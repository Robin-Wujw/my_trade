import json
import pickle
import socket
import sqlite3

import pandas as pd
import pytest

from stock_research.backtesting import frozen_replay as frozen


def test_snapshot_is_read_only_and_stable_between_repository_connections(tmp_path):
    path = tmp_path / "source.sqlite3"
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample (value INTEGER)")
        writer.execute("INSERT INTO sample VALUES (1)")
    with frozen._readonly_snapshot(path) as database:
        first = database.connect(read_only=True)
        assert first.execute("SELECT value FROM sample").fetchone()[0] == 1
        first.close()
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE sample SET value=2")
        second = database.connect(read_only=True)
        assert second.execute("SELECT value FROM sample").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            second.execute("UPDATE sample SET value=3")
        with pytest.raises(RuntimeError, match="read-only"):
            database.connect()
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT value FROM sample").fetchone()[0] == 2


def test_offline_guard_blocks_database_and_network():
    with frozen._offline():
        with pytest.raises(RuntimeError, match="database or network"):
            sqlite3.connect(":memory:")
        with pytest.raises(RuntimeError, match="database or network"):
            socket.create_connection(("example.invalid", 80))
    with sqlite3.connect(":memory:") as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)


def test_prepared_price_units_must_agree():
    frame = pd.DataFrame({"qfq_to_raw_factor": [2.], **{
        field: [5.] for field in ("open", "high", "low", "close")}, **{
        f"raw_{field}": [10.] for field in ("open", "high", "low", "close")}})
    frozen._validate_prepared({"A": frame})
    frame.loc[0, "raw_close"] = 9
    with pytest.raises(ValueError, match="units disagree"):
        frozen._validate_prepared({"A": frame})


def small_bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    parameters = frozen._engine_parameters({"engine_parameters": {
        "requested_start": "2025-01-02", "end_date": "2025-01-03"}})
    kwargs = {**parameters, "price_frames": {"A": pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]), "open": [10., 10.],
        "high": [10., 10.], "low": [10., 10.], "close": [10., 10.], "volume": [1000, 1000]})},
        "candidate_snapshots": {"2025-01-02": []}, "formula_phases": {}, "trade_plans": {},
        "corporate_actions": {}, "security_end_dates": {}}
    with (directory / "inputs.pkl").open("wb") as stream:
        pickle.dump({"engine_kwargs": kwargs, "stock_basic_display_only": pd.DataFrame()}, stream)
    manifest = {"version": "portfolio-frozen-inputs-v2", "engine_parameters": parameters,
                "input_bundle_sha256": frozen.app.file_sha256(directory / "inputs.pkl"),
                "code_sha256_at_freeze": frozen._code_hashes(), "dependencies": frozen._dependency_versions(),
                "price_coverage": {"independent_trade_calendar": True},
                "daily_limit_coverage": {"complete": True}}
    frozen._json(directory / "manifest.json", manifest)
    return directory, manifest


@pytest.mark.parametrize("damage", ["inputs", "code", "dependency", "parameters"])
def test_replay_rejects_changed_contract_before_creating_output(tmp_path, damage):
    directory, manifest = small_bundle(tmp_path)
    if damage == "inputs":
        with (directory / "inputs.pkl").open("ab") as stream:
            stream.write(b"modified")
    elif damage == "code":
        manifest["code_sha256_at_freeze"] = {}
    elif damage == "dependency":
        manifest["dependencies"]["python"] = "0.0.0"
    else:
        manifest["engine_parameters"]["initial_capital"] += 1
    frozen._json(directory / "manifest.json", manifest)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        frozen.replay(directory, output)
    assert not output.exists()


def test_cash_only_replay_passes_independent_audit(tmp_path):
    directory, _ = small_bundle(tmp_path)
    output = tmp_path / "run"
    status = frozen.replay(directory, output)
    assert status["status"] == "finished"
    assert status["accounting_status"] == "passed"
    assert status["acceptance_status"] == "research_only"
    assert status["strict_point_in_time"] is False
    assert json.loads((output / "accounting_audit.json").read_text())["error_count"] == 0
    summary = json.loads(
        (output / "portfolio_2025-01-02_2025-01-03_summary.json").read_text(encoding="utf-8")
    )
    assert summary["acceptance_status"] == "research_only"
    assert summary["research_only"] is True
    assert summary["strict_point_in_time"] is False


def test_reference_worker_imports_its_own_dependencies(tmp_path):
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "frozen_marker.py").write_text("MARKER = 'archived-dependency'", encoding="utf-8")
    engine = directory / "engine.py"
    engine.write_text("""from frozen_marker import MARKER
def run_portfolio_backtest(**kwargs):
    return {'marker': MARKER}
""", encoding="utf-8")
    bundle = tmp_path / "inputs.pkl"
    with bundle.open("wb") as stream:
        pickle.dump({"engine_kwargs": {}}, stream)
    result = frozen._run_reference(directory, bundle, tmp_path, engine)
    assert result == {"marker": "archived-dependency"}
