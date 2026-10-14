import json
import pickle

import pandas as pd
import pytest

from stock_research.backtesting import frozen_replay as frozen
from stock_research.backtesting import historical_closure as closure
from stock_research.research.selection_audit import audit_candidates


def source_bundle(tmp_path, *, future_bar=False):
    source = tmp_path / "source"
    source.mkdir()
    config = {"engine_parameters": {"requested_start": "2026-07-20", "end_date": "2026-07-21"}}
    parameters = frozen._engine_parameters(config)
    days = ["2026-07-20", "2026-07-22" if future_bar else "2026-07-21"]
    kwargs = {**parameters, "price_frames": {"A": pd.DataFrame({
        "date": pd.to_datetime(days), "open": [10., 10.], "high": [10., 10.],
        "low": [10., 10.], "close": [10., 10.], "volume": [1000, 1000]})},
        "candidate_snapshots": {"2026-07-20": []}, "formula_phases": {},
        "trade_plans": {}, "corporate_actions": {}, "security_end_dates": {}}
    with (source / "inputs.pkl").open("wb") as stream:
        pickle.dump({"engine_kwargs": kwargs, "stock_basic_display_only": pd.DataFrame()}, stream)
    manifest = {"version": "portfolio-frozen-inputs-v2", "engine_parameters": parameters,
                "input_bundle_sha256": frozen.app.file_sha256(source / "inputs.pkl"),
                "dependencies": {"python": "archived-runtime"}, "code_sha256_at_freeze": {},
                "price_coverage": {"independent_trade_calendar": True}, "daily_limit_coverage": {"complete": True}}
    frozen._json(source / "manifest.json", manifest)
    frozen._json(tmp_path / "configuration.json", config)
    return source, tmp_path / "configuration.json"


def test_refreeze_preserves_original_bytes_and_records_environment_transition(tmp_path):
    source, config = source_bundle(tmp_path)
    original_manifest = (source / "manifest.json").read_bytes()
    updated = closure.refreeze(source, tmp_path / "new", configuration=config)
    assert (source / "manifest.json").read_bytes() == original_manifest
    assert (source / "inputs.pkl").read_bytes() == (tmp_path / "new/inputs.pkl").read_bytes()
    assert updated["refreeze_provenance"]["source_dependencies"]["python"] == "archived-runtime"
    assert updated["dependencies"] == frozen._dependency_versions()
    assert (tmp_path / "new/code/config/strategy_baseline.json").is_file()


def test_refreeze_rejects_data_after_july_21(tmp_path):
    source, config = source_bundle(tmp_path, future_bar=True)
    with pytest.raises(ValueError, match="beyond the closure cutoff"):
        closure.refreeze(source, tmp_path / "new", configuration=config)
    assert not (tmp_path / "new").exists()


def test_cash_only_historical_closure_executes_archived_code_and_independent_audit(tmp_path):
    source, config = source_bundle(tmp_path)
    result = closure.run_closure(source, tmp_path / "closure", configuration=config,
                                 database=tmp_path / "absent.sqlite3", repeat=True)
    assert result["status"] == "completed"
    assert result["accounting_status"] == "passed"
    assert result["accounting_error_count"] == 0
    assert result["deterministic_repeat_verified"] is True
    assert result["selection"]["candidate_row_count"] == 0
    assert result["risk"]["cumulative_return"] == 0.0
    assert result["risk"]["maximum_drawdown"] == 0.0
    replay = tmp_path / "closure/replay"
    for name in ("run_manifest.json", "accounting_audit.json"):
        document = json.loads((replay / name).read_text(encoding="utf-8"))
        document["accounting_status" if name == "run_manifest.json" else "status"] = "data_limited"
        frozen._json(replay / name, document)
    assert closure.audit_replay_risk(replay, tmp_path / "absent.sqlite3")["accounting_status"] == "data_limited"
    accounting = json.loads((replay / "accounting_audit.json").read_text(encoding="utf-8"))
    accounting["error_count"] = 1
    frozen._json(replay / "accounting_audit.json", accounting)
    with pytest.raises(ValueError, match="independently audited"):
        closure.audit_replay_risk(replay, tmp_path / "absent.sqlite3")


def test_selection_labels_stop_at_cutoff_even_if_cache_has_later_prices(tmp_path):
    candidates = tmp_path / "candidates"
    prices = tmp_path / "prices"
    candidates.mkdir()
    prices.mkdir()
    pd.DataFrame([{"code": "sz.000001", "candidate_score": 100, "signal_eligible": True}]).to_csv(
        candidates / "day.csv", index=False)
    (candidates / "manifest.json").write_text(json.dumps({
        "strict_financial_point_in_time": True, "industry_point_in_time": True,
        "snapshots": [{"date": "2026-07-21", "file": "day.csv"}]}), encoding="utf-8")
    pd.DataFrame({"date": ["2026-07-21", "2026-07-22"], "close": [10, 100]}).to_csv(
        prices / "sz_000001.csv", index=False)
    result = audit_candidates(candidates, prices, end_date="2026-07-21")
    assert result["observations"]["forward_1d"].isna().all()
    assert result["summary"]["strict_point_in_time"] is False
