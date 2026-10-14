import pandas as pd
import pytest

from stock_research.backtesting.frozen_replay import _engine_parameters
from stock_research.research import historical_selection_controls as controls
from stock_research.research.selection_controls import evaluate


def experiment_fixture(tmp_path):
    """A tiny real archived engine run, separate from actual research evidence."""
    output = tmp_path / "experiment"
    output.mkdir()
    parameters = _engine_parameters({"engine_parameters": {
        "requested_start": "2021-01-04", "end_date": "2021-01-05"}})
    dates = ["2021-01-04", "2021-01-05"]
    prices = pd.DataFrame({"date": pd.to_datetime(dates), "open": [10., 10.],
                           "high": [10., 10.], "low": [10., 10.], "close": [10., 10.],
                           "volume": [1000, 1000]})
    controls._dump(output / "common_inputs.pkl", {"engine_kwargs": {
        **parameters, "price_frames": {"sz.000001": prices}, "formula_phases": {},
        "trade_plans": {}, "corporate_actions": {}, "security_end_dates": {}}})
    controls._dump(output / "left_candidates.pkl", {date: [] for date in dates})
    controls._write_json(output / "profile.json", {"engine_parameters": parameters})
    directory = output / "candidates/baseline_six_factor"
    directory.mkdir(parents=True)
    controls._dump(directory / "candidates.pkl", {date: [{
        "code": "sz.000001", "playbook_factor": "fixture", "allow_left": False,
        "allow_right": True, "selected_for_trading": True, "signal_eligible": True,
        "selection_engine": "quantsplaybook_factor_only"}] for date in dates})
    controls._write_json(directory / "manifest.json", {
        "control_id": "baseline_six_factor", "seed": None, "dates": dates,
        "candidate_bundle": {"file": "candidates.pkl", "sha256": controls.file_sha256(directory / "candidates.pkl")}})
    hashes = controls._archive(output)
    experiment = {"version": controls.VERSION, "dates": dates, "engine_parameters": parameters,
                  "code_sha256": hashes, "dependencies": controls._dependency_versions(),
                  "limitations": ["Cash-only integration fixture; not historical research evidence."],
                  "common_inputs_sha256": controls.file_sha256(output / "common_inputs.pkl"),
                  "source_sha256": {"profile": controls.file_sha256(output / "profile.json"),
                                    "left_candidates": controls.file_sha256(output / "left_candidates.pkl")},
                  "controls": [{"control_id": "baseline_six_factor", "seed": None,
                                "manifest": "candidates/baseline_six_factor/manifest.json",
                                "manifest_sha256": controls.file_sha256(directory / "manifest.json")}]}
    controls._write_json(output / "experiment.json", experiment)
    return output


def test_prepare_rejects_cutoff_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="2026-07-21"):
        controls.prepare(panel_directory="missing", baseline_bundle="missing", price_directory="missing",
                         database="missing", profile="missing", output_directory=tmp_path / "untouched",
                         end_date="2026-07-22")
    assert not (tmp_path / "untouched").exists()


def test_worker_requires_archived_source(tmp_path):
    output = experiment_fixture(tmp_path)
    with pytest.raises(ValueError, match="archived source"):
        controls.worker(output / "experiment.json", "baseline_six_factor")


def test_real_archived_cash_run_and_resume_and_input_tamper(tmp_path):
    output = experiment_fixture(tmp_path)
    controls.run(output / "experiment.json", profile_execution=True)
    summary = controls._read_json(output / "runs/baseline_six_factor/summary.json")
    assert summary["frozen_run"]["accounting_status"] == "passed"
    assert summary["research_diagnostics"]["risk"]["cumulative_return"] == 0
    assert summary["final_return_pct"] == 0
    assert summary["frozen_run"]["execution_profiled"] is True
    assert (output / "runs/baseline_six_factor/execution.prof").is_file()
    before = (output / "runs/baseline_six_factor/summary.json").read_bytes()
    (output / "receipts.json").unlink()
    controls.run(output / "experiment.json")
    assert (output / "runs/baseline_six_factor/summary.json").read_bytes() == before
    assert (output / "receipts.json").is_file()
    assert controls._read_json(output / "comparison.json")["comparison"] is None
    with (output / "common_inputs.pkl").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="Frozen input changed"):
        controls.run(output / "experiment.json")


def test_evaluator_rejects_changed_compact_candidate_bundle(tmp_path):
    output = experiment_fixture(tmp_path)
    controls.run(output / "experiment.json")
    with (output / "candidates/baseline_six_factor/candidates.pkl").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="Candidate bundle hash mismatch"):
        evaluate(output / "experiment.json", output / "receipts.json", output / "invalid.json")


def test_source_manifest_cannot_be_replaced_without_experiment_binding(tmp_path):
    output = experiment_fixture(tmp_path)
    source = output / "source"
    path = source / "stock_research/backtesting/engine.py"
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    manifest = controls._read_json(source / "source_manifest.json")
    manifest["sha256"]["stock_research/backtesting/engine.py"] = controls.file_sha256(path)
    controls._write_json(source / "source_manifest.json", manifest)
    with pytest.raises(ValueError, match="differs from the experiment"):
        controls.run(output / "experiment.json")


def test_legacy_audit_attestation_preserves_real_summary_and_original_receipt(tmp_path):
    output = experiment_fixture(tmp_path)
    controls.run(output / "experiment.json")
    directory = output / "runs/baseline_six_factor"
    summary = (directory / "summary.json").read_bytes()
    receipt = controls._read_json(directory / "receipt.json")
    receipt = {key: value for key, value in receipt.items() if not key.startswith("accounting_")}
    controls._write_json(directory / "receipt.json", receipt)
    result = controls.attest_accounting(output / "experiment.json", "baseline_six_factor")
    assert result["accounting_status"] == "passed"
    assert result["accounting_error_count"] == 0
    assert (directory / "summary.json").read_bytes() == summary
    assert controls._read_json(directory / "receipt_before_accounting_attestation.json") == receipt
    assert controls.attest_accounting(output / "experiment.json", "baseline_six_factor") == result


def test_calendar_trim_removes_future_market_state_without_filling_unknown_actions():
    kwargs = {"formula_phases": {"2021-01-04": {}, "2022-01-04": {}},
              "corporate_actions": {"sz.000001": [{"ex_date": "20210104"}, {"ex_date": "20220104"}]},
              "security_end_dates": {"sz.000001": pd.Timestamp("2022-01-04")}, "trade_plans": {"plans": {}}}
    result = controls._trim_calendar(kwargs, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
    assert result == {"formula_phases_removed": 1, "corporate_actions_removed": 1,
                      "future_security_end_dates_removed": 1}
    assert kwargs["corporate_actions"]["sz.000001"] == [{"ex_date": "20210104"}]
    assert kwargs["security_end_dates"] == {}


def test_fork_and_execution_parity_require_identical_fills_and_inputs(tmp_path):
    original = experiment_fixture(tmp_path)
    controls.run(original / "experiment.json")
    updated = tmp_path / "updated"
    controls.fork_for_execution_equivalence(original / "experiment.json", updated)
    controls.run(updated / "experiment.json")
    old_run = original / "runs/baseline_six_factor"
    new_run = updated / "runs/baseline_six_factor"
    report = controls.compare_execution_results(old_run, new_run, tmp_path / "parity.json")
    assert report["status"] == "identical_equity_trades_events"
    with (new_run / "trades.csv").open("a") as handle:
        handle.write("changed fill")
    with pytest.raises(ValueError, match="parity failed"):
        controls.compare_execution_results(old_run, new_run, tmp_path / "invalid_parity.json")


def test_resume_supplements_all_legacy_receipts_before_aggregate_evaluation(tmp_path):
    output = experiment_fixture(tmp_path)
    candidate = output / "candidates/random_0000"
    candidate.mkdir()
    base = output / "candidates/baseline_six_factor"
    controls._dump(candidate / "candidates.pkl", controls._load(base / "candidates.pkl"))
    manifest = controls._read_json(base / "manifest.json")
    manifest.update(control_id="random_0000", seed=0)
    manifest["candidate_bundle"]["sha256"] = controls.file_sha256(candidate / "candidates.pkl")
    controls._write_json(candidate / "manifest.json", manifest)
    experiment = controls._read_json(output / "experiment.json")
    experiment["controls"].append({"control_id": "random_0000", "seed": 0,
                                   "manifest": "candidates/random_0000/manifest.json",
                                   "manifest_sha256": controls.file_sha256(candidate / "manifest.json")})
    controls._write_json(output / "experiment.json", experiment)
    controls.run(output / "experiment.json")
    before = {}
    for name in ("baseline_six_factor", "random_0000"):
        directory = output / "runs" / name
        before[name] = controls.file_sha256(directory / "summary.json")
        receipt = controls._read_json(directory / "receipt.json")
        controls._write_json(directory / "receipt.json", {
            key: value for key, value in receipt.items() if not key.startswith("accounting_")})
    controls.run(output / "experiment.json", controls=["baseline_six_factor"])
    for name in before:
        assert controls.file_sha256(output / "runs" / name / "summary.json") == before[name]
        assert (output / "runs" / name / "receipt_before_accounting_attestation.json").is_file()
    assert controls._read_json(output / "comparison.json")["completed_random_replays"] == 1
