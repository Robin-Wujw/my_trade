import pandas as pd
import pytest

from stock_research.backtesting.frozen_replay import _engine_parameters
from stock_research.research import fixed_experiments as experiments
from stock_research.research.historical_selection_controls import _dump, _load, _read_json, _write_json


def cash_bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    parameters = _engine_parameters({"engine_parameters": {
        "requested_start": "2021-01-04", "end_date": "2021-01-06"}})
    days = ["2021-01-04", "2021-01-05", "2021-01-06"]
    prices = pd.DataFrame({"date": pd.to_datetime(days), "open": 10., "high": 10.,
                           "low": 10., "close": 10., "volume": 1000})
    _dump(directory / "inputs.pkl", {"engine_kwargs": {
        **parameters, "price_frames": {"sz.000001": prices}, "formula_phases": {},
        "candidate_snapshots": {date: [] for date in days}, "trade_plans": {},
        "corporate_actions": {}, "security_end_dates": {}}})
    _write_json(directory / "manifest.json", {
        "input_bundle_sha256": experiments.file_sha256(directory / "inputs.pkl"),
        "engine_parameters": parameters})
    return directory


def test_neighborhood_is_the_registered_cross_product_and_only_ties_change():
    design = experiments.neighborhood_design()
    assert len(design) == len({row["variant_id"] for row in design}) == 18
    assert {row["start_offset"] for row in design} == {0, 5, 20}
    panel = pd.DataFrame({"code": [f"sz.{code:06d}" for code in range(61)],
                          "_baseline_score": [1.] * 61})
    asc, pct_asc = experiments.rank_neighborhood(panel, top_n=40, tie_order="ascending")
    desc, pct_desc = experiments.rank_neighborhood(panel, top_n=40, tie_order="descending")
    assert asc.iloc[0].code == "sz.000000" and desc.iloc[0].code == "sz.000060"
    pd.testing.assert_series_equal(pct_asc, pct_desc)
    with pytest.raises(ValueError, match="registered"):
        experiments.rank_neighborhood(panel, top_n=70, tie_order="ascending")
    with pytest.raises(ValueError, match="Duplicate"):
        experiments.rank_neighborhood(pd.concat([panel, panel]), top_n=40, tie_order="ascending")


def test_real_archived_variants_bind_parameters_and_resume_unchanged(tmp_path):
    source = cash_bundle(tmp_path)
    output = tmp_path / "experiment"
    experiments.prepare_parameters(source, output, [
        {"variant_id": "baseline"},
        {"variant_id": "hard_stop", "parameter_overrides": {"enable_intraday_right_protective_stop": True}},
    ], family="R04-tail-risk")
    with pytest.raises(ValueError, match="archived source"):
        experiments.worker(output / "experiment.json", "baseline")
    experiments.run(output / "experiment.json", jobs=2)
    report = _read_json(output / "comparison.json")
    assert report["completed_replays"] == 2
    assert report["return_distribution_pct"]["median"] == 0.
    rows = {row["variant_id"]: row for row in report["results"]}
    assert rows["baseline"]["parameters"]["enable_intraday_right_protective_stop"] is False
    assert rows["hard_stop"]["parameters"]["enable_intraday_right_protective_stop"] is True
    receipt = output / "runs/baseline/receipt.json"
    before = receipt.read_bytes()
    experiments.run(output / "experiment.json")
    assert receipt.read_bytes() == before
    changed = _read_json(receipt)
    changed["engine_parameters"]["enable_intraday_right_protective_stop"] = True
    _write_json(receipt, changed)
    with pytest.raises(ValueError, match="disagrees"):
        experiments.compare(output / "experiment.json")


def test_partial_results_do_not_publish_a_distribution_and_tampering_fails(tmp_path):
    source = cash_bundle(tmp_path)
    output = tmp_path / "experiment"
    experiments.prepare_parameters(source, output, [{"variant_id": "baseline"}], family="fixture")
    assert experiments.compare(output / "experiment.json")["return_distribution_pct"] is None
    path = output / "common_inputs.pkl"
    payload = _load(path)
    payload["engine_kwargs"]["initial_capital"] = 1.
    _dump(path, payload)
    with pytest.raises(ValueError, match="common inputs changed"):
        experiments.run(output / "experiment.json")


def test_changed_output_and_bad_jobs_are_rejected(tmp_path):
    source = cash_bundle(tmp_path)
    output = tmp_path / "experiment"
    experiments.prepare_parameters(source, output, [{"variant_id": "baseline"}], family="fixture")
    for value in (True, 0, 9, 1.5):
        with pytest.raises(ValueError, match="integer"):
            experiments.run(output / "experiment.json", jobs=value)
    experiments.run(output / "experiment.json")
    with (output / "runs/baseline/equity.csv").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="output changed"):
        experiments.compare(output / "experiment.json")
