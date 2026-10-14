import json
from pathlib import Path

import pandas as pd
import pytest

from stock_research.backtesting.inputs import file_sha256, load_candidate_snapshots
from stock_research.research.selection_controls import FACTORS, _selected, eligible_panel, evaluate, generate
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    dates = ["2024-01-02", "2024-01-03"]
    panel, members = [], []
    for date in dates:
        for i in range(65):
            code = f"sz.{i:06d}"
            panel.append({"date": date, "code": code, "data_cutoff": date,
                          "average_amount_20_yuan": 40_000_000 if i < 60 else 20_000_000,
                          "turnover_rate": i + 1, "close": i + 10,
                          **{factor: -(i + 1) / 100 if factor == FACTORS[-1] else i for factor in FACTORS}})
            members.append({"date": date, "code": code, "membership_as_of": date,
                            "is_listed": True, "is_traded": True, "is_st": False, "history_days": 100})
    pd.DataFrame(panel).to_csv(tmp_path / "panel.csv", index=False)
    pd.DataFrame(members).to_csv(tmp_path / "members.csv", index=False)
    write_json(tmp_path / "source.json", {"observation_dates": dates,
               "panel_sha256": file_sha256(tmp_path / "panel.csv"),
               "membership_sha256": file_sha256(tmp_path / "members.csv"),
               "universe_definition": "dated listed/traded/nonST/history>=60",
               "adjustment_policy": "raw execution with dated factors", "availability_policy": "close_t"})
    write_json(tmp_path / "profile.json", {"engine_parameters": {"requested_start": dates[0], "end_date": dates[-1]}})
    left = tmp_path / "left"
    left.mkdir()
    snapshots = []
    for date in dates:
        path = left / f"candidates_{date}.csv"
        pd.DataFrame([{"code": "sh.600001", "allow_left": True, "allow_right": False}]).to_csv(path, index=False)
        snapshots.append({"date": date, "file": path.name, "candidate_count": 1, "sha256": file_sha256(path)})
    write_json(left / "manifest.json", {"version": SNAPSHOT_VERSION,
               "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION, "snapshots": snapshots})
    return tmp_path, dates


def create(root):
    return generate(root / "panel.csv", root / "members.csv", root / "source.json",
                    root / "profile.json", root / "left", root / "generated")


def test_generate_100_seeds_common_pool_and_lowvol_direction(inputs):
    root, dates = inputs
    result = create(root)
    assert result["completed_random_replays"] == 0
    assert len(result["controls"]) == 103
    selections = load_candidate_snapshots(root / "generated/low_residual_volatility", dates[0], dates[-1])
    assert len(selections[dates[0]]) == 50
    assert selections[dates[0]][0]["code"] == "sz.000000"
    assert selections[dates[0]][-1]["code"] == "sz.000049"
    exposures = pd.read_csv(root / "generated/style_exposures.csv")
    assert exposures["eligible_count"].eq(60).all()
    assert exposures["selected_count"].eq(50).all()
    assert result["source_sha256"]["left_candidates"] == file_sha256(root / "left/manifest.json")


def test_random_is_order_independent_and_future_dates_do_not_change_history(inputs):
    root, dates = inputs
    panel, _ = eligible_panel(root / "panel.csv", root / "members.csv", dates)
    first = panel[panel.date.eq(dates[0])]
    codes = _selected(first, "random_0007", 7)[0]["code"].tolist()
    assert codes == _selected(first.sample(frac=1, random_state=23), "random_0007", 7)[0]["code"].tolist()
    assert codes != _selected(first, "random_0008", 8)[0]["code"].tolist()
    assert first.iloc[0]["_baseline_score"] == pytest.approx((5 / 65 + 1) / 6)
    assert first.iloc[0]["_baseline_score"] != pytest.approx((5 / 60 + 1) / 6)


@pytest.mark.parametrize("failure", ["future", "missing_factor", "missing_date", "unknown_flag", "missing_adv"])
def test_coverage_failures_are_not_silent_pool_shrinkage(inputs, failure):
    root, dates = inputs
    panel = pd.read_csv(root / "panel.csv")
    members = pd.read_csv(root / "members.csv")
    if failure == "future":
        panel.loc[0, "data_cutoff"] = "2025-01-01"
    elif failure == "missing_factor":
        panel = panel.drop(columns=[FACTORS[0]])
    elif failure == "missing_date":
        panel = panel[panel.date.ne(dates[0])]
    elif failure == "unknown_flag":
        members["is_st"] = "unknown"
    else:
        panel.loc[0, "average_amount_20_yuan"] = float("nan")
    panel.to_csv(root / "panel.csv", index=False)
    members.to_csv(root / "members.csv", index=False)
    with pytest.raises(ValueError):
        eligible_panel(root / "panel.csv", root / "members.csv", dates)


def test_factor_warmup_missing_values_preserve_original_available_weight_rule(inputs):
    root, dates = inputs
    panel = pd.read_csv(root / "panel.csv")
    panel.loc[0, FACTORS[-1]] = float("nan")
    panel.to_csv(root / "panel.csv", index=False)
    eligible, _ = eligible_panel(root / "panel.csv", root / "members.csv", dates)
    first = eligible[eligible.date.eq(dates[0])]
    assert len(first) == 60
    assert first.iloc[0]["_baseline_score"] == pytest.approx(1 / 65)
    assert "sz.000000" not in set(_selected(first, "low_residual_volatility", None)[0]["code"])
    assert len(_selected(first, "random_0000", 0)[0]) == 50


def receipts(root, experiment):
    rows = []
    for index, control in enumerate(experiment["controls"]):
        audit_path = root / f"accounting_{index}.json"
        write_json(audit_path, {"status": "passed", "error_count": 0, "errors": []})
        summary = {"final_return_pct": 200 if index == 0 else index,
                   "frozen_run": {"status": "finished", "accounting_status": "passed",
                                  "acceptance_status": "research_only", "engine_sha256": "e" * 64,
                                  "code_sha256": {"engine.py": "e" * 64},
                                  "input_bundle_sha256": f"{index:064x}",
                                  "engine_parameters": experiment["engine_parameters"]}}
        path = root / f"summary_{index}.json"
        rows.append({"control_id": control["control_id"], "summary_path": path.name,
                     "accounting_status": "passed", "accounting_error_count": 0,
                     "accounting_audit_path": audit_path.name, "accounting_audit_sha256": file_sha256(audit_path),
                     "candidate_manifest_sha256": control["manifest_sha256"],
                     "profile_sha256": experiment["source_sha256"]["profile"],
                     "input_bundle_sha256": f"{index:064x}",
                     "shared_input_sha256": {**{key: "b" * 64 for key in
                         ("prices", "formula_phases", "trade_plans", "corporate_actions", "security_end_dates")},
                         "left_candidates": experiment["source_sha256"]["left_candidates"]}})
        summary["selection_control"] = {key: rows[-1][key] for key in
            ("control_id", "candidate_manifest_sha256", "profile_sha256", "shared_input_sha256")}
        write_json(path, summary)
        rows[-1]["summary_sha256"] = file_sha256(path)
    return {"experiment_sha256": file_sha256(root / "generated/experiment.json"), "replays": rows}


def test_only_completed_comparable_runs_support_percentile(inputs):
    root, _ = inputs
    experiment = create(root)
    registry = receipts(root, experiment)
    write_json(root / "receipts.json", registry)
    result = evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")
    assert result["completed_random_replays"] == 100
    assert result["comparison"]["baseline_percentile_midrank"] == 100
    assert result["comparison"]["baseline_in_top_five_percent"] is True
    registry["replays"].pop()
    write_json(root / "receipts.json", registry)
    result = evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")
    assert result["status"] == "incomplete_replays"
    assert result["comparison"] is None


@pytest.mark.parametrize("error_count", [0, 1])
def test_data_limited_accounting_requires_zero_errors_and_preserves_qualification(inputs, error_count):
    root, _ = inputs
    experiment = create(root)
    registry = receipts(root, experiment)
    registry["replays"] = registry["replays"][:1]
    receipt = registry["replays"][0]
    audit_path = root / receipt["accounting_audit_path"]
    write_json(audit_path, {"status": "data_limited", "error_count": error_count,
                           "errors": ["cash difference"] if error_count else [],
                           "data_limitations": ["pay_date_missing"]})
    receipt.update(accounting_status="data_limited", accounting_error_count=error_count,
                   accounting_audit_sha256=file_sha256(audit_path))
    path = root / receipt["summary_path"]
    summary = json.loads(path.read_text())
    summary["frozen_run"].update(accounting_status="data_limited", acceptance_status="data_limited")
    write_json(path, summary)
    receipt["summary_sha256"] = file_sha256(path)
    write_json(root / "receipts.json", registry)
    if error_count:
        with pytest.raises(ValueError, match="accounting errors"):
            evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")
    else:
        result = evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")
        assert result["status"] == "incomplete_replays"
        assert result["completed_random_replays"] == 0
        assert json.loads(path.read_text())["frozen_run"]["accounting_status"] == "data_limited"


@pytest.mark.parametrize("failure", ["engine", "left", "duplicate", "failed_accounting", "changed_summary", "reused_bundle", "unbound_summary"])
def test_invalid_replay_evidence_is_rejected(inputs, failure):
    root, _ = inputs
    experiment = create(root)
    registry = receipts(root, experiment)
    summary_path = root / "summary_1.json"
    summary = json.loads(summary_path.read_text())
    if failure == "engine":
        summary["frozen_run"]["engine_parameters"] = {"max_positions": 100}
    elif failure == "failed_accounting":
        summary["frozen_run"]["accounting_status"] = "failed"
    elif failure == "left":
        registry["replays"][1]["shared_input_sha256"]["left_candidates"] = "f" * 64
    elif failure == "duplicate":
        registry["replays"].append(registry["replays"][0])
    elif failure == "reused_bundle":
        summary["frozen_run"]["input_bundle_sha256"] = "0" * 64
        registry["replays"][1]["input_bundle_sha256"] = "0" * 64
    elif failure == "unbound_summary":
        summary.pop("selection_control")
    else:
        summary["final_return_pct"] = 99999
    write_json(summary_path, summary)
    if failure != "changed_summary":
        registry["replays"][1]["summary_sha256"] = file_sha256(summary_path)
    write_json(root / "receipts.json", registry)
    with pytest.raises(ValueError):
        evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")


@pytest.mark.parametrize("substitution", ["common_inputs", "source", "engine"])
def test_uniform_replay_substitution_cannot_override_frozen_experiment(inputs, substitution):
    root, _ = inputs
    experiment = create(root)
    experiment["common_inputs_sha256"] = "b" * 64
    experiment["code_sha256"] = {"stock_research/backtesting/engine.py": "e" * 64}
    write_json(root / "generated/experiment.json", experiment)
    registry = receipts(root, experiment)
    for receipt in registry["replays"]:
        path = root / receipt["summary_path"]
        summary = json.loads(path.read_text())
        summary["frozen_run"]["code_sha256"] = experiment["code_sha256"]
        if substitution == "common_inputs":
            for key in receipt["shared_input_sha256"]:
                if key != "left_candidates":
                    receipt["shared_input_sha256"][key] = "c" * 64
            summary["selection_control"]["shared_input_sha256"] = receipt["shared_input_sha256"]
        elif substitution == "source":
            summary["frozen_run"]["code_sha256"] = {"stock_research/backtesting/engine.py": "f" * 64}
            summary["frozen_run"]["engine_sha256"] = "f" * 64
        else:
            summary["frozen_run"]["engine_sha256"] = "f" * 64
        write_json(path, summary)
        receipt["summary_sha256"] = file_sha256(path)
    write_json(root / "receipts.json", registry)
    with pytest.raises(ValueError, match="frozen experiment"):
        evaluate(root / "generated/experiment.json", root / "receipts.json", root / "comparison.json")
