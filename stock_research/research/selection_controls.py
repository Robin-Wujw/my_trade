"""Reproducible selection controls and evidence-gated, offline replay comparison.

Generating candidates never counts as a completed backtest. Replay receipts are
an explicit contract with the external executor, not a substitute for its audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from stock_research.backtesting.inputs import file_sha256, load_candidate_snapshots
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION


VERSION = "selection-controls-v1"
FACTORS = ("network_cc", "ma_convergence", "disposition_reversal", "high_quality_momentum",
           "coin_team", "low_market_residual_volatility_20")
STYLES = ("turnover_rate", "average_amount_20_yuan", "low_market_residual_volatility_20", "close")
ADV_FLOOR = 30_000_000
TOP_N = 50


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _dates(values, name):
    text = pd.Series(values, dtype=str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"{name} requires ISO calendar dates")
    dates = pd.to_datetime(text, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    return text.to_numpy()


def _dated_frame(path, cutoff):
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"date", "code", cutoff}
    if required - set(frame):
        raise ValueError(f"{path} missing {sorted(required - set(frame))}")
    frame["date"] = _dates(frame["date"], "date")
    frame[cutoff] = _dates(frame[cutoff], cutoff)
    if frame[["date", "code"]].duplicated().any() or not frame["code"].str.fullmatch(r"(?:sh|sz|bj)\.\d{6}").all():
        raise ValueError("date/code must be unique with canonical sh./sz./bj. codes")
    if frame[cutoff].gt(frame["date"]).any():
        raise ValueError(f"Future {cutoff} is not allowed")
    return frame


def _bool(values, name):
    text = values.astype(str).str.lower().str.strip()
    if not text.isin({"true", "false", "0", "1", "0.0", "1.0"}).all():
        raise ValueError(f"Unknown membership flag: {name}")
    return text.isin({"true", "1", "1.0"})


def eligible_panel(panel_path, membership_path, dates):
    panel = _dated_frame(panel_path, "data_cutoff")
    members = _dated_frame(membership_path, "membership_as_of")
    required = {"is_listed", "is_traded", "is_st", "history_days"}
    if required - set(members):
        raise ValueError(f"Membership missing {sorted(required - set(members))}")
    if set(panel["date"]) != set(dates) or set(members["date"]) != set(dates):
        raise ValueError("Panel and membership must cover every declared observation date exactly")
    flags = {name: _bool(members[name], name) for name in ("is_listed", "is_traded", "is_st")}
    history = pd.to_numeric(members["history_days"], errors="coerce")
    if not np.isfinite(history).all() or history.lt(0).any() or history.mod(1).ne(0).any():
        raise ValueError("history_days must be finite nonnegative integers")
    members = members[flags["is_listed"] & flags["is_traded"] & ~flags["is_st"] & history.ge(60)]
    required_panel = set(FACTORS) | set(STYLES)
    if required_panel - set(panel):
        raise ValueError(f"Panel missing {sorted(required_panel - set(panel))}")
    joined = members[["date", "code"]].merge(panel, on=["date", "code"], how="left", validate="one_to_one")
    amount = pd.to_numeric(joined["average_amount_20_yuan"], errors="coerce")
    if not np.isfinite(amount).all() or amount.lt(0).any():
        raise ValueError("Missing/invalid ADV20 for an otherwise eligible member")
    # Match the frozen model: rank on the full dated tradable cross-section,
    # then apply liquidity. Ranking again after ADV screening changes the model.
    for name in FACTORS:
        joined[name] = pd.to_numeric(joined[name], errors="coerce")
        if np.isinf(joined[name]).any():
            raise ValueError(f"Infinite factor value before liquidity screening: {name}")
    joined["_baseline_score"] = joined.groupby("date")[list(FACTORS)].rank(pct=True, method="average").mean(axis=1)
    joined = joined[amount.ge(ADV_FLOOR)].copy()
    styles = list(STYLES) + (["market_cap_yuan"] if "market_cap_yuan" in panel else [])
    for name in set(FACTORS) | set(styles):
        joined[name] = pd.to_numeric(joined[name], errors="coerce")
        if name in FACTORS and np.isinf(joined[name]).any():
            raise ValueError(f"Infinite factor value in the shared eligible pool: {name}")
        if name not in FACTORS and not np.isfinite(joined[name]).all():
            raise ValueError(f"Missing factor/style coverage in the shared eligible pool: {name}")
    if joined["close"].le(0).any() or joined["turnover_rate"].lt(0).any():
        raise ValueError("Invalid close or turnover")
    if joined[FACTORS[-1]].gt(0).any():
        raise ValueError("Residual-volatility score must be negative standard deviation")
    counts = joined.groupby("date").size().reindex(dates, fill_value=0)
    if counts.lt(TOP_N).any():
        raise ValueError("Every dated eligible pool must have at least 50 complete observations")
    return joined.sort_values(["date", "code"]).reset_index(drop=True), styles


def _selected(group, control, seed):
    if control == "baseline_six_factor":
        scores = group["_baseline_score"]
    elif control == "low_turnover":
        scores = -group["turnover_rate"]
    elif control == "low_residual_volatility":
        scores = group[FACTORS[-1]]  # Negative residual std: larger means less risk.
    else:
        date = group.iloc[0]["date"]
        scores = group["code"].map(lambda code: hashlib.sha256(
            f"{seed}|{date}|{code}".encode()).hexdigest())
    ranked = group.assign(_score=scores).dropna(subset=["_score"]).sort_values(["_score", "code"], ascending=[False, True])
    if len(ranked) < TOP_N:
        raise ValueError(f"{control} has fewer than 50 available scores on {group.iloc[0]['date']}")
    return ranked.head(TOP_N), scores.rank(pct=True, method="average")


def generate(panel_path, membership_path, source_manifest_path, profile_path, left_candidates,
             output_directory, *, seeds=range(100)):
    """Write right-side candidates; bind the unchanged left source for the replay executor."""
    seeds = list(seeds)
    if len(seeds) < 100 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("At least 100 distinct nonnegative integer seeds are required")
    inputs = {"panel": Path(panel_path), "membership": Path(membership_path),
              "source_manifest": Path(source_manifest_path), "profile": Path(profile_path),
              "left_candidates": Path(left_candidates) / "manifest.json"}
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    source, profile = _read(source_manifest_path), _read(profile_path)
    dates = sorted(_dates(source.get("observation_dates", []), "observation_dates").tolist())
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Explicit unique observation_dates are required")
    for key in ("universe_definition", "adjustment_policy", "availability_policy"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Source manifest must disclose {key}")
    if source.get("panel_sha256") != hashes["panel"] or source.get("membership_sha256") != hashes["membership"]:
        raise ValueError("Source manifest input hash mismatch")
    if not isinstance(profile.get("engine_parameters"), dict) or not profile["engine_parameters"]:
        raise ValueError("Profile requires frozen engine_parameters")
    # Verify the actual shared left files, not only a directory label.
    left = load_candidate_snapshots(left_candidates, dates[0], dates[-1])
    if not set(dates).issubset(left):
        raise ValueError("Shared left source does not cover all observation dates")
    left_manifest = _read(inputs["left_candidates"])
    for row in left_manifest.get("snapshots", []):
        if row["date"] in dates and not row.get("sha256"):
            raise ValueError("Shared left snapshots require file SHA256 evidence")
    panel, styles = eligible_panel(panel_path, membership_path, dates)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    controls = [("baseline_six_factor", None), ("low_turnover", None), ("low_residual_volatility", None)]
    controls += [(f"random_{seed:04d}", seed) for seed in seeds]
    exposures, records = [], []
    groups = {date: group for date, group in panel.groupby("date", sort=True)}
    for control, seed in controls:
        directory = output / control
        directory.mkdir()
        snapshots = []
        for date in dates:
            group = groups[date]
            selected, percentile = _selected(group, control, seed)
            rows = []
            for rank, (index, row) in enumerate(selected.iterrows(), 1):
                rows.append({"code": row["code"], "name": row["code"],
                             "strategy_part": f"selection_control:{control}",
                             "selection_engine": "quantsplaybook_factor_only",
                             "selection_profile": "quantsplaybook_factor_only",
                             "playbook_factor": control, "playbook_factor_rank": rank,
                             "candidate_score": float(percentile.loc[index] * 100),
                             "selected_for_trading": True, "signal_eligible": True,
                             "allow_right": True, "allow_left": False})
            path = directory / f"candidates_{date}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            snapshots.append({"date": date, "file": path.name, "candidate_count": len(rows),
                              "sha256": file_sha256(path)})
            exposure = {"control": control, "date": date, "eligible_count": len(group), "selected_count": len(rows)}
            for style in styles:
                exposure[f"{style}_median"] = float(selected[style].median())
                exposure[f"{style}_median_percentile"] = float(group[style].rank(pct=True).loc[selected.index].median())
            exposures.append(exposure)
        manifest = {"version": SNAPSHOT_VERSION, "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
                    "control_version": VERSION, "control_id": control, "seed": seed, "snapshots": snapshots,
                    "research_only": True, "strict_point_in_time": False,
                    "uses_financial_data": False, "financial_point_in_time": True,
                    "strict_financial_point_in_time": True, "industry_point_in_time": False,
                    "industry_point_in_time_status": "unverified_do_not_bypass_existing_gate",
                    "source_sha256": hashes, "top_n": TOP_N, "adv20_floor_yuan": ADV_FLOOR,
                    "start_date": dates[0], "end_date": dates[-1],
                    "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
                    "left_candidate_policy": "external_executor_must_preserve_shared_left_source"}
        _json(directory / "manifest.json", manifest)
        records.append({"control_id": control, "seed": seed, "manifest": f"{control}/manifest.json",
                        "manifest_sha256": file_sha256(directory / "manifest.json")})
    pd.DataFrame(exposures).to_csv(output / "style_exposures.csv", index=False, encoding="utf-8-sig")
    result = {"version": VERSION, "status": "candidates_generated_not_replayed", "research_only": True,
              "strict_point_in_time": False, "completed_random_replays": 0, "seeds": seeds,
              "dates": dates, "top_n": TOP_N, "adv20_floor_yuan": ADV_FLOOR,
              "source_sha256": hashes, "source_paths": {k: str(v.resolve()) for k, v in inputs.items()},
              "engine_parameters": profile["engine_parameters"], "source_disclosures": source,
              "common_universe_sha256": _digest(panel[["date", "code"]].to_dict("records")),
              "baseline_rank_policy": "full_PIT_tradable_percentiles_available_factor_weight_renormalization_then_ADV30m_then_Top50",
              "style_exposures_sha256": file_sha256(output / "style_exposures.csv"),
              "controls": records,
              "limitations": ["Dated inputs are checked, but their external PIT provenance is not certified.",
                              "Right-only snapshots require the unchanged shared left lane at execution.",
                              "Industry metadata remains unverified; no strict-PIT replay is authorized.",
                              "Generating 100 seeds is not evidence of 100 completed experiments."]}
    for name, path in inputs.items():
        if file_sha256(path) != hashes[name]:
            raise ValueError(f"Source changed during generation: {name}")
    _json(output / "experiment.json", result)
    return result


def evaluate(experiment_path, receipts_path, output_path, *, metric="final_return_pct"):
    """Compare completed replay evidence; receipts must come from the actual executor.

    The executor attests common-input hashes; this reader does not deserialize
    pickle bundles or certify a receipt's authenticity.
    """
    experiment_path, receipts_path = Path(experiment_path), Path(receipts_path)
    experiment, registry = _read(experiment_path), _read(receipts_path)
    expected = {row["control_id"]: row for row in experiment["controls"]}
    if experiment.get("version") != VERSION or registry.get("experiment_sha256") != file_sha256(experiment_path):
        raise ValueError("Replay registry is not bound to this experiment")
    rows, seen, common = [], set(), None
    used_bundles, used_summaries = set(), set()
    for receipt in registry.get("replays", []):
        control = receipt.get("control_id")
        if control not in expected or control in seen:
            raise ValueError("Unknown or duplicate replay control")
        seen.add(control)
        manifest_path = experiment_path.parent / expected[control]["manifest"]
        manifest_hash = file_sha256(manifest_path)
        if manifest_hash != expected[control]["manifest_sha256"] or receipt.get("candidate_manifest_sha256") != manifest_hash:
            raise ValueError("Replay candidate manifest hash mismatch")
        manifest = _read(manifest_path)
        if "candidate_bundle" in manifest:
            bundle = manifest["candidate_bundle"]
            name = bundle.get("file", "")
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Candidate bundle must be a local basename")
            if file_sha256(manifest_path.parent / name) != bundle.get("sha256"):
                raise ValueError("Candidate bundle hash mismatch")
            if manifest.get("dates") != experiment["dates"]:
                raise ValueError("Candidate bundle calendar mismatch")
        else:
            load_candidate_snapshots(manifest_path.parent, experiment["dates"][0], experiment["dates"][-1])
        if manifest.get("control_id") != control or manifest.get("seed") != expected[control]["seed"]:
            raise ValueError("Candidate identity mismatch")
        summary_path = receipts_path.parent / receipt["summary_path"]
        if file_sha256(summary_path) != receipt.get("summary_sha256"):
            raise ValueError("Replay summary hash mismatch")
        summary = _read(summary_path)
        run = summary.get("frozen_run", {})
        binding = summary.get("selection_control", {})
        if binding != {key: receipt.get(key) for key in ("control_id", "candidate_manifest_sha256",
                                                       "profile_sha256", "shared_input_sha256")}:
            raise ValueError("Actual replay summary must bind the control and shared inputs")
        if run.get("status") != "finished" or run.get("accounting_status") not in {"passed", "data_limited"}:
            raise ValueError("Replay must be finished with an independently checked accounting result")
        audit_path = receipts_path.parent / receipt.get("accounting_audit_path", "")
        if not audit_path.is_file() or file_sha256(audit_path) != receipt.get("accounting_audit_sha256"):
            raise ValueError("Replay accounting audit hash mismatch or audit unavailable")
        accounting = _read(audit_path)
        if (accounting.get("status") != run.get("accounting_status")
                or receipt.get("accounting_status") != accounting.get("status")
                or accounting.get("error_count") != 0 or accounting.get("errors")
                or receipt.get("accounting_error_count") != 0):
            raise ValueError("Replay independent accounting errors/status are not acceptable")
        for key in ("accounting_audit_sha256", "accounting_error_count"):
            if key in run and run[key] != receipt[key]:
                raise ValueError("Actual replay accounting evidence differs from receipt")
        acceptance = summary.get("acceptance_status", run.get("acceptance_status"))
        if acceptance not in {"research_only", "data_limited", "passed"}:
            raise ValueError("Replay acceptance is failed or unavailable")
        if accounting["status"] == "data_limited" and acceptance != "data_limited":
            raise ValueError("Data-limited accounting must retain overall data_limited qualification")
        if run.get("engine_parameters") != experiment["engine_parameters"]:
            raise ValueError("Replay changed frozen engine/profile parameters")
        if receipt.get("profile_sha256") != experiment["source_sha256"]["profile"]:
            raise ValueError("Replay profile hash mismatch")
        shared = receipt.get("shared_input_sha256", {})
        required = {"prices", "formula_phases", "trade_plans", "corporate_actions", "security_end_dates", "left_candidates"}
        if required - set(shared) or any(not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in shared.values()):
            raise ValueError("Replay requires hashes of every shared non-right-candidate input")
        if shared["left_candidates"] != experiment["source_sha256"]["left_candidates"]:
            raise ValueError("Replay changed the common left source")
        code_hashes = run.get("code_sha256")
        if (not re.fullmatch(r"[a-f0-9]{64}", str(run.get("engine_sha256", "")))
                or not isinstance(code_hashes, dict) or not code_hashes
                or any(not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in code_hashes.values())):
            raise ValueError("Replay engine/source fingerprint missing")
        # A historical experiment freezes its inputs and executable source, not
        # just agreement between replays. Equal substitutions in every receipt
        # must not turn a different experiment into an accepted comparison.
        if "common_inputs_sha256" in experiment:
            if any(shared[key] != experiment["common_inputs_sha256"]
                   for key in required - {"left_candidates"}):
                raise ValueError("Replay common inputs differ from the frozen experiment")
        if "code_sha256" in experiment:
            expected_code = experiment["code_sha256"]
            if code_hashes != expected_code:
                raise ValueError("Replay source differs from the frozen experiment")
            expected_engine = expected_code.get("stock_research/backtesting/engine.py")
            if expected_engine is not None and run["engine_sha256"] != expected_engine:
                raise ValueError("Replay engine differs from the frozen experiment")
        if receipt.get("input_bundle_sha256") != run.get("input_bundle_sha256") or not run.get("input_bundle_sha256"):
            raise ValueError("Replay receipt must bind the actual input bundle")
        if run["input_bundle_sha256"] in used_bundles or receipt["summary_sha256"] in used_summaries:
            raise ValueError("One completed bundle/summary cannot count as multiple control runs")
        used_bundles.add(run["input_bundle_sha256"])
        used_summaries.add(receipt["summary_sha256"])
        identity = {"shared": shared, "engine": run["engine_sha256"], "code": run["code_sha256"],
                    "acceptance": acceptance, "limitations": run.get("data_limitations", [])}
        if common is None:
            common = identity
        elif common != identity:
            raise ValueError("Replay inputs, engine, source, or data limitations differ")
        value = summary.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Replay metric is missing or not finite: {metric}")
        rows.append({"control_id": control, "metric": float(value), "summary_sha256": receipt["summary_sha256"]})
    random = [row["metric"] for row in rows if row["control_id"].startswith("random_")]
    by_control = {row["control_id"]: row["metric"] for row in rows}
    complete = len(random) >= 100 and set(expected) == seen
    baseline = by_control.get("baseline_six_factor")
    comparison = None
    if complete and baseline is not None:
        less = sum(value < baseline for value in random)
        ties = sum(value == baseline for value in random)
        upper_p = (1 + sum(value >= baseline for value in random)) / (len(random) + 1)
        comparison = {"baseline_percentile_midrank": 100 * (less + 0.5 * ties) / len(random),
                      "empirical_upper_tail_p_including_baseline": upper_p,
                      "baseline_in_top_five_percent": upper_p <= 0.05,
                      "interpretation": "Descriptive research comparison; not proof of factor attribution."}
    result = {"version": VERSION, "research_only": True, "strict_point_in_time": False,
              "status": "complete_research_comparison" if complete else "incomplete_replays",
              "metric": metric, "higher_is_better": True, "completed_random_replays": len(random),
              "missing_controls": sorted(set(expected) - seen), "comparison": comparison, "replays": rows,
              "experiment_sha256": file_sha256(experiment_path), "receipts_sha256": file_sha256(receipts_path),
              "evidence_limit": "External executor receipts attest common inputs; authenticity and full PIT are not certified."}
    _json(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Top50 selection controls; no broker/network access")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("generate")
    for name in ("panel", "membership", "source-manifest", "profile", "left-candidates", "output-directory"):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--seed-start", type=int, default=0)
    create.add_argument("--seed-count", type=int, default=100)
    compare = sub.add_parser("evaluate")
    for name in ("experiment", "receipts", "output"):
        compare.add_argument(f"--{name}", required=True)
    compare.add_argument("--metric", default="final_return_pct", choices=("final_return_pct", "sharpe", "sortino", "calmar"))
    args = parser.parse_args(argv)
    if args.action == "generate":
        result = generate(args.panel, args.membership, args.source_manifest, args.profile,
                          args.left_candidates, args.output_directory,
                          seeds=range(args.seed_start, args.seed_start + args.seed_count))
    else:
        result = evaluate(args.experiment, args.receipts, args.output, metric=args.metric)
    print(json.dumps({key: result[key] for key in ("status", "completed_random_replays")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
