"""Offline, predeclared diagnostics; every variant receives an audited receipt.

This executor permits the declared parameters to differ, unlike the selection
control comparator. It never selects a winning variant or changes production.
Local pickle inputs must be trusted. Historical membership is research-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from stock_research.backtesting.engine import run_portfolio_backtest
from stock_research.backtesting.frozen_replay import _dependency_versions, _offline
from stock_research.backtesting.inputs import file_sha256
from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting
from stock_research.reporting.portfolio_research_audit import audit_frames
from stock_research.research.historical_selection_controls import (
    DATA_KEYS, ROOT, _archive, _dump, _experiment_run_lock, _load, _read_json,
    _verify_archive, _window, _write_json,
)
from stock_research.research.quantsplaybook_compare import rank_weighted_factor_combination
from stock_research.research.quantsplaybook_hybrid import merge_hybrid_rows
from stock_research.research.selection_controls import FACTORS


VERSION = "fixed-research-experiments-v1"


def neighborhood_design():
    return [{"variant_id": f"top{top}_{tie}_offset{offset:02d}", "top_n": top,
             "tie_order": tie, "start_offset": offset}
            for top in (40, 50, 60) for tie in ("ascending", "descending")
            for offset in (0, 5, 20)]


def rank_neighborhood(frame, *, top_n, tie_order):
    if top_n not in (40, 50, 60) or tie_order not in {"ascending", "descending"}:
        raise ValueError("Only the registered neighborhood design is allowed")
    if frame.code.duplicated().any():
        raise ValueError("Duplicate dated neighborhood members")
    score = pd.to_numeric(frame["_baseline_score"], errors="coerce")
    if np.isinf(score).any():
        raise ValueError("Invalid neighborhood score")
    ranked = frame.assign(_score=score).dropna(subset=["_score"]).sort_values(
        ["_score", "code"], ascending=[False, tie_order == "ascending"])
    if len(ranked) < top_n:
        raise ValueError("Insufficient members for the registered Top60 neighborhood")
    return ranked.head(top_n), score.rank(pct=True, method="average")


def _source_controls(path):
    path = Path(path).resolve()
    parent = _read_json(path)
    archive = _verify_archive(path.parent / "source")
    if archive["sha256"] != parent["code_sha256"]:
        raise ValueError("Parent source archive differs")
    for name, expected in (("common_inputs.pkl", parent["common_inputs_sha256"]),
                           ("left_candidates.pkl", parent["source_sha256"]["left_candidates"])):
        if file_sha256(path.parent / name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    _window(parent["engine_parameters"]["requested_start"], parent["engine_parameters"]["end_date"])
    return path, parent


def prepare_neighborhood(parent_experiment, panel_directory, output_directory):
    """Top40/50/60 x deterministic ties x 0/5/20 actual sessions, all fixed."""
    parent_path, parent = _source_controls(parent_experiment)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dates = parent["dates"]
    if len(dates) <= 20:
        raise ValueError("Neighborhood needs at least 21 dated observation sessions")
    common = _load(parent_path.parent / "common_inputs.pkl")
    frames = common["engine_kwargs"]["price_frames"]
    amounts = []
    for code, frame in frames.items():
        observed = pd.DataFrame({"date": pd.to_datetime(frame["date"]), "code": code,
                                 "amount": pd.to_numeric(frame["amount"], errors="coerce")})
        if observed.date.duplicated().any() or not observed.date.is_monotonic_increasing:
            raise ValueError("Prepared ADV bars must be unique and ordered")
        observed["adv"] = observed.amount.where(observed.amount.gt(0)).rolling(20, min_periods=20).mean() * 1000
        amounts.append(observed.loc[observed.date.between(dates[0], dates[-1]), ["date", "code", "adv"]])
    style = pd.concat(amounts, ignore_index=True)
    del amounts, frames, common
    left = _load(parent_path.parent / "left_candidates.pkl")
    if sorted(left) != dates:
        raise ValueError("Shared left calendar differs")
    variant_snapshots = {(top, tie): {} for top in (40, 50, 60) for tie in ("ascending", "descending")}
    panel_directory = Path(panel_directory)
    model_row = next(row for row in parent["controls"] if row["control_id"] == "baseline_six_factor")
    model_manifest = _read_json(parent_path.parent / model_row["manifest"])
    if file_sha256(parent_path.parent / model_row["manifest"]) != model_row["manifest_sha256"]:
        raise ValueError("Parent baseline manifest changed")
    model_file = parent_path.parent / model_row["manifest"]
    model_bundle = model_file.parent / model_manifest["candidate_bundle"]["file"]
    if file_sha256(model_bundle) != model_manifest["candidate_bundle"]["sha256"]:
        raise ValueError("Parent baseline candidates changed")
    parent_baseline = _load(model_bundle)
    model = next(row["playbook_factor"] for rows in parent_baseline.values() for row in rows)
    actual_panels = {}
    for name, expected in parent["factor_panel_sha256"].items():
        path = panel_directory / name
        if file_sha256(path) != expected:
            raise ValueError(f"Cached factor panel differs from parent: {name}")
        cached = pd.read_pickle(path)
        panel = cached[["date", "code", *FACTORS]].copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.loc[panel.date.between(dates[0], dates[-1])]
        panel["code"] = panel.code.astype(str)
        if panel.duplicated(["date", "code"]).any():
            raise ValueError("Duplicate historical membership")
        indexed = panel.set_index(["date", "code"]).sort_index()
        indexed["_baseline_score"] = rank_weighted_factor_combination(
            indexed, {factor: 0.166666666667 for factor in FACTORS})
        panel = indexed.reset_index().merge(style, on=["date", "code"], how="left", validate="one_to_one")
        if panel.adv.isna().any():
            raise ValueError("Missing ADV history for otherwise eligible membership")
        panel = panel.loc[panel.adv.ge(30_000_000)]
        for date, group in panel.groupby("date", sort=True):
            day = str(date.date())
            for (top, tie), snapshots in variant_snapshots.items():
                selected, pct = rank_neighborhood(group, top_n=top, tie_order=tie)
                rows = [{"code": row.code, "name": row.code,
                         "strategy_part": f"quantsplaybook:{model}",
                         "selection_engine": "quantsplaybook_factor_only",
                         "selection_profile": "quantsplaybook_factor_only", "playbook_factor": model,
                         "playbook_factor_score": float(row["_score"]), "playbook_factor_rank": rank,
                         "candidate_score": round(float(pct.loc[index] * 100), 6),
                         "selected_for_trading": True, "signal_eligible": True,
                         "allow_right": True, "allow_left": False, "candidate_failure_reason": ""}
                        for rank, (index, row) in enumerate(selected.iterrows(), 1)]
                snapshots[day] = merge_hybrid_rows(rows, left[day], right_factor=model)
        actual_panels[name] = expected
        if file_sha256(path) != expected:
            raise ValueError(f"Panel changed while preparing: {name}")
        print(f"prepared fixed neighborhood {name}", flush=True)
    if any(sorted(value) != dates for value in variant_snapshots.values()):
        raise ValueError("Neighborhood candidate calendar incomplete")
    # The offset-zero Top50 ascending selection must reproduce its parent's
    # candidate fields (the research control label is excluded from this check).
    rebuilt = variant_snapshots[(50, "ascending")]
    for day in dates:
        expected = merge_hybrid_rows(parent_baseline[day], left[day], right_factor=model)
        strip = lambda rows: [{key: value for key, value in row.items() if key != "selection_control"} for row in rows]
        if strip(rebuilt[day]) != strip(expected):
            raise ValueError(f"Neighborhood baseline selection changed: {day}")
    shutil.copy2(parent_path.parent / "common_inputs.pkl", output / "common_inputs.pkl")
    records = []
    for (top, tie), snapshots in variant_snapshots.items():
        path = output / f"candidates_top{top}_{tie}.pkl"
        _dump(path, snapshots)
    for design in neighborhood_design():
        path = output / f"candidates_top{design['top_n']}_{design['tie_order']}.pkl"
        records.append({**design, "candidates": path.name, "candidates_sha256": file_sha256(path),
                        "parameter_overrides": {"requested_start": dates[design["start_offset"]]},
                        "research_policy": None})
    return _finish_prepare(output, parent["engine_parameters"], records,
                           {"parent_experiment_sha256": file_sha256(parent_path),
                            "parent_common_inputs_sha256": parent["common_inputs_sha256"],
                            "factor_panel_sha256": actual_panels}, "R02-neighborhood")


def prepare_parameters(source_bundle, output_directory, variants, *, family):
    """Freeze explicit variants on precisely the original audited input bundle."""
    source = Path(source_bundle).resolve()
    manifest = _read_json(source / "manifest.json")
    if file_sha256(source / "inputs.pkl") != manifest["input_bundle_sha256"]:
        raise ValueError("Source bundle hash mismatch")
    payload = _load(source / "inputs.pkl")
    kwargs = payload["engine_kwargs"]
    parameters = {key: value for key, value in kwargs.items() if key not in DATA_KEYS}
    if parameters != manifest["engine_parameters"]:
        raise ValueError("Source parameters differ from manifest")
    _window(parameters["requested_start"], parameters["end_date"])
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = kwargs.pop("candidate_snapshots")
    _dump(output / "candidates.pkl", candidates)
    _dump(output / "common_inputs.pkl", payload)
    records = []
    for variant in variants:
        overrides = dict(variant.get("parameter_overrides", {}))
        if set(overrides) - set(parameters):
            raise ValueError("Variant contains unknown baseline parameters")
        _window(overrides.get("requested_start", parameters["requested_start"]),
                overrides.get("end_date", parameters["end_date"]))
        records.append({"variant_id": variant["variant_id"], "parameter_overrides": overrides,
                        "research_policy": variant.get("research_policy"), "candidates": "candidates.pkl",
                        "candidates_sha256": file_sha256(output / "candidates.pkl")})
    return _finish_prepare(output, parameters, records,
                           {"original_bundle_sha256": manifest["input_bundle_sha256"]}, family)


def _finish_prepare(output, parameters, records, provenance, family):
    ids = [row["variant_id"] for row in records]
    if not ids or len(set(ids)) != len(ids) or any(not value.replace("_", "").isalnum() for value in ids):
        raise ValueError("Variant identities must be unique safe names")
    hashes = _archive(output)
    document = {"version": VERSION, "family": family, "research_only": True,
                "strict_point_in_time": False, "baseline_parameters": parameters, "variants": records,
                "common_inputs_sha256": file_sha256(output / "common_inputs.pkl"),
                "source_sha256": hashes, "dependencies": _dependency_versions(), "provenance": provenance,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "completed_replays": 0,
                "limitations": ["Observed historical data; diagnostic only, no historical winner promotion.",
                                "Financial revisions, industry snapshots, limits and cash payment gaps remain."]}
    _write_json(output / "experiment.json", document)
    return document


def _verify(path):
    path = Path(path).resolve()
    experiment = _read_json(path)
    if experiment.get("version") != VERSION:
        raise ValueError("Unknown fixed experiment contract")
    archive = _verify_archive(path.parent / "source")
    if archive["sha256"] != experiment["source_sha256"] or archive["dependencies"] != experiment["dependencies"]:
        raise ValueError("Experiment source/environment changed")
    if file_sha256(path.parent / "common_inputs.pkl") != experiment["common_inputs_sha256"]:
        raise ValueError("Experiment common inputs changed")
    for row in experiment["variants"]:
        if Path(row["candidates"]).name != row["candidates"]:
            raise ValueError("Candidate path must be an experiment-local file")
        if file_sha256(path.parent / row["candidates"]) != row["candidates_sha256"]:
            raise ValueError("Experiment candidates changed")
    return path, experiment


def worker(experiment_path, variant_id):
    path, experiment = _verify(experiment_path)
    if ROOT.resolve() != (path.parent / "source").resolve():
        raise ValueError("Worker must execute the archived source")
    variant = next(row for row in experiment["variants"] if row["variant_id"] == variant_id)
    kwargs = _load(path.parent / "common_inputs.pkl")["engine_kwargs"]
    parameters = {key: value for key, value in kwargs.items() if key not in DATA_KEYS}
    if parameters != experiment["baseline_parameters"]:
        raise ValueError("Common parameters differ from frozen baseline")
    kwargs.update(variant["parameter_overrides"])
    parameters.update(variant["parameter_overrides"])
    start, end = _window(parameters["requested_start"], parameters["end_date"])
    candidates = _load(path.parent / variant["candidates"])
    kwargs["candidate_snapshots"] = {date: rows for date, rows in candidates.items() if start <= pd.Timestamp(date) <= end}
    fingerprint = hashlib.sha256((file_sha256(path) + experiment["common_inputs_sha256"]
                                  + variant["candidates_sha256"] + variant_id).encode()).hexdigest()
    output = path.parent / "runs" / variant_id
    output.mkdir(parents=True, exist_ok=True)
    run = {"status": "running", "variant_id": variant_id, "research_only": True,
           "experiment_sha256": file_sha256(path), "input_bundle_sha256": fingerprint,
           "engine_parameters": parameters, "research_policy": variant["research_policy"],
           "started_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run_manifest.json", run)
    try:
        with _offline():
            if variant["research_policy"] is None:
                result = run_portfolio_backtest(**kwargs)
            else:
                from stock_research.research.portfolio_replay import run_portfolio_research
                result = run_portfolio_research(**kwargs, research_policy=variant["research_policy"])
            audit = audit_portfolio_accounting(result, kwargs["price_frames"], kwargs["corporate_actions"], parameters)
        _verify(path)
        _write_json(output / "accounting_audit.json", audit)
        for key, suffix in (("equity_curve", "equity"), ("trade_ledger", "trades"), ("events", "events")):
            pd.DataFrame(result[key]).to_csv(output / f"{suffix}.csv", index=False, encoding="utf-8-sig")
        diagnostics = audit_frames(pd.DataFrame(result["equity_curve"]), pd.DataFrame(result["trade_ledger"]),
                                   path.parent / "no_external_database.sqlite3", initial_equity=1.)
        summary = {key: value for key, value in result.items() if key not in {"equity_curve", "events"}}
        summary.update(research_diagnostics=diagnostics, research_only=True, strict_point_in_time=False)
        _write_json(output / "summary.json", summary)
        if audit["error_count"] or audit["status"] not in {"passed", "data_limited"}:
            raise ValueError("Independent accounting audit failed")
        run.update(status="finished", accounting_status=audit["status"], accounting_error_count=0,
                   output_sha256={name: file_sha256(output / name) for name in
                                  ("accounting_audit.json", "summary.json", "equity.csv", "trades.csv", "events.csv")},
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output / "receipt.json", run)
        print(json.dumps({"variant_id": variant_id, "return_pct": result.get("final_return_pct"),
                          "accounting_status": audit["status"]}), flush=True)
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write_json(output / "run_manifest.json", run)
    return 0


def compare(experiment_path):
    path, experiment = _verify(experiment_path)
    results = []
    for variant in experiment["variants"]:
        directory = path.parent / "runs" / variant["variant_id"]
        if not (directory / "receipt.json").is_file():
            continue
        receipt = _read_json(directory / "receipt.json")
        effective = {**experiment["baseline_parameters"], **variant["parameter_overrides"]}
        fingerprint = hashlib.sha256((file_sha256(path) + experiment["common_inputs_sha256"]
                                      + variant["candidates_sha256"] + variant["variant_id"]).encode()).hexdigest()
        if (receipt["status"] != "finished" or receipt["accounting_error_count"] != 0
                or receipt["accounting_status"] not in {"passed", "data_limited"}
                or receipt["experiment_sha256"] != file_sha256(path)
                or receipt["input_bundle_sha256"] != fingerprint
                or receipt["engine_parameters"] != effective
                or receipt["research_policy"] != variant["research_policy"]):
            raise ValueError("Receipt disagrees with frozen experiment")
        required = {"accounting_audit.json", "summary.json", "equity.csv", "trades.csv", "events.csv"}
        if set(receipt["output_sha256"]) != required:
            raise ValueError("Incomplete receipt output evidence")
        for name, expected in receipt["output_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError("Completed experiment output changed")
        audit = _read_json(directory / "accounting_audit.json")
        if audit["error_count"] or audit.get("errors") or audit["status"] != receipt["accounting_status"]:
            raise ValueError("Receipt does not attest a zero-error independent audit")
        summary = _read_json(directory / "summary.json")
        risk = summary["research_diagnostics"]["risk"]
        results.append({"variant_id": variant["variant_id"], "parameters": effective,
                        "research_policy": variant["research_policy"], "accounting_status": audit["status"],
                        "return_pct": summary.get("final_return_pct"), "maximum_drawdown": risk.get("maximum_drawdown"),
                        "sharpe": risk.get("sharpe"), "receipt_sha256": file_sha256(directory / "receipt.json")})
    distribution = None
    if len(results) == len(experiment["variants"]):
        values = np.array([row["return_pct"] for row in results], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite completed variant return")
        distribution = {"minimum": float(values.min()), "median": float(np.median(values)),
                        "maximum": float(values.max()), "percentile_05": float(np.quantile(values, .05)),
                        "percentile_95": float(np.quantile(values, .95)), "positive_fraction": float((values > 0).mean())}
    report = {"research_only": True, "strict_point_in_time": False, "family": experiment["family"],
              "planned_replays": len(experiment["variants"]), "completed_replays": len(results),
              "experiment_sha256": file_sha256(path), "results": results, "return_distribution_pct": distribution,
              "note": "Fixed retrospective diagnosis; no winning variant selected or promoted."}
    _write_json(path.parent / "comparison.json", report)
    return report


def run(experiment_path, *, jobs=1):
    if type(jobs) is not int or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer from 1 to 8")
    path, experiment = _verify(experiment_path)
    with _experiment_run_lock(path.parent):
        report = compare(path)
        complete = {row["variant_id"] for row in report["results"]}
        pending = [row["variant_id"] for row in experiment["variants"] if row["variant_id"] not in complete]
        def execute(variant):
            subprocess.run([sys.executable, "-m", "stock_research.research.fixed_experiments", "worker",
                            "--experiment", str(path), "--variant", variant], cwd=path.parent / "source", check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        # Process only one bounded batch at a time so a failed audit cannot
        # silently launch the remaining registered variants.
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for index in range(0, len(pending), jobs):
                futures = [executor.submit(execute, variant) for variant in pending[index:index + jobs]]
                for future in as_completed(futures):
                    future.result()
                    compare(path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-neighborhood")
    prepare.add_argument("--parent-experiment", required=True)
    prepare.add_argument("--panel-directory", required=True)
    prepare.add_argument("--output-directory", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--experiment", required=True)
    execute.add_argument("--jobs", type=int, default=1)
    child = sub.add_parser("worker")
    child.add_argument("--experiment", required=True)
    child.add_argument("--variant", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare-neighborhood":
        prepare_neighborhood(args.parent_experiment, args.panel_directory, args.output_directory)
        return 0
    if args.action == "run":
        return run(args.experiment, jobs=args.jobs)
    return worker(args.experiment, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
