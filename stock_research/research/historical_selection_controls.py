"""Frozen, offline selection-control replays, bounded by 2026-07-21.

Pickles are trusted local research artifacts, never an interchange format for
untrusted uploads. Candidate generation and actual completed runs stay separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from stock_research.backtesting.engine import run_portfolio_backtest
from stock_research.backtesting.frozen_replay import _dependency_versions, _offline, _readonly_snapshot
from stock_research.backtesting.inputs import file_sha256
from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting
from stock_research.reporting.portfolio_research_audit import audit_frames
from stock_research.research.quantsplaybook_compare import rank_weighted_factor_combination
from stock_research.research.quantsplaybook_hybrid import merge_hybrid_rows
from stock_research.research.selection_controls import FACTORS, VERSION, _selected, evaluate
from stock_research.storage import TushareRepository
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION


ROOT = Path(__file__).resolve().parents[2]
CUTOFF = pd.Timestamp("2026-07-21")
DATA_KEYS = {"price_frames", "candidate_snapshots", "formula_phases", "trade_plans", "corporate_actions", "security_end_dates"}


def _write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _dump(path, value):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _window(start, end):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or start > end or end > CUTOFF:
        raise ValueError("Historical controls require a valid interval ending on/before 2026-07-21")
    return start, end


def _peak_rss_mb():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged",
                                                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.peak_ws / 1024 ** 2


def _archive(output):
    archive = output / "source"
    archive.mkdir()
    files = sorted((ROOT / "stock_research").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "config/strategy_baseline.json"]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT).as_posix()
        expected = file_sha256(source)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(source) != expected or file_sha256(target) != expected:
            raise ValueError(f"Source changed while archiving: {relative}")
        hashes[relative] = expected
    _write_json(archive / "source_manifest.json", {"sha256": hashes, "dependencies": _dependency_versions()})
    return hashes


def _verify_archive(source):
    manifest = _read_json(source / "source_manifest.json")
    if _dependency_versions() != manifest["dependencies"]:
        raise ValueError("Archived execution environment differs")
    for relative, expected in manifest["sha256"].items():
        if file_sha256(source / relative) != expected:
            raise ValueError(f"Archived source changed: {relative}")
    return manifest


def _trim_calendar(kwargs, start, end):
    counts = {}
    phases = kwargs["formula_phases"]
    kwargs["formula_phases"] = {date: value for date, value in phases.items() if start <= pd.Timestamp(date) <= end}
    counts["formula_phases_removed"] = len(phases) - len(kwargs["formula_phases"])
    actions = kwargs["corporate_actions"]
    trimmed = {}
    for code, rows in actions.items():
        retained = [row for row in rows if start <= pd.to_datetime(str(row.get("ex_date")), errors="coerce") <= end]
        if retained:
            trimmed[code] = retained
    counts["corporate_actions_removed"] = sum(map(len, actions.values())) - sum(map(len, trimmed.values()))
    kwargs["corporate_actions"] = trimmed
    ends = kwargs["security_end_dates"]
    kwargs["security_end_dates"] = {code: date for code, date in ends.items() if pd.Timestamp(date) <= end}
    counts["future_security_end_dates_removed"] = len(ends) - len(kwargs["security_end_dates"])
    if kwargs["trade_plans"].get("plans"):
        raise ValueError("Historical controls require explicitly dated plans; undated nonempty plans are not accepted")
    return counts


def trim_prepared_calendar(experiment_path):
    """Finalize an unexecuted pilot's date boundary with before/after hash evidence."""
    experiment_path = Path(experiment_path).resolve()
    directory = experiment_path.parent
    if (directory / "runs").exists():
        raise ValueError("Cannot change common inputs after any replay has started")
    experiment = _read_json(experiment_path)
    archive = _verify_archive(directory / "source")
    if archive["sha256"] != experiment["code_sha256"]:
        raise ValueError("Source archive differs from experiment")
    common_path = directory / "common_inputs.pkl"
    original_hash = file_sha256(common_path)
    if original_hash != experiment["common_inputs_sha256"]:
        raise ValueError("Common inputs changed before calendar finalization")
    common = _load(common_path)
    start, end = _window(experiment["engine_parameters"]["requested_start"], experiment["engine_parameters"]["end_date"])
    counts = _trim_calendar(common["engine_kwargs"], start, end)
    if not any(counts.values()):
        return counts
    _dump(common_path.with_suffix(".tmp"), common)
    common_path.with_suffix(".tmp").replace(common_path)
    experiment["calendar_finalization"] = {"before_sha256": original_hash, "removed": counts}
    experiment["common_inputs_sha256"] = file_sha256(common_path)
    _write_json(experiment_path, experiment)
    return counts


def compare_baseline(experiment_path, original_bundle):
    """Record exact old/new baseline selection differences, without using returns."""
    experiment_path = Path(experiment_path).resolve()
    directory, original_bundle = experiment_path.parent, Path(original_bundle)
    experiment = _read_json(experiment_path)
    original_path = original_bundle / "inputs.pkl"
    if file_sha256(original_path) != experiment["source_sha256"]["original_bundle"]:
        raise ValueError("Original baseline bundle hash mismatch")
    record = next(row for row in experiment["controls"] if row["control_id"] == "baseline_six_factor")
    manifest_path = directory / record["manifest"]
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    manifest = _read_json(manifest_path)
    candidate_path = manifest_path.parent / "candidates.pkl"
    if file_sha256(candidate_path) != manifest["candidate_bundle"]["sha256"]:
        raise ValueError("Baseline candidate bundle changed")
    original = _load(original_path)["engine_kwargs"]["candidate_snapshots"]
    generated = _load(candidate_path)
    left = _load(directory / "left_candidates.pkl")
    details = []
    left_preserved = True
    for date in experiment["dates"]:
        old = {row["code"]: row for row in original[date] if row.get("allow_right") is True}
        new = {row["code"]: row for row in generated[date]}
        old_left = [row for row in original[date] if row.get("allow_left") is True]
        left_preserved &= json.dumps(old_left, sort_keys=True, default=str) == json.dumps(left[date], sort_keys=True, default=str)
        common = sorted(set(old) & set(new))
        rank_changed = [code for code in common if old[code].get("playbook_factor_rank") != new[code].get("playbook_factor_rank")]
        max_score_error = max((abs(float(old[code]["candidate_score"]) - float(new[code]["candidate_score"]))
                               for code in common), default=0.)
        details.append({"date": date, "old_count": len(old), "new_count": len(new),
                        "only_old": sorted(set(old) - set(new)), "only_new": sorted(set(new) - set(old)),
                        "common_rank_changed": rank_changed, "max_common_candidate_score_error": max_score_error})
    report = {"research_only": True, "experiment_sha256": file_sha256(experiment_path),
              "original_bundle_sha256": file_sha256(original_path),
              "generated_candidates_sha256": file_sha256(candidate_path),
              "factor_panel_sha256": experiment["factor_panel_sha256"],
              "dates": len(details), "selection_changed_dates": sum(bool(row["only_old"] or row["only_new"]) for row in details),
              "ranking_changed_dates": sum(bool(row["common_rank_changed"]) for row in details),
              "original_joint_left_right_rows": sum(bool(row.get("allow_left") is True and row.get("allow_right") is True)
                                                   for date in experiment["dates"] for row in original[date]),
              "shared_left_exactly_preserved": left_preserved, "daily_differences": details}
    _write_json(directory / "baseline_selection_comparison.json", report)
    return report


def fork_for_execution_equivalence(experiment_path, output_directory, *, control="baseline_six_factor"):
    """Copy identical frozen inputs and archive the current code for a parity replay."""
    experiment_path = Path(experiment_path).resolve()
    source = experiment_path.parent
    experiment = _read_json(experiment_path)
    _verify_archive(source / "source")
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    record = next(row for row in experiment["controls"] if row["control_id"] == control)
    for name, expected in (("common_inputs.pkl", experiment["common_inputs_sha256"]),
                           ("profile.json", experiment["source_sha256"]["profile"]),
                           ("left_candidates.pkl", experiment["source_sha256"]["left_candidates"]),
                           (record["manifest"], record["manifest_sha256"])):
        if file_sha256(source / name) != expected:
            raise ValueError(f"Parent experiment input changed: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        if file_sha256(target) != expected:
            raise ValueError(f"Copied experiment input differs: {name}")
    manifest = _read_json(source / record["manifest"])
    candidate_relative = Path(record["manifest"]).parent / "candidates.pkl"
    if file_sha256(source / candidate_relative) != manifest["candidate_bundle"]["sha256"]:
        raise ValueError("Parent candidate bundle changed")
    shutil.copy2(source / candidate_relative, destination / candidate_relative)
    if file_sha256(destination / candidate_relative) != manifest["candidate_bundle"]["sha256"]:
        raise ValueError("Copied candidate bundle differs")
    experiment.update(controls=[record], purpose="execution_equivalence_validation_only",
                      parent_experiment_sha256=file_sha256(experiment_path),
                      code_sha256=_archive(destination), dependencies=_dependency_versions())
    _write_json(destination / "experiment.json", experiment)
    return destination / "experiment.json"


def compare_execution_results(original_directory, updated_directory, output_path):
    """Require exact serialized equity, fills and events plus identical inputs."""
    original_directory, updated_directory = Path(original_directory), Path(updated_directory)
    old, new = (_read_json(path / "summary.json") for path in (original_directory, updated_directory))
    for key in ("input_bundle_sha256", "engine_parameters"):
        if old["frozen_run"][key] != new["frozen_run"][key]:
            raise ValueError(f"Execution comparison changed {key}")
    hashes = {name: {"original": file_sha256(original_directory / name),
                     "updated": file_sha256(updated_directory / name)}
              for name in ("equity.csv", "trades.csv", "events.csv")}
    if any(item["original"] != item["updated"] for item in hashes.values()):
        raise ValueError("Execution parity failed: equity, trades or events differ")
    for path in (original_directory, updated_directory):
        audit = _read_json(path / "accounting_audit.json")
        if audit.get("status") not in {"passed", "data_limited"} or audit.get("error_count") != 0:
            raise ValueError("Execution parity requires zero-error independent accounting")
    report = {"status": "identical_equity_trades_events", "research_only": True,
              "common_input_bundle_sha256": old["frozen_run"]["input_bundle_sha256"],
              "original_summary_sha256": file_sha256(original_directory / "summary.json"),
              "updated_summary_sha256": file_sha256(updated_directory / "summary.json"),
              "csv_sha256": hashes, "original_runtime_seconds": old["frozen_run"]["elapsed_seconds"],
              "updated_runtime_seconds": new["frozen_run"]["elapsed_seconds"],
              "original_profiled": old["frozen_run"].get("execution_profiled", False),
              "updated_profiled": new["frozen_run"].get("execution_profiled", False)}
    _write_json(output_path, report)
    return report


def prepare(*, panel_directory, baseline_bundle, price_directory, database, profile,
            output_directory, start_date="2021-01-01", end_date="2026-07-21", seed_count=100):
    """Freeze common inputs and generate all declared controls from dated cached membership."""
    start, end = _window(start_date, end_date)
    if seed_count < 100:
        raise ValueError("Declare at least 100 random controls; fewer completed runs never support a percentile")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    bundle_root, price_directory, panel_directory = map(Path, (baseline_bundle, price_directory, panel_directory))
    original_manifest = _read_json(bundle_root / "manifest.json")
    original_path = bundle_root / "inputs.pkl"
    if file_sha256(original_path) != original_manifest["input_bundle_sha256"]:
        raise ValueError("Original frozen bundle hash mismatch")
    profile_path = Path(profile)
    profile_document = _read_json(profile_path)
    original = _load(original_path)
    parameters = {key: value for key, value in original["engine_kwargs"].items() if key not in DATA_KEYS}
    if parameters != profile_document["engine_parameters"]:
        raise ValueError("Baseline profile differs from the original frozen engine parameters")
    parameters.update(requested_start=str(start.date()), end_date=str(end.date()))
    derived_profile = {"research_only": True, "engine_parameters": parameters,
                       "source_profile_sha256": file_sha256(profile_path),
                       "date_override_only": [str(start.date()), str(end.date())]}
    _write_json(output / "profile.json", derived_profile)
    original_snapshots = original["engine_kwargs"].pop("candidate_snapshots")
    dates = sorted(date for date in original_snapshots if str(start.date()) <= date <= str(end.date()))
    if not dates:
        raise ValueError("No original candidate dates in requested historical interval")
    left = {date: [row for row in original_snapshots[date] if row.get("allow_left") is True] for date in dates}
    _dump(output / "left_candidates.pkl", left)
    left_hash = file_sha256(output / "left_candidates.pkl")
    coverage_path = panel_directory.parent / "data_coverage.json"
    if not coverage_path.is_file():
        raise ValueError("Cached factor membership requires its historical data_coverage.json provenance")
    coverage = _read_json(coverage_path)
    if coverage.get("old_candidate_pool_used") is not False:
        raise ValueError("Current/prior candidate pools cannot substitute for dated full-market membership")
    factor_hashes, frames, styles, price_hashes = {}, {}, [], {}
    warmup = start - pd.Timedelta(days=700)
    original_frames = original["engine_kwargs"].pop("price_frames")
    original_symbols = set(original_frames)
    files = sorted(price_directory.glob("*.pkl"))
    if not files:
        raise ValueError("Prepared full-market price files are unavailable")
    for index, path in enumerate(files, 1):
        code = path.stem.replace("_", ".")
        expected = file_sha256(path)
        price_hashes[path.name] = expected
        # Preserve original frozen execution rows for the old baseline universe.
        frame = original_frames.pop(code, None)
        if frame is None:
            frame = pd.read_pickle(path)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame.date.between(warmup, end)].sort_values("date").reset_index(drop=True)
        if frame.empty:
            continue
        required = {"date", "amount", "raw_close", "qfq_to_raw_factor", "ma5", "ma10", "ma20", "ma60", "ma120"}
        if required - set(frame) or frame.date.duplicated().any():
            raise ValueError(f"Prepared execution input is incomplete: {code}")
        adjustment = pd.to_numeric(frame["qfq_to_raw_factor"], errors="coerce")
        if not np.isfinite(adjustment).all() or adjustment.le(0).any():
            raise ValueError(f"Invalid execution adjustment factors: {code}")
        amount = pd.to_numeric(frame["amount"], errors="coerce").where(lambda values: values > 0) * 1000
        style = pd.DataFrame({"date": frame.date, "code": code,
                              "average_amount_20_yuan": amount.rolling(20, min_periods=20).mean(),
                              "close": pd.to_numeric(frame["raw_close"], errors="coerce")})
        styles.append(style.loc[style.date.between(start, end)])
        frames[code] = frame
        if file_sha256(path) != expected:
            raise ValueError(f"Prepared prices changed while freezing: {path}")
        if index % 1000 == 0:
            print(f"prepared prices {index}/{len(files)} peak_rss_mb={_peak_rss_mb()}", flush=True)
    del original_frames
    style_panel = pd.concat(styles, ignore_index=True)
    del styles
    style_panel["code"] = style_panel["code"].astype("category")
    panels = []
    with _readonly_snapshot(database) as snapshot_database:
        repository = TushareRepository(snapshot_database)
        for year in range(start.year, end.year + 1):
            path = panel_directory / f"factor_panel_{year}.pkl"
            expected = file_sha256(path)
            cached = pd.read_pickle(path)
            if {"date", "code", *FACTORS} - set(cached):
                raise ValueError(f"Required frozen factors unavailable: {path}")
            # Do not carry future_return or label_date into selection inputs.
            panel = cached[["date", "code", *FACTORS]].copy()
            del cached
            panel["date"] = pd.to_datetime(panel["date"])
            panel["code"] = panel["code"].astype(str)
            panel = panel.loc[panel.date.between(start, end)]
            if panel[["date", "code"]].duplicated().any():
                raise ValueError("Historical PIT membership has duplicate date/code rows")
            indexed = panel.set_index(["date", "code"]).sort_index()
            # Use the original arithmetic, including available-factor weight renormalization.
            indexed["_baseline_score"] = rank_weighted_factor_combination(indexed, {factor: 0.166666666667 for factor in FACTORS})
            panel = indexed.reset_index().merge(style_panel, how="left", on=["date", "code"], validate="one_to_one")
            if panel["average_amount_20_yuan"].isna().any():
                missing = panel.loc[panel.average_amount_20_yuan.isna(), ["date", "code"]].head(10).to_dict("records")
                raise ValueError(f"PIT members lack ADV20 coverage; not silently excluded: {missing}")
            panel = panel.loc[panel.average_amount_20_yuan.ge(30_000_000)].copy()
            rows = snapshot_database.connect(read_only=True).execute(
                "SELECT ts_code, trade_date, json_extract(payload_json, '$.turnover_rate'), "
                "json_extract(payload_json, '$.total_mv') FROM raw.tushare_dataset_rows "
                "WHERE dataset='daily_basic' AND trade_date>=? AND trade_date<=? ORDER BY trade_date,ts_code",
                [max(str(start.date()), f"{year}-01-01"), min(str(end.date()), f"{year}-12-31")],
            ).fetchall()
            basic = pd.DataFrame(rows, columns=["ts_code", "date", "turnover_rate", "market_cap_yuan"])
            basic["date"] = pd.to_datetime(basic["date"])
            basic["code"] = basic.ts_code.str[-2:].str.lower() + "." + basic.ts_code.str[:6]
            basic["market_cap_yuan"] = pd.to_numeric(basic["market_cap_yuan"], errors="coerce") * 10000
            basic["turnover_rate"] = pd.to_numeric(basic["turnover_rate"], errors="coerce")
            if basic[["date", "code"]].duplicated().any():
                raise ValueError("Conflicting daily_basic style observations")
            panel = panel.merge(basic.drop(columns="ts_code"), on=["date", "code"], how="left", validate="one_to_one")
            panel["date"] = panel.date.dt.strftime("%Y-%m-%d")
            panels.append(panel)
            factor_hashes[path.name] = expected
            if file_sha256(path) != expected:
                raise ValueError(f"Cached factors changed during preparation: {path}")
            print(f"prepared factor year={year} ADV_members={len(panel)}", flush=True)
        actions = original["engine_kwargs"]["corporate_actions"]
        extra_codes = sorted(set(frames) - original_symbols)
        fresh_actions = repository.load_dividend_actions(extra_codes, start_date=str(start.date()), end_date=str(end.date()))
        # Existing frozen symbols keep their original action bytes; add only new symbols.
        actions.update({code: values for code, values in fresh_actions.items() if code not in original_symbols})
        basic_master = repository.load_stock_basic_universe()
        ends = original["engine_kwargs"]["security_end_dates"]
        ends.update({str(row.code): pd.Timestamp(row.delist_date) for row in basic_master.itertuples()
                     if row.code in frames and row.code not in original_symbols and pd.notna(row.delist_date)})
    del style_panel
    panel = pd.concat(panels, ignore_index=True)
    del panels
    if set(panel.date) != set(dates):
        raise ValueError("Dated common universe does not cover the original observation calendar")
    if panel.groupby("date").size().lt(50).any():
        raise ValueError("Common ADV universe has fewer than 50 securities")
    engine_kwargs = {**original["engine_kwargs"], **parameters, "price_frames": frames}
    calendar_trim = _trim_calendar(engine_kwargs, start, end)
    common = {"engine_kwargs": engine_kwargs, "stock_basic_display_only": original.get("stock_basic_display_only")}
    _dump(output / "common_inputs.pkl", common)
    common_hash = file_sha256(output / "common_inputs.pkl")
    del original, engine_kwargs, common, frames
    gc.collect()
    controls = [("baseline_six_factor", None), ("low_turnover", None), ("low_residual_volatility", None)]
    controls += [(f"random_{seed:04d}", seed) for seed in range(seed_count)]
    groups = {date: group.sort_values("code") for date, group in panel.groupby("date", sort=True)}
    del panel
    style_names = ("turnover_rate", "average_amount_20_yuan", FACTORS[-1], "close", "market_cap_yuan")
    for group in groups.values():
        for name in style_names:
            group[f"_style_pct_{name}"] = group[name].rank(pct=True)
    model = profile_document["model_id"]
    records, exposures = [], []
    baseline_differences = []
    for control, seed in controls:
        destination = output / "candidates" / control
        destination.mkdir(parents=True)
        selected_snapshots = {}
        for date in dates:
            group = groups[date]
            selected, percentiles = _selected(group, control, seed)
            rows = []
            for rank, (index, row) in enumerate(selected.iterrows(), 1):
                rows.append({"code": row.code, "name": row.code, "strategy_part": f"quantsplaybook:{model}",
                             "selection_engine": "quantsplaybook_factor_only", "selection_profile": "quantsplaybook_factor_only",
                             "playbook_factor": model, "playbook_factor_score": float(percentiles.loc[index]) if seed is not None else float(row["_score"]),
                             "playbook_factor_rank": rank, "candidate_score": round(float(percentiles.loc[index] * 100), 6),
                             "selected_for_trading": True, "signal_eligible": True, "allow_right": True, "allow_left": False,
                             "candidate_failure_reason": "", "selection_control": control})
            selected_snapshots[date] = rows
            if control == "baseline_six_factor":
                old_codes = {row["code"] for row in original_snapshots[date] if row.get("allow_right") is True}
                if old_codes != {row["code"] for row in rows}:
                    baseline_differences.append(date)
            exposure = {"control_id": control, "date": date, "eligible_count": len(group), "selected_count": len(rows)}
            for name in style_names:
                exposure[f"{name}_coverage"] = int(selected[name].notna().sum())
                exposure[f"{name}_median"] = float(selected[name].median())
                exposure[f"{name}_median_percentile"] = float(selected[f"_style_pct_{name}"].median())
            exposures.append(exposure)
        candidate_path = destination / "candidates.pkl"
        _dump(candidate_path, selected_snapshots)
        manifest = {"version": SNAPSHOT_VERSION, "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
                    "control_version": VERSION, "control_id": control, "seed": seed,
                    "candidate_bundle": {"file": "candidates.pkl", "sha256": file_sha256(candidate_path)},
                    "dates": dates, "top_n": 50, "research_only": True, "strict_point_in_time": False}
        _write_json(destination / "manifest.json", manifest)
        records.append({"control_id": control, "seed": seed,
                        "manifest": (destination / "manifest.json").relative_to(output).as_posix(),
                        "manifest_sha256": file_sha256(destination / "manifest.json")})
        print(f"generated {control}; no completed replay implied", flush=True)
    pd.DataFrame(exposures).to_csv(output / "style_exposures.csv", index=False, encoding="utf-8-sig")
    code_hashes = _archive(output)
    experiment = {"version": VERSION, "research_only": True, "strict_point_in_time": False,
                  "status": "candidates_generated_not_replayed", "completed_random_replays": 0,
                  "dates": dates, "seeds": list(range(seed_count)), "controls": records,
                  "engine_parameters": parameters, "common_inputs_sha256": common_hash,
                  "source_sha256": {"profile": file_sha256(output / "profile.json"), "left_candidates": left_hash,
                                    "original_bundle": original_manifest["input_bundle_sha256"],
                                    "historical_membership_coverage": file_sha256(coverage_path)},
                  "factor_panel_sha256": factor_hashes, "prepared_price_file_sha256": price_hashes,
                  "code_sha256": code_hashes, "dependencies": _dependency_versions(),
                  "baseline_selection_differs_from_old_dates": baseline_differences,
                  "historical_cutoff": "2026-07-21", "source_membership_provenance": coverage,
                  "calendar_trim": calendar_trim,
                  "limitations": ["Historical cached membership and financial/industry revisions remain research-only.",
                                  "Factor warm-up NaNs retain original available-factor weight renormalization.",
                                  "Random uses the whole ADV pool; simple controls rank only their available observations.",
                                  "All controls share expanded execution prices; the old reported return is not reused.",
                                  "Original price-limit gaps and corporate-action payment-date gaps are not repaired by this experiment."],
                  "preparation_seconds": time.perf_counter() - begin, "peak_rss_mb": _peak_rss_mb()}
    _write_json(output / "experiment.json", experiment)
    return experiment


def worker(experiment_path, control_id, *, profile_execution=False):
    """Run inside the archived source directory, with database and network disabled."""
    experiment_path = Path(experiment_path).resolve()
    directory = experiment_path.parent
    experiment = _read_json(experiment_path)
    _window(experiment["engine_parameters"]["requested_start"], experiment["engine_parameters"]["end_date"])
    archive = _verify_archive(directory / "source")
    if archive["sha256"] != experiment["code_sha256"] or archive["dependencies"] != experiment["dependencies"]:
        raise ValueError("Archived source/environment differs from the experiment")
    if file_sha256(directory / "profile.json") != experiment["source_sha256"]["profile"]:
        raise ValueError("Frozen profile changed")
    if ROOT.resolve() != (directory / "source").resolve():
        raise ValueError("Worker must execute the archived source, not the mutable checkout")
    record = next(row for row in experiment["controls"] if row["control_id"] == control_id)
    manifest_path = directory / record["manifest"]
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError("Candidate manifest changed")
    manifest = _read_json(manifest_path)
    if manifest["candidate_bundle"]["file"] != "candidates.pkl":
        raise ValueError("Historical candidate bundle must be candidates.pkl")
    candidate_path = manifest_path.parent / manifest["candidate_bundle"]["file"]
    if file_sha256(candidate_path) != manifest["candidate_bundle"]["sha256"]:
        raise ValueError("Candidate bundle changed")
    if file_sha256(directory / "common_inputs.pkl") != experiment["common_inputs_sha256"]:
        raise ValueError("Common engine inputs changed")
    if file_sha256(directory / "left_candidates.pkl") != experiment["source_sha256"]["left_candidates"]:
        raise ValueError("Shared left candidates changed")
    output = directory / "runs" / control_id
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    common = _load(directory / "common_inputs.pkl")
    right, left = _load(candidate_path), _load(directory / "left_candidates.pkl")
    if sorted(right) != experiment["dates"] or sorted(left) != experiment["dates"]:
        raise ValueError("Candidate calendars differ from the experiment")
    kwargs = common["engine_kwargs"]
    start, end = _window(experiment["engine_parameters"]["requested_start"], experiment["engine_parameters"]["end_date"])
    if any(_trim_calendar(dict(kwargs), start, end).values()):
        raise ValueError("Common formula/actions/security dates extend outside the replay window")
    kwargs["candidate_snapshots"] = {date: merge_hybrid_rows(rows, left[date], right_factor=rows[0]["playbook_factor"])
                                   for date, rows in right.items()}
    parameters = {key: value for key, value in kwargs.items() if key not in DATA_KEYS}
    if parameters != experiment["engine_parameters"]:
        raise ValueError("Effective parameters differ from the frozen experiment")
    fingerprint = hashlib.sha256((experiment["common_inputs_sha256"] + record["manifest_sha256"]
                                  + experiment["source_sha256"]["left_candidates"]).encode()).hexdigest()
    run = {"status": "running", "accounting_status": "pending", "acceptance_status": "pending",
           "research_only": True, "strict_point_in_time": False, "engine_parameters": parameters,
           "engine_sha256": archive["sha256"]["stock_research/backtesting/engine.py"],
           "code_sha256": archive["sha256"], "input_bundle_sha256": fingerprint,
           "input_bundle_hash_method": "sha256(common-pickle-sha256 + candidate-manifest-sha256 + left-pickle-sha256)",
           "started_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run_manifest.json", run)
    profiler = None
    if profile_execution:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    try:
        with _offline():
            result = run_portfolio_backtest(**kwargs)
            audit = audit_portfolio_accounting(result, kwargs["price_frames"], kwargs["corporate_actions"], parameters)
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(output / "execution.prof"))
            import pstats
            with (output / "execution_profile.txt").open("w", encoding="utf-8") as handle:
                pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(45)
            run["execution_profiled"] = True
        _verify_archive(directory / "source")
        _write_json(output / "accounting_audit.json", audit)
        run.update(status="finished", accounting_status=audit["status"],
                   accounting_error_count=audit["error_count"],
                   accounting_audit_sha256=file_sha256(output / "accounting_audit.json"),
                   acceptance_status="failed" if audit["status"] == "failed" else "data_limited",
                   elapsed_seconds=time.perf_counter() - started, peak_rss_mb=_peak_rss_mb(),
                   data_limitations=experiment["limitations"], accounting_data_limitations=audit["data_limitations"])
        for key, suffix in (("equity_curve", "equity"), ("trade_ledger", "trades"), ("events", "events")):
            pd.DataFrame(result[key]).to_csv(output / f"{suffix}.csv", index=False, encoding="utf-8-sig")
        result["research_diagnostics"] = audit_frames(pd.DataFrame(result["equity_curve"]), pd.DataFrame(result["trade_ledger"]),
                                                     directory / "no_external_database.sqlite3",
                                                     initial_equity=1.0)  # Engine emits normalized NAV.
        binding = {"control_id": control_id, "candidate_manifest_sha256": record["manifest_sha256"],
                   "profile_sha256": experiment["source_sha256"]["profile"],
                   "shared_input_sha256": {**{name: experiment["common_inputs_sha256"] for name in
                       ("prices", "formula_phases", "trade_plans", "corporate_actions", "security_end_dates")},
                       "left_candidates": experiment["source_sha256"]["left_candidates"]}}
        result.update(frozen_run=run, selection_control=binding, research_only=True,
                      strict_point_in_time=False, acceptance_status=run["acceptance_status"])
        for name in ("sharpe", "sortino", "calmar"):
            result[name] = result["research_diagnostics"]["risk"].get(name)
        _write_json(output / "summary.json", {key: value for key, value in result.items() if key not in {"equity_curve", "events"}})
        _write_json(output / "receipt.json", {**binding, "summary_path": f"runs/{control_id}/summary.json",
                     "summary_sha256": file_sha256(output / "summary.json"), "input_bundle_sha256": fingerprint,
                     "accounting_status": audit["status"], "accounting_error_count": audit["error_count"],
                     "accounting_audit_path": f"runs/{control_id}/accounting_audit.json",
                     "accounting_audit_sha256": run["accounting_audit_sha256"]})
        print(json.dumps({"control_id": control_id, "status": run["status"], "accounting_status": audit["status"],
                          "return_pct": result.get("final_return_pct"), "elapsed_seconds": run["elapsed_seconds"],
                          "peak_rss_mb": run["peak_rss_mb"]}), flush=True)
    except Exception as exc:
        run.update(status="failed", acceptance_status="failed", error=f"{type(exc).__name__}: {exc}",
                   elapsed_seconds=time.perf_counter() - started, peak_rss_mb=_peak_rss_mb())
        raise
    finally:
        _write_json(output / "run_manifest.json", run)
    return 0 if run["accounting_status"] in {"passed", "data_limited"} and run["accounting_error_count"] == 0 else 1


def attest_accounting(experiment_path, control):
    """Bind a completed legacy audit without changing or relabelling its result."""
    directory = Path(experiment_path).resolve().parent
    output = directory / "runs" / control
    receipt_path, summary_path, audit_path = (output / name for name in ("receipt.json", "summary.json", "accounting_audit.json"))
    receipt, summary, audit = _read_json(receipt_path), _read_json(summary_path), _read_json(audit_path)
    if file_sha256(summary_path) != receipt["summary_sha256"]:
        raise ValueError("Completed summary changed before accounting attestation")
    run = summary["frozen_run"]
    if (run["status"] != "finished" or audit.get("status") not in {"passed", "data_limited"}
            or run["accounting_status"] != audit["status"] or audit.get("error_count") != 0 or audit.get("errors")):
        raise ValueError("Completed replay does not have an acceptable zero-error independent audit")
    evidence = {"accounting_status": audit["status"], "accounting_error_count": 0,
                "accounting_audit_path": f"runs/{control}/accounting_audit.json",
                "accounting_audit_sha256": file_sha256(audit_path)}
    if all(receipt.get(key) == value for key, value in evidence.items()):
        return receipt
    if any(key in receipt for key in evidence):
        raise ValueError("Existing accounting attestation changed")
    # Old executor already emitted the real audit. Preserve its original receipt
    # and unchanged summary; this is an evidence supplement, never another trial.
    original_receipt = output / "receipt_before_accounting_attestation.json"
    if original_receipt.exists():
        raise ValueError("Legacy accounting receipt has already been supplemented")
    shutil.copy2(receipt_path, original_receipt)
    receipt.update(evidence)
    receipt["accounting_attestation"] = {"original_receipt_sha256": file_sha256(original_receipt),
                                       "unchanged_summary_sha256": file_sha256(summary_path),
                                       "attestation_code_sha256": file_sha256(Path(__file__)),
                                       "reason": "Bind the existing zero-error data-limited audit; no strategy replay or status change."}
    _write_json(receipt_path, receipt)
    return receipt


def run(experiment_path, *, controls=None, profile_execution=False):
    experiment_path = Path(experiment_path).resolve()
    directory = experiment_path.parent
    experiment = _read_json(experiment_path)
    archive = _verify_archive(directory / "source")
    if archive["sha256"] != experiment["code_sha256"] or archive["dependencies"] != experiment["dependencies"]:
        raise ValueError("Archived source/environment differs from the experiment")
    for path, expected in ((directory / "common_inputs.pkl", experiment["common_inputs_sha256"]),
                           (directory / "left_candidates.pkl", experiment["source_sha256"]["left_candidates"]),
                           (directory / "profile.json", experiment["source_sha256"]["profile"])):
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path.name}")
    selected = controls or [row["control_id"] for row in experiment["controls"]]
    known = {row["control_id"] for row in experiment["controls"]}
    if set(selected) - known:
        raise ValueError("Unknown requested control")
    for control in selected:
        run_path = directory / "runs" / control / "run_manifest.json"
        receipt_path = run_path.parent / "receipt.json"
        if run_path.exists() and receipt_path.exists():
            previous, receipt = _read_json(run_path), _read_json(receipt_path)
            if (previous.get("status") == "finished" and previous.get("accounting_status") in {"passed", "data_limited"}
                    and file_sha256(directory / receipt["summary_path"]) == receipt["summary_sha256"]):
                reuse = True
            else:
                reuse = False
        else:
            reuse = False
        if reuse:
            print(f"reuse independently audited completed run: {control}", flush=True)
        else:
            command = [sys.executable, "-m", "stock_research.research.historical_selection_controls", "worker",
                       "--experiment", str(experiment_path), "--control", control]
            if profile_execution:
                command.append("--profile-execution")
            process = subprocess.run(command,
                           cwd=directory / "source", check=False,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if process.returncode and not receipt_path.is_file():
                raise subprocess.CalledProcessError(process.returncode, command)
        attest_accounting(experiment_path, control)
        receipts = [attest_accounting(experiment_path, path.parent.name)
                    for path in sorted((directory / "runs").glob("*/receipt.json"))]
        _write_json(directory / "receipts.json", {"experiment_sha256": file_sha256(experiment_path), "replays": receipts})
        evaluate(experiment_path, directory / "receipts.json", directory / "comparison.json")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare")
    for name in ("panel-directory", "baseline-bundle", "price-directory", "database", "profile", "output-directory"):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--start-date", default="2021-01-01")
    create.add_argument("--end-date", default="2026-07-21")
    create.add_argument("--seed-count", type=int, default=100)
    execute = sub.add_parser("run")
    execute.add_argument("--experiment", required=True)
    execute.add_argument("--controls", nargs="+")
    execute.add_argument("--profile-execution", action="store_true")
    child = sub.add_parser("worker")
    child.add_argument("--experiment", required=True)
    child.add_argument("--control", required=True)
    child.add_argument("--profile-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(panel_directory=args.panel_directory, baseline_bundle=args.baseline_bundle,
                price_directory=args.price_directory, database=args.database, profile=args.profile,
                output_directory=args.output_directory, start_date=args.start_date, end_date=args.end_date,
                seed_count=args.seed_count)
        return 0
    if args.action == "run":
        return run(args.experiment, controls=args.controls, profile_execution=args.profile_execution)
    return worker(args.experiment, args.control, profile_execution=args.profile_execution)


if __name__ == "__main__":
    raise SystemExit(main())
