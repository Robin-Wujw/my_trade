"""Audited offline closure on a previously frozen local historical input bundle."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pandas as pd

from stock_research.backtesting import frozen_replay as frozen
from stock_research.backtesting.baseline import BASELINE_PATH
from stock_research.core.paths import PATHS
from stock_research.reporting.portfolio_research_audit import build_audit, render_markdown
from stock_research.research import selection_audit


CUTOFF = "2026-07-21"


def refreeze(source_directory, output_directory, *, configuration=BASELINE_PATH):
    """Keep input bytes/parameters; create a new explicit code/runtime contract.

    This is revalidation under today's code, not reproduction of the original
    environment. The original bundle and manifest remain untouched.
    """
    source = Path(source_directory).resolve()
    source_manifest = source / "manifest.json"
    manifest_hash = frozen.app.file_sha256(source_manifest)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    bundle = source / "inputs.pkl"
    if manifest.get("version") != "portfolio-frozen-inputs-v2":
        raise ValueError("complete v2 source bundle required")
    if frozen.app.file_sha256(bundle) != manifest.get("input_bundle_sha256"):
        raise ValueError("source bundle hash mismatch")
    parameters = manifest["engine_parameters"]
    if pd.Timestamp(parameters["end_date"]) > pd.Timestamp(CUTOFF):
        raise ValueError("historical closure must stop by 2026-07-21")
    config = json.loads(Path(configuration).read_text(encoding="utf-8"))
    if frozen._engine_parameters(config) != parameters:
        raise ValueError("versioned configuration differs from the original frozen parameters")
    # Pickle is restricted to locally created, hash-verified research bundles.
    with bundle.open("rb") as stream:
        payload = pickle.load(stream)
    kwargs = payload["engine_kwargs"]
    actual = {key: value for key, value in kwargs.items() if key not in frozen.DATA_ARGUMENTS}
    if actual != parameters:
        raise ValueError("payload parameters differ from source manifest")
    end = pd.Timestamp(parameters["end_date"])
    for key in ("candidate_snapshots", "formula_phases"):
        if any(pd.Timestamp(date) > end for date in kwargs[key]):
            raise ValueError(f"{key} contains data after the closure cutoff")
    if any(pd.to_datetime(frame["date"]).gt(end).any() for frame in kwargs["price_frames"].values()):
        raise ValueError("price bars extend beyond the closure cutoff")
    if parameters.get("price_frames_prepared"):
        frozen._validate_prepared(kwargs["price_frames"])
    del payload, kwargs
    gc.collect()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(bundle, output / "inputs.pkl")
    if frozen.app.file_sha256(output / "inputs.pkl") != manifest["input_bundle_sha256"]:
        raise ValueError("copied input bundle differs")
    if frozen.app.file_sha256(source_manifest) != manifest_hash:
        raise ValueError("source manifest changed during refreeze")
    code_hashes = frozen._code_hashes()
    for relative, expected in code_hashes.items():
        target = output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen.ROOT / relative, target)
        if frozen.app.file_sha256(target) != expected:
            raise ValueError(f"code changed during refreeze: {relative}")
    updated = {**manifest, "created_at": datetime.now(timezone.utc).isoformat(),
               "dependencies": frozen._dependency_versions(), "code_sha256_at_freeze": code_hashes,
               "refreeze_provenance": {"source_directory": str(source),
                   "source_manifest_sha256": manifest_hash,
                   "source_input_bundle_sha256": manifest["input_bundle_sha256"],
                   "source_dependencies": manifest["dependencies"],
                   "source_code_sha256": manifest["code_sha256_at_freeze"],
                   "configuration_sha256": frozen.app.file_sha256(configuration),
                   "input_bytes_unchanged": True, "parameters_unchanged": True,
                   "purpose": "revalidate historical inputs under current code/environment"}}
    frozen._json(output / "manifest.json", updated)
    return updated


def audit_frozen_candidates(bundle_directory, output_directory):
    """Label candidates using only the same frozen prices used by execution."""
    bundle = Path(bundle_directory) / "inputs.pkl"
    manifest = json.loads((Path(bundle_directory) / "manifest.json").read_text(encoding="utf-8"))
    if frozen.app.file_sha256(bundle) != manifest["input_bundle_sha256"]:
        raise ValueError("bundle changed before selection audit")
    with bundle.open("rb") as stream:
        kwargs = pickle.load(stream)["engine_kwargs"]
    records = [{**row, "date": date} for date, rows in kwargs["candidate_snapshots"].items()
               for row in rows if selection_audit._as_bool(row.get("signal_eligible", True))]
    candidates = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["date", "code", "candidate_score", "candidate_source"])
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates["candidate_group"] = selection_audit._candidate_group(candidates)
    candidates["audit_score"] = pd.to_numeric(candidates["candidate_score"], errors="coerce")
    candidates = candidates.sort_values(["date", "candidate_group", "audit_score", "code"],
                                         ascending=[True, True, False, True], kind="stable")
    candidates["audit_rank"] = candidates.groupby(["date", "candidate_group"])["audit_score"].rank(method="first", ascending=False)
    prices = {code: selection_audit.price_series(frame[[key for key in ("date", "close", "tradestatus") if key in frame]])
              for code, frame in kwargs["price_frames"].items()}
    observations = selection_audit.attach_forward_returns(candidates, prices)
    metrics = selection_audit.summarize_audit(observations)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    observations.to_csv(output / "observations.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    payload = {"research_only": True, "strict_point_in_time": False, "data_limited": True,
               "candidate_directory": str(Path(bundle_directory).resolve()),
               "price_directory": "same_frozen_bundle", "benchmark_code": None,
               "candidate_snapshot_count": len(kwargs["candidate_snapshots"]), "candidate_row_count": len(candidates),
               "price_code_count": len(prices),
               "missing_price_code_count": len(set(candidates["code"]) - set(prices)),
               "label_end_date": kwargs["end_date"], "input_bundle_sha256": manifest["input_bundle_sha256"],
               "data_limitations": ["historical_research_only", "close_to_close_labels_are_not_executable_returns",
                                    "revision_vintage_and_industry_snapshot_limitations_remain",
                                    "end_of_sample_incomplete_labels_not_filled"]}
    frozen._json(output / "summary.json", payload)
    selection_audit._write_markdown(output / "report.md", payload, metrics)
    return payload


def isolated_replay(bundle_directory, output_directory):
    """Execute the archived code while the working checkout can keep evolving."""
    bundle = Path(bundle_directory).resolve()
    output = Path(output_directory).resolve()
    log_path = output.parent / (output.name + ".log")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "apps.portfolio_frozen_replay", "replay",
                        "--bundle-directory", str(bundle), "--output-directory", str(output)],
                       cwd=bundle / "code", env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))


def audit_replay_risk(replay_directory, database=PATHS.database):
    """Check NAV units against the execution summary before publishing risk."""
    replay = Path(replay_directory)
    run = json.loads((replay / "run_manifest.json").read_text(encoding="utf-8"))
    accounting = json.loads((replay / "accounting_audit.json").read_text(encoding="utf-8"))
    if (run.get("status") != "finished"
            or run.get("accounting_status") not in {"passed", "data_limited"}
            or accounting.get("status") != run["accounting_status"]
            or accounting.get("error_count") != 0 or accounting.get("errors") != []):
        raise ValueError("risk publication requires a finished, independently audited replay")
    p = run["engine_parameters"]
    stem = f"portfolio_{p['requested_start']}_{p['end_date']}"
    report = build_audit(replay / f"{stem}_equity.csv", replay / f"{stem}_trades.csv",
                         database, initial_equity=1.0)  # Engine exports normalized NAV.
    report["accounting_status"] = accounting["status"]
    report["accounting_data_limitations"] = accounting.get("data_limitations", [])
    summary = json.loads((replay / f"{stem}_summary.json").read_text(encoding="utf-8"))
    if not math.isclose(report["risk"]["cumulative_return"] * 100,
                        summary["final_return_pct"], abs_tol=0.00051):
        raise ValueError("risk cumulative return differs from the audited execution summary")
    return report


def run_closure(source_bundle, output_directory, *, configuration=BASELINE_PATH,
                database=PATHS.database, repeat=True):
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "research_only": True, "strict_point_in_time": False,
              "cutoff": CUTOFF, "started_at": datetime.now(timezone.utc).isoformat()}
    frozen._json(output / "closure.json", status)
    try:
        manifest = refreeze(source_bundle, output / "bundle", configuration=configuration)
        first = isolated_replay(output / "bundle", output / "replay")
        accounting = json.loads((output / "replay/accounting_audit.json").read_text(encoding="utf-8"))
        p = manifest["engine_parameters"]
        stem = f"portfolio_{p['requested_start']}_{p['end_date']}"
        report = audit_replay_risk(output / "replay", database)
        frozen._json(output / "risk_audit.json", report)
        (output / "risk_audit.md").write_text(render_markdown(report), encoding="utf-8")
        selection = audit_frozen_candidates(output / "bundle", output / "selection")
        gc.collect()
        repeat_hashes = {}
        if repeat:
            isolated_replay(output / "bundle", output / "repeat")
            for suffix in ("equity.csv", "trades.csv", "events.csv"):
                before = frozen.app.file_sha256(output / "replay" / f"{stem}_{suffix}")
                after = frozen.app.file_sha256(output / "repeat" / f"{stem}_{suffix}")
                if before != after:
                    raise ValueError(f"deterministic repeat differs: {suffix}")
                repeat_hashes[suffix] = before
        status.update(status="completed", accounting_status=first["accounting_status"],
                      accounting_error_count=accounting["error_count"],
                      acceptance_status=first["acceptance_status"],
                      data_limitations=first["data_limitations"],
                      input_bundle_sha256=manifest["input_bundle_sha256"],
                      deterministic_repeat_verified=bool(repeat), repeat_output_sha256=repeat_hashes,
                      selection=selection, risk=report["risk"],
                      note="historical baseline closure; selection controls have separate receipts and completion status")
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        frozen._json(output / "closure.json", status)
    return status
