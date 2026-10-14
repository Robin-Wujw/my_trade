"""Daily same-day sync, value capture and frozen paper orchestration; local only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.data.forward_sync import run_sync, utc_now
from stock_research.reporting.forward import load_forward_report
from stock_research.research.forward_left import capture_left
from stock_research.research.frozen_forward import (
    capture_clock, implementation_fingerprint, run_capture, sha256, write_json,
)


def formula_history(formula_path, observation_date, output_path, previous_entry=None):
    from stock_research.reporting.daily_report import (
        FORMULA_PHASE_FILE, FORMULA_PHASE_VERSION, RIGHT_SIDE_ACTIVE, RIGHT_SIDE_EXITED, RIGHT_SIDE_WAITING,
        RIGHT_SIDE_WATCH, advance_formula_phase, load_formula_phase_state,
    )
    frame = pd.read_csv(formula_path)
    required = {"date", "window_up_streak", "window_down_streak", "window_unique_count"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("forward Formula33 requires observed 21-day trend streaks")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.max() != pd.Timestamp(observation_date) or dates.duplicated().any():
        raise ValueError("forward Formula33 date mismatch")
    observation = frame.loc[dates.eq(pd.Timestamp(observation_date))].iloc[0].to_dict()
    for column in ("window_up_streak", "window_down_streak", "window_unique_count"):
        value = pd.to_numeric(observation[column], errors="coerce")
        if pd.isna(value) or not float(value).is_integer() or value < 0:
            raise ValueError(f"forward Formula33 requires observed nonnegative integer {column}")
        observation[column] = int(value)
    phases = {RIGHT_SIDE_ACTIVE: "active", RIGHT_SIDE_WATCH: "watch",
              RIGHT_SIDE_WAITING: "waiting", RIGHT_SIDE_EXITED: "exited"}
    previous = load_formula_phase_state(FORMULA_PHASE_FILE)
    if previous_entry:
        previous = {"version": FORMULA_PHASE_VERSION, "observation_date": previous_entry["date"],
                    "phase": {value: key for key, value in phases.items()}[previous_entry["formula33"]["phase"]]}
    state = advance_formula_phase(frame, previous)
    pd.DataFrame([{**observation, "date": observation_date, "phase": phases[state["phase"]],
                   "formula_source_sha256": sha256(formula_path)}]).to_csv(output_path, index=False)
    return str(output_path)


def _run(*, observation_date, formula_path, database=None, sync_state=None, output_root=None):
    database = Path(database or PATHS.database)
    sync_state = Path(sync_state or PATHS.state / "forward_data_sync.json")
    root = Path(output_root or PATHS.runtime_root / "forward" / "frozen_strategy").resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "daily_status.json"
    status = {"observation_date": observation_date, "started_at": utc_now(),
              "paper_only": True, "research_only": True, "status": "running"}
    write_json(status_path, status)
    try:
        capture_clock(observation_date)
        manifest_path = root / "manifest.json"
        previous_entry = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("implementation") != implementation_fingerprint():
                raise ValueError("code/dependencies changed; preregister a separate forward stream")
            entries = [entry for entry in manifest.get("snapshots", []) if entry["date"] == observation_date]
            if entries:
                bundle = load_forward_report(manifest_path, observation_date)
                source_path = root / observation_date / "formula_source.csv"
                if not source_path.is_file() or sha256(source_path) != sha256(formula_path):
                    raise ValueError("same-day Formula33 input changed after immutable capture")
                history_path = source_path.with_name("formula_history.csv")
                if not history_path.is_file() or sha256(history_path) != entries[0].get("formula_sha256"):
                    raise ValueError("archived Formula33 history changed after immutable capture")
                archived_formula = pd.read_csv(history_path)
                if (len(archived_formula) != 1 or "formula_source_sha256" not in archived_formula
                        or archived_formula.iloc[0]["formula_source_sha256"] != sha256(source_path)):
                    raise ValueError("archived Formula33 source hash mismatch")
                status.update(status="completed", reused=True, candidate_count=len(bundle["candidates"]))
                return manifest_path
            previous_entry = max(manifest.get("snapshots", []), key=lambda item: item["date"], default=None)
        state = run_sync(database_path=database, state_path=sync_state,
                         end_date=observation_date, capture_financials=True)
        if state.get("status") != "completed" or state.get("latest_session") != observation_date:
            raise ValueError("same-day forward synchronization incomplete; inspect forward_data_sync.json")
        left = capture_left(database=database, sync_state=sync_state,
                            output_root=root / "value_left", observation_date=observation_date)
        day_root = root / observation_date
        day_root.mkdir(parents=True, exist_ok=True)
        source_path = day_root / "formula_source.csv"
        source_path.write_bytes(Path(formula_path).read_bytes())
        history = formula_history(source_path, observation_date, day_root / "formula_history.csv", previous_entry)
        entry = run_capture(database=database, sync_state=sync_state, left_directory=left,
                            formula_history=history, output_root=root, observation_date=observation_date)
        load_forward_report(manifest_path, observation_date)
        status.update(status="completed", reused=False, candidate_count=entry["candidate_count"])
        return manifest_path
    except Exception as exc:
        status.update(status="blocked", error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        status["finished_at"] = utc_now()
        write_json(status_path, status)


def run(**kwargs):
    root = Path(kwargs.get("output_root") or PATHS.runtime_root / "forward" / "frozen_strategy").resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".daily.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"daily forward already running or interrupted; inspect lock: {lock}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return _run(**kwargs)
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-date", required=True)
    parser.add_argument("--formula-path", required=True)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--sync-state", default=str(PATHS.state / "forward_data_sync.json"))
    parser.add_argument("--output-root", default=str(PATHS.runtime_root / "forward" / "frozen_strategy"))
    args = parser.parse_args(argv)
    try:
        print(run(**vars(args)))
        return 0
    except Exception as exc:
        print(f"daily_forward blocked: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
