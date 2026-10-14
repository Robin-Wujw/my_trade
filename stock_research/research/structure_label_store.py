"""Immutable daily R03 labels with original event and execution source archives."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd

from stock_research.backtesting.inputs import file_sha256
from stock_research.research.structure_event_store import _json, load_structure_session
from stock_research.research.structure_events import HORIZONS, _local_time, visible_expectancy, conditional_factor_gain
from stock_research.research.structure_labels import CONTRACT, CONTRACT_SHA256, executable_structure_label


VERSION = "forward-structure-labels-v1"


def load_structure_labels(directory):
    root = Path(directory).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if document.get("version") != VERSION or document.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Structure labels have an unknown execution contract")
    for name, expected in document["files_sha256"].items():
        if Path(name).name != name or file_sha256(root / name) != expected:
            raise ValueError(f"Structure label archive changed: {name}")
    sources = {key: value for key, value in document["files_sha256"].items()
               if key not in {"labels.json", "diagnostics.json"}}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if source_hash != document["source_sha256"]:
        raise ValueError("Structure label source binding changed")
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    if len(labels) != document["label_count"]:
        raise ValueError("Structure label count differs from manifest")
    if any(row.get("contract_sha256") != CONTRACT_SHA256 or row.get("label_source_sha256") != source_hash
           or pd.Timestamp(row["available_at"]) != pd.Timestamp(document["captured_at"]) for row in labels):
        raise ValueError("Structure labels differ from source/capture receipt")
    if len({(row["event_id"], row["horizon"]) for row in labels}) != len(labels):
        raise ValueError("Duplicate structure label identities")
    previous = json.loads((root / "previous_labels.json").read_text(encoding="utf-8"))
    retained = [row for row in previous if row["status"] in {"completed", "entry_not_filled"}]
    combined = retained + labels
    if len({(row["event_id"], row["horizon"]) for row in combined}) != len(combined):
        raise ValueError("Duplicate current/retained outcome identities")
    return document, combined


def capture_structure_labels(output_directory, event_directories, raw_frames, sessions, corporate_actions, *,
                             observation_date, captured_at, source_receipts, action_coverage_complete,
                             previous_labels=()):
    """Archive current executions, retaining prior completed labels separately.

    Every daily result binds raw source bars, action rows, the real calendar,
    all immutable event receipts, the registered contract and implementation.
    Previous completed outcomes remain original evidence, never redated.
    """
    from stock_research.research.frozen_forward import capture_clock, verify_replay_prefix
    actual, captured = _local_time(capture_clock(observation_date)), _local_time(captured_at)
    if (captured > actual or captured.strftime("%Y-%m-%d") != observation_date
            or (captured.hour, captured.minute) < (16, 30)
            or captured <= _local_time(CONTRACT["registered_at"])):
        raise ValueError("Labels require real same-day completed-session capture after registration")
    if not source_receipts:
        raise ValueError("Execution labels require actual upstream source receipts")
    events, captures = [], []
    event_hashes = {}
    for directory in event_directories:
        document, rows = load_structure_session(directory)
        if _local_time(document["captured_at"]) > captured or document["observation_date"] > observation_date:
            raise ValueError("Future structure observations cannot enter execution labels")
        captures.append({key: document[key] for key in
                        ("observation_date", "captured_at", "source_sha256", "is_trading_day")})
        event_hashes[str(Path(directory).resolve())] = file_sha256(Path(directory) / "manifest.json")
        events.extend(rows)
    if len({row["event_id"] for row in events}) != len(events):
        raise ValueError("Duplicate event capture stream")
    previous = {(row["event_id"], row["horizon"]): row for row in previous_labels}
    if len(previous) != len(previous_labels):
        raise ValueError("Duplicate previous outcome identities")
    if previous_labels:
        prior_path = source_receipts.get("previous_label_manifest")
        if (prior_path is None or Path(prior_path).name != "manifest.json"
                or load_structure_labels(Path(prior_path).parent)[1] != list(previous_labels)):
            raise ValueError("Previous labels must match their immutable source manifest")
    upstream = {str(name): file_sha256(path) for name, path in sorted(source_receipts.items())}
    frames = {}
    for code, frame in raw_frames.items():
        data = frame.copy()
        data["date"] = pd.to_datetime(data.date, errors="coerce")
        if data.date.isna().any():
            raise ValueError("Outcome source contains unknown dates")
        data = data.loc[data.date.le(pd.Timestamp(observation_date))].sort_values("date").reset_index(drop=True)
        data.attrs = {}
        frames[str(code)] = data
    root = Path(output_directory).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".structure-labels-", dir=root.parent))
    pd.to_pickle(frames, staging / "execution_prices.pkl.gz", compression={"method": "gzip", "mtime": 0})
    _json(staging / "events.json", events)
    _json(staging / "captures.json", captures)
    _json(staging / "calendar.json", [str(pd.Timestamp(day).date()) for day in sessions])
    _json(staging / "actions.json", corporate_actions)
    implementation_paths = (
        Path(__file__), Path(__file__).with_name("structure_labels.py"),
        Path(__file__).with_name("structure_events.py"),
        Path(__file__).parents[1] / "backtesting/execution.py",
        Path(__file__).parents[1] / "backtesting/accounting.py",
        Path(__file__).parents[1] / "backtesting/corporate_actions.py",
        Path(__file__).parents[1] / "strategies/ohlc_execution.py",
        Path(__file__).parents[1] / "reporting/portfolio_accounting_audit.py",
    )
    implementation = {path.name: file_sha256(path) for path in implementation_paths}
    _json(staging / "upstream.json", {"files_sha256": upstream, "event_manifests_sha256": event_hashes,
          "action_coverage_complete": action_coverage_complete, "implementation": implementation})
    _json(staging / "contract.json", CONTRACT)
    _json(staging / "previous_labels.json", list(previous_labels))
    sources = {path.name: file_sha256(path) for path in staging.iterdir()}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if root.exists():
        document, _ = load_structure_labels(root)
        if document["source_sha256"] != source_hash:
            raise ValueError("Immutable execution label inputs or implementation changed")
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        return document
    current = []
    retained = []
    for event in events:
        for horizon in HORIZONS:
            prior = previous.get((event["event_id"], horizon))
            if prior and prior["status"] in {"completed", "entry_not_filled"}:
                retained.append(prior)
                continue
            subset = [day for day in sessions if pd.Timestamp(day) >= pd.Timestamp(event["observation_date"])]
            label = executable_structure_label(event, frames.get(event["code"], pd.DataFrame()), subset,
                corporate_actions.get(event["code"], []), horizon=horizon,
                available_at=captured.isoformat(), source_sha256=source_hash,
                corporate_actions_complete=action_coverage_complete.get(event["code"], False))
            if prior and "execution" in prior and "execution" in label:
                verify_replay_prefix(prior["execution"], label["execution"],
                                     prior["execution"]["equity_curve"][-1]["date"])
            current.append(label)
    all_labels = retained + current
    diagnostics = []
    for signal_type in sorted({event["signal_type"] for event in events}):
        for horizon in HORIZONS:
            estimate = visible_expectancy(events, all_labels, captures, as_of=captured.isoformat(),
                                          signal_type=signal_type, horizon=horizon, registration_at=CONTRACT["registered_at"])
            diagnostics.append({"expectancy": asdict(estimate),
                "conditional_factor_gain": conditional_factor_gain(events, all_labels, captures,
                    as_of=captured.isoformat(), signal_type=signal_type, horizon=horizon,
                    registration_at=CONTRACT["registered_at"])})
    if upstream != {str(name): file_sha256(path) for name, path in sorted(source_receipts.items())}:
        raise ValueError("Execution label upstream changed during capture")
    if implementation != {path.name: file_sha256(path) for path in implementation_paths}:
        raise ValueError("Execution label implementation changed during capture")
    for directory, expected in event_hashes.items():
        load_structure_session(directory)
        if file_sha256(Path(directory) / "manifest.json") != expected:
            raise ValueError("Execution event receipt changed during capture")
    _json(staging / "labels.json", current)
    _json(staging / "diagnostics.json", diagnostics)
    document = {"version": VERSION, "contract_sha256": CONTRACT_SHA256,
                "observation_date": observation_date, "captured_at": captured.isoformat(),
                "source_sha256": source_hash, "files_sha256": {**sources,
                    "labels.json": file_sha256(staging / "labels.json"),
                    "diagnostics.json": file_sha256(staging / "diagnostics.json")},
                "event_count": len(events), "label_count": len(current),
                "retained_original_label_count": len(retained),
                "status_counts": dict(Counter(row["status"] for row in all_labels)),
                "research_only": True, "strict_point_in_time": False,
                "historical_backfill_is_forward_evidence": False}
    _json(staging / "manifest.json", document)
    staging.rename(root)
    load_structure_labels(root)
    return document


def capture_daily_structure_labels(output_root, previous_entries, current_event_directory, repository,
                                   prices, sync_state_path, observation_date, captured_at):
    """Use already captured local data; never fetch data or fabricate a session."""
    from stock_research.research.quantsplaybook_compare import _iter_price_frames
    output = Path(output_root).resolve()
    directories = []
    for entry in previous_entries:
        manifest = (output / entry["structure_event_manifest"]).resolve()
        if not manifest.is_relative_to(output) or file_sha256(manifest) != entry["structure_event_manifest_sha256"]:
            raise ValueError("Previous structure event manifest path/hash changed")
        directories.append(manifest.parent)
    directories.append(Path(current_event_directory).resolve())
    events = [row for directory in directories for row in load_structure_session(directory)[1]]
    previous, receipts = [], {"sync_state": Path(sync_state_path)}
    if previous_entries and "structure_label_manifest" in previous_entries[-1]:
        entry = previous_entries[-1]
        path = (output / entry["structure_label_manifest"]).resolve()
        if not path.is_relative_to(output) or file_sha256(path) != entry["structure_label_manifest_sha256"]:
            raise ValueError("Previous structure label manifest path/hash changed")
        document, previous = load_structure_labels(path.parent)
        if document != entry["structure_label_receipt"]:
            raise ValueError("Previous structure label receipt changed")
        receipts["previous_label_manifest"] = path
    finished = {(row["event_id"], row["horizon"]) for row in previous
                if row["status"] in {"completed", "entry_not_filled"}}
    codes = {row["code"] for row in events if any((row["event_id"], horizon) not in finished for horizon in HORIZONS)}
    start = min([observation_date, *(row["observation_date"] for row in events)])
    sessions, actions, frames, coverage = [observation_date], {}, {}, {}
    if events:
        with repository.database.connect(read_only=True) as connection:
            rows = connection.execute(
                "SELECT DISTINCT trade_date FROM raw.tushare_dataset_rows WHERE dataset='trade_cal' "
                "AND trade_date>=? AND trade_date<=? "
                "AND json_extract(payload_json, '$.exchange')='SSE' "
                "AND json_extract(payload_json, '$.is_open') IN (1,'1') ORDER BY trade_date",
                [start, observation_date],
            ).fetchall()
        sessions = [str(row[0]) for row in rows]
        if not sessions or sessions[0] != start or sessions[-1] != observation_date:
            raise ValueError("Structure execution requires the complete captured exchange calendar")
        limits = repository.load_dataset_for_codes("stk_limit", codes, start_date=start, end_date=observation_date)
        if codes and not {"code", "date", "up_limit", "down_limit"}.issubset(limits):
            limits = pd.DataFrame(columns=["code", "date", "up_limit", "down_limit"])
        for code, frame in _iter_price_frames(prices, codes=codes):
            quote = limits.loc[limits.code.eq(code), ["date", "up_limit", "down_limit"]].drop_duplicates()
            if quote.date.duplicated().any():
                raise ValueError("Conflicting precise execution limit rows")
            frames[code] = frame.drop(columns=["up_limit", "down_limit"], errors="ignore").merge(quote, on="date", how="left")
        actions = repository.load_dividend_actions(codes, start_date=start, end_date=observation_date)
        state = json.loads(Path(sync_state_path).read_text(encoding="utf-8"))
        completed = state.get("completed", {})
        # Calendar coverage and a successful empty response are separate from
        # the presence of rows. A provider failure cannot prove no action.
        complete_actions = all(
            completed.get(f"dividend:{day}", {}).get("coverage_basis") == "successful_ex_record_announcement_date_queries"
            and completed.get(f"dividend:{day}", {}).get("dividend_payment_dates_verified") is True
            for day in sessions)
        coverage = {code: complete_actions for code in codes}
    return capture_structure_labels(output / observation_date / "structure_labels", directories,
        frames, sessions, actions, observation_date=observation_date, captured_at=captured_at,
        source_receipts=receipts, action_coverage_complete=coverage, previous_labels=previous)
