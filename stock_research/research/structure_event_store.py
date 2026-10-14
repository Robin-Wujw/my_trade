"""Immutable full-universe R03 captures, separate from the paper portfolio.

Inputs are archived even when no structure fires. Only real completed-session
captures belong here; historical backfills cannot advance the forward gate.
This module records observations, never orders or fabricated outcome labels.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from stock_research.backtesting.inputs import file_sha256
from stock_research.backtesting.prices import _prepare_frame
from stock_research.research.structure_events import (
    REGISTERED_AT, _local_time, observed_structure_events,
)


VERSION = "forward-structure-events-v1"
BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount",
               "tradestatus", "raw_to_qfq_factor", "raw_open", "raw_high", "raw_low", "raw_close")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, default=str, allow_nan=False), encoding="utf-8")


def _implementation():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "backtesting/signals.py", root / "backtesting/prices.py",
             root / "research/structure_events.py", *sorted((root / "indicators").glob("*.py"))]
    return {"source": {path.relative_to(root).as_posix(): file_sha256(path) for path in paths},
            "packages": {name: metadata.version(name) for name in ("numpy", "pandas")}}


def load_structure_session(directory):
    """Verify archived sources and observations before serving any evidence."""
    root = Path(directory).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if document.get("version") != VERSION:
        raise ValueError("Unknown structure capture contract")
    for name, expected in document["files_sha256"].items():
        if Path(name).name != name or file_sha256(root / name) != expected:
            raise ValueError(f"Structure capture archive changed: {name}")
    sources = {name: sha for name, sha in document["files_sha256"].items() if name != "events.json"}
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if fingerprint != document["source_sha256"]:
        raise ValueError("Structure capture source binding changed")
    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    if len(events) != document["event_count"] or any(
            row["source_sha256"] != fingerprint or row["observation_date"] != document["observation_date"]
            or row["captured_at"] != document["captured_at"] for row in events):
        raise ValueError("Structure observations differ from capture receipt")
    return document, events


def capture_structure_session(output_directory, price_frames, eligible_members, observation_date, *,
                              captured_at, source_receipts, factor_ranks=None):
    """Archive every member, including those outside Top50 and zero-event days.

    The caller supplies the full dated PIT membership and its source receipts.
    Each raw frame is clipped before computing adjustment/technical features;
    only one prepared stock frame is held at a time. Existing captures are
    immutable; a changed source or implementation requires a separate stream.
    """
    from stock_research.research.frozen_forward import capture_clock

    actual = _local_time(capture_clock(observation_date))
    captured = _local_time(captured_at)
    day = pd.Timestamp(observation_date).normalize()
    if (captured > actual or captured <= REGISTERED_AT
            or captured.tz_localize(None).normalize() != day
            or (captured.hour, captured.minute) < (16, 30)):
        raise ValueError("Structure capture requires a real same-day completed session")
    if not source_receipts:
        raise ValueError("Structure capture requires upstream source receipts")
    upstream = {str(name): file_sha256(path) for name, path in sorted(source_receipts.items())}
    members = pd.DataFrame(eligible_members).copy()
    if "date" not in members or "code" not in members:
        raise ValueError("Full dated structure membership required")
    members["date"] = pd.to_datetime(members["date"], errors="coerce")
    members = members.loc[members.date.eq(day)].sort_values("code").reset_index(drop=True)
    if members.empty or members.code.duplicated().any():
        raise ValueError("Full dated structure membership must be unique and nonempty")
    ranks = factor_ranks or {}
    for rank in ranks.values():
        if type(rank) is not int or rank < 1:
            raise ValueError("Factor rank must be an observed positive integer")
    raw_frames = {}
    for code in members.code.astype(str):
        if code not in price_frames:
            raise ValueError(f"Full structure member lacks prices: {code}")
        data = price_frames[code].copy()
        if {"date", "open", "high", "low", "close", "volume", "raw_to_qfq_factor"} - set(data):
            raise ValueError(f"Structure source lacks raw prices/adjustment evidence: {code}")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.loc[data.date.le(day), [name for name in BAR_COLUMNS if name in data]].sort_values("date")
        if data.empty or data.date.duplicated().any() or data.date.max() != day:
            raise ValueError(f"Structure source requires unique same-day prices: {code}")
        columns = ["open", "high", "low", "close", "raw_to_qfq_factor"]
        columns += [name for name in ("raw_open", "raw_high", "raw_low", "raw_close") if name in data]
        values = data[columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values).all().all() or not values.gt(0).all().all():
            raise ValueError(f"Structure source price/adjustment coverage incomplete: {code}")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        if not np.isfinite(volume).all() or volume.lt(0).any():
            raise ValueError(f"Structure source volume coverage incomplete: {code}")
        data = data.reset_index(drop=True)
        data.attrs = {}
        raw_frames[code] = data
    root = Path(output_directory).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".structure-capture-", dir=root.parent))
    # gzip mtime=0 makes the compressed source fingerprint reproducible.
    pd.to_pickle(raw_frames, staging / "prices.pkl.gz", compression={"method": "gzip", "mtime": 0})
    _json(staging / "membership.json", members.assign(date=members.date.dt.strftime("%Y-%m-%d")).to_dict("records"))
    _json(staging / "factor_ranks.json", ranks)
    implementation = _implementation()
    _json(staging / "upstream.json", {"source_sha256": upstream, "implementation": implementation})
    sources = {name: file_sha256(staging / name) for name in
               ("prices.pkl.gz", "membership.json", "factor_ranks.json", "upstream.json")}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if root.exists():
        previous, _ = load_structure_session(root)
        if previous["source_sha256"] != source_hash:
            raise ValueError("Immutable structure capture inputs or implementation changed")
        for name in sources:
            (staging / name).unlink()
        staging.rmdir()
        return previous
    events = []
    for member in members.to_dict("records"):
        code = str(member["code"])
        prepared = _prepare_frame(raw_frames[code])
        events.extend(observed_structure_events({code: prepared}, [member], observation_date,
                      captured_at=captured.isoformat(), source_sha256=source_hash, factor_ranks=ranks))
    if (upstream != {str(name): file_sha256(path) for name, path in sorted(source_receipts.items())}
            or _implementation() != implementation):
        raise ValueError("Structure upstream changed during capture")
    _json(staging / "events.json", events)
    document = {"version": VERSION, "observation_date": str(day.date()), "captured_at": captured.isoformat(),
                "source_sha256": source_hash, "files_sha256": {**sources, "events.json": file_sha256(staging / "events.json")},
                "member_count": len(members), "event_count": len(events), "is_trading_day": True,
                "research_only": True, "strict_point_in_time": False,
                "event_scope": "existing_highest_priority_structure_per_member; full_membership_not_Top50",
                "outcome_status": "pending_independent_executable_T20_T60_labels"}
    _json(staging / "manifest.json", document)
    # Publish once. Failed or racing attempts remain available for diagnosis.
    staging.rename(root)
    load_structure_session(root)
    return document
