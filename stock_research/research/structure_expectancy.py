"""Source-verified, dated R03 expectancy for an explicit research entry point."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_research.backtesting.inputs import file_sha256
from stock_research.research.structure_events import EventExpectancy, visible_expectancy
from stock_research.research.structure_labels import CONTRACT
from stock_research.research.structure_label_store import load_structure_labels


class StructureExpectancyBook:
    def __init__(self, receipts):
        if not isinstance(receipts, list):
            raise ValueError("Structure expectancy requires explicit label manifest receipts")
        self.receipts, self.snapshots, self.decisions = [], [], []
        for receipt in receipts:
            if not isinstance(receipt, dict) or set(receipt) != {"manifest_path", "manifest_sha256"}:
                raise ValueError("Structure expectancy requires manifest_path/manifest_sha256")
            path = Path(receipt["manifest_path"]).resolve()
            if path.name != "manifest.json":
                raise ValueError("Structure expectancy receipt must bind the exact manifest.json")
            if file_sha256(path) != receipt["manifest_sha256"]:
                raise ValueError("Structure expectancy manifest hash changed")
            document, labels = load_structure_labels(path.parent)
            events = json.loads((path.parent / "events.json").read_text(encoding="utf-8"))
            captures = json.loads((path.parent / "captures.json").read_text(encoding="utf-8"))
            self.snapshots.append((pd.Timestamp(document["captured_at"]), events, labels, captures))
            self.receipts.append({"manifest_path": str(path), "manifest_sha256": receipt["manifest_sha256"]})
        self.snapshots.sort(key=lambda item: item[0])
        if len({item[0] for item in self.snapshots}) != len(self.snapshots):
            raise ValueError("Duplicate structure expectancy snapshot times")
        self.source_sha256 = hashlib.sha256(json.dumps(self.receipts, sort_keys=True).encode()).hexdigest()
        self._cache = {}

    def estimate(self, observation_date, signal_type):
        day = pd.Timestamp(observation_date)
        if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
            raise ValueError("Structure expectancy requires a dated completed-close decision")
        # A close decision is queued for the next session. Only archives actually
        # captured by this local day's end are usable; later snapshots cannot
        # import an old label backwards into an earlier decision.
        cutoff = pd.Timestamp(f"{day:%Y-%m-%d}T23:59:59+08:00")
        key = (str(day.date()), str(signal_type))
        if key in self._cache:
            return self._cache[key]
        known = [row for row in self.snapshots if row[0] <= cutoff]
        if known:
            _, events, labels, captures = known[-1]
            estimate = visible_expectancy(events, labels, captures, as_of=cutoff.isoformat(),
                signal_type=signal_type, horizon=60, registration_at=CONTRACT["registered_at"])
        else:
            estimate = EventExpectancy(cutoff.isoformat(), signal_type, 60, 0, 0, None, None,
                                       "insufficient_visible_campaigns", hashlib.sha256(b"[]").hexdigest())
        self._cache[key] = estimate
        self.decisions.append({"observation_date": key[0], **asdict(estimate)})
        return estimate

    def verify_sources(self):
        for receipt in self.receipts:
            path = Path(receipt["manifest_path"])
            if file_sha256(path) != receipt["manifest_sha256"]:
                raise ValueError("Structure expectancy manifest changed during replay")
            load_structure_labels(path.parent)

    def report(self):
        return {"structure_expectancy_sources": self.receipts,
                "structure_expectancy_sources_sha256": self.source_sha256,
                "structure_expectancy_decisions": self.decisions,
                "structure_expectancy_status": "forward_gate_required; no_legacy_target_fallback",
                "structure_expectancy_horizon": 60,
                "structure_expectancy_decision_time": "completed_close_queued_for_next_session; source_by_local_day_end"}
