"""Existing buy-point events over the full eligible universe and visible labels.

Labels are a separate table. An event cannot borrow a label available after the
decision time. Historical diagnostics cannot satisfy the forward sample gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from stock_research.backtesting.signals import _right_signal


REGISTERED_AT = pd.Timestamp("2026-10-14T05:16:36Z")
HORIZONS = (20, 60)


def _local_time(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Event availability requires a valid timezone")
    return timestamp.tz_convert("Asia/Shanghai")


def _observed_times(values):
    # utc=True alone would silently interpret naive source strings as UTC.
    return pd.to_datetime([_local_time(value) for value in values], utc=True)


def observed_structure_events(price_frames, eligible_members, observation_date, *, captured_at,
                              source_sha256, factor_ranks=None):
    """Inspect every dated eligible member, including members outside Top50.

    Membership must already enforce listing/ST/trading/history/ADV20 rules.
    This function retains the existing engine's highest-priority structure,
    without adding buy points or treating its planned target as expected profit.
    """
    captured = _local_time(captured_at)
    day = pd.Timestamp(observation_date)
    if pd.isna(day) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Events require a valid dated source fingerprint")
    day = day.normalize()
    if day > captured.tz_localize(None).normalize():
        raise ValueError("Future event observation is not available at capture time")
    members = pd.DataFrame(eligible_members).copy()
    required = {"date", "code", "membership_as_of", "is_listed", "is_traded", "is_st", "history_days", "adv20_yuan"}
    if required - set(members):
        raise ValueError("Full dated membership eligibility fields are required")
    members["date"] = pd.to_datetime(members.date, errors="coerce")
    members = members.loc[members.date.eq(day)].copy()
    if members.empty or members.code.duplicated().any():
        raise ValueError("Dated full membership must be nonempty and unique")
    available = pd.to_datetime(members.membership_as_of, errors="coerce")
    if available.isna().any() or available.gt(day).any():
        raise ValueError("Future or unknown membership is not allowed")
    for column in ("is_listed", "is_traded", "is_st"):
        if not members[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("Membership flags must be explicit booleans")
    history = pd.to_numeric(members.history_days, errors="coerce")
    adv = pd.to_numeric(members.adv20_yuan, errors="coerce")
    if not np.isfinite(history).all() or not np.isfinite(adv).all() or history.mod(1).ne(0).any():
        raise ValueError("Membership history/liquidity coverage incomplete")
    eligible = members.loc[members.is_listed & members.is_traded & ~members.is_st & history.ge(60) & adv.ge(30_000_000)]
    ranks = factor_ranks or {}
    results = []
    for code in sorted(eligible.code.astype(str)):
        if code not in price_frames:
            raise ValueError(f"Eligible member lacks observed prices: {code}")
        frame = price_frames[code].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame.date.le(day)].sort_values("date").reset_index(drop=True)
        frame.attrs = {}
        if frame.empty or frame.date.duplicated().any() or frame.iloc[-1].date != day:
            raise ValueError(f"Eligible member lacks a unique same-day bar: {code}")
        signal = _right_signal(frame, len(frame) - 1)
        if signal is None:
            continue
        entry = float(frame.iloc[-1]["close"] if signal["order_type"] == "close" else signal["trigger"])
        stop = float(signal["stop"])
        target = signal.get("target_price")
        if not math.isfinite(entry) or not math.isfinite(stop) or not 0 < stop < entry:
            continue
        identity = f"{observation_date}|{code}|{signal['signal_type']}|{source_sha256}"
        results.append({"event_id": hashlib.sha256(identity.encode()).hexdigest(), "code": code,
                        "observation_date": str(day.date()), "captured_at": captured.isoformat(),
                        "signal_type": signal["signal_type"], "signal": json.dumps(signal, sort_keys=True, default=str),
                        "planned_entry": entry, "planned_stop": stop,
                        "planned_target": target, "planned_r": (float(target) - entry) / (entry - stop)
                        if target is not None and math.isfinite(float(target)) else None,
                        "factor_rank": ranks.get(code), "source_sha256": source_sha256,
                        "research_only": True, "strict_point_in_time": False,
                        "same_day_capture": captured.normalize().tz_localize(None) == day})
    return results


@dataclass(frozen=True)
class EventExpectancy:
    as_of: str
    signal_type: str
    horizon: int
    campaigns: int
    captured_sessions: int
    mean_net_r: float | None
    win_rate: float | None
    status: str
    dataset_sha256: str


def visible_expectancy(events, labels, captures, *, as_of, signal_type, horizon=60,
                       forward_only=True):
    """Require 252 real captured sessions and 100 nonoverlapping campaigns.

    Exit R must include actual executable costs. Close-return labels are not
    admitted as executable outcomes. A pending endpoint is never filled with 0.
    """
    if horizon not in HORIZONS:
        raise ValueError("Only registered T+20/T+60 horizons are allowed")
    cutoff = _local_time(as_of)
    event_frame, label_frame, capture_frame = map(pd.DataFrame, (events, labels, captures))
    if event_frame.empty or label_frame.empty:
        return EventExpectancy(cutoff.isoformat(), signal_type, horizon, 0, 0, None, None,
                               "insufficient_visible_campaigns", hashlib.sha256(b"[]").hexdigest())
    required = {"event_id", "code", "observation_date", "captured_at", "signal_type", "source_sha256"}
    if required - set(event_frame) or {"event_id", "horizon", "exit_date", "available_at", "net_r", "executable", "cost_included", "label_source_sha256"} - set(label_frame):
        raise ValueError("Incomplete event/outcome evidence")
    if event_frame.event_id.duplicated().any() or label_frame.duplicated(["event_id", "horizon"]).any():
        raise ValueError("Duplicate event/outcome identity")
    event_frame["capture"] = _observed_times(event_frame.captured_at)
    label_frame["known"] = _observed_times(label_frame.available_at)
    event_frame["start"] = pd.to_datetime(event_frame.observation_date, errors="coerce")
    label_frame["exit"] = pd.to_datetime(label_frame.exit_date, errors="coerce")
    selected = event_frame.loc[event_frame.signal_type.eq(signal_type) & event_frame.capture.le(cutoff)]
    selected = selected.merge(label_frame.loc[label_frame.horizon.eq(horizon) & label_frame.known.le(cutoff)],
                              on="event_id", how="inner", validate="one_to_one")
    selected["net_r"] = pd.to_numeric(selected.net_r, errors="coerce")
    valid = (selected.start.notna() & selected.exit.notna() & selected.exit.ge(selected.start)
             & selected.exit.le(cutoff.tz_localize(None).normalize()) & np.isfinite(selected.net_r)
             & selected.executable.eq(True) & selected.cost_included.eq(True)
             & selected.source_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}")
             & selected.label_source_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}")
             & selected.known.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize().ge(selected.exit))
    selected = selected.loc[valid].copy()
    if forward_only:
        day = selected.capture.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
        selected = selected.loc[selected.capture.gt(REGISTERED_AT) & day.eq(selected.start)]
    # Keep the first available campaign; repeated signals in a still-open
    # stock campaign cannot inflate the class's evidence count.
    retained, last_exit = [], {}
    for row in selected.sort_values(["start", "code", "event_id"]).to_dict("records"):
        if row["code"] in last_exit and row["start"] <= last_exit[row["code"]]:
            continue
        retained.append(row)
        last_exit[row["code"]] = row["exit"]
    sessions = 0
    if not capture_frame.empty:
        if {"observation_date", "captured_at", "source_sha256", "is_trading_day"} - set(capture_frame):
            raise ValueError("Captured-session receipts are required")
        times = pd.Series(_observed_times(capture_frame.captured_at), index=capture_frame.index)
        days = pd.to_datetime(capture_frame.observation_date, errors="coerce")
        local_days = times.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
        hashes = capture_frame.source_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}")
        valid = times.le(cutoff) & local_days.eq(days) & hashes & capture_frame.is_trading_day.eq(True)
        if forward_only:
            valid &= times.gt(REGISTERED_AT)
        sessions = int(days.loc[valid].nunique())
    dataset = [{key: str(row[key]) for key in ("event_id", "source_sha256", "exit", "known", "net_r")} for row in retained]
    fingerprint = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    values = np.array([row["net_r"] for row in retained], dtype=float)
    enough = len(values) >= 100 and sessions >= 252
    status = ("forward_sample_ready" if forward_only else "historical_diagnostic_only") if enough else "insufficient_visible_campaigns"
    return EventExpectancy(cutoff.isoformat(), signal_type, horizon, len(values), sessions,
                           float(values.mean()) if len(values) else None,
                           float((values > 0).mean()) if len(values) else None, status, fingerprint)


def expected_target(entry, stop, estimate: EventExpectancy):
    """A target is unavailable until the registered forward gate is satisfied."""
    if not math.isfinite(entry) or not math.isfinite(stop) or not 0 < stop < entry:
        raise ValueError("Invalid structural entry/stop")
    if estimate.status != "forward_sample_ready" or estimate.mean_net_r is None:
        return {"status": "data_limited", "target": None, "expectancy": asdict(estimate)}
    return {"status": "ok" if estimate.mean_net_r > 0 else "nonpositive_expectancy",
            "target": entry + (entry - stop) * estimate.mean_net_r,
            "expectancy": asdict(estimate), "research_only": True}
