"""Canonical interface between every selection model and the trade engine."""
from __future__ import annotations

import math


MAX_DAILY_CANDIDATES = 10


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_candidate(candidate):
    """Return the one candidate schema consumed by portfolio execution.

    Selection models may retain their own diagnostic columns, but they cannot
    select an execution path.  Every eligible row enters the same price-signal,
    position and exit engine.
    """
    row = dict(candidate)
    row["code"] = str(row.get("code") or "")
    row["name"] = str(row.get("name") or row["code"])
    quality = _number(row.get("quality_score"))
    growth = _number(row.get("earnings_yoy", row.get("yoy")))
    market_cap = _number(row.get("mktcap"))
    original_score = _number(row.get("fundamental_score"))
    if original_score is None:
        original_score = _number(row.get("candidate_score")) or 0.0
    source = str(row.get("candidate_source") or "")
    if quality is not None and growth is not None:
        mainline_bonus = 15.0 if "standard_mainline" in source or bool(row.get("is_mainline")) else 0.0
        valuation_bonus = 0.0
        price_to_value = _number(row.get("price_to_value"))
        if price_to_value is not None and 0.80 <= price_to_value <= 1.08:
            valuation_bonus = max(0.0, 5.0 * (1.08 - price_to_value) / 0.28)
        original_score = quality + min(max(growth, 0.0), 1.0) * 20.0 + mainline_bonus + valuation_bonus
    row["candidate_score"] = round(original_score, 6)
    row["selection_reason"] = str(
        row.get("selection_reason")
        or row.get("strategy_part")
        or row.get("candidate_source")
        or "选股模型入选"
    )
    eligible = row.get("signal_eligible")
    if eligible is None:
        eligible = True
    if isinstance(eligible, str):
        eligible = eligible.strip().lower() in {"1", "true", "yes", "y"}
    if quality is not None:
        eligible = eligible and quality >= 70.0
    if growth is not None:
        eligible = eligible and growth >= 0.10
    if market_cap is not None:
        eligible = eligible and market_cap >= 100.0
    row["signal_eligible"] = bool(eligible)
    return row


def normalize_candidate_snapshots(snapshots):
    result = {}
    for date, rows in snapshots.items():
        normalized = [normalize_candidate(item) for item in rows]
        eligible = [item for item in normalized if item["signal_eligible"]]
        eligible.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        selected = eligible[:MAX_DAILY_CANDIDATES]
        for rank, item in enumerate(selected, 1):
            item["selection_rank"] = rank
        result[date] = selected
    return result
