"""Objective price structures without conflating support and recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORT_RATIOS = (0.50, 0.625, 0.75)


def structure_price(low: float, high: float, ratio: float) -> float:
    if high <= low:
        raise ValueError("structure high must be greater than structure low")
    return float(low) + (float(high) - float(low)) * float(ratio)


def trend_amplitude_valid(low: float, high: float) -> bool:
    """The author's trend-level test: 50% price * 110% < 62.5% price."""
    level_50 = structure_price(low, high, 0.50)
    level_625 = structure_price(low, high, 0.625)
    return level_50 * 1.10 < level_625


def configured_price_structures(plan: dict | None) -> list[dict]:
    """Parse strict, pre-declared structures from a trade plan.

    ``uptrend_support`` uses one rising low/high pair and is pullback-only.
    ``pullback_recovery`` uses three anchors: the rising low/high pair plus the
    later pullback low. Its only ratio trigger is half of the pullback range.
    """
    raw_items = list((plan or {}).get("price_structures") or [])
    parsed = []
    for raw in raw_items:
        item = dict(raw or {})
        kind = str(item.get("kind") or "").strip()
        try:
            uptrend_low = float(item["uptrend_low"])
            uptrend_high = float(item["uptrend_high"])
        except (KeyError, TypeError, ValueError):
            continue
        if uptrend_high <= uptrend_low:
            continue
        levels = {
            ratio: structure_price(uptrend_low, uptrend_high, ratio)
            for ratio in SUPPORT_RATIOS
        }
        common = {
            "kind": kind,
            "uptrend_low": uptrend_low,
            "uptrend_high": uptrend_high,
            "uptrend_levels": levels,
            "amplitude_valid": trend_amplitude_valid(uptrend_low, uptrend_high),
        }
        if kind == "uptrend_support":
            try:
                ratio = float(item["ratio"])
            except (KeyError, TypeError, ValueError):
                continue
            if ratio not in SUPPORT_RATIOS or not list(item.get("confluence") or []):
                continue
            parsed.append({
                **common,
                "ratio": ratio,
                "level": levels[ratio],
                "confluence": list(item["confluence"]),
            })
        elif kind == "pullback_recovery":
            try:
                pullback_low = float(item["pullback_low"])
            except (KeyError, TypeError, ValueError):
                continue
            if not uptrend_low < pullback_low < uptrend_high:
                continue
            high_date = pd.to_datetime(item.get("uptrend_high_date"), errors="coerce")
            low_date = pd.to_datetime(item.get("pullback_low_date"), errors="coerce")
            consolidation_days = item.get("consolidation_days")
            if consolidation_days is None and pd.notna(high_date) and pd.notna(low_date):
                consolidation_days = len(pd.bdate_range(high_date, low_date)) - 1
            parsed.append({
                **common,
                "pullback_low": pullback_low,
                "recovery_half": (uptrend_high + pullback_low) / 2,
                "deep_pullback_confirmed": pullback_low < levels[0.625],
                "consolidation_days": int(consolidation_days or 0),
            })
    return parsed


def _normalize_structure_frame(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.tail(int(lookback)).copy()
    required = {"date", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.dropna(subset=list(required))
        .sort_values("date")
        .reset_index(drop=True)
    )


def confirmed_turning_points(
    frame: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
) -> dict[str, list[int]]:
    """Return right-confirmed swing highs/lows without using the decision day."""
    if frame is None or len(frame) < left + right + 1:
        return {"lows": [], "highs": []}
    window = int(left) + int(right) + 1
    low_values = pd.to_numeric(frame["low"], errors="coerce")
    high_values = pd.to_numeric(frame["high"], errors="coerce")
    rolling_low = low_values.rolling(window, center=True, min_periods=window).min()
    rolling_high = high_values.rolling(window, center=True, min_periods=window).max()
    lows = np.flatnonzero(
        np.isclose(low_values.to_numpy(dtype=float), rolling_low.to_numpy(dtype=float), equal_nan=False)
    ).tolist()
    highs = np.flatnonzero(
        np.isclose(high_values.to_numpy(dtype=float), rolling_high.to_numpy(dtype=float), equal_nan=False)
    ).tolist()
    return {"lows": lows, "highs": highs}


def _volume_launch_positions(data: pd.DataFrame) -> list[int]:
    """Objective proxy for a volume-led third-wave launch described by the author."""
    returns = data["close"].pct_change()
    volume_ready = (
        (data["volume"] >= data["volume"].shift(5))
        & (data["volume"] >= data["volume"].shift(10))
    )
    launch = (returns >= 0.02) & volume_ready
    return [int(position) for position in data.index[launch.fillna(False)]]


def infer_uptrend_anchors(
    frame: pd.DataFrame,
    *,
    lookback: int = 320,
    pivot_left: int = 3,
    pivot_right: int = 3,
    minimum_advance: float = 0.20,
) -> list[dict]:
    """Enumerate multi-level L-H anchors from confirmed lows before volume launches.

    This deliberately returns several levels.  A later support-first matching
    stage chooses the anchor whose ratio overlaps an independently known
    technical support; it must not silently collapse the chart to one global
    low/high pair.
    """
    data = _normalize_structure_frame(frame, lookback)
    if len(data) < 40:
        return []
    pivots = confirmed_turning_points(
        data, left=pivot_left, right=pivot_right,
    )
    launches = _volume_launch_positions(data)
    high_values = data["high"].to_numpy(dtype=float)
    suffix_high_positions = np.empty(len(data), dtype=int)
    best_position = len(data) - 1
    for position in range(len(data) - 1, -1, -1):
        if high_values[position] >= high_values[best_position]:
            best_position = position
        suffix_high_positions[position] = best_position
    anchors = {}
    for launch_pos in launches:
        prior_lows = [position for position in pivots["lows"] if position < launch_pos]
        if not prior_lows:
            continue
        high_pos = int(suffix_high_positions[launch_pos])
        if high_pos <= launch_pos or len(data) - 1 - high_pos < 1:
            continue
        high = float(data.iloc[high_pos]["high"])
        # Keep two levels per launch: the nearest confirmed low represents the
        # small operating wave; the lowest of the recent pivots represents the
        # broader wave-1/trend start (as in the Duofuduo example).  Enumerating
        # every pivot made daily portfolio reconstruction unnecessarily cubic.
        recent_lows = prior_lows[-12:]
        low_positions = {
            recent_lows[-1],
            min(recent_lows, key=lambda position: float(data.iloc[position]["low"])),
        }
        for low_pos in low_positions:
            low = float(data.iloc[low_pos]["low"])
            if low <= 0 or high / low - 1 < float(minimum_advance):
                continue
            key = (low_pos, high_pos)
            anchors[key] = {
                "kind": "uptrend_anchor",
                "uptrend_low": low,
                "uptrend_high": high,
                "uptrend_low_date": pd.Timestamp(data.iloc[low_pos]["date"]).strftime("%Y-%m-%d"),
                "uptrend_high_date": pd.Timestamp(data.iloc[high_pos]["date"]).strftime("%Y-%m-%d"),
                "launch_date": pd.Timestamp(data.iloc[launch_pos]["date"]).strftime("%Y-%m-%d"),
                "uptrend_levels": {
                    ratio: structure_price(low, high, ratio)
                    for ratio in SUPPORT_RATIOS
                },
                "amplitude_valid": trend_amplitude_valid(low, high),
                "bars_since_high": len(data) - 1 - high_pos,
                "_high_pos": high_pos,
                "_data": data,
            }
    result = sorted(
        anchors.values(),
        key=lambda item: (
            item["uptrend_high_date"], item["uptrend_low_date"]
        ),
        reverse=True,
    )
    return result


def infer_price_structures(frame: pd.DataFrame, *, lookback: int = 320) -> list[dict]:
    """Build uptrend anchors and their separately confirmed H-P pullbacks."""
    structures = []
    for anchor in infer_uptrend_anchors(frame, lookback=lookback):
        data = anchor.pop("_data")
        high_pos = int(anchor.pop("_high_pos"))
        structures.append(dict(anchor))
        # P must already have at least three bars to its right.  A new lower P
        # therefore resets the recovery trigger only after it is confirmable.
        confirmed = data.iloc[high_pos + 1 : max(high_pos + 1, len(data) - 3)]
        if confirmed.empty:
            continue
        pullback_pos = int(confirmed["low"].idxmin())
        pullback_low = float(data.iloc[pullback_pos]["low"])
        if not anchor["uptrend_low"] < pullback_low < anchor["uptrend_high"]:
            continue
        structures.append({
            **anchor,
            "kind": "pullback_recovery",
            "pullback_low": pullback_low,
            "pullback_low_date": pd.Timestamp(data.iloc[pullback_pos]["date"]).strftime("%Y-%m-%d"),
            "recovery_half": (anchor["uptrend_high"] + pullback_low) / 2,
            "deep_pullback_confirmed": pullback_low < anchor["uptrend_levels"][0.625],
            "consolidation_days": int(anchor["bars_since_high"]),
        })
    return structures


def infer_pullback_recovery(
    frame: pd.DataFrame,
    *,
    lookback: int = 500,
    min_drawdown: float = 0.12,
) -> dict | None:
    """Compatibility helper returning the newest qualifying recovery structure."""
    candidates = [
        item for item in infer_price_structures(frame, lookback=lookback)
        if item["kind"] == "pullback_recovery"
        and item["pullback_low"] / item["uptrend_high"] - 1 <= -abs(float(min_drawdown))
    ]
    return candidates[0] if candidates else None
