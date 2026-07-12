"""Auditable stop-loss and staged take-profit rules for an open position."""
from __future__ import annotations

import pandas as pd


def _price(value):
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def position_exit_snapshot(
    frame: pd.DataFrame,
    cost: float,
    entry_date,
    *,
    entry_mode: str = "right",
    exit_tranches: int = 3,
    thesis_valid: bool = True,
    condition_stop: float | None = None,
    space_stop_pct: float | None = None,
    market_weak: bool = False,
    bearish_divergence: bool = False,
    time_limit_days: int | None = None,
) -> dict:
    """Calculate independent stop and take-profit conditions.

    Structural conditions use the latest close. The right-side 10% space stop
    is a hard intraday stop and triggers when the latest low touches it.
    ``frame`` and ``cost`` must use the same adjustment basis. Each take-profit
    trigger represents one independently managed tranche; callers decide the
    actual share count.
    """
    if frame is None or frame.empty or cost <= 0 or entry_mode not in {"left", "right"}:
        return {"position_risk_available": False, "position_risk_reason": "invalid position data"}
    if exit_tranches not in {3, 5}:
        return {"position_risk_available": False, "position_risk_reason": "exit tranches must be 3 or 5"}
    data = frame.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    for column in ("open", "close", "high", "low", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    entry = pd.Timestamp(entry_date).normalize()
    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    data = data[data["date"].dt.normalize() >= entry]
    if data.empty:
        return {"position_risk_available": False, "position_risk_reason": "no bars since entry"}

    current = float(data.iloc[-1]["close"])
    current_low = _price(data.iloc[-1].get("low"))
    if current_low is None:
        current_low = current
    high_series = data["high"].fillna(data["close"])
    peak_index = high_series.idxmax()
    peak = float(high_series.loc[peak_index])
    peak_date = pd.Timestamp(data.loc[peak_index, "date"]).strftime("%Y-%m-%d")
    holding_days = len(data)
    days_since_peak = len(data.loc[peak_index:]) - 1
    current_return = current / cost - 1
    maximum_return = peak / cost - 1

    effective_space_pct = 0.10 if space_stop_pct is None and entry_mode == "right" else space_stop_pct
    space_stop = cost * (1 - effective_space_pct) if effective_space_pct is not None else None
    condition_stop_value = _price(condition_stop)
    initial_candidates = [value for value in (space_stop, condition_stop_value) if value is not None]
    initial_stop = max(initial_candidates) if initial_candidates else None
    condition_stop_triggered = condition_stop_value is not None and current < condition_stop_value
    space_stop_triggered = space_stop is not None and current_low <= space_stop
    hard_stop_triggered = condition_stop_triggered or space_stop_triggered or not thesis_valid

    profit_floor = cost * 1.05 if maximum_return >= 0.10 else None
    half_profit_stop = cost + (peak - cost) * 0.50 if maximum_return > 0 else None
    trailing_10_stop = peak * 0.90 if maximum_return >= 0.20 else None
    profit_floor_triggered = profit_floor is not None and current < profit_floor
    half_profit_triggered = maximum_return >= 0.10 and current < half_profit_stop
    trailing_10_triggered = trailing_10_stop is not None and current < trailing_10_stop

    volume = data["volume"].fillna(0)
    current_volume = float(volume.iloc[-1])
    deduct5 = float(volume.iloc[-6]) if len(volume) >= 6 else None
    deduct10 = float(volume.iloc[-11]) if len(volume) >= 11 else None
    baseline_values = [value for value in (deduct5, deduct10) if value is not None and value > 0]
    trend_volume_baseline = max(baseline_values) if baseline_values else None
    volume_recovered = trend_volume_baseline is not None and current_volume > trend_volume_baseline

    limit = int(time_limit_days or (5 if market_weak or bearish_divergence else 13))
    entry_time_stop = entry_mode == "right" and holding_days >= limit and maximum_return < 0.10
    divergence_time_take_profit = (
        bearish_divergence and days_since_peak >= 5 and not volume_recovered
    )

    take_profit_tranches = min(
        exit_tranches,
        sum((profit_floor_triggered, trailing_10_triggered, half_profit_triggered, divergence_time_take_profit)),
    )
    take_profit_trigger_ids = [
        trigger_id
        for trigger_id, triggered in (
            ("profit_floor", profit_floor_triggered),
            ("trailing_10", trailing_10_triggered),
            ("half_profit", half_profit_triggered),
            ("divergence_time", divergence_time_take_profit),
        )
        if triggered
    ][:exit_tranches]
    if hard_stop_triggered:
        action = "止损清仓"
        priority = 5
    elif entry_time_stop:
        action = "时间止损"
        priority = 4
    elif take_profit_tranches:
        action = f"分仓止盈{take_profit_tranches}份"
        priority = 3
    else:
        action = "继续持有"
        priority = 1

    active_stops = [value for value in (initial_stop,) if value is not None]
    if profit_floor is not None:
        active_stops.append(profit_floor)
    return {
        "position_risk_available": True,
        "entry_mode": entry_mode,
        "entry_parts": 5 if entry_mode == "left" else 1,
        "exit_tranches": exit_tranches,
        "thesis_valid": thesis_valid,
        "entry_date": entry.strftime("%Y-%m-%d"),
        "holding_days": holding_days,
        "cost": round(cost, 3), "close": round(current, 3),
        "peak": round(peak, 3), "peak_date": peak_date,
        "days_since_peak": days_since_peak,
        "current_return_pct": round(current_return * 100, 2),
        "maximum_return_pct": round(maximum_return * 100, 2),
        "condition_stop": condition_stop_value,
        "space_stop": None if space_stop is None else round(space_stop, 3),
        "initial_stop": None if initial_stop is None else round(initial_stop, 3),
        "condition_stop_triggered": condition_stop_triggered,
        "space_stop_triggered": space_stop_triggered,
        "hard_stop_triggered": hard_stop_triggered,
        "profit_floor": None if profit_floor is None else round(profit_floor, 3),
        "profit_floor_triggered": profit_floor_triggered,
        "half_profit_stop": None if half_profit_stop is None else round(half_profit_stop, 3),
        "half_profit_triggered": half_profit_triggered,
        "trailing_10_stop": None if trailing_10_stop is None else round(trailing_10_stop, 3),
        "trailing_10_triggered": trailing_10_triggered,
        "time_limit_days": limit,
        "entry_time_stop": entry_time_stop,
        "trend_volume_baseline": trend_volume_baseline,
        "volume_recovered": volume_recovered,
        "divergence_time_take_profit": divergence_time_take_profit,
        "take_profit_tranches": take_profit_tranches,
        "take_profit_trigger_ids": take_profit_trigger_ids,
        "active_protection_stop": round(max(active_stops), 3) if active_stops else None,
        "position_action": action,
        "position_action_priority": priority,
    }
