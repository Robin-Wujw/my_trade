"""Conservative execution rules for pre-placed orders using daily OHLC bars."""
from __future__ import annotations

import math

import pandas as pd


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def daily_limit_pct(code: str, *, trade_date=None, is_st: bool = False) -> float:
    if is_st:
        date = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(date) or date < pd.Timestamp("2026-07-06"):
            return 0.05
        return 0.10
    digits = str(code).split(".")[-1]
    if digits.startswith(("300", "301", "688")):
        return 0.20
    if digits.startswith(("4", "8")):
        return 0.30
    return 0.10


def locked_limit_direction(row, previous_close, code: str, *, is_st: bool = False) -> str | None:
    """Return up/down only for a one-price bar near the board limit."""
    prices = [_number(row.get(name)) for name in ("open", "high", "low", "close")]
    previous = _number(previous_close)
    if previous is None or previous <= 0 or any(price is None for price in prices):
        return None
    if max(prices) - min(prices) > max(prices) * 1e-6:
        return None
    change = prices[0] / previous - 1
    threshold = daily_limit_pct(
        code, trade_date=row.get("date"), is_st=is_st,
    ) - 0.005
    if change >= threshold:
        return "up"
    if change <= -threshold:
        return "down"
    return None


def fill_limit_order(
    row, *, side: str, limit_price: float, previous_close, code: str,
    is_st: bool = False,
) -> dict:
    """Simulate a resting limit order; an exact touch is not assumed to fill."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    price = float(limit_price)
    day_open = _number(row.get("open"))
    day_high = _number(row.get("high"))
    day_low = _number(row.get("low"))
    if None in (day_open, day_high, day_low):
        return {"filled": False, "status": "missing_ohlc", "price": None}
    locked = locked_limit_direction(row, previous_close, code, is_st=is_st)
    if (side == "buy" and locked == "up") or (side == "sell" and locked == "down"):
        return {"filled": False, "status": f"locked_limit_{locked}", "price": None}
    if side == "buy":
        if day_open <= price:
            return {"filled": True, "status": "gap_or_open_fill", "price": day_open}
        if day_low < price:
            return {"filled": True, "status": "intraday_cross", "price": price}
        status = "touch_unconfirmed" if day_low == price else "not_reached"
    else:
        if day_open >= price:
            return {"filled": True, "status": "gap_or_open_fill", "price": day_open}
        if day_high > price:
            return {"filled": True, "status": "intraday_cross", "price": price}
        status = "touch_unconfirmed" if day_high == price else "not_reached"
    return {"filled": False, "status": status, "price": None}


def fill_sell_stop(
    row, *, stop_price: float, previous_close, code: str, is_st: bool = False,
) -> dict:
    """Simulate a resting sell stop, including an unfilled locked limit-down."""
    stop = float(stop_price)
    day_open = _number(row.get("open"))
    day_low = _number(row.get("low"))
    if day_open is None or day_low is None:
        return {"filled": False, "status": "missing_ohlc", "price": None}
    if locked_limit_direction(row, previous_close, code, is_st=is_st) == "down":
        return {"filled": False, "status": "locked_limit_down", "price": None}
    if day_open <= stop:
        return {"filled": True, "status": "gap_stop", "price": day_open}
    if day_low < stop:
        return {"filled": True, "status": "intraday_stop", "price": stop}
    status = "touch_unconfirmed" if day_low == stop else "not_reached"
    return {"filled": False, "status": status, "price": None}


def fill_buy_stop(
    row, *, trigger_price: float, previous_close, code: str,
    is_st: bool = False, max_gap_pct: float = 0.05,
) -> dict:
    """Simulate a pre-placed breakout buy stop using only the daily bar."""
    trigger = float(trigger_price)
    day_open = _number(row.get("open"))
    day_high = _number(row.get("high"))
    if day_open is None or day_high is None:
        return {"filled": False, "status": "missing_ohlc", "price": None}
    if locked_limit_direction(row, previous_close, code, is_st=is_st) == "up":
        return {"filled": False, "status": "locked_limit_up", "price": None}
    if day_open >= trigger:
        if day_open / trigger - 1 > max_gap_pct:
            return {"filled": False, "status": "gap_above_chase_limit", "price": None}
        return {"filled": True, "status": "gap_breakout", "price": day_open}
    if day_high > trigger:
        return {"filled": True, "status": "intraday_breakout", "price": trigger}
    status = "touch_unconfirmed" if day_high == trigger else "not_reached"
    return {"filled": False, "status": status, "price": None}
