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


def _untradable_status(row) -> str | None:
    trade_status = _number(row.get("tradestatus"))
    if trade_status is not None and trade_status <= 0:
        return "suspended_or_no_trade"
    volume = _number(row.get("volume"))
    if volume is not None and volume <= 0:
        return "suspended_or_no_trade"
    return None


def daily_limit_pct(code: str, *, trade_date=None, is_st: bool = False) -> float:
    normalized = str(code).strip().lower()
    digits = normalized.split(".")[-1]
    # Beijing keeps its 30% board limit for ST stocks as well. Accept the
    # exchange-qualified code first, including Tushare's suffix notation.
    if normalized.startswith("bj.") or normalized.endswith(".bj"):
        return 0.30
    if digits.startswith(("4", "8", "920")):
        return 0.30
    if is_st:
        date = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(date) or date < pd.Timestamp("2026-07-06"):
            return 0.05
        return 0.10
    if digits.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def locked_limit_direction(row, previous_close, code: str, *, is_st: bool = False) -> str | None:
    """Return up/down only for a one-price bar near the board limit."""
    prices = [_number(row.get(name)) for name in ("open", "high", "low", "close")]
    if any(price is None or price <= 0 for price in prices):
        return None
    if max(prices) - min(prices) > max(prices) * 1e-6:
        return None
    return opening_limit_direction(row, previous_close, code, is_st=is_st)


def opening_limit_direction(
    row, previous_close, code: str, *, is_st: bool = False,
) -> str | None:
    """Return the limit direction using only information visible at the open."""
    day_open = _number(row.get("open"))
    previous = _number(previous_close)
    if day_open is None or day_open <= 0:
        return None
    up_limit = _number(row.get("up_limit"))
    down_limit = _number(row.get("down_limit"))
    tolerance = max(1e-4, day_open * 1e-6)
    if up_limit is not None and day_open >= up_limit - tolerance:
        return "up"
    if down_limit is not None and day_open <= down_limit + tolerance:
        return "down"
    if previous is None or previous <= 0:
        return None
    change = day_open / previous - 1.0
    threshold = daily_limit_pct(
        code, trade_date=row.get("date"), is_st=is_st,
    ) - 0.005
    # An explicit limit governs its side even when the price does not touch it.
    if up_limit is None and change >= threshold:
        return "up"
    if down_limit is None and change <= -threshold:
        return "down"
    return None


def fill_market_on_open(
    row, *, side: str, previous_close, code: str, is_st: bool = False,
) -> dict:
    """Conservatively simulate a market-on-open order without later bar data."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    untradable = _untradable_status(row)
    if untradable:
        return {"filled": False, "status": untradable, "price": None}
    day_open = _number(row.get("open"))
    if day_open is None or day_open <= 0:
        return {"filled": False, "status": "missing_open", "price": None}
    direction = opening_limit_direction(
        row, previous_close, code, is_st=is_st,
    )
    if (side == "buy" and direction == "up") or (
        side == "sell" and direction == "down"
    ):
        return {
            "filled": False,
            "status": f"locked_limit_{direction}",
            "price": None,
        }
    return {"filled": True, "status": "open_fill", "price": day_open}


def fill_limit_order(
    row, *, side: str, limit_price: float, previous_close, code: str,
    is_st: bool = False,
) -> dict:
    """Simulate a resting limit order; an exact touch is not assumed to fill."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    untradable = _untradable_status(row)
    if untradable:
        return {"filled": False, "status": untradable, "price": None}
    price = float(limit_price)
    day_open = _number(row.get("open"))
    day_high = _number(row.get("high"))
    day_low = _number(row.get("low"))
    if None in (day_open, day_high, day_low):
        return {"filled": False, "status": "missing_ohlc", "price": None}
    opening = day_open <= price if side == "buy" else day_open >= price
    locked = (opening_limit_direction if opening else locked_limit_direction)(
        row, previous_close, code, is_st=is_st,
    )
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
    untradable = _untradable_status(row)
    if untradable:
        return {"filled": False, "status": untradable, "price": None}
    stop = float(stop_price)
    day_open = _number(row.get("open"))
    day_low = _number(row.get("low"))
    if day_open is None or day_low is None:
        return {"filled": False, "status": "missing_ohlc", "price": None}
    direction = (opening_limit_direction if day_open <= stop else locked_limit_direction)(
        row, previous_close, code, is_st=is_st,
    )
    if direction == "down":
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
    untradable = _untradable_status(row)
    if untradable:
        return {"filled": False, "status": untradable, "price": None}
    trigger = float(trigger_price)
    day_open = _number(row.get("open"))
    day_high = _number(row.get("high"))
    if day_open is None or day_high is None:
        return {"filled": False, "status": "missing_ohlc", "price": None}
    direction = (opening_limit_direction if day_open >= trigger else locked_limit_direction)(
        row, previous_close, code, is_st=is_st,
    )
    if direction == "up":
        return {"filled": False, "status": "locked_limit_up", "price": None}
    if day_open >= trigger:
        if day_open / trigger - 1 > max_gap_pct:
            return {"filled": False, "status": "gap_above_chase_limit", "price": None}
        return {"filled": True, "status": "gap_breakout", "price": day_open}
    if day_high > trigger:
        return {"filled": True, "status": "intraday_breakout", "price": trigger}
    status = "touch_unconfirmed" if day_high == trigger else "not_reached"
    return {"filled": False, "status": status, "price": None}
