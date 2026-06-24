# -*- coding: utf-8 -*-
"""
Wave-position helpers for right-side holding and pullback monitoring.

The functions are intentionally simple and explainable: identify a recent
low-high wave, calculate where the current price sits in that wave, and expose
50% / 62.5% / 75% levels for staged holding or reduction decisions.
"""
import pandas as pd


def calc_wave_pct(low, high, current):
    if high <= low:
        raise ValueError(f"高点({high})必须大于低点({low})")
    return round((current - low) / (high - low) * 100, 2)


def level_price(low, high, pct):
    return round(low + (high - low) * pct / 100, 2)


def wave_levels(low, high, current=None, retracement_low=None):
    result = {
        "low": float(low),
        "high": float(high),
        "rise_pct": high / low - 1 if low else None,
        "level_50": level_price(low, high, 50),
        "level_625": level_price(low, high, 62.5),
        "level_75": level_price(low, high, 75),
    }
    if current is not None:
        result["current"] = float(current)
        result["wave_pct"] = calc_wave_pct(low, high, current)
    if retracement_low is not None:
        result["retracement_low"] = float(retracement_low)
        result["retracement_50"] = round((high + retracement_low) / 2, 2)
    return result


def infer_recent_wave(df, lookback=120):
    """Infer the latest low-to-high wave from OHLC history.

    It finds the lowest low in the lookback window, then the highest high after
    that low. This is not a full wave theory parser; it is a deterministic
    helper for daily monitoring.
    """
    if df is None or df.empty:
        return None
    data = df.copy().tail(lookback)
    required = {"low", "high", "close"}
    if not required.issubset(data.columns):
        return None
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(required))
    if len(data) < 20:
        return None

    low_pos = data["low"].idxmin()
    after_low = data.loc[low_pos:]
    if after_low.empty:
        return None
    high_pos = after_low["high"].idxmax()
    low = float(data.loc[low_pos, "low"])
    high = float(data.loc[high_pos, "high"])
    current = float(data.iloc[-1]["close"])
    if high <= low:
        return None

    after_high = data.loc[high_pos:]
    retracement_low = float(after_high["low"].min()) if not after_high.empty else None
    result = wave_levels(low, high, current, retracement_low)
    result["low_date"] = str(data.loc[low_pos].get("date", low_pos))
    result["high_date"] = str(data.loc[high_pos].get("date", high_pos))
    return result


def wave_signal(levels):
    if not levels:
        return "波段不足"
    pct = levels.get("wave_pct")
    current = levels.get("current")
    if pct is None or current is None:
        return "波段位置未知"
    if pct >= 75:
        return "75%以上，右侧强势但适合分批止盈/盯背离"
    if pct >= 62.5:
        return "62.5%-75%，趋势持仓区"
    if pct >= 50:
        return "50%-62.5%，右侧观察区"
    return "50%以下，右侧强度不足或回撤偏深"
