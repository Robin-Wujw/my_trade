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


def infer_downtrend_recovery(df, lookback=240, cross_window=21, min_drawdown=0.12):
    """Measure recovery from the strongest high-to-later-low downtrend.

    Only rows already present in ``df`` are used.  The anchor pair is the
    maximum drawdown in the lookback window, with the high strictly before the
    low.  Recovery levels are measured upward from that low toward the prior
    high, which is the right-side 50% / 62.5% setup used by the strategy.
    """
    if df is None or df.empty:
        return None
    data = df.copy().tail(lookback)
    required = {"high", "low", "close"}
    if not required.issubset(data.columns):
        return None
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(required)).reset_index(drop=True)
    if len(data) < 40:
        return None

    running_high = data["high"].cummax()
    drawdowns = data["low"] / running_high - 1
    low_pos = int(drawdowns.idxmin())
    if low_pos <= 0:
        return None
    high_pos = int(data.loc[: low_pos - 1, "high"].idxmax())
    high = float(data.loc[high_pos, "high"])
    low = float(data.loc[low_pos, "low"])
    if high <= low or low / high - 1 > -abs(min_drawdown):
        return None

    current = float(data.iloc[-1]["close"])
    level_50 = level_price(low, high, 50)
    level_625 = level_price(low, high, 62.5)
    recovery_pct = calc_wave_pct(low, high, current)
    post_low = data.loc[low_pos:].copy()

    def crossed_recently(level):
        above = post_low["close"] >= level
        crossed = above & ~above.shift(1, fill_value=False)
        positions = crossed[crossed].index.tolist()
        if not positions:
            return False, None
        last_pos = positions[-1]
        bars_ago = len(data) - 1 - last_pos
        return bars_ago < cross_window, int(bars_ago)

    crossed_50, bars_since_50 = crossed_recently(level_50)
    crossed_625, bars_since_625 = crossed_recently(level_625)
    if current >= level_625:
        zone = "62.5%以上确认"
    elif current >= level_50:
        zone = "50%-62.5%右侧启动"
    else:
        zone = "50%以下未确认"

    return {
        "downtrend_high": high,
        "downtrend_low": low,
        "downtrend_drawdown": low / high - 1,
        "recovery_level_50": level_50,
        "recovery_level_625": level_625,
        "recovery_pct": recovery_pct,
        "recovery_zone": zone,
        "above_recovery_50": current >= level_50,
        "above_recovery_625": current >= level_625,
        "crossed_recovery_50_recently": crossed_50,
        "crossed_recovery_625_recently": crossed_625,
        "bars_since_recovery_50_cross": bars_since_50,
        "bars_since_recovery_625_cross": bars_since_625,
        "downtrend_high_date": str(data.loc[high_pos].get("date", high_pos)),
        "downtrend_low_date": str(data.loc[low_pos].get("date", low_pos)),
    }


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
