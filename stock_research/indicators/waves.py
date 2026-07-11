# -*- coding: utf-8 -*-
"""Right-side recovery levels for a prior high-to-later-low downtrend."""
import pandas as pd


def calc_wave_pct(low, high, current):
    if high <= low:
        raise ValueError(f"高点({high})必须大于低点({low})")
    raw = (current - low) / (high - low) * 100
    return round(min(100.0, max(0.0, raw)), 2)


def calc_wave_progress_pct(low, high, current):
    """Unbounded recovery progress; use only to describe breaks beyond anchors."""
    if high <= low:
        raise ValueError(f"高点({high})必须大于低点({low})")
    return round((current - low) / (high - low) * 100, 2)


def level_price(low, high, pct):
    return round(low + (high - low) * pct / 100, 2)


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
    recovery_progress_pct = calc_wave_progress_pct(low, high, current)
    recovery_pct = calc_wave_pct(low, high, current)
    breakout_above_high_pct = max(0.0, round((current / high - 1) * 100, 2))
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
        "recovery_progress_pct": recovery_progress_pct,
        "breakout_above_high_pct": breakout_above_high_pct,
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
