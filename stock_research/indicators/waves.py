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


def infer_downtrend_recovery(df, lookback=500, cross_window=21, min_drawdown=0.12):
    """Measure the latest active significant pullback and its recovery.

    A pullback starts only after price falls ``min_drawdown`` from the latest
    rising high. A new high completes and clears that pullback, so stale
    anchors from an old cycle cannot produce meaningless multi-hundred-percent
    recovery readings.
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

    high_pos = 0
    high = float(data.loc[0, "high"])
    low_pos = None
    low = None
    for pos in range(1, len(data)):
        day_high = float(data.loc[pos, "high"])
        day_low = float(data.loc[pos, "low"])
        if day_high >= high:
            high_pos = pos
            high = day_high
            low_pos = None
            low = None
            continue
        if day_low / high - 1 <= -abs(min_drawdown):
            if low is None or day_low <= low:
                low_pos = pos
                low = day_low
    if low_pos is None or low is None or high <= low:
        return None

    current = float(data.iloc[-1]["close"])
    # The prior rising wave and the subsequent pullback are two different
    # ranges. Keep both so their 50% levels cannot be confused in reports.
    uptrend_low_pos = int(data.loc[:high_pos, "low"].idxmin())
    uptrend_low = float(data.loc[uptrend_low_pos, "low"])
    close_high_pos = int(data.loc[:low_pos, "close"].idxmax())
    close_high = float(data.loc[close_high_pos, "close"])
    close_uptrend_low_pos = int(data.loc[:close_high_pos, "close"].idxmin())
    close_uptrend_low = float(data.loc[close_uptrend_low_pos, "close"])
    post_high_close_low_pos = int(data.loc[close_high_pos:, "close"].idxmin())
    close_pullback_low = float(data.loc[post_high_close_low_pos, "close"])
    uptrend_level_50 = level_price(uptrend_low, high, 50)
    uptrend_close_level_50 = level_price(close_uptrend_low, close_high, 50)
    pullback_close_level_50 = level_price(close_pullback_low, close_high, 50)
    level_50 = level_price(low, high, 50)
    level_625 = level_price(low, high, 62.5)
    recovery_progress_pct = calc_wave_progress_pct(low, high, current)
    recovery_pct = calc_wave_pct(low, high, current)
    breakout_above_high_pct = max(0.0, round((current / high - 1) * 100, 2))
    if current >= high:
        trend_stage = "uptrend"
        stage_level_50 = uptrend_level_50
        stage_level_50_passed = current >= uptrend_level_50
    else:
        trend_stage = "pullback_recovery"
        stage_level_50 = level_50
        stage_level_50_passed = current >= level_50
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
        "uptrend_low": uptrend_low,
        "uptrend_level_50": uptrend_level_50,
        "close_wave_high": close_high,
        "close_uptrend_low": close_uptrend_low,
        "uptrend_close_level_50": uptrend_close_level_50,
        "close_pullback_low": close_pullback_low,
        "pullback_close_level_50": pullback_close_level_50,
        "downtrend_drawdown": low / high - 1,
        "recovery_level_50": level_50,
        "recovery_level_625": level_625,
        "recovery_pct": recovery_pct,
        "recovery_progress_pct": recovery_progress_pct,
        "breakout_above_high_pct": breakout_above_high_pct,
        "recovery_zone": zone,
        "trend_stage": trend_stage,
        "stage_level_50": stage_level_50,
        "stage_level_50_passed": stage_level_50_passed,
        "above_recovery_50": current >= level_50,
        "above_recovery_625": current >= level_625,
        "crossed_recovery_50_recently": crossed_50,
        "crossed_recovery_625_recently": crossed_625,
        "bars_since_recovery_50_cross": bars_since_50,
        "bars_since_recovery_625_cross": bars_since_625,
        "downtrend_high_date": str(data.loc[high_pos].get("date", high_pos)),
        "downtrend_low_date": str(data.loc[low_pos].get("date", low_pos)),
        "uptrend_low_date": str(data.loc[uptrend_low_pos].get("date", uptrend_low_pos)),
    }
