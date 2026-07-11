import pandas as pd

from stock_research.indicators.waves import (
    calc_wave_progress_pct,
    calc_wave_pct,
    infer_downtrend_recovery,
    level_price,
)


def test_recovery_levels_are_deterministic():
    assert calc_wave_pct(10, 20, 15) == 50.0
    assert level_price(10, 20, 62.5) == 16.25


def test_wave_percentile_is_bounded_and_breakout_progress_is_separate():
    assert calc_wave_pct(10, 20, 25) == 100.0
    assert calc_wave_pct(10, 20, 5) == 0.0
    assert calc_wave_progress_pct(10, 20, 25) == 150.0


def test_breakout_is_measured_against_prior_high_not_wave_range():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "high": [20.0] * 10 + [18.0] * 50,
            "low": [19.0] * 10 + [10.0] * 50,
            "close": [19.5] * 10 + [12.0] * 49 + [25.0],
        }
    )

    result = infer_downtrend_recovery(frame)

    assert result["recovery_pct"] == 100.0
    assert result["breakout_above_high_pct"] == 25.0


def test_downtrend_recovery_uses_high_before_later_low():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "high": [20.0] * 10 + [18.0] * 20 + [16.0] * 30,
            "low": [19.0] * 10 + [12.0] * 20 + [10.0] * 30,
            "close": [19.5] * 10 + [13.0] * 20 + [15.0] * 30,
        }
    )

    result = infer_downtrend_recovery(frame)

    assert result["downtrend_high"] == 20.0
    assert result["downtrend_low"] == 10.0
    assert result["recovery_level_50"] == 15.0
    assert result["recovery_level_625"] == 16.25
    assert result["recovery_zone"] == "50%-62.5%右侧启动"


def test_wave_reports_prior_uptrend_and_pullback_midpoints_separately():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "high": [15.0] * 10 + [37.2] * 10 + [30.0] * 40,
            "low": [14.13] * 10 + [35.0] * 10 + [23.06] * 40,
            "close": [14.5] * 10 + [37.0] * 10 + [25.0] * 40,
        }
    )

    result = infer_downtrend_recovery(frame)

    assert result["uptrend_level_50"] == 25.66
    assert result["recovery_level_50"] == 30.13
    assert result["uptrend_level_50"] != result["recovery_level_50"]
    assert result["trend_stage"] == "pullback_recovery"
    assert result["stage_level_50"] == result["recovery_level_50"]
    assert result["stage_level_50_passed"] is False


def test_default_window_keeps_an_uptrend_start_older_than_240_bars():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=320),
            "high": [15.5] + [30.0] * 268 + [64.98] + [60.0] * 50,
            "low": [14.92] + [20.0] * 268 + [60.0] + [39.9] * 50,
            "close": [15.0] + [25.0] * 268 + [64.0] + [40.0] * 50,
        }
    )

    result = infer_downtrend_recovery(frame)

    assert result["uptrend_low"] == 14.92
    assert result["downtrend_high"] == 64.98
    assert result["uptrend_level_50"] == 39.95


def test_breakout_clears_old_pullback_and_waits_for_a_new_cycle():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "high": [20.0] * 10 + [18.0] * 49 + [22.0],
            "low": [10.0] * 10 + [8.0] * 49 + [21.0],
            "close": [15.0] * 10 + [12.0] * 49 + [21.0],
        }
    )

    result = infer_downtrend_recovery(frame)

    assert result is None
