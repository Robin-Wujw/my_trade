import pandas as pd

from stock_research.indicators.waves import (
    calc_wave_pct,
    infer_downtrend_recovery,
    level_price,
)


def test_recovery_levels_are_deterministic():
    assert calc_wave_pct(10, 20, 15) == 50.0
    assert level_price(10, 20, 62.5) == 16.25


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
