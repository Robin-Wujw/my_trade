import pandas as pd

from stock_research.indicators.technical_quant import moving_average_deduction_snapshot, technical_snapshot


def make_bars(count=40):
    close = [10 + index * 0.2 for index in range(count)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=count, freq="B"),
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000 + index * 20 for index in range(count)],
        }
    )


def test_snapshot_quantifies_all_requested_indicators():
    result = technical_snapshot(make_bars())

    assert result["technical_available"] is True
    for key in (
        "kd_k_close", "kd_d_close", "kd_k_high", "kd_d_high",
        "kd_k_low", "kd_d_low", "kd_gap", "rsi999", "macd_hist",
        "ene_upper", "ene_mid", "ene_lower", "wr10", "wr20",
        "bias10", "volume_ma5", "volume_ma10", "base_volume_ratio",
        "technical_opportunity_score", "technical_risk_score",
        "technical_confidence", "technical_action_score",
    ):
        assert result[key] is not None
    assert 0 <= result["technical_action_score"] <= 100
    assert result["kd_k_high"] >= result["kd_k_close"]
    assert result["kd_k_low"] <= result["kd_k_close"]
    assert 0 <= result["volume_baseline_count"] <= 4
    assert result["volume_baseline_ok"] == (result["volume_baseline_count"] == 4)


def test_snapshot_rejects_insufficient_history():
    result = technical_snapshot(make_bars(10))

    assert result["technical_available"] is False
    assert "need 20" in result["technical_reason"]


def test_ma_deduction_separates_long_support_from_short_down_drag():
    close = pd.Series([100.0] * 180 + [90.0] + [90.0] * 40 + [98.0] * 19 + [97.0])
    volume = pd.Series([1000.0] * 221 + [2000.0] * 19 + [900.0])

    result = moving_average_deduction_snapshot(close, volume)

    assert "60" in result["long_ma_support_periods"]
    assert result["short_ma_down_drag_count"] >= 1
    assert result["ma_deduction_details"]["60"]["support"] is True


def test_ma_deduction_marks_rising_long_ma_below_price_as_upward_pull():
    close = pd.Series([100.0] * 62 + [50.0] + [120.0] * 59 + [110.0])
    volume = pd.Series([100.0] * 122 + [200.0])

    result = moving_average_deduction_snapshot(close, volume, periods=(60,), proximity_pct=1.0)

    assert result["long_ma_upward_pull_periods"] == "60"
    assert result["long_ma_support_periods"] == ""
