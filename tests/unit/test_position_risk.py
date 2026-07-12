import pandas as pd

from stock_research.indicators.position_risk import position_exit_snapshot


def test_space_stop_triggers_on_intraday_low_even_when_close_recovers():
    frame = pd.DataFrame({
        "date": ["2026-01-02", "2026-01-05"],
        "open": [100.0, 96.0],
        "high": [101.0, 98.0],
        "low": [99.0, 89.5],
        "close": [100.0, 95.0],
        "volume": [1000, 1000],
    })

    result = position_exit_snapshot(frame, 100.0, "2026-01-02", entry_mode="right")

    assert result["space_stop"] == 90.0
    assert result["space_stop_triggered"] is True
    assert result["hard_stop_triggered"] is True


def bars(closes, highs=None, volumes=None):
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=len(closes)),
        "close": closes,
        "high": highs or closes,
        "volume": volumes or [1000] * len(closes),
    })


def test_condition_or_space_stop_uses_first_higher_protection():
    result = position_exit_snapshot(
        bars([100, 96, 92]), 100, "2026-01-01", condition_stop=95,
    )

    assert result["initial_stop"] == 95
    assert result["hard_stop_triggered"] is True
    assert result["position_action"] == "止损清仓"


def test_large_profit_retracement_triggers_independent_tranches():
    result = position_exit_snapshot(
        bars([34, 45, 57.5, 41.1]), 34, "2026-01-01",
    )

    assert result["half_profit_stop"] == 45.75
    assert result["trailing_10_stop"] == 51.75
    assert result["half_profit_triggered"] is True
    assert result["trailing_10_triggered"] is True
    assert result["take_profit_trigger_ids"] == ["trailing_10", "half_profit"]
    assert result["position_action"] == "分仓止盈2份"


def test_profit_floor_protects_position_after_ten_percent_gain():
    result = position_exit_snapshot(
        bars([100, 112, 104]), 100, "2026-01-01",
    )

    assert result["profit_floor"] == 105
    assert result["profit_floor_triggered"] is True
    assert result["take_profit_tranches"] == 2
    assert result["take_profit_trigger_ids"] == ["profit_floor", "half_profit"]
    assert result["position_action"] == "分仓止盈2份"


def test_weak_market_uses_five_day_time_stop():
    result = position_exit_snapshot(
        bars([100, 102, 101, 103, 102]), 100, "2026-01-01", market_weak=True,
    )

    assert result["time_limit_days"] == 5
    assert result["entry_time_stop"] is True
    assert result["position_action"] == "时间止损"


def test_bearish_divergence_timer_resets_when_volume_recovers():
    volumes = [1000] * 12 + [2000]
    result = position_exit_snapshot(
        bars([100, 102, 105, 120, 118, 117, 116, 115, 114, 113, 112, 111, 110], volumes=volumes),
        100, "2026-01-01", bearish_divergence=True,
    )

    assert result["days_since_peak"] >= 5
    assert result["volume_recovered"] is True
    assert result["divergence_time_take_profit"] is False


def test_left_entry_uses_five_entry_parts_and_has_no_default_time_or_space_stop():
    result = position_exit_snapshot(
        bars([100] * 20), 100, "2026-01-01", entry_mode="left", exit_tranches=5,
    )

    assert result["entry_parts"] == 5
    assert result["exit_tranches"] == 5
    assert result["space_stop"] is None
    assert result["entry_time_stop"] is False


def test_left_entry_exits_when_fundamental_thesis_is_invalidated():
    result = position_exit_snapshot(
        bars([100, 98]), 100, "2026-01-01", entry_mode="left", thesis_valid=False,
    )

    assert result["hard_stop_triggered"] is True
    assert result["position_action"] == "止损清仓"
