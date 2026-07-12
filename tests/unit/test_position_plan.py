import pandas as pd

from stock_research.strategies.position_plan import backtest_position_plan


def test_left_grid_fills_only_touched_levels_without_time_stop():
    dates = pd.bdate_range("2026-01-01", periods=70)
    closes = [25.0] * 67 + [23.8, 22.5, 21.5]
    lows = [24.8] * 67 + [23.2, 22.2, 21.2]
    frame = pd.DataFrame({
        "date": dates, "open": closes, "high": closes,
        "low": lows, "close": closes, "volume": 1000,
    })

    result = backtest_position_plan(frame, dates[67], value_line=23.3)

    assert [event["price"] for event in result["events"]] == [23.3, 22.3, 21.3]
    assert result["final_mode"] == "left"
    assert result["final_position_pct"] == 9.0
    assert result["final_cost"] == 22.3


def test_backtest_is_point_in_time_for_existing_events():
    dates = pd.bdate_range("2026-01-01", periods=75)
    closes = [25.0] * 67 + [23.8, 22.5, 21.5, 22, 24, 23, 22, 21]
    frame = pd.DataFrame({
        "date": dates, "open": closes, "high": closes,
        "low": [value - 0.5 for value in closes], "close": closes,
        "volume": [1000] * 72 + [3000, 1000, 1000],
    })

    prefix = backtest_position_plan(frame.iloc[:70], dates[67], value_line=23.3)
    full = backtest_position_plan(frame, dates[67], value_line=23.3)

    assert full["events"][:len(prefix["events"])] == prefix["events"]


def test_left_grid_sells_layers_on_rebound_and_keeps_core_lot():
    dates = pd.bdate_range("2026-01-01", periods=73)
    closes = [25.0] * 67 + [23.8, 22.5, 21.5, 22.5, 23.5, 24.5]
    lows = [24.8] * 67 + [23.2, 22.2, 21.2, 22.0, 23.0, 24.0]
    frame = pd.DataFrame({
        "date": dates, "open": closes, "high": closes,
        "low": lows, "close": closes, "volume": 1000,
    })

    result = backtest_position_plan(frame, dates[67], value_line=23.3)
    sells = [event for event in result["events"] if event["action"] == "左侧网格卖出一层"]

    assert [event["reason"] for event in sells] == ["触及预设卖价22.30", "触及预设卖价23.30"]
    assert result["filled_left_levels"] == [23.3]
    assert result["final_position_pct"] == 3.0
    assert result["grid_round_trips"][21.3] == 1
    assert result["grid_round_trips"][22.3] == 1


def test_flat_long_term_averages_do_not_trigger_left_to_right_add():
    dates = pd.bdate_range("2026-01-01", periods=75)
    closes = [25.0] * 67 + [23.0, 25.5, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0]
    lows = [24.8] * 67 + [22.9, 25.0, 23.8, 23.8, 23.8, 23.8, 23.8, 23.8]
    volumes = [1000] * 68 + [3000] + [1000] * 6
    frame = pd.DataFrame({
        "date": dates, "open": closes, "high": closes,
        "low": lows, "close": closes, "volume": volumes,
    })

    result = backtest_position_plan(frame, dates[67], value_line=23.3)
    actions = [event["action"] for event in result["events"]]

    assert "左转右加仓" not in actions
    assert result["final_mode"] == "left"
    assert result["final_position_pct"] == 3.0
    assert result["right_position_pct"] == 0.0


def test_explicit_grid_uses_configured_sizes_and_sell_prices():
    dates = pd.bdate_range("2026-01-01", periods=70)
    closes = [26.0] * 67 + [23.5, 21.8, 24.0]
    lows = [25.5] * 67 + [23.2, 21.5, 23.5]
    frame = pd.DataFrame({
        "date": dates, "open": closes, "high": closes,
        "low": lows, "close": closes, "volume": 1000,
    })
    plan = [
        {"buy_price": 23.3, "sell_price": 25.0, "position_pct": 0.05, "core": True},
        {"buy_price": 22.0, "sell_price": 24.0, "position_pct": 0.10},
    ]

    result = backtest_position_plan(frame, dates[67], value_line=None, left_grid_plan=plan)
    changes = [event["position_change_pct"] for event in result["events"]]

    assert changes == [5.0, 10.0, -10.0]
    assert result["final_position_pct"] == 5.0
    assert result["filled_left_levels"] == [23.3]
    assert result["grid_plan_source"] == "explicit"
    assert result["grid_round_trips"][22.0] == 1


def test_explicit_grid_rejects_sell_price_not_above_buy_price():
    frame = pd.DataFrame({
        "date": ["2026-01-02"], "open": [10], "high": [10],
        "low": [10], "close": [10], "volume": [1000],
    })

    try:
        backtest_position_plan(
            frame, "2026-01-02", value_line=None,
            left_grid_plan=[{"buy_price": 10, "sell_price": 10, "position_pct": 0.05}],
        )
    except ValueError as exc:
        assert "sell_price > buy_price" in str(exc)
    else:
        raise AssertionError("invalid explicit grid must fail")
