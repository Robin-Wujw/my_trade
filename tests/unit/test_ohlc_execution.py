from stock_research.strategies.ohlc_execution import (
    daily_limit_pct,
    fill_buy_stop,
    fill_limit_order,
    fill_market_on_open,
    fill_sell_stop,
)


def bar(open_, high, low, close):
    return {"open": open_, "high": high, "low": low, "close": close}


def test_exact_limit_touch_is_not_assumed_to_fill():
    result = fill_limit_order(bar(11, 12, 10, 11), side="buy", limit_price=10, previous_close=11, code="sh.600000")
    assert result == {"filled": False, "status": "touch_unconfirmed", "price": None}


def test_gap_through_buy_limit_gets_open_price_improvement():
    result = fill_limit_order(bar(9, 10, 8, 9.5), side="buy", limit_price=10, previous_close=11, code="sh.600000")
    assert result["filled"] is True
    assert result["price"] == 9


def test_one_price_limit_up_blocks_buy_but_allows_sell():
    locked = bar(11, 11, 11, 11)
    buy = fill_limit_order(locked, side="buy", limit_price=12, previous_close=10, code="sh.600000")
    sell = fill_limit_order(locked, side="sell", limit_price=10.5, previous_close=10, code="sh.600000")
    assert buy["status"] == "locked_limit_up"
    assert sell["filled"] is True


def test_one_price_limit_down_blocks_sell_stop():
    result = fill_sell_stop(bar(9, 9, 9, 9), stop_price=9.5, previous_close=10, code="sh.600000")
    assert result == {"filled": False, "status": "locked_limit_down", "price": None}


def test_st_limit_changes_from_five_to_ten_percent_on_2026_07_06():
    assert daily_limit_pct("sh.600000", trade_date="2026-07-03", is_st=True) == 0.05
    assert daily_limit_pct("sh.600000", trade_date="2026-07-06", is_st=True) == 0.10


def test_old_five_percent_st_one_price_limit_up_blocks_buy():
    row = {"date": "2026-07-03", **bar(10.5, 10.5, 10.5, 10.5)}
    result = fill_limit_order(
        row, side="buy", limit_price=11, previous_close=10,
        code="sh.600000", is_st=True,
    )
    assert result["status"] == "locked_limit_up"


def test_explicit_daily_limit_takes_precedence_over_board_rule():
    row = {
        "date": "2026-07-03",
        "up_limit": 10.5,
        "down_limit": 9.5,
        **bar(10.5, 10.5, 10.5, 10.5),
    }
    result = fill_limit_order(
        row,
        side="buy",
        limit_price=11,
        previous_close=10,
        code="sh.600000",
        is_st=False,
    )
    assert result["status"] == "locked_limit_up"


def test_market_on_open_uses_opening_limit_without_later_bar_information():
    up_then_opened = {
        "date": "2024-01-02",
        "volume": 1_000_000,
        "up_limit": 11.0,
        "down_limit": 9.0,
        **bar(11.0, 11.2, 10.7, 10.9),
    }
    down_then_opened = {
        "date": "2024-01-02",
        "volume": 1_000_000,
        "up_limit": 11.0,
        "down_limit": 9.0,
        **bar(9.0, 9.3, 8.9, 9.2),
    }

    assert fill_market_on_open(
        up_then_opened,
        side="buy",
        previous_close=10.0,
        code="sh.600000",
    )["status"] == "locked_limit_up"
    assert fill_market_on_open(
        down_then_opened,
        side="sell",
        previous_close=10.0,
        code="sh.600000",
    )["status"] == "locked_limit_down"


def test_market_on_open_rejects_suspended_trade_status():
    row = {"tradestatus": 0, "volume": 1_000_000, **bar(10, 10.2, 9.8, 10)}

    assert fill_market_on_open(
        row,
        side="buy",
        previous_close=10.0,
        code="sh.600000",
    )["status"] == "suspended_or_no_trade"


def test_breakout_stop_fills_at_trigger_or_small_gap_but_rejects_large_gap():
    intraday = fill_buy_stop(bar(9.8, 10.2, 9.7, 10.1), trigger_price=10, previous_close=9.8, code="sh.600000")
    small_gap = fill_buy_stop(bar(10.2, 10.4, 10.1, 10.3), trigger_price=10, previous_close=9.8, code="sh.600000")
    large_gap = fill_buy_stop(bar(10.6, 10.8, 10.5, 10.7), trigger_price=10, previous_close=9.8, code="sh.600000")
    assert intraday["price"] == 10
    assert small_gap["price"] == 10.2
    assert large_gap["status"] == "gap_above_chase_limit"


def test_exact_limits_do_not_fall_back_to_an_estimated_st_limit():
    row = {"date": "2024-01-02", "up_limit": 11, "down_limit": 9,
           **bar(10.5, 10.5, 10.5, 10.5)}
    for previous in (10, None):
        assert fill_market_on_open(
            row, side="buy", previous_close=previous, code="sh.600000", is_st=True,
        )["filled"] is True
        assert fill_limit_order(
            row, side="buy", limit_price=10.6, previous_close=previous,
            code="sh.600000", is_st=True,
        )["filled"] is True
    row = {"up_limit": 11, "down_limit": 9, **bar(11, 11, 11, 11)}
    assert fill_market_on_open(
        row, side="buy", previous_close=None, code="sh.600000",
    )["status"] == "locked_limit_up"


def test_opening_stop_and_limit_cannot_use_later_unlocking():
    row = {"up_limit": 11, "down_limit": 9, **bar(11, 11, 10, 10.5)}
    assert fill_buy_stop(
        row, trigger_price=10.8, previous_close=10, code="sh.600000",
    )["status"] == "locked_limit_up"
    assert fill_limit_order(
        row, side="buy", limit_price=11, previous_close=10, code="sh.600000",
    )["status"] == "locked_limit_up"
    row = {"up_limit": 11, "down_limit": 9, **bar(9, 10, 9, 9.5)}
    assert fill_sell_stop(
        row, stop_price=9.5, previous_close=10, code="sh.600000",
    )["status"] == "locked_limit_down"
    assert fill_limit_order(
        row, side="sell", limit_price=9, previous_close=10, code="sh.600000",
    )["status"] == "locked_limit_down"


def test_suspended_placeholder_bar_never_fills_orders():
    suspended = {"volume": 0, **bar(10, 12, 8, 11)}

    assert fill_limit_order(
        suspended,
        side="buy",
        limit_price=10,
        previous_close=10,
        code="sh.600000",
    )["status"] == "suspended_or_no_trade"
    assert fill_sell_stop(
        suspended,
        stop_price=9,
        previous_close=10,
        code="sh.600000",
    )["status"] == "suspended_or_no_trade"
    assert fill_buy_stop(
        suspended,
        trigger_price=11,
        previous_close=10,
        code="sh.600000",
    )["status"] == "suspended_or_no_trade"
