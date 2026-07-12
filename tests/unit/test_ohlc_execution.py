from stock_research.strategies.ohlc_execution import daily_limit_pct, fill_buy_stop, fill_limit_order, fill_sell_stop


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


def test_breakout_stop_fills_at_trigger_or_small_gap_but_rejects_large_gap():
    intraday = fill_buy_stop(bar(9.8, 10.2, 9.7, 10.1), trigger_price=10, previous_close=9.8, code="sh.600000")
    small_gap = fill_buy_stop(bar(10.2, 10.4, 10.1, 10.3), trigger_price=10, previous_close=9.8, code="sh.600000")
    large_gap = fill_buy_stop(bar(10.6, 10.8, 10.5, 10.7), trigger_price=10, previous_close=9.8, code="sh.600000")
    assert intraday["price"] == 10
    assert small_gap["price"] == 10.2
    assert large_gap["status"] == "gap_above_chase_limit"
