import pandas as pd

from stock_research.reporting.trade_reminders import build_trade_reminders


def bars(closes, lows=None, highs=None):
    closes = list(closes)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-09-01", periods=len(closes)),
        "open": closes,
        "high": highs or closes,
        "low": lows or closes,
        "close": closes,
        "volume": [1000] * len(closes),
    })


def test_explicit_plan_reminds_before_unfilled_grid_buy():
    frame = bars([25.0] * 70 + [23.3, 22.5, 21.55], lows=[25.0] * 70 + [23.2, 22.4, 21.5])
    config = {
        "proximity_pct": 0.02,
        "plans": {
            "sh.600699": {
                "name": "均胜电子", "start_date": "2025-09-01", "value_line": 23.3,
                "grid": [
                    {"buy_price": 23.3, "sell_price": 25.3, "position_pct": 0.06, "core": True},
                    {"buy_price": 21.3, "sell_price": 22.3, "position_pct": 0.06},
                ],
            }
        },
    }

    reminders = build_trade_reminders(pd.DataFrame(), "2026-01-01", lambda *_: frame, config)

    assert len(reminders) == 1
    assert reminders[0]["kind"] == "计划买入"
    assert reminders[0]["target"] == 21.3
    assert reminders[0]["position_pct"] == 6.0


def test_selected_value_stock_is_candidate_not_order_without_plan():
    stocks = pd.DataFrame([{
        "code": "sh.600001", "name": "候选", "strategy_part": "1.基本价值线或附近",
        "close": 10.1, "value_line": 10.0,
    }])

    reminders = build_trade_reminders(stocks, "2026-07-10", lambda *_: pd.DataFrame(), {
        "proximity_pct": 0.02, "plans": {},
    })

    assert reminders[0]["kind"] == "左侧候选"
    assert reminders[0]["position_pct"] is None
    assert "待制定显式网格" in reminders[0]["message"]


def test_candidate_outside_proximity_is_not_reminded():
    stocks = pd.DataFrame([{
        "code": "sh.600001", "name": "候选", "strategy_part": "1.基本价值线或附近",
        "close": 10.3, "value_line": 10.0,
    }])

    assert build_trade_reminders(stocks, "2026-07-10", lambda *_: pd.DataFrame(), {
        "proximity_pct": 0.02, "plans": {},
    }) == []


def test_explicit_right_position_reminds_before_protection_stop(monkeypatch):
    monkeypatch.setattr(
        "stock_research.reporting.trade_reminders.backtest_position_plan",
        lambda *_args, **_kwargs: {
            "filled_left_levels": [], "right_position_pct": 15.0,
            "right_risk": {
                "position_risk_available": True, "initial_stop": 9.0,
                "profit_floor": None, "maximum_return_pct": 5.0,
                "trailing_10_stop": None, "holding_days": 3, "time_limit_days": 13,
            },
        },
    )
    frame = bars([9.1] * 20)
    config = {
        "proximity_pct": 0.02,
        "plans": {"A": {"name": "甲", "start_date": "2026-01-01", "grid": []}},
    }

    reminders = build_trade_reminders(pd.DataFrame(), "2026-07-10", lambda *_: frame, config)

    assert reminders[0]["kind"] == "计划卖出"
    assert reminders[0]["target"] == 9.0
    assert "条件/空间止损" in reminders[0]["message"]
