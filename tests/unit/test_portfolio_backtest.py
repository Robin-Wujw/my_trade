import pandas as pd
import pytest

from stock_research.strategies.historical_candidates import report_period_for
from stock_research.strategies.portfolio_backtest import (
    build_formula_phase_history,
    run_portfolio_backtest,
)


def breakout_bars():
    dates = pd.bdate_range("2026-01-01", periods=80)
    closes = [10.0] * 79 + [11.0]
    return pd.DataFrame({
        "date": dates,
        "open": [10.0] * 80,
        "high": [10.0] * 79 + [11.0],
        "low": [10.0] * 80,
        "close": closes,
        "volume": [1000] * 79 + [3000],
    })


def test_conservative_report_period_switches_after_april():
    assert report_period_for("2026-01-05") == "2025-06-30"
    assert report_period_for("2026-04-30") == "2025-06-30"
    assert report_period_for("2026-05-01") == "2026-03-31"


def test_formula_phase_history_uses_three_five_and_exit_streaks():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=4),
        "window_up_streak": [0, 3, 5, 0],
        "window_down_streak": [0, 0, 0, 5],
    })

    result = build_formula_phase_history(frame)

    assert list(result.values()) == ["waiting", "watch", "active", "exited"]


def test_portfolio_does_not_chain_open_symbols_on_the_same_day():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    codes = ["A", "B", "C", "D"]
    snapshots = {date: [{"code": code, "strategy_part": "2.正常基本面选股"} for code in codes]}

    result = run_portfolio_backtest(
        {code: bars for code in codes}, snapshots,
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date, max_positions=3,
    )

    assert len(result["final_positions"]) == 1
    assert result["coverage_complete"] is True
    assert sum(item["total_position_pct"] for item in result["final_positions"]) == 20.0


def test_portfolio_rejects_missing_candidate_history():
    with pytest.raises(ValueError, match="candidate snapshot history is empty"):
        run_portfolio_backtest({}, {}, {}, requested_start="2026-01-01", end_date="2026-01-02")


def test_three_day_formula_decline_blocks_new_entry_without_profit_buffer():
    dates = pd.bdate_range("2026-01-01", periods=81)

    def frame(closes, volumes):
        return pd.DataFrame({
            "date": dates, "open": [10.0] * len(dates),
            "high": closes, "low": [10.0] * len(dates),
            "close": closes, "volume": volumes,
        })

    first = frame([10.0] * 79 + [11.0, 11.0], [1000] * 79 + [3000, 1000])
    second = frame([10.0] * 80 + [11.0], [1000] * 80 + [3000])
    day1, day2 = (dates[-2].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
    snapshots = {
        day1: [{"code": "A", "strategy_part": "2.正常基本面选股"}],
        day2: [
            {"code": "A", "strategy_part": "2.正常基本面选股"},
            {"code": "B", "strategy_part": "2.正常基本面选股"},
        ],
    }
    formula = {
        day1: {"phase": "active", "window_down_streak": 0},
        day2: {"phase": "exited", "window_down_streak": 3},
    }

    result = run_portfolio_backtest(
        {"A": first, "B": second}, snapshots, formula,
        requested_start=day1, end_date=day2, max_positions=2,
    )

    assert [item["code"] for item in result["final_positions"]] == ["A"]


def test_portfolio_executes_intraday_space_stop_at_stop_price():
    bars = breakout_bars()
    entry_date = bars.iloc[-1]["date"]
    next_date = entry_date + pd.offsets.BDay(1)
    bars = pd.concat([bars, pd.DataFrame([{
        "date": next_date, "open": 9.5, "high": 9.7,
        "low": 8.8, "close": 9.4, "volume": 1000,
    }])], ignore_index=True)
    snapshot_date = entry_date.strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {snapshot_date: [{"code": "A", "strategy_part": "normal"}]},
        {snapshot_date: {"phase": "active", "window_up_streak": 5}},
        requested_start=snapshot_date,
        end_date=next_date.strftime("%Y-%m-%d"),
        max_positions=1,
    )

    sell = [event for event in result["events"] if event["position_change_pct"] < 0][0]
    assert sell["price"] == 9.0
    assert sell["realized_account_pct"] == -2.0


def test_unlimited_symbols_still_obeys_sequential_right_side_entry():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    codes = ["A", "B", "C", "D", "E"]
    result = run_portfolio_backtest(
        {code: bars for code in codes},
        {date: [{"code": code, "name": code} for code in codes]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date,
        end_date=date,
        max_positions=None,
    )

    assert len(result["final_positions"]) == 1
    assert sum(item["total_position_pct"] for item in result["final_positions"]) == 20.0


def test_unconfirmed_market_opens_only_first_right_side_symbol_without_profit_buffer():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    codes = ["A", "B", "C"]
    result = run_portfolio_backtest(
        {code: bars for code in codes},
        {date: [{"code": code, "name": code} for code in codes]},
        {date: {"phase": "waiting", "window_up_streak": 0}},
        requested_start=date,
        end_date=date,
        max_positions=None,
    )

    assert len(result["final_positions"]) == 1
    assert sum(item["right_position_pct"] for item in result["final_positions"]) == 20.0


def test_up_market_opens_only_first_right_side_symbol_without_ten_percent_profit():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    codes = ["A", "B", "C"]

    result = run_portfolio_backtest(
        {code: bars for code in codes},
        {date: [{"code": code, "name": code} for code in codes]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date,
        end_date=date,
        max_positions=None,
    )

    assert len(result["final_positions"]) == 1
    assert result["final_positions"][0]["right_position_pct"] == 20.0
    assert result["events"][0]["reason"].find("直接突破首仓") >= 0


def test_new_breakout_lot_waits_for_next_open_because_of_t_plus_one():
    bars = breakout_bars()
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [10.0, 11.0, 9.8, 9.9]
    entry_date = bars.iloc[-1]["date"]
    next_date = entry_date + pd.offsets.BDay(1)
    bars = pd.concat([bars, pd.DataFrame([{
        "date": next_date, "open": 9.8, "high": 9.9,
        "low": 9.7, "close": 9.8, "volume": 1000,
    }])], ignore_index=True)
    date = entry_date.strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A"}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date,
        end_date=next_date.strftime("%Y-%m-%d"),
        max_positions=None,
    )

    assert [event["action"] for event in result["events"]] == ["右侧买入", "突破次日未确认退出"]
    assert result["events"][-1]["date"] == next_date.strftime("%Y-%m-%d")
    assert result["events"][-1]["price"] == 9.8
    assert result["final_positions"] == []


def test_failed_breakout_can_sell_then_rebuy_on_next_day_breakout():
    bars = breakout_bars()
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [10.0, 11.0, 9.8, 9.9]
    entry_date = bars.iloc[-1]["date"]
    next_date = entry_date + pd.offsets.BDay(1)
    bars = pd.concat([bars, pd.DataFrame([{
        "date": next_date, "open": 9.8, "high": 11.5,
        "low": 9.7, "close": 11.2, "volume": 3000,
    }])], ignore_index=True)
    date = entry_date.strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A"}]},
        {
            date: {"phase": "active", "window_up_streak": 5},
            next_date.strftime("%Y-%m-%d"): {"phase": "active", "window_up_streak": 5},
        },
        requested_start=date,
        end_date=next_date.strftime("%Y-%m-%d"),
        max_positions=None,
    )

    actions = [event["action"] for event in result["events"]]
    assert actions == ["右侧买入", "突破次日未确认退出", "右侧买入"]
    assert result["events"][0]["reason"].startswith("R1;")
    assert result["events"][-1]["reason"].startswith("R2;")


def test_left_only_candidate_cannot_open_automatic_right_position():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A", "allow_right": False}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
    )

    assert result["events"] == []
    assert result["final_positions"] == []


def test_after_close_snapshot_becomes_effective_on_next_trading_day():
    bars = breakout_bars()
    signal_date = bars.iloc[-1]["date"]
    next_date = signal_date + pd.offsets.BDay(1)
    bars = pd.concat([bars, pd.DataFrame([{
        "date": next_date, "open": 10.0, "high": 11.5,
        "low": 10.0, "close": 11.2, "volume": 3000,
    }])], ignore_index=True)
    snapshot_date = signal_date.strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {snapshot_date: [{"code": "A", "name": "A", "allow_right": True}]},
        {snapshot_date: {"phase": "active", "window_up_streak": 5}},
        requested_start=snapshot_date,
        end_date=next_date.strftime("%Y-%m-%d"),
        signals_effective_next_day=True,
    )

    assert len(result["events"]) == 1
    assert result["events"][0]["date"] == next_date.strftime("%Y-%m-%d")
