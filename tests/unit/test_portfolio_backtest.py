import pandas as pd
import pytest

from apps.portfolio_backtest import (
    default_data_end_date,
    load_candidate_snapshots,
    validate_backtest_input_coverage,
)
from stock_research.indicators.price_structure import (
    infer_uptrend_anchors,
    structure_price,
    trend_amplitude_valid,
)
from stock_research.strategies.historical_candidates import (
    _trade_basis_snapshot,
    _validate_required_financial_periods,
    report_period_for,
)
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.strategies.portfolio_backtest import (
    _affordable_buy_notional,
    _capped_entry_size,
    _price_structure_signal,
    _prepare_frame,
    board_lot_size,
    build_formula_phase_history,
    run_portfolio_backtest,
)


def test_affordable_notional_reserves_commission_and_slippage():
    notional = _affordable_buy_notional(
        10_000,
        10_000,
        commission_rate=0.000085,
        minimum_commission=5,
        slippage_rate=0.0005,
    )

    assert notional < 10_000
    assert notional + max(notional * 0.000085, 5) + notional * 0.0005 == pytest.approx(10_000)


def test_board_lot_rules_use_two_hundred_for_star_market():
    assert board_lot_size("sh.688072") == 200
    assert board_lot_size("688072") == 200
    assert board_lot_size("sh.600699") == 100
    assert board_lot_size("sz.300308") == 100

    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    regular = run_portfolio_backtest(
        {"sh.600699": bars}, {date: [{"code": "sh.600699"}]}, {},
        requested_start=date, end_date=date, initial_capital=15_000,
    )
    star = run_portfolio_backtest(
        {"sh.688072": bars}, {date: [{"code": "sh.688072"}]}, {},
        requested_start=date, end_date=date, initial_capital=15_000,
    )

    assert regular["trade_ledger"][0]["quantity"] == 300
    assert star["trade_ledger"][0]["quantity"] == 200


def test_unified_candidate_pool_applies_gates_caps_growth_and_keeps_top_ten():
    rows = []
    for index in range(25):
        rows.append({
            "code": f"sh.60{index:04d}",
            "name": f"candidate-{index}",
            "quality_score": 70 + index,
            "earnings_yoy": 0.10 + index,
            "mktcap": 100,
            "candidate_source": "standard_mainline",
        })
    rows.extend([
        {"code": "LOWQ", "quality_score": 69, "earnings_yoy": 1, "mktcap": 100},
        {"code": "LOWG", "quality_score": 99, "earnings_yoy": 0.09, "mktcap": 100},
        {"code": "SMALL", "quality_score": 99, "earnings_yoy": 1, "mktcap": 99},
    ])

    selected = normalize_candidate_snapshots({"2026-07-10": rows})["2026-07-10"]

    assert len(selected) == 10
    assert [item["selection_rank"] for item in selected] == list(range(1, 11))
    assert not {"LOWQ", "LOWG", "SMALL"} & {item["code"] for item in selected}
    # Growth above 100% is capped: adjacent rows differ by quality, not giant yoy.
    assert selected[0]["candidate_score"] - selected[1]["candidate_score"] == pytest.approx(1)


def test_trade_basis_snapshot_scores_visible_ma_volume_and_breakout_setup():
    dates = pd.bdate_range("2026-01-01", periods=80)
    closes = [10 + index * 0.03 for index in range(79)] + [13.0]
    frame = pd.DataFrame({
        "open": closes,
        "high": [value + 0.1 for value in closes],
        "low": [value - 0.1 for value in closes],
        "close": closes,
        "volume": [1000] * 79 + [1800],
    }, index=dates)

    result = _trade_basis_snapshot(frame, dates[-1])

    assert result["trade_basis_score"] >= 7
    assert result["technical_alignment"] == "trade_ready"
    assert result["near_21d_close_high"] is True
    assert "MA20/MA60同步上扬" in result["trade_basis_reason"]
    assert result["ima_web_validation"] == "aligned"


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


def test_point_in_time_report_period_switches_at_disclosure_deadlines():
    assert report_period_for("2024-09-24") == "2024-06-30"
    assert report_period_for("2024-10-30") == "2024-06-30"
    assert report_period_for("2024-10-31") == "2024-09-30"
    assert report_period_for("2026-01-05") == "2025-09-30"
    assert report_period_for("2026-04-29") == "2025-09-30"
    assert report_period_for("2026-04-30") == "2026-03-31"
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
    assert sum(item["position_pct"] for item in result["final_positions"]) == 20.0


def test_portfolio_result_does_not_depend_on_price_frame_insertion_order():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    snapshots = {date: [{"code": "A"}, {"code": "B"}]}
    kwargs = {
        "requested_start": date,
        "end_date": date,
        "initial_capital": 250_000,
        "commission_rate": 0.000085,
        "minimum_commission": 5,
    }

    forward = run_portfolio_backtest(
        {"A": bars, "B": bars}, snapshots, {}, **kwargs,
    )
    reversed_input = run_portfolio_backtest(
        {"B": bars, "A": bars}, snapshots, {}, **kwargs,
    )

    assert forward["trade_ledger"] == reversed_input["trade_ledger"]
    assert forward["final_positions"] == reversed_input["final_positions"]


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
        commission_rate=0.000085,
        minimum_commission=5,
        sell_stamp_duty_rate=0.0005,
    )

    sell = [event for event in result["events"] if event["position_change_pct"] < 0][0]
    assert sell["price"] == 9.0
    assert sell["realized_account_pct"] == pytest.approx(-2.0105, abs=0.0001)
    assert len(result["trade_ledger"]) == 2
    buy, sell = result["trade_ledger"]
    assert buy["trade_side"] == "买入"
    assert buy["trade_amount"] == pytest.approx(200_000)
    assert buy["commission_amount"] == pytest.approx(17)
    assert buy["profit_loss_amount"] == pytest.approx(-17)
    assert sell["trade_side"] == "卖出"
    assert sell["quantity"] == pytest.approx(20_000)
    assert sell["trade_amount"] == pytest.approx(180_000)
    assert sell["cost_amount"] == pytest.approx(200_000)
    assert sell["allocated_entry_fee_amount"] == pytest.approx(17)
    assert sell["transaction_cost_amount"] == pytest.approx(105.3)
    assert sell["gross_pnl_amount"] == pytest.approx(-20_000)
    assert sell["profit_loss_amount"] == pytest.approx(-20_122.3)
    assert sell["reason"]
    assert result["trade_summary"]["closed_trade_net_pnl_amount"] == pytest.approx(-20_122.3)


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
    assert sum(item["position_pct"] for item in result["final_positions"]) == 20.0


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
    assert sum(item["position_pct"] for item in result["final_positions"]) == 20.0


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
    assert result["final_positions"][0]["position_pct"] == 20.0
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


def test_legacy_allow_right_does_not_create_a_separate_execution_route():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A", "allow_right": False}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
    )

    assert len(result["trade_ledger"]) == 1


def test_unified_candidate_interface_honors_signal_eligible():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A", "signal_eligible": False}]},
        {}, requested_start=date, end_date=date,
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


def test_price_breakout_condition_order_does_not_use_final_daily_volume():
    bars = breakout_bars()
    bars.loc[bars.index[-1], "volume"] = 1
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
    )

    assert len(result["events"]) == 1
    assert "价格突破21日收盘高点" in result["events"][0]["reason"]


def test_configured_price_structure_ratio_is_a_pre_known_condition_order():
    bars = breakout_bars()
    bars.loc[bars.index[-2], "close"] = 10.1
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [9.8, 10.2, 9.7, 10.1, 1]
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    plans = {"plans": {"A": {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 8.0,
        "uptrend_high": 12.0, "ratio": 0.50,
        "confluence": ["MA20"],
    }]}}}

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]}, {},
        requested_start=date, end_date=date, trade_plans=plans,
    )

    buy = result["events"][0]
    assert buy["price"] == 9.8
    assert "上涨波段50.0%拉回支撑" in buy["reason"]


def test_symbol_cap_is_independent_from_price_structure_ratios():
    assert _capped_entry_size(0.60, 0.20) == pytest.approx(0.025)
    assert _capped_entry_size(0.30, 0.15) == pytest.approx(0.15)


def test_unproven_latest_batch_blocks_another_addition():
    dates = pd.bdate_range("2026-01-01", periods=82)
    bars = pd.DataFrame({
        "date": dates,
        "open": [9.8] * 79 + [9.8, 12.0, 11.4],
        "high": [9.9] * 79 + [12.3, 12.2, 11.6],
        "low": [9.7] * 79 + [9.7, 10.9, 10.9],
        "close": [9.8] * 79 + [12.2, 11.5, 11.4],
        "volume": [1000] * 82,
    })
    start = dates[-3].strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    snapshots = {start: [{"code": "A"}]}
    plans = {"plans": {"A": {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 8.0,
        "uptrend_high": 12.0, "ratio": 0.75,
        "confluence": ["volume_node"],
    }]}}}

    result = run_portfolio_backtest(
        {"A": bars}, snapshots, {}, requested_start=start, end_date=end,
        trade_plans=plans,
    )

    buys = [event for event in result["events"] if event["action"] == "右侧买入"]
    assert len(buys) == 2


def test_uptrend_ratio_has_no_break_back_above_order():
    bars = breakout_bars()
    bars.loc[bars.index[-2], "close"] = 9.8
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [9.8, 11.2, 9.7, 11.0]
    plan = {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 8.0,
        "uptrend_high": 12.0, "ratio": 0.625,
        "confluence": ["MA20"],
    }]}

    signal = _price_structure_signal(bars, len(bars) - 1, plan, auto_structure=False)

    assert signal is None


def test_pullback_half_breakout_requires_deep_pullback_amplitude_and_time():
    bars = breakout_bars()
    bars.loc[bars.index[-2], "close"] = 7.5
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [7.5, 8.5, 7.4, 8.2]
    valid = {"price_structures": [{
        "kind": "pullback_recovery", "uptrend_low": 2.0,
        "uptrend_high": 10.0, "pullback_low": 6.0,
        "consolidation_days": 13,
    }]}
    shallow = {"price_structures": [{
        "kind": "pullback_recovery", "uptrend_low": 2.0,
        "uptrend_high": 10.0, "pullback_low": 8.0,
        "consolidation_days": 13,
    }]}

    signal = _price_structure_signal(bars, len(bars) - 1, valid, auto_structure=False)
    rejected = _price_structure_signal(bars, len(bars) - 1, shallow, auto_structure=False)

    assert signal["trigger"] == 8.0
    assert signal["reason"] == "回调波段50%向上突破"
    assert rejected is None


def test_monotonic_window_without_confirmed_start_pivot_is_not_anchored():
    dates = pd.bdate_range("2026-01-01", periods=80)
    rise = list(pd.Series(range(65)).map(lambda value: 2.0 + value * 8.0 / 64))
    pullback = [9.4, 8.8, 8.2, 7.6, 7.0, 6.5, 6.0, 6.3, 6.5, 6.7, 6.9, 7.0, 7.05, 7.1]
    closes = rise + pullback + [7.0]
    bars = _prepare_frame(pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value + 0.05 for value in closes[:-1]] + [7.15],
        "low": [value - 0.05 for value in closes[:-1]] + [6.95],
        "close": closes,
        "volume": [1000] * len(dates),
    }))

    signal = _price_structure_signal(bars, len(bars) - 1, auto_structure=True)

    assert signal is None


def test_volume_launch_uses_confirmed_preceding_swing_low_not_window_low():
    dates = pd.bdate_range("2026-01-01", periods=55)
    base = [10.0, 9.8, 9.5, 9.2, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0]
    rise = list(pd.Series(range(25)).map(lambda value: 10.3 + value * 9.7 / 24))
    pullback = [19.5 - value * 0.35 for value in range(20)]
    closes = base + rise + pullback
    frame = pd.DataFrame({
        "date": dates, "close": closes, "volume": [1000] * len(dates),
        "high": [value + 0.05 for value in closes],
        "low": [value - 0.05 for value in closes],
    })

    anchors = infer_uptrend_anchors(frame)

    assert any(
        item["uptrend_low_date"] == dates[4].strftime("%Y-%m-%d")
        and item["uptrend_low"] == pytest.approx(8.95)
        and item["uptrend_high"] == pytest.approx(20.05)
        for item in anchors
    )


def test_author_labeled_anchor_examples_reproduce_published_ratios():
    # Junsheng Electronics: qfq L=13.30, H=39.98, U50=26.64.
    assert structure_price(13.30, 39.98, 0.50) == pytest.approx(26.64)
    assert (39.98 + 22.68) / 2 == pytest.approx(31.33)
    assert trend_amplitude_valid(13.30, 39.98)

    # Duofuduo: L=9.70 and H≈43.876 reproduce the published U625=31.06.
    dfd_high = (31.06 - 9.70 * 0.375) / 0.625
    assert structure_price(9.70, dfd_high, 0.625) == pytest.approx(31.06)
    assert trend_amplitude_valid(9.70, dfd_high)

    # Tinci local operating wave: 2025-12-16 L=36.31 to H=64.98.
    assert structure_price(36.31, 64.98, 0.50) == pytest.approx(50.645)
    assert structure_price(36.31, 64.98, 0.75) == pytest.approx(57.8125)
    # This smaller wave can locate support but is too narrow for the larger
    # trend-change/recovery-half amplitude test.
    assert not trend_amplitude_valid(36.31, 64.98)


def test_minimum_five_yuan_commission_is_applied_to_small_order():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]}, {},
        requested_start=date, end_date=date,
        commission_rate=0.000085, minimum_commission=5,
        initial_capital=10_000,
    )

    assert result["transaction_cost_pct"] == pytest.approx(0.05)
    assert result["events"][0]["realized_account_pct"] == pytest.approx(-0.05)
    assert result["events"][0]["execution_quantity"] == pytest.approx(
        10_000 * 0.20 / result["events"][0]["price"]
    )
    assert result["final_cash"] == pytest.approx(7_995.0)
    assert result["final_positions"][0]["batches"][0]["quantity"] == pytest.approx(200.0)


def test_default_data_end_date_waits_until_daily_bar_is_ready():
    assert default_data_end_date("2026-07-13 15:30") == "2026-07-10"
    assert default_data_end_date("2026-07-13 16:01") == "2026-07-13"
    assert default_data_end_date("2026-07-12 18:00") == "2026-07-10"


def test_backtest_input_coverage_must_match_between_candidates_and_formula():
    snapshots = {"2026-07-10": [{"code": "A"}]}
    formula = pd.DataFrame({"date": ["2026-07-09", "2026-07-10"]})

    assert validate_backtest_input_coverage(
        snapshots, formula, "2026-07-10", "2026-07-10",
    ) == "2026-07-10"

    with pytest.raises(RuntimeError, match="no dates in requested backtest range"):
        validate_backtest_input_coverage(
            snapshots,
            pd.DataFrame({"date": ["2026-07-09"]}),
            "2026-07-10",
            "2026-07-10",
        )


def test_backtest_input_coverage_rejects_empty_candidate_days():
    formula = pd.DataFrame({"date": ["2024-09-24", "2024-09-25"]})

    with pytest.raises(RuntimeError, match="empty selection days"):
        validate_backtest_input_coverage(
            {
                "2024-09-24": [{"code": "A"}],
                "2024-09-25": [],
            },
            formula,
            "2024-09-24",
            "2024-09-25",
        )


def test_backtest_input_coverage_requires_every_formula_trade_date():
    formula = pd.DataFrame({"date": ["2024-09-24", "2024-09-25"]})

    with pytest.raises(RuntimeError, match="do not cover every Formula33 trade date"):
        validate_backtest_input_coverage(
            {"2024-09-24": [{"code": "A"}]},
            formula,
            "2024-09-24",
            "2024-09-25",
        )


def test_required_financial_periods_must_exist_for_candidate_history():
    with pytest.raises(RuntimeError, match="2024-06-30"):
        _validate_required_financial_periods({
            "2024-06-30": {},
            "2024-09-30": {"600000": {"quality_score": 80}},
        })


def test_empty_candidate_snapshot_file_loads_as_zero_candidates(tmp_path):
    (tmp_path / "candidates_2024-09-24.csv").write_text("", encoding="utf-8")

    snapshots = load_candidate_snapshots(tmp_path, "2024-09-24", "2024-09-24")

    assert snapshots == {"2024-09-24": []}
