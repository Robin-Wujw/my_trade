import pandas as pd
import pytest

from apps.portfolio_backtest import (
    candidate_manifest_empty_dates,
    default_data_end_date,
    financial_cache_file_count,
    formula33_refresh_window_args,
    invalidate_formula33_manifest_if_kline_cache_incomplete,
    load_candidate_snapshots,
    report_period_visible_date,
    summarize_kline_cache_coverage,
    validate_backtest_input_coverage,
)
from stock_research.indicators.price_structure import (
    configured_price_structures,
    infer_uptrend_anchors,
    structure_price,
    trend_amplitude_valid,
)
from stock_research.strategies.historical_candidates import (
    _leadership_snapshot,
    _trade_basis_snapshot,
    _validate_required_financial_periods,
    report_period_for,
)
from stock_research.indicators.technical_entries import (
    _valid_volume_price_nodes,
    apply_entry_confluence,
    infer_technical_entry,
)
from stock_research.strategies.candidate_interface import (
    normalize_candidate,
    normalize_candidate_snapshots,
)
from stock_research.strategies.portfolio_backtest import (
    PositionState,
    _active_profit_trigger_ids,
    _affordable_buy_notional,
    _capped_entry_size,
    _entry_risk_still_controls_lot,
    _effective_profit_tranches,
    _price_structure_signal,
    _prepare_frame,
    _is_profit_tail,
    _profit_ids_to_execute,
    _qualifies_profit_tail,
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


def test_only_qualified_high_profit_tail_releases_old_entry_stop():
    tail = PositionState(
        right=[{"merged": True}], right_parts=1,
        right_sold={"profit_floor"}, right_tail_capacity_free=True,
    )

    assert not _entry_risk_still_controls_lot({"merged": True}, tail)
    assert _entry_risk_still_controls_lot({"merged": False}, tail)
    assert _entry_risk_still_controls_lot(
        {"merged": True}, PositionState(right_sold={"profit_floor"}),
    )


def test_board_lot_rules_use_two_hundred_for_star_market():
    assert board_lot_size("sh.688072") == 200
    assert board_lot_size("688072") == 200
    assert board_lot_size("sh.600699") == 100
    assert board_lot_size("sz.300308") == 100

    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    regular = run_portfolio_backtest(
        {"sh.600699": bars}, {date: [{"code": "sh.600699"}]}, {},
        requested_start=date, end_date=date, initial_capital=33_000,
    )
    star = run_portfolio_backtest(
        {"sh.688072": bars}, {date: [{"code": "sh.688072"}]}, {},
        requested_start=date, end_date=date, initial_capital=33_000,
    )

    assert regular["trade_ledger"][0]["quantity"] == 900
    assert star["trade_ledger"][0]["quantity"] == 800


def test_profit_parts_adapt_to_actual_board_lots_instead_of_forcing_small_exit():
    assert _effective_profit_tranches("sz.300308", 200, 5) == 2
    assert _effective_profit_tranches("sh.688408", 200, 5) == 1
    assert _effective_profit_tranches("sz.300308", 1000, 5) == 5
    assert _effective_profit_tranches("sz.300308", 1000, 3) == 3


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


def test_leadership_snapshot_rewards_visible_multi_horizon_strength():
    dates = pd.bdate_range("2025-01-01", periods=140)
    closes = [10 + index * 0.05 for index in range(140)]
    frame = pd.DataFrame({
        "open": closes,
        "high": [value + 0.1 for value in closes],
        "low": [value - 0.1 for value in closes],
        "close": closes,
        "volume": [1000] * 140,
    }, index=dates)

    result = _leadership_snapshot(frame, dates[-1])

    assert result["return_20d"] > 0
    assert result["return_60d"] > result["return_20d"]
    assert result["leadership_score"] >= 15
    assert result["long_term_structure_favorable"] is True


def test_unified_candidate_score_includes_bounded_leadership():
    selected = normalize_candidate_snapshots({
        "2026-07-10": [{
            "code": "sz.300001",
            "quality_score": 80,
            "earnings_yoy": 0.50,
            "mktcap": 500,
            "trade_basis_score": 8,
            "leadership_score": 25,
            "candidate_source": "growth_leadership",
        }],
    })["2026-07-10"]

    assert selected[0]["candidate_score"] == pytest.approx(123)


def test_unified_pool_reserves_five_core_candidates_from_leadership_crowding():
    core = [{
        "code": f"CORE{index}",
        "quality_score": 70,
        "earnings_yoy": 0.10,
        "mktcap": 100,
        "candidate_source": "value_model",
    } for index in range(5)]
    leaders = [{
        "code": f"LEADER{index}",
        "quality_score": 100,
        "earnings_yoy": 1.0,
        "mktcap": 1000,
        "trade_basis_score": 12,
        "leadership_score": 30,
        "candidate_source": "growth_leadership",
    } for index in range(10)]

    selected = normalize_candidate_snapshots({"2026-07-10": core + leaders})["2026-07-10"]

    assert len(selected) == 10
    assert {item["code"] for item in selected if item["code"].startswith("CORE")} == {
        f"CORE{index}" for index in range(5)
    }


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
    assert sum(item["position_pct"] for item in result["final_positions"]) == pytest.approx(29.92)


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
        "date": next_date, "open": 10.1, "high": 10.2,
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
    assert sell["price"] == 9.9
    assert sell["realized_account_pct"] == pytest.approx(-3.0078, abs=0.0001)
    assert len(result["trade_ledger"]) == 2
    buy, sell = result["trade_ledger"]
    assert buy["trade_side"] == "买入"
    assert buy["trade_amount"] == pytest.approx(299_200)
    assert buy["commission_amount"] == pytest.approx(25.43)
    assert buy["profit_loss_amount"] == pytest.approx(-25.43)
    assert sell["trade_side"] == "卖出"
    assert sell["quantity"] == pytest.approx(27_200)
    assert sell["trade_amount"] == pytest.approx(269_280)
    assert sell["cost_amount"] == pytest.approx(299_200)
    assert sell["allocated_entry_fee_amount"] == pytest.approx(25.43)
    assert sell["transaction_cost_amount"] == pytest.approx(157.53)
    assert sell["gross_pnl_amount"] == pytest.approx(-29_920)
    assert sell["profit_loss_amount"] == pytest.approx(-30_102.96)
    assert sell["reason"]
    assert result["trade_summary"]["closed_trade_net_pnl_amount"] == pytest.approx(-30_102.96)


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
    assert sum(item["position_pct"] for item in result["final_positions"]) == pytest.approx(29.92)


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
    assert sum(item["position_pct"] for item in result["final_positions"]) == pytest.approx(29.92)


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
    assert result["final_positions"][0]["position_pct"] == pytest.approx(29.92)
    assert "整理平台收盘放量突破" in result["events"][0]["reason"]


def test_new_breakout_lot_waits_for_next_open_because_of_t_plus_one():
    bars = breakout_bars()
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [10.0, 11.0, 9.8, 11.0]
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
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [10.0, 11.0, 9.8, 11.0]
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


def test_explicit_right_permission_blocks_a_direct_right_entry():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {date: [{"code": "A", "name": "A", "allow_right": False}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
    )

    assert result["trade_ledger"] == []


def test_candidate_sources_grant_left_and_right_permissions_independently():
    selected = normalize_candidate_snapshots({"2026-07-10": [
        {"code": "VALUE", "candidate_source": "value_model", "mktcap": 150},
        {
            "code": "BOTH",
            "candidate_source": "value_model+growth_leadership",
            "mktcap": 150,
        },
        {"code": "RIGHT", "candidate_source": "growth_leadership", "mktcap": 150},
    ]})["2026-07-10"]
    by_code = {item["code"]: item for item in selected}

    assert by_code["VALUE"]["allow_left"] is True
    assert by_code["VALUE"]["allow_right"] is False
    assert by_code["BOTH"]["allow_left"] is True
    assert by_code["BOTH"]["allow_right"] is True
    assert by_code["RIGHT"]["allow_left"] is False
    assert by_code["RIGHT"]["allow_right"] is True


def test_candidate_diagnostics_are_opt_in_and_not_tradable():
    snapshots = {"2026-07-10": [
        {"code": "A", "candidate_score": 10},
        {
            "code": "B", "candidate_source": "value_model",
            "signal_eligible": False, "selected_for_trading": False,
            "value_falsification_reason": "earnings_yoy_below_10pct",
            "candidate_failure_reason": "value_financial_falsification",
        },
    ]}

    default = normalize_candidate_snapshots(snapshots)["2026-07-10"]
    diagnostic = normalize_candidate_snapshots(
        snapshots, include_diagnostics=True,
    )["2026-07-10"]

    assert [item["code"] for item in default] == ["A"]
    by_code = {item["code"]: item for item in diagnostic}
    assert by_code["B"]["signal_eligible"] is False
    assert by_code["B"]["selected_for_trading"] is False
    assert by_code["B"]["value_falsified"] is True


def test_candidate_nan_text_fields_do_not_falsify_value_model():
    selected = normalize_candidate_snapshots({"2026-07-10": [{
        "code": "A",
        "candidate_source": "value_model",
        "quality_score": 80,
        "earnings_yoy": 0.20,
        "mktcap": 150,
        "value_falsification_reason": float("nan"),
        "candidate_failure_reason": float("nan"),
    }]})["2026-07-10"]

    assert selected[0]["allow_left"] is True
    assert selected[0]["value_falsification_reason"] == ""
    assert selected[0]["candidate_failure_reason"] == ""
    assert selected[0]["value_falsified"] is False


def test_model_candidates_require_visible_100b_market_cap():
    missing = normalize_candidate({
        "code": "MISSING",
        "candidate_source": "growth_leadership",
        "quality_score": 90,
        "earnings_yoy": 0.30,
    })
    small = normalize_candidate({
        "code": "SMALL",
        "candidate_source": "standard_mainline",
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 99.9,
    })
    enough = normalize_candidate({
        "code": "ENOUGH",
        "candidate_source": "value_model",
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 100.0,
    })

    assert missing["signal_eligible"] is False
    assert missing["candidate_failure_reason"] == "mktcap_missing"
    assert small["signal_eligible"] is False
    assert small["candidate_failure_reason"] == "mktcap_below_100"
    assert enough["signal_eligible"] is True

    diagnostic = normalize_candidate_snapshots(
        {"2026-07-10": [missing, small, enough]},
        include_diagnostics=True,
    )["2026-07-10"]

    assert {item["code"] for item in diagnostic} == {"MISSING", "SMALL", "ENOUGH"}


def test_value_grid_keeps_core_sells_two_upper_units_and_rebuys_them():
    dates = pd.bdate_range("2026-01-01", periods=84)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 84,
        "high": [100.5] * 80 + [100.5, 106.0, 111.0, 100.5],
        "low": [99.5] * 80 + [99.0, 100.0, 100.0, 99.0],
        "close": [100.0] * 84,
        "volume": [1000.0] * 84,
    })
    first = dates[80].strftime("%Y-%m-%d")
    last = dates[83].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {first: [{
            "code": "A", "candidate_source": "value_model",
            "value_line": 100.0, "industry": "test", "mktcap": 150,
        }]},
        {}, requested_start=first, end_date=last,
    )

    left_buys = [event for event in result["events"] if event["action"] == "左侧网格买入"]
    left_sells = [event for event in result["events"] if event["action"] == "左侧网格卖出"]
    assert len(left_buys) == 7
    assert [event["grid_slot"] for event in left_sells] == [3, 4]
    assert not [event for event in result["events"] if event["action"] == "右侧买入"]
    assert result["final_positions"][0]["position_mode"] == "left"
    assert result["final_positions"][0]["left_position_pct"] == pytest.approx(10)
    assert {item["grid_slot"] for item in result["final_positions"][0]["left_batches"]} == set(range(5))


def test_value_grid_candidate_removal_stops_new_left_buys_without_clearing():
    dates = pd.bdate_range("2026-01-01", periods=83)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 81 + [95.0, 90.0],
        "high": [100.5] * 83,
        "low": [99.5] * 81 + [94.0, 89.0],
        "close": [100.0] * 83,
        "volume": [1000.0] * 83,
    })
    first = dates[80].strftime("%Y-%m-%d")
    last = dates[82].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {
            first: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
            dates[81].strftime("%Y-%m-%d"): [],
        },
        {}, requested_start=first, end_date=last,
    )

    assert [event["action"] for event in result["events"]].count("左侧网格买入") == 5
    assert result["final_positions"][0]["position_mode"] == "left"
    assert {item["grid_slot"] for item in result["final_positions"][0]["left_batches"]} == set(range(5))


def test_left_to_right_stops_grid_and_uses_right_side_batches(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=83)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 81 + [109.0, 110.0],
        "high": [100.5] * 81 + [110.0, 111.0],
        "low": [99.5] * 83,
        "close": [100.0] * 81 + [109.0, 110.0],
        "volume": [1000.0] * 83,
    })
    first = dates[80].strftime("%Y-%m-%d")
    turn = dates[81].strftime("%Y-%m-%d")
    last = dates[82].strftime("%Y-%m-%d")

    def fake_signal(data, index, plan=None, **kwargs):
        if data.iloc[index]["date"].strftime("%Y-%m-%d") != turn:
            return None
        return {
            "rank": 2,
            "stop": 90.0,
            "trigger": 100.0,
            "order_type": "close",
            "reason": "test left-to-right",
            "known_volume_ratio": 1.0,
            "signal_type": "uptrend_50_reclaim",
            "entry_evidence_score": 6,
        }

    monkeypatch.setattr(
        "stock_research.strategies.portfolio_backtest._right_signal",
        fake_signal,
    )

    result = run_portfolio_backtest(
        {"A": bars},
        {
            first: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
            turn: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
            last: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
        },
        {}, requested_start=first, end_date=last,
    )

    actions = [event["action"] for event in result["events"]]
    assert "左转右接管左仓" in actions
    assert "左侧网格卖出" not in actions
    position = result["final_positions"][0]
    assert position["position_mode"] == "right"
    assert position["left_batches"] == []
    assert all(item["batch"].startswith("R") for item in position["batches"])
    promoted = [
        item for item in position["batches"]
        if item.get("origin_account") == "left"
    ]
    assert {item["origin_batch"] for item in promoted} == {
        "L1", "L2", "L3", "L4", "L5",
    }
    assert all(item["batch"].startswith("R") for item in promoted)
    assert {round(item["stop"], 3) for item in position["batches"]} == {90.0}
    switch_index = actions.index("左转右接管左仓")
    managed_after_switch = result["events"][switch_index + 1:]
    assert not any(
        str(event["reason"]).startswith("L")
        for event in managed_after_switch
    )


def test_value_grid_financial_falsification_exits_left_at_next_open():
    dates = pd.bdate_range("2026-01-01", periods=83)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 82 + [90.0],
        "high": [100.5] * 83,
        "low": [99.5] * 82 + [89.5],
        "close": [100.0] * 83,
        "volume": [1000.0] * 83,
    })
    first = dates[80].strftime("%Y-%m-%d")
    fail = dates[81].strftime("%Y-%m-%d")
    last = dates[82].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {
            first: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
            fail: [{
                "code": "A", "candidate_source": "value_model",
                "signal_eligible": False,
                "selected_for_trading": False,
                "value_falsified": True,
                "value_falsification_reason": "earnings_yoy_below_10pct",
                "candidate_failure_reason": "value_financial_falsification",
            }],
        },
        {}, requested_start=first, end_date=last,
    )

    exits = [event for event in result["events"] if event["action"] == "左侧价值证伪清仓"]
    assert len(exits) == 5
    assert {event["date"] for event in exits} == {last}
    assert result["final_positions"] == []


def test_left_core_only_position_does_not_consume_capacity():
    dates = pd.bdate_range("2026-01-01", periods=82)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 82,
        "high": [100.5] * 81 + [111.0],
        "low": [99.5] * 82,
        "close": [100.0] * 82,
        "volume": [1000.0] * 82,
    })
    first = dates[80].strftime("%Y-%m-%d")
    last = dates[81].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars},
        {
            first: [{
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "industry": "test", "mktcap": 150,
            }],
            last: [],
        },
        {}, requested_start=first, end_date=last,
    )

    position = result["final_positions"][0]
    assert position["left_position_pct"] == pytest.approx(6)
    assert {item["grid_slot"] for item in position["left_batches"]} == {0, 1, 2}
    assert position["capacity_counted"] is False


def test_all_market_phases_allow_only_one_left_symbol():
    dates = pd.bdate_range("2026-01-01", periods=81)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 81,
        "high": [100.5] * 81,
        "low": [99.5] * 80 + [99.0],
        "close": [100.0] * 81,
        "volume": [1000.0] * 81,
    })
    date = dates[-1].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars, "B": bars},
        {date: [
            {
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "candidate_score": 20, "mktcap": 150,
            },
            {
                "code": "B", "candidate_source": "value_model",
                "value_line": 100.0, "candidate_score": 10, "mktcap": 150,
            },
        ]},
        {date: {"phase": "waiting", "window_up_streak": 0}},
        requested_start=date, end_date=date, max_positions=5,
    )

    assert [item["code"] for item in result["final_positions"]] == ["A"]
    assert result["maximum_left_side_symbols"] == 1
    assert result["maximum_right_market_left_side_symbols"] == 0


def test_waiting_market_does_not_seed_multiple_left_symbols():
    dates = pd.bdate_range("2026-01-01", periods=83)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 83,
        "high": [100.5] * 83,
        "low": [99.5] * 80 + [99.0, 99.0, 99.0],
        "close": [100.0] * 83,
        "volume": [1000.0] * 83,
    })
    first = dates[80].strftime("%Y-%m-%d")
    second_day = dates[81].strftime("%Y-%m-%d")
    third_day = dates[82].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars, "B": bars},
        {first: [
            {
                "code": "A", "candidate_source": "value_model",
                "value_line": 100.0, "candidate_score": 20, "mktcap": 150,
            },
            {
                "code": "B", "candidate_source": "value_model",
                "value_line": 100.0, "candidate_score": 10, "mktcap": 150,
            },
        ]},
        {
            first: {"phase": "waiting", "window_up_streak": 0},
            second_day: {"phase": "waiting", "window_up_streak": 0},
            third_day: {"phase": "watch", "window_up_streak": 3},
        },
        requested_start=first, end_date=third_day, max_positions=5,
    )

    assert [item["code"] for item in result["final_positions"]] == ["A"]
    quota_exits = [
        event for event in result["events"]
        if event["action"] == "左侧全行情限额清仓"
    ]
    assert quota_exits == []
    assert result["maximum_left_side_symbols"] == 1


def test_left_core_still_blocks_a_second_left_symbol():
    dates = pd.bdate_range("2026-01-01", periods=83)
    bars = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 83,
        "high": [100.5] * 81 + [111.0, 100.5],
        "low": [99.5] * 81 + [99.5, 99.0],
        "close": [100.0] * 83,
        "volume": [1000.0] * 83,
    })
    first = dates[80].strftime("%Y-%m-%d")
    sell_non_core = dates[81].strftime("%Y-%m-%d")
    try_sixth = dates[82].strftime("%Y-%m-%d")
    initial_codes = ["A", "B", "C", "D", "E"]

    result = run_portfolio_backtest(
        {code: bars for code in initial_codes + ["F"]},
        {
            first: [
                {
                    "code": code, "candidate_source": "value_model",
                    "value_line": 100.0, "candidate_score": 100 - index,
                    "industry": f"industry-{code}", "mktcap": 150,
                }
                for index, code in enumerate(initial_codes)
            ],
            sell_non_core: [],
            try_sixth: [{
                "code": "F", "candidate_source": "value_model",
                "value_line": 100.0, "candidate_score": 200,
                "industry": "industry-F", "mktcap": 150,
            }],
        },
        {}, requested_start=first, end_date=try_sixth,
        max_positions=5, max_total_held_symbols=5,
    )

    assert [item["code"] for item in result["final_positions"]] == ["A"]
    assert result["maximum_left_side_symbols"] == 1
    assert result["maximum_total_held_symbols"] == 1
    assert all(item["capacity_counted"] is False for item in result["final_positions"])


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


def test_consolidation_breakout_requires_final_daily_volume_confirmation():
    bars = breakout_bars()
    bars.loc[bars.index[-1], "volume"] = 1
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
    )

    assert result["events"] == []


def test_entry_evidence_floor_rejects_single_unconfirmed_structure():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date, end_date=date,
        min_entry_evidence_score=6,
    )

    assert result["events"] == []
    assert result["min_entry_evidence_score"] == pytest.approx(6.0)


def test_generic_high_is_replaced_by_close_confirmed_consolidation_breakout():
    bars = _prepare_frame(breakout_bars())

    signal = infer_technical_entry(bars, len(bars) - 1)

    assert signal["signal_type"] == "consolidation_breakout"
    assert signal["order_type"] == "close"
    assert signal["trigger"] == pytest.approx(10.0)
    assert signal["stop"] == pytest.approx(10.0)


def test_strong_trend_breakout_requires_ma_trend_leadership_and_volume():
    dates = pd.bdate_range("2025-01-01", periods=121)
    closes = [60.0 + index * 0.5 for index in range(120)] + [125.0]
    volumes = [1000.0] * 120 + [1500.0]
    data = _prepare_frame(pd.DataFrame({
        "date": dates,
        "open": [value * 0.99 for value in closes],
        "high": [value * 1.01 for value in closes],
        "low": [value * 0.98 for value in closes],
        "close": closes,
        "volume": volumes,
    }))

    signal = infer_technical_entry(data, len(data) - 1)

    assert signal["signal_type"] == "strong_trend_breakout"
    assert signal["order_type"] == "close"
    assert signal["entry_evidence_score"] >= 7


def test_long_cycle_deduction_only_scores_an_existing_large_wave_structure():
    dates = pd.bdate_range("2025-01-01", periods=151)
    closes = [5.0 + index * 10.0 / 150 for index in range(151)]
    volumes = [1000.0] * 151
    volumes[29] = 500.0
    volumes[149] = 1200.0
    data = _prepare_frame(pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value * 1.01 for value in closes],
        "low": [value * 0.99 for value in closes],
        "close": closes,
        "volume": volumes,
    }))
    level = float(data.iloc[149]["ma120"])

    scored = apply_entry_confluence(data, 150, {
        "rank": 4,
        "trigger": level,
        "stop": level,
        "order_type": "stop",
        "signal_type": "uptrend_50_reclaim",
        "anchor_low": 5.0,
        "anchor_high": 15.0,
    })

    assert "deduction_low_price_volume_ma120" in scored["entry_evidence"]
    assert "large_wave_structure" in scored["entry_evidence"]
    assert scored["entry_evidence_score"] >= 9


def test_volume_price_node_is_usable_only_after_two_bar_confirmation():
    dates = pd.bdate_range("2026-01-01", periods=14)
    closes = [10.0] * 14
    closes[11] = 10.4
    volumes = [1000.0] * 14
    volumes[11] = 2000.0
    frame = pd.DataFrame({
        "date": dates,
        "open": [value - 0.1 for value in closes],
        "high": [value + 0.2 for value in closes],
        "low": [value - 0.2 for value in closes],
        "close": closes,
        "volume": volumes,
    })

    assert _valid_volume_price_nodes(frame.iloc[:12]) == []
    nodes = _valid_volume_price_nodes(frame.iloc[:13])

    assert len(nodes) == 1
    assert nodes[0]["date"] == dates[11].strftime("%Y-%m-%d")
    assert nodes[0]["confirmed_on"] == dates[12].strftime("%Y-%m-%d")
    assert nodes[0]["support"] == pytest.approx(9.8)


def test_breached_volume_price_node_never_revives_after_price_recovers():
    dates = pd.bdate_range("2026-01-01", periods=16)
    closes = [10.0] * 16
    closes[11] = 10.4
    closes[14] = 10.5
    closes[15] = 10.6
    volumes = [1000.0] * 16
    volumes[11] = 2000.0
    lows = [value - 0.2 for value in closes]
    lows[13] = 9.7
    frame = pd.DataFrame({
        "date": dates,
        "open": [value - 0.1 for value in closes],
        "high": [value + 0.2 for value in closes],
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    assert _valid_volume_price_nodes(frame) == []


def test_configured_price_structure_ratio_is_a_pre_known_condition_order():
    bars = breakout_bars()
    bars.loc[bars.index[-2], "close"] = 10.1
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [9.8, 10.2, 9.7, 10.1, 1]
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    plan = {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 8.0,
        "uptrend_high": 12.0, "ratio": 0.50,
        "confluence": ["MA20"],
    }]}
    plans = {"plans": {"A": {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 8.0,
        "uptrend_high": 12.0, "ratio": 0.50,
        "confluence": ["MA20"],
    }]}}}

    signal = _price_structure_signal(bars, len(bars) - 1, plan, auto_structure=False)
    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]}, {},
        requested_start=date, end_date=date, trade_plans=plans,
    )

    assert signal["signal_type"] == "uptrend_support_pullback"
    assert result["events"] == []


def test_symbol_cap_is_independent_from_price_structure_ratios():
    assert _capped_entry_size(0.60, 0.20) == pytest.approx(0.025)
    assert _capped_entry_size(0.30, 0.15) == pytest.approx(0.15)


def test_support_pullback_can_only_add_after_float_profit_buffer():
    dates = pd.bdate_range("2026-01-01", periods=82)
    bars = pd.DataFrame({
        "date": dates,
        "open": [10.0] * 79 + [10.0, 12.2, 12.3],
        "high": [10.0] * 79 + [11.0, 12.4, 12.5],
        "low": [10.0] * 79 + [10.0, 12.0, 11.9],
        "close": [10.0] * 79 + [11.0, 12.2, 12.2],
        "volume": [1000] * 79 + [3000, 1000, 1000],
    })
    start = dates[-3].strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    plans = {"plans": {"A": {"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 10.0,
        "uptrend_high": 14.0, "ratio": 0.50,
        "confluence": ["MA20"],
    }]}}}

    result = run_portfolio_backtest(
        {"A": bars}, {start: [{"code": "A"}]}, {},
        requested_start=start, end_date=end, trade_plans=plans,
    )

    buys = [event for event in result["events"] if event["position_change_pct"] > 0]
    assert len(buys) == 2
    assert buys[-1]["signal_type"] == "uptrend_support_pullback"


def test_close_confirmed_initial_batch_must_prove_before_addition():
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
    assert len(buys) == 1


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
    assert signal["reason"] == "回调波段50%收盘放量向上突破"
    assert signal["order_type"] == "close"
    assert rejected is None


def test_pullback_half_breakout_gap_up_gets_priority_bonus():
    bars = breakout_bars()
    bars.loc[bars.index[-2], ["open", "high", "low", "close"]] = [7.5, 7.9, 7.3, 7.8]
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [8.1, 8.5, 8.0, 8.3]
    plan = {"price_structures": [{
        "kind": "pullback_recovery", "uptrend_low": 2.0,
        "uptrend_high": 10.0, "pullback_low": 6.0,
        "consolidation_days": 13,
    }]}

    signal = _price_structure_signal(bars, len(bars) - 1, plan, auto_structure=False)

    assert signal["rank"] == 5
    assert signal["gap_up"] is True
    assert "跳空向上加分" in signal["reason"]


def test_pullback_half_breakout_takes_priority_over_support_pullback():
    bars = breakout_bars()
    bars.loc[bars.index[-2], ["open", "high", "low", "close"]] = [7.8, 7.95, 7.4, 7.9]
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [7.9, 8.5, 6.9, 8.3]
    plan = {"price_structures": [
        {
            "kind": "uptrend_support", "uptrend_low": 2.0,
            "uptrend_high": 10.0, "ratio": 0.625,
            "confluence": ["MA20"],
        },
        {
            "kind": "pullback_recovery", "uptrend_low": 2.0,
            "uptrend_high": 10.0, "pullback_low": 6.0,
            "consolidation_days": 13,
        },
    ]}

    signal = _price_structure_signal(bars, len(bars) - 1, plan, auto_structure=False)

    assert signal["signal_type"] == "pullback_50_breakout"


def test_preferred_right_breakout_uses_three_tenths_until_three_down_days():
    bars = breakout_bars()
    bars.loc[bars.index[-2], ["open", "high", "low", "close"]] = [7.5, 7.9, 7.3, 7.8]
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [8.1, 8.5, 8.0, 8.3]
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")
    plan = {"plans": {"A": {"price_structures": [{
        "kind": "pullback_recovery", "uptrend_low": 2.0,
        "uptrend_high": 10.0, "pullback_low": 6.0,
        "consolidation_days": 13,
    }]}}}

    up_result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]},
        {date: {"phase": "active", "window_up_streak": 3, "window_down_streak": 0}},
        requested_start=date, end_date=date, trade_plans=plan,
    )
    down_result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]},
        {date: {"phase": "watch", "window_up_streak": 0, "window_down_streak": 3}},
        requested_start=date, end_date=date, trade_plans=plan,
    )

    assert up_result["events"][0]["requested_position_pct"] == pytest.approx(30)
    assert "33未三日下行首仓" in up_result["events"][0]["reason"]
    assert down_result["events"][0]["requested_position_pct"] == pytest.approx(20)
    assert "33三日下行试错首仓" in down_result["events"][0]["reason"]


def test_tranche_profit_protection_is_independent_from_formula33():
    history = pd.DataFrame({
        "close": [100.0] * 14 + [80.0],
        "ma20": [90.0] * 6 + [91.0] * 9,
    })
    assert "profit_floor" in _active_profit_trigger_ids({
        "maximum_return_pct": 15,
        "profit_floor_triggered": True,
        "close": 80.0,
    }, history, 5)
    assert "trailing_10" in _active_profit_trigger_ids({
        "maximum_return_pct": 25,
        "trailing_10_triggered": True,
        "close": 80.0,
    }, history, 5)
    assert "divergence_time" in _active_profit_trigger_ids({
        "maximum_return_pct": 25,
        "divergence_time_take_profit": True,
        "close": 80.0,
    }, history, 5)


def test_valid_volume_node_break_sells_one_intermediate_profit_tranche():
    dates = pd.bdate_range("2026-01-01", periods=16)
    closes = [10.0] * 16
    closes[11] = 10.4
    volumes = [1000.0] * 16
    volumes[11] = 2000.0
    lows = [value - 0.2 for value in closes]
    lows[15] = 9.7
    history = pd.DataFrame({
        "date": dates,
        "open": [value - 0.1 for value in closes],
        "high": [value + 0.2 for value in closes],
        "low": lows,
        "close": closes,
        "volume": volumes,
        "ma20": [9.0] * 16,
    })

    trigger_ids = _active_profit_trigger_ids({
        "maximum_return_pct": 25,
        "close": closes[-1],
    }, history, 3)

    assert "volume_node_break:2026-01-16" in trigger_ids
    assert _profit_ids_to_execute(trigger_ids, 3) == [
        "volume_node_break:2026-01-16",
    ]


def test_maximum_profit_half_is_reserved_for_last_tranche():
    history = pd.DataFrame({"close": [100.0] * 10, "ma20": [90.0] * 10})
    profit = {
        "maximum_return_pct": 100.0,
        "half_profit_triggered": True,
        "close": 100.0,
    }

    assert "maximum_profit_half" not in _active_profit_trigger_ids(profit, history, 2)
    assert "maximum_profit_half" in _active_profit_trigger_ids(profit, history, 1)


def test_only_the_last_sold_down_profit_tranche_is_a_non_capacity_tail():
    state = PositionState(
        right=[{"size": 0.06}],
        right_parts=1,
        right_sold={"trailing_10"},
        right_tail_capacity_free=True,
    )
    assert _is_profit_tail(state)

    state.right_parts = 2
    assert not _is_profit_tail(state)

    state.right_parts = 1
    state.right_sold.clear()
    assert not _is_profit_tail(state)

    state.right_sold.add("trailing_10")
    state.right_tail_capacity_free = False
    assert not _is_profit_tail(state)


def test_profit_tail_requires_one_part_and_fifty_percent_current_return():
    assert _qualifies_profit_tail({"current_return_pct": 50.0}, 1, 0.50)
    assert not _qualifies_profit_tail({"current_return_pct": 49.9}, 1, 0.50)
    assert not _qualifies_profit_tail({"current_return_pct": 80.0}, 2, 0.50)


def test_three_or_five_parts_always_reserve_the_final_profit_tranche():
    signals = {"profit_floor", "trailing_10", "divergence_time", "ma20_break"}
    assert len(_profit_ids_to_execute(signals, 5)) == 4
    assert len(_profit_ids_to_execute(signals, 3)) == 2
    assert _profit_ids_to_execute({"trailing_10"}, 1) == []
    assert _profit_ids_to_execute({"maximum_profit_half"}, 1) == [
        "maximum_profit_half",
    ]


def test_four_symbol_cap_allows_four_but_blocks_third_same_industry(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=74)
    targets = {1001: 70, 1002: 71, 1003: 72, 1004: 73, 1005: 73}

    def frame(marker):
        closes = [10.0] * len(dates)
        target = targets[marker]
        closes[target:] = [12.0] * (len(dates) - target)
        return pd.DataFrame({
            "date": dates,
            "open": [10.0] * len(dates),
            "high": [12.0 if index == target else close for index, close in enumerate(closes)],
            "low": [10.0] * len(dates),
            "close": closes,
            "volume": [marker] * len(dates),
        })

    def fake_signal(data, index, plan=None, **kwargs):
        marker = int(data.iloc[0]["volume"])
        if index != targets[marker]:
            return None
        return {
            "rank": 2, "stop": 10.0, "trigger": 10.0,
            "order_type": "stop", "reason": "test breakout",
            "known_volume_ratio": 1.0,
        }

    monkeypatch.setattr(
        "stock_research.strategies.portfolio_backtest._right_signal",
        fake_signal,
    )
    codes = {f"S{marker}": frame(marker) for marker in targets}
    snapshots = {}
    for index in range(70, 74):
        date = dates[index].strftime("%Y-%m-%d")
        snapshots[date] = [
            {"code": "S1001", "candidate_score": 10, "mainline_boards": "通信设备"},
            {"code": "S1002", "candidate_score": 20, "mainline_boards": "通信网络设备及器件"},
            {"code": "S1003", "candidate_score": 30, "mainline_boards": "半导体"},
            {"code": "S1005", "candidate_score": 50, "mainline_boards": "通信设备"},
            {"code": "S1004", "candidate_score": 40, "mainline_boards": "电力设备"},
        ]
    formula = {
        date: {"phase": "active", "window_up_streak": 3, "window_down_streak": 0}
        for date in snapshots
    }

    result = run_portfolio_backtest(
        codes, snapshots, formula,
        requested_start=min(snapshots), end_date=max(snapshots),
        max_positions=4, max_same_industry=2,
    )

    held = {item["code"] for item in result["final_positions"]}
    assert held == {"S1001", "S1002", "S1003", "S1004"}
    assert "S1005" not in held
    assert result["max_positions"] == 4
    assert result["max_same_industry"] == 2


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


def test_author_case_junsheng_keeps_support_and_recovery_half_separate():
    structures = configured_price_structures({"price_structures": [
        {
            "kind": "uptrend_support", "uptrend_low": 13.30,
            "uptrend_high": 39.98, "ratio": 0.50,
            "confluence": ["annual_ma"],
        },
        {
            "kind": "pullback_recovery", "uptrend_low": 13.30,
            "uptrend_high": 39.98, "pullback_low": 22.68,
            "consolidation_days": 13,
        },
    ]})

    support = next(item for item in structures if item["kind"] == "uptrend_support")
    recovery = next(item for item in structures if item["kind"] == "pullback_recovery")
    assert support["level"] == pytest.approx(26.64)
    assert recovery["recovery_half"] == pytest.approx(31.33)
    assert recovery["deep_pullback_confirmed"] is True
    assert recovery["amplitude_valid"] is True


def test_author_case_duofuduo_reproduces_u625_with_node_and_season_ma_context():
    dfd_high = (31.06 - 9.70 * 0.375) / 0.625
    structures = configured_price_structures({"price_structures": [{
        "kind": "uptrend_support", "uptrend_low": 9.70,
        "uptrend_high": dfd_high, "ratio": 0.625,
        "confluence": ["volume_node_29.62", "season_ma_low_deduction"],
    }]})

    assert structures[0]["level"] == pytest.approx(31.06)
    assert structures[0]["amplitude_valid"] is True
    assert len(structures[0]["confluence"]) == 2


def test_author_case_tinci_distinguishes_large_support_from_local_wave():
    assert structure_price(15.36, 61.44, 0.50) == pytest.approx(38.40)
    assert structure_price(36.31, 64.98, 0.50) == pytest.approx(50.645)
    assert structure_price(36.31, 64.98, 0.75) == pytest.approx(57.8125)
    # The local wave can locate an operating support, but is too narrow to
    # certify the larger trend-change/recovery-half entry by itself.
    assert not trend_amplitude_valid(36.31, 64.98)


def test_minimum_five_yuan_commission_is_applied_to_small_order():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_portfolio_backtest(
        {"A": bars}, {date: [{"code": "A"}]}, {},
        requested_start=date, end_date=date,
        commission_rate=0.000085, minimum_commission=5,
        initial_capital=20_000,
    )

    assert result["transaction_cost_pct"] == pytest.approx(0.025)
    assert result["events"][0]["realized_account_pct"] == pytest.approx(-0.025)
    assert result["events"][0]["execution_quantity"] == pytest.approx(500.0)
    assert result["final_cash"] == pytest.approx(14_495.0)
    assert result["final_positions"][0]["batches"][0]["quantity"] == pytest.approx(500.0)


def test_default_data_end_date_waits_until_daily_bar_is_ready():
    assert default_data_end_date("2026-07-13 15:30") == "2026-07-10"
    assert default_data_end_date("2026-07-13 16:01") == "2026-07-13"
    assert default_data_end_date("2026-07-12 18:00") == "2026-07-10"


def test_formula33_refresh_window_covers_long_backtest_range():
    window = formula33_refresh_window_args("2024-09-24", "2026-07-13")

    assert window["lookback"] >= 680
    assert window["history_days"] == 420


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


def test_kline_preflight_invalidates_manifest_when_cache_starts_late(tmp_path):
    universe = tmp_path / "stock_universe.csv"
    universe.write_text("code\nsh.600000\nsz.000001\n", encoding="utf-8")
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    pd.DataFrame({"date": ["2024-09-25", "2024-09-26"]}).to_csv(
        kline_dir / "sh_600000.csv", index=False,
    )
    pd.DataFrame({"date": ["2024-09-24", "2024-09-26"]}).to_csv(
        kline_dir / "sz_000001.csv", index=False,
    )
    manifest = tmp_path / "formula33.json"
    manifest.write_text("{}", encoding="utf-8")

    coverage = invalidate_formula33_manifest_if_kline_cache_incomplete(
        manifest_path=manifest,
        kline_directory=kline_dir,
        universe_path=universe,
        start_date="2024-09-24",
        end_date="2024-09-26",
    )

    assert coverage["complete"] is False
    assert coverage["missing_start_count"] == 1
    assert not manifest.exists()


def test_kline_coverage_summary_accepts_complete_cache(tmp_path):
    universe = tmp_path / "stock_universe.csv"
    universe.write_text("code\nsh.600000\n", encoding="utf-8")
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    pd.DataFrame({"date": ["2024-09-24", "2024-09-26"]}).to_csv(
        kline_dir / "sh_600000.csv", index=False,
    )

    coverage = summarize_kline_cache_coverage(
        kline_dir, universe, "2024-09-24", "2024-09-26",
    )

    assert coverage["complete"] is True
    assert coverage["universe_count"] == 1


def test_kline_coverage_summary_allows_post_start_ipo(tmp_path):
    universe = tmp_path / "stock_universe.csv"
    universe.write_text("code\nsz.001220\n", encoding="utf-8")
    basic = tmp_path / "stock_basic_20260713.csv"
    basic.write_text("code,ipoDate\nsz.001220,2026-02-03\n", encoding="utf-8")
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    pd.DataFrame({"date": ["2026-02-03", "2026-07-13"]}).to_csv(
        kline_dir / "sz_001220.csv", index=False,
    )

    coverage = summarize_kline_cache_coverage(
        kline_dir,
        universe,
        "2024-09-24",
        "2026-07-13",
        stock_basic_path=basic,
    )

    assert coverage["complete"] is True
    assert coverage["post_ipo_start_count"] == 1
    assert coverage["missing_start_count"] == 0


def test_financial_cache_period_helpers(tmp_path):
    (tmp_path / "600000_20240630.json").write_text("{}", encoding="utf-8")
    (tmp_path / "000001_20240630.json").write_text("{}", encoding="utf-8")
    (tmp_path / "000001_20240930.json").write_text("{}", encoding="utf-8")

    assert report_period_visible_date("2024-06-30") == pd.Timestamp("2024-08-31")
    assert report_period_visible_date("2024-09-30") == pd.Timestamp("2024-10-31")
    assert financial_cache_file_count(tmp_path, "2024-06-30") == 2
    assert financial_cache_file_count(tmp_path, "2024-09-30") == 1


def test_candidate_manifest_empty_dates(tmp_path):
    (tmp_path / "manifest.json").write_text(
        """
        {
          "snapshots": [
            {"date": "2024-09-24", "candidate_count": 10},
            {"date": "2024-09-25", "candidate_count": 0},
            {"date": "2024-09-26", "candidate_count": null}
          ]
        }
        """,
        encoding="utf-8",
    )

    assert candidate_manifest_empty_dates(tmp_path) == [
        "2024-09-25",
        "2024-09-26",
    ]


def test_empty_candidate_snapshot_file_loads_as_zero_candidates(tmp_path):
    (tmp_path / "candidates_2024-09-24.csv").write_text("", encoding="utf-8")

    snapshots = load_candidate_snapshots(tmp_path, "2024-09-24", "2024-09-24")

    assert snapshots == {"2024-09-24": []}
