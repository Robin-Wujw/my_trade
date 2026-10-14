import copy
import json

import pandas as pd
import pytest

from stock_research.research.structure_labels import executable_structure_label, CONTRACT


def example(periods=24, order_type="close"):
    dates = pd.bdate_range("2026-10-15", periods=periods)
    bars = pd.DataFrame({"date": dates, "open": 10., "high": 10.2, "low": 9.8,
                         "close": 10., "volume": 1_000_000., "tradestatus": 1,
                         "up_limit": 11., "down_limit": 9., "raw_to_qfq_factor": 1.})
    event = {"event_id": "event", "code": "sz.000001", "observation_date": "2026-10-15",
             "captured_at": "2026-10-15T17:00:00+08:00", "price_basis": "observation_raw_anchor",
             "planned_stop": 9., "planned_entry": 10.,
             "signal": json.dumps({"order_type": order_type})}
    kwargs = dict(horizon=20, available_at=f"{dates[-1].date()}T17:00:00+08:00",
                  source_sha256="c" * 64, corporate_actions_complete=True)
    return event, bars, dates, [], kwargs


def run(values):
    event, bars, sessions, actions, kwargs = values
    return executable_structure_label(event, bars, sessions, actions, **kwargs)


def test_horizon_is_visible_close_then_next_open_with_board_lots_and_all_costs():
    values = example()
    event, bars, dates, actions, kwargs = values
    bars.loc[20, ["close", "high"]] = 10.9
    bars.loc[21, ["open", "high", "low", "close", "up_limit", "down_limit"]] = [11.5, 11.6, 11.4, 11.5, 11.99, 9.81]
    result = run(values)
    assert result["status"] == "completed" and result["accounting_audit"]["status"] == "passed"
    buy, sell = [row for row in result["execution"]["events"] if row.get("execution_quantity")]
    assert buy["date"] == str(dates[1].date()) and buy["execution_quantity"] == 9900
    assert sell["date"] == str(dates[21].date()) and sell["execution_price"] == 11.5
    assert result["initial_risk_amount"] == 9900.
    expected = 9900 * 1.5 - 99000 * (.000085 + .001) - 113850 * (.000085 + .001 + .0005)
    assert result["net_profit_amount"] == pytest.approx(expected)
    assert result["net_r"] == pytest.approx(expected / 9900)


def test_daily_limit_touch_does_not_fill_or_roll_forward():
    values = example(order_type="limit")
    values[1].loc[1, ["open", "low", "close", "high"]] = [10.1, 10., 10.1, 10.2]
    result = run(values)
    assert result["status"] == "entry_not_filled" and result["reason"] == "touch_unconfirmed"
    assert result["net_r"] is None and result["executable"] is False


def test_buy_day_stop_cannot_sell_same_day_and_gap_stop_uses_next_day_price():
    values = example()
    values[1].loc[1, ["low", "down_limit"]] = [8., 8.]
    values[1].loc[2, ["open", "high", "low", "close", "down_limit"]] = [8., 8.2, 7.9, 8., 7.]
    result = run(values)
    fills = [row for row in result["execution"]["events"] if row.get("execution_quantity")]
    assert len(fills) == 2 and fills[1]["date"] == str(values[2][2].date())
    assert fills[1]["execution_price"] == 8. and result["net_r"] < -2.


def test_triggered_locked_stop_persists_after_price_rebound():
    values = example()
    values[1].loc[2, ["open", "high", "low", "close"]] = 9.
    values[1].loc[3, ["open", "high", "low", "close", "up_limit", "down_limit"]] = [10., 10.2, 9.8, 10., 10.5, 8.1]
    result = run(values)
    assert result["blocked_exits"] == [{"date": str(values[2][2].date()), "status": "locked_limit_down"}]
    assert result["exit_date"] == str(values[2][3].date())
    assert result["execution"]["events"][-1]["execution_price"] == 10.


def test_horizon_locked_exit_and_unsettled_open_position_remain_pending():
    values = example(periods=22)
    values[1].loc[21, ["open", "high", "low", "close"]] = 9.
    result = run(values)
    assert result["status"] == "pending_exit_or_settlement" and result["net_r"] is None
    assert result["execution"]["final_positions"][0]["quantity"] == 9900
    assert result["accounting_audit"]["error_count"] == 0


@pytest.mark.parametrize("change,reason", [
    ("missing_bar", "missing_exchange_session_bar_not_assumed_suspended"),
    ("missing_limit", "incomplete_execution_prices_limits_or_status"),
    ("unknown_adjustment", "adjustment_change_without_documented_corporate_action"),
    ("unknown_actions", "corporate_action_coverage_not_proven"),
    ("unproven_basis", "observation_raw_signal_basis_not_proven"),
])
def test_incomplete_sources_never_become_no_trade_or_zero_return(change, reason):
    values = list(example())
    if change == "missing_bar":
        values[1] = values[1].drop(index=3)
    elif change == "missing_limit":
        values[1].loc[3, "down_limit"] = float("nan")
    elif change == "unknown_adjustment":
        values[1].loc[3:, "raw_to_qfq_factor"] = 1.2
    elif change == "unknown_actions":
        values[4]["corporate_actions_complete"] = False
    else:
        values[0].pop("price_basis")
    result = run(values)
    assert result["status"] == "data_limited" and result["reason"] == reason and result["net_r"] is None


def test_pending_horizon_and_future_sources_do_not_supply_current_returns():
    values = example(periods=21)
    assert run(values)["status"] == "pending_horizon"
    original = run(values)
    future = values[1].iloc[[-1]].assign(date=pd.Timestamp("2030-01-01"), close=999., high=999.)
    extended = list(values)
    extended[1] = pd.concat([values[1], future])
    extended[2] = list(values[2]) + [pd.Timestamp("2030-01-01")]
    assert run(extended) == original


def test_before_close_outcome_capture_cannot_use_same_day_execution_or_close():
    values = example(periods=22)
    values[4]["available_at"] = f"{values[2][-1].date()}T09:00:00+08:00"
    result = run(values)
    assert result["status"] == "pending_horizon" and result["net_r"] is None
    assert not any(row.get("execution_quantity", 0) < 0 for row in result["execution"]["events"])


def test_raw_quote_outside_precise_limit_cannot_become_an_executable_label():
    values = example()
    values[1].loc[21, ["open", "high", "close"]] = 12.
    result = run(values)
    assert result["status"] == "data_limited" and result["reason"] == "raw_prices_conflict_with_exact_limits"


def test_registered_dividend_waits_for_actual_pay_and_reconciles_independently():
    values = list(example(periods=24))
    _, bars, dates, _, _ = values
    values[3] = [{"code": "sz.000001", "record_date": str(dates[3].date()),
                  "ex_date": str(dates[4].date()), "pay_date": str(dates[23].date()), "cash_div": 1.}]
    bars.loc[4:, ["open", "close"]] = 9.
    bars.loc[4:, "high"] = 9.2
    bars.loc[4:, "low"] = 8.8
    bars.loc[4:, "down_limit"] = 8.1
    bars.loc[4:, "up_limit"] = 9.9
    bars.loc[4:, "raw_to_qfq_factor"] = 1.1
    result = run(values)
    assert result["status"] == "completed" and result["accounting_audit"]["error_count"] == 0
    assert result["execution"]["equity_curve"][-1]["date"] == str(dates[23].date())
    assert result["net_profit_amount"] < 0  # cash dividend offsets price fall; fees remain
    pending = copy.deepcopy(values)
    pending[4]["available_at"] = f"{dates[21].date()}T17:00:00+08:00"
    assert run(pending)["status"] == "pending_exit_or_settlement"
    missing = copy.deepcopy(values)
    missing[3][0].pop("pay_date")
    assert run(missing)["status"] == "data_limited"


def test_bonus_shares_are_not_sold_before_listing_and_cost_is_preserved():
    values = list(example(periods=24))
    _, bars, dates, _, _ = values
    values[3] = [{"code": "sz.000001", "record_date": str(dates[3].date()),
                  "ex_date": str(dates[4].date()), "div_listdate": str(dates[23].date()), "stk_div": 1.}]
    for field in ["open", "high", "low", "close", "up_limit", "down_limit"]:
        bars.loc[4:, field] /= 2
    bars.loc[4:, "raw_to_qfq_factor"] = 2.
    result = run(values)
    assert result["status"] == "completed" and result["accounting_audit"]["error_count"] == 0
    sells = [row for row in result["execution"]["events"] if row.get("execution_quantity", 0) < 0]
    assert [(row["date"], row["execution_quantity"]) for row in sells] == [
        (str(dates[21].date()), -9900), (str(dates[23].date()), -9900)]
    assert result["initial_risk_amount"] == 9900. and result["net_profit_amount"] < 0


def test_historical_stamp_duty_uses_actual_sell_date():
    values = list(example())
    dates = pd.bdate_range("2023-07-03", periods=24)
    values[0].update(observation_date=str(dates[0].date()), captured_at=f"{dates[0].date()}T17:00:00+08:00")
    values[1]["date"] = dates
    values[2] = dates
    values[4]["available_at"] = f"{dates[-1].date()}T17:00:00+08:00"
    result = run(values)
    sell = [row for row in result["execution"]["events"] if row.get("execution_quantity", 0) < 0][0]
    assert sell["stamp_duty_amount"] == 99000 * CONTRACT["sell_stamp_duty_rate_before_2023_08_28"]


@pytest.mark.parametrize("value", ["unavailable", float("inf"), -1., True])
def test_invalid_action_terms_cannot_be_silently_normalized_to_no_distribution(value):
    values = list(example())
    dates = values[2]
    values[3] = [{"code": "sz.000001", "record_date": str(dates[3].date()),
                  "ex_date": str(dates[4].date()), "pay_date": str(dates[5].date()), "cash_div": value}]
    result = run(values)
    assert result["status"] == "data_limited" and result["reason"] == "invalid_corporate_action_terms"


@pytest.mark.parametrize("terms", [{"cash_div": None}, {"cash_div": float("nan")}, {"rights_ratio": .3}])
def test_missing_or_unsupported_action_terms_do_not_prove_zero_distribution(terms):
    values = list(example())
    dates = values[2]
    values[3] = [{"code": "sz.000001", "record_date": str(dates[3].date()),
                  "ex_date": str(dates[4].date()), "pay_date": str(dates[5].date()), **terms}]
    result = run(values)
    assert result["status"] == "data_limited" and result["net_r"] is None
