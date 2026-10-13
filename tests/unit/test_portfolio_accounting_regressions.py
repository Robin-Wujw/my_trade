"""Small, deterministic account scenarios independent of stock selection."""
import pandas as pd
import pytest

from stock_research.backtesting import engine as engine
from stock_research.reporting.portfolio_attribution import build_portfolio_attribution


def scenario(monkeypatch, *, actions=None, fee=0, capital=10_000,
             future_name=None, first_open=10, security_end=None, split=1,
             profit=False, execution="next_open", order_type="close", target=14, missing_ex_bar=False):
    dates = pd.bdate_range("2025-01-02", periods=70)
    bars = pd.DataFrame({"date": dates, "open": 10., "high": 10.1,
                         "low": 9.9, "close": 10., "volume": 100_000})
    bars.loc[61, ["open", "high", "low", "close"]] = [first_open] * 4
    if split != 1:
        bars["raw_to_qfq_factor"] = float(split)
        bars.loc[62:, "raw_to_qfq_factor"] = 1.
        bars.loc[62:, ["open", "high", "low", "close"]] /= split
    if profit:
        bars.loc[63:, ["open", "high", "low", "close"]] *= 1.2

    if missing_ex_bar:
        bars = bars.drop(index=62).reset_index(drop=True)

    def signal(data, index, *args, **kwargs):
        if data.iloc[index]["date"] != dates[60]:
            return None
        factor = float(data.iloc[index]["qfq_to_raw_factor"])
        return {"rank": 4, "trigger": 10 / factor, "stop": 9.5 / factor,
                "order_type": order_type, "reason": "account fixture",
                "known_volume_ratio": 1., "signal_type": "consolidation_breakout"}

    def gate(data, index, candidate, signal, **kwargs):
        return True, {**signal, "semantic_entry_ok": True, "entry_risk_pct": 5.,
                      "entry_reward_risk": 8., "entry_target_price":
                      target / float(data.iloc[index]["qfq_to_raw_factor"])}, "ok"

    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", gate)
    monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **kw: {})
    monkeypatch.setattr(engine, "_active_profit_trigger_ids",
                        lambda p, h, n: ({"maximum_profit_half"} if n == 1 else {"profit_a", "profit_b", "profit_c"})
                        if profit and h.iloc[-1]["date"] >= dates[64] else set())
    candidate = {"code": "sz.000001", "name": "普通股", "candidate_score": 99,
                 "allow_right": True, "allow_left": False,
                 "signal_eligible": True, "selected_for_trading": True}
    snapshots = {str(dates[59].date()): [candidate]}
    if future_name:
        snapshots[str((dates[-1] + pd.Timedelta(days=10)).date())] = [{**candidate, "name": future_name}]
    result = engine.run_portfolio_backtest(
        {"sz.000001": bars}, snapshots, {}, requested_start=str(dates[59].date()),
        end_date=str(dates[-1].date()), signals_effective_next_day=True,
        close_confirmed_execution=execution, initial_capital=capital,
        commission_rate=0, minimum_commission=fee,
        estimated_slippage_rate=0,
        sell_stamp_duty_rate_before_2023_08_28=None,
        corporate_actions={"sz.000001": actions(dates)} if actions else None,
        security_end_dates={"sz.000001": dates[63]} if security_end else None,
    )
    return result, dates


def audited_scenario(monkeypatch, **options):
    import inspect
    from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting

    original = engine.run_portfolio_backtest
    captured = {}

    def run(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "run_portfolio_backtest", run)
    result, dates = scenario(monkeypatch, **options)

    def audit(value=result):
        return audit_portfolio_accounting(value, captured["price_frames"],
                                          captured["corporate_actions"], captured)

    return result, dates, audit


@pytest.mark.parametrize("options", [{"fee": 5}, {"fee": 5, "security_end": True},
                                     {"profit": True, "execution": "close_proxy"}])
def test_independent_ledger_rebuilds_fills_and_fees(monkeypatch, options):
    _, _, audit = audited_scenario(monkeypatch, **options)
    report = audit()
    assert report["status"] == "passed", report


def test_independent_ledger_rebuilds_registered_dividends_and_bonus(monkeypatch):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "pay_date": str(dates[65].date()), "div_listdate": str(dates[66].date()),
                 "cash_div": .3, "stk_div": .5}]
    _, _, audit = audited_scenario(monkeypatch, actions=actions, split=1.5, fee=5)
    report = audit()
    assert report["status"] == "passed", report


@pytest.mark.parametrize("damage", ["cash", "shares", "fees", "phase", "empty"])
def test_independent_ledger_rejects_corrupted_results(monkeypatch, damage):
    from copy import deepcopy
    result, _, audit = audited_scenario(monkeypatch, fee=5)
    corrupted = deepcopy(result)
    if damage == "cash":
        corrupted["equity_curve"][-1]["cash"] += 100
        corrupted["equity_curve"][-1]["equity"] += .01
        corrupted["events"][0]["cash_change_amount"] += 100
        corrupted["events"][0]["cash_change_amount_raw"] += 100
    elif damage == "shares":
        corrupted["final_positions"][0]["quantity"] += 1
    elif damage == "fees":
        corrupted["events"][0]["transaction_cost_amount_raw"] = 0
    elif damage == "phase":
        corrupted["events"][0]["execution_phase"] = "unknown"
    else:
        corrupted["events"] = []
        corrupted["equity_curve"] = []
        corrupted["final_positions"] = []
    assert audit(corrupted)["status"] == "failed"


def test_later_st_name_cannot_change_an_earlier_open_order(monkeypatch):
    normal, _ = scenario(monkeypatch, first_open=10.49)
    future_st, _ = scenario(monkeypatch, first_open=10.49, future_name="*ST未来")
    key = lambda r: [(t["date"], t["execution_quantity"], t["execution_price"]) for t in r["trade_ledger"]]
    assert key(normal)
    assert key(normal) == key(future_st)


def test_initial_cash_is_the_first_drawdown_peak(monkeypatch):
    result, _ = scenario(monkeypatch, fee=5)
    expected = min(row["equity"] for row in result["equity_curve"]) - 1
    assert result["maximum_drawdown_pct"] == pytest.approx(expected * 100, abs=.001)


def test_account_channel_closed_pnl_does_not_double_charge_entry_fees(monkeypatch):
    result, _ = scenario(monkeypatch, fee=5, security_end=True)
    sells = [t for t in result["trade_ledger"] if t["execution_quantity"] < 0]
    assert sells
    assert sum(result["trade_mix_summary"]["net_pnl_by_account"].values()) == pytest.approx(
        sum(t["profit_loss_amount"] for t in sells))


@pytest.mark.parametrize("execution", ["next_open", "close_proxy"])
def test_final_profit_exit_sells_the_odd_lot_without_losing_shares(monkeypatch, execution):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "div_listdate": str(dates[62].date()), "stk_div": .5}]
    result, _ = scenario(monkeypatch, actions=actions, split=1.5, profit=True, execution=execution)
    buys = sum(t["execution_quantity"] for t in result["trade_ledger"] if t["execution_quantity"] > 0)
    sells = -sum(t["execution_quantity"] for t in result["trade_ledger"] if t["execution_quantity"] < 0)
    assert buys > 0
    assert sells == buys * 1.5
    assert not result["final_positions"]
    assert result["capital_risk_attribution"]["origin_pnl_reconciliation_difference"] == pytest.approx(0, abs=.01)


def test_cash_dividend_is_receivable_until_pay_date(monkeypatch):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "pay_date": str(dates[66].date()), "cash_div": 5.}]
    result, dates = scenario(monkeypatch, actions=actions)
    curve = {r["date"]: r for r in result["equity_curve"]}
    buy = result["trade_ledger"][0]
    amount = buy["execution_quantity"] * 5
    before = curve[str(dates[61].date())]["cash"]
    assert curve[str(dates[62].date())]["cash"] == before
    assert curve[str(dates[62].date())]["dividend_receivable"] == amount
    assert curve[str(dates[66].date())]["cash"] == before + amount
    assert result["final_positions"][0]["quantity"] == buy["execution_quantity"]
    concentration = result["profit_concentration_summary"]
    assert concentration["net_total_pnl_amount_raw"] == pytest.approx(amount)
    assert concentration["top1_symbol"]["dividend_receivable_amount"] == 0.
    assert result["capital_risk_attribution"]["origin_pnl_reconciliation_difference"] == pytest.approx(0, abs=.01)


def test_zero_equity_attribution_has_defined_return_and_no_next_year_division():
    curve = [{"date": "2024-01-02", "equity": 1., "cash": 100,
              "market_exposure_pct": 0, "total_held_symbol_count": 0},
             {"date": "2024-12-31", "equity": 0., "cash": 0,
              "market_exposure_pct": None, "total_held_symbol_count": 0},
             {"date": "2025-01-02", "equity": 0., "cash": 0,
              "market_exposure_pct": None, "total_held_symbol_count": 0}]
    result = build_portfolio_attribution([], curve, [], 100)
    assert result["cagr_pct"] == -100
    assert result["annual"][0]["return_pct"] == -100
    assert result["annual"][1]["return_pct"] is None


def test_five_percent_open_gap_is_not_rejected_by_float_rounding(monkeypatch):
    result, _ = scenario(monkeypatch, first_open=10.5)
    assert result["trade_ledger"]
    assert not any(e["reason"] == "next_open_gap_above_chase_limit" for e in result["entry_blocks"])


def test_preplaced_stop_rechecks_reward_risk_at_the_actual_gap_fill(monkeypatch):
    result, _ = scenario(monkeypatch, first_open=10.49, order_type="stop", target=11.5)
    assert not result["trade_ledger"]
    assert any(e["reason"] == "next_open_reward_risk_too_low" for e in result["entry_blocks"])


def test_unlisted_bonus_is_written_off_on_delisting(monkeypatch):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "div_listdate": str(dates[67].date()), "stk_div": .5}]
    result, _, audit = audited_scenario(monkeypatch, actions=actions, split=1.5, security_end=True)
    assert not result["final_positions"]
    assert not any(e["action"] == "红股上市可售" for e in result["events"])
    assert audit()["status"] == "passed", audit()


def test_action_only_pay_date_is_not_delayed_until_next_price_bar(monkeypatch):
    def actions(dates):
        saturday = dates[62] + pd.offsets.Week(weekday=5)
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "pay_date": str(saturday.date()), "cash_div": .3}]
    result, dates, audit = audited_scenario(monkeypatch, actions=actions)
    pay = actions(dates)[0]["pay_date"]
    curve = {point["date"]: point for point in result["equity_curve"]}
    assert curve[pay]["dividend_receivable"] == 0
    assert not curve[pay]["is_trading_day"]
    assert result["capital_risk_attribution"]["trading_days"] == sum(row["is_trading_day"] for row in curve.values())
    assert any(e["date"] == pay and e["action"] == "现金分红到账" for e in result["events"])
    assert audit()["status"] == "passed", audit()


def test_bonus_listed_after_original_sale_can_finish_profit_exit(monkeypatch):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "div_listdate": str(dates[67].date()), "stk_div": .5}]
    result, _, audit = audited_scenario(monkeypatch, actions=actions, split=1.5, profit=True)
    assert not result["final_positions"]
    assert audit()["status"] == "passed", audit()


def test_conflicting_action_terms_block_credit_and_are_independently_flagged(monkeypatch):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "pay_date": str(dates[66].date()), "cash_div": amount} for amount in (.22, .81)]
    result, _, audit = audited_scenario(monkeypatch, actions=actions)
    assert not any(e.get("corporate_action_cash_amount") for e in result["events"])
    result["corporate_action_limitations"] = []
    checked = audit()
    assert checked["error_count"] == 0
    assert checked["status"] == "data_limited"
    assert checked["data_limitations"][0]["reason"] == "conflicting_action_revisions"


def test_date_enrichment_is_one_dividend_in_both_ledgers(monkeypatch):
    def actions(dates):
        base = {"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()), "cash_div": .1}
        return [base, {**base, "pay_date": str(dates[66].date())}]
    result, _, audit = audited_scenario(monkeypatch, actions=actions)
    assert result["final_dividend_receivable"] == 0
    assert len([e for e in result["events"] if e.get("corporate_action_cash_amount")]) == 1
    assert audit()["status"] == "passed", audit()


@pytest.mark.parametrize("cash", [0., .5])
def test_no_ex_date_bar_does_not_create_a_phantom_split_gain(monkeypatch, cash):
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()),
                 "div_listdate": str(dates[62].date()), "stk_div": .5,
                 "cash_div": cash, "pay_date": str(dates[66].date())}]
    result, dates, audit = audited_scenario(monkeypatch, actions=actions, split=1.5, missing_ex_bar=True)
    curve = {row["date"]: row for row in result["equity_curve"]}
    assert curve[str(dates[62].date())]["equity"] == pytest.approx(curve[str(dates[61].date())]["equity"])
    assert audit()["status"] == "passed", audit()


def test_unpaid_dividend_is_included_in_symbol_profit_and_rendered_report(monkeypatch):
    from stock_research.reporting.backtest_trade_report import render_trade_report_markdown
    def actions(dates):
        return [{"record_date": str(dates[61].date()), "ex_date": str(dates[62].date()), "cash_div": .123456}]
    result, _, audit = audited_scenario(monkeypatch, actions=actions, fee=5)
    summary = result["profit_concentration_summary"]
    expected = (result["equity_curve"][-1]["equity"] - 1) * 10_000
    assert summary["net_total_pnl_amount_raw"] == pytest.approx(expected)
    assert summary["top1_symbol"]["dividend_receivable_amount"] > 0
    assert audit()["error_count"] == 0
    assert "应收股利" in render_trade_report_markdown(result)


def test_initial_r_coverage_reports_all_sells_without_fabricating_missing_r():
    result = engine._portfolio_r_multiple_summary([
        {"execution_quantity": -100, "realized_r_multiple": 1.},
        {"execution_quantity": -100, "realized_r_multiple": None}])
    assert result["all_sell_count"] == 2
    assert result["sell_count"] == 1
    assert result["missing_initial_r_count"] == 1
    assert result["r_coverage_pct"] == 50.


def test_mixed_left_and_right_exit_has_no_whole_order_r(monkeypatch):
    from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
    dates = pd.bdate_range("2025-01-02", periods=70)
    bars = pd.DataFrame({"date": dates, "open": 100., "high": 100.5,
                         "low": 99.5, "close": 100., "volume": 100_000})
    bars.loc[61, ["open", "high", "low", "close"]] = [109., 110., 99.5, 109.]
    bars.loc[62, ["open", "high", "low", "close"]] = [110., 111., 109., 110.]
    bars.loc[63:, ["open", "high", "low", "close"]] = [125., 126., 124., 125.]
    def signal(data, index, *args, **kwargs):
        if data.iloc[index]["date"] != dates[61]:
            return None
        return {"rank": 4, "trigger": 109., "stop": 105., "order_type": "close",
                "known_volume_ratio": 1., "signal_type": "consolidation_breakout", "reason": "fixture"}
    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", lambda data, index, candidate, signal, **kw:
                        (True, {**signal, "semantic_entry_ok": True, "entry_target_price": 140.,
                                "entry_risk_pct": 5., "entry_reward_risk": 8.}, "ok"))
    monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **kw: {})
    monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda p, history, n:
                        ({"maximum_profit_half"} if n == 1 else {"profit_a", "profit_b"})
                        if history.iloc[-1]["date"] >= dates[66] else set())
    candidate = {"code": "A", "candidate_source": "value_model+standard_mainline", "value_line": 100.,
                 "quality_score": 90, "earnings_yoy": .3, "mktcap": 150, "price_to_value": 1.,
                 "industry": "汽车电子电气系统", "value_industry_allowed": True,
                 "value_industry_allowlist_match": "汽车电子电气系统", "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION}
    result = engine.run_portfolio_backtest({"A": bars}, {str(dates[59].date()): [candidate]}, {},
                                          requested_start=str(dates[59].date()), end_date=str(dates[-1].date()),
                                          profit_tranches=3)
    mixed = [e for e in result["trade_ledger"] if e["execution_quantity"] < 0
             and {a["origin_account"] for a in e["origin_allocations"]} == {"left", "right"}]
    assert mixed
    assert all(e.get("realized_r_multiple") is None for e in mixed)
