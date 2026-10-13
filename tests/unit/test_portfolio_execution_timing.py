"""Portfolio event order: frozen opening orders cannot use later bar outcomes."""
import pandas as pd
import pytest

from stock_research.backtesting import engine as engine


def _scenario(monkeypatch, *, execution_close=10.8, locked_open=False,
              intraday_stop=False, new_lot_stop=False):
    dates = pd.bdate_range("2025-01-02", periods=92)
    frames = {}
    for code in ("sz.000001", "sz.000002"):
        frames[code] = pd.DataFrame({
            "date": dates, "open": 10.0, "high": 10.1, "low": 9.9,
            "close": 10.0, "volume": 1_000_000, "test_code": code,
        })
    held = frames["sz.000001"]
    held.loc[65:84, ["open", "high", "low", "close"]] = [12, 12.1, 11.9, 12]
    held.loc[85:, ["open", "high", "low", "close"]] = [10.8, 10.9, 10.7, 10.8]
    held.loc[90, ["open", "high", "low", "close"]] = [
        9.72 if locked_open else 10.8,
        max(10.9, execution_close), 9.7 if locked_open else 10.7, execution_close,
    ]
    if intraday_stop:
        held.loc[85:, ["open", "high", "low", "close"]] = [12, 12.1, 11.9, 12]
        held.loc[90, "low"] = 8.5
    if new_lot_stop:
        frames["sz.000002"].loc[90, ["low", "close"]] = [8.5, 9.4]

    def signal(data, index, plan=None, **kwargs):
        code = data.iloc[index]["test_code"]
        when = dates[60] if code == "sz.000001" else dates[89]
        if data.iloc[index]["date"] != when:
            return None
        return {
            "rank": 4, "trigger": 10.0, "stop": 9.5,
            "order_type": "close", "reason": "timing fixture",
            "known_volume_ratio": 1.0, "signal_type": "consolidation_breakout",
        }

    def gate(data, index, candidate, signal, **kwargs):
        return True, {**signal, "semantic_entry_ok": True,
                      "entry_risk_pct": 5.0, "entry_reward_risk": 8.0,
                      "entry_target_price": 14.0}, "ok"

    def exit_snapshot(history, cost, entry_date, **kwargs):
        row = history.iloc[-1]
        if row["date"] == dates[90] and float(row["low"]) < 9.0:
            return {"space_stop_triggered": True, "space_stop": cost * 0.9,
                    "condition_stop_triggered": float(row["close"]) < 9.5,
                    "position_action": "空间止损"}
        return {}

    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", gate)
    monkeypatch.setattr(engine, "position_exit_snapshot", exit_snapshot)
    monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda *a: set())
    candidates = [{"code": code, "candidate_score": 99,
                   "allow_right": True, "allow_left": False,
                   "signal_eligible": True, "selected_for_trading": True}
                  for code in frames]
    result = engine.run_portfolio_backtest(
        frames, {str(dates[59].date()): candidates}, {},
        requested_start=str(dates[59].date()), end_date=str(dates[91].date()),
        close_confirmed_execution="next_open", signals_effective_next_day=True,
        right_entry_size_multiplier=3.0, commission_rate=0,
        minimum_commission=0, estimated_slippage_rate=0, sell_stamp_duty_rate=0,
    )
    return result, str(dates[90].date()), str(dates[91].date())


def test_open_rotation_is_invariant_to_execution_day_close(monkeypatch):
    weak, execution_date, _ = _scenario(monkeypatch, execution_close=10.8)
    strong, _, _ = _scenario(monkeypatch, execution_close=12.5)
    def opening_trades(result):
        return [(t["code"], t["trade_side"], t["execution_price"], t["quantity"])
                for t in result["trade_ledger"] if t["date"] == execution_date]
    assert opening_trades(weak) == opening_trades(strong)
    rotation = [t for t in weak["trade_ledger"] if t["action"] == "汰弱让位"]
    assert len(rotation) == 1
    assert rotation[0]["execution_price"] == pytest.approx(10.8)
    assert rotation[0]["execution_phase"] == "open"
    assert rotation[0]["decision_date"] < execution_date


def test_intraday_exit_cannot_fund_earlier_open_buy(monkeypatch):
    result, execution_date, _ = _scenario(monkeypatch, intraday_stop=True)
    trades = [t for t in result["trade_ledger"] if t["date"] == execution_date]
    assert [(t["code"], t["trade_side"]) for t in trades] == [
        ("sz.000002", "买入"), ("sz.000001", "卖出"),
    ]
    assert trades[0]["trade_amount"] == pytest.approx(300_000)
    assert trades[0]["cash_after"] >= 0


def test_locked_open_rotation_does_not_release_exposure(monkeypatch):
    result, execution_date, _ = _scenario(monkeypatch, locked_open=True)
    trades = [t for t in result["trade_ledger"] if t["date"] == execution_date]
    assert not any(t["action"] == "汰弱让位" for t in trades)
    buys = [t for t in trades if t["code"] == "sz.000002" and t["trade_side"] == "买入"]
    assert len(buys) == 1
    assert buys[0]["trade_amount"] == pytest.approx(300_000)


def test_open_purchase_obeys_t_plus_one_even_if_stop_hit(monkeypatch):
    result, execution_date, next_date = _scenario(monkeypatch, new_lot_stop=True)
    trades = [t for t in result["trade_ledger"] if t["code"] == "sz.000002"]
    assert trades[0]["date"] == execution_date and trades[0]["trade_side"] == "买入"
    assert not any(t["date"] == execution_date and t["trade_side"] == "卖出" for t in trades)
    assert any(t["date"] == next_date and t["trade_side"] == "卖出" for t in trades)


def test_open_rotation_cannot_expand_the_preallocated_open_buy_budget(monkeypatch):
    result, execution_date, _ = _scenario(monkeypatch)
    trades = [e for e in result["trade_ledger"] if e["date"] == execution_date]
    assert trades[0]["trade_side"] == "卖出"
    assert trades[1]["trade_side"] == "买入"
    spent = sum(e["trade_amount"] + e["transaction_cost_amount_raw"]
                for e in trades if e["trade_side"] == "买入")
    assert spent <= trades[0]["cash_before"]
    assert trades[1]["cash_before"] > trades[0]["cash_before"]


def test_preplaced_priority_never_reads_the_execution_day_proximity():
    signal = {"rank": 4, "technical_signal_timing": "previous_close_preplaced_order"}
    candidate = {"candidate_score": 80}
    item = (80, 80, 4, "A", 0, signal, candidate, None)
    low = {"A": pd.DataFrame({"structure_proximity_score": [0.]})}
    high = {"A": pd.DataFrame({"structure_proximity_score": [100.]})}
    assert engine._right_entry_priority_sort_key(item, low, mode=engine.ENTRY_PRIORITY_MODE_STRUCTURE) == engine._right_entry_priority_sort_key(item, high, mode=engine.ENTRY_PRIORITY_MODE_STRUCTURE)


def test_intraday_left_grid_sale_cannot_change_an_opening_right_buy(monkeypatch):
    from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
    dates = pd.bdate_range("2025-01-02", periods=83)
    def run(high):
        frames = {code: pd.DataFrame({"date": dates, "open": price, "high": price * 1.005,
                                      "low": price * .995, "close": price, "volume": 100_000,
                                      "test_code": code})
                  for code, price in (("A", 100.), ("B", 10.), ("C", 10.))}
        frames["A"].loc[81, "high"] = high
        frames["B"].loc[65:, ["open", "high", "low", "close"]] = [12.1, 12.15, 12.05, 12.1]
        def signal(data, index, *args, **kwargs):
            code = data.iloc[index]["test_code"]
            target = 60 if code == "B" else 80 if code == "C" else -1
            if index == target:
                return {"rank": 4, "trigger": 10., "stop": 9.5, "order_type": "close",
                        "known_volume_ratio": 1., "signal_type": "consolidation_breakout", "reason": "fixture"}
            return None
        monkeypatch.setattr(engine, "_right_signal", signal)
        monkeypatch.setattr(engine, "_semantic_right_entry_gate", lambda data, index, candidate, signal, **kw:
                            (True, {**signal, "semantic_entry_ok": True, "entry_target_price": 14.,
                                    "entry_risk_pct": 5., "entry_reward_risk": 8.}, "ok"))
        monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **kw: {})
        monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda *a: set())
        left = {"code": "A", "candidate_source": "value_model", "value_line": 100.,
                "quality_score": 90, "earnings_yoy": .3, "mktcap": 150, "price_to_value": 1.,
                "industry": "汽车电子电气系统", "value_industry_allowed": True,
                "value_industry_allowlist_match": "汽车电子电气系统", "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION}
        right = [{"code": code, "candidate_score": 99, "allow_left": False, "allow_right": True}
                 for code in ("B", "C")]
        return engine.run_portfolio_backtest(frames, {str(dates[59].date()): [left, *right]}, {},
                                            requested_start=str(dates[59].date()), end_date=str(dates[-1].date()),
                                            right_entry_size_multiplier=2.)
    flat, selling = run(100.5), run(106.)
    date = str(dates[81].date())
    def opening_buy(result):
        return [(e["execution_price"], e["execution_quantity"]) for e in result["trade_ledger"]
                if e["date"] == date and e["code"] == "C" and e["execution_quantity"] > 0]
    assert opening_buy(flat)
    assert opening_buy(flat) == opening_buy(selling)
    sells = [e for e in selling["trade_ledger"] if e["date"] == date and e["code"] == "A" and e["execution_quantity"] < 0]
    assert sells and all(e["execution_phase"] == "intraday" for e in sells)
