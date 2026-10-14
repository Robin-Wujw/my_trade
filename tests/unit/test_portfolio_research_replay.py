"""Research controls are tested on actual fills and visible information cutoffs."""
import inspect

import pandas as pd
import pytest

from stock_research.backtesting import engine
from stock_research.research.portfolio_policy import gap_tail_estimate
from stock_research.research.portfolio_replay import freeze_research_policy, run_portfolio_research
from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting


def _fixture(monkeypatch, *, periods=320, first_signal=255, second_signal=315,
             held_open=12., execution_close=12., down_streak=0, signal_type="consolidation_breakout",
             diversified=False):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    frames = {code: pd.DataFrame({"date": dates, "open": 10., "high": 10.1, "low": 9.9,
                                  "close": 10., "volume": 1_000_000, "test_code": code})
              for code in ("sz.000001", "sz.000002")}
    frames["sz.000001"].loc[first_signal + 5:, ["open", "high", "low", "close"]] = [12., 12.1, 11.9, 12.]
    if second_signal + 1 < periods:
        frames["sz.000001"].loc[second_signal + 1, ["open", "high", "low", "close"]] = [
            held_open, max(held_open, execution_close) + .1,
            min(held_open, execution_close) - .1, execution_close]

    def signal(data, index, plan=None, **kwargs):
        code = data.iloc[index]["test_code"]
        day = first_signal if code == "sz.000001" else second_signal
        if data.iloc[index]["date"] != dates[day]:
            return None
        return {"rank": 4, "trigger": 10., "stop": 9.5, "order_type": "close",
                "reason": "visible-policy fixture", "known_volume_ratio": 1., "signal_type": signal_type}

    observed_streaks = []

    def gate(data, index, candidate, signal, **kwargs):
        observed_streaks.append(kwargs.get("current_down_streak"))
        return True, {**signal, "semantic_entry_ok": True, "entry_risk_pct": 5.,
                      "entry_reward_risk": 8., "entry_target_price": 14.}, "ok"

    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", gate)
    monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda *a: set())
    candidates = [{"code": code, "candidate_score": 99, "allow_right": True, "allow_left": False,
                   "signal_eligible": True, "selected_for_trading": True,
                   "selection_profile": "quantsplaybook_factor_only" if diversified or signal_type == "uptrend_support_pullback" else ""}
                  for code in frames]
    start = str(dates[max(0, first_signal - 65)].date())
    kwargs = dict(price_frames=frames, candidate_snapshots={start: candidates},
                  formula_phases={start: {"phase": "active", "window_up_streak": 5,
                                         "window_down_streak": down_streak, "window_unique_count": 188}},
                  requested_start=start, end_date=str(dates[-1].date()),
                  close_confirmed_execution="next_open", signals_effective_next_day=True,
                  commission_rate=0, minimum_commission=0, estimated_slippage_rate=0,
                  sell_stamp_duty_rate=0, enable_diversified_factor_sleeve=diversified)
    return kwargs, dates, observed_streaks


def _buys(result, code=None):
    return [event for event in result["events"] if event["trade_side"] == "买入"
            and (code is None or event["code"] == code)]


def test_policy_off_preserves_production_results_and_public_signature(monkeypatch):
    kwargs, _, _ = _fixture(monkeypatch)
    production = engine.run_portfolio_backtest(**kwargs)
    research = run_portfolio_research(research_policy={}, **kwargs)
    assert _buys(production)
    for key, value in production.items():
        assert research[key] == value, key
    assert "research_policy" not in inspect.signature(engine.run_portfolio_backtest).parameters
    assert "research_risk_decisions" not in research
    assert research["cash_income_status"] == "not_applied_no_identified_source_quote"


def test_explicit_empty_structure_evidence_blocks_actual_research_fills(monkeypatch):
    from stock_research.backtesting import signals
    kwargs, _, _ = _fixture(monkeypatch)
    baseline = engine.run_portfolio_backtest(**kwargs)
    assert _buys(baseline)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", signals._semantic_right_entry_gate)
    result = run_portfolio_research(research_policy={}, structure_label_manifests=[], **kwargs)
    assert not _buys(result)
    assert any(row["reason"] == "structure_expectancy_data_limited" for row in result["entry_blocks"])
    assert result["structure_expectancy_decisions"]
    assert all(row["mean_net_r"] is None for row in result["structure_expectancy_decisions"])


def test_nav_sizing_uses_open_marks_and_keeps_initial_capital_accounting(monkeypatch):
    kwargs, days, _ = _fixture(monkeypatch, execution_close=12.)
    policy = {"sizing_basis": "visible_nav"}
    result = run_portfolio_research(research_policy=policy, **kwargs)
    second = _buys(result, "sz.000002")[0]
    assert second["date"] == str(days[316].date())
    assert second["research_observed_nav"] == pytest.approx(1_060_000)
    assert second["trade_amount"] == pytest.approx(318_000)
    assert second["position_change_pct"] == pytest.approx(31.8)
    assert second["execution_quantity"] == 31800
    assert second["execution_phase"] == "open"
    point = next(point for point in result["equity_curve"] if point["date"] == second["date"])
    assert point["gross_exposure_basis"] == "current_nav_market_value"
    assert point["cost_exposure_pct"] == pytest.approx(61.8)
    assert point["nav_market_exposure_pct"] == pytest.approx(678000 / 1060000 * 100)
    final = next(row for row in result["final_positions"] if row["code"] == "sz.000001")
    assert final["position_pct_basis"] == "current_nav_market_value"
    assert final["cost_position_pct"] == 30.
    assert final["nav_market_value"] == 360000
    kwargs, _, _ = _fixture(monkeypatch, execution_close=60.)
    future_close = run_portfolio_research(research_policy=policy, **kwargs)
    changed_second = _buys(future_close, "sz.000002")[0]
    assert changed_second["execution_quantity"] == second["execution_quantity"]
    assert changed_second["research_observed_nav"] == second["research_observed_nav"]


def test_formula_ablation_disables_down_streak_size_and_keeps_real_metadata(monkeypatch):
    kwargs, _, seen = _fixture(monkeypatch, down_streak=4)
    baseline = engine.run_portfolio_backtest(**kwargs)
    result = run_portfolio_research(research_policy={"formula33_enabled": False}, **kwargs)
    buy = _buys(result)[0]
    assert _buys(baseline)[0]["trade_amount"] == 200_000
    assert buy["trade_amount"] == 300_000
    assert buy["frozen_formula_down_streak"] == 4
    assert buy["frozen_formula_up_streak"] == 5
    assert 0 in seen and 4 in seen


def test_formula_ablation_restores_pullback_without_replacing_observation(monkeypatch):
    kwargs, _, _ = _fixture(monkeypatch, down_streak=4, signal_type="uptrend_support_pullback")
    baseline = engine.run_portfolio_backtest(**kwargs)
    result = run_portfolio_research(research_policy={"formula33_enabled": False}, **kwargs)
    assert not _buys(baseline)
    assert _buys(result)
    assert _buys(result)[0]["frozen_formula_down_streak"] == 4


def test_gap_tail_uses_250_completed_bars_and_rejects_missing_history(monkeypatch):
    policy = {"sizing_basis": "visible_nav", "gap_tail_enabled": True}
    kwargs, _, _ = _fixture(monkeypatch)
    result = run_portfolio_research(research_policy=policy, **kwargs)
    buy = _buys(result)[0]
    assert buy["trade_amount"] == 160_000
    assert buy["research_gap_observations"] == 250
    assert buy["research_gap_loss_quantile"] == 0
    assert buy["research_equity_risk_budget"] == 8000
    kwargs, _, _ = _fixture(monkeypatch, periods=120, first_signal=65, second_signal=115)
    result = run_portfolio_research(research_policy=policy, **kwargs)
    assert not _buys(result)
    assert any(row["reason"] == "insufficient_visible_gap_history" for row in result["research_risk_decisions"])


def test_visible_open_drawdown_blocks_new_risk_before_future_close(monkeypatch):
    kwargs, _, _ = _fixture(monkeypatch, held_open=1., execution_close=12., diversified=True)
    result = run_portfolio_research(research_policy={"sizing_basis": "visible_nav", "maximum_drawdown": .2}, **kwargs)
    assert _buys(result, "sz.000001")
    assert not _buys(result, "sz.000002")
    assert any(row["reason"] == "drawdown_new_risk_block" for row in result["research_risk_decisions"])


def test_portfolio_volatility_warmup_is_explicit_and_has_actual_fills(monkeypatch):
    kwargs, _, _ = _fixture(monkeypatch)
    policy = {"sizing_basis": "visible_nav", "annual_volatility_target": .15}
    result = run_portfolio_research(research_policy=policy, **kwargs)
    assert _buys(result)
    assert _buys(result)[0]["research_portfolio_scale"] == 1
    kwargs, days, _ = _fixture(monkeypatch)
    kwargs["requested_start"] = str(days[254].date())
    short = run_portfolio_research(research_policy=policy, **kwargs)
    assert any(row["reason"] == "insufficient_visible_equity_history" for row in short["research_risk_decisions"])


def test_policy_validation_rejects_ambiguous_values_and_hashes_canonical_json():
    first, canonical, digest = freeze_research_policy({"sizing_basis": "visible_nav", "formula33_enabled": False})
    assert digest == freeze_research_policy({"formula33_enabled": False, "sizing_basis": "visible_nav"})[2]
    assert first.active and '"schema_version":1' in canonical
    for value in (None, {"enabled": True}, {"formula33_enabled": "false"}, {"risk_budget": float("nan")},
                  {"gap_tail_enabled": True}, {"volatility_window": 60.5}, {"maximum_drawdown": 1}):
        with pytest.raises(ValueError):
            freeze_research_policy(value)
    with pytest.raises(ValueError, match="cutoff"):
        gap_tail_estimate(pd.DataFrame(), None)


def test_active_policy_retains_independent_cash_share_fee_audit(monkeypatch):
    kwargs, days, _ = _fixture(monkeypatch)
    kwargs.update(commission_rate=.0003, minimum_commission=5., estimated_slippage_rate=.001,
                  sell_stamp_duty_rate=.0005, security_end_dates={"sz.000002": str(days[318].date())})
    result = run_portfolio_research(research_policy={"sizing_basis": "visible_nav", "gap_tail_enabled": True}, **kwargs)
    assert _buys(result)
    assert any(event["trade_side"] == "卖出" for event in result["events"])
    bound = inspect.signature(engine.run_portfolio_backtest).bind(**kwargs)
    bound.apply_defaults()
    audit = audit_portfolio_accounting(result, kwargs["price_frames"], {}, bound.arguments)
    assert audit["status"] == "passed", audit
    assert all(buy["quantity"] % 100 == 0 and buy["cash_after"] >= 0 for buy in _buys(result))


def test_formula_sizing_ablation_preserves_observed_down_streak_exit_horizon(monkeypatch):
    kwargs, days, _ = _fixture(monkeypatch, down_streak=4)
    for frame in kwargs["price_frames"].values():
        frame[["open", "high", "low", "close"]] = [10., 10.1, 9.9, 10.]
    seen = []
    def timed_exit(history, cost, entry_date, **settings):
        limit = settings["time_limit_days"]
        if limit == 9999:
            return {}
        seen.append(limit)
        observed = int(history.date.ge(entry_date).sum())
        return {"entry_time_stop": observed >= limit}
    monkeypatch.setattr(engine, "position_exit_snapshot", timed_exit)
    baseline = engine.run_portfolio_backtest(**kwargs)
    result = run_portfolio_research(research_policy={"formula33_enabled": False}, **kwargs)
    def exits(value):
        return [(row["code"], row["date"], row["execution_price"], row["reason"])
                for row in value["events"] if row["trade_side"] == "卖出"]
    assert exits(result) and exits(result) == exits(baseline)
    assert set(seen) == {8}


def test_nav_minimum_trimmed_entry_uses_current_nav_when_cash_is_insufficient(monkeypatch):
    codes = ["sz.000001", "sz.000002", "sz.000003", "sz.000004"]
    dates = pd.bdate_range("2023-01-02", periods=140)
    signal_dates = dict(zip(codes, [70, 85, 100, 130]))
    frames = {}
    for code in codes:
        frame = pd.DataFrame({"date": dates, "open": 10., "high": 10.1, "low": 9.9,
                              "close": 10., "volume": 1_000_000, "test_code": code})
        if code != codes[-1]:
            frame.loc[signal_dates[code] + 5:, ["open", "high", "low", "close"]] = [11., 11.1, 10.9, 11.]
            frame.loc[120:, ["open", "high", "low", "close"]] = [20., 20.1, 19.9, 20.]
        frames[code] = frame

    def signal(data, index, plan=None, **kwargs):
        code = data.iloc[index]["test_code"]
        if data.iloc[index]["date"] != dates[signal_dates[code]]:
            return None
        return {"rank": 4, "trigger": 10., "stop": 9.5, "order_type": "close",
                "reason": "visible NAV capacity", "known_volume_ratio": 1.,
                "signal_type": "consolidation_breakout"}

    def gate(data, index, candidate, signal, **kwargs):
        return True, {**signal, "semantic_entry_ok": True, "entry_risk_pct": 5.,
                      "entry_reward_risk": 8., "entry_target_price": 14.}, "ok"

    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", gate)
    monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda *a: set())
    start = str(dates[5].date())
    candidates = [{"code": code, "candidate_score": 99, "allow_right": True, "allow_left": False,
                   "signal_eligible": True, "selected_for_trading": True} for code in codes]
    kwargs = dict(price_frames=frames, candidate_snapshots={start: candidates},
                  formula_phases={start: {"phase": "active", "window_up_streak": 5, "window_down_streak": 0}},
                  requested_start=start, end_date=str(dates[-1].date()),
                  close_confirmed_execution="next_open", signals_effective_next_day=True,
                  commission_rate=0, minimum_commission=0, estimated_slippage_rate=0,
                  sell_stamp_duty_rate=0, right_entry_size_multiplier=5 / 6)
    result = run_portfolio_research(research_policy={"sizing_basis": "visible_nav"}, **kwargs)
    assert [row["trade_amount"] for row in _buys(result)] == [250_000., 256_000., 262_000.]
    decision = next(row for row in result["research_risk_decisions"] if row["code"] == codes[-1])
    assert decision["research_observed_nav"] == 1_768_000.
    assert decision["approved_notional_before_execution"] == 442_000.
    # Cash 232k is above 20% of initial capital but below the 353.6k NAV floor.
    assert result["final_cash"] == 232_000.
    assert any(row["code"] == codes[-1] and row["reason"] == "gross_exposure_cap"
               for row in result["entry_blocks"])
    production = engine.run_portfolio_backtest(**kwargs)
    inactive = run_portfolio_research(research_policy={}, **kwargs)
    assert _buys(production, codes[-1])[0]["trade_amount"] == 250_000.
    assert all(inactive[key] == value for key, value in production.items())
