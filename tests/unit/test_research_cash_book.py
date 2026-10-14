"""Cash instruments lock actual principal and use dated executable source contracts."""
from copy import deepcopy
import hashlib
import inspect
import json

import pandas as pd
import pytest

from stock_research.backtesting import engine
from stock_research.research.cash_book import CashBook, audit_cash_research
from stock_research.research.portfolio_replay import run_portfolio_research


def contract(**changes):
    return {"schema_version": 1, "instrument": "sh.204001", "instrument_type": "exchange_reverse_repo",
            "currency": "CNY", "timezone": "Asia/Shanghai", "execution_time": "15:05:00",
            "allocation_fraction": 1., "reserve_cash": 0., **changes}


def source(tmp_path, **changes):
    quote = {"quote_id": "fixture-repo-20240105", "instrument": "sh.204001",
             "instrument_type": "exchange_reverse_repo", "currency": "CNY",
             "source": "isolated executable quote fixture; not actual market evidence",
             "quote_kind": "executable_lend_rate", "trading_date": "2024-01-05",
             "observed_at": "2024-01-05T15:04:59+08:00", "available_at": "2024-01-05T15:04:59+08:00",
             "valid_until": "2024-01-05T15:05:01+08:00", "annual_rate": .0365, "day_count": 365,
             "accrual_start_date": "2024-01-05", "accrual_end_date": "2024-01-08",
             "principal_release_at": "2024-01-08T09:00:00+08:00",
             "interest_release_at": "2024-01-08T09:30:00+08:00", "principal_increment": 1000.,
             "minimum_principal": 1000., "maximum_principal": 2_000_000., "entry_fee_rate": 0.,
             "minimum_entry_fee": 1., "settlement_fee": 2., "interest_tax_rate": .1, **changes}
    path = tmp_path / f"{quote['quote_id']}.json"
    path.write_text(json.dumps(quote, sort_keys=True), encoding="utf-8")
    return {"source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def book_result(book, initial, cash):
    return {"initial_capital": initial, "final_cash": cash, "events": [], "equity_curve": [], **book.report()}


def test_cash_contract_locks_principal_and_records_fees_interest_and_release_separately(tmp_path):
    book = CashBook(contract(allocation_fraction=.5, reserve_cash=10_000.), [source(tmp_path)])
    cash = book.invest("2024-01-05", 100_000.)
    assert cash == 55_999. and book.principal == 44_000.
    assert [row["action"] for row in book.ledger] == ["principal_lock", "entry_fee"]
    assert book.report()["cash_gross_interest_amount"] == 0
    cash = book.advance("2024-01-08T08:59:59+08:00", cash)
    assert cash == 55_999. and book.principal == 44_000.
    cash = book.advance("2024-01-08T09:00:00+08:00", cash)
    assert cash == 99_997. and book.principal == 0
    assert book.report()["cash_gross_interest_amount"] == 0
    cash = book.advance("2024-01-08T09:30:00+08:00", cash)
    assert cash == pytest.approx(100_008.88)
    assert [row["action"] for row in book.ledger] == ["principal_lock", "entry_fee", "principal_release", "settlement_fee", "interest_payment"]
    assert book.report()["cash_gross_interest_amount"] == 13.2
    assert audit_cash_research(book_result(book, 100_000., cash))["status"] == "passed"
    assert book.advance("2024-01-09T09:30:00+08:00", cash) == cash
    assert len(book.ledger) == 5


@pytest.mark.parametrize("changes", [
    {"available_at": "2024-01-05T16:30:00+08:00"},
    {"observed_at": "2024-01-05T15:05:01+08:00", "available_at": "2024-01-05T15:05:01+08:00"},
    {"valid_until": "2024-01-05T15:04:59+08:00"},
])
def test_late_or_expired_quotes_cannot_create_same_day_cash_fills(tmp_path, changes):
    book = CashBook(contract(), [source(tmp_path, **changes)])
    assert book.invest("2024-01-05", 1_000_000.) == 1_000_000.
    assert book.ledger == [] and book.principal == 0
    assert book.report()["cash_income_status"] == "data_limited"


def test_source_hash_exact_fixed_instrument_and_unique_quote_are_required(tmp_path):
    receipt = source(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        CashBook(contract(), [{**receipt, "source_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="preselected"):
        CashBook(contract(instrument="sz.131810"), [receipt])
    with pytest.raises(ValueError, match="Duplicate"):
        CashBook(contract(), [receipt, receipt])
    with pytest.raises(ValueError, match="timezone"):
        CashBook(contract(), [source(tmp_path, available_at="2024-01-05T15:04:59")])
    with pytest.raises(ValueError, match="executable"):
        CashBook(contract(), [source(tmp_path, quote_kind="daily_closing_yield")])
    with pytest.raises(ValueError, match="fixed time"):
        CashBook(contract(execution_time="16:30:00"), [])


def test_incomplete_or_ambiguous_settlement_contract_fails_before_booking(tmp_path):
    receipt = source(tmp_path)
    from pathlib import Path
    path = Path(receipt["source_path"])
    quote = json.loads(path.read_text())
    quote.pop("principal_release_at")
    path.write_text(json.dumps(quote))
    with pytest.raises(ValueError, match="complete execution"):
        CashBook(contract(), [{**receipt, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
    with pytest.raises(ValueError, match="inconsistent"):
        CashBook(contract(), [source(tmp_path, principal_release_at="2024-01-05T15:04:00+08:00")])


@pytest.mark.parametrize("release", ["2024-01-07T23:59:59+08:00", "2024-01-07T15:59:59Z"])
def test_interest_cannot_be_paid_before_source_accrual_end(tmp_path, release):
    with pytest.raises(ValueError, match="inconsistent"):
        CashBook(contract(), [source(tmp_path, interest_release_at=release)])


def _engine_inputs(monkeypatch, *, end_date="2024-01-09"):
    dates = pd.bdate_range(end="2024-01-09", periods=100)
    frames = {code: pd.DataFrame({"date": dates, "open": 10., "high": 10.1, "low": 9.9,
                                  "close": 10., "volume": 1_000_000, "test_code": code})
              for code in ("sz.000001", "sz.000002")}
    signals = {"sz.000001": pd.Timestamp("2024-01-05"), "sz.000002": pd.Timestamp("2024-01-08")}
    def signal(data, index, plan=None, **kwargs):
        if data.iloc[index]["date"] != signals[data.iloc[index]["test_code"]]:
            return None
        return {"rank": 4, "trigger": 10., "stop": 9.5, "order_type": "close", "reason": "cash timing fixture",
                "known_volume_ratio": 1., "signal_type": "consolidation_breakout"}
    def gate(data, index, candidate, signal, **kwargs):
        return True, {**signal, "semantic_entry_ok": True, "entry_risk_pct": 5.,
                      "entry_reward_risk": 8., "entry_target_price": 14.}, "ok"
    monkeypatch.setattr(engine, "_right_signal", signal)
    monkeypatch.setattr(engine, "_semantic_right_entry_gate", gate)
    monkeypatch.setattr(engine, "position_exit_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(engine, "_active_profit_trigger_ids", lambda *a: set())
    candidates = [{"code": code, "candidate_score": 99, "allow_right": True, "allow_left": False,
                   "signal_eligible": True, "selected_for_trading": True} for code in frames]
    return dict(price_frames=frames, candidate_snapshots={"2024-01-04": candidates},
                formula_phases={"2024-01-04": {"phase": "active", "window_up_streak": 5, "window_down_streak": 0}},
                requested_start="2024-01-05", end_date=end_date, close_confirmed_execution="next_open",
                signals_effective_next_day=True, commission_rate=0, minimum_commission=0,
                estimated_slippage_rate=0, sell_stamp_duty_rate=0)


def test_engine_cannot_spend_locked_principal_or_future_afternoon_settlement(tmp_path, monkeypatch):
    inputs = _engine_inputs(monkeypatch)
    receipt = source(tmp_path, principal_release_at="2024-01-08T16:00:00+08:00",
                     interest_release_at="2024-01-08T16:00:00+08:00", interest_tax_rate=0.)
    result = run_portfolio_research(research_policy={"sizing_basis": "visible_nav"},
                                    cash_contract=contract(), cash_quotes=[receipt], **inputs)
    buys = [row for row in result["events"] if row["trade_side"] == "买入"]
    assert [(row["code"], row["date"], row["trade_amount"]) for row in buys] == [("sz.000002", "2024-01-09", 300_000.)]
    first = result["equity_curve"][0]
    assert first["cash"] == 999. and first["cash_instrument_principal"] == 999_000.
    assert first["equity"] == .999999 and first["stock_market_value"] == 0
    assert first["market_exposure_pct"] == 0
    assert buys[0]["research_observed_nav"] == pytest.approx(1_000_296.7)
    assert result["cash_gross_interest_amount"] == 299.7
    assert result["cash_income_amount"] == pytest.approx(296.7)
    assert result["final_cash"] == pytest.approx(700_296.7)
    assert result["cash_accounting_audit"]["status"] == "passed"


def test_release_before_open_can_finance_stock_and_stock_fees_reconcile(tmp_path, monkeypatch):
    inputs = _engine_inputs(monkeypatch)
    inputs.update(commission_rate=.0003, minimum_commission=5., estimated_slippage_rate=.001)
    result = run_portfolio_research(research_policy={}, cash_contract=contract(), cash_quotes=[source(tmp_path)], **inputs)
    buys = [row for row in result["events"] if row["trade_side"] == "买入"]
    assert buys and buys[0]["code"] == "sz.000001" and buys[0]["date"] == "2024-01-08"
    assert result["cash_accounting_audit"]["status"] == "passed"
    tampered = deepcopy(result)
    tampered["cash_instrument_ledger"][2]["timestamp"] = "2024-01-05T16:00:00+08:00"
    assert "cash_source_or_settlement_time" in audit_cash_research(tampered)["errors"]


def test_weekend_settlement_adds_nontrading_accounting_date_without_inventing_stock_fill(tmp_path, monkeypatch):
    inputs = _engine_inputs(monkeypatch)
    receipt = source(tmp_path, principal_release_at="2024-01-07T10:00:00+08:00",
                     interest_release_at="2024-01-07T10:00:00+08:00", accrual_end_date="2024-01-07")
    result = run_portfolio_research(research_policy={}, cash_contract=contract(), cash_quotes=[receipt], **inputs)
    sunday = next(point for point in result["equity_curve"] if point["date"] == "2024-01-07")
    assert sunday["is_trading_day"] is False and sunday["cash_instrument_principal"] == 0
    assert not any(row["date"] == "2024-01-07" for row in result["events"])
    assert result["cash_accounting_audit"]["status"] == "passed"


def test_end_before_maturity_keeps_principal_asset_and_zero_unpaid_interest(tmp_path, monkeypatch):
    inputs = _engine_inputs(monkeypatch, end_date="2024-01-05")
    result = run_portfolio_research(research_policy={}, cash_contract=contract(), cash_quotes=[source(tmp_path)], **inputs)
    assert result["cash_locked_principal"] == 999_000. and result["final_cash"] == 999.
    assert result["cash_gross_interest_amount"] == 0 and result["cash_income_amount"] == -1.
    assert result["equity_curve"][-1]["equity"] == .999999
    assert result["cash_accounting_audit"]["status"] == "passed"


def test_no_quotes_means_data_limited_no_interest_and_default_api_unchanged(monkeypatch):
    inputs = _engine_inputs(monkeypatch)
    production = engine.run_portfolio_backtest(**inputs)
    inactive = run_portfolio_research(research_policy={}, **inputs)
    assert all(inactive[key] == value for key, value in production.items())
    assert "cash_contract" not in inspect.signature(engine.run_portfolio_backtest).parameters
    assert "research_cash_book" not in inspect.signature(engine.run_portfolio_backtest).parameters
    result = run_portfolio_research(research_policy={}, cash_contract=contract(), cash_quotes=[], **inputs)
    assert result["cash_income_status"] == "data_limited"
    assert result["cash_instrument_ledger"] == [] and result["cash_gross_interest_amount"] == 0
    assert result["events"] == production["events"] and result["final_cash"] == production["final_cash"]
    assert result["final_return_pct"] == production["final_return_pct"]
    assert result["cash_accounting_audit"]["status"] == "passed"
    with pytest.raises(ValueError, match="preselected"):
        run_portfolio_research(research_policy={}, cash_quotes=[], **inputs)


def test_source_modification_and_independent_ledger_arithmetic_are_detected(tmp_path):
    receipt = source(tmp_path)
    book = CashBook(contract(), [receipt])
    cash = book.invest("2024-01-05", 100_000.)
    cash = book.advance("2024-01-08T10:00:00+08:00", cash)
    result = book_result(book, 100_000., cash)
    tampered = deepcopy(result)
    tampered["cash_instrument_ledger"][-1]["gross_interest"] += 100.
    assert "gross_interest" in audit_cash_research(tampered)["errors"]
    from pathlib import Path
    Path(receipt["source_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        book.verify_sources()
    assert "cash_quote_source_binding" in audit_cash_research(result)["errors"]


@pytest.mark.parametrize("phase,release,processed", [
    ("open", "16:00:00", "16:00:00"),
    ("intraday", "11:00:00", "11:00:00"),
    ("close", "16:00:00", "16:00:00"),
])
def test_independent_audit_rejects_later_settlement_funding_earlier_stock_phase(tmp_path, phase, release, processed):
    receipt = source(tmp_path, principal_release_at=f"2024-01-08T{release}+08:00",
                     interest_release_at=f"2024-01-08T{release}+08:00")
    book = CashBook(contract(), [receipt])
    cash = book.invest("2024-01-05", 100_000.)
    cash = book.advance(f"2024-01-08T{processed}+08:00", cash)
    result = book_result(book, 100_000., cash - 50_000.)
    # All balances and source contracts reconcile, but the forged checkpoint
    # places this earlier stock purchase after the later cash settlement.
    result["events"] = [{"date": "2024-01-08", "execution_phase": phase,
                         "cash_before": cash, "cash_after": cash - 50_000.,
                         "cash_change_amount_raw": -50_000.}]
    audit = audit_cash_research(result)
    assert audit["status"] == "failed"
    assert set(audit["errors"]) == {"cash_stock_checkpoint_chronology"}


def test_midday_settlement_processed_at_close_can_fund_close_stock_phase(tmp_path):
    receipt = source(tmp_path, principal_release_at="2024-01-08T11:00:00+08:00",
                     interest_release_at="2024-01-08T11:00:00+08:00")
    book = CashBook(contract(), [receipt])
    cash = book.invest("2024-01-05", 100_000.)
    cash = book.advance("2024-01-08T15:00:00+08:00", cash)
    result = book_result(book, 100_000., cash - 50_000.)
    result["events"] = [{"date": "2024-01-08", "execution_phase": "close",
                         "cash_before": cash, "cash_after": cash - 50_000.,
                         "cash_change_amount_raw": -50_000.}]
    assert audit_cash_research(result)["status"] == "passed"


def test_cash_audit_rejects_processing_time_moving_backwards(tmp_path):
    book = CashBook(contract(), [source(tmp_path)])
    cash = book.invest("2024-01-05", 100_000.)
    cash = book.advance("2024-01-08T10:00:00+08:00", cash)
    result = book_result(book, 100_000., cash)
    result["cash_instrument_ledger"][-1]["processed_at"] = "2024-01-08T09:30:00+08:00"
    assert audit_cash_research(result)["errors"] == ["cash_processing_chronology"]


def test_quoted_accrual_days_are_independent_of_principal_availability_and_cash_can_reinvest(tmp_path):
    friday = source(tmp_path, accrual_start_date="2024-01-08", accrual_end_date="2024-01-09",
                    interest_release_at="2024-01-09T09:00:00+08:00", interest_tax_rate=0.)
    monday = source(tmp_path, quote_id="fixture-repo-20240108", trading_date="2024-01-08",
                    observed_at="2024-01-08T15:04:59+08:00", available_at="2024-01-08T15:04:59+08:00",
                    valid_until="2024-01-08T15:05:01+08:00", accrual_start_date="2024-01-09",
                    accrual_end_date="2024-01-10", principal_release_at="2024-01-09T09:00:00+08:00",
                    interest_release_at="2024-01-10T09:00:00+08:00", interest_tax_rate=0.)
    book = CashBook(contract(), [friday, monday])
    cash = book.invest("2024-01-05", 100_000.)
    assert book.positions[0]["gross_interest"] == 9.9  # one source-defined accrual day, not the weekend lock length
    cash = book.advance("2024-01-08T09:30:00+08:00", cash)
    assert cash == pytest.approx(99_997.)
    assert book.report()["cash_gross_interest_amount"] == 0
    cash = book.invest("2024-01-08", cash)
    assert len(book.positions) == 2 and book.principal == 99_000.
    assert cash == pytest.approx(996.)
    cash = book.advance("2024-01-09T09:30:00+08:00", cash)
    assert cash == pytest.approx(100_003.9)
    assert book.report()["cash_gross_interest_amount"] == 9.9
    cash = book.advance("2024-01-10T09:30:00+08:00", cash)
    assert cash == pytest.approx(100_013.8)
    assert audit_cash_research(book_result(book, 100_000., cash))["status"] == "passed"
