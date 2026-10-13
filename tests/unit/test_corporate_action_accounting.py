from copy import deepcopy

import pandas as pd
import pytest

from stock_research.backtesting.corporate_actions import CorporateActionBook
from stock_research.backtesting.engine import PositionState, _corporate_action_adjusted_quantities


def fixture(*, bonus=.05, cash=.1, pay="2025-01-08", listing="2025-01-09"):
    lot = {"batch": "R1", "date": pd.Timestamp("2025-01-02"), "quantity": 100.,
           "cost": 10., "size": .1, "entry_fee_cash": 5.,
           "initial_risk_per_share_raw": 1., "initial_stop_raw_price": 9.}
    states = {"A": PositionState(right=[lot])}
    book = CorporateActionBook({"A": {pd.Timestamp("2025-01-06"): [
        {"record_date": "2025-01-03", "pay_date": pay, "div_listdate": listing,
         "stk_div": bonus, "cash_div": cash}]}}, "2025-01-02")
    book.register(pd.Timestamp("2025-01-03"), states)
    return book, states, lot


def advance(book, states, date):
    return book.advance(pd.Timestamp(date), states, _corporate_action_adjusted_quantities)


def test_small_explicit_bonus_preserves_cost_fees_and_total_risk_without_price_factors():
    book, states, lot = fixture()
    events, cash = advance(book, states, "2025-01-06")
    assert cash == 0
    assert book.receivable == 10
    assert lot["quantity"] == 100
    assert book.lots("A")[0]["quantity"] == 5
    lots = states["A"].right + book.lots("A")
    assert sum(p["quantity"] * p["cost"] for p in lots) == pytest.approx(1000)
    assert sum(p["size"] for p in lots) == pytest.approx(.1)
    assert sum(p["entry_fee_cash"] for p in lots) == pytest.approx(5)
    assert sum(p["quantity"] * p["initial_risk_per_share_raw"] for p in lots) == pytest.approx(100)
    assert any(e.get("corporate_action_quantity_delta") == 5 for e in events)


def test_selling_original_position_does_not_cancel_registered_cash_or_bonus():
    book, states, lot = fixture()
    advance(book, states, "2025-01-06")
    states["A"].right.remove(lot)
    events, cash = advance(book, states, "2025-01-08")
    assert cash == 10
    assert book.receivable == 0
    assert states["A"].right == []
    assert book.lots("A")[0]["quantity"] == 5
    events, cash = advance(book, states, "2025-01-09")
    assert cash == 0
    assert states["A"].right[0]["quantity"] == 5
    assert states["A"].right[0]["sellable_from"] == pd.Timestamp("2025-01-09")
    assert book.lots() == []
    assert advance(book, states, "2025-01-10") == ([], 0.)


def test_missing_pay_and_listing_dates_do_not_make_rights_spendable_or_sellable():
    book, states, lot = fixture(pay=None, listing=None)
    events, cash = advance(book, states, "2025-02-01")
    assert cash == 0
    assert book.receivable == 10
    assert book.lots("A")[0]["quantity"] == 5
    assert states["A"].right == [lot]
    assert {x["reason"] for x in book.limitations} == {
        "pay_date_missing_or_before_ex_date", "share_listing_date_missing_or_before_ex_date"}


def test_large_cash_dividend_never_creates_shares_or_changes_cost():
    book, states, lot = fixture(bonus=0, cash=5)
    before = deepcopy(lot)
    advance(book, states, "2025-01-08")
    for key in ("quantity", "cost", "size", "entry_fee_cash", "initial_risk_per_share_raw"):
        assert lot[key] == before[key]
    assert lot["initial_stop_raw_price"] == 4.
    assert not book.pending
    assert book.receivable == 0


def test_new_buyer_after_record_date_is_not_entitled():
    book, states, lot = fixture()
    states["A"].right.append({**lot, "batch": "R2", "quantity": 200.})
    events, cash = advance(book, states, "2025-01-08")
    assert cash == 10
    assert sum(p["quantity"] for p in book.lots()) == 5


def test_duplicate_vendor_rows_do_not_double_dividends():
    actions = {"A": {pd.Timestamp("2025-01-06"): [
        {"record_date": "2025-01-03", "pay_date": "2025-01-08", "cash_div": .1}]*2}}
    book = CorporateActionBook(actions, "2025-01-02")
    _, states, _ = fixture(bonus=0)
    book.register(pd.Timestamp("2025-01-03"), states)
    _, cash = advance(book, states, "2025-01-08")
    assert cash == 10


def test_cash_and_bonus_rebase_risk_prices_without_changing_total_initial_risk():
    book, states, lot = fixture(bonus=.5, cash=1.)
    lot["initial_entry_raw_price"] = 10.
    lot["initial_target_raw_price"] = 13.
    # Capture a new record-date snapshot after adding reference prices.
    book.actions[0]["registered"] = None
    book.register(pd.Timestamp("2025-01-03"), states)
    advance(book, states, "2025-01-06")
    lots = states["A"].right + book.lots("A")
    assert all(p["initial_stop_raw_price"] == pytest.approx(8 / 1.5) for p in lots)
    assert all(p["initial_entry_raw_price"] == pytest.approx(9 / 1.5) for p in lots)
    assert sum(p["quantity"] * p["initial_risk_per_share_raw"] for p in lots) == pytest.approx(100.)
    assert sum(p["quantity"] * p["cost"] for p in lots) == pytest.approx(1000.)


def test_conflicting_revisions_do_not_credit_two_dividends():
    actions = {"A": {pd.Timestamp("2025-01-06"): [
        {"record_date": "2025-01-03", "pay_date": "2025-01-08", "cash_div": .22},
        {"record_date": "2025-01-03", "pay_date": "2025-01-08", "cash_div": .81}]}}
    book = CorporateActionBook(actions, "2025-01-02")
    _, states, _ = fixture(bonus=0)
    book.register(pd.Timestamp("2025-01-03"), states)
    _, cash = advance(book, states, "2025-01-08")
    assert cash == 0
    assert book.receivable == 0
    assert any(item["reason"] == "conflicting_action_revisions" for item in book.limitations)


def test_enriched_settlement_date_does_not_duplicate_the_registered_right():
    actions = {"A": {pd.Timestamp("2025-01-06"): [
        {"record_date": "2025-01-03", "cash_div": .1},
        {"record_date": "2025-01-03", "pay_date": "2025-01-08", "cash_div": .1}]}}
    book = CorporateActionBook(actions, "2025-01-02")
    _, states, _ = fixture(bonus=0)
    book.register(pd.Timestamp("2025-01-03"), states)
    _, cash = advance(book, states, "2025-01-08")
    assert cash == 10
    assert book.receivable == 0
    assert not book.limitations
