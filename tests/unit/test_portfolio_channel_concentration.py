from copy import deepcopy
import csv
import hashlib
import json

import pytest

from stock_research.reporting.portfolio_channel_concentration import (
    build_channel_contribution, render_markdown, report_from_files,
)


def allocation(channel, quantity, cost, fee=0.):
    return {"origin_account": channel, "quantity": quantity, "cost_amount": cost,
            "entry_fee_amount": fee, "origin_batch": "L1" if channel == "left" else "R1"}


def sale(code, price, allocations, fee=0.):
    return {"code": code, "date": "2026-07-20", "execution_quantity": -sum(row["quantity"] for row in allocations),
            "execution_price": price, "transaction_cost_amount_raw": fee, "origin_allocations": allocations,
            "account_mode": "right"}


def channel(report, name):
    return next(row for row in report["by_channel"] if row["origin_channel"] == name)


def test_mixed_sell_preserves_left_origin_and_splits_costs_without_double_counting_entry_fee():
    events = [sale("A", 12., [allocation("left", 100, 1000., 2.), allocation("right", 100, 1100., 3.)], fee=4.)]
    report = build_channel_contribution(events, [], initial_capital=10_000.)
    left, right = channel(report, "left"), channel(report, "right")
    assert left["realized_net_profit"] == 196.
    assert right["realized_net_profit"] == 95.
    assert left["fees_paid"] == right["fees_paid"] == 2.
    assert report["covered_realized_net_profit"] == 291.
    assert left["realized_contribution_pct_initial_capital"] == 1.96
    assert report["attribution_coverage_status"] == "complete"
    assert not report["independent_channel_effect_established"]


def test_partial_sale_and_unlisted_bonus_keep_unrealized_and_unsettled_outcomes_separate():
    sold = allocation("left", 50, 500., 1.)
    remaining = allocation("left", 50, 500., 1.)
    events = [sale("A", 12., [sold], fee=1.)]
    positions = [{"code": "A", "close": 13., "quantity": 50,
                  "batches": [{"quantity": 50, "origin_allocation": remaining, "corporate_action_pending": True}]}]
    report = build_channel_contribution(events, positions, initial_capital=10_000.)
    left = channel(report, "left")
    assert left["realized_net_profit"] == 98. and left["unrealized_net_profit"] == 149.
    assert left["closed_and_settled_realized_profit"] == 0.
    assert left["unsettled_stock_count"] == 1
    stock = left["stocks"][0]
    assert stock["outcome_status"] == "open_position" and stock["pending_listing_quantity"] == 50
    assert left["top_stock_concentration"][0]["profit"] == 98.
    markdown = render_markdown(report)
    assert "open_position" in markdown and "不是可交易反事实收益" in markdown


def test_dividend_receivable_is_not_realized_before_cash_settles():
    origin = allocation("left", 100, 1000., 2.)
    events = [sale("A", 12., [origin], fee=2.),
              {"code": "A", "origin_allocations": [origin], "dividend_receivable_change": 10.}]
    pending = build_channel_contribution(events, [], initial_capital=10_000.)
    left = channel(pending, "left")
    assert left["realized_net_profit"] == 196. and left["settled_dividends"] == 0.
    assert left["stocks"][0]["outcome_status"] == "shares_closed_pending_dividend"
    events.append({"code": "A", "origin_allocations": [origin], "corporate_action_cash_amount": 10.,
                   "dividend_receivable_change": -10.})
    settled = channel(build_channel_contribution(events, [], initial_capital=10_000.), "left")
    assert settled["realized_net_profit"] == 206. and settled["dividend_receivable"] == 0.
    assert settled["stocks"][0]["outcome_status"] == "closed_and_settled"


def test_top_stock_concentration_aggregates_repeated_tranches_and_does_not_drop_losers():
    events = [sale("A", 12., [allocation("left", 100, 1000.)]),
              sale("A", 11., [allocation("left", 100, 1000.)]),
              sale("B", 11., [allocation("left", 100, 1000.)]),
              sale("C", 8., [allocation("left", 100, 1000.)])]
    left = channel(build_channel_contribution(events, [], initial_capital=10_000.), "left")
    assert left["realized_net_profit"] == 200.
    top1, top3 = left["top_stock_concentration"]
    assert top1["codes"] == ["A"] and top1["profit"] == 300.
    assert top1["share_of_net_realized_profit"] == 1.5
    assert top1["share_of_positive_stock_profit"] == .75
    assert top1["remaining_static_realized_profit"] == -100.
    assert top3["codes"] == ["A", "B"] and top3["remaining_static_realized_profit"] == -200.
    assert top3["selected_stock_count"] == 2 and top3["counterfactual_is_tradable"] is False


def test_nonpositive_channel_net_profit_never_gets_misleading_concentration_ratio():
    events = [sale("A", 11., [allocation("right", 100, 1000.)]),
              sale("B", 8., [allocation("right", 100, 1000.)])]
    top = channel(build_channel_contribution(events, [], initial_capital=10_000.), "right")["top_stock_concentration"][0]
    assert top["share_of_net_realized_profit"] is None
    assert top["share_of_positive_stock_profit"] == 1.
    assert top["remaining_static_realized_profit"] == -200.


@pytest.mark.parametrize("change,reason", [
    ({"origin_allocations": []}, "missing_origin_allocations"),
    ({"execution_quantity": -200}, "allocated_quantity_mismatch"),
    ({"transaction_cost_amount_raw": float("inf")}, "invalid_transaction_cost"),
    ({"profit_loss_amount_raw": 9999.}, "allocated_profit_mismatch"),
])
def test_missing_or_inconsistent_sources_remain_data_limited_without_guessed_channel(change, reason):
    event = sale("A", 12., [allocation("left", 100, 1000.)])
    report = build_channel_contribution([{**event, **change}], [], initial_capital=10_000.)
    assert report["attribution_coverage_status"] == "data_limited"
    assert report["coverage"]["excluded_cash_or_trade_rows"] == 1
    assert report["issues"][0]["reason"] == reason
    assert channel(report, "right")["stock_count"] == 0  # current account_mode cannot substitute for origin


def test_unknown_terminal_inventory_does_not_claim_positions_are_closed():
    report = build_channel_contribution([sale("A", 12., [allocation("left", 100, 1000.)])], None, initial_capital=10_000.)
    assert report["attribution_coverage_status"] == "data_limited"
    assert channel(report, "left")["stocks"][0]["outcome_status"] == "terminal_inventory_unknown"


def test_buy_fee_is_deducted_only_when_allocated_to_sale_or_terminal_inventory():
    origin = allocation("left", 100, 1000., 2.)
    buy = {"code": "A", "execution_quantity": 100., "origin_allocations": [origin], "transaction_cost_amount_raw": 2.}
    events = [buy, sale("A", 12., [allocation("left", 50, 500., 1.)], fee=1.)]
    positions = [{"code": "A", "close": 12., "quantity": 50,
                  "left_batches": [{"quantity": 50, "origin_allocation": allocation("left", 50, 500., 1.)}]}]
    left = channel(build_channel_contribution(events, positions, initial_capital=10_000.), "left")
    assert left["fees_paid"] == 3.
    assert left["realized_net_profit"] == 98. and left["unrealized_net_profit"] == 99.
    assert left["realized_net_profit"] + left["unrealized_net_profit"] == 197.


def test_invalid_terminal_inventory_is_disclosed_without_double_counting():
    position = {"code": "A", "close": 12., "quantity": 100,
                "batches": [{"quantity": 100, "origin_allocation": allocation("left", 100, 1000.)}]}
    report = build_channel_contribution([], [position, deepcopy(position)], initial_capital=10_000.)
    assert report["coverage"]["excluded_final_position_rows"] == 1
    left = channel(report, "left")
    assert left["unrealized_net_profit"] == 200. and left["stocks"][0]["open_quantity"] == 100.
    assert left["stocks"][0]["outcome_status"] == "terminal_inventory_unknown"


def test_missing_cost_and_rounded_fallback_are_visible_in_coverage():
    event = sale("A", 12., [allocation("left", 100, 1000.)])
    missing = deepcopy(event)
    missing["origin_allocations"][0].pop("entry_fee_amount")
    report = build_channel_contribution([event, missing], [], initial_capital=10_000.)
    assert report["coverage"]["excluded_cash_or_trade_rows"] == 1
    assert channel(report, "left")["stocks"][0]["outcome_status"] == "attribution_incomplete"
    rounded = {**event, "transaction_cost_amount_raw": "", "transaction_cost_amount": 1.}
    report = build_channel_contribution([rounded], [], initial_capital=10_000.)
    assert report["coverage"]["rounded_fee_fallback_rows"] == 1
    assert report["attribution_coverage_status"] == "data_limited"
    assert report["covered_realized_net_profit"] == 199.


def test_file_adapter_without_receipt_does_not_claim_observed_hashes_bind_an_audited_run(tmp_path):
    event = sale("A", 12., [allocation("left", 100, 1000.)])
    path = tmp_path / "events.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=event.keys())
        writer.writeheader()
        writer.writerow(event)
    summary = {"initial_capital": 10_000., "final_positions": [], "event_count": 1, "end_date": "2026-07-21",
               "accounting_audit": {"status": "data_limited", "error_count": 0, "errors": []}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    report = report_from_files(path, summary_path)
    assert report["declared_source_accounting_status"] == "data_limited"
    assert report["source_binding_status"] == "unverified_source_pair"
    assert report["status"] == "data_limited"
    assert report["source_receipt"] is None
    assert len(report["source_files"]["events"]["sha256"]) == 64
    assert channel(report, "left")["realized_net_profit"] == 200.
    for changes, error in [({"event_count": 2}, "row counts"),
                           ({"end_date": "2026-07-19"}, "cutoff"),
                           ({"accounting_audit": {"status": "failed", "error_count": 1}}, "zero-error")]:
        summary_path.write_text(json.dumps({**summary, **changes}))
        with pytest.raises(ValueError, match=error):
            report_from_files(path, summary_path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_events(path, events):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in events for key in row)))
        writer.writeheader()
        writer.writerows(events)


@pytest.fixture
def archived_source_pair(tmp_path):
    buy = {"code": "A", "date": "2026-07-20", "execution_phase": "open", "event_sequence": 1,
           "execution_quantity": 100, "execution_price": 10., "transaction_cost_amount_raw": 2.,
           "profit_loss_amount_raw": -2., "cash_change_amount_raw": -1002., "cash_before": 10_000.,
           "cash_after": 8998., "origin_allocations": [allocation("left", 100, 1000., 2.)]}
    neutral = {"code": "A", "date": "2026-07-20", "execution_quantity": 0., "event_sequence": 2, "action": "transfer"}
    sell = {**sale("A", 12., [allocation("left", 50, 500., 1.)], fee=1.), "event_sequence": 3,
            "execution_phase": "intraday", "profit_loss_amount_raw": 98., "cash_change_amount_raw": 599.,
            "cash_before": 8998., "cash_after": 9597.}
    positions = [{"code": "A", "close": 12., "quantity": 50,
                  "batches": [{"quantity": 50, "origin_allocation": allocation("left", 50, 500., 1.)}]}]
    summary = {"initial_capital": 10_000., "final_positions": positions, "event_count": 3, "end_date": "2026-07-21",
               "accounting_audit": {"status": "data_limited", "error_count": 0, "errors": []},
               "frozen_run": {"input_bundle_sha256": "a" * 64,
                              "engine_parameters": {"initial_capital": 10_000., "end_date": "2026-07-21"}},
               "trade_ledger": [buy, sell],
               "capital_risk_attribution": {"by_origin": [
                   {"origin_account": "left", "realized_net_pnl": 98., "unrealized_net_pnl": 99.,
                    "dividend_cash": 0., "dividend_receivable": 0., "transaction_cost": 3., "total_net_pnl": 197.}]}}
    events_path, summary_path = tmp_path / "events.csv", tmp_path / "summary.json"
    events = [buy, neutral, sell]
    write_events(events_path, events)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt_path = tmp_path / "comparison.json"
    receipt = {"input_bundle_sha256": "a" * 64,
               "determinism": {"events": {"identical": True, "sha256": [digest(events_path)] * 2}},
               "repaired": {"run_manifest": summary["frozen_run"], "accounting_audit": summary["accounting_audit"],
                            "capital_risk_attribution": summary["capital_risk_attribution"]}}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return events_path, summary_path, events, summary, {"path": receipt_path, "sha256": digest(receipt_path)}


def test_legacy_receipt_checks_complete_events_trade_economics_and_terminal_origin(archived_source_pair):
    events_path, summary_path, _, _, receipt = archived_source_pair
    report = report_from_files(events_path, summary_path, source_receipt=receipt)
    assert report["source_binding_status"] == "verified_legacy_comparison_scope"
    assert report["source_receipt"]["sha256"] == receipt["sha256"]
    assert channel(report, "left")["unrealized_net_profit"] == 99.
    assert any("no summary byte hash" in item for item in report["limitations"])


def test_legacy_receipt_rejects_same_count_csv_with_different_nontrade_event(archived_source_pair):
    events_path, summary_path, events, _, receipt = archived_source_pair
    events[1]["action"] = "different_run"
    write_events(events_path, events)
    with pytest.raises(ValueError, match="entire events CSV"):
        report_from_files(events_path, summary_path, source_receipt=receipt)


@pytest.mark.parametrize("change,error", [
    ("audit", "accounting_audit"), ("attribution", "capital_risk_attribution"),
    ("ledger", "cash_after"), ("valuation", "unrealized_net_pnl"), ("inventory", "Terminal origin inventory"),
])
def test_legacy_receipt_rejects_same_count_summary_from_different_run(archived_source_pair, change, error):
    events_path, summary_path, _, summary, receipt = archived_source_pair
    if change == "audit":
        summary["accounting_audit"]["amount_tolerance"] = .01
    elif change == "attribution":
        summary["capital_risk_attribution"]["by_origin"][0]["realized_net_pnl"] = 99.
    elif change == "ledger":
        summary["trade_ledger"][0]["cash_after"] += 1.
    elif change == "valuation":
        summary["final_positions"][0]["close"] += 1.
    else:
        # Keep terminal channel PnL exactly 99 while changing quantity and cost.
        position = summary["final_positions"][0]
        position["quantity"] = 100.
        position["batches"][0].update(quantity=100., origin_allocation=allocation("left", 100, 1100., 1.))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        report_from_files(events_path, summary_path, source_receipt=receipt)


def test_receipt_sha_is_external_evidence_and_cannot_be_silently_recomputed(archived_source_pair):
    events_path, summary_path, _, _, receipt = archived_source_pair
    receipt["path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA256 mismatch"):
        report_from_files(events_path, summary_path, source_receipt=receipt)


@pytest.mark.parametrize("changed_source", ["events", "summary"])
def test_new_output_receipt_binds_both_exact_source_files(archived_source_pair, changed_source):
    events_path, summary_path, _, _, receipt = archived_source_pair
    receipt["path"].write_text(json.dumps({"output_sha256": {
        events_path.name: digest(events_path), summary_path.name: digest(summary_path)}}), encoding="utf-8")
    receipt["sha256"] = digest(receipt["path"])
    report = report_from_files(events_path, summary_path, source_receipt=receipt)
    assert report["source_binding_status"] == "verified_output_hashes"
    path = events_path if changed_source == "events" else summary_path
    path.write_bytes(path.read_bytes() + b"\n")  # same rows/data, different archived bytes
    with pytest.raises(ValueError, match="bind both events and summary"):
        report_from_files(events_path, summary_path, source_receipt=receipt)
