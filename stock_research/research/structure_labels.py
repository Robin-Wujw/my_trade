"""Independent, executable R03 campaigns; never broker or portfolio orders.

The fixed contract was recorded before outcome production. A close return is
not a fill. Missing bars, limits or action coverage never become zero returns.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stock_research.backtesting.corporate_actions import (
    CorporateActionBook, _corporate_action_adjusted_quantities, _normalize_corporate_actions,
)
from stock_research.backtesting.execution import ExecutionCosts, fill_cash_buy, fill_cash_sell
from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting
from stock_research.research.structure_events import HORIZONS, _local_time
from stock_research.strategies.ohlc_execution import (
    fill_buy_stop, fill_limit_order, fill_market_on_open, fill_sell_stop,
)


CONTRACT = {
    "version": "structure-label-execution-v1",
    "registered_at": "2026-10-14T09:35:40Z", "initial_cash": 100_000.,
    "entry": "next_session_only; close_at_open; limit_or_buy_stop_day_order",
    "stop": "preplaced_from_session_after_buy; blocked_stop_persists_at_next_open",
    "horizon": "observation_calendar_T20_T60_close_decision; next_open_liquidation",
    "target_profit_exit": False, "commission_rate": .000085, "minimum_commission": 5.,
    "estimated_slippage_rate": .001, "sell_stamp_duty_rate": .0005,
    "sell_stamp_duty_rate_before_2023_08_28": .001,
    "r_denominator": "initial_filled_quantity_times_raw_entry_minus_structural_stop",
    "income": "settled_net_cash_after_all_shares_and_entitlements",
    "research_only": True, "strict_point_in_time": False,
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()
COSTS = ExecutionCosts(CONTRACT["commission_rate"], CONTRACT["minimum_commission"],
                       CONTRACT["estimated_slippage_rate"], CONTRACT["sell_stamp_duty_rate"],
                       CONTRACT["sell_stamp_duty_rate_before_2023_08_28"])


def _date(value):
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("Execution requires explicit local session dates")
    return date


def _base(event, horizon, available_at, source_sha256):
    if horizon not in HORIZONS or type(horizon) is not int:
        raise ValueError("Only registered T20/T60 label horizons are allowed")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)):
        raise ValueError("Execution labels require a source SHA256")
    available = _local_time(available_at)
    if _local_time(event["captured_at"]) > available:
        raise ValueError("Event is not available before outcome capture")
    return {"event_id": event["event_id"], "horizon": horizon,
            "available_at": available.isoformat(), "exit_date": None,
            "net_r": None, "executable": False, "cost_included": True,
            "label_source_sha256": source_sha256, "contract_sha256": CONTRACT_SHA256,
            "forward_contract_eligible": _local_time(event["captured_at"]) > _local_time(CONTRACT["registered_at"]),
            "research_only": True, "strict_point_in_time": False}


def executable_structure_label(event, raw_bars, sessions, corporate_actions, *, horizon,
                               available_at, source_sha256, corporate_actions_complete):
    """Replay one independent campaign, retaining pending/blocked evidence.

    ``sessions`` must come from an archived exchange calendar. The source must
    contain explicit suspension rows; an absent quote is unknown. Source raw
    prices and precise limits stay distinct from the observation's signal basis.
    This pure research executor does not assert authenticity of caller receipts;
    the immutable capture adapter binds and dates those files separately.
    """
    label = _base(event, horizon, available_at, source_sha256)
    def limited(reason):
        return {**label, "status": "data_limited", "reason": reason}
    if corporate_actions_complete is not True:
        return limited("corporate_action_coverage_not_proven")
    if event.get("price_basis") != "observation_raw_anchor":
        return limited("observation_raw_signal_basis_not_proven")
    observed = _date(event["observation_date"])
    available = _local_time(available_at)
    cutoff = available.tz_localize(None).normalize()
    if (available.hour, available.minute) < (16, 30):
        # The daily bar includes the close. It is not an opening-time source.
        cutoff -= pd.Timedelta(days=1)
    calendar = pd.DatetimeIndex([_date(day) for day in sessions])
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or observed not in calendar:
        raise ValueError("Unique ordered exchange calendar including observation is required")
    calendar = calendar[(calendar >= observed) & (calendar <= cutoff)]
    if len(calendar) < 2:
        return {**label, "status": "pending_entry_session"}
    data = pd.DataFrame(raw_bars).copy()
    required = {"date", "open", "high", "low", "close", "volume", "tradestatus",
                "up_limit", "down_limit", "raw_to_qfq_factor"}
    if required - set(data):
        return limited("missing_raw_execution_fields")
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    if data.date.isna().any() or data.date.dt.tz is not None:
        raise ValueError("Unknown or nonlocal execution quote dates")
    data = data.loc[data.date.isin(calendar)].sort_values("date")
    if data.date.duplicated().any():
        raise ValueError("Duplicate execution quote dates")
    if set(data.date) != set(calendar):
        return limited("missing_exchange_session_bar_not_assumed_suspended")
    for field in ("open", "high", "low", "close"):
        if f"raw_{field}" in data:
            data[field] = data[f"raw_{field}"]
    numeric = data[list(required - {"date"})].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric).all().all():
        return limited("incomplete_execution_prices_limits_or_status")
    if (not numeric[["open", "high", "low", "close", "up_limit", "down_limit", "raw_to_qfq_factor"]].gt(0).all().all()
            or numeric.volume.lt(0).any() or not numeric.tradestatus.isin([0, 1]).all()
            or numeric.high.lt(numeric[["open", "close", "low"]].max(axis=1)).any()
            or numeric.low.gt(numeric[["open", "close", "high"]].min(axis=1)).any()
            or numeric.up_limit.lt(numeric.down_limit).any()):
        return limited("invalid_raw_execution_bar")
    if (numeric.high.gt(numeric.up_limit + .010000001).any()
            or numeric.low.lt(numeric.down_limit - .010000001).any()):
        return limited("raw_prices_conflict_with_exact_limits")
    data[list(numeric)] = numeric
    data = data.reset_index(drop=True)
    code = str(event["code"])
    actions = corporate_actions.to_dict("records") if hasattr(corporate_actions, "to_dict") else list(corporate_actions)
    for row in actions:
        if (not isinstance(row, dict) or row.get("code", code) != code
                or pd.isna(pd.to_datetime(row.get("ex_date"), errors="coerce"))):
            return limited("incomplete_or_mismatched_action_source")
        supported_amounts = []
        for field in ("cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate"):
            value = row.get(field)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                return limited("invalid_corporate_action_terms")
            if isinstance(value, bool) or not math.isfinite(amount) or amount < 0:
                return limited("invalid_corporate_action_terms")
            supported_amounts.append(amount)
        if not supported_amounts or not any(amount > 0 for amount in supported_amounts):
            return limited("missing_supported_corporate_action_terms")
        for field in ("rights_ratio", "rights_price", "rights_issue_ratio", "allotment_ratio"):
            if row.get(field) is not None:
                return limited("unsupported_corporate_action_entitlement")
        stated_bonus = pd.to_numeric(row.get("stk_div"), errors="coerce")
        bonus_parts = sum(float(row.get(key) or 0.) for key in ("stk_bo_rate", "stk_co_rate")
                          if pd.notna(row.get(key)))
        if (pd.notna(stated_bonus) and bonus_parts > 0
                and not math.isclose(float(stated_bonus), bonus_parts, rel_tol=1e-8, abs_tol=1e-10)):
            return limited("conflicting_corporate_action_share_terms")
    action_lookup = _normalize_corporate_actions({code: actions})
    book = CorporateActionBook(action_lookup, observed)
    # A price-factor jump without a documented action must not silently adjust
    # stops or create dividend/shares. Rights issues are outside this contract.
    action_days = set(action_lookup.get(code, {}))
    changed = data.raw_to_qfq_factor.pct_change().abs().gt(1e-10)
    if any(day not in action_days for day in data.loc[changed, "date"]):
        return limited("adjustment_change_without_documented_corporate_action")
    signal = json.loads(event["signal"]) if isinstance(event["signal"], str) else event["signal"]
    order_type = signal.get("order_type")
    if order_type not in {"close", "limit", "buy_stop", "stop"}:
        return limited("unregistered_entry_order_type")
    stop = float(event["planned_stop"])
    trigger = float(event["planned_entry"])
    if not math.isfinite(stop) or not math.isfinite(trigger) or not 0 < stop < trigger:
        raise ValueError("Invalid raw structural entry/stop")
    for action in book.actions:
        if observed < action["ex"] <= calendar[1]:
            if action["ambiguous"]:
                return limited("conflicting_action_revisions")
            stop = (stop - action["cash"]) / (1 + action["bonus"])
            trigger = (trigger - action["cash"]) / (1 + action["bonus"])
    if stop <= 0 or trigger <= stop:
        return limited("invalid_entry_after_corporate_action")
    state = SimpleNamespace(left=[], right=[])
    states = {code: state}
    cash = float(CONTRACT["initial_cash"])
    capital = cash
    result = {"events": [], "equity_curve": [], "final_positions": []}
    initial_risk = None
    stopped = False
    last_sell = None
    blocks = []
    def emit(day, item):
        result["events"].append({**item, "date": str(day.date()), "code": code,
                                  "event_sequence": len(result["events"]) + 1})
    def trade(day, quantity, price, phase):
        nonlocal cash
        before = cash
        costs = COSTS.components(abs(quantity) * price, sell=quantity < 0, trade_date=day)
        if quantity > 0:
            cash -= quantity * price + costs["transaction_cost_amount"]
        else:
            fill = fill_cash_sell(-quantity, price, 0., initial_capital=capital,
                                  cash_balance=cash, trade_date=day, costs=COSTS)
            cash = fill.cash_after
        emit(day, {"execution_quantity": quantity, "execution_price": price,
                   "execution_phase": phase, "cash_before": before, "cash_after": cash,
                   "cash_change_amount_raw": cash - before,
                   "transaction_cost_amount_raw": costs["transaction_cost_amount"], **costs})
    for index, row in data.iterrows():
        day = row.date
        events, delta = book.advance(day, states, _corporate_action_adjusted_quantities)
        cash += delta
        for item in events:
            emit(day, {**item, "execution_phase": "corporate_action"})
        previous = float(data.iloc[index - 1].close) if index else float(row.close)
        if index == 1:
            if order_type == "close":
                entry = fill_market_on_open(row, side="buy", previous_close=previous, code=code)
                phase = "open"
            elif order_type == "limit":
                entry = fill_limit_order(row, side="buy", limit_price=trigger,
                                         previous_close=previous, code=code)
                phase = "open" if row.open <= trigger else "intraday"
            else:
                entry = fill_buy_stop(row, trigger_price=trigger, previous_close=previous, code=code)
                phase = "open" if row.open >= trigger else "intraday"
            if not entry["filled"]:
                return {**label, "status": "entry_not_filled", "reason": entry["status"],
                        "entry_session": str(day.date())}
            price = float(entry["price"])
            if price <= stop:
                return {**label, "status": "entry_not_filled", "reason": "open_below_structural_stop"}
            fill = fill_cash_buy(code, 1., price, initial_capital=capital, cash_balance=cash,
                                 phase_buy_budget=cash, costs=COSTS)
            if fill.quantity <= 0:
                return {**label, "status": "entry_not_filled", "reason": "cash_or_board_lot"}
            initial_risk = fill.quantity * (price - stop)
            state.right.append({"quantity": fill.quantity, "cost": price, "size": fill.size,
                                "entry_fee_cash": fill.fee_cash, "batch": "R03", "entry_date": day,
                                "initial_stop_raw_price": stop})
            trade(day, fill.quantity, price, phase)
        if index >= 2 and (state.right or book.lots(code)):
            horizon_exit = index > horizon
            if stopped or horizon_exit:
                exit_fill = fill_market_on_open(row, side="sell", previous_close=previous, code=code)
                phase = "open"
            else:
                live_stop = float(state.right[0]["initial_stop_raw_price"]) if state.right else stop
                exit_fill = fill_sell_stop(row, stop_price=live_stop, previous_close=previous, code=code)
                phase = "open" if row.open <= live_stop else "intraday"
                if (exit_fill["status"] in {"gap_stop", "intraday_stop", "locked_limit_down"}
                        and (row.open <= live_stop or row.low < live_stop)):
                    stopped = True
                # Suspension cannot show a touch, so only a previously triggered
                # stop or the known horizon can establish a continuing exit.
            if (stopped or horizon_exit) and not exit_fill["filled"]:
                blocks.append({"date": str(day.date()), "status": exit_fill["status"]})
            if exit_fill["filled"]:
                eligible = [lot for lot in state.right if day > lot["entry_date"]
                            and day >= pd.Timestamp(lot.get("sellable_from", day))]
                quantity = sum(lot["quantity"] for lot in eligible)
                if quantity:
                    trade(day, -quantity, float(exit_fill["price"]), phase)
                    state.right[:] = [lot for lot in state.right if lot not in eligible]
                    last_sell = day
        book.register(day, states)
        quantity = sum(lot["quantity"] for lot in state.right + book.lots(code))
        result["equity_curve"].append({"date": str(day.date()), "cash": cash,
            "equity": (cash + quantity * row.close + book.receivable) / capital,
            "dividend_receivable": book.receivable, "share_balances": {code: quantity}})
        outstanding = any(action["registered"] and (not action["applied"]
                          or (action["amount"] and not action["paid"])) for action in book.actions)
        if index > horizon and quantity == 0 and book.receivable == 0 and not outstanding:
            break
    quantity = sum(lot["quantity"] for lot in state.right + book.lots(code))
    if quantity:
        result["final_positions"] = [{"code": code, "quantity": quantity}]
    result["corporate_action_limitations"] = book.limitations
    parameters = {**CONTRACT, "initial_capital": capital}
    audit = audit_portfolio_accounting(result, {code: data}, {code: actions}, parameters)
    if audit["error_count"]:
        raise ValueError(f"Independent structure accounting failed: {audit['errors']}")
    label.update({"accounting_audit": audit, "execution": result, "initial_risk_amount": initial_risk,
                  "blocked_exits": blocks, "costs": asdict(COSTS),
                  "exit_date": str(last_sell.date()) if last_sell is not None else None})
    if book.limitations:
        return {**label, "status": "data_limited", "reason": "incomplete_corporate_action_settlement"}
    if len(calendar) <= horizon + 1:
        return {**label, "status": "pending_horizon"}
    if quantity or book.receivable or outstanding:
        return {**label, "status": "pending_exit_or_settlement"}
    return {**label, "status": "completed", "net_r": (cash - capital) / initial_risk,
            "net_profit_amount": cash - capital, "executable": True}
