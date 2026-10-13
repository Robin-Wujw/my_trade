"""Independent cash/share/valuation reconstruction from fills and source actions.

Engine cash changes, reported costs and PnL are assertions, never ledger inputs.
Amounts use unrounded execution prices and quantities. The fixed tolerances
below account for emitted six-decimal balances, not unexplained ledger gaps.
"""
from collections import defaultdict
import math

import pandas as pd


AMOUNT_TOLERANCE = .0001
SHARE_TOLERANCE = 1e-8


def audit_portfolio_accounting(result, price_frames, corporate_actions, parameters):
    capital = float(parameters.get("initial_capital", 1_000_000.))
    cash = capital
    shares = defaultdict(float)
    receivable = 0.
    errors = []
    limits = list(result.get("corporate_action_limitations") or [])
    max_cash_difference = max_equity_difference = 0.
    daily_events = defaultdict(list)
    curve_dates = [str(point["date"]) for point in result["equity_curve"]]
    if not curve_dates:
        errors.append({"check": "missing_equity_curve"})
    if curve_dates != sorted(set(curve_dates)):
        errors.append({"check": "duplicate_or_unordered_curve_dates"})
    event_dates = [str(event["date"]) for event in result["events"]]
    if event_dates != sorted(event_dates):
        errors.append({"check": "unordered_event_dates"})
    for index, event in enumerate(result["events"], 1):
        if str(event["date"]) not in curve_dates:
            errors.append({"check": "event_outside_equity_curve", "date": event["date"]})
        if event.get("event_sequence") != index:
            errors.append({"check": "event_sequence", "expected": index})
        daily_events[str(event["date"])].append(event)
    closes = {}
    for code, frame in price_frames.items():
        column = "raw_close" if "raw_close" in frame else "close"
        series = pd.Series(pd.to_numeric(frame[column], errors="coerce").to_numpy(),
                           index=pd.to_datetime(frame["date"]).dt.normalize()).sort_index()
        closes[code] = series[~series.index.duplicated(keep="last")]
    actions = []
    pending_bonus = []
    seen = set()
    def date_value(value):
        value = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(value) else pd.Timestamp(value).normalize()
    def number(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.
        return value if math.isfinite(value) else 0.
    if hasattr(corporate_actions, "to_dict"):
        grouped = defaultdict(list)
        for row in corporate_actions.to_dict("records"):
            grouped[str(row.get("code") or row.get("ts_code"))].append(row)
        corporate_actions = grouped
    for code, rows in (corporate_actions or {}).items():
        iterator = rows.items() if isinstance(rows, dict) else ((None, row) for row in rows)
        for date_key, row in iterator:
            if not isinstance(row, dict):
                row = {"stk_div": float(row) - 1, "ex_date": date_key}
            ex = date_value(row.get("ex_date") or date_key)
            record = date_value(row.get("record_date"))
            pay = date_value(row.get("pay_date"))
            bonus = number(row.get("stk_bo_rate")) + number(row.get("stk_co_rate"))
            bonus = bonus or number(row.get("stk_div"))
            dividend = number(row.get("cash_div")) or number(row.get("cash_div_tax"))
            key = (code, record, ex, pay, bonus, dividend, date_value(row.get("div_listdate")))
            if key in seen or ex is None:
                continue
            seen.add(key)
            actions.append({"code": code, "ex": ex, "record": record, "pay": pay,
                            "bonus": bonus, "dividend": dividend, "quantity": None,
                            "listing": date_value(row.get("div_listdate")),
                            "applied": False, "paid": False})
    grouped_actions = defaultdict(list)
    for action in actions:
        grouped_actions[(action["code"], action["record"], action["ex"])].append(action)
    actions = []
    for revisions in grouped_actions.values():
        action = dict(revisions[0])
        pay_dates = {item["pay"] for item in revisions if item["pay"] is not None}
        listing_dates = {item["listing"] for item in revisions if item["listing"] is not None}
        terms = {(item["bonus"], item["dividend"]) for item in revisions}
        action["ambiguous"] = len(terms) > 1 or len(pay_dates) > 1 or len(listing_dates) > 1
        action["pay"] = next(iter(pay_dates)) if len(pay_dates) == 1 else None
        action["listing"] = next(iter(listing_dates)) if len(listing_dates) == 1 else None
        actions.append(action)

    actions_by_code = defaultdict(list)
    for action in sorted(actions, key=lambda item: item["ex"]):
        actions_by_code[action["code"]].append(action)

    def limitation(action, reason):
        item = {"code": action["code"], "ex_date": str(action["ex"].date()), "reason": reason}
        if item not in limits:
            limits.append(item)

    def check(label, actual, expected, tolerance=AMOUNT_TOLERANCE):
        if (actual is None or not math.isfinite(float(actual)) or not math.isfinite(expected)
                or abs(float(actual) - expected) > tolerance):
            errors.append({"check": label, "actual": actual, "expected": expected})
    for point in result["equity_curve"]:
        date = pd.Timestamp(point["date"])
        text = str(point["date"])
        expected_dividend = defaultdict(float)
        expected_bonus = defaultdict(float)
        for action in actions:
            qty = action["quantity"]
            if qty is None:
                continue
            if not action["applied"] and action["ex"] <= date:
                action["applied"] = True
                if action["ambiguous"]:
                    if qty:
                        limitation(action, "conflicting_action_revisions")
                    action["paid"] = True
                    continue
                if qty * action["dividend"] and (action["pay"] is None or action["pay"] < action["ex"]):
                    limitation(action, "pay_date_missing_or_before_ex_date")
                if qty * action["bonus"] and (action["listing"] is None or action["listing"] < action["ex"]):
                    limitation(action, "share_listing_date_missing_or_before_ex_date")
                delta = int(math.floor(qty * (1 + action["bonus"]) + .5)) - qty
                shares[action["code"]] += delta
                expected_bonus[action["code"]] += delta
                if delta and (action["listing"] is None or action["listing"] < action["ex"] or action["listing"] > date):
                    pending_bonus.append({"code": action["code"], "quantity": delta,
                                          "listing": action["listing"] if action["listing"] is not None and action["listing"] >= action["ex"] else None})
                receivable += qty * action["dividend"]
            if (action["applied"] and not action["paid"] and action["pay"] is not None
                    and action["ex"] <= action["pay"] <= date):
                action["paid"] = True
                amount = qty * action["dividend"]
                cash += amount
                receivable -= amount
                expected_dividend[action["code"]] += amount
        pending_bonus = [item for item in pending_bonus if item["listing"] is None or item["listing"] > date]
        sellable = dict(shares)
        for item in pending_bonus:
            sellable[item["code"]] = sellable.get(item["code"], 0.) - item["quantity"]
        actual_dividend = defaultdict(float)
        actual_bonus = defaultdict(float)
        prior_phase = -1
        phase_budget = cash
        for event in daily_events[text]:
            code = str(event["code"])
            qty = float(event.get("execution_quantity") or 0.)
            actual_dividend[code] += float(event.get("corporate_action_cash_amount") or 0.)
            actual_bonus[code] += float(event.get("corporate_action_quantity_delta") or 0.)
            if not qty:
                continue
            phase = {"corporate_action": 0, "open": 1, "intraday": 2, "close": 3}.get(event.get("execution_phase"), -1)
            if phase < prior_phase or phase < 0:
                errors.append({"check": f"{text}:{code}:execution_phase", "actual": event.get("execution_phase")})
            if phase != prior_phase:
                phase_budget = cash
            prior_phase = phase
            check(f"{text}:{code}:integer_shares", qty, round(qty), SHARE_TOLERANCE)
            price = float(event["execution_price"])
            turnover = abs(qty) * price
            commission = max(turnover * float(parameters.get("commission_rate", 0.)),
                             float(parameters.get("minimum_commission", 0.))) if turnover else 0.
            stamp = 0.
            if qty < 0:
                stamp_rate = parameters.get("sell_stamp_duty_rate_before_2023_08_28")
                if stamp_rate is None or date >= pd.Timestamp("2023-08-28"):
                    stamp_rate = parameters.get("sell_stamp_duty_rate", 0.)
                stamp = turnover * float(stamp_rate)
            fee = commission + stamp + turnover * float(parameters.get("estimated_slippage_rate", 0.))
            if qty > 0:
                if turnover + fee > phase_budget + AMOUNT_TOLERANCE:
                    errors.append({"check": f"{text}:{code}:same_phase_sale_funds_buy"})
                phase_budget -= turnover + fee
            elif event.get("terminal_write_off"):
                end_date = (parameters.get("security_end_dates") or {}).get(code)
                if end_date is None or date < pd.Timestamp(end_date):
                    errors.append({"check": f"{text}:{code}:unjustified_terminal_write_off"})
                pending_bonus = [item for item in pending_bonus if item["code"] != code]
            else:
                if -qty > sellable.get(code, 0.) + SHARE_TOLERANCE:
                    errors.append({"check": f"{text}:{code}:t_plus_one_or_unlisted_shares_sold"})
                sellable[code] = sellable.get(code, 0.) + qty
            check(f"{text}:{code}:cash_before", event.get("cash_before"), cash)
            check(f"{text}:{code}:cash_change", event.get("cash_change_amount_raw", event.get("cash_change_amount")),
                  -qty * price - fee, AMOUNT_TOLERANCE if "cash_change_amount_raw" in event else .00501)
            cash -= qty * price + fee
            shares[code] += qty
            if cash < -AMOUNT_TOLERANCE or shares[code] < -SHARE_TOLERANCE:
                errors.append({"check": f"{text}:{code}:negative_balance", "cash": cash, "shares": shares[code]})
            check(f"{text}:{code}:cash_after", event.get("cash_after"), cash)
            check(f"{text}:{code}:fees", event.get("transaction_cost_amount_raw", event.get("transaction_cost_amount")),
                  fee, AMOUNT_TOLERANCE if "transaction_cost_amount_raw" in event else .00501)
        for code in set(expected_dividend) | set(actual_dividend):
            check(f"{text}:{code}:dividend_source", actual_dividend[code], expected_dividend[code])
        for code in set(expected_bonus) | set(actual_bonus):
            check(f"{text}:{code}:bonus_source", actual_bonus[code], expected_bonus[code], SHARE_TOLERANCE)
        for action in actions:
            if action["record"] == date:
                action["quantity"] = shares[action["code"]]
        market_value = 0.
        for code, qty in shares.items():
            if abs(qty) < SHARE_TOLERANCE:
                continue
            visible = closes[code].loc[:date]
            if visible.empty or not math.isfinite(float(visible.iloc[-1])):
                errors.append({"check": f"{text}:{code}:missing_valuation"})
                continue
            close = float(visible.iloc[-1])
            quote_date = visible.index[-1]
            for action in actions_by_code[code]:
                if quote_date < action["ex"] <= date and not action["ambiguous"]:
                    close = max(0., (close - action["dividend"]) / (1 + action["bonus"]))
            market_value += qty * close
        expected_equity = cash + market_value + receivable
        check(f"{text}:cash", point.get("cash"), cash)
        check(f"{text}:receivable", point.get("dividend_receivable", 0.), receivable)
        check(f"{text}:equity", float(point["equity"]) * capital, expected_equity)
        if "share_balances" in point:
            for code in set(shares) | set(point["share_balances"]):
                check(f"{text}:{code}:inventory", point["share_balances"].get(code, 0.), shares[code], SHARE_TOLERANCE)
        max_cash_difference = max(max_cash_difference, abs(float(point["cash"]) - cash))
        max_equity_difference = max(max_equity_difference, abs(float(point["equity"]) * capital - expected_equity))
    final_shares = {str(p["code"]): float(p["quantity"]) for p in result["final_positions"]}
    for code in set(shares) | set(final_shares):
        check(f"{code}:final_inventory", final_shares.get(code, 0.), shares[code], SHARE_TOLERANCE)
    concentration = result.get("profit_concentration_summary") or {}
    if result["equity_curve"] and "net_total_pnl_amount_raw" in concentration:
        check("symbol_pnl_reconciliation", concentration["net_total_pnl_amount_raw"],
              (float(result["equity_curve"][-1]["equity"]) - 1) * capital)
    attribution = result.get("capital_risk_attribution") or {}
    if "origin_pnl_reconciliation_difference" in attribution:
        check("origin_pnl_reconciliation", attribution["origin_pnl_reconciliation_difference"], 0.)
    if attribution.get("missing_origin_events", 0):
        errors.append({"check": "missing_origin_events", "actual": attribution["missing_origin_events"]})
    return {"status": "failed" if errors else "data_limited" if limits else "passed",
            "amount_tolerance": AMOUNT_TOLERANCE, "share_tolerance": SHARE_TOLERANCE,
            "max_cash_difference": max_cash_difference, "max_equity_difference": max_equity_difference,
            "error_count": len(errors), "errors": errors[:100], "data_limitations": limits,
            "basis": "fills_and_fees_recomputed_from_frozen_parameters_and_original_actions"}
