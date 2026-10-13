"""Registered dividend rights, receivables and shares awaiting listing.

The caller supplies actual lots at each record-date close. Price adjustment
factors never create securities or reduce historical total acquisition cost.
"""
from copy import deepcopy
import math

import pandas as pd


def action_date(value):
    value = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value).normalize()


def number(value, default=0.):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def origin(lot, quantity=None):
    qty = float(lot["quantity"] if quantity is None else quantity)
    fraction = qty / float(lot["quantity"]) if lot["quantity"] else 0.
    return {"batch": lot["batch"], "origin_batch": lot.get("origin_batch", lot["batch"]),
            "origin_account": lot.get("origin_account") or ("left" if str(lot["batch"]).startswith("L") else "right"),
            "quantity": qty, "cost_amount": qty * float(lot["cost"]),
            "entry_fee_amount": float(lot.get("entry_fee_cash") or 0.) * fraction}


class CorporateActionBook:
    def __init__(self, actions, start):
        self.actions = []
        self.pending = []
        self.receivable = 0.
        self.limitations = []
        self.start = pd.Timestamp(start).normalize()
        by_event = {}
        for code, by_date in actions.items():
            for ex_date, rows in by_date.items():
                for row in rows:
                    row = row if isinstance(row, dict) else {"stk_div": float(row) - 1}
                    record = action_date(row.get("record_date"))
                    pay = action_date(row.get("pay_date"))
                    listing = action_date(row.get("div_listdate"))
                    bonus = number(row.get("stk_bo_rate")) + number(row.get("stk_co_rate"))
                    bonus = bonus or number(row.get("stk_div"))
                    cash = number(row.get("cash_div")) or number(row.get("cash_div_tax"))
                    key = (code, record, ex_date)
                    if key in by_event:
                        previous = by_event[key]
                        if (previous["bonus"], previous["cash"]) != (bonus, cash):
                            previous["ambiguous"] = True
                        for field, value in (("pay", pay), ("listing", listing)):
                            if previous[field] is None:
                                previous[field] = value
                            elif value is not None and previous[field] != value:
                                previous["ambiguous"] = True
                        continue
                    by_event[key] = {"code": code, "record": record, "ex": ex_date,
                                     "pay": pay, "listing": listing, "bonus": bonus,
                                     "cash": cash, "ambiguous": False}
        self.actions_by_code = {}
        for item in by_event.values():
            action = {**item, "registered": None, "applied": False,
                      "paid": False, "amount": 0., "id": len(self.actions) + 1}
            self.actions.append(action)
            self.actions_by_code.setdefault(item["code"], []).append(action)
        for rows in self.actions_by_code.values():
            rows.sort(key=lambda action: action["ex"])

    def limitation(self, action, reason):
        item = {"code": action["code"], "ex_date": str(action["ex"].date()), "reason": reason}
        if item not in self.limitations:
            self.limitations.append(item)

    def lots(self, code=None):
        return [p["lot"] for p in self.pending if code is None or p["code"] == code]

    def register(self, date, states):
        for action in self.actions:
            if action["record"] != date or action["registered"] is not None:
                continue
            state = states.get(action["code"])
            if state is None:
                action["registered"] = []
                continue
            action["registered"] = [
                {"ref": lot, "lot": deepcopy(lot), "account": account}
                for account in ("left", "right") for lot in getattr(state, account)
            ]
            action["registered"] += [
                {"ref": p["lot"], "lot": deepcopy(p["lot"]), "account": p["account"]}
                for p in self.pending if p["code"] == action["code"]
            ]

    def advance(self, date, states, allocate_quantities):
        """Return non-trading events and cash credited before this session."""
        events = []
        cash_delta = 0.
        for action in self.actions:
            code = action["code"]
            state = states.get(code)
            if state is None or action["ex"] > date:
                continue
            if not action["applied"]:
                registered = action["registered"]
                if registered is None:
                    # No starting positions exist before the requested interval.
                    if action["record"] is not None and action["record"] < self.start:
                        registered = []
                    else:
                        if state.left or state.right or self.lots(code):
                            self.limitation(action, "record_date_or_registered_holdings_missing")
                        registered = []
                action["registered"] = registered
                allocations = [origin(item["lot"]) for item in registered]
                quantity = sum(a["quantity"] for a in allocations)
                if action["ambiguous"]:
                    if quantity:
                        self.limitation(action, "conflicting_action_revisions")
                    action["applied"] = True
                    action["paid"] = True
                    continue
                action["amount"] = quantity * action["cash"]
                action["allocations"] = allocations
                action["applied"] = True
                base = {"code": code, "corporate_action_id": action["id"],
                        "corporate_action_effective_date": str(action["ex"].date()),
                        "corporate_action_quantity_before": quantity,
                        "origin_allocations": allocations, "account_mode": "corporate_action"}
                if action["amount"]:
                    self.receivable += action["amount"]
                    events.append({**base, "action": "现金分红应收确认",
                                   "dividend_receivable_change": action["amount"],
                                   "income_amount": action["amount"],
                                   "corporate_action_cash_dividend_per_share": action["cash"]})
                    if action["pay"] is None or action["pay"] < action["ex"]:
                        self.limitation(action, "pay_date_missing_or_before_ex_date")
                if action["cash"]:
                    for item in registered:
                        lot, old = item["ref"], item["lot"]
                        if not any(lot is present for present in getattr(state, item["account"]) + self.lots(code)):
                            continue
                        for key in ("initial_stop_raw_price", "initial_entry_raw_price", "initial_target_raw_price"):
                            if old.get(key) is not None:
                                lot.setdefault("original_" + key, old[key])
                                old[key] = float(old[key]) - action["cash"]
                                lot[key] = old[key]
                if action["bonus"] and quantity:
                    originals = [float(item["lot"]["quantity"]) for item in registered]
                    adjusted = allocate_quantities(originals, 1 + action["bonus"])
                    added = 0
                    for item, new_quantity in zip(registered, adjusted):
                        old = item["lot"]
                        lot = item["ref"]
                        old_quantity = float(old["quantity"])
                        extra = int(new_quantity - old_quantity)
                        if extra <= 0:
                            continue
                        live = any(lot is present for present in getattr(state, item["account"]) + self.lots(code))
                        if not live or abs(float(lot["quantity"]) - old_quantity) > 1e-8:
                            self.limitation(action, "registered_lot_changed_before_ex_date")
                            continue
                        ratio = new_quantity / old_quantity
                        bonus_lot = deepcopy(lot)
                        for target in (lot, bonus_lot):
                            target["cost"] = float(old["cost"]) / ratio
                            for key in ("initial_risk_per_share_raw", "initial_reward_per_share_raw",
                                        "initial_stop_raw_price", "initial_entry_raw_price", "initial_target_raw_price"):
                                if target.get(key) is not None:
                                    target.setdefault("original_" + key, target[key])
                                    target[key] = float(old[key]) / ratio
                        lot["size"] = float(old["size"]) / ratio
                        lot["entry_fee_cash"] = float(old.get("entry_fee_cash") or 0.) / ratio
                        bonus_lot.update(quantity=float(extra), size=float(old["size"]) - lot["size"],
                                         entry_fee_cash=float(old.get("entry_fee_cash") or 0.) - lot["entry_fee_cash"],
                                         batch=f"{old['batch']}:CA{action['id']}",
                                         origin_batch=old.get("origin_batch", old["batch"]),
                                         origin_account=origin(old)["origin_account"],
                                         corporate_action_pending=True)
                        self.pending.append({"code": code, "account": item["account"], "lot": bonus_lot,
                                             "listing": action["listing"], "action_id": action["id"]})
                        added += extra
                    if action["listing"] is None or action["listing"] < action["ex"]:
                        self.limitation(action, "share_listing_date_missing_or_before_ex_date")
                    events.append({**base, "action": "除权持仓调整",
                                   "corporate_action_quantity_multiplier": 1 + action["bonus"],
                                   "corporate_action_quantity_after": quantity + added,
                                   "corporate_action_quantity_delta": float(added)})
            if (not action["paid"] and action["amount"] and action["pay"] is not None
                    and action["ex"] <= action["pay"] <= date):
                amount = action["amount"]
                action["paid"] = True
                self.receivable -= amount
                cash_delta += amount
                events.append({"code": code, "action": "现金分红到账", "account_mode": "corporate_action",
                               "corporate_action_id": action["id"], "corporate_action_cash_amount": amount,
                               "corporate_action_cash_dividend_per_share": action["cash"],
                               "corporate_action_effective_date": str(action["pay"].date()),
                               "cash_change_amount": amount, "dividend_receivable_change": -amount,
                               "origin_allocations": action["allocations"]})
        for pending in list(self.pending):
            if pending["listing"] is None or pending["listing"] > date:
                continue
            action = self.actions[pending["action_id"] - 1]
            if pending["listing"] < action["ex"]:
                continue
            lot = pending["lot"]
            lot["corporate_action_pending"] = False
            lot["sellable_from"] = pending["listing"]
            getattr(states[pending["code"]], pending["account"]).append(lot)
            self.pending.remove(pending)
            events.append({"code": pending["code"], "action": "红股上市可售", "account_mode": "corporate_action",
                           "corporate_action_id": pending["action_id"], "listed_quantity": float(lot["quantity"]),
                           "origin_allocations": [origin(lot)]})
        return events, cash_delta


def _normalize_action_date(value):
    converted = pd.to_datetime(value, errors="coerce")
    if pd.isna(converted):
        return None
    return pd.Timestamp(converted).normalize()



def _action_share_multiplier(action):
    if action is None:
        return None
    if isinstance(action, (int, float)):
        value = float(action)
        return value if value > 0 else None
    if not isinstance(action, dict):
        return None
    components = []
    for key in ("stk_bo_rate", "stk_co_rate"):
        value = pd.to_numeric(action.get(key), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            components.append(float(value))
    if components:
        return 1.0 + sum(components)
    value = pd.to_numeric(action.get("stk_div"), errors="coerce")
    if pd.notna(value) and float(value) > 0:
        return 1.0 + float(value)
    return None



def _action_cash_dividend(action):
    if action is None or not isinstance(action, dict):
        return None
    for key in ("cash_div", "cash_div_tax"):
        value = pd.to_numeric(action.get(key), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return None



def _normalize_corporate_actions(corporate_actions):
    lookup = {}
    if hasattr(corporate_actions, "to_dict"):
        rows = corporate_actions.to_dict("records")
        corporate_actions = {}
        for row in rows:
            code = str(row.get("code") or row.get("ts_code") or "").strip()
            if code:
                corporate_actions.setdefault(code, []).append(row)
    for code, rows in dict(corporate_actions or {}).items():
        by_date = lookup.setdefault(str(code), {})
        if isinstance(rows, dict):
            iterator = rows.items()
        else:
            iterator = ((None, row) for row in rows or [])
        for date_key, row in iterator:
            if isinstance(row, dict):
                action_date = _normalize_action_date(
                    row.get("ex_date") or row.get("div_listdate") or row.get("date") or date_key,
                )
            else:
                action_date = _normalize_action_date(date_key)
            if action_date is None:
                continue
            multiplier = _action_share_multiplier(row)
            cash_dividend = _action_cash_dividend(row)
            if multiplier is None and cash_dividend is None:
                continue
            by_date.setdefault(action_date, []).append(row)
    return lookup



def _corporate_action_adjusted_quantities(quantities, multiplier) -> list[int]:
    originals = [max(0.0, float(quantity)) for quantity in quantities]
    if not originals:
        return []
    multiplier = max(0.0, float(multiplier))
    expected = [quantity * multiplier for quantity in originals]
    adjusted = [int(math.floor(value)) for value in expected]
    positive_count = sum(1 for quantity in originals if quantity > 0)
    target_total = max(positive_count, int(math.floor(sum(expected) + 0.5)))
    delta = target_total - sum(adjusted)
    if delta > 0:
        order = sorted(
            range(len(expected)),
            key=lambda index: (expected[index] - adjusted[index], originals[index]),
            reverse=True,
        )
        for index in order[:delta]:
            adjusted[index] += 1
    elif delta < 0:
        order = sorted(
            range(len(expected)),
            key=lambda index: (expected[index] - adjusted[index], originals[index]),
        )
        for index in order[:abs(delta)]:
            if adjusted[index] > 1:
                adjusted[index] -= 1
    return adjusted

