"""Source-bound, fixed-instrument cash contracts for research portfolios.

Only executable fixed-rate exchange reverse-repo quotes are supported. Daily
fund NAVs and end-of-day indicative yields are not executable quotes. Principal
is a separate locked asset; income is recognized only at contractual payment.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd


CONTRACT_FIELDS = {"schema_version", "instrument", "instrument_type", "currency", "timezone",
                   "execution_time", "allocation_fraction", "reserve_cash"}
QUOTE_FIELDS = {"quote_id", "instrument", "instrument_type", "currency", "source", "quote_kind",
                "trading_date", "observed_at", "available_at", "valid_until", "annual_rate", "day_count",
                "accrual_start_date", "accrual_end_date", "principal_release_at", "interest_release_at",
                "principal_increment", "minimum_principal", "maximum_principal", "entry_fee_rate",
                "minimum_entry_fee", "settlement_fee", "interest_tax_rate"}
MONEY = Decimal("0.01")


def _money(value):
    return Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)


def _finite_number(value, field, *, minimum=0, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid cash contract number: {field}")
    return value


def _time(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Cash evidence timestamps require an explicit timezone")
    return result.tz_convert("Asia/Shanghai")


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Cash evidence dates must be explicit local calendar dates")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Invalid cash evidence date")
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze_cash_contract(contract):
    if not isinstance(contract, dict) or set(contract) != CONTRACT_FIELDS:
        raise ValueError("Cash research requires the complete explicit fixed-instrument contract")
    result = json.loads(json.dumps(contract, allow_nan=False))
    if (type(result["schema_version"]) is not int or result["schema_version"] != 1
            or not isinstance(result["instrument"], str)
            or not re.fullmatch(r"(?:sh\.204\d{3}|sz\.131\d{3})", result["instrument"])
            or result["instrument_type"] != "exchange_reverse_repo" or result["currency"] != "CNY"
            or result["timezone"] != "Asia/Shanghai"):
        raise ValueError("Only explicitly identified CNY exchange reverse-repo contracts are supported")
    clock = result["execution_time"]
    if not isinstance(clock, str) or not re.fullmatch(r"15:[0-2]\d:[0-5]\d", clock) or clock == "15:00:00":
        raise ValueError("Cash execution must be a fixed time after 15:00 and before 15:30")
    fraction = _finite_number(result["allocation_fraction"], "allocation_fraction", maximum=1)
    if fraction <= 0:
        raise ValueError("Cash allocation_fraction must be positive")
    _finite_number(result["reserve_cash"], "reserve_cash")
    return result, _digest(result)


def load_cash_quotes(receipts, contract):
    """Read exact archived quote contracts, rejecting missing provenance and revisions."""
    if not isinstance(receipts, list):
        raise ValueError("cash_quotes must be an explicit list of local source receipts")
    quotes, seen_dates, seen_ids = [], set(), set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {"source_path", "source_sha256"}:
            raise ValueError("Cash quotes require source_path and source_sha256 receipts")
        path = Path(receipt["source_path"]).resolve()
        expected = receipt["source_sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Cash quote source hash is invalid")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Cash quote source hash mismatch")
        quote = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(quote, dict) or set(quote) != QUOTE_FIELDS:
            raise ValueError("Cash quote requires a complete execution and settlement contract")
        for field in ("instrument", "instrument_type", "currency"):
            if quote[field] != contract[field]:
                raise ValueError("Cash quote differs from the preselected fixed instrument")
        if (not isinstance(quote["source"], str) or not quote["source"].strip()
                or not isinstance(quote["quote_id"], str) or not quote["quote_id"].strip()
                or quote["quote_kind"] != "executable_lend_rate"):
            raise ValueError("Cash quotes must identify an executable lending quote and actual source")
        day = _day(quote["trading_date"])
        if quote["trading_date"] in seen_dates or quote["quote_id"] in seen_ids:
            raise ValueError("Duplicate cash quote day or identity; retrospective rate selection is forbidden")
        seen_dates.add(quote["trading_date"])
        seen_ids.add(quote["quote_id"])
        execution = _time(f"{quote['trading_date']}T{contract['execution_time']}+08:00")
        observed, available, valid_until = map(_time, (quote["observed_at"], quote["available_at"], quote["valid_until"]))
        principal_release, interest_release = map(_time, (quote["principal_release_at"], quote["interest_release_at"]))
        accrual_start, accrual_end = map(_day, (quote["accrual_start_date"], quote["accrual_end_date"]))
        if (observed > available or valid_until < observed or observed.tz_localize(None).normalize() != day
                or accrual_end <= accrual_start or accrual_start < day
                or principal_release <= execution or interest_release <= execution
                or interest_release.tz_localize(None) < accrual_end):
            raise ValueError("Cash quote observation, accrual or release dates are inconsistent")
        _finite_number(quote["annual_rate"], "annual_rate", maximum=1)
        if type(quote["day_count"]) is not int or quote["day_count"] not in {360, 365}:
            raise ValueError("Cash quote must state an explicit 360/365 day-count basis")
        for field in ("principal_increment", "minimum_principal", "maximum_principal",
                      "minimum_entry_fee", "settlement_fee"):
            _finite_number(quote[field], field)
            if Decimal(str(quote[field])) != _money(quote[field]):
                raise ValueError("Cash notional and fee amounts must use exact currency cents")
        for field in ("entry_fee_rate", "interest_tax_rate"):
            _finite_number(quote[field], field, maximum=1)
        increment, minimum, maximum = map(_money, (quote["principal_increment"], quote["minimum_principal"], quote["maximum_principal"]))
        if increment <= 0 or minimum < increment or maximum < minimum or minimum % increment:
            raise ValueError("Cash quote principal increments/limits are inconsistent")
        if _money(quote["settlement_fee"]) > minimum:
            raise ValueError("Cash settlement fee must be covered by returned principal")
        quote.update(source_path=str(path), source_sha256=expected)
        quotes.append(quote)
    return sorted(quotes, key=lambda row: row["trading_date"])


class CashBook:
    """Independent lock/fee/release/interest ledger, driven by visible timestamps."""

    def __init__(self, contract, quote_receipts):
        self.contract, self.contract_sha256 = freeze_cash_contract(contract)
        self.quotes = load_cash_quotes(quote_receipts, self.contract)
        self.quote_by_date = {row["trading_date"]: row for row in self.quotes}
        self.quotes_sha256 = _digest([{key: row[key] for key in ("quote_id", "source_sha256")} for row in self.quotes])
        self.ledger, self.decisions, self.positions = [], [], []
        self._last_time = None

    @property
    def principal(self):
        return float(sum((_money(row["principal"]) for row in self.positions if not row["principal_released"]), Decimal(0)))

    def execution_at(self, day):
        return _time(f"{pd.Timestamp(day):%Y-%m-%d}T{self.contract['execution_time']}+08:00")

    def calendar_dates(self, start, end):
        return {timestamp.tz_localize(None).normalize() for quote in self.quotes
                for timestamp in (_time(quote["principal_release_at"]), _time(quote["interest_release_at"]))
                if start <= timestamp.tz_localize(None).normalize() <= end}

    def verify_sources(self):
        for quote in self.quotes:
            if hashlib.sha256(Path(quote["source_path"]).read_bytes()).hexdigest() != quote["source_sha256"]:
                raise ValueError("Cash quote source changed during replay")

    def _append(self, position, action, timestamp, cash, stock_event_count, *, principal_delta=0, interest=0, fee=0, tax=0,
                processed_at=None):
        principal_delta, interest, fee, tax = map(_money, (principal_delta, interest, fee, tax))
        change = -principal_delta + interest - fee - tax
        before = float(cash)
        after = before + float(change)
        if after < -1e-8:
            raise ValueError("Cash contract settlement cannot borrow against locked principal")
        row = {"sequence": len(self.ledger) + 1, "position_id": position["position_id"],
               "instrument": self.contract["instrument"], "quote_id": position["quote_id"],
               "source_sha256": position["source_sha256"], "action": action,
               "timestamp": _time(timestamp).isoformat(), "stock_event_count": int(stock_event_count),
               "processed_at": _time(timestamp if processed_at is None else processed_at).isoformat(),
               "principal_delta": float(principal_delta), "gross_interest": float(interest),
               "fee": float(fee), "tax": float(tax), "cash_change": float(change),
               "cash_before": before, "cash_after": after,
               "income": float(interest - fee - tax), "research_only": True}
        self.ledger.append(row)
        return after

    def advance(self, timestamp, cash, *, stock_event_count=0):
        now = _time(timestamp)
        if self._last_time is not None and now < self._last_time:
            raise ValueError("Cash ledger cannot move backwards in time")
        due = []
        for position in self.positions:
            for kind in ("principal", "interest"):
                if not position[f"{kind}_released"] and _time(position[f"{kind}_release_at"]) <= now:
                    due.append((_time(position[f"{kind}_release_at"]), 0 if kind == "principal" else 1, position, kind))
        for at, _, position, kind in sorted(due, key=lambda row: (row[0], row[1], row[2]["position_id"])):
            if kind == "principal":
                cash = self._append(position, "principal_release", at, cash, stock_event_count,
                                    principal_delta=-position["principal"], processed_at=now)
                if position["settlement_fee"]:
                    cash = self._append(position, "settlement_fee", at, cash, stock_event_count,
                                        fee=position["settlement_fee"], processed_at=now)
            else:
                cash = self._append(position, "interest_payment", at, cash, stock_event_count,
                                    interest=position["gross_interest"], tax=position["interest_tax"], processed_at=now)
            position[f"{kind}_released"] = True
        self._last_time = now
        return float(cash)

    def invest(self, day, cash, *, stock_event_count=0):
        at = self.execution_at(day)
        cash = self.advance(at, cash, stock_event_count=stock_event_count)
        date = at.strftime("%Y-%m-%d")
        if any(row["date"] == date for row in self.decisions):
            raise ValueError("Cash contract may execute only once per dated quote")
        quote = self.quote_by_date.get(date)
        decision = {"date": date, "execution_at": at.isoformat(), "instrument": self.contract["instrument"]}
        if quote is None:
            self.decisions.append({**decision, "status": "data_limited", "reason": "missing_identified_executable_quote"})
            return cash
        if not (_time(quote["observed_at"]) <= at and _time(quote["available_at"]) <= at <= _time(quote["valid_until"])):
            self.decisions.append({**decision, "status": "data_limited", "reason": "quote_not_visible_or_expired_at_fixed_execution",
                                   "quote_id": quote["quote_id"], "source_sha256": quote["source_sha256"]})
            return cash
        budget = max(Decimal(0), Decimal(str(cash)) - _money(self.contract["reserve_cash"])) * Decimal(str(self.contract["allocation_fraction"]))
        increment = _money(quote["principal_increment"])
        rate = Decimal(str(quote["entry_fee_rate"]))
        principal = min(_money(quote["maximum_principal"]), budget / (1 + rate))
        principal = (principal / increment).to_integral_value(rounding=ROUND_DOWN) * increment
        while principal >= _money(quote["minimum_principal"]):
            fee = max(_money(quote["minimum_entry_fee"]), _money(principal * rate))
            if principal + fee <= budget:
                break
            principal -= increment
        if principal < _money(quote["minimum_principal"]):
            self.decisions.append({**decision, "status": "not_invested", "reason": "insufficient_unlocked_cash",
                                   "quote_id": quote["quote_id"]})
            return cash
        days = (_day(quote["accrual_end_date"]) - _day(quote["accrual_start_date"])).days
        interest = _money(principal * Decimal(str(quote["annual_rate"])) * days / quote["day_count"])
        tax = _money(max(Decimal(0), interest) * Decimal(str(quote["interest_tax_rate"])))
        position = {"position_id": _digest([self.contract_sha256, quote["quote_id"], at.isoformat()]),
                    "instrument": self.contract["instrument"], "quote_id": quote["quote_id"],
                    "source_sha256": quote["source_sha256"], "execution_at": at.isoformat(),
                    "principal": float(principal), "entry_fee": float(fee), "gross_interest": float(interest),
                    "interest_tax": float(tax), "settlement_fee": quote["settlement_fee"],
                    "principal_release_at": quote["principal_release_at"], "interest_release_at": quote["interest_release_at"],
                    "principal_released": False, "interest_released": False, "accrual_days": days,
                    "annual_rate": quote["annual_rate"], "day_count": quote["day_count"]}
        self.positions.append(position)
        cash = self._append(position, "principal_lock", at, cash, stock_event_count, principal_delta=principal)
        if fee:
            cash = self._append(position, "entry_fee", at, cash, stock_event_count, fee=fee)
        self.decisions.append({**decision, "status": "invested", "quote_id": quote["quote_id"],
                               "position_id": position["position_id"], "principal": float(principal)})
        return cash

    def report(self):
        limited = not self.decisions or any(row["status"] == "data_limited" for row in self.decisions)
        return {"cash_contract": self.contract, "cash_contract_sha256": self.contract_sha256,
                "cash_quotes_sha256": self.quotes_sha256, "cash_quote_sources": self.quotes,
                "cash_instrument_ledger": self.ledger, "cash_instrument_positions": self.positions,
                "cash_instrument_decisions": self.decisions,
                "cash_income_status": "data_limited" if limited else "contractual_settlement_only",
                "cash_income_amount": sum(row["income"] for row in self.ledger),
                "cash_gross_interest_amount": sum(row["gross_interest"] for row in self.ledger),
                "cash_instrument_fees": sum(row["fee"] for row in self.ledger),
                "cash_instrument_tax": sum(row["tax"] for row in self.ledger),
                "cash_locked_principal": self.principal,
                "cash_unpaid_contract_interest": sum(row["gross_interest"] - row["interest_tax"]
                                                      for row in self.positions if not row["interest_released"]),
                "cash_valuation_basis": "principal_at_cost; interest_recognized_only_when_contractually_paid",
                "cash_execution_model": "fixed_after_stock_close_time_executable_lend_quote; no_broker_orders"}


def audit_cash_research(result):
    """Recompute source terms, principal inventory and shared cash independently."""
    errors = []
    if _digest(result["cash_contract"]) != result["cash_contract_sha256"]:
        errors.append("cash_contract_hash")
    quotes = {row["quote_id"]: row for row in result["cash_quote_sources"]}
    if (len(quotes) != len(result["cash_quote_sources"])
            or _digest([{key: row[key] for key in ("quote_id", "source_sha256")}
                        for row in result["cash_quote_sources"]]) != result["cash_quotes_sha256"]):
        errors.append("cash_quotes_identity")
    for quote in quotes.values():
        payload = Path(quote["source_path"]).read_bytes()
        if (hashlib.sha256(payload).hexdigest() != quote["source_sha256"]
                or json.loads(payload.decode("utf-8-sig")) != {key: quote[key] for key in QUOTE_FIELDS}):
            errors.append("cash_quote_source_binding")
    positions = {row["position_id"]: row for row in result["cash_instrument_positions"]}
    cash, principal = float(result["initial_capital"]), 0.0
    stock_events = result.get("events", [])
    stock_cursor, observed = 0, {}
    # The stock engine has no intraday fill timestamps. Its conservative cash
    # contract therefore admits payments only before the opening or closing
    # phase. Derive these bounds from the stock phases, independently of the
    # cash ledger's claimed checkpoints.
    phase_bounds = {"corporate_action": ("00:00:00", "00:00:00"),
                    "open": ("09:30:00", "09:30:00"),
                    "intraday": ("09:30:00", "15:00:00"),
                    "close": ("15:00:00", "15:00:00")}
    stock_bounds = []
    previous_stock_key = None
    for event in stock_events:
        phase = event.get("execution_phase")
        if phase not in phase_bounds:
            errors.append("unknown_stock_execution_phase")
            stock_bounds.append(None)
            continue
        day = _day(event["date"])
        key = (day, list(phase_bounds).index(phase))
        if previous_stock_key is not None and key < previous_stock_key:
            errors.append("stock_execution_chronology")
        previous_stock_key = key
        stock_bounds.append(tuple(_time(f"{event['date']}T{clock}+08:00") for clock in phase_bounds[phase]))
    previous_processed_at = None
    def equal(actual, expected, name):
        if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-5:
            errors.append(name)
    def stock_flow(event):
        nonlocal cash
        equal(event["cash_before"], cash, "stock_cash_before")
        cash += float(event.get("cash_change_amount_raw", event.get("cash_change_amount", 0)))
        equal(event["cash_after"], cash, "stock_cash_after")
    for row in result["cash_instrument_ledger"]:
        checkpoint = row["stock_event_count"]
        if type(checkpoint) is not int or not stock_cursor <= checkpoint <= len(stock_events):
            errors.append("invalid_stock_checkpoint")
            continue
        processed_at = _time(row["processed_at"])
        if previous_processed_at is not None and processed_at < previous_processed_at:
            errors.append("cash_processing_chronology")
        previous_processed_at = processed_at
        preceding = stock_bounds[checkpoint - 1] if checkpoint else None
        following = stock_bounds[checkpoint] if checkpoint < len(stock_bounds) else None
        if ((preceding is not None and processed_at < preceding[1])
                or (following is not None and processed_at > following[0])):
            errors.append("cash_stock_checkpoint_chronology")
        while stock_cursor < checkpoint:
            stock_flow(stock_events[stock_cursor])
            stock_cursor += 1
        position = positions.get(row["position_id"])
        if position is None or position["quote_id"] not in quotes:
            errors.append("missing_position_quote")
            continue
        quote = quotes[position["quote_id"]]
        if any(quote[field] != result["cash_contract"][field] for field in ("instrument", "instrument_type", "currency")):
            errors.append("cash_instrument_selection")
        equal(row["cash_before"], cash, "cash_contract_before")
        actions = observed.setdefault(row["position_id"], set())
        if row["action"] in actions:
            errors.append("duplicate_cash_action")
        actions.add(row["action"])
        days = (pd.Timestamp(quote["accrual_end_date"]) - pd.Timestamp(quote["accrual_start_date"])).days
        amount = Decimal(str(position["principal"]))
        if (amount < Decimal(str(quote["minimum_principal"])) or amount > Decimal(str(quote["maximum_principal"]))
                or amount % Decimal(str(quote["principal_increment"]))):
            errors.append("cash_principal_contract_limits")
        execution = _time(position["execution_at"])
        if _time(quote["interest_release_at"]).tz_localize(None) < _day(quote["accrual_end_date"]):
            errors.append("interest_paid_before_accrual_end")
        if (not _time(quote["observed_at"]) <= _time(quote["available_at"]) <= execution <= _time(quote["valid_until"])
                or execution.strftime("%Y-%m-%d") != quote["trading_date"]
                or execution.strftime("%H:%M:%S") != result["cash_contract"]["execution_time"]):
            errors.append("cash_quote_not_visible_at_execution")
        expected_interest = float((amount * Decimal(str(quote["annual_rate"])) * days / quote["day_count"]).quantize(MONEY, rounding=ROUND_HALF_UP))
        expected_fee = float(max(Decimal(str(quote["minimum_entry_fee"])),
                                 (amount * Decimal(str(quote["entry_fee_rate"]))).quantize(MONEY, rounding=ROUND_HALF_UP)))
        expected_tax = float((Decimal(str(max(0., expected_interest))) * Decimal(str(quote["interest_tax_rate"]))).quantize(MONEY, rounding=ROUND_HALF_UP))
        for field, expected_value in (("gross_interest", expected_interest), ("entry_fee", expected_fee),
                                      ("interest_tax", expected_tax), ("accrual_days", days)):
            equal(position[field], expected_value, f"position_{field}")
        if row["action"] == "principal_lock":
            budget = max(Decimal(0), Decimal(str(cash)) - Decimal(str(result["cash_contract"]["reserve_cash"]))) * Decimal(str(result["cash_contract"]["allocation_fraction"]))
            increment = Decimal(str(quote["principal_increment"]))
            next_amount = amount + increment
            next_fee = max(Decimal(str(quote["minimum_entry_fee"])),
                           (next_amount * Decimal(str(quote["entry_fee_rate"]))).quantize(MONEY, rounding=ROUND_HALF_UP))
            if (amount + Decimal(str(expected_fee)) > budget
                    or (next_amount <= Decimal(str(quote["maximum_principal"])) and next_amount + next_fee <= budget)):
                errors.append("cash_allocation_not_fixed_contract")
        expected = {"principal_lock": (position["principal"], 0., 0., 0., position["execution_at"]),
                    "entry_fee": (0., 0., expected_fee, 0., position["execution_at"]),
                    "principal_release": (-position["principal"], 0., 0., 0., quote["principal_release_at"]),
                    "settlement_fee": (0., 0., quote["settlement_fee"], 0., quote["principal_release_at"]),
                    "interest_payment": (0., expected_interest, 0., expected_tax, quote["interest_release_at"])}.get(row["action"])
        if expected is None:
            errors.append("unknown_cash_action")
            continue
        for field, value in zip(("principal_delta", "gross_interest", "fee", "tax"), expected[:4]):
            equal(row[field], value, field)
        if (_time(row["timestamp"]) != _time(expected[4]) or row["source_sha256"] != quote["source_sha256"]
                or _time(row["processed_at"]) < _time(row["timestamp"])):
            errors.append("cash_source_or_settlement_time")
        expected_change = -expected[0] + expected[1] - expected[2] - expected[3]
        equal(row["cash_change"], expected_change, "cash_change")
        principal += expected[0]
        cash += expected_change
        equal(row["cash_after"], cash, "cash_contract_after")
        if cash < -1e-8 or principal < -1e-8:
            errors.append("negative_cash_or_principal")
    for event in stock_events[stock_cursor:]:
        stock_flow(event)
    equal(result["final_cash"], cash, "final_cash")
    equal(result["cash_locked_principal"], principal, "locked_principal")
    for position_id, position in positions.items():
        actions = observed.get(position_id, set())
        expected_actions = {"principal_lock"}
        if position["entry_fee"]:
            expected_actions.add("entry_fee")
        if position["principal_released"]:
            expected_actions.add("principal_release")
            if position["settlement_fee"]:
                expected_actions.add("settlement_fee")
        if position["interest_released"]:
            expected_actions.add("interest_payment")
        if actions != expected_actions:
            errors.append("cash_position_ledger_incomplete")
    dated_flows = {}
    for row in stock_events:
        flows = dated_flows.setdefault(row["date"], [0., 0.])
        flows[0] += float(row.get("cash_change_amount_raw", row.get("cash_change_amount", 0)))
    for row in result["cash_instrument_ledger"]:
        flows = dated_flows.setdefault(_time(row["processed_at"]).strftime("%Y-%m-%d"), [0., 0.])
        flows[0] += row["cash_change"]
        flows[1] += row["principal_delta"]
    dates = sorted(dated_flows)
    cursor, daily_cash, daily_principal = 0, float(result["initial_capital"]), 0.
    for point in result.get("equity_curve", []):
        while cursor < len(dates) and dates[cursor] <= point["date"]:
            daily_cash += dated_flows[dates[cursor]][0]
            daily_principal += dated_flows[dates[cursor]][1]
            cursor += 1
        equal(point["cash"], daily_cash, "daily_available_cash")
        equal(point["cash_instrument_principal"], daily_principal, "daily_locked_principal")
        expected = point["cash"] + point["dividend_receivable"] + point["stock_market_value"] + point["cash_instrument_principal"]
        equal(point["equity"] * result["initial_capital"], expected, "equity_components")
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "cash_contract_entries": len(result["cash_instrument_ledger"]), "stock_entries": len(stock_events),
            "recomputed_final_cash": cash, "recomputed_locked_principal": principal,
            "scope": "source_contract_terms_cash_flows_principal_inventory_and_equity_components; stock_share_audit_separate"}
