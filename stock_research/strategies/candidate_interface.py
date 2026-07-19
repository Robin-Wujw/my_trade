"""Canonical interface between every selection model and the trade engine."""
from __future__ import annotations

import math


MAX_DAILY_CANDIDATES = 60
MIN_CORE_DAILY_CANDIDATES = 0
FACTOR_QUANT_SOURCES = {"factor_quant", "quant_right"}
LEFT_VALUE_EXPLOSIVE_GROWTH_YOY = 1.0
LEFT_VALUE_SMALL_CAP_BUFFER = 150.0
LEFT_VALUE_DEEP_DISCOUNT_RATIO = 0.90
LEFT_VALUE_MIN_PRICE_TO_VALUE = 0.80
LEFT_VALUE_MAX_PRICE_TO_VALUE = 1.08


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _append_failure_reason(row, reason: str) -> None:
    if not reason:
        return
    existing = _clean_text(row.get("candidate_failure_reason"))
    parts = [item.strip() for item in existing.split(";") if item.strip()]
    if reason not in parts:
        parts.append(reason)
    row["candidate_failure_reason"] = ";".join(parts)


def left_value_safety_reasons(row) -> list[str]:
    """Return visible-data reasons why a value-line row is not left executable."""
    price_to_value = _number(row.get("price_to_value"))
    growth = _number(row.get("earnings_yoy", row.get("yoy")))
    market_cap = _number(row.get("mktcap"))
    if price_to_value is None or growth is None or market_cap is None:
        return []
    if (
        growth >= LEFT_VALUE_EXPLOSIVE_GROWTH_YOY
        and market_cap < LEFT_VALUE_SMALL_CAP_BUFFER
        and price_to_value > LEFT_VALUE_DEEP_DISCOUNT_RATIO
    ):
        return ["left_high_growth_small_cap_needs_deeper_discount"]
    return []


def left_value_permission_reasons(row) -> list[str]:
    """Return reasons why a value-model row cannot open a left-side plan."""
    reasons = []
    if _number(row.get("quality_score")) is None:
        reasons.append("quality_score_missing")
    if _number(row.get("earnings_yoy", row.get("yoy"))) is None:
        reasons.append("earnings_yoy_missing")
    if _number(row.get("mktcap")) is None:
        reasons.append("mktcap_missing")
    price_to_value = _number(row.get("price_to_value"))
    if price_to_value is None:
        reasons.append("price_to_value_missing")
    elif not LEFT_VALUE_MIN_PRICE_TO_VALUE <= price_to_value <= LEFT_VALUE_MAX_PRICE_TO_VALUE:
        reasons.append("price_to_value_out_of_range")
    if reasons:
        return reasons
    return left_value_safety_reasons(row)


def normalize_candidate(candidate):
    """Return the one candidate schema consumed by portfolio execution.

    Selection models may retain their own diagnostic columns, but they cannot
    select an execution path.  Every eligible row enters the same price-signal,
    position and exit engine.
    """
    row = dict(candidate)
    row["code"] = str(row.get("code") or "")
    row["name"] = str(row.get("name") or row["code"])
    quality = _number(row.get("quality_score"))
    growth = _number(row.get("earnings_yoy", row.get("yoy")))
    market_cap = _number(row.get("mktcap"))
    leadership_score = _number(row.get("leadership_score")) or 0.0
    original_score = _number(row.get("fundamental_score"))
    if original_score is None:
        original_score = _number(row.get("candidate_score")) or 0.0
    source = _clean_text(row.get("candidate_source"))
    sources = {item for item in source.split("+") if item}
    if sources:
        row["allow_left"] = "value_model" in sources
        row["allow_right"] = bool(
            sources & {
                "standard_mainline", "growth_leadership",
                "quant_right", "factor_quant",
            }
        )
    else:
        row["allow_left"] = bool(row.get("allow_left", False))
        row["allow_right"] = bool(row.get("allow_right", True))
    left_permission_reasons = (
        left_value_permission_reasons(row)
        if "value_model" in sources or row["allow_left"] else []
    )
    if left_permission_reasons:
        row["allow_left"] = False
        for reason in left_permission_reasons:
            _append_failure_reason(row, reason)
    if quality is not None and growth is not None:
        mainline_bonus = 0.0
        valuation_bonus = 0.0
        price_to_value = _number(row.get("price_to_value"))
        if price_to_value is not None and 0.80 <= price_to_value <= 1.08:
            valuation_bonus = max(0.0, 5.0 * (1.08 - price_to_value) / 0.28)
        trade_basis_bonus = _number(row.get("trade_basis_score")) or 0.0
        right_quant_score = _number(row.get("right_quant_score"))
        right_quant_bonus = (
            min(max(right_quant_score, 0.0) * 0.35, 30.0)
            if right_quant_score is not None else 0.0
        )
        core_score = (
            quality
            + min(max(growth, 0.0), 1.0) * 20.0
            + mainline_bonus
            + valuation_bonus
            + min(max(trade_basis_bonus, 0.0), 12.0)
        )
        row["core_candidate_score"] = round(core_score, 6)
        original_score = core_score + max(
            min(max(leadership_score, 0.0), 30.0),
            right_quant_bonus,
        )
        if sources & FACTOR_QUANT_SOURCES and right_quant_score is not None:
            quant_score = max(float(right_quant_score), 0.0)
            setup_bonus = 5.0 if _clean_text(row.get("right_quant_setup")) == "高盈亏比" else 0.0
            row["core_candidate_score"] = round(quant_score + setup_bonus, 6)
            original_score = max(original_score, quant_score + setup_bonus)
    row["candidate_score"] = round(original_score, 6)
    row["value_falsification_reason"] = _clean_text(
        row.get("value_falsification_reason")
    )
    row["candidate_failure_reason"] = _clean_text(
        row.get("candidate_failure_reason")
    )
    row["selected_for_trading"] = _truthy(
        row.get("selected_for_trading", True)
    )
    row["value_falsified"] = _truthy(row.get("value_falsified")) or bool(
        row["value_falsification_reason"]
    )
    row["selection_reason"] = _clean_text(
        row.get("selection_reason")
        or row.get("strategy_part")
        or row.get("candidate_source")
        or "选股模型入选"
    )
    eligible = row.get("signal_eligible")
    if eligible is None:
        eligible = True
    if isinstance(eligible, str):
        eligible = eligible.strip().lower() in {"1", "true", "yes", "y"}
    eligibility_reasons = []
    requires_full_fundamentals = bool(
        sources & {
            "standard_mainline", "growth_leadership",
            "quant_right", "factor_quant",
        }
    )
    if requires_full_fundamentals and quality is None:
        eligibility_reasons.append("quality_score_missing")
        eligible = False
    if quality is not None:
        if quality < 70.0:
            eligibility_reasons.append("quality_score_below_70")
        eligible = eligible and quality >= 70.0
    if requires_full_fundamentals and growth is None:
        eligibility_reasons.append("earnings_yoy_missing")
        eligible = False
    if growth is not None:
        if growth < 0.10:
            eligibility_reasons.append("earnings_yoy_below_10pct")
        eligible = eligible and growth >= 0.10
    requires_market_cap = bool(sources) or quality is not None or growth is not None
    if requires_market_cap:
        if market_cap is None:
            eligibility_reasons.append("mktcap_missing")
        elif market_cap < 100.0:
            eligibility_reasons.append("mktcap_below_100")
        eligible = eligible and market_cap is not None and market_cap >= 100.0
    elif market_cap is not None:
        if market_cap < 100.0:
            eligibility_reasons.append("mktcap_below_100")
        eligible = eligible and market_cap >= 100.0
    data_status = _clean_text(row.get("data_status"))
    if data_status and data_status != "traded":
        eligibility_reasons.append(f"data_status_{data_status}")
        eligible = False
    valid_price_bar = row.get("valid_price_bar")
    if valid_price_bar is not None and not _truthy(valid_price_bar):
        eligibility_reasons.append("invalid_price_bar")
        eligible = False
    is_traded_bar = row.get("is_traded_bar")
    if is_traded_bar is not None and not _truthy(is_traded_bar):
        eligibility_reasons.append("not_traded_bar")
        eligible = False
    for reason in eligibility_reasons:
        _append_failure_reason(row, reason)
    if not row["allow_left"] and not row["allow_right"]:
        eligible = False
        _append_failure_reason(row, "no_executable_lane")
    row["signal_eligible"] = bool(eligible)
    return row


def _is_diagnostic_row(candidate):
    return bool(
        candidate.get("candidate_failure_reason")
        or candidate.get("value_falsification_reason")
        or candidate.get("value_falsified")
        or candidate.get("selected_for_trading") is False
    )


def normalize_candidate_snapshots(snapshots, *, include_diagnostics=False):
    result = {}
    for date, rows in snapshots.items():
        normalized = [normalize_candidate(item) for item in rows]
        eligible = [
            item for item in normalized
            if item["signal_eligible"] and item["selected_for_trading"]
        ]
        eligible.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        selected = []
        selected_codes = set()
        for item in eligible:
            if len(selected) >= MAX_DAILY_CANDIDATES:
                break
            if item["code"] not in selected_codes:
                selected.append(item)
                selected_codes.add(item["code"])
        selected.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        for rank, item in enumerate(selected, 1):
            item["selection_rank"] = rank
            item["selected_for_trading"] = True
        if include_diagnostics:
            selected_codes = {item["code"] for item in selected}
            diagnostics = []
            for item in normalized:
                if item["code"] in selected_codes or not _is_diagnostic_row(item):
                    continue
                item = dict(item)
                item["signal_eligible"] = False
                item["selected_for_trading"] = False
                item["selection_rank"] = None
                diagnostics.append(item)
            result[date] = selected + sorted(
                diagnostics, key=lambda item: item["code"],
            )
        else:
            result[date] = selected
    return result
