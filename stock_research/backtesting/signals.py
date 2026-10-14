"""Point-in-time signals and semantic entry gates, independent of portfolio state."""
from __future__ import annotations

import math
import re

import pandas as pd

from stock_research.indicators.price_structure import configured_price_structures, infer_price_structures
from stock_research.indicators.technical_entries import (
    SUPPORT_ZONE_PCT, _valid_volume_price_nodes, apply_entry_confluence, infer_technical_entry,
)

MIN_DAILY_RIGHT_RISK_PCT = 0.01
MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT = 0.018
PULLBACK_PILOT_BASE_SIZE = 0.08
ENTRY_PRIORITY_MODE_LEGACY = "legacy_weighted"
ENTRY_PRIORITY_MODE_STRUCTURE = "structure_lexicographic"


def build_formula_phase_history(formula_rows) -> dict[str, str]:
    frame = pd.DataFrame(formula_rows).copy()
    if frame.empty or "date" not in frame:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    phase = "waiting"
    result = {}

    def streak(value):
        number = pd.to_numeric(value, errors="coerce")
        return 0 if pd.isna(number) else int(number)

    for _, row in frame.iterrows():
        up = streak(row.get("window_up_streak"))
        down = streak(row.get("window_down_streak"))
        if down >= 5:
            phase = "exited"
        elif up >= 5:
            phase = "active"
        elif up >= 3 and phase != "active":
            phase = "watch"
        result[row["date"].strftime("%Y-%m-%d")] = phase
    return result


def _enabled(value, *, default=True):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clean_trade_basis_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    return str(value).replace(
        "距离21日收盘高点3%以内",
        "候选观察：接近21日收盘高点（非买点）",
    )


def _minimum_trimmed_entry_size(
    planned_size,
    *,
    starting_new_right_campaign,
    pullback_pilot,
):
    planned_size = max(0.0, float(planned_size))
    if pullback_pilot:
        minimum = PULLBACK_PILOT_BASE_SIZE
    elif starting_new_right_campaign:
        minimum = 0.20
    else:
        minimum = 0.10
    return min(planned_size, minimum)


def _active_profit_trigger_ids(profit, history, remaining_parts) -> set[str]:
    """Map source-backed protection signals without treating examples as a checklist."""
    maximum_return = float(profit.get("maximum_return_pct") or 0.0)
    if maximum_return < 10:
        return set()
    ids = set()
    if profit.get("profit_floor_triggered"):
        ids.add("profit_floor")
    if maximum_return >= 20 and profit.get("trailing_10_triggered"):
        ids.add("trailing_10")
    if profit.get("divergence_time_take_profit"):
        ids.add("divergence_time")

    if (
        maximum_return >= 20
        and len(history) >= 14
        and {"date", "volume", "low"}.issubset(history.columns)
    ):
        prior_history = history.iloc[:-1]
        valid_nodes = _valid_volume_price_nodes(prior_history)
        if valid_nodes:
            prior_close = float(prior_history.iloc[-1]["close"])
            supporting_nodes = [
                node for node in valid_nodes
                if float(node["support"]) <= prior_close
            ]
            if supporting_nodes:
                closest_node = max(
                    supporting_nodes, key=lambda node: float(node["support"]),
                )
                if float(history.iloc[-1]["low"]) < float(closest_node["support"]):
                    ids.add(f"volume_node_break:{closest_node['date']}")

    close = float(profit.get("close") or 0.0)
    latest_ma20 = pd.to_numeric(history.iloc[-1].get("ma20"), errors="coerce")
    prior_ma20 = (
        pd.to_numeric(history.iloc[-6].get("ma20"), errors="coerce")
        if len(history) >= 6 else pd.NA
    )
    if (
        pd.notna(latest_ma20)
        and pd.notna(prior_ma20)
        and float(latest_ma20) > float(prior_ma20)
        and close < float(latest_ma20) * 0.97
    ):
        ids.add("rising_ma20_break_3pct")

    if (
        int(remaining_parts) == 1
        and maximum_return >= 20
        and profit.get("half_profit_triggered")
    ):
        ids.add("maximum_profit_half")
    return ids


def _price_structure_signal(
    data, index, plan=None, *, auto_structure=True, allow_pullback=True,
):
    if index < 40:
        return None
    row = data.iloc[index]
    previous = data.iloc[:index]
    prior_close = float(previous.iloc[-1]["close"])
    volume_baseline = max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )
    close_volume_ratio = (
        float(row["volume"]) / volume_baseline if volume_baseline > 0 else 0.0
    )

    def gap_up_rank(base_rank: int) -> tuple[int, bool]:
        prior_high = pd.to_numeric(previous.iloc[-1].get("high"), errors="coerce")
        day_open = pd.to_numeric(row.get("open"), errors="coerce")
        gap_up = bool(
            pd.notna(prior_high)
            and pd.notna(day_open)
            and float(day_open) > float(prior_high)
        )
        return base_rank, gap_up

    structures = configured_price_structures(plan)
    if auto_structure:
        inferred_structures = infer_price_structures(previous)
        structures.extend(inferred_structures)
        for inferred in (
            item for item in inferred_structures
            if item["kind"] == "uptrend_anchor"
        ):
            # Ratios of the preceding uptrend remain pullback candidates. They
            # become executable only when a pre-known moving average overlaps
            # the level.  Auto-inferred supports must be trend-level waves, not
            # short relay swings that only look good after sorting by the
            # highest nearby level.
            if not inferred.get("amplitude_valid"):
                continue
            # All three ratios stay available at their useful ages.
            maximum_days = {0.75: 8, 0.625: 21, 0.50: 34}
            age = int(inferred.get("bars_since_high") or 0)
            prior = previous.iloc[-1]
            for ratio, level in inferred["uptrend_levels"].items():
                if age > maximum_days[ratio]:
                    continue
                confluence = []
                for name in ("ma5", "ma10", "ma20", "ma60"):
                    value = pd.to_numeric(prior.get(name), errors="coerce")
                    if pd.notna(value) and abs(float(value) / float(level) - 1) <= SUPPORT_ZONE_PCT:
                        confluence.append(name.upper())
                if confluence:
                    structures.append({
                        "kind": "uptrend_support", "ratio": ratio,
                        "level": float(level), "confluence": confluence,
                        "uptrend_low": inferred["uptrend_low"],
                        "uptrend_high": inferred["uptrend_high"],
                        "uptrend_low_date": inferred["uptrend_low_date"],
                        "uptrend_high_date": inferred["uptrend_high_date"],
                        "amplitude_valid": inferred.get("amplitude_valid"),
                    })

    # A close-confirmed breach of the uptrend U50 followed by a reclaim is a
    # trend-restoration entry.  It is not the H-P pullback-half breakout.
    reclaim_candidates = []
    for structure in structures:
        if structure.get("kind") != "uptrend_anchor":
            continue
        if not structure.get("amplitude_valid"):
            continue
        trigger = float(structure["uptrend_levels"][0.50])
        high_date = pd.Timestamp(structure["uptrend_high_date"])
        after_high = previous[previous["date"] > high_date]
        effective_breach = bool((after_high["close"] < trigger).any())
        if (
            effective_breach
            and prior_close <= trigger < float(row["close"])
            and float(row["close"]) <= trigger * (1 + SUPPORT_ZONE_PCT)
            and close_volume_ratio >= 1.0
        ):
            reclaim_candidates.append((trigger, structure))
    if reclaim_candidates:
        trigger, structure = max(reclaim_candidates, key=lambda item: item[0])
        rank, gap_up = gap_up_rank(5)
        reason = "上涨波段50%有效跌破后收盘放量收复"
        if gap_up:
            reason += "; 跳空向上加分"
        return {
            "rank": rank, "stop": trigger, "trigger": trigger,
            "order_type": "close", "reason": reason,
            "known_volume_ratio": close_volume_ratio,
            "requires_next_day_confirmation": True,
            "structure_ratio": 0.50,
            "signal_type": "uptrend_50_reclaim", "gap_up": gap_up,
            "anchor_low": structure["uptrend_low"],
            "anchor_high": structure["uptrend_high"],
            "anchor_low_date": structure["uptrend_low_date"],
            "anchor_high_date": structure["uptrend_high_date"],
        }

    # Pullback-half breakout is a separate three-anchor structure. It is valid
    # only after the same uptrend's 62.5% was breached, the trend-level span is
    # wide enough, and consolidation has lasted at least 13 trading days.
    for structure in structures:
        if structure["kind"] != "pullback_recovery":
            continue
        if not (
            structure.get("deep_pullback_confirmed")
            and structure.get("amplitude_valid")
            and int(structure.get("consolidation_days") or 0) >= 13
        ):
            continue
        trigger = float(structure["recovery_half"])
        if (
            prior_close <= trigger < float(row["close"])
            and float(row["close"]) <= trigger * (1 + SUPPORT_ZONE_PCT)
            and close_volume_ratio >= 1.0
        ):
            rank, gap_up = gap_up_rank(5)
            reason = "回调波段50%收盘放量向上突破"
            if gap_up:
                reason += "; 跳空向上加分"
            return {
                "rank": rank, "stop": trigger, "trigger": trigger,
                "order_type": "close", "reason": reason,
                "known_volume_ratio": close_volume_ratio,
                "requires_next_day_confirmation": True,
                "structure_ratio": 0.50,
                "signal_type": "pullback_50_breakout", "gap_up": gap_up,
                "anchor_low": structure.get("uptrend_low"),
                "anchor_high": structure.get("uptrend_high"),
                "anchor_pullback_low": structure.get("pullback_low"),
                "anchor_low_date": structure.get("uptrend_low_date"),
                "anchor_high_date": structure.get("uptrend_high_date"),
                "anchor_pullback_low_date": structure.get("pullback_low_date"),
            }

    # U75/U625 are pullback-only and require declared technical confluence.
    # They are lower priority than the author's two right-side breakout entries.
    if allow_pullback:
        supports = sorted(
            (item for item in structures if item["kind"] == "uptrend_support"),
            key=lambda item: item["level"], reverse=True,
        )
        for structure in supports:
            if not structure.get("amplitude_valid"):
                continue
            level = float(structure["level"])
            ratio = float(structure["ratio"])
            zone_top = level * (1 + SUPPORT_ZONE_PCT)
            if (
                prior_close > level
                and float(row["low"]) <= zone_top
                and float(row["close"]) >= level
                and float(row["close"]) <= zone_top
            ):
                confluence = ",".join(map(str, structure["confluence"]))
                return {
                    "rank": 4, "stop": level, "trigger": float(row["close"]),
                    "order_type": "close",
                    "reason": f"上涨波段{ratio:.1%}拉回支撑5%区域; 共振={confluence}",
                    "known_volume_ratio": 1.0, "structure_ratio": ratio,
                    "signal_type": "uptrend_support_pullback",
                    "support_confluence": list(structure["confluence"]),
                    "anchor_low": structure.get("uptrend_low"),
                    "anchor_high": structure.get("uptrend_high"),
                    "anchor_low_date": structure.get("uptrend_low_date"),
                    "anchor_high_date": structure.get("uptrend_high_date"),
                }
    return None


def _right_signal(
    data, index, plan=None, *, auto_price_structure=True,
    allow_structure_pullback=True,
):
    if index < 60:
        return None
    structure_signal = _price_structure_signal(
        data, index, plan, auto_structure=auto_price_structure,
        allow_pullback=allow_structure_pullback,
    )
    if structure_signal:
        structure_signal = apply_entry_confluence(data, index, structure_signal)
    if configured_price_structures(plan) and structure_signal:
        return structure_signal
    technical_signal = infer_technical_entry(data, index)
    return max(
        (signal for signal in (structure_signal, technical_signal) if signal),
        key=lambda signal: signal["rank"],
        default=None,
    )


def _leader_trend_add_signal(data, index):
    if index < 80:
        return None
    row = data.iloc[index]
    previous = data.iloc[:index]
    prior = previous.iloc[-1]
    close_high = float(previous["close"].tail(21).max())
    volume_baseline = max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )
    volume_ratio = float(row["volume"]) / volume_baseline if volume_baseline > 0 else 0.0
    ma20 = pd.to_numeric(prior.get("ma20"), errors="coerce")
    ma60 = pd.to_numeric(prior.get("ma60"), errors="coerce")
    prior_ma20 = pd.to_numeric(previous.iloc[-6].get("ma20"), errors="coerce")
    if (
        pd.isna(ma20)
        or pd.isna(ma60)
        or pd.isna(prior_ma20)
        or close_high <= 0
    ):
        return None
    if not (
        float(prior["close"]) <= close_high < float(row["close"])
        and float(row["close"]) <= close_high * 1.10
        and float(prior["close"]) > float(ma20) >= float(ma60)
        and float(ma20) > float(prior_ma20)
        and volume_ratio >= 0.85
    ):
        return None
    stop = max(close_high * 0.95, float(ma20) * 0.98)
    target = close_high + (close_high - stop) * 3.0
    signal = {
        "rank": 5,
        "trigger": close_high,
        "stop": stop,
        "target_price": target,
        "order_type": "close",
        "reason": "proven leader trend add on 21d close high; close stop",
        "known_volume_ratio": volume_ratio,
        "signal_type": "leader_trend_add",
        "requires_next_day_confirmation": True,
    }
    return apply_entry_confluence(data, index, signal)


def _candidate_number(candidate, key):
    if not candidate:
        return None
    value = pd.to_numeric(candidate.get(key), errors="coerce")
    return float(value) if pd.notna(value) else None


def _candidate_first_number(candidate, *keys):
    for key in keys:
        value = _candidate_number(candidate, key)
        if value is not None:
            return value
    return None


def _channel_candidate_number(candidate, channel, key):
    return _candidate_first_number(candidate, f"{channel}_{key}", key)


def _signal_number(signal, key):
    if not signal:
        return None
    value = pd.to_numeric(signal.get(key), errors="coerce")
    return float(value) if pd.notna(value) else None


def _right_entry_priority_sort_key(item, frames, *, mode):
    """Return a deterministic same-day cash-allocation priority key."""
    (
        legacy_priority, candidate_score, signal_rank, code, index, signal,
        candidate, _,
    ) = item
    if mode == ENTRY_PRIORITY_MODE_LEGACY:
        return (
            float(legacy_priority), float(candidate_score),
            float(signal_rank), str(code), int(index),
        )
    if mode != ENTRY_PRIORITY_MODE_STRUCTURE:
        raise ValueError(f"unsupported right entry priority mode: {mode}")

    entry_evidence = _signal_number(signal, "entry_evidence_score")
    structure_proximity = _signal_number(
        signal, "structure_proximity_score",
    )
    if structure_proximity is None:
        structure_proximity = _candidate_number(
            candidate, "structure_proximity_score",
        )
    if (
        structure_proximity is None
        and not signal.get("standard_right_next_open")
        and signal.get("technical_signal_timing") != "previous_close_preplaced_order"
    ):
        value = pd.to_numeric(
            frames[str(code)].iloc[int(index)].get(
                "structure_proximity_score",
            ),
            errors="coerce",
        )
        structure_proximity = float(value) if pd.notna(value) else None
    return (
        float(signal_rank),
        float(entry_evidence) if entry_evidence is not None else -math.inf,
        (
            float(structure_proximity)
            if structure_proximity is not None else -math.inf
        ),
        float(candidate_score),
        float(legacy_priority),
        str(code),
        int(index),
    )


def _ordered_right_entry_candidates(candidates, frames, *, mode):
    return sorted(
        candidates,
        key=lambda item: _right_entry_priority_sort_key(
            item, frames, mode=mode,
        ),
        reverse=True,
    )


def _signal_support_has_fast_ma_confluence(signal) -> bool:
    if not signal:
        return True
    confluence = signal.get("support_confluence")
    if confluence is None:
        reason = str(signal.get("reason") or "")
        if "共振=" not in reason:
            return True
        confluence_text = reason.split("共振=", 1)[1]
        confluence = re.split(r"[,，、;；]+", confluence_text)
    if isinstance(confluence, str):
        items = re.split(r"[,，、;；]+", confluence)
    else:
        items = list(confluence)
    fast_ma = {"MA5", "MA10", "MA20"}
    return any(str(item).strip() in fast_ma for item in items)








def _candidate_text(candidate, key) -> str:
    if not candidate:
        return ""
    value = candidate.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _has_semantic_candidate_profile(candidate) -> bool:
    if not candidate:
        return False
    fields = {
        "candidate_source", "quality_score", "earnings_yoy", "mktcap",
        "trade_basis_score", "leadership_score", "right_quant_score",
        "return_20d", "return_60d", "avg_amount_20",
    }
    return any(candidate.get(field) not in (None, "") for field in fields)


def _factor_only_selection_profile(candidate) -> bool:
    return (
        _candidate_text(candidate, "selection_profile")
        == "quantsplaybook_factor_only"
        or any(
            _enabled(candidate.get(field), default=False)
            for field in (
                "compressed_structure_channel_eligible",
                "factor_horizon_channel_eligible",
                "breadth_reversal_channel_eligible",
            )
        )
    )


def _compact_attack_core_profile(candidate) -> bool:
    setup = _candidate_text(candidate, "right_quant_setup")
    right_quant = _candidate_number(candidate, "right_quant_score")
    structure_rank = _candidate_first_number(
        candidate, "quant_structure_rank", "right_structure_rank", "structure_rank",
    )
    low_risk_rank = _candidate_first_number(
        candidate, "quant_low_risk_rank", "right_risk_rank", "low_risk_rank",
    )
    volume_confirm_rank = _candidate_first_number(
        candidate,
        "quant_volume_confirm_rank", "right_volume_node_rank", "volume_confirm_rank",
    )
    ret60 = _candidate_number(candidate, "return_60d")
    drawdown_60 = _candidate_number(candidate, "drawdown_60")
    return bool(
        setup == "compact_attack_core"
        and right_quant is not None and right_quant >= 85.0
        and structure_rank is not None and structure_rank >= 80.0
        and low_risk_rank is not None and low_risk_rank >= 60.0
        and volume_confirm_rank is not None and volume_confirm_rank >= 60.0
        and ret60 is not None and ret60 >= 0.15
        and (drawdown_60 is None or drawdown_60 >= -0.08)
    )


def _large_liquid_trend_core_profile(candidate) -> bool:
    """Broadens strong right-side candidates without relaxing structure quality."""
    if not candidate:
        return False
    sources = set(str(candidate.get("candidate_source") or "").split("+"))
    setup = _candidate_text(candidate, "right_quant_setup")
    quality = _candidate_number(candidate, "quality_score")
    growth = _candidate_number(candidate, "earnings_yoy")
    market_cap = _candidate_number(candidate, "mktcap")
    avg_amount = _candidate_number(candidate, "avg_amount_20")
    trade_basis = _candidate_number(candidate, "trade_basis_score")
    right_quant = _candidate_number(candidate, "right_quant_score")
    ret20 = _candidate_number(candidate, "return_20d")
    ret60 = _candidate_number(candidate, "return_60d")
    drawdown_60 = _candidate_number(candidate, "drawdown_60")
    alpha_rank = _candidate_first_number(candidate, "quant_alpha_rank", "alpha_rank")
    trend_stability_rank = _candidate_first_number(
        candidate, "quant_trend_stability_rank", "trend_stability_rank",
    )
    structure_rank = _candidate_first_number(
        candidate, "quant_structure_rank", "right_structure_rank", "structure_rank",
    )
    volume_confirm_rank = _candidate_first_number(
        candidate,
        "quant_volume_confirm_rank", "right_volume_node_rank", "volume_confirm_rank",
    )
    low_risk_rank = _candidate_first_number(
        candidate, "quant_low_risk_rank", "right_risk_rank", "low_risk_rank",
    )
    return bool(
        sources & {"factor_quant", "quant_right"}
        and setup in {"强趋势", "large_liquid_trend_core"}
        and quality is not None and quality >= 75.0
        and growth is not None and growth >= 0.20
        and market_cap is not None and market_cap >= 3000.0
        and avg_amount is not None and avg_amount >= 3_000_000_000.0
        and trade_basis is not None and trade_basis >= 6.0
        and right_quant is not None and right_quant >= 85.0
        and ret20 is not None and 0.0 <= ret20 <= 0.40
        and ret60 is not None and ret60 >= 0.25
        and drawdown_60 is not None and drawdown_60 >= -0.16
        and (alpha_rank is None or alpha_rank >= 55.0)
        and (trend_stability_rank is None or trend_stability_rank >= 70.0)
        and (structure_rank is None or structure_rank >= 35.0)
        and (volume_confirm_rank is None or volume_confirm_rank >= 45.0)
        and (low_risk_rank is None or low_risk_rank >= 25.0)
    )


def _planned_right_target(data, index, signal, entry_price, risk_amount, candidate=None):
    candidates = []
    explicit = _signal_number(signal, "target_price")
    if explicit is not None:
        candidates.append((explicit, "explicit_signal_target"))
    signal_type = str(signal.get("signal_type") or "")
    trigger = _signal_number(signal, "trigger")
    stop = _signal_number(signal, "stop")
    if (
        signal_type == "leader_pivot_breakout"
        and _compact_attack_core_profile(candidate)
        and trigger is not None
        and stop is not None
        and trigger > stop > 0
    ):
        pivot_risk = trigger - stop
        candidates.append((
            trigger + pivot_risk * 3.5,
            "compact_attack_core_3_5r_pivot_continuation",
        ))
    anchor_high = _signal_number(signal, "anchor_high")
    if anchor_high is not None:
        candidates.append((anchor_high, "visible_anchor_high"))
    visible = data.iloc[:index + 1]
    if not visible.empty:
        recent_high = pd.to_numeric(
            visible.tail(120)["high"], errors="coerce",
        ).max()
        if pd.notna(recent_high):
            candidates.append((float(recent_high), "visible_120d_high"))
    valid = [
        (price, label) for price, label in candidates
        if price is not None and price > entry_price
    ]
    if valid:
        return max(valid, key=lambda item: item[0])
    return entry_price + risk_amount * 2.8, "planned_2_8r_trend_continuation"


def _trailing_amount_observation(data, index, *, amount_multiplier=1.0):
    """Return signal-day and trailing-20 turnover using only visible bars."""
    if "amount" not in data or index < 0:
        return None, None
    start = max(0, int(index) - 19)
    values = pd.to_numeric(
        data.iloc[start:int(index) + 1]["amount"],
        errors="coerce",
    )
    values = values[(values > 0) & values.notna()] * float(amount_multiplier)
    current = pd.to_numeric(data.iloc[int(index)].get("amount"), errors="coerce")
    signal_amount = (
        None
        if pd.isna(current) or float(current) <= 0
        else float(current) * float(amount_multiplier)
    )
    average_amount = float(values.mean()) if len(values) >= 20 else None
    return signal_amount, average_amount


def _semantic_right_entry_gate(
    data,
    index,
    candidate,
    signal,
    *,
    current_down_streak=0,
    management_transition=False,
    price_amount_multiplier=1.0,
    factor_only_min_avg_amount=None,
    research_expectancy=None,
):
    """Convert a technical signal into an auditable high R/R entry decision."""
    if not signal:
        return False, signal, "missing_signal"
    enriched = dict(signal)
    expected = None
    if research_expectancy is not None:
        from stock_research.research.structure_events import expected_target
        entry = _signal_number(signal, "trigger") if signal.get("order_type") != "close" else float(data.iloc[index]["close"])
        stop = _signal_number(signal, "stop")
        if entry is None or stop is None or not 0 < stop < entry:
            enriched.update(semantic_entry_ok=False, entry_gate_reason="invalid_or_non_protective_stop")
            return False, enriched, "invalid_or_non_protective_stop"
        expected = expected_target(entry, stop, research_expectancy)
        enriched["research_structure_expectancy"] = expected["expectancy"]
        if expected["status"] != "ok":
            reason = "structure_expectancy_" + expected["status"]
            enriched.update(semantic_entry_ok=False, entry_gate_reason=reason)
            return False, enriched, reason
    factor_only = _factor_only_selection_profile(candidate)
    if not factor_only and not _has_semantic_candidate_profile(candidate):
        enriched.update({
            "semantic_entry_ok": True,
            "semantic_entry_setup": "legacy_minimal_candidate_profile",
            "entry_gate_reason": "legacy_minimal_candidate_profile",
        })
        if expected is not None:
            enriched.update(entry_target_price=expected["target"], entry_target_basis="visible_forward_structure_net_expectancy",
                            entry_reward_risk=research_expectancy.mean_net_r)
        return True, enriched, "ok"

    close = float(data.iloc[index]["close"])
    entry_price = close if signal.get("order_type") == "close" else (
        _signal_number(signal, "trigger") or close
    )
    stop = _signal_number(signal, "stop")
    if entry_price <= 0 or stop is None or stop <= 0 or stop >= entry_price:
        enriched["semantic_entry_ok"] = False
        enriched["entry_gate_reason"] = "invalid_or_non_protective_stop"
        return False, enriched, "invalid_or_non_protective_stop"

    quality = _candidate_number(candidate, "quality_score")
    growth = _candidate_number(candidate, "earnings_yoy")
    market_cap = _candidate_number(candidate, "mktcap")
    if not factor_only and not management_transition:
        if quality is None or quality < 70:
            return False, enriched, "quality_hard_gate_failed"
        if growth is None or growth < 0.10:
            return False, enriched, "growth_hard_gate_failed"
        if market_cap is None or market_cap < 100:
            return False, enriched, "market_cap_hard_gate_failed"

    trade_basis = (
        None if management_transition
        else _candidate_number(candidate, "trade_basis_score")
    )
    leadership = (
        None if management_transition
        else _candidate_number(candidate, "leadership_score")
    )
    right_quant = (
        None if management_transition
        else _candidate_number(candidate, "right_quant_score")
    )
    ret20 = (
        None if management_transition
        else _candidate_number(candidate, "return_20d")
    )
    ret60 = (
        None if management_transition
        else _candidate_number(candidate, "return_60d")
    )
    candidate_avg_amount = (
        None if management_transition
        else _candidate_number(candidate, "avg_amount_20")
    )
    observed_signal_amount, observed_avg_amount = _trailing_amount_observation(
        data,
        index,
        amount_multiplier=price_amount_multiplier,
    )
    avg_amount = (
        observed_avg_amount
        if factor_only and observed_avg_amount is not None
        else candidate_avg_amount
    )
    large_liquid_trend_core = (
        False if management_transition
        else _large_liquid_trend_core_profile(candidate)
    )
    trade_basis_floor = 6.0 if large_liquid_trend_core else 6.5
    # factor_only 候选的 trade_basis 检查同样保持"候选无该字段"时的语义
    # （跳过）：候选画像补出的 trade_basis_score 不应改变 factor_only 通道行为。
    if trade_basis is not None and not factor_only and trade_basis < trade_basis_floor:
        return False, enriched, "trade_basis_too_weak"
    liquidity_floor = 500_000_000.0
    liquidity_profile = "standard_500m"
    compact_attack_liquidity_ok = _compact_attack_core_profile(candidate)
    if compact_attack_liquidity_ok:
        liquidity_floor = 450_000_000.0
        liquidity_profile = "compact_attack_core_450m"
    if large_liquid_trend_core:
        liquidity_floor = 3_000_000_000.0
        liquidity_profile = "large_liquid_trend_core_3b"
    if factor_only:
        liquidity_floor = (
            None
            if factor_only_min_avg_amount is None
            else float(factor_only_min_avg_amount)
        )
        liquidity_profile = (
            "factor_only_visible_amount_not_enforced"
            if liquidity_floor is None
            else "factor_only_visible_20d_amount_floor"
        )
    enriched.update({
        "observed_signal_amount": observed_signal_amount,
        "observed_avg_amount_20": observed_avg_amount,
        "observed_amount_multiplier": float(price_amount_multiplier),
    })
    if (
        factor_only
        and liquidity_floor is not None
        and (avg_amount is None or avg_amount < liquidity_floor)
    ):
        enriched.update({
            "semantic_entry_ok": False,
            "entry_gate_reason": "liquidity_too_thin",
            "entry_liquidity_floor": liquidity_floor,
            "entry_liquidity_profile": liquidity_profile,
        })
        return False, enriched, "liquidity_too_thin"
    if avg_amount is not None and not factor_only and avg_amount < liquidity_floor:
        enriched.update({
            "semantic_entry_ok": False,
            "entry_gate_reason": "liquidity_too_thin",
            "entry_liquidity_floor": liquidity_floor,
            "entry_liquidity_profile": liquidity_profile,
        })
        return False, enriched, "liquidity_too_thin"
    if ret20 is not None and not factor_only and ret20 < -0.02:
        return False, enriched, "weak_20d_relative_strength"

    strong_profile = any((
        leadership is not None and leadership >= 18.0,
        right_quant is not None and right_quant >= 60.0,
        ret60 is not None and ret60 >= 0.20,
        trade_basis is not None and trade_basis >= 8.0,
    ))
    if not factor_only and not management_transition and not strong_profile:
        return False, enriched, "weak_relative_strength"

    signal_type = str(signal.get("signal_type") or "")
    technical_alignment = _candidate_text(candidate, "technical_alignment")
    if (
        signal_type == "uptrend_support_pullback"
        and large_liquid_trend_core
        and technical_alignment
        and technical_alignment != "trade_ready"
    ):
        enriched.update({
            "semantic_entry_ok": False,
            "entry_gate_reason": "large_liquid_trend_watch_only",
        })
        return False, enriched, "large_liquid_trend_watch_only"
    if (
        signal_type == "uptrend_support_pullback"
        and not _signal_support_has_fast_ma_confluence(signal)
    ):
        enriched.update({
            "semantic_entry_ok": False,
            "entry_gate_reason": "support_pullback_without_fast_ma_confluence",
        })
        return False, enriched, "support_pullback_without_fast_ma_confluence"
    volume_ratio = float(signal.get("known_volume_ratio") or 0.0)
    breakout_types = {
        "uptrend_50_reclaim", "pullback_50_breakout", "gap_long_ma_breakout",
        "consolidation_breakout", "leader_pivot_breakout",
        "volume_price_node", "leader_trend_add",
    }
    if signal_type in breakout_types and volume_ratio < 1.0:
        return False, enriched, "volume_confirmation_missing"
    if ret20 is not None and not factor_only and ret20 > 0.70:
        return False, enriched, "late_acceleration_chase"
    if ret20 is not None and not factor_only and ret20 > 0.45 and volume_ratio >= 3.0:
        return False, enriched, "late_acceleration_chase"

    risk_amount = entry_price - stop
    risk_pct = risk_amount / entry_price
    if risk_pct < MIN_DAILY_RIGHT_RISK_PCT:
        enriched.update({
            "semantic_entry_ok": False,
            "entry_risk_pct": round(risk_pct * 100, 3),
            "entry_minimum_risk_pct": round(MIN_DAILY_RIGHT_RISK_PCT * 100, 3),
            "entry_gate_reason": "risk_pct_too_tight",
        })
        return False, enriched, "risk_pct_too_tight"
    if signal.get("order_type") == "close" and risk_pct < MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT:
        enriched.update({
            "semantic_entry_ok": False,
            "entry_risk_pct": round(risk_pct * 100, 3),
            "entry_minimum_close_confirmed_risk_pct": round(
                MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT * 100,
                3,
            ),
            "entry_gate_reason": "close_confirmed_risk_pct_too_tight",
        })
        return False, enriched, "close_confirmed_risk_pct_too_tight"
    risk_limit = 0.065 if signal_type == "uptrend_support_pullback" else 0.10
    if int(current_down_streak or 0) >= 3:
        risk_limit = min(risk_limit, 0.07)
    if risk_pct > risk_limit:
        enriched.update({
            "semantic_entry_ok": False,
            "entry_risk_pct": round(risk_pct * 100, 3),
            "entry_gate_reason": "risk_pct_too_wide",
        })
        return False, enriched, "risk_pct_too_wide"

    if expected is None:
        target, target_basis = _planned_right_target(
            data, index, signal, entry_price, risk_amount, candidate,
        )
    else:
        target, target_basis = expected["target"], "visible_forward_structure_net_expectancy"
    reward_risk = (target - entry_price) / risk_amount if risk_amount > 0 else 0.0
    required_rr = 3.0 if signal_type == "uptrend_support_pullback" else 2.5
    if signal_type == "uptrend_support_pullback" and large_liquid_trend_core:
        required_rr = 3.5
    if int(current_down_streak or 0) >= 3:
        required_rr = max(required_rr, 3.0)
    if reward_risk < required_rr:
        enriched.update({
            "semantic_entry_ok": False,
            "entry_risk_pct": round(risk_pct * 100, 3),
            "entry_reward_risk": round(reward_risk, 3),
            "entry_target_price": round(target, 3),
            "entry_target_basis": target_basis,
            "entry_gate_reason": "reward_risk_too_low",
        })
        return False, enriched, "reward_risk_too_low"

    enriched.update({
        "semantic_entry_ok": True,
        "semantic_entry_setup": (
            "factor_only_high_rr_right_entry"
            if factor_only else "high_rr_right_entry"
        ),
        "entry_gate_reason": "ok",
        "entry_risk_pct": round(risk_pct * 100, 3),
        "entry_reward_risk": round(reward_risk, 3),
        "entry_target_price": round(target, 3),
        "entry_target_basis": target_basis,
        "entry_liquidity_floor": liquidity_floor,
        "entry_liquidity_profile": liquidity_profile,
    })
    return True, enriched, "ok"

# Historical imports remain supported; rules live in the research package.
from stock_research.research.channels import (
    _prepare_bull_half_reversal_signal,
    _confirm_bull_half_reversal_signal,
    _semantic_bull_half_reversal_gate,
)
