"""Point-in-time portfolio replay over dated selection snapshots."""
from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np
import pandas as pd

from stock_research.strategies.candidate_interface import (
    MAX_DAILY_CANDIDATES,
    normalize_candidate_snapshots,
)
from stock_research.indicators.position_risk import position_exit_snapshot
from stock_research.indicators.price_structure import (
    configured_price_structures,
    infer_price_structures,
)
from stock_research.indicators.technical_entries import (
    SUPPORT_ZONE_PCT,
    _valid_volume_price_nodes,
    apply_entry_confluence,
    infer_technical_entry,
)
from stock_research.reporting.portfolio_attribution import build_portfolio_attribution
from stock_research.backtesting.corporate_actions import CorporateActionBook
from stock_research.reporting.signal_text import (
    localize_breadth_reversal_observation,
    localize_entry_block,
    localize_event,
    localize_signal_counts,
    summary_code_text,
)
from stock_research.strategies.ohlc_execution import (
    fill_buy_stop,
    fill_limit_order,
    fill_market_on_open,
    fill_sell_stop,
    locked_limit_direction,
)


from stock_research.backtesting.values import (
    _finite_float,
    _split_industry_tags,
)
from stock_research.backtesting.prices import (
    _divergence_series as _divergence_series,
    _prepare_frame as _prepare_frame,
    prepare_portfolio_price_frames,
)
from stock_research.backtesting.accounting import (
    sell_stamp_duty_rate_for_date,
    _capped_entry_size,
    _affordable_buy_notional,
    board_lot_size,
    _board_lot_quantity,
    _effective_profit_tranches,
)
from stock_research.backtesting.state import (
    PositionState,
    _is_profit_tail,
    _qualifies_profit_tail,
    _profit_ids_to_execute,
    _entry_risk_still_controls_lot,
)
from stock_research.reporting.portfolio_metrics import (
    _portfolio_r_multiple_summary,
    _portfolio_profit_concentration_summary,
)
from stock_research.backtesting.corporate_actions import (
    _normalize_corporate_actions,
    _corporate_action_adjusted_quantities,
)
from stock_research.backtesting.defaults import (
    DEFAULT_CLOSE_CONFIRMED_EXECUTION,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    DEFAULT_MINIMUM_COMMISSION,
    DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28,
    DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY,
)

MIN_DAILY_RIGHT_RISK_PCT = 0.01
MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT = 0.018
LARGE_LIQUID_TREND_PILOT_SIZE = 0.06
PULLBACK_PILOT_BASE_SIZE = 0.08
PULLBACK_PILOT_STRONG_SIZE = 0.10
DEFAULT_BREADTH_REVERSAL_MAX_UNIQUE_COUNT = 50
DEFAULT_BREADTH_REVERSAL_RISK_BUDGET = 0.006
DEFAULT_BREADTH_REVERSAL_MAX_POSITION = 0.20
DEFAULT_BREADTH_REVERSAL_TIME_LIMIT_DAYS = 20
DEFAULT_FACTOR_HORIZON_RISK_BUDGET = 0.015
DEFAULT_FACTOR_HORIZON_STOP_LOSS_PCT = 0.08
DEFAULT_FACTOR_HORIZON_MAX_POSITION = 0.20
DEFAULT_FACTOR_HORIZON_HOLDING_DAYS = 20
DEFAULT_COMPRESSED_STRUCTURE_RISK_BUDGET = 0.015
DEFAULT_COMPRESSED_STRUCTURE_STOP_LOSS_PCT = 0.08
DEFAULT_COMPRESSED_STRUCTURE_MAX_POSITION = 0.20
DEFAULT_COMPRESSED_STRUCTURE_HOLDING_DAYS = 20
DEFAULT_COMPRESSED_STRUCTURE_MIN_ENTRY_EVIDENCE_SCORE = 9.0
DEFAULT_COMPRESSED_STRUCTURE_MIN_PROXIMITY_SCORE = 40.0
DEFAULT_COMPRESSED_STRUCTURE_MIN_AVG_AMOUNT = 0.0
ENTRY_PRIORITY_MODE_LEGACY = "legacy_weighted"
ENTRY_PRIORITY_MODE_STRUCTURE = "structure_lexicographic"
ENTRY_PRIORITY_MODES = {
    ENTRY_PRIORITY_MODE_LEGACY,
    ENTRY_PRIORITY_MODE_STRUCTURE,
}


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


def _prepare_bull_half_reversal_signal(data, index, signal):
    """Turn the observed half-pullback into a protective close signal."""
    if not signal or signal.get("signal_type") != "bull_run_half_pullback":
        return signal
    row = data.iloc[index]
    trigger = _signal_number(signal, "trigger")
    signal_low = pd.to_numeric(row.get("low"), errors="coerce")
    if trigger is None or pd.isna(signal_low):
        return signal
    result = dict(signal)
    result.update({
        "order_type": "close",
        "stop": min(float(trigger), float(signal_low)) * 0.995,
        "reason": "连阳一半拉回；信号日低点下方止损",
        "requires_next_day_confirmation": True,
        "bull_half_signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
    })
    return result


def _confirm_bull_half_reversal_signal(
    data,
    index,
    signal,
    *,
    maximum_extension=0.05,
):
    """Confirm yesterday's half-pullback at today's close without backdating."""
    if not signal or signal.get("signal_type") != "bull_run_half_pullback":
        return False, signal, "missing_signal"
    row = data.iloc[index]
    trigger = _signal_number(signal, "trigger")
    close = pd.to_numeric(row.get("close"), errors="coerce")
    if trigger is None or pd.isna(close) or float(close) < trigger:
        return False, signal, "next_close_confirmation_failed"
    if float(close) > trigger * (1.0 + float(maximum_extension)):
        return False, signal, "next_close_confirmation_too_extended"
    result = dict(signal)
    result.update({
        "order_type": "close",
        "requires_next_day_confirmation": False,
        "reason": "连阳一半拉回；次日收盘确认；信号日低点下方止损",
        "close_fill_status": "next_close_confirmed",
        "technical_signal_date": signal.get("bull_half_signal_date"),
        "technical_signal_timing": "signal_close_then_next_day_close_confirmation",
    })
    return True, result, "ok"


def _semantic_bull_half_reversal_gate(data, index, candidate, signal):
    """Separate contrarian semantics from the trend-breakout entry gate."""
    enriched = dict(signal or {})
    if not signal or signal.get("signal_type") != "bull_run_half_pullback":
        return False, enriched, "missing_signal"
    entry_price = float(data.iloc[index]["close"])
    stop = _signal_number(signal, "stop")
    if stop is None or stop <= 0 or stop >= entry_price:
        return False, enriched, "invalid_or_non_protective_stop"
    average_amount = _candidate_number(candidate, "avg_amount_20")
    if average_amount is None or average_amount < 500_000_000.0:
        return False, enriched, "liquidity_too_thin"
    return_60d = _candidate_number(candidate, "return_60d")
    if return_60d is None or return_60d > -0.10:
        return False, enriched, "reversal_60d_decline_not_deep_enough"
    risk = entry_price - stop
    risk_pct = risk / entry_price
    if risk_pct < MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT:
        return False, enriched, "close_confirmed_risk_pct_too_tight"
    if risk_pct > 0.10:
        return False, enriched, "risk_pct_too_wide"
    target = entry_price + risk * 3.0
    enriched.update({
        "semantic_entry_ok": True,
        "semantic_entry_setup": "breadth_reversal_half_pullback",
        "entry_gate_reason": "ok",
        "entry_risk_pct": round(risk_pct * 100, 3),
        "entry_reward_risk": 3.0,
        "entry_target_price": round(target, 3),
        "entry_target_basis": "fixed_3r_after_next_close_confirmation",
        "entry_liquidity_floor": 500_000_000.0,
        "entry_liquidity_profile": "standard_500m",
    })
    return True, enriched, "ok"


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
):
    """Convert a technical signal into an auditable high R/R entry decision."""
    if not signal:
        return False, signal, "missing_signal"
    enriched = dict(signal)
    factor_only = _factor_only_selection_profile(candidate)
    if not factor_only and not _has_semantic_candidate_profile(candidate):
        enriched.update({
            "semantic_entry_ok": True,
            "semantic_entry_setup": "legacy_minimal_candidate_profile",
            "entry_gate_reason": "legacy_minimal_candidate_profile",
        })
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

    target, target_basis = _planned_right_target(
        data, index, signal, entry_price, risk_amount, candidate,
    )
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


def run_portfolio_backtest(
    price_frames,
    candidate_snapshots,
    formula_phases,
    *,
    requested_start,
    end_date,
    trade_plans=None,
    max_daily_candidates=MAX_DAILY_CANDIDATES,
    max_positions=5,
    max_total_held_symbols=5,
    max_left_positions=1,
    max_same_industry=2,
    same_theme_correlation=0.60,
    min_entry_evidence_score=0,
    right_entry_priority_mode=ENTRY_PRIORITY_MODE_LEGACY,
    right_entry_size_multiplier=1.0,
    right_entry_risk_budget_pct=None,
    enable_right_inverse_volatility_sizing=False,
    enable_right_market_volatility_sizing=False,
    price_amount_multiplier=1.0,
    factor_only_min_avg_amount=None,
    max_entry_signal_amount_participation=None,
    max_entry_avg_amount_participation=None,
    profit_tranches=5,
    profit_tail_min_return=0.50,
    left_grid_unit=0.02,
    left_grid_step=0.05,
    left_grid_max_exposure=0.10,
    pe_pb_left_max_positions=1,
    pe_pb_left_max_exposure=0.05,
    max_symbol_exposure=0.70,
    exit_tail_on_candidate_removal=False,
    signals_effective_next_day=DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY,
    auto_price_structure=True,
    allow_structure_pullback=True,
    allow_pullback_pilot=True,
    enable_breadth_reversal_half_pullback=False,
    factor_only_breadth_reversal_only=False,
    breadth_reversal_max_unique_count=DEFAULT_BREADTH_REVERSAL_MAX_UNIQUE_COUNT,
    breadth_reversal_risk_budget=DEFAULT_BREADTH_REVERSAL_RISK_BUDGET,
    breadth_reversal_max_position=DEFAULT_BREADTH_REVERSAL_MAX_POSITION,
    breadth_reversal_max_positions=5,
    breadth_reversal_time_limit_days=DEFAULT_BREADTH_REVERSAL_TIME_LIMIT_DAYS,
    enable_factor_horizon_channel=False,
    factor_horizon_risk_budget=DEFAULT_FACTOR_HORIZON_RISK_BUDGET,
    factor_horizon_stop_loss_pct=DEFAULT_FACTOR_HORIZON_STOP_LOSS_PCT,
    factor_horizon_max_position=DEFAULT_FACTOR_HORIZON_MAX_POSITION,
    factor_horizon_max_positions=5,
    factor_horizon_holding_days=DEFAULT_FACTOR_HORIZON_HOLDING_DAYS,
    factor_horizon_trend_management_return=None,
    factor_horizon_max_return_60d=None,
    factor_horizon_require_ma20_not_rising=False,
    factor_horizon_exclude_watch_phase=False,
    factor_horizon_min_avg_amount=None,
    enable_compressed_structure_channel=False,
    factor_only_compressed_structure_only=False,
    compressed_structure_risk_budget=DEFAULT_COMPRESSED_STRUCTURE_RISK_BUDGET,
    compressed_structure_stop_loss_pct=DEFAULT_COMPRESSED_STRUCTURE_STOP_LOSS_PCT,
    compressed_structure_max_position=DEFAULT_COMPRESSED_STRUCTURE_MAX_POSITION,
    compressed_structure_max_positions=5,
    compressed_structure_holding_days=DEFAULT_COMPRESSED_STRUCTURE_HOLDING_DAYS,
    compressed_structure_min_entry_evidence_score=(
        DEFAULT_COMPRESSED_STRUCTURE_MIN_ENTRY_EVIDENCE_SCORE
    ),
    compressed_structure_min_proximity_score=(
        DEFAULT_COMPRESSED_STRUCTURE_MIN_PROXIMITY_SCORE
    ),
    compressed_structure_min_avg_amount=(
        DEFAULT_COMPRESSED_STRUCTURE_MIN_AVG_AMOUNT
    ),
    enable_intraday_right_protective_stop=False,
    enable_diversified_factor_sleeve=False,
    diversified_factor_risk_budget=0.01,
    diversified_factor_max_position=0.20,
    close_confirmed_execution=DEFAULT_CLOSE_CONFIRMED_EXECUTION,
    commission_rate=DEFAULT_COMMISSION_RATE,
    minimum_commission=DEFAULT_MINIMUM_COMMISSION,
    initial_capital=1_000_000.0,
    sell_stamp_duty_rate=DEFAULT_SELL_STAMP_DUTY_RATE,
    sell_stamp_duty_rate_before_2023_08_28=(
        DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28
    ),
    estimated_slippage_rate=DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    corporate_actions=None,
    security_end_dates=None,
    price_frames_prepared=False,
):
    """Replay a portfolio without filling dates before candidate coverage."""
    if close_confirmed_execution not in {"next_open", "close_proxy"}:
        raise ValueError("close_confirmed_execution must be next_open or close_proxy")
    configured_entry_priority_mode = str(right_entry_priority_mode).strip()
    if configured_entry_priority_mode not in ENTRY_PRIORITY_MODES:
        raise ValueError(
            "right_entry_priority_mode must be one of "
            f"{sorted(ENTRY_PRIORITY_MODES)}"
        )
    configured_right_entry_size_multiplier = float(
        right_entry_size_multiplier
    )
    if not 0 < configured_right_entry_size_multiplier <= 3.0:
        raise ValueError(
            "right_entry_size_multiplier must be greater than 0 and at most 3"
        )
    configured_right_entry_risk_budget_pct = (
        None
        if right_entry_risk_budget_pct is None
        else float(right_entry_risk_budget_pct)
    )
    if (
        configured_right_entry_risk_budget_pct is not None
        and not 0 < configured_right_entry_risk_budget_pct <= 0.03
    ):
        raise ValueError(
            "right_entry_risk_budget_pct must be greater than 0 and at most 0.03"
        )
    if (
        configured_right_entry_risk_budget_pct is not None
        and not math.isclose(
            configured_right_entry_size_multiplier, 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "right_entry_risk_budget_pct and right_entry_size_multiplier "
            "cannot be enabled together"
        )
    configured_right_inverse_volatility_sizing = bool(
        enable_right_inverse_volatility_sizing
    )
    configured_right_market_volatility_sizing = bool(
        enable_right_market_volatility_sizing
    )
    if (
        configured_right_inverse_volatility_sizing
        and configured_right_market_volatility_sizing
    ):
        raise ValueError(
            "cross-sectional and market volatility sizing cannot be enabled together"
        )
    if (
        configured_right_entry_risk_budget_pct is not None
        and (
            configured_right_inverse_volatility_sizing
            or configured_right_market_volatility_sizing
        )
    ):
        raise ValueError(
            "right entry risk budget and volatility sizing "
            "cannot be enabled together"
        )
    configured_price_amount_multiplier = float(price_amount_multiplier)
    if configured_price_amount_multiplier <= 0:
        raise ValueError("price_amount_multiplier must be positive")
    configured_factor_only_min_avg_amount = (
        None
        if factor_only_min_avg_amount is None
        else float(factor_only_min_avg_amount)
    )
    if (
        configured_factor_only_min_avg_amount is not None
        and configured_factor_only_min_avg_amount <= 0
    ):
        raise ValueError("factor_only_min_avg_amount must be positive")

    def optional_participation(value, *, label):
        if value is None:
            return None
        result = float(value)
        if not 0 < result <= 1:
            raise ValueError(f"{label} must be greater than 0 and at most 1")
        return result

    configured_max_entry_signal_amount_participation = optional_participation(
        max_entry_signal_amount_participation,
        label="max_entry_signal_amount_participation",
    )
    configured_max_entry_avg_amount_participation = optional_participation(
        max_entry_avg_amount_participation,
        label="max_entry_avg_amount_participation",
    )
    if float(initial_capital) <= 0:
        raise ValueError("initial_capital must be positive")
    normalized_snapshots = normalize_candidate_snapshots(
        candidate_snapshots,
        include_diagnostics=True,
        max_daily_candidates=max_daily_candidates,
    )
    snapshots = {
        pd.Timestamp(date).normalize(): {str(item["code"]): dict(item) for item in rows}
        for date, rows in normalized_snapshots.items()
    }
    if not snapshots:
        raise ValueError("candidate snapshot history is empty")
    coverage_start = min(snapshots)
    requested = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(end_date).normalize()
    start = max(requested, coverage_start)
    # Cash is shared, so same-day processing order can affect which order gets
    # the remaining balance.  Sort symbols to keep reruns independent of file
    # discovery/dictionary insertion order.
    frames = (
        {code: price_frames[code] for code in sorted(price_frames)}
        if price_frames_prepared
        else prepare_portfolio_price_frames(price_frames)
    )
    frame_date_values = {
        code: frame["date"].to_numpy(dtype="datetime64[ns]")
        for code, frame in frames.items()
    }

    def frame_index_on_date(code, date):
        values = frame_date_values.get(code)
        if values is None or not len(values):
            return None
        target = np.datetime64(pd.Timestamp(date).normalize())
        position = int(np.searchsorted(values, target, side="left"))
        if position >= len(values) or values[position] != target:
            return None
        return position

    def latest_frame_index(code, date, *, inclusive=True):
        values = frame_date_values.get(code)
        if values is None or not len(values):
            return None
        side = "right" if inclusive else "left"
        position = int(np.searchsorted(
            values,
            np.datetime64(pd.Timestamp(date).normalize()),
            side=side,
        )) - 1
        return position if position >= 0 else None
    corporate_action_lookup = _normalize_corporate_actions(corporate_actions or {})
    normalized_security_end_dates = {
        str(code): pd.Timestamp(value).normalize()
        for code, value in dict(security_end_dates or {}).items()
        if pd.notna(pd.to_datetime(value, errors="coerce"))
    }
    corporate_book = CorporateActionBook(corporate_action_lookup, start)
    plans = dict((trade_plans or {}).get("plans") or {})
    states = {code: PositionState() for code in frames}
    active_position_codes = set()
    events = []
    realized = 0.0
    transaction_costs = 0.0
    cash_balance = float(initial_capital)
    equity_curve = []
    current_candidates = {}
    current_phase = "waiting"
    current_down_streak = 0
    current_up_streak = 0
    next_batch_id = 1
    last_candidate_scores = {}
    last_candidates = {}
    previous_candidate_codes = set()
    pending_candidate_exits = set()
    pending_weak_exits = set()
    pending_left_exits = {}
    pending_left_quota_exits = {}
    pending_breadth_reversal_entries = {}
    pending_compressed_structure_entries = {}
    pending_standard_right_close_entries = {}
    pending_preplaced_right_entries = {}
    factor_horizon_trend_managed_batches = set()
    factor_horizon_trend_managed_codes = set()

    candidate_names = {
        str(item["code"]): str(item.get("name") or "")
        for rows in candidate_snapshots.values()
        for item in rows
    }

    def candidate_display_name(code, candidate=None):
        if candidate:
            name = str(candidate.get("name") or "").strip()
            if name and name.lower() not in {"nan", "none"}:
                return name
        return candidate_names.get(code) or str(plans.get(code, {}).get("name") or code)

    def risk_stock_name_reason(name):
        text = str(name or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        upper = text.upper()
        if "ST" in upper or "退" in text:
            return "risk_name_or_status"
        return ""

    def candidate_risk_reason(candidate):
        if not candidate:
            return ""
        return risk_stock_name_reason(
            candidate.get("name") or "",
        )

    risk_names_as_of = {}

    def is_st(code):
        return bool(risk_stock_name_reason(risk_names_as_of.get(code, "")))

    def raw_price(row, column="close"):
        value = row.get(f"raw_{column}", row.get(column))
        number = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(number) else float(number)

    def execution_bar(row):
        item = row.copy()
        for column in ("open", "high", "low", "close"):
            value = raw_price(row, column)
            if value is not None:
                item[column] = value
        return item

    def close_proxy_sell_status(row, previous_close, code):
        if locked_limit_direction(
            execution_bar(row), previous_close, code, is_st=is_st(code),
        ) == "down":
            return "locked_limit_down"
        return "close_proxy"

    def standard_right_next_open_fill(row, previous_close, code):
        execution_row = execution_bar(row)
        fill = fill_market_on_open(
            execution_row,
            side="buy",
            previous_close=previous_close,
            code=code,
            is_st=is_st(code),
        )
        if not fill["filled"]:
            return {
                "filled": False,
                "price": None,
                "status": f"standard_right_next_open_{fill['status']}",
            }
        return {
            "filled": True,
            "price": float(fill["price"]),
            "status": "standard_right_close_confirmed_next_open",
        }

    def qfq_to_raw_price(row, price):
        factor = pd.to_numeric(row.get("qfq_to_raw_factor"), errors="coerce")
        if pd.isna(factor) or float(factor) <= 0:
            raise ValueError("qfq_to_raw_factor must be complete and positive")
        factor = float(factor)
        return float(price) * factor

    def raw_to_qfq_price(row, price):
        factor = pd.to_numeric(row.get("qfq_to_raw_factor"), errors="coerce")
        if pd.isna(factor) or float(factor) <= 0:
            raise ValueError("qfq_to_raw_factor must be complete and positive")
        factor = float(factor)
        return float(price) / factor

    def candidate_raw_price_to_signal(row, price):
        return raw_to_qfq_price(row, price)

    def frozen_raw_price_on_execution_date(row, signal, key):
        value = _finite_float(signal.get(key))
        if value is None:
            return None
        frozen_factor = _finite_float(signal.get("frozen_qfq_to_raw_factor"))
        if frozen_factor is None:
            signal_close = _finite_float(signal.get("technical_signal_close"))
            raw_close = _finite_float(signal.get("technical_signal_close_raw_price"))
            if signal_close and raw_close:
                frozen_factor = raw_close / signal_close
        if frozen_factor is None:
            return value
        return value * float(row["qfq_to_raw_factor"]) / frozen_factor

    def right_entry_risk_metadata(row, signal, entry_raw_price, quantity=None):
        stop = pd.to_numeric(signal.get("stop"), errors="coerce")
        if pd.isna(stop):
            return {}
        frozen_stop_raw = frozen_raw_price_on_execution_date(
            row, signal, "frozen_structure_stop_raw_price",
        )
        stop_raw = (
            float(frozen_stop_raw)
            if pd.notna(frozen_stop_raw) and float(frozen_stop_raw) > 0
            else qfq_to_raw_price(row, float(stop))
        )
        risk_per_share = float(entry_raw_price) - float(stop_raw)
        if risk_per_share <= 0:
            return {}
        target = pd.to_numeric(signal.get("entry_target_price"), errors="coerce")
        target_raw = None
        reward_per_share = None
        if pd.notna(target):
            target_raw = qfq_to_raw_price(row, float(target))
            reward_per_share = float(target_raw) - float(entry_raw_price)
        metadata = {
            "initial_stop_raw_price": round(float(stop_raw), 3),
            "initial_risk_per_share_raw": round(float(risk_per_share), 6),
            "initial_entry_raw_price": round(float(entry_raw_price), 3),
            "entry_reward_risk": signal.get("entry_reward_risk"),
            "entry_risk_pct": signal.get("entry_risk_pct"),
        }
        if target_raw is not None:
            metadata["initial_target_raw_price"] = round(float(target_raw), 3)
        if reward_per_share is not None:
            metadata["initial_reward_per_share_raw"] = round(float(reward_per_share), 6)
            metadata["planned_reward_r_multiple"] = round(
                float(reward_per_share) / float(risk_per_share),
                3,
            )
        if quantity is not None:
            metadata["initial_risk_amount"] = round(
                abs(float(quantity)) * float(risk_per_share),
                2,
            )
        return metadata

    def lot_origin_allocation(lot, quantity=None):
        quantity = abs(float(lot["quantity"] if quantity is None else quantity))
        total = float(lot["quantity"])
        origin = lot.get("origin_account") or ("left" if str(lot["batch"]).startswith("L") else "right")
        return {
            "batch": lot["batch"], "origin_account": origin,
            "origin_batch": lot.get("origin_batch", lot["batch"]),
            "quantity": quantity, "cost_amount": quantity * float(lot["cost"]),
            "entry_fee_amount": float(lot.get("entry_fee_cash") or 0.0) * quantity / total if total else 0.0,
        }

    def lot_risk_metadata(lot, quantity=None):
        risk_per_share = _finite_float(lot.get("initial_risk_per_share_raw"))
        origin = lot_origin_allocation(lot, quantity)
        provenance = {
            "batch": origin["batch"], "origin_account": origin["origin_account"],
            "origin_batch": origin["origin_batch"], "origin_allocations": [origin],
        }
        if risk_per_share is None or risk_per_share <= 0:
            return provenance
        sold_quantity = abs(float(quantity if quantity is not None else lot.get("quantity") or 0.0))
        metadata = {
            "initial_risk_per_share_raw": round(risk_per_share, 6),
            "initial_risk_amount": round(sold_quantity * risk_per_share, 2),
            "initial_stop_raw_price": lot.get("initial_stop_raw_price"),
            "initial_entry_raw_price": lot.get("initial_entry_raw_price"),
            "initial_target_raw_price": lot.get("initial_target_raw_price"),
            "initial_reward_per_share_raw": lot.get("initial_reward_per_share_raw"),
            "planned_reward_r_multiple": lot.get("planned_reward_r_multiple"),
            "entry_reward_risk": lot.get("entry_reward_risk"),
            "entry_risk_pct": lot.get("entry_risk_pct"),
            "r_multiple_source": "single_lot",
        }
        return {**provenance, **{key: value for key, value in metadata.items() if value is not None}}

    def lot_is_sellable(lot, date):
        return (not lot.get("corporate_action_pending")
                and pd.Timestamp(lot.get("acquired_date", lot["date"])).normalize() < date
                and pd.Timestamp(lot.get("sellable_from", lot["date"])).normalize() <= date)

    def lot_signal_cost(lot):
        return float(lot.get("signal_cost", lot.get("cost")))

    def with_raw_signal_price_metadata(row, metadata):
        enriched = dict(metadata)
        for key in (
            "entry_target_price",
            "trigger",
            "stop",
            "anchor_low",
            "anchor_high",
            "anchor_pullback_low",
            "value_line",
            "left_value_line",
        ):
            value = pd.to_numeric(enriched.get(key), errors="coerce")
            if pd.notna(value) and float(value) > 0:
                enriched[f"{key}_raw_price"] = round(
                    qfq_to_raw_price(row, float(value)),
                    3,
                )
        if enriched:
            enriched.setdefault("signal_price_basis", "qfq_end_anchored")
            enriched.setdefault("execution_price_basis", "raw_unadjusted")
        return enriched

    def occupied_codes():
        pending_codes = {p["code"] for p in corporate_book.pending}
        active_position_codes.update(pending_codes)
        stale = {
            code for code in active_position_codes
            if not states[code].left and not states[code].right and code not in pending_codes
        }
        active_position_codes.difference_update(stale)
        return set(active_position_codes)

    def left_side_codes():
        return {
            code for code in occupied_codes()
            for state in (states[code],)
            if state.left or state.left_grid_started
            or any(p["code"] == code and p["account"] == "left" for p in corporate_book.pending)
        }

    def left_unrealized_return_before(code, date):
        state = states[code]
        quantity = sum(float(lot.get("quantity") or 0.0) for lot in state.left)
        if quantity <= 1e-9:
            return 0.0
        cost = sum(
            float(lot.get("quantity") or 0.0) * lot_signal_cost(lot)
            for lot in state.left
        ) / quantity
        if cost <= 1e-9:
            return 0.0
        visible = frames[code][frames[code]["date"].dt.normalize() < date]
        if visible.empty:
            return 0.0
        return float(visible.iloc[-1]["close"]) / cost - 1.0

    def left_position_counts_capacity(state):
        if not state.left:
            return False
        return any(not bool(lot.get("core")) for lot in state.left)

    def capacity_codes():
        return {
            code for code in occupied_codes()
            for state in (states[code],)
            if left_position_counts_capacity(state)
            or (state.right and not _is_profit_tail(state))
            or bool(corporate_book.lots(code))
        }

    def gross_exposure():
        return sum(
            sum(float(lot["size"]) for lot in state.left + state.right)
            for code in occupied_codes()
            for state in (states[code],)
        ) + sum(float(lot["size"]) for lot in corporate_book.lots())

    def symbol_exposure(code):
        state = states[code]
        return sum(float(lot["size"]) for lot in state.left + state.right + corporate_book.lots(code))

    configured_candidate_limit = max(1, int(max_daily_candidates))
    configured_positions = (
        5 if max_positions is None or int(max_positions) <= 0
        else min(5, int(max_positions))
    )
    configured_total_held_symbols = max(1, min(10, int(max_total_held_symbols)))
    configured_left_positions = max(
        0, min(configured_total_held_symbols, int(max_left_positions)),
    )
    configured_same_industry = max(1, int(max_same_industry))
    configured_theme_correlation = float(same_theme_correlation)
    configured_min_entry_evidence_score = max(
        0.0, float(min_entry_evidence_score),
    )
    configured_allow_pullback_pilot = bool(allow_pullback_pilot)
    configured_factor_only_breadth_reversal_only = bool(
        factor_only_breadth_reversal_only
    )
    configured_breadth_reversal_enabled = bool(
        enable_breadth_reversal_half_pullback
        or configured_factor_only_breadth_reversal_only
    )
    configured_breadth_reversal_max_unique_count = max(
        1, int(breadth_reversal_max_unique_count),
    )
    configured_breadth_reversal_risk_budget = min(
        0.03, max(0.0001, float(breadth_reversal_risk_budget)),
    )
    configured_breadth_reversal_max_position = min(
        0.30, max(0.01, float(breadth_reversal_max_position)),
    )
    configured_breadth_reversal_max_positions = max(
        1, min(5, int(breadth_reversal_max_positions)),
    )
    configured_breadth_reversal_time_limit_days = max(
        1, int(breadth_reversal_time_limit_days),
    )
    configured_factor_horizon_enabled = bool(enable_factor_horizon_channel)
    configured_factor_horizon_risk_budget = min(
        0.03, max(0.0001, float(factor_horizon_risk_budget)),
    )
    configured_factor_horizon_stop_loss_pct = min(
        0.15, max(0.02, float(factor_horizon_stop_loss_pct)),
    )
    configured_factor_horizon_max_position = min(
        0.30, max(0.01, float(factor_horizon_max_position)),
    )
    configured_factor_horizon_max_positions = max(
        1, min(5, int(factor_horizon_max_positions)),
    )
    configured_factor_horizon_holding_days = max(
        1, int(factor_horizon_holding_days),
    )
    configured_factor_horizon_trend_management_return = (
        None
        if factor_horizon_trend_management_return is None
        else max(0.0, float(factor_horizon_trend_management_return))
    )
    configured_factor_horizon_max_return_60d = (
        None
        if factor_horizon_max_return_60d is None
        else float(factor_horizon_max_return_60d)
    )
    configured_factor_horizon_require_ma20_not_rising = bool(
        factor_horizon_require_ma20_not_rising
    )
    configured_factor_horizon_exclude_watch_phase = bool(
        factor_horizon_exclude_watch_phase
    )
    configured_factor_horizon_min_avg_amount = (
        500_000_000.0
        if factor_horizon_min_avg_amount is None
        else max(0.0, float(factor_horizon_min_avg_amount))
    )
    configured_factor_only_compressed_structure_only = bool(
        factor_only_compressed_structure_only
    )
    configured_compressed_structure_enabled = bool(
        enable_compressed_structure_channel
        or configured_factor_only_compressed_structure_only
    )
    configured_compressed_structure_risk_budget = min(
        0.03, max(0.0001, float(compressed_structure_risk_budget)),
    )
    configured_compressed_structure_stop_loss_pct = min(
        0.15, max(0.02, float(compressed_structure_stop_loss_pct)),
    )
    configured_compressed_structure_max_position = min(
        0.30, max(0.01, float(compressed_structure_max_position)),
    )
    configured_compressed_structure_max_positions = max(
        1, min(5, int(compressed_structure_max_positions)),
    )
    configured_compressed_structure_holding_days = max(
        1, int(compressed_structure_holding_days),
    )
    configured_compressed_structure_min_entry_evidence_score = max(
        0.0, float(compressed_structure_min_entry_evidence_score),
    )
    configured_compressed_structure_min_proximity_score = max(
        0.0, float(compressed_structure_min_proximity_score),
    )
    configured_compressed_structure_min_avg_amount = max(
        0.0, float(compressed_structure_min_avg_amount),
    )
    configured_intraday_right_protective_stop = bool(
        enable_intraday_right_protective_stop
    )
    configured_diversified_factor_sleeve = bool(
        enable_diversified_factor_sleeve
    )
    configured_diversified_factor_risk_budget = min(
        0.03, max(0.0001, float(diversified_factor_risk_budget)),
    )
    configured_diversified_factor_max_position = min(
        0.30, max(0.01, float(diversified_factor_max_position)),
    )
    configured_profit_tranches = max(2, min(5, int(profit_tranches)))
    configured_profit_tail_min_return = max(0.0, float(profit_tail_min_return))
    configured_left_grid_unit = max(0.0, float(left_grid_unit))
    configured_left_grid_step = max(0.05, float(left_grid_step))
    configured_left_grid_max_exposure = min(
        0.30, max(0.0, float(left_grid_max_exposure)),
    )
    configured_pe_pb_left_max_positions = max(
        0,
        min(configured_left_positions, int(pe_pb_left_max_positions)),
    )
    configured_pe_pb_left_max_exposure = min(
        configured_left_grid_max_exposure,
        max(0.0, float(pe_pb_left_max_exposure)),
    )
    if configured_left_grid_unit > 0 and configured_left_grid_max_exposure > 0:
        configured_left_initial_slots = max(
            1,
            min(
                5,
                int(math.ceil(configured_left_grid_max_exposure / configured_left_grid_unit - 1e-9)),
            ),
        )
    else:
        configured_left_initial_slots = 0
    configured_left_core_slots = min(3, configured_left_initial_slots)
    if configured_left_grid_unit > 0 and configured_pe_pb_left_max_exposure > 0:
        configured_pe_pb_left_initial_slots = max(
            1,
            min(
                5,
                int(math.ceil(
                    configured_pe_pb_left_max_exposure
                    / configured_left_grid_unit
                    - 1e-9
                )),
            ),
        )
    else:
        configured_pe_pb_left_initial_slots = 0
    configured_pe_pb_left_core_slots = min(
        configured_pe_pb_left_initial_slots,
        int(math.ceil(configured_pe_pb_left_initial_slots * 0.60)),
    )
    for state in states.values():
        state.right_parts = configured_profit_tranches
    concentration_blocks = []
    entry_blocks = []
    breadth_reversal_observations = []

    phase_buy_budget = float(initial_capital)
    phase_reserved_codes = set()
    phase_reserved_capacity = set()
    phase_reserved_left = set()
    phase_reserved_pe_pb = set()
    phase_industry_counts = {}

    def begin_execution_phase(phase):
        nonlocal execution_phase, phase_buy_budget, phase_reserved_codes
        nonlocal phase_reserved_capacity, phase_reserved_left, phase_reserved_pe_pb, phase_industry_counts
        execution_phase = phase
        phase_buy_budget = float(cash_balance)
        phase_reserved_codes = occupied_codes()
        phase_reserved_capacity = capacity_codes()
        phase_reserved_left = left_side_codes()
        phase_reserved_pe_pb = pe_pb_left_codes()
        phase_industry_counts = held_industry_counts()

    def symbol_limit_reached():
        configured = configured_positions
        return len(capacity_codes() | phase_reserved_capacity) >= configured

    def total_symbol_limit_reached(code=None):
        occupied = occupied_codes() | phase_reserved_codes
        if code is not None and code in occupied:
            return False
        return len(occupied) >= configured_total_held_symbols

    def right_market_active():
        return current_phase in {"watch", "active"} or current_up_streak >= 3

    def left_symbol_limit_reached(code=None):
        left_codes = left_side_codes() | phase_reserved_left
        if code is not None and code in left_codes:
            return False
        return len(left_codes) >= configured_left_positions

    def left_candidate_kind(candidate):
        sources = set(str((candidate or {}).get("candidate_source") or "").split("+"))
        return "pe_pb" if sources & {"pe_model", "pb_model"} else "value"

    def pe_pb_left_codes():
        return {
            code
            for code in left_side_codes()
            if left_candidate_kind(states[code].left_candidate_profile) == "pe_pb"
        }

    def pe_pb_left_limit_reached(code=None):
        if (
            code is not None
            and code in left_side_codes()
            and left_candidate_kind(states[code].left_candidate_profile) == "pe_pb"
        ):
            return False
        return len(pe_pb_left_codes() | phase_reserved_pe_pb) >= configured_pe_pb_left_max_positions

    def right_codes():
        return {
            code for code in occupied_codes()
            if states[code].right
        }

    def channel_position_count(lot_field):
        return sum(
            any(lot.get(lot_field) for lot in state.right)
            for state in states.values()
        )

    def candidate_industry_tags(candidate):
        raw = candidate.get("industry") or candidate.get("mainline_boards") or ""
        if pd.isna(raw):
            return set()
        aliases = {
            "通信网络设备及器件": "通信设备",
            "光通信设备": "通信设备",
            "通信终端及配件": "通信设备",
        }
        tags = {
            aliases.get(tag.strip(), tag.strip())
            for tag in re.split(r"[、,，;；/|+]", str(raw))
            if tag.strip()
        }
        return tags

    def left_value_falsification_reason(candidate):
        if not candidate:
            return ""
        reason = str(candidate.get("value_falsification_reason") or "").strip()
        if reason:
            return reason
        if _enabled(candidate.get("value_falsified"), default=False):
            return str(
                candidate.get("candidate_failure_reason")
                or "value thesis falsified by financial snapshot"
            ).strip()
        return ""

    def candidate_failure_reason(candidate):
        if not candidate:
            return ""
        return str(candidate.get("candidate_failure_reason") or "").strip()

    def candidate_is_daily_quota_diagnostic(candidate):
        reason = candidate_failure_reason(candidate)
        return reason.startswith(
            "not_selected_for_trading: factor_rank_not_in_observation_pool"
        ) or reason.startswith(
            "not_selected_for_trading: daily_top10_quota_or_core_reservation"
        )

    def candidate_has_exceptional_right_profile(candidate):
        """Keep the over-quota escape hatch only for clear right-side leaders."""
        if not candidate:
            return False
        sources = set(str(candidate.get("candidate_source") or "").split("+"))
        if not sources & {
            "standard_mainline", "growth_leadership",
            "quant_right", "factor_quant",
        }:
            return False
        right_quant_score = pd.to_numeric(
            candidate.get("right_quant_score"), errors="coerce",
        )
        if pd.notna(right_quant_score) and float(right_quant_score) >= 95.0:
            return True
        leadership = pd.to_numeric(candidate.get("leadership_score"), errors="coerce")
        return_20 = pd.to_numeric(candidate.get("return_20d"), errors="coerce")
        return_60 = pd.to_numeric(candidate.get("return_60d"), errors="coerce")
        drawdown_60 = pd.to_numeric(candidate.get("drawdown_60"), errors="coerce")
        avg_amount_20 = pd.to_numeric(candidate.get("avg_amount_20"), errors="coerce")
        visible_right_strength = (
            pd.notna(leadership)
            and float(leadership) >= 23.0
        ) or (
            pd.notna(right_quant_score)
            and float(right_quant_score) >= 75.0
        )
        return bool(
            visible_right_strength
            and pd.notna(return_20)
            and pd.notna(return_60)
            and pd.notna(drawdown_60)
            and pd.notna(avg_amount_20)
            and 0.05 <= float(return_20) <= 0.60
            and float(return_60) >= 0.35
            and float(drawdown_60) >= -0.12
            and float(avg_amount_20) >= 5_000_000_000.0
        )

    def candidate_passes_hard_gate(candidate):
        if not candidate:
            return False
        quality = pd.to_numeric(candidate.get("quality_score"), errors="coerce")
        growth = pd.to_numeric(candidate.get("earnings_yoy"), errors="coerce")
        market_cap = pd.to_numeric(candidate.get("mktcap"), errors="coerce")
        return (
            pd.notna(quality)
            and pd.notna(growth)
            and pd.notna(market_cap)
            and float(quality) >= 70.0
            and float(growth) >= 0.10
            and float(market_cap) >= 100.0
        )

    def right_candidate_can_evaluate(candidate):
        if not candidate:
            return False
        if candidate_risk_reason(candidate):
            return False
        if (
            _enabled(candidate.get("selected_for_trading"), default=True)
            and _enabled(candidate.get("signal_eligible"), default=True)
        ):
            return True
        # Ranking out of the daily Top10 is a report/display limit.  It should
        # not prevent a right-side model candidate from waiting for its own
        # high-quality technical setup.
        return (
            candidate_is_daily_quota_diagnostic(candidate)
            and _enabled(candidate.get("allow_right"), default=False)
            and candidate_has_exceptional_right_profile(candidate)
            and candidate_passes_hard_gate(candidate)
            and not left_value_falsification_reason(candidate)
        )

    def left_candidate_can_add(candidate):
        if not candidate:
            return False
        if candidate_risk_reason(candidate):
            return False
        sources = set(str(candidate.get("candidate_source") or "").split("+"))
        quality = pd.to_numeric(candidate.get("quality_score"), errors="coerce")
        market_cap = pd.to_numeric(candidate.get("mktcap"), errors="coerce")
        if "pe_model" in sources or "pb_model" in sources:
            # PE/PB 低估通道：低估比率 = 当前估值/历史年度低估均值 <= 1.0，
            # 质量/市值门槛放宽（quality>=60、mktcap>=100亿）。
            pe_ratio = pd.to_numeric(candidate.get("pe_ratio"), errors="coerce")
            pb_ratio = pd.to_numeric(candidate.get("pb_ratio"), errors="coerce")
            ratios = [v for v in (pe_ratio, pb_ratio) if pd.notna(v)]
            low_ratio = min(ratios) if ratios else np.nan
            return (
                _enabled(candidate.get("selected_for_trading"), default=True)
                and _enabled(candidate.get("signal_eligible"), default=True)
                and _enabled(candidate.get("allow_left"), default=False)
                and pd.notna(low_ratio)
                and 0.0 < float(low_ratio) <= 1.0
                and pd.notna(quality)
                and float(quality) >= 60.0
                and pd.notna(market_cap)
                and float(market_cap) >= 100.0
                and not left_value_falsification_reason(candidate)
            )
        price_to_value = pd.to_numeric(candidate.get("price_to_value"), errors="coerce")
        growth = pd.to_numeric(candidate.get("earnings_yoy"), errors="coerce")
        return (
            _enabled(candidate.get("selected_for_trading"), default=True)
            and _enabled(candidate.get("signal_eligible"), default=True)
            and _enabled(candidate.get("allow_left"), default=False)
            and "value_model" in sources
            and _enabled(candidate.get("value_industry_allowed"), default=False)
            and pd.notna(price_to_value)
            and 0.88 <= float(price_to_value) <= 1.03
            and pd.notna(quality)
            and float(quality) >= 80.0
            and pd.notna(growth)
            and float(growth) >= 0.20
            and pd.notna(market_cap)
            and float(market_cap) >= 150.0
            and not left_value_falsification_reason(candidate)
        )

    def held_industry_counts():
        counts = {}
        for code in occupied_codes():
            state = states[code]
            if _is_profit_tail(state) and not state.left and not corporate_book.lots(code):
                continue
            tags = {
                tag
                for lot in state.left + state.right + corporate_book.lots(code)
                for tag in lot.get("industry_tags", [])
            }
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def return_correlation(code, held_code, date):
        left = frames[code]
        right = frames[held_code]
        left = left[left["date"] < date][["date", "close"]].tail(61)
        right = right[right["date"] < date][["date", "close"]].tail(61)
        aligned = left.merge(right, on="date", suffixes=("_left", "_right"))
        if len(aligned) < 41:
            return None
        returns = aligned[["close_left", "close_right"]].pct_change().dropna()
        if len(returns) < 40:
            return None
        if (returns.std(ddof=0) <= 1e-12).any():
            return None
        correlation = returns["close_left"].corr(returns["close_right"])
        return None if pd.isna(correlation) else float(correlation)

    def industry_limit_reached(code, candidate, date):
        if code in capacity_codes():
            return False
        counts = held_industry_counts()
        counts = {tag: max(counts.get(tag, 0), phase_industry_counts.get(tag, 0))
                  for tag in set(counts) | set(phase_industry_counts)}
        tags = candidate_industry_tags(candidate)
        tag_blocked = any(
            counts.get(tag, 0) >= configured_same_industry
            for tag in tags
        )
        correlations = {
            held_code: return_correlation(code, held_code, date)
            for held_code in capacity_codes()
        }
        related = {
            held_code: correlation
            for held_code, correlation in correlations.items()
            if correlation is not None
            and correlation >= configured_theme_correlation
        }
        correlation_blocked = len(related) >= configured_same_industry
        if tag_blocked or correlation_blocked:
            concentration_blocks.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "code": code,
                "name": candidate.get("name") or code,
                "industry_tags": "、".join(sorted(tags)),
                "correlated_held_codes": sorted(related),
                "correlations": {
                    held_code: round(correlation, 4)
                    for held_code, correlation in related.items()
                },
                "reason": "industry_tag" if tag_blocked else "return_correlation",
            })
            return True
        return False

    def record_entry_block(date, code, candidate, signal, reason):
        signal_metadata = {}
        if signal:
            visible = frames.get(code, pd.DataFrame())
            if not visible.empty:
                day_row = visible[visible["date"].dt.normalize() == date]
                if not day_row.empty:
                    signal_metadata = with_raw_signal_price_metadata(
                        day_row.iloc[-1],
                        {
                            key: signal.get(key)
                            for key in (
                                "entry_target_price", "trigger", "stop",
                                "anchor_low", "anchor_high",
                                "anchor_pullback_low",
                            )
                            if signal.get(key) is not None
                        },
                    )
        entry_blocks.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "code": code,
            "name": candidate.get("name") or code,
            "reason": reason,
            "signal_type": signal.get("signal_type") if signal else None,
            "entry_evidence_score": (
                signal.get("entry_evidence_score") if signal else None
            ),
            "semantic_entry_ok": signal.get("semantic_entry_ok") if signal else None,
            "entry_risk_pct": signal.get("entry_risk_pct") if signal else None,
            "entry_reward_risk": signal.get("entry_reward_risk") if signal else None,
            "entry_target_price": signal.get("entry_target_price") if signal else None,
            "entry_target_basis": signal.get("entry_target_basis") if signal else None,
            "entry_target_raw_price": signal_metadata.get("entry_target_price_raw_price"),
            "signal_trigger_raw_price": signal_metadata.get("trigger_raw_price"),
            "signal_stop_raw_price": signal_metadata.get("stop_raw_price"),
            "signal_price_basis": signal_metadata.get("signal_price_basis"),
            "execution_price_basis": signal_metadata.get("execution_price_basis"),
            "candidate_score": candidate.get("candidate_score"),
            "trade_basis_score": candidate.get("trade_basis_score"),
            "leadership_score": candidate.get("leadership_score"),
            "candidate_failure_reason": candidate_failure_reason(candidate),
            "technical_signal_date": (
                signal.get("technical_signal_date") if signal else None
            ),
            "technical_signal_timing": (
                signal.get("technical_signal_timing") if signal else None
            ),
            "execution_date": signal.get("execution_date") if signal else None,
            "execution_timing": (
                signal.get("execution_timing") if signal else None
            ),
            "execution_status": (
                signal.get("execution_status") if signal else None
            ),
            "entry_channel": signal.get("entry_channel") if signal else None,
            "formula_window_unique_count": (
                signal.get("formula_window_unique_count") if signal else None
            ),
            "structure_proximity_score": (
                signal.get("structure_proximity_score") if signal else None
            ),
            "structure_proximity_score_source": (
                signal.get("structure_proximity_score_source") if signal else None
            ),
        })

    def has_twenty_percent_float(date):
        for code in occupied_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty:
                continue
            close = float(visible.iloc[-1]["close"])
            if any(close / lot_signal_cost(lot) - 1 >= 0.20 for lot in state.right):
                return True
            if any(float(lot.get("max_return") or 0.0) >= 0.20 for lot in state.right):
                return True
        return False

    def symbol_has_float_profit(code, date, threshold):
        state = states[code]
        visible = frames[code][frames[code]["date"] < date]
        if visible.empty or not state.right:
            return False
        close = float(visible.iloc[-1]["close"])
        return any(close / lot_signal_cost(lot) - 1 >= float(threshold) for lot in state.right)

    def add_on_float_threshold():
        return 0.20 if current_down_streak >= 3 else 0.10

    def right_symbol_returns(date):
        returns = []
        for code in capacity_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty or not state.right:
                continue
            close = float(visible.iloc[-1]["close"])
            lot_returns = [
                max(
                    float(lot.get("max_return") or 0.0),
                    close / lot_signal_cost(lot) - 1,
                )
                for lot in state.right
            ]
            if lot_returns:
                returns.append(max(lot_returns))
        return sorted(returns, reverse=True)

    def can_open_new_right_symbol(date, *, down_streak=None):
        returns = right_symbol_returns(date)
        down_streak = (
            current_down_streak
            if down_streak is None else int(down_streak)
        )
        if not returns:
            return True
        if len(returns) >= configured_positions:
            return False
        if down_streak < 3:
            return all(value >= 0.10 for value in returns)
        if len(returns) == 1:
            return returns[0] >= 0.20
        return returns[0] >= 0.20 and all(value >= 0.10 for value in returns[1:])

    def leading_pullback_pilot_size(candidate, signal):
        if not configured_allow_pullback_pilot:
            return 0.0
        if not candidate or not signal:
            return 0.0
        if signal.get("signal_type") != "uptrend_support_pullback":
            return 0.0
        if not right_market_active() or current_down_streak >= 3:
            return 0.0
        sources = set(str(candidate.get("candidate_source") or "").split("+"))
        factor_only = _factor_only_selection_profile(candidate)
        if not factor_only and not sources & {
            "factor_quant", "standard_mainline", "growth_leadership",
            "quant_right",
        }:
            return 0.0
        leadership = pd.to_numeric(candidate.get("leadership_score"), errors="coerce")
        right_quant = pd.to_numeric(candidate.get("right_quant_score"), errors="coerce")
        trade_basis = pd.to_numeric(candidate.get("trade_basis_score"), errors="coerce")
        return_20 = pd.to_numeric(candidate.get("return_20d"), errors="coerce")
        return_60 = pd.to_numeric(candidate.get("return_60d"), errors="coerce")
        drawdown_60 = pd.to_numeric(candidate.get("drawdown_60"), errors="coerce")
        avg_amount_20 = pd.to_numeric(candidate.get("avg_amount_20"), errors="coerce")
        support_distance = 1.0
        trigger = pd.to_numeric(signal.get("trigger"), errors="coerce")
        stop = pd.to_numeric(signal.get("stop"), errors="coerce")
        if (
            pd.notna(trigger)
            and pd.notna(stop)
            and float(trigger) > 0
            and float(stop) > 0
        ):
            support_distance = max(0.0, float(trigger) / float(stop) - 1.0)
        known_volume_ratio = float(signal.get("known_volume_ratio") or 0.0)
        if factor_only:
            return (
                PULLBACK_PILOT_BASE_SIZE
                if known_volume_ratio >= 1.0 and support_distance <= 0.055
                else 0.0
            )
        strong_quant_profile = (
            pd.notna(leadership) and float(leadership) >= 25.0
        ) or (
            pd.notna(right_quant) and float(right_quant) >= 85.0
        ) or (
            pd.notna(trade_basis) and float(trade_basis) >= 7.5
        )
        large_liquid_trend_core = _large_liquid_trend_core_profile(candidate)
        base_ok = (
            strong_quant_profile
            and known_volume_ratio >= 1.0
            and support_distance <= 0.055
        )
        if base_ok and pd.notna(trade_basis):
            base_ok = float(trade_basis) >= 7.0
        if base_ok and pd.notna(return_20):
            base_ok = 0.0 <= float(return_20) <= 0.45
        if base_ok and pd.notna(return_60):
            base_ok = float(return_60) >= 0.10
        if base_ok and pd.notna(drawdown_60):
            base_ok = float(drawdown_60) >= -0.18
        if base_ok and pd.notna(avg_amount_20):
            base_ok = float(avg_amount_20) >= 800_000_000.0
        large_liquid_ok = (
            large_liquid_trend_core
            and known_volume_ratio >= 1.0
            and support_distance <= 0.055
        )
        if large_liquid_ok and pd.notna(return_20):
            large_liquid_ok = 0.0 <= float(return_20) <= 0.40
        if large_liquid_ok and pd.notna(return_60):
            large_liquid_ok = float(return_60) >= 0.25
        if large_liquid_ok and pd.notna(drawdown_60):
            large_liquid_ok = float(drawdown_60) >= -0.16
        if not base_ok:
            return LARGE_LIQUID_TREND_PILOT_SIZE if large_liquid_ok else 0.0
        strong_ok = (
            strong_quant_profile
            and known_volume_ratio >= 1.10
            and support_distance <= 0.045
        )
        if strong_ok and pd.notna(return_20):
            strong_ok = 0.0 <= float(return_20) <= 0.45
        if strong_ok and pd.notna(return_60):
            strong_ok = float(return_60) >= 0.20
        if strong_ok and pd.notna(drawdown_60):
            strong_ok = float(drawdown_60) >= -0.12
        if strong_ok and pd.notna(avg_amount_20):
            strong_ok = float(avg_amount_20) >= 1_500_000_000.0
        return PULLBACK_PILOT_STRONG_SIZE if strong_ok else PULLBACK_PILOT_BASE_SIZE

    def leading_pullback_pilot_starter(candidate, signal):
        return leading_pullback_pilot_size(candidate, signal) > 0.0

    def high_conviction_right_entry_size(candidate, signal):
        if not candidate or not signal:
            return 0.0
        reward_risk = pd.to_numeric(signal.get("entry_reward_risk"), errors="coerce")
        risk_pct = pd.to_numeric(signal.get("entry_risk_pct"), errors="coerce")
        trade_basis = pd.to_numeric(candidate.get("trade_basis_score"), errors="coerce")
        leadership = pd.to_numeric(candidate.get("leadership_score"), errors="coerce")
        right_quant = pd.to_numeric(candidate.get("right_quant_score"), errors="coerce")
        return_20 = pd.to_numeric(candidate.get("return_20d"), errors="coerce")
        if pd.isna(reward_risk) or pd.isna(risk_pct):
            return 0.0
        strong_profile = (
            (pd.notna(leadership) and float(leadership) >= 25.0)
            or (pd.notna(right_quant) and float(right_quant) >= 90.0)
            or (pd.notna(trade_basis) and float(trade_basis) >= 8.0)
        )
        decisive_pivot_asymmetric_setup = (
            str(signal.get("signal_type") or "") == "leader_pivot_breakout"
            and pd.notna(right_quant)
            and float(right_quant) >= 60.0
            and pd.notna(return_20)
            and 0.08 <= float(return_20) <= 0.25
            and float(reward_risk) >= 5.0
            and 4.5 <= float(risk_pct) <= 6.0
            and float(signal.get("known_volume_ratio") or 0.0) >= 1.0
        )
        if decisive_pivot_asymmetric_setup:
            return 0.30
        if (
            strong_profile
            and float(reward_risk) >= 2.5
            and float(risk_pct) <= 7.0
        ):
            return 0.30
        return 0.0

    def trade_fee_components(turnover, *, sell=False, trade_date=None):
        turnover = abs(float(turnover))
        if turnover <= 0:
            return {
                "commission_amount": 0.0,
                "stamp_duty_amount": 0.0,
                "slippage_amount": 0.0,
                "transaction_cost_amount": 0.0,
            }
        commission = max(
            turnover * float(commission_rate), float(minimum_commission),
        )
        stamp_duty_rate = (
            sell_stamp_duty_rate_for_date(
                trade_date,
                sell_stamp_duty_rate,
                sell_stamp_duty_rate_before_2023_08_28,
            )
            if sell else 0.0
        )
        stamp_duty = turnover * stamp_duty_rate
        slippage = turnover * float(estimated_slippage_rate)
        return {
            "commission_amount": commission,
            "stamp_duty_amount": stamp_duty,
            "slippage_amount": slippage,
            "transaction_cost_amount": commission + stamp_duty + slippage,
        }

    def add_event(
        date, code, action, price, size, reason, pnl=0.0, *,
        cost_basis=None, execution_quantity=None, entry_fee_cash=0.0, **metadata,
    ):
        nonlocal realized
        metadata = dict(metadata)
        realized += pnl
        execution_basis = (
            float(price) if float(size) > 0
            else float(cost_basis) if float(size) < 0 and cost_basis is not None
            else None
        )
        if execution_quantity is None:
            execution_quantity = (
                float(initial_capital) * float(size) / execution_basis
                if execution_basis not in {None, 0.0} else 0.0
            )
        execution_quantity = float(execution_quantity)
        trade_side = "买入" if execution_quantity > 0 else "卖出" if execution_quantity < 0 else "状态"
        trade_amount = abs(execution_quantity) * float(price)
        fee_components = trade_fee_components(
            trade_amount,
            sell=execution_quantity < 0,
            trade_date=date,
        )
        allocated_entry_fee = float(entry_fee_cash) if execution_quantity < 0 else 0.0
        cost_amount = (
            abs(execution_quantity) * float(cost_basis)
            if execution_quantity < 0 and cost_basis is not None else None
        )
        gross_pnl_amount = (
            trade_amount - cost_amount
            if cost_amount is not None else None
        )
        round_trip_pnl_amount = (
            gross_pnl_amount
            - fee_components["transaction_cost_amount"]
            - allocated_entry_fee
            if gross_pnl_amount is not None else None
        )
        if execution_quantity > 0:
            profit_loss_amount = -fee_components["transaction_cost_amount"]
            profit_loss_pct = (
                profit_loss_amount / trade_amount * 100 if trade_amount else None
            )
            cash_change_amount = -trade_amount - fee_components["transaction_cost_amount"]
        elif execution_quantity < 0:
            profit_loss_amount = round_trip_pnl_amount
            profit_loss_pct = (
                round_trip_pnl_amount / (cost_amount + allocated_entry_fee) * 100
                if cost_amount is not None and cost_amount + allocated_entry_fee > 0 else None
            )
            cash_change_amount = trade_amount - fee_components["transaction_cost_amount"]
        else:
            profit_loss_amount = None
            profit_loss_pct = None
            cash_change_amount = 0.0
        risk_per_share = _finite_float(metadata.get("initial_risk_per_share_raw"))
        initial_risk_amount = _finite_float(metadata.get("initial_risk_amount"))
        if (
            initial_risk_amount is None
            and risk_per_share is not None
            and risk_per_share > 0
            and abs(execution_quantity) > 0
        ):
            initial_risk_amount = abs(execution_quantity) * risk_per_share
        if initial_risk_amount is not None and initial_risk_amount > 0:
            metadata["initial_risk_amount"] = round(initial_risk_amount, 2)
            if execution_quantity < 0:
                if profit_loss_amount is not None:
                    metadata["realized_r_multiple"] = round(
                        float(profit_loss_amount) / initial_risk_amount,
                        3,
                    )
                if gross_pnl_amount is not None:
                    metadata["gross_r_multiple"] = round(
                        float(gross_pnl_amount) / initial_risk_amount,
                        3,
                    )
                total_fees = fee_components["transaction_cost_amount"] + allocated_entry_fee
                metadata["transaction_cost_r_multiple"] = round(
                    total_fees / initial_risk_amount,
                    3,
                )
                stop_raw = _finite_float(metadata.get("initial_stop_raw_price"))
                if stop_raw is not None and risk_per_share is not None and risk_per_share > 0:
                    metadata["exit_vs_initial_stop_r"] = round(
                        (float(price) - stop_raw) / risk_per_share,
                        3,
                    )
        events.append({
            "date": date.strftime("%Y-%m-%d"), "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "action": action, "price": round(float(price), 3),
            "execution_price": float(price),
            "position_change_pct": round(size * 100, 2),
            "reason": reason, "phase": current_phase,
            "realized_account_pct": round(pnl * 100, 4),
            # Quantity is emitted for independent broker/accounting replays.
            # Exposure is defined against initial capital in this engine, so
            # buys use their fill and exits use the same exact lot quantity.
            # Keeping the unrounded value avoids reconciliation
            # drift from the display-only percentage fields above.
            "execution_quantity": execution_quantity,
            "cost_basis": execution_basis,
            "trade_side": trade_side,
            "quantity": round(abs(execution_quantity), 8),
            "trade_amount": round(trade_amount, 2),
            "commission_amount": round(fee_components["commission_amount"], 2),
            "stamp_duty_amount": round(fee_components["stamp_duty_amount"], 2),
            "slippage_amount": round(fee_components["slippage_amount"], 2),
            "transaction_cost_amount": round(fee_components["transaction_cost_amount"], 2),
            "allocated_entry_fee_amount": round(allocated_entry_fee, 2),
            "cost_amount": None if cost_amount is None else round(cost_amount, 2),
            "gross_pnl_amount": None if gross_pnl_amount is None else round(gross_pnl_amount, 2),
            "profit_loss_amount": None if profit_loss_amount is None else round(profit_loss_amount, 2),
            "profit_loss_pct": None if profit_loss_pct is None else round(profit_loss_pct, 4),
            "cash_change_amount": round(cash_change_amount, 2),
            "cash_change_amount_raw": float(metadata.get("cash_change_amount", cash_change_amount)),
            "transaction_cost_amount_raw": fee_components["transaction_cost_amount"],
            "profit_loss_amount_raw": profit_loss_amount,
            "execution_phase": execution_phase,
            "event_sequence": len(events) + 1,
            "cash_before": round(cash_balance - float(metadata.get("cash_change_amount", cash_change_amount)), 6),
            "cash_after": round(cash_balance, 6),
            **metadata,
        })
        if execution_quantity > 0:
            active_position_codes.add(code)
        elif (
            execution_quantity < 0
            and not states[code].left
            and not states[code].right
        ):
            active_position_codes.discard(code)

    def trade_fee(turnover, *, sell=False, trade_date=None):
        return trade_fee_components(
            turnover,
            sell=sell,
            trade_date=trade_date,
        )["transaction_cost_amount"]

    def execute_buy(code, requested_size, price):
        """Buy against real shared cash, shrinking rather than borrowing."""
        nonlocal cash_balance, transaction_costs, phase_buy_budget
        available_cash = min(cash_balance, phase_buy_budget)
        requested_notional = max(0.0, float(requested_size)) * float(initial_capital)
        price = float(price)
        if requested_notional <= 0 or price <= 0 or available_cash <= float(minimum_commission):
            return 0.0, 0.0, 0.0, 0.0, False
        affordable_notional = _affordable_buy_notional(
            available_cash,
            requested_notional,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_rate=estimated_slippage_rate,
        )
        if affordable_notional <= 1e-8:
            return 0.0, 0.0, 0.0, 0.0, True
        cash_limited = affordable_notional < requested_notional - 1e-8
        quantity = _board_lot_quantity(code, affordable_notional / price)
        if quantity <= 0:
            return 0.0, 0.0, 0.0, 0.0, cash_limited
        notional = quantity * price
        fee_cash = trade_fee(notional)
        cash_balance = max(0.0, cash_balance - notional - fee_cash)
        phase_buy_budget = max(0.0, phase_buy_budget - notional - fee_cash)
        fee = fee_cash / float(initial_capital)
        transaction_costs += fee
        return notional / float(initial_capital), float(quantity), -fee, fee_cash, cash_limited

    def execute_sell(quantity, sell, cost):
        """Sell exact shares and release net proceeds into shared cash."""
        nonlocal cash_balance, transaction_costs
        quantity = max(0.0, float(quantity))
        sell = float(sell)
        cost = float(cost)
        if quantity <= 0:
            return 0.0, 0.0
        turnover = quantity * sell
        fee_cash = trade_fee(turnover, sell=True, trade_date=date)
        cash_balance += turnover - fee_cash
        fee = fee_cash / float(initial_capital)
        transaction_costs += fee
        cost_value = quantity * cost
        size = cost_value / float(initial_capital)
        pnl = (turnover - cost_value - fee_cash) / float(initial_capital)
        return size, pnl

    def apply_corporate_actions(date):
        nonlocal cash_balance
        action_events, _ = corporate_book.advance(date, states, _corporate_action_adjusted_quantities)
        for event in action_events:
            event = dict(event)
            code = event.pop("code")
            action = event.pop("action")
            cash = float(event.get("cash_change_amount") or 0.)
            cash_balance += cash
            index = latest_frame_index(code, date, inclusive=False)
            price = raw_price(frames[code].iloc[index], "close") if index is not None else 0.
            add_event(date, code, action, price or 0., 0., "按登记权益及实际生效日期记账",
                      float(event.pop("income_amount", 0.)) / float(initial_capital), **event)
            if action == "红股上市可售" and states[code].right:
                states[code].right_parts = min(
                    states[code].right_parts,
                    _effective_profit_tranches(code, sum(float(lot["quantity"]) for lot in states[code].right),
                                               configured_profit_tranches))
            if states[code].left or states[code].right:
                active_position_codes.add(code)
        for code in sorted(occupied_codes()):
            index = frame_index_on_date(code, date)
            if index is None or index == 0:
                continue
            data = frames[code]
            before = _finite_float(data.iloc[index - 1].get("raw_to_qfq_factor"))
            after = _finite_float(data.iloc[index].get("raw_to_qfq_factor"))
            if before and after and abs(before / after - 1.) > 1e-4:
                if not any(data.iloc[index - 1]["date"] < ex_date <= date
                           for ex_date in corporate_action_lookup.get(code, {})):
                    corporate_book.limitation({"code": code, "ex": date},
                                              "adjustment_factor_changed_without_explicit_action")

    def consume_merged_lots(code, state, lots, requested_quantity):
        """Allocate an account-level sell order to FIFO lots without rounding each lot."""
        remaining = max(0, int(requested_quantity))
        sold_quantity = 0.0
        sold_cost_value = 0.0
        sold_entry_fee = 0.0
        sold_risk_amount = 0.0
        sold_risk_quantity = 0.0
        sold_reward_amount = 0.0
        sold_entry_risk_pct_amount = 0.0
        origin_allocations = []
        for lot in list(lots):
            if remaining <= 0:
                break
            original_quantity = float(lot["quantity"])
            take = min(original_quantity, remaining)
            if take <= 0:
                continue
            ratio = take / original_quantity
            origin_allocations.append(lot_origin_allocation(lot, take))
            sold_quantity += take
            sold_cost_value += take * float(lot["cost"])
            sold_entry_fee += float(lot.get("entry_fee_cash") or 0.0) * ratio
            risk_per_share = _finite_float(lot.get("initial_risk_per_share_raw"))
            if risk_per_share is not None and risk_per_share > 0:
                sold_risk_amount += take * risk_per_share
                sold_risk_quantity += take
                reward_per_share = _finite_float(lot.get("initial_reward_per_share_raw"))
                if reward_per_share is not None:
                    sold_reward_amount += take * reward_per_share
                entry_risk_pct = _finite_float(lot.get("entry_risk_pct"))
                if entry_risk_pct is not None:
                    sold_entry_risk_pct_amount += take * entry_risk_pct
            lot["quantity"] = original_quantity - take
            lot["size"] *= 1.0 - ratio
            lot["entry_fee_cash"] = float(lot.get("entry_fee_cash") or 0.0) * (1.0 - ratio)
            remaining -= take
            if lot["quantity"] <= 1e-9:
                state.right.remove(lot)
        sold_cost = sold_cost_value / sold_quantity if sold_quantity else 0.0
        risk_metadata = {}
        if (sold_quantity > 0 and sold_risk_amount > 0
                and abs(sold_risk_quantity - sold_quantity) <= 1e-8):
            risk_metadata = {
                "initial_risk_amount": round(sold_risk_amount, 2),
                "initial_risk_per_share_raw": round(sold_risk_amount / sold_quantity, 6),
                "r_multiple_source": "merged_weighted",
            }
            if sold_reward_amount:
                risk_metadata["initial_reward_per_share_raw"] = round(
                    sold_reward_amount / sold_quantity,
                    6,
                )
                risk_metadata["planned_reward_r_multiple"] = round(
                    sold_reward_amount / sold_risk_amount,
                    3,
                )
                risk_metadata["entry_reward_risk"] = risk_metadata["planned_reward_r_multiple"]
            if sold_entry_risk_pct_amount:
                risk_metadata["entry_risk_pct"] = round(
                    sold_entry_risk_pct_amount / sold_quantity,
                    3,
                )
        risk_metadata["origin_allocations"] = origin_allocations
        return sold_quantity, sold_cost, sold_entry_fee, risk_metadata

    def rotate_weak_right_symbols_for_entry(
        date,
        incoming_code,
        candidate,
        signal,
        required_size,
        current_candidates,
        exited_today,
    ):
        # A close-only replacement cannot finance an earlier intraday order.
        if execution_phase != "open" and signal.get("order_type") != "close":
            return []
        reward_risk = pd.to_numeric(signal.get("entry_reward_risk"), errors="coerce")
        risk_pct = pd.to_numeric(signal.get("entry_risk_pct"), errors="coerce")
        if (
            pd.isna(reward_risk)
            or pd.isna(risk_pct)
            or float(reward_risk) < 5.0
            or float(risk_pct) > 6.0
            or signal.get("semantic_entry_ok") is not True
        ):
            return []
        needed = gross_exposure() + float(required_size) - 1.0
        if needed <= 1e-9:
            return []
        weak = []
        for held_code in capacity_codes() - {incoming_code}:
            if held_code in exited_today or held_code in pending_weak_exits:
                continue
            held_state = states[held_code]
            if not held_state.right or any(not lot_is_sellable(lot, date) for lot in held_state.right):
                continue
            visible = frames[held_code][
                frames[held_code]["date"] < date
                if execution_phase == "open" else frames[held_code]["date"] <= date
            ]
            if visible.empty:
                continue
            held_row = visible.iloc[-1]
            held_quantity = sum(float(lot["quantity"]) for lot in held_state.right)
            if held_quantity <= 0:
                continue
            held_cost = sum(
                float(lot["quantity"]) * lot_signal_cost(lot)
                for lot in held_state.right
            ) / held_quantity
            held_return = float(held_row["close"]) / held_cost - 1.0
            first_date = min(lot["date"] for lot in held_state.right)
            holding_days = max(
                len(visible[visible["date"] >= first_date]), 1,
            )
            ma20 = pd.to_numeric(held_row.get("ma20"), errors="coerce")
            below_ma20 = pd.notna(ma20) and float(held_row["close"]) < float(ma20)
            proven = any(lot.get("proven", False) for lot in held_state.right)
            max_return = max(
                float(lot.get("max_return") or 0.0) for lot in held_state.right
            )
            removed = held_code not in current_candidates
            weak_unproven = (
                not proven
                and holding_days >= 20
                and max_return < 0.20
                and (held_return < 0.08 or below_ma20 or removed)
            )
            stale_low_efficiency = (
                holding_days >= 60
                and held_return < 0.03
                and max_return < 0.20
            )
            profit_giveback = 0.0
            if max_return > -0.99:
                profit_giveback = (1.0 + held_return) / (1.0 + max_return) - 1.0
            proven_stalling = (
                proven
                and below_ma20
                and holding_days >= 20
                and held_return < 0.50
                and profit_giveback <= -0.08
            )
            if not (weak_unproven or stale_low_efficiency or proven_stalling):
                continue
            held_score = (
                held_return * 100.0
                + (20.0 if proven else 0.0)
                + (-10.0 if below_ma20 else 5.0)
                + profit_giveback * 50.0
                + (-5.0 if removed else 0.0)
            )
            weak.append((held_score, held_code, held_return, holding_days, below_ma20, held_row["date"]))
        rotated = []
        for _, held_code, held_return, holding_days, below_ma20, decision_date in sorted(weak):
            if needed <= 1e-9:
                break
            held_state = states[held_code]
            index = frame_index_on_date(held_code, date)
            if index is None:
                continue
            row = frames[held_code].iloc[index]
            previous_close = raw_price(frames[held_code].iloc[index - 1], "close") if index > 0 else None
            if execution_phase == "open":
                fill = fill_market_on_open(
                    execution_bar(row), side="sell", previous_close=previous_close,
                    code=held_code, is_st=is_st(held_code),
                )
                if not fill["filled"]:
                    continue
                sell = float(fill["price"])
                timing = "previous_close_decision_open_fill"
            else:
                if close_proxy_sell_status(row, previous_close, held_code) == "locked_limit_down":
                    continue
                sell = raw_price(row, "close")
                timing = "close_proxy"
            for lot in list(held_state.right):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                held_state.right.remove(lot)
                add_event(
                    date, held_code, "汰弱让位", sell, -size,
                    (
                        f"{lot['batch']}; 高盈亏比新机会{incoming_code}让位; "
                        f"持有{holding_days}日; 当前收益{held_return:.1%}; "
                        f"{'跌破MA20' if below_ma20 else '效率不足'}; {timing}"
                    ),
                    pnl,
                    cost_basis=lot["cost"],
                    execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    account_mode="right",
                    decision_date=pd.Timestamp(decision_date).strftime("%Y-%m-%d"),
                    execution_timing=timing,
                    replacement_code=incoming_code,
                    replacement_name=candidate.get("name") if candidate else None,
                    replacement_reward_risk=signal.get("entry_reward_risk"),
                    replacement_risk_pct=signal.get("entry_risk_pct"),
                    **lot_risk_metadata(lot, quantity),
                )
                needed -= float(lot["size"])
            held_state.pending_lot_exits.clear()
            held_state.pending_profit_ids.clear()
            held_state.right_sold.clear()
            held_state.right_parts = configured_profit_tranches
            held_state.right_plan_date = None
            pending_weak_exits.discard(held_code)
            exited_today.add(held_code)
            rotated.append(held_code)
        return rotated

    def left_grid_plan(value_line, *, initial_slots=None, core_slots=None):
        """Value grid: capped initial units at value line, then lower units."""
        anchor = float(value_line)
        initial_slots = (
            configured_left_initial_slots
            if initial_slots is None else int(initial_slots)
        )
        core_slots = (
            configured_left_core_slots
            if core_slots is None else int(core_slots)
        )
        plan = []
        for slot in range(10):
            if slot < initial_slots:
                buy_price = anchor
            else:
                buy_price = anchor * (1.0 - configured_left_grid_step) ** (
                    slot - initial_slots + 1
                )
            if slot < core_slots:
                sell_price = None
            elif slot < initial_slots:
                sell_price = anchor * (
                    1.0 + configured_left_grid_step * (slot - core_slots + 1)
                )
            else:
                sell_price = anchor * (1.0 - configured_left_grid_step) ** (
                    slot - initial_slots
                )
            plan.append({
                "slot": slot,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "core": slot < core_slots,
            })
        return plan

    def left_initial_slot_set(initial_slots=None):
        count = (
            configured_left_initial_slots
            if initial_slots is None else int(initial_slots)
        )
        return set(range(count))

    def promote_left_grid_to_right_campaign(code, state, row, signal, first_right_batch_id):
        if not state.left:
            return [], first_right_batch_id
        promoted = []
        next_right_batch_id = first_right_batch_id
        origin_candidate_profile = dict(state.left_candidate_profile or {})
        reconfirm_on_next_day = (
            bool(signal.get("requires_next_day_confirmation"))
            or (
                signal["order_type"] == "stop"
                and float(row["close"]) < float(signal["trigger"])
            )
        )
        for lot in list(state.left):
            state.left.remove(lot)
            promoted_lot = dict(lot)
            origin_batch = lot["batch"]
            promoted_lot.update({
                "batch": f"R{next_right_batch_id}",
                "date": row["date"],
                "stop": float(signal["stop"]),
                "merged": True,
                "proven": False,
                "max_return": 0.0,
                "acquired_date": lot.get("acquired_date", lot["date"]),
                "origin_account": "left",
                "origin_batch": origin_batch,
                "origin_left_value_line": state.left_value_line,
                "origin_left_candidate_profile": origin_candidate_profile,
                "left_grid_slot": int(lot["slot"]),
                "reconfirm_level": float(signal["trigger"]),
                "reconfirm_on_next_day": reconfirm_on_next_day,
            })
            next_right_batch_id += 1
            state.right.append(promoted_lot)
            promoted.append(promoted_lot)
        state.left_value_line = None
        state.left_grid_started = False
        state.left_candidate_profile = None
        return promoted, next_right_batch_id

    def exit_promoted_left_lot(code, state, lot, date, price, reason):
        quantity = float(lot["quantity"])
        entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
        size, pnl = execute_sell(quantity, price, lot["cost"])
        state.right.remove(lot)
        state.pending_lot_exits.pop(lot["batch"], None)
        add_event(
            date, code, "左转右批次退出", price, -size,
            (
                f"{lot['batch']}(原{lot.get('origin_batch', lot['batch'])}); "
                f"{reason}; 左转右后不恢复左仓"
            ),
            pnl,
            cost_basis=lot["cost"], execution_quantity=-quantity,
            entry_fee_cash=entry_fee_cash, account_mode="right",
            **lot_risk_metadata(lot, quantity),
        )

    def left_position_detached_from_value(row, state):
        if state.left_value_line is None:
            return False
        close = pd.to_numeric(row.get("close"), errors="coerce")
        if pd.isna(close):
            return False
        return float(close) / float(state.left_value_line) > 1.08

    def left_to_right_switch_signal(code, date, candidate):
        state = states[code]
        if not state.left or state.right or code in pending_left_exits:
            return None
        if not candidate:
            return None
        data = frames[code]
        index = frame_index_on_date(code, date)
        if index is None:
            return None
        row = data.iloc[index]
        if not left_position_detached_from_value(row, state):
            return None
        if (
            candidate
            and not _enabled(candidate.get("signal_eligible"), default=True)
            and not candidate_is_daily_quota_diagnostic(candidate)
        ):
            return None
        if left_value_falsification_reason(candidate):
            return None
        signal = _right_signal(
            data, index, plans.get(code),
            auto_price_structure=auto_price_structure,
            allow_structure_pullback=allow_structure_pullback,
        )
        if not signal:
            return None
        ok, signal, _ = _semantic_right_entry_gate(
            data, index, candidate, signal,
            current_down_streak=current_down_streak,
            management_transition=True,
        )
        if not ok:
            return None
        return signal

    def execute_right_entries(candidates, date, exited_today, left_right_switch_candidate_current):
        nonlocal next_batch_id
        for (
            _, fundamental_score, __, code, index, signal, candidate,
            entry_channel,
        ) in _ordered_right_entry_candidates(
            candidates,
            frames,
            mode=configured_entry_priority_mode,
        ):
            signal = dict(signal)
            order_type = signal.get("order_type")
            if signal.get("standard_right_next_open"):
                order_phase = "open"
            elif order_type == "close":
                order_phase = "close"
            else:
                execution_row = frames[code].iloc[index]
                opening = raw_price(execution_row, "open")
                trigger = qfq_to_raw_price(execution_row, signal["trigger"])
                at_open = opening is not None and (
                    opening >= trigger if order_type == "stop" else opening <= trigger
                )
                order_phase = "open" if at_open else "intraday"
            if order_phase != execution_phase:
                continue
            reopening_profit_tail = _is_profit_tail(states[code])
            # A left-held symbol is already part of the portfolio. Its
            # management transition must not be gated as a new symbol.
            opening_right_symbol = code not in occupied_codes()
            breadth_reversal_entry = (
                entry_channel == "breadth_reversal_half_pullback"
            )
            factor_horizon_entry = (
                entry_channel == "factor_horizon_match"
            )
            compressed_structure_entry = (
                entry_channel == "compressed_structure_continuation"
            )
            diversified_factor_entry = bool(
                configured_diversified_factor_sleeve
                and entry_channel == "standard_right_entry"
                and _factor_only_selection_profile(candidate)
            )
            entry_down_streak = int(
                signal.get(
                    "frozen_formula_down_streak", current_down_streak,
                )
            )
            entry_up_streak = int(
                signal.get(
                    "frozen_formula_up_streak", current_up_streak,
                )
            )
            if opening_right_symbol:
                channel_capacity_full = (
                    breadth_reversal_entry
                    and channel_position_count("breadth_reversal_entry")
                    >= configured_breadth_reversal_max_positions
                ) or (
                    factor_horizon_entry
                    and channel_position_count("factor_horizon_entry")
                    >= configured_factor_horizon_max_positions
                ) or (
                    compressed_structure_entry
                    and channel_position_count("compressed_structure_entry")
                    >= configured_compressed_structure_max_positions
                )
                if channel_capacity_full:
                    record_entry_block(
                        date, code, candidate, signal,
                        "specialized_channel_capacity_full",
                    )
                    continue
                if total_symbol_limit_reached(code):
                    record_entry_block(date, code, candidate, signal, "total_symbol_limit")
                    continue
                if industry_limit_reached(code, candidate, date):
                    record_entry_block(date, code, candidate, signal, "industry_or_correlation_limit")
                    continue
                if (
                    not breadth_reversal_entry
                    and not factor_horizon_entry
                    and not compressed_structure_entry
                    and not diversified_factor_entry
                    and not can_open_new_right_symbol(
                        date, down_streak=entry_down_streak,
                    )
                ):
                    record_entry_block(date, code, candidate, signal, "new_symbol_unlock_not_met")
                    continue
                if symbol_limit_reached():
                    if factor_horizon_entry or compressed_structure_entry:
                        record_entry_block(
                            date, code, candidate, signal,
                            (
                                "compressed_structure_capacity_full"
                                if compressed_structure_entry
                                else "factor_horizon_capacity_full"
                            ),
                        )
                        continue
                    incoming_reference_close = float(
                        signal.get("technical_signal_close")
                        or frames[code].iloc[index]["close"]
                    )
                    incoming_stop_pct = max(
                        0.0,
                        1
                        - float(signal["stop"])
                        / incoming_reference_close,
                    )
                    incoming_score = fundamental_score + signal["rank"] * 10 + min(float(signal["known_volume_ratio"]), 5.0) * 2 - incoming_stop_pct * 20
                    weak = []
                    for held_code in capacity_codes() - pending_weak_exits:
                        held_state = states[held_code]
                        if not held_state.right:
                            continue
                        if execution_phase != "close":
                            visible = frames[held_code][
                                frames[held_code]["date"] < date
                            ]
                        else:
                            visible = frames[held_code][
                                frames[held_code]["date"] <= date
                            ]
                        if visible.empty:
                            continue
                        held_row = visible.iloc[-1]
                        held_quantity = sum(lot["quantity"] for lot in held_state.right)
                        held_cost = sum(
                            lot["quantity"] * lot_signal_cost(lot) for lot in held_state.right
                        ) / held_quantity
                        held_return = float(held_row["close"]) / held_cost - 1
                        holding_days = max(
                            len(visible[visible["date"] >= min(lot["date"] for lot in held_state.right)]), 1,
                        )
                        below_ma20 = pd.notna(held_row["ma20"]) and float(held_row["close"]) < float(held_row["ma20"])
                        removed = held_code not in current_candidates
                        eligible = below_ma20 and holding_days >= 5 and held_return < 0.10
                        if not eligible:
                            continue
                        held_fundamental = float(last_candidate_scores.get(held_code, 0.0))
                        held_score = held_fundamental + held_return * 100 + (20 if any(lot.get("proven") for lot in held_state.right) else 0) + (-10 if below_ma20 else 10)
                        replacement_margin = 5 if removed else 15
                        if incoming_score >= held_score + replacement_margin:
                            weak.append((held_score, held_code))
                    if weak:
                        pending_weak_exits.add(min(weak)[1])
                        record_entry_block(date, code, candidate, signal, "weak_replacement_scheduled")
                    else:
                        record_entry_block(date, code, candidate, signal, "symbol_limit_no_weak_replacement")
                    continue
            row = frames[code].iloc[index].copy()
            if execution_phase == "open":
                for column in ("high", "low", "close"):
                    row[column] = row["open"]
                    row[f"raw_{column}"] = raw_price(row, "open")
            previous_close = raw_price(frames[code].iloc[index - 1], "close") if index > 0 else None
            standard_right_next_open = bool(
                entry_channel == "standard_right_entry"
                and signal.get("standard_right_next_open")
            )
            if execution_phase == "open" and not standard_right_next_open:
                fill = fill_market_on_open(
                    execution_bar(row), side="buy", previous_close=previous_close,
                    code=code, is_st=is_st(code),
                )
            elif signal["order_type"] == "stop":
                fill = fill_buy_stop(
                    execution_bar(row),
                    trigger_price=qfq_to_raw_price(row, signal["trigger"]),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
            elif signal["order_type"] == "close":
                if standard_right_next_open:
                    fill = standard_right_next_open_fill(
                        row, previous_close, code,
                    )
                elif (
                    breadth_reversal_entry
                    and locked_limit_direction(
                        execution_bar(row), previous_close, code,
                        is_st=is_st(code),
                    ) == "up"
                ):
                    fill = {
                        "filled": False,
                        "price": None,
                        "status": "locked_limit_up",
                    }
                else:
                    fill = {
                        "filled": True,
                        "price": raw_price(row, "close"),
                        "status": "close_confirmed",
                    }
            else:
                fill = fill_limit_order(
                    execution_bar(row), side="buy",
                    limit_price=qfq_to_raw_price(row, signal["trigger"]),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
            if not fill["filled"]:
                if standard_right_next_open:
                    block_reason = str(fill.get("status"))
                    signal["execution_status"] = block_reason
                else:
                    block_reason = (
                        "confirmation_locked_limit_up"
                        if breadth_reversal_entry
                        and fill.get("status") == "locked_limit_up"
                        else "order_not_filled"
                    )
                record_entry_block(
                    date, code, candidate, signal, block_reason,
                )
                continue
            if standard_right_next_open or signal.get("technical_signal_timing") == "previous_close_preplaced_order":
                signal["execution_status"] = fill.get("status")
                fill_price = float(fill["price"])
                signal_close_raw = frozen_raw_price_on_execution_date(
                    row, signal, "technical_signal_close_raw_price",
                )
                stop_raw = frozen_raw_price_on_execution_date(
                    row, signal, "frozen_structure_stop_raw_price",
                )
                target_raw = frozen_raw_price_on_execution_date(
                    row, signal, "frozen_entry_target_raw_price",
                )
                if (
                    signal_close_raw is None
                    or stop_raw is None
                    or target_raw is None
                ):
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "next_open_revalidation_metadata_missing",
                    )
                    continue
                if fill_price / signal_close_raw - 1.0 > 0.05 + 1e-12:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "next_open_gap_above_chase_limit",
                    )
                    continue
                open_risk = fill_price - stop_raw
                open_risk_pct = open_risk / fill_price
                signal_type = str(signal.get("signal_type") or "")
                open_risk_limit = (
                    0.065
                    if signal_type == "uptrend_support_pullback"
                    else 0.10
                )
                if entry_down_streak >= 3:
                    open_risk_limit = min(open_risk_limit, 0.07)
                if (
                    open_risk <= 0
                    or open_risk_pct < MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT
                    or open_risk_pct > open_risk_limit
                ):
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "next_open_risk_distance_invalid",
                    )
                    continue
                open_reward_risk = (target_raw - fill_price) / open_risk
                required_reward_risk = (
                    3.0
                    if signal_type == "uptrend_support_pullback"
                    else 2.5
                )
                if entry_down_streak >= 3:
                    required_reward_risk = max(
                        required_reward_risk,
                        3.0,
                    )
                if open_reward_risk < required_reward_risk:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "next_open_reward_risk_too_low",
                    )
                    continue
                signal.update({
                    "fill_revalidated": True,
                    "next_open_revalidated": execution_phase == "open",
                    "next_open_risk_pct": round(
                        open_risk_pct * 100,
                        3,
                    ),
                    "next_open_reward_risk": round(
                        open_reward_risk,
                        3,
                    ),
                })
            existing_size = sum(lot["size"] for lot in states[code].right)
            starting_new_right_campaign = not states[code].right or reopening_profit_tail
            left_to_right = False
            left_to_right_management_only = False
            pullback_pilot = False
            if starting_new_right_campaign:
                left_to_right = bool(states[code].left)
                left_to_right_management_only = (
                    left_to_right
                    and not bool(
                        signal.get(
                            "frozen_left_to_right_candidate_current",
                            left_right_switch_candidate_current.get(
                                code, False,
                            ),
                        )
                    )
                )
                preferred_breakout = signal.get("signal_type") in {
                    "uptrend_50_reclaim",
                    "pullback_50_breakout",
                    "gap_long_ma_breakout",
                    "consolidation_breakout",
                    "leader_pivot_breakout",
                    "leader_trend_add",
                    "volume_price_node",
                    "bull_run_half_pullback",
                }
                inferred_entry = signal.get("signal_type") in {
                    "consolidation_breakout",
                    "leader_pivot_breakout",
                    "leader_trend_add",
                    "volume_price_node",
                    "bull_run_half_pullback",
                }
                pullback_pilot = (
                    False
                    if breadth_reversal_entry
                    or factor_horizon_entry
                    or compressed_structure_entry
                    else bool(signal.get("frozen_pullback_pilot"))
                    if signal.get("standard_right_next_open")
                    else leading_pullback_pilot_starter(candidate, signal)
                )
                direct_breakout = signal["order_type"] == "stop"
                high_conviction_size = float(
                    signal.get("frozen_high_conviction_size")
                    if signal.get("standard_right_next_open")
                    else high_conviction_right_entry_size(candidate, signal)
                )
                pilot_size = float(
                    signal.get("frozen_pullback_pilot_size")
                    if signal.get("standard_right_next_open")
                    else leading_pullback_pilot_size(candidate, signal)
                )
                size = (
                    float(signal["planned_position_size"])
                    if breadth_reversal_entry
                    or factor_horizon_entry
                    or compressed_structure_entry
                    else min(
                        configured_diversified_factor_max_position,
                        configured_diversified_factor_risk_budget
                        / max(
                            float(signal.get("entry_risk_pct") or 0.0)
                            / 100.0,
                            1e-6,
                        ),
                    )
                    if diversified_factor_entry
                    else 0.20 if entry_down_streak >= 3
                    else pilot_size if pullback_pilot
                    else high_conviction_size if high_conviction_size > 0
                    else 0.30 if preferred_breakout
                    else 0.20
                )
                ordinary_standard_right_first_entry = (
                    entry_channel == "standard_right_entry"
                    and not diversified_factor_entry
                    and not states[code].right
                    and not reopening_profit_tail
                    and not left_to_right
                )
                if (
                    configured_right_entry_risk_budget_pct is not None
                    and ordinary_standard_right_first_entry
                ):
                    frozen_stop_raw = frozen_raw_price_on_execution_date(
                        row, signal, "frozen_structure_stop_raw_price",
                    )
                    stop_raw = (
                        float(frozen_stop_raw)
                        if pd.notna(frozen_stop_raw)
                        and float(frozen_stop_raw) > 0
                        else qfq_to_raw_price(row, signal["stop"])
                    )
                    entry_raw_price = float(fill["price"])
                    risk_pct = (
                        entry_raw_price - float(stop_raw)
                    ) / entry_raw_price
                    if risk_pct <= 0:
                        record_entry_block(
                            date, code, candidate, signal,
                            "right_entry_risk_budget_invalid_stop_distance",
                        )
                        continue
                    size = min(
                        0.30,
                        configured_right_entry_risk_budget_pct / risk_pct,
                    )
                if not any((
                    breadth_reversal_entry,
                    factor_horizon_entry,
                    compressed_structure_entry,
                    diversified_factor_entry,
                )):
                    size *= configured_right_entry_size_multiplier
                    if (
                        configured_right_inverse_volatility_sizing
                        and ordinary_standard_right_first_entry
                    ):
                        inverse_volatility_multiplier = pd.to_numeric(
                            candidate.get(
                                "right_inverse_volatility_multiplier"
                            ),
                            errors="coerce",
                        )
                        if (
                            pd.isna(inverse_volatility_multiplier)
                            or not math.isfinite(
                                float(inverse_volatility_multiplier)
                            )
                            or float(inverse_volatility_multiplier) <= 0
                        ):
                            record_entry_block(
                                date, code, candidate, signal,
                                "right_inverse_volatility_multiplier_invalid",
                            )
                            continue
                        size *= float(inverse_volatility_multiplier)
                    if (
                        configured_right_market_volatility_sizing
                        and ordinary_standard_right_first_entry
                    ):
                        market_volatility_multiplier = pd.to_numeric(
                            candidate.get(
                                "right_market_volatility_multiplier"
                            ),
                            errors="coerce",
                        )
                        if (
                            pd.isna(market_volatility_multiplier)
                            or not math.isfinite(
                                float(market_volatility_multiplier)
                            )
                            or not 0 < float(market_volatility_multiplier) <= 1
                        ):
                            record_entry_block(
                                date, code, candidate, signal,
                                "right_market_volatility_multiplier_invalid",
                            )
                            continue
                        size *= float(market_volatility_multiplier)
                if breadth_reversal_entry:
                    entry_kind = "低广度反转连阳拉回首仓"
                elif factor_horizon_entry:
                    entry_kind = "因子周期匹配首仓"
                elif compressed_structure_entry:
                    entry_kind = "紧结构平台延续首仓"
                elif diversified_factor_entry:
                    entry_kind = "纯因子多标的风险预算首仓"
                elif pullback_pilot:
                    entry_kind = "支撑拉回试错首仓"
                elif preferred_breakout:
                    entry_kind = "33未三日下行首仓" if entry_down_streak < 3 else "33三日下行试错首仓"
                elif inferred_entry:
                    entry_kind = "技术合理买点试探仓"
                elif direct_breakout:
                    entry_kind = "直接突破首仓"
                else:
                    entry_kind = "33上行首仓" if entry_up_streak >= 3 else "33未上行首仓"
                if reopening_profit_tail:
                    entry_kind = f"尾仓再入; {entry_kind}"
                elif left_to_right:
                    left_size = sum(float(lot["size"]) for lot in states[code].left)
                    if left_to_right_management_only:
                        size = 0.0
                        entry_kind = f"左转右仅接管原左仓; 非当日候选不新增右仓; {entry_kind}"
                    else:
                        size = min(size, left_size * 0.50)
                        entry_kind = f"左转右切换; 加仓不超过原左仓一半; {entry_kind}"
            else:
                if (
                    breadth_reversal_entry
                    or factor_horizon_entry
                    or compressed_structure_entry
                ):
                    continue
                if any(not lot.get("proven", False) for lot in states[code].right):
                    continue
                pullback_entry = signal.get("signal_type") in {
                    "uptrend_support_pullback",
                    "bull_run_half_pullback",
                }
                ratio = 0.50 if pullback_entry else 1 / 3
                size = (
                    existing_size * ratio
                    * configured_right_entry_size_multiplier
                )
                entry_kind = f"浮盈加仓{ratio:.0%}"
            if left_to_right_management_only:
                states[code].pending_tail_capacity_free = False
                states[code].right_tail_capacity_free = False
                states[code].right_sold.clear()
                states[code].pending_profit_ids.clear()
                promoted_left_lots, next_batch_id = promote_left_grid_to_right_campaign(
                    code, states[code], row, signal, next_batch_id,
                )
                states[code].right_parts = _effective_profit_tranches(
                    code,
                    sum(float(lot["quantity"]) for lot in states[code].right),
                    configured_profit_tranches,
                )
                states[code].right_plan_date = row["date"]
                promoted_batch_labels = ",".join(
                    f"{lot['batch']}(原{lot.get('origin_batch', lot['batch'])})"
                    for lot in promoted_left_lots
                )
                add_event(
                    date, code, "左转右接管左仓", fill["price"], 0.0,
                    (
                        f"接管左侧批次={promoted_batch_labels}; "
                        f"右侧止损={float(signal['stop']):.3f}; 左侧网格暂停; "
                        "非当日候选，仅切换管理状态，未新增右侧仓位"
                    ),
                    account_mode="right",
                )
                record_entry_block(
                    date, code, candidate, signal,
                    "left_to_right_management_only_without_current_candidate",
                )
                continue
            size = _capped_entry_size(
                symbol_exposure(code), size, max_symbol_exposure,
            )
            if size <= 1e-9:
                record_entry_block(date, code, candidate, signal, "symbol_exposure_cap")
                continue
            requested_size = size
            signal_amount = _finite_float(
                signal.get("observed_signal_amount")
            )
            average_amount = _finite_float(
                signal.get("observed_avg_amount_20")
            )
            if signal_amount is None or average_amount is None:
                liquidity_date = signal.get("technical_signal_date") or row["date"]
                liquidity_index = frame_index_on_date(code, liquidity_date)
                if liquidity_index is not None:
                    observed_signal_amount, observed_avg_amount = (
                        _trailing_amount_observation(
                            frames[code],
                            liquidity_index,
                            amount_multiplier=(
                                configured_price_amount_multiplier
                            ),
                        )
                    )
                    signal_amount = signal_amount or observed_signal_amount
                    average_amount = average_amount or observed_avg_amount
            participation_caps = []
            if configured_max_entry_signal_amount_participation is not None:
                if signal_amount is None:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "signal_amount_unavailable_for_participation_cap",
                    )
                    continue
                participation_caps.append(
                    signal_amount
                    * configured_max_entry_signal_amount_participation
                )
            if configured_max_entry_avg_amount_participation is not None:
                if average_amount is None:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "average_amount_unavailable_for_participation_cap",
                    )
                    continue
                participation_caps.append(
                    average_amount
                    * configured_max_entry_avg_amount_participation
                )
            if participation_caps:
                requested_notional = size * float(initial_capital)
                liquidity_notional_cap = min(participation_caps)
                signal = dict(signal)
                signal.update({
                    "observed_signal_amount": signal_amount,
                    "observed_avg_amount_20": average_amount,
                    "entry_liquidity_notional_cap": liquidity_notional_cap,
                    "entry_liquidity_requested_pct": round(
                        requested_size * 100,
                        3,
                    ),
                    "entry_liquidity_limited": (
                        liquidity_notional_cap < requested_notional - 1e-8
                    ),
                })
                size = min(
                    size,
                    liquidity_notional_cap / float(initial_capital),
                )
                if size <= 1e-9:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "liquidity_participation_cap",
                    )
                    continue
                signal["entry_liquidity_approved_pct"] = round(
                    size * 100,
                    3,
                )
            if gross_exposure() + size > 1.0 + 1e-9:
                rotated = rotate_weak_right_symbols_for_entry(
                    date,
                    code,
                    candidate,
                    signal,
                    size,
                    current_candidates,
                    exited_today,
                )
                if rotated:
                    record_entry_block(
                        date,
                        code,
                        candidate,
                        signal,
                        "weak_replacement_executed:" + ",".join(rotated),
                    )
            if gross_exposure() + size > 1.0 + 1e-9:
                available_size = max(0.0, 1.0 - gross_exposure())
                minimum_trimmed_size = _minimum_trimmed_entry_size(
                    size,
                    starting_new_right_campaign=starting_new_right_campaign,
                    pullback_pilot=pullback_pilot,
                )
                if available_size + 1e-9 < minimum_trimmed_size:
                    record_entry_block(date, code, candidate, signal, "gross_exposure_cap")
                    continue
                signal = dict(signal)
                signal["gross_exposure_trimmed_from_pct"] = round(size * 100, 3)
                signal["gross_exposure_available_pct"] = round(available_size * 100, 3)
                size = min(size, available_size)
            execution_requested_size = size
            size, quantity, pnl, entry_fee_cash, cash_limited = execute_buy(
                code, execution_requested_size, float(fill["price"]),
            )
            if size <= 1e-9:
                if left_to_right:
                    states[code].pending_tail_capacity_free = False
                    states[code].right_tail_capacity_free = False
                    states[code].right_sold.clear()
                    states[code].pending_profit_ids.clear()
                    promoted_left_lots, next_batch_id = promote_left_grid_to_right_campaign(
                        code, states[code], row, signal, next_batch_id,
                    )
                    states[code].right_parts = _effective_profit_tranches(
                        code,
                        sum(float(lot["quantity"]) for lot in states[code].right),
                        configured_profit_tranches,
                    )
                    states[code].right_plan_date = row["date"]
                    promoted_batch_labels = ",".join(
                        f"{lot['batch']}(原{lot.get('origin_batch', lot['batch'])})"
                        for lot in promoted_left_lots
                    )
                    add_event(
                        date, code, "左转右接管左仓", fill["price"], 0.0,
                        (
                            f"接管左侧批次={promoted_batch_labels}; "
                            f"右侧止损={float(signal['stop']):.3f}; 左侧网格暂停; "
                            "现金或一手股数限制，仅切换管理状态，未新增右侧仓"
                        ),
                        account_mode="right",
                    )
                    record_entry_block(
                        date, code, candidate, signal,
                        "left_to_right_add_on_cash_or_board_lot_limit",
                    )
                    continue
                record_entry_block(date, code, candidate, signal, "cash_or_board_lot_limit")
                continue
            if reopening_profit_tail:
                for lot in states[code].right:
                    lot["merged"] = True
                states[code].right_parts = configured_profit_tranches
                states[code].right_sold.clear()
                states[code].pending_profit_ids.clear()
                states[code].pending_tail_capacity_free = False
                states[code].right_tail_capacity_free = False
                states[code].right_plan_date = row["date"]
            elif not states[code].right:
                states[code].pending_tail_capacity_free = False
                states[code].right_tail_capacity_free = False
            promoted_left_lots = []
            if left_to_right:
                promoted_left_lots, next_batch_id = promote_left_grid_to_right_campaign(
                    code, states[code], row, signal, next_batch_id,
                )
            batch = f"R{next_batch_id}"
            next_batch_id += 1
            risk_metadata = right_entry_risk_metadata(
                row,
                signal,
                float(fill["price"]),
                quantity,
            )
            right_lot = {
                "cost": float(fill["price"]), "date": row["date"],
                "signal_cost": raw_to_qfq_price(row, float(fill["price"])),
                "stop": float(signal["stop"]), "size": size,
                "quantity": quantity,
                "entry_fee_cash": entry_fee_cash,
                "batch": batch,
                "merged": reopening_profit_tail or starting_new_right_campaign,
                "proven": False,
                "max_return": 0.0,
                "time_limit_days": (
                    configured_breadth_reversal_time_limit_days
                    if breadth_reversal_entry
                    else configured_compressed_structure_holding_days
                    if compressed_structure_entry
                    else configured_factor_horizon_holding_days
                    if factor_horizon_entry
                    else 8 if pullback_pilot else None
                ),
                "space_stop_pct": (
                    configured_factor_horizon_stop_loss_pct
                    if factor_horizon_entry
                    else configured_compressed_structure_stop_loss_pct
                    if compressed_structure_entry
                    else 0.07 if pullback_pilot else None
                ),
                "breadth_reversal_entry": breadth_reversal_entry,
                "factor_horizon_entry": factor_horizon_entry,
                "factor_horizon_trend_managed": False,
                "compressed_structure_entry": compressed_structure_entry,
                "diversified_factor_entry": diversified_factor_entry,
                "intraday_protective_stop": (
                    breadth_reversal_entry
                    or factor_horizon_entry
                    or compressed_structure_entry
                    or configured_intraday_right_protective_stop
                ),
                "fixed_target_price": (
                    float(signal["entry_target_price"])
                    if breadth_reversal_entry else None
                ),
                "industry_tags": sorted(candidate_industry_tags(candidate)),
                "reconfirm_level": float(signal["trigger"]),
                "reconfirm_on_next_day": (
                    bool(signal.get("requires_next_day_confirmation"))
                    or (
                        signal["order_type"] == "stop"
                        and float(row["close"]) < float(signal["trigger"])
                    )
                ),
                **risk_metadata,
            }
            states[code].right.append(right_lot)
            if starting_new_right_campaign:
                states[code].right_parts = _effective_profit_tranches(
                    code,
                    sum(float(lot["quantity"]) for lot in states[code].right),
                    configured_profit_tranches,
                )
                if promoted_left_lots:
                    promoted_batch_labels = ",".join(
                        f"{lot['batch']}(原{lot.get('origin_batch', lot['batch'])})"
                        for lot in promoted_left_lots
                    )
                    add_event(
                        date, code, "左转右接管左仓", fill["price"], 0.0,
                        (
                            f"接管左侧批次={promoted_batch_labels}; "
                            f"右侧止损={float(signal['stop']):.3f}; 左侧网格暂停"
                        ),
                        account_mode="right",
                    )
            if not states[code].right_plan_date:
                states[code].right_plan_date = row["date"]
            structure_metadata = {
                key: signal.get(key)
                for key in (
                    "structure_ratio", "anchor_low", "anchor_high",
                    "anchor_pullback_low", "anchor_low_date",
                    "anchor_high_date", "anchor_pullback_low_date",
                    "signal_type", "gap_up",
                    "entry_evidence_score", "entry_evidence",
                    "support_confluence",
                    "semantic_entry_ok", "semantic_entry_setup",
                    "entry_gate_reason", "entry_risk_pct",
                    "entry_reward_risk", "entry_target_price",
                    "entry_target_basis", "entry_liquidity_floor",
                    "entry_liquidity_profile",
                    "observed_signal_amount", "observed_avg_amount_20",
                    "observed_amount_multiplier",
                    "entry_liquidity_notional_cap",
                    "entry_liquidity_requested_pct",
                    "entry_liquidity_approved_pct",
                    "entry_liquidity_limited",
                    "next_open_revalidated", "next_open_risk_pct",
                    "next_open_reward_risk",
                    "entry_channel", "formula_observation_date",
                    "formula_window_unique_count", "planned_position_size",
                    "source_signal_type", "original_entry_gate_reason",
                    "structure_proximity_score",
                    "structure_proximity_score_source",
                    "volume_node_date", "volume_node_confirmed_on",
                    "valid_volume_node_count",
                    "frozen_formula_phase",
                    "frozen_formula_down_streak",
                    "frozen_formula_up_streak",
                )
                if signal.get(key) is not None
            }
            structure_metadata.update({
                key: signal.get(key)
                for key in ("trigger", "stop")
                if signal.get(key) is not None
            })
            signal_metadata_row = row
            if standard_right_next_open:
                signal_metadata_index = frame_index_on_date(
                    code,
                    pd.Timestamp(signal["technical_signal_date"]),
                )
                if signal_metadata_index is not None:
                    signal_metadata_row = frames[code].iloc[
                        signal_metadata_index
                    ]
            structure_metadata = with_raw_signal_price_metadata(
                signal_metadata_row, structure_metadata,
            )
            structure_metadata["industry_tags"] = "、".join(
                sorted(candidate_industry_tags(candidate))
            )
            buy_metadata = {**lot_risk_metadata(right_lot, quantity), **risk_metadata, **structure_metadata}
            buy_metadata.update({
                "candidate_snapshot_date": candidate.get(
                    "candidate_snapshot_date",
                ),
                "technical_signal_date": (
                    signal.get("technical_signal_date")
                    or date.strftime("%Y-%m-%d")
                ),
                "technical_signal_timing": (
                    signal.get("technical_signal_timing")
                    or (
                        "same_day_close_confirmed_close_proxy"
                        if close_confirmed_execution == "close_proxy"
                        else "close_confirmed_next_open"
                    )
                ),
                "execution_date": signal.get("execution_date"),
                "execution_timing": signal.get("execution_timing"),
                "execution_status": signal.get("execution_status"),
            })
            add_event(
                date, code, "右侧买入", fill["price"], size,
                f"{batch}; {entry_kind}; {signal['reason']}; {fill['status']}",
                pnl, execution_quantity=quantity,
                entry_fee_cash=entry_fee_cash,
                requested_position_pct=requested_size * 100,
                cash_limited=cash_limited,
                lot_rounded=(
                    size < execution_requested_size - 1e-9
                    and not cash_limited
                ),
                board_lot_size=board_lot_size(code),
                selection_reason=_clean_trade_basis_text(
                    candidate.get("selection_reason"),
                ),
                trade_basis_score=candidate.get("trade_basis_score"),
                trade_basis_reason=_clean_trade_basis_text(
                    candidate.get("trade_basis_reason"),
                ),
                technical_alignment=candidate.get("technical_alignment"),
                ima_web_validation=candidate.get("ima_web_validation"),
                **buy_metadata,
            )


    def fill_matches_phase(fill):
        status = str(fill.get("status") or "")
        phase = ("open" if status in {"open_fill", "gap_or_open_fill", "gap_stop", "gap_breakout",
                                      "standard_right_close_confirmed_next_open"}
                 else "close" if status == "close_confirmed" else "intraday")
        return phase == execution_phase

    def execute_left_grids(left_targets, date, side, sold_grid_slots):
        for code, grid_anchor, anchor_is_candidate_raw in left_targets:
            state = states[code]
            if code in pending_left_exits:
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            if anchor_is_candidate_raw:
                grid_anchor = candidate_raw_price_to_signal(row, grid_anchor)
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            candidate = current_candidates.get(code, {})
            can_add_left = left_candidate_can_add(candidate)
            left_profile = candidate or state.left_candidate_profile or {}
            pe_pb_left = left_candidate_kind(left_profile) == "pe_pb"
            effective_left_max_exposure = (
                configured_pe_pb_left_max_exposure
                if pe_pb_left else configured_left_grid_max_exposure
            )
            effective_initial_slots = (
                configured_pe_pb_left_initial_slots
                if pe_pb_left else configured_left_initial_slots
            )
            effective_core_slots = (
                configured_pe_pb_left_core_slots
                if pe_pb_left else configured_left_core_slots
            )
            plan_by_slot = {
                item["slot"]: item
                for item in left_grid_plan(
                    grid_anchor,
                    initial_slots=effective_initial_slots,
                    core_slots=effective_core_slots,
                )
            }
            sold_slots = sold_grid_slots.setdefault(code, set())

            # Only non-core grid units have resting sell orders.  Selling a
            # right-side campaign elsewhere never reaches this list.
            for lot in list(state.left) if side == "sell" else []:
                if not lot_is_sellable(lot, date):
                    continue
                sell_price = lot.get("sell_price")
                if sell_price is None:
                    continue
                fill = fill_limit_order(
                    execution_bar(row), side="sell",
                    limit_price=qfq_to_raw_price(row, float(sell_price)),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
                if not fill["filled"] or not fill_matches_phase(fill):
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, float(fill["price"]), lot["cost"])
                state.left.remove(lot)
                sold_slots.add(int(lot["slot"]))
                add_event(
                    date, code, "左侧网格卖出", fill["price"], -size,
                    f"{lot['batch']}; 上一格卖出; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    **lot_risk_metadata(lot, quantity),
                    **with_raw_signal_price_metadata(row, {
                        "value_line": grid_anchor,
                        "account_mode": "left",
                    }),
                )

            if side != "buy":
                continue
            held_slots = {int(lot["slot"]) for lot in state.left + corporate_book.lots(code)}
            starting_new_left_plan = (
                not state.left_grid_started and code not in left_side_codes()
            )
            starting_new_symbol = starting_new_left_plan and code not in capacity_codes()
            if starting_new_left_plan:
                if (
                    total_symbol_limit_reached(code)
                    or left_symbol_limit_reached(code)
                    or (pe_pb_left and pe_pb_left_limit_reached(code))
                ):
                    continue
                if starting_new_symbol:
                    if symbol_limit_reached() or industry_limit_reached(code, candidate, date):
                        continue
            for planned in plan_by_slot.values():
                if not can_add_left:
                    continue
                slot = int(planned["slot"])
                if slot in held_slots or slot in sold_slots:
                    continue
                # Initial slots are capped by left_grid_max_exposure.  Lower
                # resting orders are only active once the capped initial grid
                # has actually started.
                if (
                    not state.left_grid_started
                    and slot >= effective_initial_slots
                    and not left_initial_slot_set(
                        effective_initial_slots
                    ).issubset(held_slots)
                ):
                    continue
                current_left = sum(float(lot["size"]) for lot in state.left)
                if (
                    not left_position_counts_capacity(state)
                    and not bool(planned["core"])
                    and (
                        symbol_limit_reached()
                        or total_symbol_limit_reached(code)
                        or left_symbol_limit_reached(code)
                        or industry_limit_reached(code, candidate, date)
                    )
                ):
                    continue
                requested_size = min(
                    configured_left_grid_unit,
                    effective_left_max_exposure - current_left,
                )
                requested_size = _capped_entry_size(
                    symbol_exposure(code), requested_size, max_symbol_exposure,
                )
                if requested_size <= 1e-9 or gross_exposure() + requested_size > 1.0 + 1e-9:
                    continue
                fill = fill_limit_order(
                    execution_bar(row), side="buy",
                    limit_price=qfq_to_raw_price(row, float(planned["buy_price"])),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
                if not fill["filled"] or not fill_matches_phase(fill):
                    continue
                size, quantity, pnl, entry_fee_cash, cash_limited = execute_buy(
                    code, requested_size, float(fill["price"]),
                )
                if size <= 1e-9:
                    continue
                lot = {
                    "cost": float(fill["price"]), "date": row["date"],
                    "signal_cost": raw_to_qfq_price(row, float(fill["price"])),
                    "buy_price": float(planned["buy_price"]),
                    "sell_price": planned["sell_price"], "size": size,
                    "quantity": quantity, "entry_fee_cash": entry_fee_cash,
                    "slot": slot, "batch": f"L{slot + 1}",
                    "core": bool(planned["core"]),
                    "industry_tags": sorted(
                        candidate_industry_tags(candidate)
                        or {
                            tag for existing in state.left
                            for tag in existing.get("industry_tags", [])
                        }
                    ),
                }
                if state.left_value_line is None:
                    state.left_value_line = grid_anchor
                if state.left_candidate_profile is None and candidate:
                    state.left_candidate_profile = dict(candidate)
                state.left.append(lot)
                held_slots.add(slot)
                if left_initial_slot_set(
                    effective_initial_slots
                ).issubset(held_slots):
                    state.left_grid_started = True
                add_event(
                    date, code, "左侧网格买入", fill["price"], size,
                    f"{lot['batch']}; 价值线网格第{slot + 1}份; {fill['status']}", pnl,
                    execution_quantity=quantity, entry_fee_cash=entry_fee_cash,
                    requested_position_pct=requested_size * 100,
                    cash_limited=cash_limited,
                    lot_rounded=size < requested_size - 1e-9 and not cash_limited,
                    board_lot_size=board_lot_size(code), grid_slot=slot,
                    **lot_risk_metadata(lot, quantity),
                    **with_raw_signal_price_metadata(row, {
                        "value_line": grid_anchor,
                        "account_mode": "left",
                        "candidate_snapshot_date": candidate.get(
                            "candidate_snapshot_date",
                        ),
                        "technical_signal_timing": (
                            "preplanned_value_grid_limit_no_technical_signal"
                        ),
                    }),
                )

    def execute_right_exits(date, exited_today):
        for code in sorted(occupied_codes()):
            state = states[code]
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            history = data.iloc[: index + 1]
            if execution_phase == "open":
                row = row.copy()
                for column in ("high", "low", "close"):
                    row[column] = row["open"]
                    row[f"raw_{column}"] = raw_price(row, "open")
                for column in row.index:
                    if str(column).startswith("ma") or "divergence" in str(column):
                        row[column] = data.iloc[index - 1].get(column) if index > 0 else None
                history = pd.concat([data.iloc[:index], row.to_frame().T], ignore_index=True)
            prior_close = float(data.iloc[index - 1]["close"]) if index > 0 else None
            if state.right:
                for lot in list(state.right):
                    if not lot_is_sellable(lot, date):
                        if execution_phase == "close" and float(row["close"]) < float(lot["stop"]):
                            state.pending_lot_exits.setdefault(lot["batch"], "condition stop; T+1")
                        continue
                    if not _entry_risk_still_controls_lot(lot, state):
                        continue
                    protective_stop_entry = bool(
                        lot.get("intraday_protective_stop")
                    )
                    if (
                        protective_stop_entry
                        and pd.Timestamp(row["date"]).normalize()
                        > pd.Timestamp(lot["date"]).normalize()
                    ):
                        previous_close_raw = (
                            raw_price(data.iloc[index - 1], "close")
                            if index > 0 else None
                        )
                        stop_raw = qfq_to_raw_price(row, float(lot["stop"]))
                        stop_fill = fill_sell_stop(
                            execution_bar(row), stop_price=stop_raw,
                            previous_close=previous_close_raw,
                            code=code, is_st=is_st(code),
                        )
                        if stop_fill["filled"] and fill_matches_phase(stop_fill):
                            quantity = float(lot["quantity"])
                            sell = float(stop_fill["price"])
                            entry_fee_cash = float(
                                lot.get("entry_fee_cash") or 0.0
                            )
                            size, pnl = execute_sell(
                                quantity, sell, lot["cost"],
                            )
                            state.right.remove(lot)
                            state.pending_lot_exits.pop(lot["batch"], None)
                            factor_horizon_entry = bool(
                                lot.get("factor_horizon_entry")
                            )
                            compressed_structure_entry = bool(
                                lot.get("compressed_structure_entry")
                            )
                            breadth_reversal_entry = bool(
                                lot.get("breadth_reversal_entry")
                            )
                            add_event(
                                date, code,
                                (
                                    "因子周期匹配保护止损"
                                    if factor_horizon_entry
                                    else "紧结构平台延续保护止损"
                                    if compressed_structure_entry
                                    else "低广度反转保护止损"
                                    if breadth_reversal_entry
                                    else "右侧结构保护止损"
                                ),
                                sell, -size,
                                (
                                    f"{lot['batch']}; "
                                    + (
                                        "固定失效距离保护止损"
                                        if factor_horizon_entry
                                        or compressed_structure_entry
                                        else "信号日低点保护止损"
                                        if breadth_reversal_entry
                                        else "预设结构保护止损"
                                    )
                                    + f"; {stop_fill['status']}"
                                ),
                                pnl, cost_basis=lot["cost"],
                                execution_quantity=-quantity,
                                entry_fee_cash=entry_fee_cash,
                                **lot_risk_metadata(lot, quantity),
                            )
                            exited_today.add(code)
                            continue
                        stop_touched = (
                            raw_price(row, "low") is not None
                            and raw_price(row, "low") <= stop_raw
                        )
                        if stop_touched and execution_phase == "close":
                            state.pending_lot_exits.setdefault(
                                lot["batch"], "condition stop",
                            )
                            continue
                        target = pd.to_numeric(
                            lot.get("fixed_target_price"), errors="coerce",
                        )
                        if (
                            lot.get("breadth_reversal_entry")
                            and pd.notna(target)
                            and float(target) > 0
                        ):
                            target_fill = fill_limit_order(
                                execution_bar(row), side="sell",
                                limit_price=qfq_to_raw_price(row, float(target)),
                                previous_close=previous_close_raw,
                                code=code, is_st=is_st(code),
                            )
                            if target_fill["filled"] and fill_matches_phase(target_fill):
                                quantity = float(lot["quantity"])
                                sell = float(target_fill["price"])
                                entry_fee_cash = float(
                                    lot.get("entry_fee_cash") or 0.0
                                )
                                size, pnl = execute_sell(
                                    quantity, sell, lot["cost"],
                                )
                                state.right.remove(lot)
                                state.pending_lot_exits.pop(
                                    lot["batch"], None,
                                )
                                add_event(
                                    date, code, "低广度反转3R止盈", sell, -size,
                                    (
                                        f"{lot['batch']}; 达到固定3R目标; "
                                        f"{target_fill['status']}"
                                    ),
                                    pnl, cost_basis=lot["cost"],
                                    execution_quantity=-quantity,
                                    entry_fee_cash=entry_fee_cash,
                                    **lot_risk_metadata(lot, quantity),
                                )
                                exited_today.add(code)
                                continue
                    factor_horizon_trend_managed = bool(
                        lot.get("factor_horizon_entry")
                        and configured_factor_horizon_trend_management_return
                        is not None
                        and float(lot.get("max_return") or 0.0)
                        >= configured_factor_horizon_trend_management_return
                    )
                    if factor_horizon_trend_managed:
                        factor_horizon_trend_managed_batches.add(lot["batch"])
                        factor_horizon_trend_managed_codes.add(code)
                        lot["factor_horizon_trend_managed"] = True
                    fixed_horizon_entry = bool(
                        lot.get("compressed_structure_entry")
                        or (
                            lot.get("factor_horizon_entry")
                            and not lot.get("factor_horizon_trend_managed")
                        )
                    )
                    if fixed_horizon_entry:
                        fixed_holding_days = (
                            configured_compressed_structure_holding_days
                            if lot.get("compressed_structure_entry")
                            else configured_factor_horizon_holding_days
                        )
                        entry_date = pd.Timestamp(lot["date"]).normalize()
                        holding_bars = int(
                            pd.to_datetime(
                                history["date"], errors="coerce",
                            ).dt.normalize().ge(entry_date).sum()
                        )
                        if execution_phase == "close" and holding_bars >= fixed_holding_days:
                            reason = (
                                f"固定持有{fixed_holding_days}"
                                "个交易日"
                            )
                            previous_close_raw = (
                                raw_price(data.iloc[index - 1], "close")
                                if index > 0 else None
                            )
                            if close_confirmed_execution == "close_proxy":
                                sell_status = close_proxy_sell_status(
                                    row, previous_close_raw, code,
                                )
                                if sell_status == "locked_limit_down":
                                    state.pending_lot_exits.setdefault(
                                        lot["batch"], reason,
                                    )
                                    continue
                                quantity = float(lot["quantity"])
                                sell = raw_price(row, "close")
                                entry_fee_cash = float(
                                    lot.get("entry_fee_cash") or 0.0
                                )
                                size, pnl = execute_sell(
                                    quantity, sell, lot["cost"],
                                )
                                state.right.remove(lot)
                                state.pending_lot_exits.pop(
                                    lot["batch"], None,
                                )
                                add_event(
                                    date, code,
                                    (
                                        "紧结构平台延续到期退出"
                                        if lot.get("compressed_structure_entry")
                                        else "因子周期匹配到期退出"
                                    ),
                                    sell, -size,
                                    f"{lot['batch']}; {reason}; 收盘代理",
                                    pnl, cost_basis=lot["cost"],
                                    execution_quantity=-quantity,
                                    entry_fee_cash=entry_fee_cash,
                                    **lot_risk_metadata(lot, quantity),
                                )
                                exited_today.add(code)
                            else:
                                state.pending_lot_exits.setdefault(
                                    lot["batch"], reason,
                                )
                            continue
                        continue
                    time_limit = int(lot.get("time_limit_days") or (8 if current_down_streak >= 3 else 13))
                    space_stop_pct = lot.get("space_stop_pct")
                    risk = position_exit_snapshot(
                        history, lot_signal_cost(lot), lot["date"], entry_mode="right",
                        condition_stop=lot["stop"], time_limit_days=time_limit,
                        space_stop_pct=space_stop_pct,
                        bearish_divergence=bool(
                            pd.to_numeric(row.get("bearish_divergence_count"), errors="coerce") >= 2
                        ),
                    )
                    if risk.get("space_stop_triggered"):
                        quantity = float(lot["quantity"])
                        stop_price = qfq_to_raw_price(row, float(risk["space_stop"]))
                        previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
                        fill = fill_sell_stop(
                            execution_bar(row), stop_price=stop_price,
                            previous_close=previous_close, code=code, is_st=is_st(code),
                        )
                        if not fill["filled"] or not fill_matches_phase(fill):
                            continue
                        execution_price = float(fill["price"])
                        execution_reason = f"hard space stop {stop_price:.3f}; {fill['status']}"
                        if lot.get("origin_account") == "left":
                            exit_promoted_left_lot(
                                code, state, lot, date, execution_price,
                                execution_reason,
                            )
                            exited_today.add(code)
                            continue
                        entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                        size, pnl = execute_sell(quantity, execution_price, lot["cost"])
                        state.right.remove(lot)
                        state.pending_lot_exits.pop(lot["batch"], None)
                        exited_today.add(code)
                        add_event(
                            date, code, risk["position_action"], execution_price,
                            -size, f"{lot['batch']}; {execution_reason}", pnl,
                            cost_basis=lot["cost"], execution_quantity=-quantity,
                            entry_fee_cash=entry_fee_cash,
                            **lot_risk_metadata(lot, quantity),
                        )
                    elif execution_phase == "close" and (risk.get("condition_stop_triggered") or risk.get("entry_time_stop")):
                        reason = "condition stop" if risk.get("condition_stop_triggered") else "entry time condition"
                        if close_confirmed_execution == "close_proxy":
                            previous_close_raw = (
                                raw_price(data.iloc[index - 1], "close") if index > 0 else None
                            )
                            sell_status = close_proxy_sell_status(
                                row, previous_close_raw, code,
                            )
                            if sell_status == "locked_limit_down":
                                state.pending_lot_exits.setdefault(lot["batch"], reason)
                                continue
                            if lot.get("origin_account") == "left":
                                exit_promoted_left_lot(
                                    code, state, lot, date, raw_price(row, "close"),
                                    f"{reason}; 14:55/close proxy",
                                )
                                exited_today.add(code)
                                continue
                            quantity = float(lot["quantity"])
                            sell = raw_price(row, "close")
                            entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                            size, pnl = execute_sell(quantity, sell, lot["cost"])
                            state.right.remove(lot)
                            state.pending_lot_exits.pop(lot["batch"], None)
                            add_event(
                                date, code, "收盘条件退出", sell, -size,
                                f"{lot['batch']}; {reason}; 14:55/close proxy", pnl,
                                cost_basis=lot["cost"], execution_quantity=-quantity,
                                entry_fee_cash=entry_fee_cash,
                                **lot_risk_metadata(lot, quantity),
                            )
                            exited_today.add(code)
                        else:
                            state.pending_lot_exits.setdefault(lot["batch"], reason)
                if execution_phase != "close":
                    continue
                if state.right and all(
                    lot.get("breadth_reversal_entry")
                    or lot.get("compressed_structure_entry")
                    or (
                        lot.get("factor_horizon_entry")
                        and not lot.get("factor_horizon_trend_managed")
                    )
                    for lot in state.right
                ):
                    continue
                if state.right:
                    if not any(lot["merged"] for lot in state.right):
                        state.right_parts = configured_profit_tranches
                        state.right_sold.clear()
                    merge_threshold = 0.20 if current_down_streak >= 3 else 0.10
                    merged_new_batch = False
                    for lot in state.right:
                        if lot_is_sellable(lot, date) and not lot["merged"] and prior_close is not None and prior_close / lot_signal_cost(lot) - 1 >= merge_threshold:
                            lot["merged"] = True
                            merged_new_batch = True
                            add_event(date, code, "加仓批次合并", row["close"], 0.0, f"{lot['batch']}; 浮盈达到{merge_threshold:.0%}")

                    if merged_new_batch:
                        # A new add-on belongs to the existing campaign.  Keep
                        # already executed profit tranches and the original
                        # peak-observation window instead of resetting both.
                        state.right_parts = max(1, state.right_parts)

                    merged = [lot for lot in state.right if lot["merged"] and lot_is_sellable(lot, date)]
                    if merged and state.right_parts > 0:
                        merged_quantity = sum(float(lot["quantity"]) for lot in merged)
                        merged_cost = sum(
                            float(lot["quantity"]) * lot_signal_cost(lot)
                            for lot in merged
                        ) / merged_quantity
                        merged_entry = state.right_plan_date or min(lot["date"] for lot in merged)
                        profit = position_exit_snapshot(
                            history, merged_cost, merged_entry, entry_mode="right",
                            condition_stop=None, time_limit_days=9999,
                            exit_tranches=configured_profit_tranches,
                        )
                        active_profit_ids = _active_profit_trigger_ids(
                            profit, history, state.right_parts,
                        )
                        new_ids = active_profit_ids - state.right_sold
                        if new_ids:
                            if close_confirmed_execution == "close_proxy":
                                executed_ids = _profit_ids_to_execute(
                                    new_ids, state.right_parts,
                                )
                                parts = len(executed_ids)
                                if not parts:
                                    continue
                                previous_close_raw = (
                                    raw_price(data.iloc[index - 1], "close") if index > 0 else None
                                )
                                sell_status = close_proxy_sell_status(
                                    row, previous_close_raw, code,
                                )
                                if sell_status == "locked_limit_down":
                                    state.pending_profit_ids.update(executed_ids)
                                    state.pending_tail_capacity_free = _qualifies_profit_tail(
                                        profit,
                                        state.right_parts - parts,
                                        configured_profit_tail_min_return,
                                    )
                                    continue
                                sell = raw_price(row, "close")
                                planned_ratio = parts / state.right_parts
                                desired_quantity = merged_quantity * planned_ratio
                                quantity = (
                                    merged_quantity if parts == state.right_parts
                                    else float(_board_lot_quantity(code, desired_quantity))
                                )
                                if quantity <= 0:
                                    continue
                                quantity, sold_cost, entry_fee_cash, risk_metadata = consume_merged_lots(
                                    code, state, merged, quantity,
                                )
                                size, pnl = execute_sell(quantity, sell, sold_cost)
                                state.right_sold.update(executed_ids)
                                state.right_parts -= parts
                                if _qualifies_profit_tail(
                                    profit, state.right_parts,
                                    configured_profit_tail_min_return,
                                ):
                                    state.right_tail_capacity_free = True
                                action = (
                                    "统一尾仓收盘退出" if state.right_parts == 0
                                    else f"统一分仓收盘止盈{parts}份"
                                )
                                add_event(
                                    date, code, action, sell, -size,
                                    f"止盈条件={','.join(executed_ids)}; 14:55/close proxy", pnl,
                                    cost_basis=sold_cost, execution_quantity=-quantity,
                                    entry_fee_cash=entry_fee_cash,
                                    **risk_metadata,
                                )
                                if state.right_parts == 0:
                                    state.right_parts = configured_profit_tranches
                                    state.right_sold.clear()
                                    state.right_plan_date = None
                                    state.right_tail_capacity_free = False
                                    exited_today.add(code)
                            else:
                                executable_ids = _profit_ids_to_execute(
                                    new_ids, state.right_parts,
                                )
                                state.pending_profit_ids.update(executable_ids)
                                scheduled_parts = len(executable_ids)
                                state.pending_tail_capacity_free = _qualifies_profit_tail(
                                    profit, state.right_parts - scheduled_parts,
                                    configured_profit_tail_min_return,
                                )
                    if not state.right:
                        state.right_parts = configured_profit_tranches
                        state.right_sold.clear()
                        state.right_plan_date = None


    def valuation_price(code, row, date):
        price = raw_price(row, "close")
        quote_date = pd.Timestamp(row["date"]).normalize()
        for action in corporate_book.actions_by_code.get(code, []):
            if quote_date < action["ex"] <= date and not action["ambiguous"]:
                price = max(0., (price - action["cash"]) / (1 + action["bonus"]))
        return price

    def record_equity(date):
        corporate_book.register(date, states)
        unrealized = 0.0
        pending_value = 0.
        for pending in corporate_book.pending:
            index = latest_frame_index(pending["code"], date)
            if index is not None:
                pending_value += float(pending["lot"]["quantity"]) * valuation_price(pending["code"], frames[pending["code"]].iloc[index], date)
        market_value = float(cash_balance) + corporate_book.receivable + pending_value
        for code in sorted(occupied_codes()):
            state = states[code]
            data = frames[code]
            latest_index = latest_frame_index(code, date)
            if latest_index is None:
                continue
            close = valuation_price(code, data.iloc[latest_index], date)
            if state.left or state.right:
                unrealized += sum(
                    float(lot["quantity"]) * (close - float(lot["cost"]))
                    / float(initial_capital)
                    for lot in state.left + state.right
                )
                market_value += sum(
                    float(lot["quantity"]) * close
                    for lot in state.left + state.right
                )
        equity = market_value / float(initial_capital)
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": float(cash_balance),
            "is_trading_day": date in market_calendar_set,
            "share_balances": {
                code: sum(float(lot["quantity"]) for lot in
                          states[code].left + states[code].right + corporate_book.lots(code))
                for code in sorted(occupied_codes())
            },
            "cash_pct": round(float(cash_balance) / market_value * 100, 4) if market_value else None,
            "gross_exposure_pct": round(gross_exposure() * 100, 2),
            "stock_market_value": market_value - cash_balance - corporate_book.receivable,
            "dividend_receivable": corporate_book.receivable,
            "unlisted_stock_market_value": pending_value,
            "market_exposure_pct": (market_value - cash_balance - corporate_book.receivable) / market_value * 100 if market_value else None,
            "cost_exposure_pct": gross_exposure() * 100,
            "gross_exposure_basis": "initial_capital_cost_exposure",
            "capacity_position_count": len(capacity_codes()),
            "total_held_symbol_count": len(occupied_codes()),
            "left_side_symbol_count": len(left_side_codes()),
            "right_market_left_side_symbol_count": (
                len(left_side_codes()) if right_market_active() else 0
            ),
            "profit_tail_count": sum(
                _is_profit_tail(state) for state in states.values()
            ),
            "candidate_count": len(current_candidates),
        })

    market_calendar = sorted({
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[frame["date"] <= end, "date"]
    })
    market_calendar_set = set(market_calendar)
    action_dates = {
        action[key] for action in corporate_book.actions for key in ("record", "ex", "pay", "listing")
        if action[key] is not None and start <= action[key] <= end
    }
    calendar = sorted({date for date in market_calendar if date >= start} | action_dates)
    next_calendar_date = {
        market_calendar[index]: market_calendar[index + 1]
        for index in range(len(market_calendar) - 1)
    }
    snapshot_dates = sorted(snapshots)
    formula_by_date = {
        pd.Timestamp(date).normalize(): value
        for date, value in formula_phases.items()
    }
    formula_dates = sorted(formula_by_date)
    risk_snapshot_index = 0
    for date in calendar:
        while risk_snapshot_index < len(snapshot_dates) and (
            snapshot_dates[risk_snapshot_index] < date
            or (not signals_effective_next_day and snapshot_dates[risk_snapshot_index] == date)
        ):
            for code, candidate in snapshots[snapshot_dates[risk_snapshot_index]].items():
                name = candidate.get("name")
                if name and str(name).lower() not in {"nan", "none"}:
                    risk_names_as_of[code] = str(name)
            risk_snapshot_index += 1
        execution_phase = "corporate_action"
        apply_corporate_actions(date)
        if date not in market_calendar_set:
            record_equity(date)
            continue
        begin_execution_phase("open")
        exited_today = set()
        for code in sorted(occupied_codes()):
            state = states[code]
            security_end = normalized_security_end_dates.get(code)
            if security_end is None or date < security_end:
                continue
            held_lots = list(state.left) + list(state.right) + corporate_book.lots(code)
            corporate_book.pending = [p for p in corporate_book.pending if p["code"] != code]
            if not held_lots:
                continue
            for lot in held_lots:
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, 0.0, lot["cost"])
                add_event(
                    date,
                    code,
                    "退市持仓归零",
                    0.0,
                    -size,
                    (
                        f"{lot['batch']}; 已到退市日"
                        f"{security_end:%Y-%m-%d}; 无可交易价格，保守按零核销"
                    ),
                    pnl,
                    cost_basis=lot["cost"],
                    execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    security_end_date=security_end.strftime("%Y-%m-%d"),
                    terminal_write_off=True,
                    **lot_risk_metadata(lot, quantity),
                )
            state.left.clear()
            state.right.clear()
            state.left_value_line = None
            state.left_grid_started = False
            state.left_candidate_profile = None
            state.right_parts = configured_profit_tranches
            state.right_sold.clear()
            state.right_plan_date = None
            state.pending_lot_exits.clear()
            state.pending_profit_ids.clear()
            state.pending_tail_capacity_free = False
            state.right_tail_capacity_free = False
            pending_candidate_exits.discard(code)
            pending_weak_exits.discard(code)
            pending_left_exits.pop(code, None)
            pending_left_quota_exits.pop(code, None)
            active_position_codes.discard(code)
            exited_today.add(code)
        # Close-confirmed exits become market-at-open orders on the next bar.
        for code in sorted(occupied_codes()):
            state = states[code]
            if not state.pending_lot_exits and not state.pending_profit_ids:
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            fill = fill_market_on_open(
                execution_bar(row), side="sell", previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                reason = state.pending_lot_exits.get(lot["batch"])
                if not reason:
                    continue
                if lot.get("origin_account") == "left":
                    exit_promoted_left_lot(
                        code, state, lot, date, sell,
                        f"{reason}; {fill['status']}",
                    )
                    exited_today.add(code)
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                state.pending_lot_exits.pop(lot["batch"], None)
                add_event(
                    date, code, "收盘条件次日退出", sell, -size,
                    f"{lot['batch']}; {reason}; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    **lot_risk_metadata(lot, quantity),
                )
                exited_today.add(code)

            merged = [lot for lot in state.right if lot.get("merged")]
            if state.pending_profit_ids and merged and state.right_parts > 0:
                executed_ids = _profit_ids_to_execute(
                    state.pending_profit_ids, state.right_parts,
                )
                parts = len(executed_ids)
                if not parts:
                    state.pending_profit_ids.clear()
                    state.pending_tail_capacity_free = False
                    continue
                merged_quantity = sum(float(lot["quantity"]) for lot in merged)
                planned_ratio = parts / state.right_parts
                desired_quantity = merged_quantity * planned_ratio
                quantity = (merged_quantity if parts == state.right_parts
                            else float(_board_lot_quantity(code, desired_quantity)))
                if quantity <= 0:
                    continue
                quantity, sold_cost, entry_fee_cash, risk_metadata = consume_merged_lots(
                    code, state, merged, quantity,
                )
                size, pnl = execute_sell(quantity, sell, sold_cost)
                state.right_sold.update(executed_ids)
                state.pending_profit_ids.difference_update(executed_ids)
                state.right_parts -= parts
                if state.right_parts == 1 and state.pending_tail_capacity_free:
                    state.right_tail_capacity_free = True
                state.pending_tail_capacity_free = False
                action = (
                    "统一尾仓次日退出" if state.right_parts == 0
                    else f"统一分仓次日止盈{parts}份"
                )
                add_event(
                    date, code, action, sell, -size,
                    f"止盈条件={','.join(executed_ids)}; 收盘确认; {fill['status']}", pnl,
                    cost_basis=sold_cost, execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    **risk_metadata,
                )
                exited_today.add(code)
            if not state.right:
                state.right_parts = configured_profit_tranches
                state.right_sold.clear()
                state.right_plan_date = None
                state.pending_lot_exits.clear()
                state.pending_profit_ids.clear()
                state.pending_tail_capacity_free = False
                state.right_tail_capacity_free = False

        eligible_formula_dates = [
            item for item in formula_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        formula_state = (
            formula_by_date[eligible_formula_dates[-1]]
            if eligible_formula_dates else current_phase
        )
        if isinstance(formula_state, dict):
            current_phase = str(formula_state.get("phase") or current_phase)
            current_down_streak = int(formula_state.get("window_down_streak") or 0)
            current_up_streak = int(formula_state.get("window_up_streak") or 0)
        else:
            current_phase = str(formula_state)
            current_down_streak = 0
            current_up_streak = 0

        if len(left_side_codes()) > configured_left_positions:
            def left_keep_rank(code):
                state = states[code]
                score = float(last_candidate_scores.get(code, 0.0) or 0.0)
                return (
                    bool(state.right),
                    left_unrealized_return_before(code, date),
                    score,
                    symbol_exposure(code),
                    code,
                )

            keep_codes = set(sorted(
                left_side_codes(), key=left_keep_rank, reverse=True,
            )[:configured_left_positions])
            keep_label = ",".join(sorted(keep_codes)) or "无"
            for code in left_side_codes() - keep_codes:
                if code in pending_left_exits or code in pending_left_quota_exits:
                    continue
                pending_left_quota_exits[code] = (
                    f"全行情左侧标的限额; 保留={keep_label}"
                )

        for code in sorted(occupied_codes()):
            state = states[code]
            pending_lots = [
                lot for lot in state.right
                if lot.get("reconfirm_on_next_day") and pd.Timestamp(lot["date"]).normalize() < date
            ]
            if not pending_lots:
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            for lot in pending_lots:
                level = float(lot["reconfirm_level"])
                if float(row["open"]) >= level:
                    lot["reconfirm_on_next_day"] = False
                    add_event(date, code, "突破次日确认持有", raw_price(row, "open"), 0.0, f"{lot['batch']}; 开盘仍在{level:.3f}上方")
                    continue
                fill = fill_market_on_open(
                    execution_bar(row), side="sell", previous_close=previous_close,
                    code=code, is_st=is_st(code),
                )
                if not fill["filled"]:
                    continue
                if lot.get("origin_account") == "left":
                    exit_promoted_left_lot(
                        code, state, lot, date, float(fill["price"]),
                        f"开盘未站上{level:.3f}; {fill['status']}",
                    )
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                sell = float(fill["price"])
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "突破次日未确认退出", sell, -size,
                    f"{lot['batch']}; 开盘未站上{level:.3f}; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    **lot_risk_metadata(lot, quantity),
                )
            if not state.right:
                state.right_parts = configured_profit_tranches
                state.right_sold.clear()
                state.right_plan_date = None
        for code in list(pending_candidate_exits):
            state = states.get(code)
            right_size = 0.0 if state is None else sum(lot["size"] for lot in state.right)
            if state is None or not state.right or right_size >= 0.10 - 1e-9:
                pending_candidate_exits.discard(code)
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            fill = fill_market_on_open(
                execution_bar(row), side="sell", previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "落选尾仓退出", sell, -size,
                    f"{lot['batch']}; 右侧总仓低于10%; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    **lot_risk_metadata(lot, quantity),
                )
            state.right_parts = configured_profit_tranches
            state.right_sold.clear()
            state.right_plan_date = None
            pending_candidate_exits.discard(code)
            exited_today.add(code)
        for code in list(pending_weak_exits):
            state = states.get(code)
            if state is None or not state.right:
                pending_weak_exits.discard(code)
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            fill = fill_market_on_open(
                execution_bar(row), side="sell", previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "汰弱退出", sell, -size,
                    f"{lot['batch']}; 低效率仓次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                    **lot_risk_metadata(lot, quantity),
                )
            state.right_parts = configured_profit_tranches
            state.right_sold.clear()
            state.right_plan_date = None
            pending_weak_exits.discard(code)
            exited_today.add(code)
        for code, reason in list(pending_left_exits.items()):
            state = states.get(code)
            if state is None or not state.left:
                pending_left_exits.pop(code, None)
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            fill = fill_market_on_open(
                execution_bar(row), side="sell", previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.left):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.left.remove(lot)
                add_event(
                    date, code, "左侧价值证伪清仓", sell, -size,
                    f"{lot['batch']}; {reason}; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    **lot_risk_metadata(lot, quantity),
                    value_line=state.left_value_line, account_mode="left",
                )
            state.left_value_line = None
            state.left_grid_started = False
            state.left_candidate_profile = None
            pending_left_exits.pop(code, None)
            exited_today.add(code)
        for code, reason in list(pending_left_quota_exits.items()):
            state = states.get(code)
            if state is None or not state.left:
                if state is not None:
                    state.left_value_line = None
                    state.left_grid_started = False
                    state.left_candidate_profile = None
                pending_left_quota_exits.pop(code, None)
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            row = data.iloc[index]
            previous_close = raw_price(data.iloc[index - 1], "close") if index > 0 else None
            fill = fill_market_on_open(
                execution_bar(row), side="sell", previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.left):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.left.remove(lot)
                add_event(
                    date, code, "左侧全行情限额清仓", sell, -size,
                    f"{lot['batch']}; {reason}; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    **lot_risk_metadata(lot, quantity),
                    value_line=state.left_value_line, account_mode="left",
                )
            state.left_value_line = None
            state.left_grid_started = False
            state.left_candidate_profile = None
            pending_left_quota_exits.pop(code, None)
            exited_today.add(code)
        eligible_snapshots = [
            item for item in snapshot_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        if eligible_snapshots:
            candidate_snapshot_date = eligible_snapshots[-1]
            current_candidates = {
                code: {
                    **candidate,
                    "candidate_snapshot_date": candidate_snapshot_date.strftime(
                        "%Y-%m-%d",
                    ),
                }
                for code, candidate in snapshots[candidate_snapshot_date].items()
            }
            for candidate_code, candidate in current_candidates.items():
                last_candidate_scores[candidate_code] = float(candidate.get("candidate_score") or 0.0)
                last_candidates[candidate_code] = dict(candidate)
        current_candidate_codes = set(current_candidates)
        if exit_tail_on_candidate_removal and previous_candidate_codes:
            for code in set(states) - current_candidate_codes:
                state = states.get(code)
                if state and state.right and sum(lot["size"] for lot in state.right) < 0.10 - 1e-9:
                    pending_candidate_exits.add(code)
        previous_candidate_codes = current_candidate_codes
        for code in sorted(left_side_codes()):
            state = states[code]
            if not state.left or code in pending_left_exits:
                continue
            reason = left_value_falsification_reason(current_candidates.get(code))
            if reason:
                pending_left_exits[code] = reason
        # Opening orders use previous-session state and settled inventory.
        for code in sorted(occupied_codes()):
            index = frame_index_on_date(code, date)
            if index is None or index == 0:
                continue
            prior_close = float(frames[code].iloc[index - 1]["close"])
            for lot in states[code].right:
                prior_return = prior_close / lot_signal_cost(lot) - 1
                lot["max_return"] = max(float(lot.get("max_return") or 0.0), prior_return)
                if prior_return >= 0.10:
                    lot["proven"] = True
        candidates = []
        resolved_compressed_structure_codes = set()
        for code in sorted(pending_compressed_structure_entries):
            pending = pending_compressed_structure_entries[code]
            execution_date = pd.Timestamp(
                pending["execution_date"]
            ).normalize()
            if date < execution_date:
                continue
            pending_compressed_structure_entries.pop(code, None)
            resolved_compressed_structure_codes.add(code)
            candidate = dict(pending["candidate"])
            signal = dict(pending["signal"])
            if date != execution_date:
                record_entry_block(
                    date, code, candidate, signal,
                    "compressed_structure_execution_date_missed",
                )
                continue
            if code not in states:
                record_entry_block(
                    date, code, candidate, signal,
                    "compressed_structure_price_frame_missing",
                )
                continue
            index = frame_index_on_date(code, date)
            if index is None:
                record_entry_block(
                    date, code, candidate, signal,
                    "compressed_structure_no_trade_on_execution_date",
                )
                continue
            if code in exited_today or code in occupied_codes():
                record_entry_block(
                    date, code, candidate, signal,
                    "compressed_structure_symbol_already_held",
                )
                continue
            data = frames[code]
            signal_open = pd.to_numeric(
                data.iloc[index].get("open"), errors="coerce",
            )
            if pd.isna(signal_open) or float(signal_open) <= 0:
                record_entry_block(
                    date, code, candidate, signal,
                    "compressed_structure_missing_entry_price",
                )
                continue
            planned_size = min(
                configured_compressed_structure_max_position,
                configured_compressed_structure_risk_budget
                / configured_compressed_structure_stop_loss_pct,
            )
            signal.update({
                "trigger": float(signal_open),
                "stop": float(signal_open) * (
                    1.0 - configured_compressed_structure_stop_loss_pct
                ),
                "order_type": "limit",
                "reason": (
                    "紧结构平台延续；下一交易日开盘执行；"
                    f"固定{configured_compressed_structure_stop_loss_pct:.0%}保护止损；"
                    f"持有{configured_compressed_structure_holding_days}个交易日"
                ),
                "entry_channel": "compressed_structure_continuation",
                "semantic_entry_ok": True,
                "semantic_entry_setup": "compressed_structure_continuation",
                "entry_gate_reason": "compressed_structure_rule_matched",
                "entry_risk_pct": round(
                    configured_compressed_structure_stop_loss_pct * 100, 3,
                ),
                "planned_position_size": planned_size,
                "requires_next_day_confirmation": False,
                "technical_signal_timing": (
                    "候选日与平台信号在收盘冻结，下一全市场交易日开盘执行"
                ),
            })
            fundamental_score = float(candidate.get("candidate_score") or 0.0)
            entry_evidence_score = float(
                signal.get("entry_evidence_score") or 0.0
            )
            structure_proximity_score = float(
                signal.get("structure_proximity_score") or 0.0
            )
            factor_rank = float(
                _channel_candidate_number(
                    candidate, "compressed_structure", "playbook_factor_rank",
                ) or 999.0
            )
            entry_priority = (
                entry_evidence_score * 1000.0
                + structure_proximity_score * 10.0
                - factor_rank
            )
            candidates.append((
                entry_priority, fundamental_score, signal.get("rank") or 0,
                code, index, signal, candidate,
                "compressed_structure_continuation",
            ))

        resolved_standard_next_open_codes = set()
        for code in sorted(pending_standard_right_close_entries):
            pending = pending_standard_right_close_entries[code]
            execution_date = pd.Timestamp(
                pending["execution_date"]
            ).normalize()
            if date < execution_date:
                continue
            pending_standard_right_close_entries.pop(code, None)
            resolved_standard_next_open_codes.add(code)
            candidate = dict(pending["candidate"])
            signal = dict(pending["signal"])
            if date != execution_date:
                signal["execution_status"] = (
                    "standard_right_next_open_execution_date_missed"
                )
                record_entry_block(
                    date, code, candidate, signal,
                    "standard_right_next_open_execution_date_missed",
                )
                continue
            if code not in states:
                signal["execution_status"] = (
                    "standard_right_next_open_price_frame_missing"
                )
                record_entry_block(
                    date, code, candidate, signal,
                    "standard_right_next_open_price_frame_missing",
                )
                continue
            index = frame_index_on_date(code, date)
            if index is None:
                signal["execution_status"] = (
                    "standard_right_next_open_no_kline"
                )
                record_entry_block(
                    date, code, candidate, signal,
                    "standard_right_next_open_no_kline",
                )
                continue
            if code in exited_today:
                signal["execution_status"] = (
                    "standard_right_next_open_symbol_exited_today"
                )
                record_entry_block(
                    date, code, candidate, signal,
                    "standard_right_next_open_symbol_exited_today",
                )
                continue
            candidates.append((
                pending["entry_priority"], pending["fundamental_score"],
                pending["signal_rank"], code, index, signal, candidate,
                "standard_right_entry",
            ))

        resolved_preplaced_codes = set()
        for code, pending in list(pending_preplaced_right_entries.items()):
            if date < pending["execution_date"]:
                continue
            pending_preplaced_right_entries.pop(code)
            resolved_preplaced_codes.add(code)
            index = frame_index_on_date(code, date)
            if date != pending["execution_date"] or index is None or code in exited_today:
                continue
            candidates.append((pending["entry_priority"], pending["fundamental_score"],
                               pending["signal"]["rank"], code, index,
                               pending["signal"], pending["candidate"], "standard_right_entry"))

        for code in sorted(current_candidates):
            if code not in states or code in exited_today:
                continue
            candidate = current_candidates[code]
            candidate_is_current = True
            if not right_candidate_can_evaluate(candidate):
                continue
            if not _enabled(candidate.get("allow_right"), default=True):
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None:
                continue
            factor_horizon_candidate = (
                configured_factor_horizon_enabled
                and candidate_is_current
                and not states[code].left
                and _enabled(
                    candidate.get("factor_horizon_channel_eligible"),
                    default=False,
                )
            )
            if factor_horizon_candidate and not (
                states[code].left or states[code].right
            ):
                if (
                    configured_factor_horizon_exclude_watch_phase
                    and current_phase == "watch"
                ):
                    record_entry_block(
                        date, code, candidate, {},
                        "factor_horizon_watch_phase_excluded",
                    )
                    continue
                factor_rank = _channel_candidate_number(
                    candidate, "factor_horizon", "playbook_factor_rank",
                )
                structure_score = _channel_candidate_number(
                    candidate, "factor_horizon", "structure_proximity_score",
                )
                if structure_score is None:
                    structure_score = pd.to_numeric(
                        data.iloc[index - 1].get("structure_proximity_score") if index > 0 else None,
                        errors="coerce",
                    )
                    structure_score = (
                        float(structure_score)
                        if pd.notna(structure_score) else None
                    )
                average_amount = _channel_candidate_number(
                    candidate, "factor_horizon", "avg_amount_20",
                )
                return_60d = _channel_candidate_number(
                    candidate, "factor_horizon", "return_60d",
                )
                if (
                    factor_rank is None
                    or factor_rank > 10
                    or structure_score is None
                    or structure_score <= 40
                ):
                    record_entry_block(
                        date, code, candidate, {},
                        "factor_horizon_rule_mismatch",
                    )
                    continue
                if (
                    configured_factor_horizon_max_return_60d is not None
                    and (
                        return_60d is None
                        or return_60d
                        > configured_factor_horizon_max_return_60d
                    )
                ):
                    record_entry_block(
                        date, code, candidate, {},
                        "factor_horizon_return_60d_filter_mismatch",
                    )
                    continue
                if (
                    configured_factor_horizon_require_ma20_not_rising
                    and _enabled(
                        candidate.get("ma20_rising"), default=True,
                    )
                ):
                    record_entry_block(
                        date, code, candidate, {},
                        "factor_horizon_ma20_filter_mismatch",
                    )
                    continue
                if (
                    average_amount is None
                    or average_amount < configured_factor_horizon_min_avg_amount
                ):
                    record_entry_block(
                        date, code, candidate, {}, "liquidity_too_thin",
                    )
                    continue
                signal_open = pd.to_numeric(
                    data.iloc[index].get("open"), errors="coerce",
                )
                if pd.isna(signal_open) or float(signal_open) <= 0:
                    record_entry_block(
                        date, code, candidate, {}, "missing_entry_price",
                    )
                    continue
                planned_size = min(
                    configured_factor_horizon_max_position,
                    configured_factor_horizon_risk_budget
                    / configured_factor_horizon_stop_loss_pct,
                )
                factor_horizon_rule_parts = [
                    "训练期冻结的因子前10名与结构接近分大于40条件",
                ]
                if configured_factor_horizon_max_return_60d is not None:
                    factor_horizon_rule_parts.append(
                        "60日收益不高于"
                        f"{configured_factor_horizon_max_return_60d:.1%}"
                    )
                if configured_factor_horizon_require_ma20_not_rising:
                    factor_horizon_rule_parts.append("MA20尚未上扬")
                if configured_factor_horizon_exclude_watch_phase:
                    factor_horizon_rule_parts.append(
                        "Formula33观察阶段不买入"
                    )
                factor_horizon_exit_plan = (
                    "历史最大浮盈达到"
                    f"{configured_factor_horizon_trend_management_return:.1%}"
                    "后转趋势管理，否则"
                    f"{configured_factor_horizon_holding_days}日到期"
                    if configured_factor_horizon_trend_management_return
                    is not None
                    else f"计划持有{configured_factor_horizon_holding_days}日"
                )
                signal = {
                    "rank": 4,
                    "trigger": float(signal_open),
                    "stop": float(signal_open) * (
                        1.0 - configured_factor_horizon_stop_loss_pct
                    ),
                    "order_type": "limit",
                    "reason": "；".join(factor_horizon_rule_parts) + (
                        f"；{factor_horizon_exit_plan}"
                    ),
                    "known_volume_ratio": float(
                        _candidate_number(
                            candidate, "known_volume_ratio",
                        ) or 1.0
                    ),
                    "signal_type": "factor_horizon_match",
                    "entry_channel": "factor_horizon_match",
                    "semantic_entry_ok": True,
                    "semantic_entry_setup": "factor_horizon_match",
                    "entry_gate_reason": "factor_horizon_rule_matched",
                    "entry_risk_pct": round(
                        configured_factor_horizon_stop_loss_pct * 100,
                        3,
                    ),
                    "planned_position_size": planned_size,
                    "entry_liquidity_floor": (
                        configured_factor_horizon_min_avg_amount
                    ),
                    "technical_signal_date": candidate.get(
                        "candidate_snapshot_date",
                    ),
                    "technical_signal_timing": (
                        "候选日收盘冻结，下一交易日开盘执行"
                    ),
                }
                fundamental_score = float(
                    _channel_candidate_number(
                        candidate, "factor_horizon", "candidate_score",
                    ) or 0.0
                )
                entry_priority = fundamental_score + 100.0
                candidates.append((
                    entry_priority, fundamental_score, signal["rank"],
                    code, index, signal, candidate,
                    "factor_horizon_match",
                ))
                continue
            if factor_horizon_candidate and states[code].right:
                continue
        left_right_switch_signals = {}
        left_right_switch_candidates = {}
        left_right_switch_candidate_current = {}
        # Existing grids remain active even after dropping out of the daily
        # top ten. New campaigns, however, compete by candidate score so file
        # or ticker ordering can never decide which four symbols get capital.
        active_left_codes = [
            code for code in left_side_codes()
            for state in (states[code],)
            if state.left_value_line is not None and (state.left or state.left_grid_started)
        ]
        new_left_candidates = []
        for code, candidate in current_candidates.items():
            if (
                code not in states
                or states[code].left_value_line is not None
                or states[code].right
            ):
                continue
            if not left_candidate_can_add(candidate):
                continue
            value_line = pd.to_numeric(candidate.get("value_line"), errors="coerce")
            if pd.notna(value_line) and float(value_line) > 0:
                new_left_candidates.append((
                    0 if left_candidate_kind(candidate) == "value" else 1,
                    float(candidate.get("candidate_score") or 0.0),
                    code, float(value_line),
                ))
        left_targets = [
            (code, float(states[code].left_value_line), False)
            for code in active_left_codes
        ] + [
            (code, value_line, True)
            for _, _, code, value_line in sorted(
                new_left_candidates,
                key=lambda item: (item[0], -item[1], item[2]),
            )
        ]

        sold_grid_slots = {}
        execute_left_grids(left_targets, date, "sell", sold_grid_slots)
        execute_right_exits(date, exited_today)
        execute_right_entries(candidates, date, exited_today, {})
        execute_left_grids(left_targets, date, "buy", sold_grid_slots)
        begin_execution_phase("intraday")
        execute_right_entries(candidates, date, exited_today, {})
        execute_left_grids(left_targets, date, "buy", sold_grid_slots)
        execute_left_grids(left_targets, date, "sell", sold_grid_slots)
        execute_right_exits(date, exited_today)
        begin_execution_phase("close")
        for code in sorted(left_side_codes()):
            state = states[code]
            if not state.left or state.right:
                continue
            current_candidate = current_candidates.get(code)
            switch_candidate = current_candidate or state.left_candidate_profile or {}
            signal = left_to_right_switch_signal(
                code, date, switch_candidate,
            )
            if signal:
                left_right_switch_signals[code] = signal
                left_right_switch_candidates[code] = dict(switch_candidate)
                left_right_switch_candidate_current[code] = current_candidate is not None
        execute_right_exits(date, exited_today)

        candidates = []
        entry_allowed = (
            current_down_streak < 3
            or not capacity_codes()
            or has_twenty_percent_float(date)
        )
        resolved_breadth_reversal_codes = set()
        for code in sorted(pending_breadth_reversal_entries):
            pending = pending_breadth_reversal_entries[code]
            if code not in states:
                pending_breadth_reversal_entries.pop(code, None)
                continue
            data = frames[code]
            index = frame_index_on_date(code, date)
            if index is None or index <= int(pending["signal_index"]):
                continue
            pending_breadth_reversal_entries.pop(code, None)
            resolved_breadth_reversal_codes.add(code)
            candidate = dict(pending["candidate"])
            signal = dict(pending["signal"])
            confirmed, signal, confirmation_reason = (
                _confirm_bull_half_reversal_signal(data, index, signal)
            )
            observation = {
                "signal_date": signal.get("bull_half_signal_date"),
                "confirmation_date": date.strftime("%Y-%m-%d"),
                "code": code,
                "name": candidate.get("name") or code,
                "signal_type": "bull_run_half_pullback",
                "entry_channel": "breadth_reversal_half_pullback",
                "formula_window_unique_count": signal.get(
                    "formula_window_unique_count"
                ),
                "status": (
                    "confirmed" if confirmed else confirmation_reason
                ),
            }
            breadth_reversal_observations.append(observation)
            if not confirmed:
                record_entry_block(
                    date, code, candidate, signal, confirmation_reason,
                )
                continue
            if code in exited_today or code in occupied_codes():
                record_entry_block(
                    date, code, candidate, signal,
                    "reversal_symbol_already_held",
                )
                continue
            signal["entry_channel"] = "breadth_reversal_half_pullback"
            ok, signal, gate_reason = _semantic_bull_half_reversal_gate(
                data, index, candidate, signal,
            )
            if not ok:
                record_entry_block(
                    date, code, candidate, signal, gate_reason,
                )
                continue
            risk_pct = float(signal["entry_risk_pct"]) / 100.0
            signal["planned_position_size"] = min(
                configured_breadth_reversal_max_position,
                configured_breadth_reversal_risk_budget / risk_pct,
            )
            fundamental_score = float(
                _channel_candidate_number(
                    candidate, "compressed_structure", "candidate_score",
                ) or 0.0
            )
            entry_priority = (
                fundamental_score
                + float(signal["rank"]) * 8
                + min(float(signal["known_volume_ratio"]), 5.0) * 2
            )
            candidates.append((
                entry_priority, fundamental_score, signal["rank"],
                code, index, signal, candidate,
                "breadth_reversal_half_pullback",
            ))


        if (
            entry_allowed
            or configured_breadth_reversal_enabled
            or configured_factor_horizon_enabled
            or configured_compressed_structure_enabled
            or bool(candidates)
        ):
            right_entry_codes = sorted(
                set(current_candidates) | set(left_right_switch_signals) | capacity_codes()
            )
            for code in right_entry_codes:
                if (
                    code not in states
                    or code in exited_today
                    or code in resolved_breadth_reversal_codes
                    or code in resolved_compressed_structure_codes
                    or code in resolved_standard_next_open_codes
                    or code in resolved_preplaced_codes
                ):
                    continue
                candidate_is_current = code in current_candidates
                candidate = current_candidates.get(code) or (
                    left_right_switch_candidates.get(code, {})
                    if code in left_right_switch_signals else {}
                )
                if code not in left_right_switch_signals:
                    if not candidate_is_current:
                        continue
                    if not right_candidate_can_evaluate(candidate):
                        continue
                allow_right = _enabled(candidate.get("allow_right"), default=True)
                # A pure value candidate can switch only after the left
                # position has left the value zone and a right signal appears.
                if not allow_right and not states[code].left:
                    continue
                if states[code].left and code not in left_right_switch_signals:
                    continue
                data = frames[code]
                index = frame_index_on_date(code, date)
                if index is None:
                    continue
                if (
                    configured_factor_horizon_enabled
                    and _enabled(candidate.get("factor_horizon_channel_eligible"), default=False)
                    and code not in left_right_switch_signals
                ):
                    continue
                compressed_structure_candidate = (
                    configured_compressed_structure_enabled
                    and code not in left_right_switch_signals
                    and (
                        _enabled(
                            candidate.get(
                                "compressed_structure_channel_eligible"
                            ),
                            default=False,
                        )
                        or (
                            configured_factor_only_compressed_structure_only
                            and _factor_only_selection_profile(candidate)
                        )
                    )
                )
                breadth_reversal_candidate = (
                    configured_breadth_reversal_enabled
                    and code not in left_right_switch_signals
                    and (
                        _enabled(
                            candidate.get(
                                "breadth_reversal_channel_eligible"
                            ),
                            default=False,
                        )
                        or (
                            configured_factor_only_breadth_reversal_only
                            and _factor_only_selection_profile(candidate)
                        )
                    )
                )
                if compressed_structure_candidate and not breadth_reversal_candidate:
                    continue
                signal = left_right_switch_signals.get(code) or _right_signal(
                    data, index, plans.get(code),
                    auto_price_structure=auto_price_structure,
                    allow_structure_pullback=allow_structure_pullback,
                )
                if (
                    not signal
                    and states[code].right
                    and any(lot.get("proven", False) for lot in states[code].right)
                ):
                    signal = _leader_trend_add_signal(data, index)

                if breadth_reversal_candidate:
                    if not signal:
                        continue
                    if signal.get("signal_type") != "bull_run_half_pullback":
                        record_entry_block(
                            date, code, candidate, signal,
                            "factor_only_non_reversal_signal_disabled",
                        )
                        continue
                    if states[code].left or states[code].right:
                        record_entry_block(
                            date, code, candidate, signal,
                            "reversal_symbol_already_held",
                        )
                        continue
                    signal = _prepare_bull_half_reversal_signal(
                        data, index, signal,
                    )
                    formula_observation = formula_by_date.get(date)
                    unique_count = None
                    if isinstance(formula_observation, dict):
                        unique_value = pd.to_numeric(
                            formula_observation.get("window_unique_count"),
                            errors="coerce",
                        )
                        if pd.notna(unique_value):
                            unique_count = int(unique_value)
                    signal["entry_channel"] = "breadth_reversal_half_pullback"
                    signal["formula_observation_date"] = date.strftime(
                        "%Y-%m-%d",
                    )
                    signal["formula_window_unique_count"] = unique_count
                    if unique_count is None:
                        record_entry_block(
                            date, code, candidate, signal,
                            "breadth_unique_count_missing",
                        )
                        continue
                    if unique_count >= configured_breadth_reversal_max_unique_count:
                        record_entry_block(
                            date, code, candidate, signal,
                            "breadth_not_low_enough",
                        )
                        continue
                    average_amount = _channel_candidate_number(
                        candidate, "breadth_reversal", "avg_amount_20",
                    )
                    if average_amount is None or average_amount < 500_000_000.0:
                        record_entry_block(
                            date, code, candidate, signal, "liquidity_too_thin",
                        )
                        continue
                    return_60d = _channel_candidate_number(
                        candidate, "breadth_reversal", "return_60d",
                    )
                    if return_60d is None or return_60d > -0.10:
                        record_entry_block(
                            date, code, candidate, signal,
                            "reversal_60d_decline_not_deep_enough",
                        )
                        continue
                    pending_breadth_reversal_entries[code] = {
                        "candidate": dict(candidate),
                        "signal": dict(signal),
                        "signal_index": index,
                    }
                    breadth_reversal_observations.append({
                        "signal_date": date.strftime("%Y-%m-%d"),
                        "confirmation_date": None,
                        "code": code,
                        "name": candidate.get("name") or code,
                        "signal_type": "bull_run_half_pullback",
                        "entry_channel": "breadth_reversal_half_pullback",
                        "formula_window_unique_count": unique_count,
                        "status": "awaiting_next_close_confirmation",
                    })
                    continue

                if not entry_allowed:
                    continue
                if not signal:
                    continue
                if signal.get("signal_type") == "uptrend_support_pullback":
                    switching_left_to_right = code in left_right_switch_signals
                    if not states[code].right and not switching_left_to_right:
                        if not leading_pullback_pilot_starter(candidate, signal):
                            record_entry_block(date, code, candidate, signal, "support_pullback_not_first_entry")
                            continue
                    if not symbol_has_float_profit(code, date, add_on_float_threshold()):
                        if states[code].right:
                            record_entry_block(date, code, candidate, signal, "add_current_profit_not_met")
                            continue
                ok, signal, gate_reason = _semantic_right_entry_gate(
                    data, index, candidate, signal,
                    current_down_streak=current_down_streak,
                    management_transition=code in left_right_switch_signals,
                    price_amount_multiplier=(
                        configured_price_amount_multiplier
                    ),
                    factor_only_min_avg_amount=(
                        configured_factor_only_min_avg_amount
                    ),
                )
                if not ok:
                    record_entry_block(date, code, candidate, signal, gate_reason)
                    continue
                fundamental_score = float(candidate.get("candidate_score") or 0.0)
                entry_priority = (
                    fundamental_score
                    + float(signal["rank"]) * 8
                    + min(float(signal["known_volume_ratio"]), 5.0) * 2
                )
                if signal.get("order_type") != "close":
                    if _signal_number(signal, "structure_proximity_score") is None:
                        proximity = _finite_float(data.iloc[index].get("structure_proximity_score"))
                        if proximity is not None:
                            signal["structure_proximity_score"] = proximity
                            signal["structure_proximity_score_source"] = "technical_signal_date_visible_price_history"
                    execution_date = next_calendar_date.get(date)
                    if execution_date is not None:
                        target = _signal_number(signal, "entry_target_price")
                        if target is None:
                            target, _ = _planned_right_target(
                                data, index, signal, float(signal["trigger"]),
                                float(signal["trigger"]) - float(signal["stop"]), candidate)
                        signal = {**signal, "technical_signal_date": date.strftime("%Y-%m-%d"),
                                  "technical_signal_close_raw_price": raw_price(data.iloc[index], "close"),
                                  "frozen_qfq_to_raw_factor": float(data.iloc[index]["qfq_to_raw_factor"]),
                                  "frozen_structure_stop_raw_price": qfq_to_raw_price(data.iloc[index], signal["stop"]),
                                  "frozen_entry_target_raw_price": qfq_to_raw_price(data.iloc[index], target),
                                  "technical_signal_timing": "previous_close_preplaced_order",
                                  "frozen_formula_down_streak": current_down_streak,
                                  "frozen_formula_up_streak": current_up_streak}
                        pending_preplaced_right_entries[code] = {
                            "execution_date": execution_date, "candidate": dict(candidate),
                            "signal": signal, "entry_priority": entry_priority,
                            "fundamental_score": fundamental_score}
                    continue
                if (
                    close_confirmed_execution == "next_open"
                    and signal.get("order_type") == "close"
                ):
                    execution_date = next_calendar_date.get(date)
                    frozen_target = _signal_number(
                        signal,
                        "entry_target_price",
                    )
                    if frozen_target is None:
                        signal_close = float(data.iloc[index]["close"])
                        signal_stop = float(signal["stop"])
                        frozen_target, target_basis = _planned_right_target(
                            data,
                            index,
                            signal,
                            signal_close,
                            signal_close - signal_stop,
                            candidate,
                        )
                        minimum_rr_target = signal_close + (
                            signal_close - signal_stop
                        ) * 2.8
                        if frozen_target < minimum_rr_target:
                            frozen_target = minimum_rr_target
                            target_basis = (
                                "planned_2_8r_trend_continuation"
                            )
                        signal["entry_target_price"] = frozen_target
                        signal["entry_target_basis"] = target_basis
                    structure_proximity = _signal_number(
                        signal, "structure_proximity_score",
                    )
                    if structure_proximity is None:
                        structure_proximity = pd.to_numeric(
                            data.iloc[index].get("structure_proximity_score"),
                            errors="coerce",
                        )
                        if pd.notna(structure_proximity):
                            signal["structure_proximity_score"] = float(
                                structure_proximity
                            )
                            signal["structure_proximity_score_source"] = (
                                "technical_signal_date_visible_price_history"
                            )
                    signal.update({
                        "technical_signal_date": date.strftime("%Y-%m-%d"),
                        "technical_signal_timing": (
                            "previous_market_close_confirmed_next_open"
                        ),
                        "technical_signal_close": float(
                            data.iloc[index]["close"]
                        ),
                        "technical_signal_close_raw_price": raw_price(
                            data.iloc[index], "close",
                        ),
                        "execution_date": (
                            execution_date.strftime("%Y-%m-%d")
                            if execution_date is not None else None
                        ),
                        "execution_timing": "next_market_open_once",
                        "execution_status": "awaiting_next_market_open",
                        "standard_right_next_open": True,
                        "frozen_qfq_to_raw_factor": float(data.iloc[index]["qfq_to_raw_factor"]),
                        "frozen_structure_stop_raw_price": qfq_to_raw_price(
                            data.iloc[index], signal["stop"],
                        ),
                        "frozen_entry_target_raw_price": qfq_to_raw_price(
                            data.iloc[index], frozen_target,
                        ),
                        "frozen_formula_phase": current_phase,
                        "frozen_formula_down_streak": current_down_streak,
                        "frozen_formula_up_streak": current_up_streak,
                        "frozen_left_to_right_candidate_current": (
                            left_right_switch_candidate_current.get(
                                code, False,
                            )
                        ),
                        "frozen_pullback_pilot": (
                            leading_pullback_pilot_starter(candidate, signal)
                        ),
                        "frozen_pullback_pilot_size": (
                            leading_pullback_pilot_size(candidate, signal)
                        ),
                        "frozen_high_conviction_size": (
                            high_conviction_right_entry_size(candidate, signal)
                        ),
                    })
                    if execution_date is None:
                        signal["execution_status"] = (
                            "standard_right_next_open_date_unavailable"
                        )
                        record_entry_block(
                            date, code, candidate, signal,
                            "standard_right_next_open_date_unavailable",
                        )
                        continue
                    pending_standard_right_close_entries[code] = {
                        "candidate": dict(candidate),
                        "signal": dict(signal),
                        "signal_index": index,
                        "execution_date": execution_date,
                        "entry_priority": entry_priority,
                        "fundamental_score": fundamental_score,
                        "signal_rank": signal["rank"],
                    }
                    continue
                candidates.append((
                    entry_priority, fundamental_score, signal["rank"],
                    code, index, signal, candidate, "standard_right_entry",
                ))

            if configured_compressed_structure_enabled:
                signal_date_candidates = snapshots.get(date, {})
                for code in sorted(signal_date_candidates):
                    if (
                        code not in states
                        or code in resolved_compressed_structure_codes
                        or code in pending_compressed_structure_entries
                        or code in occupied_codes()
                    ):
                        continue
                    candidate = {
                        **signal_date_candidates[code],
                        "candidate_snapshot_date": date.strftime("%Y-%m-%d"),
                    }
                    if (
                        not (
                            _enabled(
                                candidate.get(
                                    "compressed_structure_channel_eligible"
                                ),
                                default=False,
                            )
                            or (
                                configured_factor_only_compressed_structure_only
                                and _factor_only_selection_profile(candidate)
                            )
                        )
                        or not right_candidate_can_evaluate(candidate)
                    ):
                        continue
                    data = frames[code]
                    index = frame_index_on_date(code, date)
                    if index is None:
                        continue
                    signal = _right_signal(
                        data, index, plans.get(code),
                        auto_price_structure=auto_price_structure,
                        allow_structure_pullback=allow_structure_pullback,
                    )
                    if not signal:
                        continue
                    if signal.get("signal_type") != "consolidation_breakout":
                        record_entry_block(
                            date, code, candidate, signal,
                            "compressed_structure_non_platform_signal",
                        )
                        continue
                    original_gate_candidate = dict(candidate)
                    for field in (
                        "avg_amount_20", "return_20d", "return_60d",
                    ):
                        original_gate_candidate.pop(field, None)
                    ok, gated_signal, gate_reason = _semantic_right_entry_gate(
                        data, index, original_gate_candidate, signal,
                        current_down_streak=current_down_streak,
                    )
                    if ok or gate_reason not in {
                        "risk_pct_too_tight",
                        "close_confirmed_risk_pct_too_tight",
                    }:
                        record_entry_block(
                            date, code, candidate, gated_signal,
                            "compressed_structure_original_gate_mismatch",
                        )
                        continue
                    entry_evidence_score = _signal_number(
                        gated_signal, "entry_evidence_score",
                    )
                    structure_proximity_score = _channel_candidate_number(
                        candidate,
                        "compressed_structure",
                        "structure_proximity_score",
                    )
                    structure_proximity_score_source = "candidate_snapshot"
                    if structure_proximity_score is None:
                        structure_proximity_score = pd.to_numeric(
                            data.iloc[index].get("structure_proximity_score"),
                            errors="coerce",
                        )
                        structure_proximity_score = (
                            float(structure_proximity_score)
                            if pd.notna(structure_proximity_score) else None
                        )
                        structure_proximity_score_source = (
                            "signal_date_visible_price_history"
                        )
                    gated_signal.update({
                        "structure_proximity_score": structure_proximity_score,
                        "structure_proximity_score_source": (
                            structure_proximity_score_source
                        ),
                    })
                    average_amount = _channel_candidate_number(
                        candidate, "compressed_structure", "avg_amount_20",
                    )
                    gated_signal.update({
                        "entry_liquidity_floor": (
                            configured_compressed_structure_min_avg_amount
                        ),
                        "entry_liquidity_profile": (
                            "紧结构专用日均成交额门"
                        ),
                        "candidate_avg_amount_20": average_amount,
                    })
                    if (
                        configured_compressed_structure_min_avg_amount > 0
                        and (
                            average_amount is None
                            or average_amount
                            < configured_compressed_structure_min_avg_amount
                        )
                    ):
                        record_entry_block(
                            date, code, candidate, gated_signal,
                            "compressed_structure_liquidity_too_thin",
                        )
                        continue
                    if (
                        entry_evidence_score is None
                        or entry_evidence_score
                        < configured_compressed_structure_min_entry_evidence_score
                        or structure_proximity_score is None
                        or structure_proximity_score
                        <= configured_compressed_structure_min_proximity_score
                    ):
                        record_entry_block(
                            date, code, candidate, gated_signal,
                            "compressed_structure_quality_gate_failed",
                        )
                        continue
                    execution_date = next_calendar_date.get(date)
                    gated_signal.update({
                        "source_signal_type": "consolidation_breakout",
                        "signal_type": "compressed_structure_continuation",
                        "entry_channel": "compressed_structure_continuation",
                        "original_entry_gate_reason": gate_reason,
                        "technical_signal_date": date.strftime("%Y-%m-%d"),
                    })
                    if execution_date is None:
                        record_entry_block(
                            date, code, candidate, gated_signal,
                            "compressed_structure_next_trade_date_unavailable",
                        )
                        continue
                    pending_compressed_structure_entries[code] = {
                        "candidate": dict(candidate),
                        "signal": dict(gated_signal),
                        "signal_index": index,
                        "execution_date": execution_date,
                    }
            execute_right_entries(
                candidates, date, exited_today, left_right_switch_candidate_current,
            )

        record_equity(date)

    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        maximum_drawdown = 0.0
        final_equity = 1.0
    else:
        drawdown = curve["equity"] / curve["equity"].cummax().clip(lower=1.0) - 1
        maximum_drawdown = float(drawdown.min())
        final_equity = float(curve.iloc[-1]["equity"])
    precoverage_dates = {
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[
            (frame["date"] >= requested) & (frame["date"] < coverage_start),
            "date",
        ]
    }

    def _right_batch_snapshot(lot, row=None):
        item = {
            "origin_allocation": lot_origin_allocation(lot),
            "corporate_action_pending": bool(lot.get("corporate_action_pending")),
            "batch": lot["batch"],
            "position_pct": round(lot["size"] * 100, 2),
            "quantity": round(lot["quantity"], 8),
            "cost": round(lot["cost"], 3),
            "signal_cost": round(lot_signal_cost(lot), 3),
            "stop": round(lot["stop"], 3),
            "stop_raw_price": (
                None if row is None
                else round(qfq_to_raw_price(row, float(lot["stop"])), 3)
            ),
            "merged": lot["merged"],
            "proven": lot.get("proven", False),
            "max_return_pct": round(float(lot.get("max_return") or 0.0) * 100, 2),
            "factor_horizon_trend_managed": bool(
                lot.get("factor_horizon_trend_managed")
            ),
        }
        if lot.get("origin_account"):
            item["origin_account"] = lot["origin_account"]
        if lot.get("origin_batch"):
            item["origin_batch"] = lot["origin_batch"]
        if lot.get("left_grid_slot") is not None:
            item["left_grid_slot"] = int(lot["left_grid_slot"])
        return item

    # Include still-unlisted securities in closing inventory and attribution.
    for pending in corporate_book.pending:
        getattr(states[pending["code"]], pending["account"]).append(pending["lot"])
        active_position_codes.add(pending["code"])
    corporate_book.pending.clear()
    final_position_details = []
    final_portfolio_market_value = final_equity * float(initial_capital)
    for code in sorted(occupied_codes()):
        state = states[code]
        frame = frames[code]
        visible = frame[frame["date"] <= end]
        last_row = None if visible.empty else visible.iloc[-1].copy()
        close = None if last_row is None else valuation_price(code, last_row, end)
        if last_row is not None and raw_price(last_row, "close") > 0:
            last_row["qfq_to_raw_factor"] *= close / raw_price(last_row, "close")
        left_size = sum(float(lot["size"]) for lot in state.left)
        left_quantity = sum(float(lot["quantity"]) for lot in state.left)
        left_value = sum(float(lot["quantity"]) * float(lot["cost"]) for lot in state.left)
        left_entry_fees = sum(float(lot.get("entry_fee_cash") or 0.0) for lot in state.left)
        right_size = sum(float(lot["size"]) for lot in state.right)
        right_quantity = sum(float(lot["quantity"]) for lot in state.right)
        right_value = sum(float(lot["quantity"]) * float(lot["cost"]) for lot in state.right)
        right_entry_fees = sum(float(lot.get("entry_fee_cash") or 0.0) for lot in state.right)
        total_quantity = left_quantity + right_quantity
        total_value = left_value + right_value
        invested_amount = total_value + left_entry_fees + right_entry_fees
        market_value = None if close is None else total_quantity * close
        unrealized_pnl_amount = None if market_value is None else market_value - invested_amount
        position_industry_tags = sorted({
            tag
            for lot in state.left + state.right
            for tag in _split_industry_tags(lot.get("industry_tags"))
        })
        final_position_details.append({
            "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "industry_tags": position_industry_tags,
            "close": close,
            "position_pct": round((left_size + right_size) * 100, 2),
            "position_pct_basis": "initial_capital_cost_exposure",
            "market_weight_pct": (
                None if market_value is None or final_portfolio_market_value <= 0
                else round(market_value / final_portfolio_market_value * 100, 4)
            ),
            "market_weight_pct_basis": "final_equity_market_value",
            "left_position_pct": round(left_size * 100, 2),
            "right_position_pct": round(right_size * 100, 2),
            "position_mode": (
                "left+right" if state.left and state.right
                else "left" if state.left else "right"
            ),
            "capacity_counted": left_position_counts_capacity(state)
            or (bool(state.right) and not _is_profit_tail(state)),
            "profit_tail": _is_profit_tail(state),
            "quantity": round(total_quantity, 8),
            "invested_amount": round(invested_amount, 2),
            "market_value": None if market_value is None else round(market_value, 2),
            "unrealized_pnl_amount": (
                None if unrealized_pnl_amount is None else round(unrealized_pnl_amount, 2)
            ),
            "unrealized_pnl_pct": (
                None if unrealized_pnl_amount is None or invested_amount <= 0
                else round(unrealized_pnl_amount / invested_amount * 100, 4)
            ),
            "cost": None if not total_quantity else round(total_value / total_quantity, 3),
            "left_value_line": state.left_value_line,
            "left_value_line_raw_price": (
                None if state.left_value_line is None or last_row is None
                else round(qfq_to_raw_price(last_row, float(state.left_value_line)), 3)
            ),
            "left_batches": [
                {
                    "origin_allocation": lot_origin_allocation(lot),
                    "corporate_action_pending": bool(lot.get("corporate_action_pending")),
                    "batch": lot["batch"], "grid_slot": int(lot["slot"]),
                    "position_pct": round(lot["size"] * 100, 2),
                    "quantity": round(lot["quantity"], 8),
                    "cost": round(lot["cost"], 3),
                    "signal_cost": round(lot_signal_cost(lot), 3),
                    "sell_price": (
                        None if lot.get("sell_price") is None
                        else round(float(lot["sell_price"]), 3)
                    ),
                    "sell_price_raw_price": (
                        None if lot.get("sell_price") is None or last_row is None
                        else round(qfq_to_raw_price(last_row, float(lot["sell_price"])), 3)
                    ),
                    "core": bool(lot.get("core")),
                }
                for lot in state.left
            ],
            "batches": [
                _right_batch_snapshot(lot, last_row)
                for lot in state.right
            ],
        })
    trade_ledger = [
        event for event in events
        if abs(float(event.get("execution_quantity") or 0.0)) > 1e-12
    ]
    buy_trades = [event for event in trade_ledger if event["trade_side"] == "买入"]
    sell_trades = [event for event in trade_ledger if event["trade_side"] == "卖出"]
    profitable_sells = [
        event for event in sell_trades if float(event.get("profit_loss_amount") or 0.0) > 0
    ]
    losing_sells = [
        event for event in sell_trades if float(event.get("profit_loss_amount") or 0.0) < 0
    ]
    trade_summary = {
        "buy_count": len(buy_trades),
        "sell_count": len(sell_trades),
        "buy_amount": round(sum(float(event["trade_amount"]) for event in buy_trades), 2),
        "sell_amount": round(sum(float(event["trade_amount"]) for event in sell_trades), 2),
        "transaction_cost_amount": round(
            sum(float(event["transaction_cost_amount"]) for event in trade_ledger), 2,
        ),
        "closed_trade_net_pnl_amount": round(
            sum(float(event.get("profit_loss_amount") or 0.0) for event in sell_trades), 2,
        ),
        "profitable_sell_count": len(profitable_sells),
        "losing_sell_count": len(losing_sells),
        "sell_win_rate_pct": round(
            len(profitable_sells) / len(sell_trades) * 100, 3,
        ) if sell_trades else 0.0,
    }
    def trade_bucket(event):
        account_mode = str(event.get("account_mode") or "").strip()
        if account_mode:
            return account_mode
        action = str(event.get("action") or "")
        if action.startswith("左侧"):
            return "left"
        if action.startswith("右侧"):
            return "right"
        return "right"

    trade_count_by_account = Counter(trade_bucket(event) for event in trade_ledger)
    buy_count_by_account = Counter(trade_bucket(event) for event in buy_trades)
    sell_count_by_account = Counter(trade_bucket(event) for event in sell_trades)
    trade_amount_by_account = Counter()
    pnl_by_account = Counter()
    traded_symbols_by_account = {}
    grid_trades_by_symbol = Counter()
    for event in trade_ledger:
        bucket = trade_bucket(event)
        trade_amount_by_account[bucket] += float(event.get("trade_amount") or 0.0)
        if event["trade_side"] == "卖出":
            pnl_by_account[bucket] += float(event.get("profit_loss_amount") or 0.0)
        traded_symbols_by_account.setdefault(bucket, set()).add(str(event.get("code") or ""))
        if bucket == "left" and event.get("grid_slot") is not None:
            grid_trades_by_symbol[
                f"{event.get('code')} {event.get('name')}"
            ] += 1
    trade_mix_summary = {
        "trade_count_by_account": dict(trade_count_by_account),
        "buy_count_by_account": dict(buy_count_by_account),
        "sell_count_by_account": dict(sell_count_by_account),
        "trade_amount_by_account": {
            key: round(value, 2) for key, value in trade_amount_by_account.items()
        },
        "net_pnl_by_account": {
            key: round(value, 2) for key, value in pnl_by_account.items()
        },
        "traded_symbol_count_by_account": {
            key: len({code for code in value if code})
            for key, value in traded_symbols_by_account.items()
        },
        "left_grid_trade_count_by_symbol_top": [
            {"symbol": symbol, "trade_count": count}
            for symbol, count in grid_trades_by_symbol.most_common(10)
        ],
        "note": (
            "net_pnl_by_account is closed-trade net PnL after allocated entry fees; "
            "max_total_held_symbols limits simultaneous symbols; grid batches "
            "can create many trades inside one left-side symbol."
        ),
    }
    right_buys = [event for event in events if event["action"] == "右侧买入"]
    right_buy_signal_type_counts = Counter(
        str(event.get("signal_type") or "unknown")
        for event in right_buys
    )
    structure_signal_counts = {
        "uptrend_ratio_pullback": int(
            right_buy_signal_type_counts.get("uptrend_support_pullback", 0)
        ),
        "uptrend_50_reclaim": int(
            right_buy_signal_type_counts.get("uptrend_50_reclaim", 0)
        ),
        "pullback_50_breakout": int(
            right_buy_signal_type_counts.get("pullback_50_breakout", 0)
        ),
    }
    final_return_pct = round((final_equity - 1) * 100, 3)
    profit_concentration_summary = _portfolio_profit_concentration_summary(
        events,
        final_position_details,
        initial_capital=initial_capital,
        final_return_pct=final_return_pct,
    )
    r_multiple_summary = _portfolio_r_multiple_summary(events)
    localized_events = [localize_event(event) for event in events]
    localized_trade_ledger = [localize_event(event) for event in trade_ledger]
    localized_entry_blocks = [
        localize_entry_block(item) for item in entry_blocks
    ]
    for item in breadth_reversal_observations:
        if (
            item.get("status") == "awaiting_next_close_confirmation"
            and item.get("code") in pending_breadth_reversal_entries
        ):
            item["status"] = "confirmation_not_available_by_end"
    localized_breadth_reversal_observations = [
        localize_breadth_reversal_observation(item)
        for item in breadth_reversal_observations
    ]
    return {
        "requested_start": requested.strftime("%Y-%m-%d"),
        "actual_start": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "candidate_coverage_start": coverage_start.strftime("%Y-%m-%d"),
        "coverage_complete": not precoverage_dates,
        "missing_candidate_trade_dates": sorted(date.strftime("%Y-%m-%d") for date in precoverage_dates),
        "events": localized_events,
        "trade_ledger": localized_trade_ledger,
        "trade_summary": trade_summary,
        "trade_mix_summary": trade_mix_summary,
        "profit_concentration_summary": profit_concentration_summary,
        "r_multiple_summary": r_multiple_summary,
        "execution_model_version": "opening-orders-first-v1",
        "capital_risk_attribution": build_portfolio_attribution(
            events, equity_curve, final_position_details, initial_capital,
        ),
        "equity_curve": equity_curve,
        "event_count": len(events),
        "cash_limited_order_count": sum(
            bool(event.get("cash_limited")) for event in events
        ),
        "lot_rounded_order_count": sum(
            bool(event.get("lot_rounded")) for event in events
        ),
        "liquidity_limited_order_count": sum(
            bool(event.get("entry_liquidity_limited")) for event in events
        ),
        "candidate_pool_limit": configured_candidate_limit,
        "max_positions": configured_positions,
        "max_total_held_symbols": configured_total_held_symbols,
        "max_left_positions": configured_left_positions,
        "profit_tranches": configured_profit_tranches,
        "profit_tails_consume_capacity": False,
        "profit_tail_minimum_maximum_return_pct": round(
            configured_profit_tail_min_return * 100, 2,
        ),
        "left_grid_unit_pct": round(configured_left_grid_unit * 100, 2),
        "left_grid_step_pct": round(configured_left_grid_step * 100, 2),
        "left_grid_max_exposure_pct": round(
            configured_left_grid_max_exposure * 100, 2,
        ),
        "pe_pb_left_max_positions": configured_pe_pb_left_max_positions,
        "pe_pb_left_max_exposure_pct": round(
            configured_pe_pb_left_max_exposure * 100, 2,
        ),
        "pe_pb_left_initial_slots": configured_pe_pb_left_initial_slots,
        "pe_pb_left_core_slots": configured_pe_pb_left_core_slots,
        "left_grid_initial_slots": configured_left_initial_slots,
        "left_grid_core_slots": configured_left_core_slots,
        "maximum_total_held_symbols": max(
            (row["total_held_symbol_count"] for row in equity_curve), default=0,
        ),
        "maximum_left_side_symbols": max(
            (row["left_side_symbol_count"] for row in equity_curve), default=0,
        ),
        "maximum_right_market_left_side_symbols": max(
            (
                row["right_market_left_side_symbol_count"]
                for row in equity_curve
            ),
            default=0,
        ),
        "maximum_profit_tail_count": max(
            (row["profit_tail_count"] for row in equity_curve), default=0,
        ),
        "max_same_industry": configured_same_industry,
        "same_theme_correlation": configured_theme_correlation,
        "min_entry_evidence_score": configured_min_entry_evidence_score,
        "right_entry_priority_mode": configured_entry_priority_mode,
        "right_entry_size_multiplier": (
            configured_right_entry_size_multiplier
        ),
        "right_entry_risk_budget_pct": (
            configured_right_entry_risk_budget_pct
        ),
        "right_inverse_volatility_sizing_enabled": (
            configured_right_inverse_volatility_sizing
        ),
        "right_inverse_volatility_sizing_scope": (
            "ordinary_standard_right_first_entry_only"
        ),
        "right_market_volatility_sizing_enabled": (
            configured_right_market_volatility_sizing
        ),
        "price_amount_multiplier": configured_price_amount_multiplier,
        "factor_only_min_avg_amount": configured_factor_only_min_avg_amount,
        "max_entry_signal_amount_participation": (
            configured_max_entry_signal_amount_participation
        ),
        "max_entry_avg_amount_participation": (
            configured_max_entry_avg_amount_participation
        ),
        "entry_gate_model": "semantic_high_reward_risk",
        "entry_gate_model_cn": summary_code_text("semantic_high_reward_risk"),
        "entry_evidence_score_role": "legacy_explanation_only",
        "entry_evidence_score_role_cn": summary_code_text(
            "legacy_explanation_only"
        ),
        "allow_pullback_pilot": configured_allow_pullback_pilot,
        "breadth_reversal_half_pullback_enabled": (
            configured_breadth_reversal_enabled
        ),
        "factor_only_breadth_reversal_only": (
            configured_factor_only_breadth_reversal_only
        ),
        "breadth_reversal_max_unique_count_exclusive": (
            configured_breadth_reversal_max_unique_count
        ),
        "breadth_reversal_risk_budget_pct": round(
            configured_breadth_reversal_risk_budget * 100, 3,
        ),
        "breadth_reversal_max_position_pct": round(
            configured_breadth_reversal_max_position * 100, 3,
        ),
        "breadth_reversal_max_positions": (
            configured_breadth_reversal_max_positions
        ),
        "breadth_reversal_time_limit_days": (
            configured_breadth_reversal_time_limit_days
        ),
        "breadth_reversal_observation_count": len(
            breadth_reversal_observations
        ),
        "breadth_reversal_observations": (
            localized_breadth_reversal_observations
        ),
        "pending_breadth_reversal_codes": sorted(
            pending_breadth_reversal_entries
        ),
        "factor_horizon_channel_enabled": configured_factor_horizon_enabled,
        "factor_horizon_risk_budget_pct": round(
            configured_factor_horizon_risk_budget * 100, 3,
        ),
        "factor_horizon_stop_loss_pct": round(
            configured_factor_horizon_stop_loss_pct * 100, 3,
        ),
        "factor_horizon_max_position_pct": round(
            configured_factor_horizon_max_position * 100, 3,
        ),
        "factor_horizon_max_positions": configured_factor_horizon_max_positions,
        "factor_horizon_holding_days": (
            configured_factor_horizon_holding_days
        ),
        "factor_horizon_trend_management_return_pct": (
            None
            if configured_factor_horizon_trend_management_return is None
            else round(
                configured_factor_horizon_trend_management_return * 100,
                3,
            )
        ),
        "factor_horizon_trend_managed_batch_count": len(
            factor_horizon_trend_managed_batches
        ),
        "factor_horizon_trend_managed_codes": sorted(
            factor_horizon_trend_managed_codes
        ),
        "factor_horizon_max_return_60d_pct": (
            None
            if configured_factor_horizon_max_return_60d is None
            else round(configured_factor_horizon_max_return_60d * 100, 3)
        ),
        "factor_horizon_require_ma20_not_rising": (
            configured_factor_horizon_require_ma20_not_rising
        ),
        "factor_horizon_exclude_watch_phase": (
            configured_factor_horizon_exclude_watch_phase
        ),
        "factor_horizon_min_avg_amount": round(
            configured_factor_horizon_min_avg_amount, 2,
        ),
        "factor_only_compressed_structure_only": (
            configured_factor_only_compressed_structure_only
        ),
        "compressed_structure_channel_enabled": (
            configured_compressed_structure_enabled
        ),
        "compressed_structure_risk_budget_pct": round(
            configured_compressed_structure_risk_budget * 100, 3,
        ),
        "compressed_structure_stop_loss_pct": round(
            configured_compressed_structure_stop_loss_pct * 100, 3,
        ),
        "compressed_structure_max_position_pct": round(
            configured_compressed_structure_max_position * 100, 3,
        ),
        "compressed_structure_max_positions": (
            configured_compressed_structure_max_positions
        ),
        "compressed_structure_holding_days": (
            configured_compressed_structure_holding_days
        ),
        "compressed_structure_min_entry_evidence_score": (
            configured_compressed_structure_min_entry_evidence_score
        ),
        "compressed_structure_min_proximity_score_exclusive": (
            configured_compressed_structure_min_proximity_score
        ),
        "compressed_structure_min_avg_amount": (
            configured_compressed_structure_min_avg_amount
        ),
        "compressed_structure_buy_count": int(
            right_buy_signal_type_counts.get(
                "compressed_structure_continuation", 0,
            )
        ),
        "pending_compressed_structure_codes": sorted(
            pending_compressed_structure_entries
        ),
        "pending_standard_right_close_codes": sorted(
            pending_standard_right_close_entries
        ),
        "intraday_right_protective_stop_enabled": (
            configured_intraday_right_protective_stop
        ),
        "diversified_factor_sleeve_enabled": (
            configured_diversified_factor_sleeve
        ),
        "diversified_factor_risk_budget_pct": round(
            configured_diversified_factor_risk_budget * 100, 3,
        ),
        "diversified_factor_max_position_pct": round(
            configured_diversified_factor_max_position * 100, 3,
        ),
        "factor_horizon_buy_count": int(
            right_buy_signal_type_counts.get("factor_horizon_match", 0)
        ),
        "concentration_block_count": len(concentration_blocks),
        "concentration_blocks": concentration_blocks,
        "entry_block_count": len(entry_blocks),
        "entry_blocks": localized_entry_blocks,
        "board_lot_policy": {"sh.688": 200, "default": 100},
        "structure_signal_counts": structure_signal_counts,
        "structure_signal_counts_cn": localize_signal_counts(
            structure_signal_counts
        ),
        "right_buy_signal_type_counts": dict(right_buy_signal_type_counts),
        "right_buy_signal_type_counts_cn": localize_signal_counts(
            dict(right_buy_signal_type_counts)
        ),
        "realized_return_pct": round(realized * 100, 3),
        "unrealized_return_pct": round((final_equity - 1 - realized) * 100, 3),
        "final_return_pct": final_return_pct,
        "maximum_drawdown_pct": round(maximum_drawdown * 100, 3),
        "maximum_gross_exposure_pct": round(
            max((row["gross_exposure_pct"] for row in equity_curve), default=0.0), 2,
        ),
        "transaction_cost_pct": round(transaction_costs * 100, 3),
        "final_cash": round(float(cash_balance), 6),
        "final_dividend_receivable": corporate_book.receivable,
        "corporate_action_limitations": corporate_book.limitations,
        "corporate_action_accounting_status": "data_limited" if corporate_book.limitations else "complete",
        "final_cash_pct": round(
            float(cash_balance) / (final_equity * float(initial_capital)) * 100, 6,
        ) if final_equity else None,
        "commission_rate": float(commission_rate),
        "minimum_commission": float(minimum_commission),
        "initial_capital": float(initial_capital),
        "sell_stamp_duty_rate": float(sell_stamp_duty_rate),
        "sell_stamp_duty_rate_before_2023_08_28": (
            None
            if sell_stamp_duty_rate_before_2023_08_28 is None
            else float(sell_stamp_duty_rate_before_2023_08_28)
        ),
        "sell_stamp_duty_reduction_date": "2023-08-28",
        "estimated_slippage_rate": float(estimated_slippage_rate),
        "max_symbol_exposure_pct": round(float(max_symbol_exposure) * 100, 2),
        "exit_tail_on_candidate_removal": bool(exit_tail_on_candidate_removal),
        "signals_effective_next_day": bool(signals_effective_next_day),
        "auto_price_structure": bool(auto_price_structure),
        "allow_structure_pullback": bool(allow_structure_pullback),
        "close_confirmed_execution": close_confirmed_execution,
        "pending_candidate_exit_codes": sorted(pending_candidate_exits),
        "final_positions": final_position_details,
    }
