"""Opt-in research channel rules. No cash, position or order state is owned here.

The frozen strategy disables these channels. The replay engine retains timing,
pending orders and execution; these functions return signals or block reasons.
"""
from __future__ import annotations

import pandas as pd

def _prepare_bull_half_reversal_signal(data, index, signal):
    """Turn the observed half-pullback into a protective close signal."""
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
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
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
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
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
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


def factor_horizon_signal(data, index, candidate, *, phase, exclude_watch_phase, max_return_60d, require_ma20_not_rising, min_avg_amount, max_position, risk_budget, stop_loss_pct, holding_days, trend_management_return):
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
    if (
        exclude_watch_phase
        and phase == "watch"
    ):
        return None, None, "factor_horizon_watch_phase_excluded"
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
        return None, None, "factor_horizon_rule_mismatch"
    if (
        max_return_60d is not None
        and (
            return_60d is None
            or return_60d
            > max_return_60d
        )
    ):
        return None, None, "factor_horizon_return_60d_filter_mismatch"
    if (
        require_ma20_not_rising
        and _enabled(
            candidate.get("ma20_rising"), default=True,
        )
    ):
        return None, None, "factor_horizon_ma20_filter_mismatch"
    if (
        average_amount is None
        or average_amount < min_avg_amount
    ):
        return None, None, "liquidity_too_thin"
    signal_open = pd.to_numeric(
        data.iloc[index].get("open"), errors="coerce",
    )
    if pd.isna(signal_open) or float(signal_open) <= 0:
        return None, None, "missing_entry_price"
    planned_size = min(
        max_position,
        risk_budget
        / stop_loss_pct,
    )
    factor_horizon_rule_parts = [
        "训练期冻结的因子前10名与结构接近分大于40条件",
    ]
    if max_return_60d is not None:
        factor_horizon_rule_parts.append(
            "60日收益不高于"
            f"{max_return_60d:.1%}"
        )
    if require_ma20_not_rising:
        factor_horizon_rule_parts.append("MA20尚未上扬")
    if exclude_watch_phase:
        factor_horizon_rule_parts.append(
            "Formula33观察阶段不买入"
        )
    factor_horizon_exit_plan = (
        "历史最大浮盈达到"
        f"{trend_management_return:.1%}"
        "后转趋势管理，否则"
        f"{holding_days}日到期"
        if trend_management_return
        is not None
        else f"计划持有{holding_days}日"
    )
    signal = {
        "rank": 4,
        "trigger": float(signal_open),
        "stop": float(signal_open) * (
            1.0 - stop_loss_pct
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
            stop_loss_pct * 100,
            3,
        ),
        "planned_position_size": planned_size,
        "entry_liquidity_floor": (
            min_avg_amount
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
    return signal, fundamental_score, ""


def compressed_structure_open_signal(data, index, candidate, signal, *, max_position, risk_budget, stop_loss_pct, holding_days):
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
    signal = dict(signal)
    signal_open = pd.to_numeric(
        data.iloc[index].get("open"), errors="coerce",
    )
    if pd.isna(signal_open) or float(signal_open) <= 0:
        return None, None, None, "compressed_structure_missing_entry_price"
    planned_size = min(
        max_position,
        risk_budget
        / stop_loss_pct,
    )
    signal.update({
        "trigger": float(signal_open),
        "stop": float(signal_open) * (
            1.0 - stop_loss_pct
        ),
        "order_type": "limit",
        "reason": (
            "紧结构平台延续；下一交易日开盘执行；"
            f"固定{stop_loss_pct:.0%}保护止损；"
            f"持有{holding_days}个交易日"
        ),
        "entry_channel": "compressed_structure_continuation",
        "semantic_entry_ok": True,
        "semantic_entry_setup": "compressed_structure_continuation",
        "entry_gate_reason": "compressed_structure_rule_matched",
        "entry_risk_pct": round(
            stop_loss_pct * 100, 3,
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
    return signal, fundamental_score, entry_priority, ""


def breadth_observation_gate(data, index, candidate, signal, formula_observation, date, *, max_unique_count):
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
    signal = _prepare_bull_half_reversal_signal(
        data, index, signal,
    )
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
        return signal, unique_count, "breadth_unique_count_missing"
    if unique_count >= max_unique_count:
        return signal, unique_count, "breadth_not_low_enough"
    average_amount = _channel_candidate_number(
        candidate, "breadth_reversal", "avg_amount_20",
    )
    if average_amount is None or average_amount < 500_000_000.0:
        return signal, unique_count, "liquidity_too_thin"
    return_60d = _channel_candidate_number(
        candidate, "breadth_reversal", "return_60d",
    )
    if return_60d is None or return_60d > -0.10:
        return signal, unique_count, "reversal_60d_decline_not_deep_enough"
    return signal, unique_count, ""


def compressed_structure_setup_gate(data, index, candidate, signal, *, semantic_gate, current_down_streak, min_avg_amount, min_entry_evidence_score, min_proximity_score):
    from stock_research.backtesting.signals import (
            _candidate_number, _channel_candidate_number, _signal_number,
            _enabled, _factor_only_selection_profile, MIN_CLOSE_CONFIRMED_RIGHT_RISK_PCT,
        )
    if signal.get("signal_type") != "consolidation_breakout":
        return signal, None, "compressed_structure_non_platform_signal"
    original_gate_candidate = dict(candidate)
    for field in (
        "avg_amount_20", "return_20d", "return_60d",
    ):
        original_gate_candidate.pop(field, None)
    ok, gated_signal, gate_reason = semantic_gate(
        data, index, original_gate_candidate, signal,
        current_down_streak=current_down_streak,
    )
    if ok or gate_reason not in {
        "risk_pct_too_tight",
        "close_confirmed_risk_pct_too_tight",
    }:
        return gated_signal, gate_reason, "compressed_structure_original_gate_mismatch"
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
            min_avg_amount
        ),
        "entry_liquidity_profile": (
            "紧结构专用日均成交额门"
        ),
        "candidate_avg_amount_20": average_amount,
    })
    if (
        min_avg_amount > 0
        and (
            average_amount is None
            or average_amount
            < min_avg_amount
        )
    ):
        return gated_signal, gate_reason, "compressed_structure_liquidity_too_thin"
    if (
        entry_evidence_score is None
        or entry_evidence_score
        < min_entry_evidence_score
        or structure_proximity_score is None
        or structure_proximity_score
        <= min_proximity_score
    ):
        return gated_signal, gate_reason, "compressed_structure_quality_gate_failed"
    return gated_signal, gate_reason, ""


def channel_candidate_eligible(candidate, channel, *, enabled, blocked=False, factor_only=False):
    from stock_research.backtesting.signals import _enabled, _factor_only_selection_profile
    return bool(enabled and not blocked and (
        _enabled(candidate.get(f"{channel}_channel_eligible"), default=False)
        or (factor_only and _factor_only_selection_profile(candidate))
    ))


def diversified_factor_entry(candidate, entry_channel, *, enabled):
    from stock_research.backtesting.signals import _factor_only_selection_profile
    return bool(enabled and entry_channel == "standard_right_entry"
                and _factor_only_selection_profile(candidate))


def diversified_factor_size(signal, *, max_position, risk_budget):
    return min(max_position, risk_budget / max(float(signal.get("entry_risk_pct") or 0.0) / 100.0, 1e-6))
