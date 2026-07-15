"""Point-in-time technical entries described by the trading framework."""
from __future__ import annotations

import pandas as pd

from stock_research.indicators.price_structure import confirmed_turning_points


SUPPORT_ZONE_PCT = 0.05
TIGHT_STRUCTURE_ZONE_PCT = 0.02


def _volume_baseline(previous: pd.DataFrame) -> float:
    return max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )


def _in_zone(value: float, level: float, pct: float = SUPPORT_ZONE_PCT) -> bool:
    return level * (1 - pct) <= value <= level * (1 + pct)


def _number(value) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _close_signal(kind, rank, trigger, stop, reason, volume_ratio):
    return {
        "rank": int(rank),
        "trigger": float(trigger),
        "stop": float(stop),
        "order_type": "close",
        "reason": reason,
        "known_volume_ratio": float(volume_ratio),
        "signal_type": kind,
        "requires_next_day_confirmation": True,
    }


def _valid_volume_price_nodes(previous: pd.DataFrame) -> list[dict]:
    """Return confirmed nodes whose two-bar low has never been breached.

    A node cannot be used on its formation day.  Its support is only known
    after the following bar, and any later low below that support invalidates
    the original node permanently; a subsequent recovery does not revive it.
    """
    window = previous.tail(60).copy().reset_index(drop=True)
    if len(window) < 12:
        return []
    median_volume = float(window["volume"].median())
    returns = window["close"].pct_change()
    nodes = []
    for position in range(len(window) - 1):
        row = window.iloc[position]
        if float(row["volume"]) < median_volume * 1.5 or float(returns.iloc[position]) < 0.02:
            continue
        support = min(
            float(window.iloc[position]["low"]),
            float(window.iloc[position + 1]["low"]),
        )
        later = window.iloc[position + 2:]
        if not later.empty and float(later["low"].min()) < support:
            continue
        nodes.append({
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "confirmed_on": pd.Timestamp(
                window.iloc[position + 1]["date"],
            ).strftime("%Y-%m-%d"),
            "support": support,
            "status": "valid",
        })
    return nodes


def apply_entry_confluence(data: pd.DataFrame, index: int, signal: dict) -> dict:
    """Score evidence around a primary structure; evidence never creates a trade."""
    scored = dict(signal)
    decision_index = index if signal.get("order_type") == "close" else index - 1
    decision = data.iloc[decision_index]
    history = data.iloc[:decision_index]
    score = int(signal.get("rank") or 0)
    evidence = [
        f"primary:{signal.get('signal_type') or 'configured_structure'}+{score}",
    ]
    context_bonus = int(signal.get("context_bonus") or 0)
    if context_bonus:
        evidence.extend(signal.get("context_evidence") or [])
        evidence.append(f"context_bonus+{context_bonus}")
        score += context_bonus
    else:
        evidence.extend(signal.get("context_evidence") or [])
    if len(history) >= 20:
        volume_baseline = _volume_baseline(history)
        if volume_baseline > 0 and float(decision["volume"]) >= volume_baseline:
            evidence.append("volume_confirmed+1")
            score += 1
            if float(decision["volume"]) >= volume_baseline * 1.30:
                evidence.append("volume_expanded_1_3x+1")
                score += 1

        level = float(signal.get("stop") or signal.get("trigger"))
        entry_reference = (
            float(decision["close"])
            if signal.get("order_type") == "close"
            else float(signal.get("trigger") or decision["close"])
        )
        if _in_zone(entry_reference, level):
            evidence.append("entry_within_structure_zone")
            if _in_zone(entry_reference, level, TIGHT_STRUCTURE_ZONE_PCT):
                evidence.append("entry_tight_to_structure+1")
                score += 1
        elif entry_reference > level * (1 + SUPPORT_ZONE_PCT):
            evidence.append("entry_extended_from_structure-2")
            score -= 2
        if level > 0:
            stop_distance = entry_reference / level - 1
            if stop_distance > 0.10:
                evidence.append("stop_distance_above_10pct-2")
                score -= 2
            elif stop_distance > 0.07:
                evidence.append("stop_distance_above_7pct-1")
                score -= 1

        ma20 = _number(decision.get("ma20"))
        ma60 = _number(decision.get("ma60"))
        prior_close = float(data.iloc[decision_index - 1]["close"]) if decision_index > 0 else None
        prior_ma20 = _number(data.iloc[decision_index - 1].get("ma20")) if decision_index > 0 else None
        prior_ma60 = _number(data.iloc[decision_index - 1].get("ma60")) if decision_index > 0 else None
        five_day_ma20 = _number(data.iloc[decision_index - 6].get("ma20")) if decision_index >= 6 else None
        five_day_ma60 = _number(data.iloc[decision_index - 6].get("ma60")) if decision_index >= 6 else None
        ma20_rising = (
            prior_ma20 is not None
            and five_day_ma20 is not None
            and prior_ma20 > five_day_ma20
        )
        medium_long_rising = (
            prior_ma20 is not None
            and prior_ma60 is not None
            and five_day_ma20 is not None
            and five_day_ma60 is not None
            and prior_ma20 > prior_ma60
            and prior_ma20 > five_day_ma20
            and prior_ma60 > five_day_ma60
        )
        if ma20 is not None:
            reclaimed_ma20 = bool(
                prior_close is not None
                and prior_close < ma20 * 0.995
                and entry_reference >= ma20
            )
            if entry_reference >= ma20 and (ma20_rising or reclaimed_ma20):
                evidence.append("close_above_rising_or_reclaimed_ma20+1")
                score += 1
            low = _number(decision.get("low"))
            close = _number(decision.get("close"))
            near_rising_ma20 = ma20_rising and abs(level / ma20 - 1) <= SUPPORT_ZONE_PCT
            pulled_to_rising_ma20 = (
                ma20_rising
                and low is not None
                and close is not None
                and low <= ma20 * (1 + SUPPORT_ZONE_PCT)
                and close >= ma20 * (1 - SUPPORT_ZONE_PCT)
            )
            if reclaimed_ma20 or near_rising_ma20 or pulled_to_rising_ma20:
                evidence.append("entry_near_or_reclaims_ma20+2")
                score += 2
            if ma60 is not None and entry_reference < ma20 and ma20 < ma60:
                evidence.append("weak_ma20_below_ma60_and_price_below_ma20-2")
                score -= 2
        if medium_long_rising:
            evidence.append("ma20_ma60_rising_alignment+2")
            score += 2

        low_volume_cutoff = float(data.iloc[max(0, decision_index - 60):decision_index]["volume"].median()) * 0.8
        for period, weight in ((20, 1), (60, 2), (120, 3)):
            if decision_index < period + 5 or pd.isna(decision[f"ma{period}"]):
                continue
            ma = float(decision[f"ma{period}"])
            deducted_price = float(data.iloc[decision_index - period]["close"])
            deducted_volume = float(data.iloc[decision_index - period]["volume"])
            deduction_confluence = (
                abs(level / ma - 1) <= SUPPORT_ZONE_PCT
                and ma > float(data.iloc[decision_index - 5][f"ma{period}"])
                and deducted_price <= ma * 0.95
                and deducted_volume <= low_volume_cutoff
                and float(decision["volume"]) >= deducted_volume
            )
            if deduction_confluence:
                evidence.append(f"deduction_low_price_volume_ma{period}+{weight}")
                score += weight

    anchor_low = pd.to_numeric(signal.get("anchor_low"), errors="coerce")
    anchor_high = pd.to_numeric(signal.get("anchor_high"), errors="coerce")
    if (
        pd.notna(anchor_low)
        and pd.notna(anchor_high)
        and float(anchor_high) / float(anchor_low) - 1 >= 0.50
    ):
        evidence.append("large_wave_structure+2")
        score += 2

    prior = data.iloc[index - 1]
    row = data.iloc[index]
    if float(row["open"]) > float(prior["high"]):
        evidence.append("gap_up+1")
        score += 1
    scored["rank"] = max(0, score)
    scored["entry_evidence_score"] = scored["rank"]
    scored["entry_evidence"] = evidence
    return scored


def infer_technical_entry(data: pd.DataFrame, index: int) -> dict | None:
    """Return the strongest objective entry visible by the decision close."""
    if index < 60:
        return None
    row = data.iloc[index]
    previous = data.iloc[:index]
    prior = previous.iloc[-1]
    baseline = _volume_baseline(previous)
    volume_ratio = float(row["volume"]) / baseline if baseline > 0 else 0.0
    signals = []

    # W bottom: two confirmed lows of similar depth with a neckline between.
    recent = previous.tail(120).reset_index(drop=True)
    pivots = confirmed_turning_points(recent, left=3, right=3)
    if len(pivots["lows"]) >= 2:
        first_low, second_low = pivots["lows"][-2:]
        if second_low - first_low >= 10:
            low1 = float(recent.iloc[first_low]["low"])
            low2 = float(recent.iloc[second_low]["low"])
            similar_lows = abs(low2 / low1 - 1) <= 0.08
            between = recent.iloc[first_low + 1:second_low]
            if similar_lows and not between.empty:
                neckline = float(between["high"].max())
                if (
                    float(prior["close"]) <= neckline < float(row["close"])
                    and float(row["close"]) <= neckline * (1 + SUPPORT_ZONE_PCT)
                    and volume_ratio >= 1.0
                ):
                    signals.append(_close_signal(
                        "w_bottom_neckline", 6, neckline, neckline,
                        "W底收盘放量突破颈线", volume_ratio,
                    ))

    # Gap long candle crossing MA20/MA60.
    ma_barrier = max(float(prior["ma20"]), float(prior["ma60"]))
    if (
        float(row["open"]) > float(prior["high"])
        and float(row["close"]) / float(row["open"]) - 1 >= 0.03
        and float(prior["close"]) <= ma_barrier < float(row["close"])
        and float(row["close"]) <= ma_barrier * (1 + SUPPORT_ZONE_PCT)
        and volume_ratio >= 1.0
    ):
        signals.append(_close_signal(
            "gap_long_ma_breakout", 5, float(row["close"]), float(prior["high"]),
            "跳空长阳越过MA20/MA60; 缺口下沿止损", volume_ratio,
        ))

    # Tight consolidation may trigger, but a generic 21-day close high may not.
    box = previous.tail(21)
    box_high = float(box["high"].max())
    box_low = float(box["low"].min())
    if (
        box_high / box_low - 1 <= 0.15
        and float(prior["close"]) <= box_high < float(row["close"])
        and float(row["close"]) <= box_high * (1 + SUPPORT_ZONE_PCT)
        and volume_ratio >= 1.0
    ):
        signals.append(_close_signal(
            "consolidation_breakout", 4, box_high, box_high,
            "整理平台收盘放量突破; 平台上沿止损", volume_ratio,
        ))

    # Most recent high-volume bullish node, entered only 2%-4% above it.
    valid_nodes = _valid_volume_price_nodes(previous)
    positive_trend = (
        float(prior["close"]) > float(prior["ma20"]) > float(prior["ma60"])
        and float(prior["ma20"]) > float(previous.iloc[-6]["ma20"])
    )
    if valid_nodes and (positive_trend or len(valid_nodes) >= 3):
        node = valid_nodes[-1]
        support = float(node["support"])
        trigger = support * 1.02
        if (
            float(prior["close"]) <= trigger < float(row["close"]) <= support * 1.04
            and float(row["low"]) >= support
            and volume_ratio >= 1.0
        ):
            signal = _close_signal(
                "volume_price_node", 3, trigger, support,
                "量价节点上浮2%-4%收盘确认; 节点止损", volume_ratio,
            )
            signal["stop"] = support
            signal["volume_node_date"] = node["date"]
            signal["volume_node_confirmed_on"] = node["confirmed_on"]
            signal["valid_volume_node_count"] = len(valid_nodes)
            signal["context_evidence"] = []
            signal["context_bonus"] = 0
            if positive_trend:
                signal["context_evidence"].append("positive_trend")
                signal["context_bonus"] += 1
            if len(valid_nodes) >= 3:
                signal["context_evidence"].append("three_valid_volume_nodes")
                signal["context_bonus"] += 2
            signals.append(signal)

    # Pullback to half of the latest run of at least three bullish candles.
    run_end = len(previous) - 1
    run_start = run_end
    while run_start >= 0 and float(previous.iloc[run_start]["close"]) > float(previous.iloc[run_start]["open"]):
        run_start -= 1
    run_start += 1
    if run_end - run_start + 1 >= 3:
        run = previous.iloc[run_start:run_end + 1]
        half = (float(run.iloc[0]["low"]) + float(run.iloc[-1]["high"])) / 2
        if (
            float(prior["close"]) > half * (1 + SUPPORT_ZONE_PCT)
            and float(row["low"]) <= half * (1 + SUPPORT_ZONE_PCT)
            and float(row["close"]) >= half
        ):
            signals.append({
                "rank": 3,
                "trigger": half,
                "stop": half,
                "order_type": "limit",
                "reason": "连阳波段一半拉回; 一半位置止损",
                "known_volume_ratio": volume_ratio,
                "signal_type": "bull_run_half_pullback",
            })

    scored = [apply_entry_confluence(data, index, signal) for signal in signals]
    return max(scored, key=lambda item: item["rank"], default=None)
