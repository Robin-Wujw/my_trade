"""Point-in-time technical entries described by the trading framework."""
from __future__ import annotations

import pandas as pd

from stock_research.indicators.price_structure import confirmed_turning_points


def _volume_baseline(previous: pd.DataFrame) -> float:
    return max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )


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
    evidence = [f"primary:{signal.get('signal_type') or 'configured_structure'}"]
    evidence.extend(signal.get("context_evidence") or [])
    bonus = int(signal.get("context_bonus") or 0)
    if len(history) >= 20:
        volume_baseline = _volume_baseline(history)
        if volume_baseline > 0 and float(decision["volume"]) >= volume_baseline:
            evidence.append("volume_confirmed")
            bonus += 1

        level = float(signal.get("stop") or signal.get("trigger"))
        low_volume_cutoff = float(data.iloc[max(0, decision_index - 60):decision_index]["volume"].median()) * 0.8
        for period, weight in ((20, 1), (60, 2), (120, 3)):
            if decision_index < period + 5 or pd.isna(decision[f"ma{period}"]):
                continue
            ma = float(decision[f"ma{period}"])
            deducted_price = float(data.iloc[decision_index - period]["close"])
            deducted_volume = float(data.iloc[decision_index - period]["volume"])
            deduction_confluence = (
                abs(level / ma - 1) <= 0.03
                and ma > float(data.iloc[decision_index - 5][f"ma{period}"])
                and deducted_price <= ma * 0.95
                and deducted_volume <= low_volume_cutoff
                and float(decision["volume"]) >= deducted_volume
            )
            if deduction_confluence:
                evidence.append(f"deduction_low_price_volume_ma{period}")
                bonus += weight

    anchor_low = pd.to_numeric(signal.get("anchor_low"), errors="coerce")
    anchor_high = pd.to_numeric(signal.get("anchor_high"), errors="coerce")
    if (
        pd.notna(anchor_low)
        and pd.notna(anchor_high)
        and float(anchor_high) / float(anchor_low) - 1 >= 0.50
    ):
        evidence.append("large_wave_structure")
        bonus += 2

    prior = data.iloc[index - 1]
    row = data.iloc[index]
    if float(row["open"]) > float(prior["high"]):
        evidence.append("gap_up")
        bonus += 1
    scored["rank"] = int(signal.get("rank") or 0) + bonus
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
                if float(prior["close"]) <= neckline < float(row["close"]) and volume_ratio >= 1.0:
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
        and volume_ratio >= 1.0
    ):
        signals.append(_close_signal(
            "gap_long_ma_breakout", 5, float(row["close"]), float(prior["high"]),
            "跳空长阳越过MA20/MA60; 缺口下沿止损", volume_ratio,
        ))

    # Tight consolidation rather than a generic 21-day high touch.
    trend_high = float(previous["close"].tail(21).max())
    high_120 = float(previous["close"].tail(120).max())
    return_60 = float(row["close"]) / float(previous.iloc[-60]["close"]) - 1
    strong_ma_trend = (
        float(prior["ma20"]) > float(prior["ma60"])
        and float(prior["ma20"]) > float(previous.iloc[-6]["ma20"])
        and float(prior["ma60"]) > float(previous.iloc[-6]["ma60"])
    )
    durable_price_leadership = (
        return_60 >= 0.25 and float(row["close"]) >= high_120 * 0.95
    )
    if (
        strong_ma_trend
        and durable_price_leadership
        and float(prior["close"]) <= trend_high < float(row["close"])
        and volume_ratio >= 1.0
    ):
        signal = _close_signal(
            "strong_trend_breakout", 4, trend_high, trend_high,
            "强趋势收盘放量突破21日收盘高点; 突破位止损", volume_ratio,
        )
        signal["context_evidence"] = [
            "ma20_ma60_strong_trend", "durable_price_leadership",
        ]
        signal["context_bonus"] = 2
        signals.append(signal)

    # Tight consolidation rather than a generic 21-day high touch.
    box = previous.tail(21)
    box_high = float(box["high"].max())
    box_low = float(box["low"].min())
    if (
        box_high / box_low - 1 <= 0.15
        and float(prior["close"]) <= box_high < float(row["close"])
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
        if float(prior["close"]) > half and float(row["low"]) <= half <= float(row["close"]):
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
