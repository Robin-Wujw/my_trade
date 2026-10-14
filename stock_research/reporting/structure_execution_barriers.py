"""Describe source-bound R03 exit barriers without inventing order attempts.

The older portfolio CSVs contain fills, not a complete exit-attempt ledger.
They cannot establish a zero count of blocked exits. This report consumes the
explicit barriers in executable structure labels; entry-day OHLC observations
remain separate from failed orders and do not alter the registered contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import pandas as pd

from stock_research.backtesting.inputs import file_sha256


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("Barrier diagnostics require explicit local session dates")
    return day


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _entry_day_observation(label, buy, bar):
    """An intraday low may precede an intraday buy; do not infer its order."""
    if bar is None:
        return "unavailable_entry_bar", None
    low = _number(bar.get("raw_low", bar.get("low")))
    close = _number(bar.get("raw_close", bar.get("close")))
    price, quantity = _number(buy.get("execution_price")), _number(buy.get("execution_quantity"))
    risk = _number(label.get("initial_risk_amount"))
    if None in (low, close, price, quantity, risk) or min(low, close, price, quantity, risk) <= 0:
        return "unavailable_entry_bar_or_risk", None
    stop = price - risk / quantity
    if stop <= 0 or close < low:
        return "unavailable_entry_bar_or_risk", None
    if buy.get("execution_phase") == "open":
        return ("confirmed_entry_day_breach" if low < stop else "no_entry_day_breach"), stop
    if buy.get("execution_phase") == "intraday":
        if close < stop:
            return "confirmed_entry_day_close_breach", stop
        return ("ambiguous_intraday_order" if low < stop else "no_entry_day_breach"), stop
    return "unavailable_entry_phase", stop


def execution_barrier_diagnostics(labels, sessions, *, entry_bars=None):
    """Count explicit attempts and calendar-session delays, separately by T20/60.

    entry_bars is keyed by (label_source_sha256, code, date); source mismatches
    cannot be silently replaced with current prices. Pending episodes have a
    known age and an unknown eventual delay. R tails use fully settled labels.
    """
    calendar = [_day(day) for day in sessions]
    if calendar != sorted(set(calendar)):
        raise ValueError("Barrier diagnostics require a unique ordered exchange calendar")
    positions = {day: index for index, day in enumerate(calendar)}
    rows, identities = [], set()
    for label in labels:
        identity = (label["event_id"], label["horizon"])
        if type(identity[1]) is not int or identity[1] not in (20, 60):
            raise ValueError("Only registered T20/T60 horizons are supported")
        if identity in identities:
            raise ValueError("Duplicate event/horizon barrier identities")
        identities.add(identity)
        row = {"event_id": identity[0], "horizon": identity[1], "label_status": label["status"],
               "source_sha256": label.get("label_source_sha256"), "episodes": [],
               "blocked_attempts": None, "entry_day_observation": "unavailable_execution",
               "net_r": None}
        execution = label.get("execution")
        if not execution or "blocked_exits" not in label:
            row["status"] = "unavailable_attempt_ledger"
            rows.append(row)
            continue
        events = execution.get("events", [])
        buys = [event for event in events if (_number(event.get("execution_quantity")) or 0) > 0]
        sells = [event for event in events if (_number(event.get("execution_quantity")) or 0) < 0]
        if len(buys) != 1:
            raise ValueError("R03 barrier report requires one independent entry")
        buy = buys[0]
        code, buy_day = buy["code"], _day(buy["date"])
        row["code"] = code
        source_key = (label.get("label_source_sha256"), code, str(buy_day.date()))
        observation, stop = _entry_day_observation(label, buy, (entry_bars or {}).get(source_key))
        row.update(entry_day_observation=observation, initial_stop_raw=stop)
        curve_days = [_day(point["date"]) for point in execution.get("equity_curve", [])]
        sell_days = [_day(event["date"]) for event in sells]
        blocks = label["blocked_exits"]
        block_days = [_day(item["date"]) for item in blocks]
        if curve_days != sorted(set(curve_days)) or block_days != sorted(set(block_days)):
            raise ValueError("R03 attempt/valuation dates must be unique and ordered")
        if not curve_days or any(day not in positions for day in [buy_day, *curve_days, *sell_days, *block_days]):
            raise ValueError("Execution dates are outside the archived exchange calendar")
        if any(day not in curve_days for day in [buy_day, *sell_days, *block_days]):
            raise ValueError("Attempts and fills are outside the actually observed execution")
        if any(day <= buy_day or day > curve_days[-1] for day in block_days):
            raise ValueError("Exit barriers must occur after entry and within observed execution")
        if any(day <= buy_day for day in sell_days):
            raise ValueError("R03 sell violates T+1")
        remaining = list(zip(block_days, blocks))
        while remaining:
            first = remaining[0][0]
            next_fill = next((day for day in sorted(sell_days) if day >= first), None)
            episode = [item for item in remaining if next_fill is None or item[0] <= next_fill]
            remaining = [item for item in remaining if next_fill is not None and item[0] > next_fill]
            last_observed = curve_days[-1]
            row["episodes"].append({
                "first_block_date": str(first.date()), "next_sell_date": str(next_fill.date()) if next_fill is not None else None,
                "status": "resolved" if next_fill is not None else "pending",
                "blocked_attempts": len(episode), "reasons": dict(Counter(item[1]["status"] for item in episode)),
                "delay_sessions": positions[next_fill] - positions[first] if next_fill is not None else None,
                "observed_age_sessions": positions[last_observed] - positions[first] if next_fill is None else None,
            })
        row.update(status="covered", blocked_attempts=len(blocks),
                   blocked_reasons=dict(Counter(item["status"] for item in blocks)))
        if label["status"] == "completed":
            row["net_r"] = _number(label.get("net_r"))
        rows.append(row)
    horizons = []
    for horizon in sorted({row["horizon"] for row in rows}):
        subset = [row for row in rows if row["horizon"] == horizon]
        covered = [row for row in subset if row["status"] == "covered"]
        episodes = [episode for row in covered for episode in row["episodes"]]
        reasons = Counter()
        for row in covered:
            reasons.update(row["blocked_reasons"])
        realized = [row["net_r"] for row in covered if row["net_r"] is not None]
        delays = [episode["delay_sessions"] for episode in episodes if episode["status"] == "resolved"]
        observations = Counter(row["entry_day_observation"] for row in covered)
        horizons.append({
            "horizon": horizon, "label_count": len(subset), "attempt_ledger_covered_labels": len(covered),
            "unavailable_attempt_labels": len(subset) - len(covered),
            "blocked_attempts": sum(row["blocked_attempts"] for row in covered) if covered else None,
            "blocked_label_count": sum(row["blocked_attempts"] > 0 for row in covered) if covered else None,
            "blocked_reasons": dict(reasons), "resolved_episode_count": len(delays) if covered else None,
            "pending_episode_count": sum(episode["status"] == "pending" for episode in episodes) if covered else None,
            "maximum_resolved_delay_sessions": max(delays) if delays else None,
            "entry_day_observations": dict(observations), "settled_net_r_count": len(realized),
            "loss_tails": [{"threshold_r": threshold, "count": sum(value < -threshold for value in realized),
                            "fraction": sum(value < -threshold for value in realized) / len(realized) if realized else None}
                           for threshold in (1., 1.5)],
        })
    status = ("unavailable_no_execution_labels" if not rows else
              "unavailable_attempt_ledger" if all(row["status"] != "covered" for row in rows) else "descriptive")
    return {"version": "structure-execution-barriers-v1", "research_only": True, "strict_point_in_time": False,
            "status": status, "by_horizon": horizons, "rows": rows,
            "note": "只统计可执行标签的明确阻断；历史组合成交CSV不能证明零阻断。买入日触及为T+1不可售观察，不是已提交止损；盘中买入前后顺序不明单列。延迟从首次明确阻断到下一笔卖出按交易日历计；未结束为空。T20/T60分开，重复/重叠事件不算独立样本。净R仅取完整清仓及权益结算标签。"}


def build_barrier_report(manifest_path, *, manifest_sha256):
    """Verify an exact manifest and its archives before consuming local prices."""
    from stock_research.research.structure_label_store import load_structure_labels

    path = Path(manifest_path).resolve()
    if path.name != "manifest.json" or file_sha256(path) != manifest_sha256:
        raise ValueError("Barrier report requires the exact source manifest/hash")
    document, labels = load_structure_labels(path.parent)
    sessions = json.loads((path.parent / "calendar.json").read_text(encoding="utf-8"))
    frames = pd.read_pickle(path.parent / "execution_prices.pkl.gz")
    entry_bars = {}
    for code, frame in frames.items():
        for bar in frame.to_dict("records"):
            entry_bars[(document["source_sha256"], code, str(_day(bar["date"]).date()))] = bar
    report = execution_barrier_diagnostics(labels, sessions, entry_bars=entry_bars)
    # Reverify after reading, so a changed source cannot get a valid receipt.
    load_structure_labels(path.parent)
    if file_sha256(path) != manifest_sha256:
        raise ValueError("Barrier report manifest changed during read")
    report["source"] = {"manifest_path": str(path), "manifest_sha256": manifest_sha256,
                        "captured_at": document["captured_at"], "files_sha256": document["files_sha256"]}
    report["implementation_sha256"] = file_sha256(__file__)
    return report


def render_markdown(report):
    lines = ["# 形态执行阻断与T+1观察", "", report["note"], "",
             "| 期限 | 标签 | 有阻断账本 | 阻断尝试 | 未结束延迟 | 最大已结束延迟（交易日） | 完整净R |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["by_horizon"]:
        values = [row[key] for key in ("horizon", "label_count", "attempt_ledger_covered_labels", "blocked_attempts",
                                       "pending_episode_count", "maximum_resolved_delay_sessions", "settled_net_r_count")]
        lines.append("| " + " | ".join("未知" if value is None else str(value) for value in values) + " |")
    for row in report["by_horizon"]:
        lines.extend(["", f"T{row['horizon']} 阻断原因：{json.dumps(row['blocked_reasons'], ensure_ascii=False)}；"
                          f"买入日观察：{json.dumps(row['entry_day_observations'], ensure_ascii=False)}。", ""])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读汇总来源绑定的形态执行阻断、T+1观察和净R尾部")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_barrier_report(args.manifest, manifest_sha256=args.manifest_sha256)
    output = Path(args.output).resolve()
    source = Path(args.manifest).resolve().parent
    if output == source or source in output.parents:
        raise ValueError("Barrier report cannot overwrite immutable source archives")
    output.mkdir(parents=True, exist_ok=True)
    (output / "barriers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "barriers.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
