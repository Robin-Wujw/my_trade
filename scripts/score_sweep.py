"""Sweep right-side entry evidence scoring without changing production defaults."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import (
    load_candidate_snapshots,
    load_price_frames,
    validate_backtest_input_coverage,
)
from stock_research.core.paths import PATHS
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies import portfolio_backtest as pb
from stock_research.indicators import technical_entries as te


TARGET_CODES = {
    "sz.300308": "中际旭创",
    "sz.300502": "新易盛",
    "sh.601138": "工业富联",
    "sh.601872": "招商轮船",  # kept as a neutral sanity target if present
    "sh.601899": "紫金矿业",
    "sh.603606": "东方电缆",
    "sh.600489": "中金黄金",
    "sh.601600": "中国铝业",
    "sh.601677": "明泰铝业",
    "sh.601168": "西部矿业",
    "sh.600549": "厦门钨业",
    "sh.603993": "洛阳钼业",
    "sh.601869": "长飞光纤",
}


def truthy(value, default=False):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "<na>"}:
            return default
        return text in {"1", "true", "yes", "y"}
    return bool(value)


def prefilter_snapshots(snapshots):
    """Keep rows that can affect production execution; drop dead diagnostics."""
    result = {}
    for date, rows in snapshots.items():
        kept = []
        for row in rows:
            failure = str(row.get("candidate_failure_reason") or "")
            selected = truthy(row.get("selected_for_trading"), default=True)
            signal_eligible = truthy(row.get("signal_eligible"), default=True)
            value_falsified = truthy(row.get("value_falsified"), default=False)
            if selected or signal_eligible:
                kept.append(row)
            elif failure.startswith("not_selected_for_trading: daily_top10_quota_or_core_reservation"):
                kept.append(row)
            elif row.get("value_falsification_reason") or value_falsified:
                kept.append(row)
        result[date] = kept
    return result


BASE_RANKS = {
    "w_bottom_neckline": 6,
    "gap_long_ma_breakout": 5,
    "uptrend_50_reclaim": 5,
    "pullback_50_breakout": 5,
    "consolidation_breakout": 4,
    "volume_price_node": 3,
    "bull_run_half_pullback": 3,
    "uptrend_support_pullback": 3,
}


SWEEP_CONTEXT = {}


def init_worker(context):
    global SWEEP_CONTEXT
    SWEEP_CONTEXT = context


def run_one_worker(payload):
    index, total, config = payload
    print(f"[{index}/{total}] running {config['name']}", flush=True)
    scorer = make_confluence(config)
    te.apply_entry_confluence = scorer
    pb.apply_entry_confluence = scorer
    context = SWEEP_CONTEXT
    result = pb.run_portfolio_backtest(
        context["price_frames"],
        context["snapshots"],
        context["phases"],
        requested_start=context["start_date"],
        end_date=context["end_date"],
        trade_plans=context["trade_plans"],
        max_positions=3,
        max_total_held_symbols=5,
        max_same_industry=2,
        same_theme_correlation=0.60,
        min_entry_evidence_score=config["threshold"],
        profit_tranches=5,
        profit_tail_min_return=0.50,
        exit_tail_on_candidate_removal=False,
        signals_effective_next_day=True,
        auto_price_structure=True,
        allow_structure_pullback=True,
        allow_pullback_pilot=False,
        close_confirmed_execution="close_proxy",
        commission_rate=0.000085,
        minimum_commission=5.0,
        initial_capital=1_000_000.0,
        sell_stamp_duty_rate=0.0005,
        estimated_slippage_rate=0.0005,
    )
    row = compact_result(config, result)
    output = Path(context["output_directory"])
    (output / f"{index:02d}_{config['name']}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"[{index}/{total}] {row['name']} "
        f"return={row['final_return_pct']} dd={row['maximum_drawdown_pct']} "
        f"targets={list(row['target_buys'])}",
        flush=True,
    )
    return row


def number(value) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return float(parsed) if pd.notna(parsed) else None


def make_confluence(config):
    base_ranks = {**BASE_RANKS, **config.get("base_ranks", {})}
    weights = {
        "context_scale": 1.0,
        "volume": 1,
        "volume_13x": 1,
        "tight": 1,
        "extended": -2,
        "stop_7": -1,
        "stop_10": -2,
        "above_or_reclaim_ma20": 1,
        "near_or_reclaim_ma20": 2,
        "weak_ma": -2,
        "ma_alignment": 2,
        "deduction_scale": 1.0,
        "large_wave": 2,
        "gap_up": 1,
        **config.get("weights", {}),
    }

    def apply_entry_confluence(data: pd.DataFrame, index: int, signal: dict) -> dict:
        scored = dict(signal)
        signal_type = signal.get("signal_type") or "configured_structure"
        decision_index = index if signal.get("order_type") == "close" else index - 1
        decision = data.iloc[decision_index]
        history = data.iloc[:decision_index]
        score = int(base_ranks.get(signal_type, signal.get("rank") or 0))
        evidence = [f"primary:{signal_type}+{score}"]

        context_bonus = int(signal.get("context_bonus") or 0)
        if context_bonus:
            scaled = int(round(context_bonus * float(weights["context_scale"])))
            evidence.extend(signal.get("context_evidence") or [])
            evidence.append(f"context_bonus+{scaled}")
            score += scaled
        else:
            evidence.extend(signal.get("context_evidence") or [])

        if len(history) >= 20:
            volume_baseline = te._volume_baseline(history)
            if volume_baseline > 0 and float(decision["volume"]) >= volume_baseline:
                delta = int(weights["volume"])
                evidence.append(f"volume_confirmed+{delta}")
                score += delta
                if float(decision["volume"]) >= volume_baseline * 1.30:
                    delta = int(weights["volume_13x"])
                    evidence.append(f"volume_expanded_1_3x+{delta}")
                    score += delta

            level = float(signal.get("stop") or signal.get("trigger"))
            entry_reference = (
                float(decision["close"])
                if signal.get("order_type") == "close"
                else float(signal.get("trigger") or decision["close"])
            )
            if te._in_zone(entry_reference, level):
                evidence.append("entry_within_structure_zone")
                if te._in_zone(entry_reference, level, te.TIGHT_STRUCTURE_ZONE_PCT):
                    delta = int(weights["tight"])
                    evidence.append(f"entry_tight_to_structure+{delta}")
                    score += delta
            elif entry_reference > level * (1 + te.SUPPORT_ZONE_PCT):
                delta = int(weights["extended"])
                evidence.append(f"entry_extended_from_structure{delta}")
                score += delta
            if level > 0:
                stop_distance = entry_reference / level - 1
                if stop_distance > 0.10:
                    delta = int(weights["stop_10"])
                    evidence.append(f"stop_distance_above_10pct{delta}")
                    score += delta
                elif stop_distance > 0.07:
                    delta = int(weights["stop_7"])
                    evidence.append(f"stop_distance_above_7pct{delta}")
                    score += delta

            ma20 = number(decision.get("ma20"))
            ma60 = number(decision.get("ma60"))
            prior_close = float(data.iloc[decision_index - 1]["close"]) if decision_index > 0 else None
            prior_ma20 = number(data.iloc[decision_index - 1].get("ma20")) if decision_index > 0 else None
            prior_ma60 = number(data.iloc[decision_index - 1].get("ma60")) if decision_index > 0 else None
            five_day_ma20 = number(data.iloc[decision_index - 6].get("ma20")) if decision_index >= 6 else None
            five_day_ma60 = number(data.iloc[decision_index - 6].get("ma60")) if decision_index >= 6 else None
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
                    delta = int(weights["above_or_reclaim_ma20"])
                    evidence.append(f"close_above_rising_or_reclaimed_ma20+{delta}")
                    score += delta
                low = number(decision.get("low"))
                close = number(decision.get("close"))
                near_rising_ma20 = ma20_rising and abs(level / ma20 - 1) <= te.SUPPORT_ZONE_PCT
                pulled_to_rising_ma20 = (
                    ma20_rising
                    and low is not None
                    and close is not None
                    and low <= ma20 * (1 + te.SUPPORT_ZONE_PCT)
                    and close >= ma20 * (1 - te.SUPPORT_ZONE_PCT)
                )
                if reclaimed_ma20 or near_rising_ma20 or pulled_to_rising_ma20:
                    delta = int(weights["near_or_reclaim_ma20"])
                    evidence.append(f"entry_near_or_reclaims_ma20+{delta}")
                    score += delta
                if ma60 is not None and entry_reference < ma20 and ma20 < ma60:
                    delta = int(weights["weak_ma"])
                    evidence.append(f"weak_ma20_below_ma60_and_price_below_ma20{delta}")
                    score += delta
            if medium_long_rising:
                delta = int(weights["ma_alignment"])
                evidence.append(f"ma20_ma60_rising_alignment+{delta}")
                score += delta

            low_volume_cutoff = float(data.iloc[max(0, decision_index - 60):decision_index]["volume"].median()) * 0.8
            for period, default_weight in ((20, 1), (60, 2), (120, 3)):
                if decision_index < period + 5 or pd.isna(decision[f"ma{period}"]):
                    continue
                ma = float(decision[f"ma{period}"])
                deducted_price = float(data.iloc[decision_index - period]["close"])
                deducted_volume = float(data.iloc[decision_index - period]["volume"])
                deduction_confluence = (
                    abs(level / ma - 1) <= te.SUPPORT_ZONE_PCT
                    and ma > float(data.iloc[decision_index - 5][f"ma{period}"])
                    and deducted_price <= ma * 0.95
                    and deducted_volume <= low_volume_cutoff
                    and float(decision["volume"]) >= deducted_volume
                )
                if deduction_confluence:
                    delta = int(round(default_weight * float(weights["deduction_scale"])))
                    evidence.append(f"deduction_low_price_volume_ma{period}+{delta}")
                    score += delta

        anchor_low = pd.to_numeric(signal.get("anchor_low"), errors="coerce")
        anchor_high = pd.to_numeric(signal.get("anchor_high"), errors="coerce")
        if (
            pd.notna(anchor_low)
            and pd.notna(anchor_high)
            and float(anchor_high) / float(anchor_low) - 1 >= 0.50
        ):
            delta = int(weights["large_wave"])
            evidence.append(f"large_wave_structure+{delta}")
            score += delta

        prior = data.iloc[index - 1]
        row = data.iloc[index]
        if float(row["open"]) > float(prior["high"]):
            delta = int(weights["gap_up"])
            evidence.append(f"gap_up+{delta}")
            score += delta
        scored["rank"] = max(0, score)
        scored["entry_evidence_score"] = scored["rank"]
        scored["entry_evidence"] = evidence
        return scored

    return apply_entry_confluence


def configs():
    return [
        {"name": "current_score8", "threshold": 8.0, "base_ranks": {}, "weights": {}},
        {"name": "threshold7", "threshold": 7.0, "base_ranks": {}, "weights": {}},
        {"name": "threshold65", "threshold": 6.5, "base_ranks": {}, "weights": {}},
        {"name": "platform_node_plus1_t7", "threshold": 7.0, "base_ranks": {"consolidation_breakout": 5, "volume_price_node": 4}, "weights": {}},
        {"name": "pullback_plus1_t7", "threshold": 7.0, "base_ranks": {"bull_run_half_pullback": 4, "uptrend_support_pullback": 4}, "weights": {}},
        {"name": "u_r_plus1_t75", "threshold": 7.5, "base_ranks": {"uptrend_50_reclaim": 6, "pullback_50_breakout": 6}, "weights": {}},
        {"name": "structure_flatter_t7", "threshold": 7.0, "base_ranks": {"w_bottom_neckline": 5, "gap_long_ma_breakout": 5, "uptrend_50_reclaim": 5, "pullback_50_breakout": 5, "consolidation_breakout": 5, "volume_price_node": 4, "bull_run_half_pullback": 4, "uptrend_support_pullback": 4}, "weights": {}},
        {"name": "less_ma_bonus_t7", "threshold": 7.0, "base_ranks": {}, "weights": {"near_or_reclaim_ma20": 1, "ma_alignment": 1}},
        {"name": "more_volume_t75", "threshold": 7.5, "base_ranks": {}, "weights": {"volume": 2, "volume_13x": 2}},
        {"name": "more_close_structure_t75", "threshold": 7.5, "base_ranks": {}, "weights": {"tight": 2, "extended": -3}},
        {"name": "less_stop_penalty_t75", "threshold": 7.5, "base_ranks": {}, "weights": {"stop_7": 0, "stop_10": -1}},
        {"name": "strong_trend_t8", "threshold": 8.0, "base_ranks": {}, "weights": {"ma_alignment": 3, "large_wave": 3, "gap_up": 2}},
        {"name": "big_wave_t75", "threshold": 7.5, "base_ranks": {}, "weights": {"large_wave": 4, "deduction_scale": 1.5}},
        {"name": "breakout_bias_t7", "threshold": 7.0, "base_ranks": {"w_bottom_neckline": 6, "gap_long_ma_breakout": 6, "consolidation_breakout": 5, "volume_price_node": 4}, "weights": {"volume": 2}},
        {"name": "pullback_bias_t7", "threshold": 7.0, "base_ranks": {"uptrend_50_reclaim": 6, "pullback_50_breakout": 6, "bull_run_half_pullback": 4, "uptrend_support_pullback": 4}, "weights": {"near_or_reclaim_ma20": 2, "large_wave": 3}},
        {"name": "quality_over_quantity_t8", "threshold": 8.0, "base_ranks": {"consolidation_breakout": 5, "volume_price_node": 4}, "weights": {"volume": 2, "tight": 2, "extended": -3, "weak_ma": -3}},
        {"name": "early_structure_t65", "threshold": 6.5, "base_ranks": {"consolidation_breakout": 5, "volume_price_node": 4, "bull_run_half_pullback": 4}, "weights": {"stop_7": -2, "stop_10": -4}},
        {"name": "balanced_candidate", "threshold": 7.0, "base_ranks": {"consolidation_breakout": 5, "volume_price_node": 4, "uptrend_50_reclaim": 6, "pullback_50_breakout": 6}, "weights": {"volume": 2, "tight": 2, "ma_alignment": 2, "large_wave": 3, "extended": -3}},
    ]


def target_summary(result):
    buys = [event for event in result["events"] if event.get("trade_side") == "买入"]
    by_code = {}
    for event in buys:
        code = event.get("code")
        if code not in TARGET_CODES or code in by_code:
            continue
        by_code[code] = {
            "name": event.get("name") or TARGET_CODES[code],
            "date": event.get("date"),
            "action": event.get("action"),
            "price": event.get("price"),
            "reason": event.get("reason"),
            "entry_evidence_score": event.get("entry_evidence_score"),
            "signal_type": event.get("signal_type"),
        }
    final_codes = {item["code"] for item in result.get("final_positions", [])}
    for code, item in by_code.items():
        item["held_to_end"] = code in final_codes
    return by_code


def compact_result(config, result):
    targets = target_summary(result)
    score_blocks = [
        block for block in result.get("entry_blocks", [])
        if str(block.get("reason", "")).startswith("entry_evidence_score_below")
    ]
    final_return = float(result.get("final_return_pct") or 0.0)
    max_dd = float(result.get("maximum_drawdown_pct") or 0.0)
    target_bonus = len(targets) * 3.0
    risk_adjusted = final_return + max_dd * 1.25 + target_bonus
    return {
        "name": config["name"],
        "threshold": config["threshold"],
        "base_ranks": config.get("base_ranks", {}),
        "weights": config.get("weights", {}),
        "final_return_pct": result.get("final_return_pct"),
        "maximum_drawdown_pct": result.get("maximum_drawdown_pct"),
        "realized_return_pct": result.get("realized_return_pct"),
        "unrealized_return_pct": result.get("unrealized_return_pct"),
        "event_count": result.get("event_count"),
        "buy_count": result.get("trade_summary", {}).get("buy_count"),
        "sell_win_rate_pct": result.get("trade_summary", {}).get("sell_win_rate_pct"),
        "entry_block_count": result.get("entry_block_count"),
        "score_block_count": len(score_blocks),
        "concentration_block_count": result.get("concentration_block_count"),
        "structure_signal_counts": result.get("structure_signal_counts"),
        "target_buys": targets,
        "final_positions": [
            {
                "code": item.get("code"),
                "name": item.get("name"),
                "position_pct": item.get("position_pct"),
                "unrealized_pnl_pct": item.get("unrealized_pnl_pct"),
            }
            for item in result.get("final_positions", [])
        ],
        "risk_adjusted_score": round(risk_adjusted, 3),
        "objective_score": final_return,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2024-09-24")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--output-directory", default=str(PATHS.runtime_root / "backtests" / "score_sweep"))
    parser.add_argument("--candidate-directory", default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"))
    parser.add_argument("--formula-history", default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config-name", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--codes",
        help="comma-separated explicit codes; builds a synthetic always-visible candidate set",
    )
    args = parser.parse_args(argv)

    formula = pd.read_csv(args.formula_history)
    explicit_codes = [item.strip() for item in (args.codes or "").split(",") if item.strip()]
    if explicit_codes:
        formula_dates = pd.to_datetime(formula["date"], errors="coerce").dropna()
        snapshots = {
            date.strftime("%Y-%m-%d"): [
                {
                    "code": code,
                    "name": TARGET_CODES.get(code, code),
                    "strategy_part": "explicit score sweep",
                    "selected_for_trading": True,
                    "signal_eligible": True,
                    "allow_right": True,
                    "allow_left": False,
                    "candidate_score": 100.0,
                    "trade_basis_score": 8.0,
                    "leadership_score": 30.0,
                }
                for code in explicit_codes
            ]
            for date in formula_dates
            if pd.Timestamp(args.start_date) <= date.normalize() <= pd.Timestamp(args.end_date)
        }
    else:
        snapshots = load_candidate_snapshots(args.candidate_directory, args.start_date, args.end_date)
        before = sum(len(rows) for rows in snapshots.values())
        snapshots = normalize_candidate_snapshots(snapshots, include_diagnostics=True)
        normalized = sum(len(rows) for rows in snapshots.values())
        snapshots = prefilter_snapshots(snapshots)
        after = sum(len(rows) for rows in snapshots.values())
        print(f"[input] normalized/prefiltered snapshots rows {before} -> {normalized} -> {after}", flush=True)
        validate_backtest_input_coverage(snapshots, formula, args.start_date, args.end_date)
    codes = {str(row["code"]) for rows in snapshots.values() for row in rows}
    price_frames = load_price_frames(
        codes,
        PATHS.cache / "formula33_kline" / "akshare",
        start_date=(pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=args.end_date,
    )
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    trade_plans = load_trade_plans(PATHS.project_root / "config" / "trade_plans.json")

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    all_configs = configs()
    if args.config_name:
        wanted = set(args.config_name)
        all_configs = [item for item in all_configs if item["name"] in wanted]
    if args.limit:
        all_configs = all_configs[:args.limit]

    tasks = list(enumerate(all_configs, 1))
    task_payloads = [(index, len(all_configs), config) for index, config in tasks]
    context = {
        "price_frames": price_frames,
        "snapshots": snapshots,
        "phases": phases,
        "trade_plans": trade_plans,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "output_directory": str(output),
    }
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(context,),
        ) as pool:
            rows = []
            for row in pool.imap_unordered(run_one_worker, task_payloads):
                rows.append(row)
                partial = sorted(rows, key=lambda item: item["objective_score"], reverse=True)
                (output / f"score_sweep_{args.start_date}_{args.end_date}.partial.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                pd.DataFrame(partial).to_csv(
                    output / f"score_sweep_{args.start_date}_{args.end_date}.partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
    else:
        init_worker(context)
        rows = []
        for payload in task_payloads:
            row = run_one_worker(payload)
            rows.append(row)
            partial = sorted(rows, key=lambda item: item["objective_score"], reverse=True)
            (output / f"score_sweep_{args.start_date}_{args.end_date}.partial.json").write_text(
                json.dumps(partial, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pd.DataFrame(partial).to_csv(
                output / f"score_sweep_{args.start_date}_{args.end_date}.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
    rows.sort(key=lambda item: item["objective_score"], reverse=True)
    (output / f"score_sweep_{args.start_date}_{args.end_date}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        output / f"score_sweep_{args.start_date}_{args.end_date}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(rows[:8], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
