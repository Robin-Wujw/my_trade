"""Sweep portfolio parameters without DB writes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import load_candidate_snapshots, load_price_frames, validate_backtest_input_coverage
from scripts.candidate_filter_experiments import compact, transform_snapshots
from stock_research.core.paths import PATHS
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def load_inputs(start_date: str, end_date: str):
    formula = pd.read_csv(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv")
    raw = load_candidate_snapshots(
        PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4",
        start_date,
        end_date,
    )
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
    validate_backtest_input_coverage(snapshots, formula, start_date, end_date)
    codes = {str(row["code"]) for rows in snapshots.values() for row in rows}
    price_frames = load_price_frames(
        codes,
        PATHS.cache / "formula33_kline" / "akshare",
        start_date=(pd.Timestamp(start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=end_date,
    )
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    return price_frames, snapshots, phases, load_trade_plans(PATHS.project_root / "config" / "trade_plans.json")


def experiment_grid(mode: str) -> list[dict]:
    base = {
        "mode": mode,
        "max_positions": 3,
        "max_total_held_symbols": 5,
        "profit_tranches": 5,
        "profit_tail_min_return": 0.50,
        "min_entry_evidence_score": 7.0,
    }
    configs = [base]
    for tranches in [2, 3, 4]:
        configs.append({**base, "profit_tranches": tranches})
    for max_positions in [4, 5]:
        configs.append({**base, "max_positions": max_positions})
    for total_symbols in [4, 6, 7]:
        configs.append({**base, "max_total_held_symbols": total_symbols})
    for tail_return in [0.30, 0.70, 1.00]:
        configs.append({**base, "profit_tail_min_return": tail_return})
    configs.append({**base, "profit_tranches": 2, "max_positions": 4})
    configs.append({**base, "profit_tranches": 2, "max_positions": 5})
    configs.append({**base, "profit_tranches": 3, "max_total_held_symbols": 6})
    return configs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--mode", default="no_value_source")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-directory", default=str(PATHS.runtime_root / "backtests" / "portfolio_parameter_experiments"))
    args = parser.parse_args(argv)

    price_frames, snapshots, phases, trade_plans = load_inputs(args.start_date, args.end_date)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    transformed = transform_snapshots(snapshots, args.mode)
    configs = experiment_grid(args.mode)
    if args.limit:
        configs = configs[:max(1, int(args.limit))]
    for index, config in enumerate(configs, start=1):
        result = run_portfolio_backtest(
            price_frames,
            transformed,
            phases,
            requested_start=args.start_date,
            end_date=args.end_date,
            trade_plans=trade_plans,
            max_positions=config["max_positions"],
            max_total_held_symbols=config["max_total_held_symbols"],
            max_same_industry=2,
            same_theme_correlation=0.60,
            min_entry_evidence_score=config["min_entry_evidence_score"],
            profit_tranches=config["profit_tranches"],
            profit_tail_min_return=config["profit_tail_min_return"],
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
        row = {
            **compact(f"{args.mode}_{index}", result),
            **config,
        }
        rows.append(row)
        (output / f"parameter_experiment_{index:02d}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda item: item["final_return_pct"], reverse=True)
    (output / "parameter_experiments.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output / "parameter_experiments.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
