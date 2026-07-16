"""Run 2026 candidate-filter portfolio experiments without DB writes."""
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
from stock_research.core.paths import PATHS
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def source_set(row) -> set[str]:
    return {item for item in str(row.get("candidate_source") or "").split("+") if item}


def disable(row: dict, reason: str) -> dict:
    item = dict(row)
    item["signal_eligible"] = False
    item["selected_for_trading"] = False
    old = str(item.get("candidate_failure_reason") or "").strip()
    item["candidate_failure_reason"] = f"{old}; {reason}".strip("; ")
    return item


def transform_snapshots(snapshots, mode: str):
    result = {}
    for date, rows in snapshots.items():
        transformed = []
        for row in rows:
            sources = source_set(row)
            rank = pd.to_numeric(row.get("selection_rank"), errors="coerce")
            trade_basis = pd.to_numeric(row.get("trade_basis_score"), errors="coerce")
            leadership = pd.to_numeric(row.get("leadership_score"), errors="coerce")
            item = dict(row)
            if mode == "baseline":
                pass
            elif mode == "no_value_source" and "value_model" in sources:
                item = disable(row, "experiment_no_value_source")
            elif mode == "mainline_only" and "standard_mainline" not in sources:
                item = disable(row, "experiment_mainline_only")
            elif mode == "mainline_growth_overlap" and not (
                {"standard_mainline", "growth_leadership"} <= sources
            ):
                item = disable(row, "experiment_mainline_growth_overlap")
            elif mode == "no_pure_growth" and sources == {"growth_leadership"}:
                item = disable(row, "experiment_no_pure_growth")
            elif mode == "top7_only" and (pd.isna(rank) or float(rank) > 7):
                item = disable(row, "experiment_top7_only")
            elif mode == "avoid_overheated_trade_basis" and pd.notna(trade_basis) and float(trade_basis) >= 10:
                item = disable(row, "experiment_avoid_trade_basis_10plus")
            elif mode == "leadership_15_to_23" and not (
                pd.notna(leadership) and 15 <= float(leadership) < 24
            ):
                item = disable(row, "experiment_leadership_15_to_23")
            transformed.append(item)
        result[date] = transformed
    return result


def compact(name, result):
    return {
        "name": name,
        "final_return_pct": result.get("final_return_pct"),
        "realized_return_pct": result.get("realized_return_pct"),
        "unrealized_return_pct": result.get("unrealized_return_pct"),
        "maximum_drawdown_pct": result.get("maximum_drawdown_pct"),
        "transaction_cost_pct": result.get("transaction_cost_pct"),
        "buy_count": result.get("trade_summary", {}).get("buy_count"),
        "sell_count": result.get("trade_summary", {}).get("sell_count"),
        "sell_win_rate_pct": result.get("trade_summary", {}).get("sell_win_rate_pct"),
        "entry_block_count": result.get("entry_block_count"),
        "right_buy_signal_type_counts": result.get("right_buy_signal_type_counts"),
        "final_positions": [
            {
                "code": item.get("code"),
                "name": item.get("name"),
                "position_pct": item.get("position_pct"),
                "unrealized_pnl_pct": item.get("unrealized_pnl_pct"),
            }
            for item in result.get("final_positions", [])
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--output-directory", default=str(PATHS.runtime_root / "backtests" / "candidate_filter_experiments"))
    parser.add_argument("--mode", action="append")
    args = parser.parse_args(argv)

    formula = pd.read_csv(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv")
    raw = load_candidate_snapshots(
        PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4",
        args.start_date,
        args.end_date,
    )
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
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
    modes = args.mode or [
        "baseline",
        "no_value_source",
        "mainline_only",
        "mainline_growth_overlap",
        "no_pure_growth",
        "top7_only",
        "avoid_overheated_trade_basis",
        "leadership_15_to_23",
    ]
    rows = []
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        result = run_portfolio_backtest(
            price_frames,
            transform_snapshots(snapshots, mode),
            phases,
            requested_start=args.start_date,
            end_date=args.end_date,
            trade_plans=trade_plans,
            max_positions=3,
            max_total_held_symbols=5,
            max_same_industry=2,
            same_theme_correlation=0.60,
            min_entry_evidence_score=7.0,
            profit_tranches=2,
            profit_tail_min_return=0.50,
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
        row = compact(mode, result)
        rows.append(row)
        (output / f"{mode}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda item: item["final_return_pct"], reverse=True)
    (output / "candidate_filter_experiments.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output / "candidate_filter_experiments.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
