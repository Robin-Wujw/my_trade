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

from apps.portfolio_backtest import (
    DEFAULT_FINANCIAL_CHUNK_SIZE,
    DEFAULT_FINANCIAL_TARGET_COVERAGE,
    DEFAULT_FINANCIAL_TIMEOUT,
    infer_price_frame_source,
    load_candidate_snapshots,
    load_price_frames,
    refresh_backtest_inputs,
    refresh_price_cache_directory,
    validate_backtest_input_coverage,
    validate_price_frame_coverage,
)
from scripts.candidate_filter_experiments import compact, transform_snapshots
from stock_research.core.paths import PATHS
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def load_inputs(
    start_date: str,
    end_date: str,
    *,
    candidate_directory,
    formula_history,
    price_kline_directory,
    no_price_database=False,
    allow_unsafe_financial=False,
):
    formula = pd.read_csv(formula_history)
    raw = load_candidate_snapshots(
        candidate_directory,
        start_date,
        end_date,
    )
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
    validate_backtest_input_coverage(
        snapshots,
        formula,
        start_date,
        end_date,
        candidate_directory=candidate_directory,
        allow_unsafe_financial=allow_unsafe_financial,
    )
    codes = {str(row["code"]) for rows in snapshots.values() for row in rows}
    use_price_database = (
        not no_price_database
        and Path(price_kline_directory).name != "akshare_raw"
    )
    price_frames = load_price_frames(
        codes,
        price_kline_directory,
        start_date=(pd.Timestamp(start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=end_date,
        source=infer_price_frame_source(price_kline_directory),
        prefer_database=use_price_database,
    )
    validate_price_frame_coverage(price_frames, codes, start_date, end_date)
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
        "left_grid_unit": 0.02,
        "left_grid_max_exposure": 0.20,
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
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
    )
    parser.add_argument(
        "--formula-history",
        default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"),
    )
    parser.add_argument(
        "--price-kline-directory",
        default="",
    )
    parser.add_argument("--no-price-database", action="store_true")
    parser.add_argument(
        "--no-refresh-inputs",
        action="store_true",
        help="research only: skip automatic K-line/financial/Formula33/candidate refresh; requires --allow-unsafe-financial",
    )
    parser.add_argument(
        "--refresh-price-source",
        choices=("akshare", "miniqmt", "miniqmt-akshare"),
        default="miniqmt",
    )
    parser.add_argument(
        "--refresh-metadata-source",
        choices=("akshare", "baostock", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--refresh-market-cap-source",
        choices=("auto", "tushare", "akshare", "akshare-capital", "none"),
        default="auto",
    )
    parser.add_argument(
        "--close-confirmed-execution",
        choices=("close_proxy", "next_open"),
        default="close_proxy",
    )
    parser.add_argument("--allow-pullback-pilot", action="store_true")
    parser.add_argument(
        "--allow-unsafe-financial",
        action="store_true",
        help="research only: allow candidate manifests marked financial_point_in_time=false",
    )
    parser.add_argument(
        "--financial-target-coverage",
        type=float,
        default=DEFAULT_FINANCIAL_TARGET_COVERAGE,
    )
    parser.add_argument(
        "--financial-chunk-size",
        type=int,
        default=DEFAULT_FINANCIAL_CHUNK_SIZE,
    )
    parser.add_argument(
        "--financial-timeout",
        type=int,
        default=DEFAULT_FINANCIAL_TIMEOUT,
    )
    parser.add_argument(
        "--left-grid-unit",
        type=float,
        action="append",
        help="left-grid unit fractions to test; can be repeated, e.g. 0 and 0.01",
    )
    parser.add_argument(
        "--left-grid-max-exposure",
        type=float,
        action="append",
        help="left-grid max exposure fractions to test; can be repeated",
    )
    args = parser.parse_args(argv)
    if not args.price_kline_directory:
        args.price_kline_directory = str(refresh_price_cache_directory(args.refresh_price_source))
    if args.no_refresh_inputs and not args.allow_unsafe_financial:
        raise RuntimeError(
            "--no-refresh-inputs is research-only because parameter sweeps must "
            "auto-refresh K-line, financial, Formula33, and candidate inputs first."
        )
    if not args.no_refresh_inputs:
        refresh_backtest_inputs(args)

    price_frames, snapshots, phases, trade_plans = load_inputs(
        args.start_date,
        args.end_date,
        candidate_directory=args.candidate_directory,
        formula_history=args.formula_history,
        price_kline_directory=args.price_kline_directory,
        no_price_database=args.no_price_database,
        allow_unsafe_financial=args.allow_unsafe_financial,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    transformed = transform_snapshots(snapshots, args.mode)
    configs = experiment_grid(args.mode)
    left_units = args.left_grid_unit or [0.02]
    left_caps = args.left_grid_max_exposure or [0.20]
    expanded_configs = []
    for config in configs:
        for left_unit in left_units:
            for left_cap in left_caps:
                expanded_configs.append({
                    **config,
                    "left_grid_unit": float(left_unit),
                    "left_grid_max_exposure": float(left_cap),
                })
    configs = expanded_configs
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
            left_grid_unit=config["left_grid_unit"],
            left_grid_max_exposure=config["left_grid_max_exposure"],
            signals_effective_next_day=True,
            auto_price_structure=True,
            allow_structure_pullback=True,
            allow_pullback_pilot=args.allow_pullback_pilot,
            close_confirmed_execution=args.close_confirmed_execution,
            commission_rate=0.000085,
            minimum_commission=5.0,
            initial_capital=1_000_000.0,
            sell_stamp_duty_rate=0.0005,
            estimated_slippage_rate=0.0005,
        )
        row = {
            **compact(f"{args.mode}_{index}", result),
            **config,
            "close_confirmed_execution": args.close_confirmed_execution,
            "allow_pullback_pilot": bool(args.allow_pullback_pilot),
            "allow_unsafe_financial": bool(args.allow_unsafe_financial),
            "price_kline_directory": str(args.price_kline_directory),
            "price_database_enabled": not args.no_price_database,
            "trade_mix_summary": result.get("trade_mix_summary"),
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
