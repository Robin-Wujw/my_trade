"""Run the point-in-time candidate portfolio backtest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def load_candidate_snapshots(directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    snapshots = {}
    for path in sorted(Path(directory).glob("candidates_*.csv")):
        date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
        if pd.isna(date) or not start <= date.normalize() <= end:
            continue
        snapshots[date.strftime("%Y-%m-%d")] = pd.read_csv(
            path, dtype={"code": str}, low_memory=False,
        ).to_dict("records")
    return snapshots


def load_price_frames(codes, directory):
    frames = {}
    for code in codes:
        path = Path(directory) / f"{str(code).replace('.', '_')}.csv"
        try:
            frames[str(code)] = pd.read_csv(path)
        except (OSError, ValueError):
            continue
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description="候选池、Formula33和三持仓组合回测")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-10")
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="maximum symbols; 0 disables the symbol-count cap while exposure stays <=100%",
    )
    parser.add_argument(
        "--codes",
        help="comma-separated explicit candidate universe; overrides snapshot members",
    )
    parser.add_argument(
        "--exit-tail-on-candidate-removal", action="store_true",
        help="exit right-side tails below 10% at the next open after candidate removal",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("rolling", "fixed-first"),
        default="rolling",
        help="rolling逐日更新候选；fixed-first冻结首个交易日候选",
    )
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "mainline-left-manual-v2"),
    )
    parser.add_argument(
        "--formula-history",
        default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"),
    )
    parser.add_argument(
        "--trade-plans",
        default=str(PATHS.project_root / "config" / "trade_plans.json"),
    )
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "portfolio"),
    )
    parser.add_argument(
        "--no-refresh-inputs", action="store_true",
        help="skip resumable mainline/candidate refresh; intended only for verified offline inputs",
    )
    args = parser.parse_args(argv)
    if not args.no_refresh_inputs and not args.codes:
        from apps import rebuild_candidate_history, rebuild_mainline_history

        rebuild_mainline_history.main([
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--candidate-directory", args.candidate_directory,
        ])
        rebuild_candidate_history.main([
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--output-directory", args.candidate_directory,
        ])
    snapshots = load_candidate_snapshots(
        args.candidate_directory, args.start_date, args.end_date,
    )
    explicit_items = [item.strip() for item in (args.codes or "").split(",") if item.strip()]
    explicit_pairs = [item.split("=", 1) for item in explicit_items]
    explicit_codes = [parts[0].strip() for parts in explicit_pairs]
    explicit_names = {
        parts[0].strip(): parts[1].strip()
        for parts in explicit_pairs if len(parts) == 2 and parts[1].strip()
    }
    if explicit_codes:
        first_date = min(snapshots) if snapshots else args.start_date
        snapshots = {
            first_date: [
                {
                    "code": code,
                    "name": explicit_names.get(code, code),
                    "strategy_part": "explicit candidate",
                }
                for code in explicit_codes
            ]
        }
    if args.candidate_mode == "fixed-first" and snapshots:
        first_date = min(snapshots)
        snapshots = {first_date: snapshots[first_date]}
    codes = {
        str(row["code"])
        for rows in snapshots.values()
        for row in rows
    }
    codes.update(load_trade_plans(args.trade_plans).get("plans", {}))
    price_frames = load_price_frames(
        codes, PATHS.cache / "formula33_kline" / "akshare",
    )
    formula = pd.read_csv(args.formula_history)
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    result = run_portfolio_backtest(
        price_frames,
        snapshots,
        phases,
        requested_start=args.start_date,
        end_date=args.end_date,
        trade_plans=load_trade_plans(args.trade_plans),
        max_positions=None if args.max_positions == 0 else args.max_positions,
        exit_tail_on_candidate_removal=args.exit_tail_on_candidate_removal,
        signals_effective_next_day=True,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{args.start_date}_{args.end_date}"
    pd.DataFrame(result["events"]).to_csv(
        output / f"{stem}_events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["equity_curve"]).to_csv(
        output / f"{stem}_equity.csv", index=False, encoding="utf-8-sig",
    )
    summary = {key: value for key, value in result.items() if key not in {"events", "equity_curve"}}
    summary["candidate_mode"] = args.candidate_mode
    if explicit_codes:
        summary["candidate_mode"] = "explicit_codes"
        summary["explicit_codes"] = explicit_codes
    summary["candidate_snapshot_dates"] = sorted(snapshots)
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
