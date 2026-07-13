"""Run the point-in-time candidate portfolio backtest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from stock_research.core.paths import PATHS
from stock_research.reporting.backtest_trade_report import (
    build_readable_trade_frame,
    render_trade_report_markdown,
)
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.storage import Database, KlineRepository, ResearchRepository
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def load_candidate_snapshots(directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    snapshots = {}
    for path in sorted(Path(directory).glob("candidates_*.csv")):
        date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
        if pd.isna(date) or not start <= date.normalize() <= end:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except EmptyDataError:
            frame = pd.DataFrame()
        snapshots[date.strftime("%Y-%m-%d")] = frame.to_dict("records")
    return snapshots


def load_price_frames(codes, directory, *, start_date=None, end_date=None):
    """Use DuckDB as the batch-read authority, with CSV as a compatibility fallback."""
    normalized_codes = sorted({str(code) for code in codes})
    frames = {}
    if PATHS.database.is_file() and normalized_codes:
        repository = KlineRepository(Database(PATHS.database))
        loaded = repository.load_stock_klines(
            "akshare",
            normalized_codes,
            start_date=str(start_date or "1900-01-01"),
            end_date=str(end_date or "2999-12-31"),
        )
        if not loaded.empty:
            frames.update({
                str(code): group.drop(columns=["code"]).reset_index(drop=True)
                for code, group in loaded.groupby("code", sort=False)
            })
    for code in (item for item in normalized_codes if item not in frames):
        path = Path(directory) / f"{str(code).replace('.', '_')}.csv"
        try:
            frame = pd.read_csv(path)
            if start_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(start_date)]
            if end_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") <= pd.Timestamp(end_date)]
            frames[str(code)] = frame
        except (OSError, ValueError):
            continue
    return frames


def default_data_end_date(now=None):
    """Return the latest session whose daily bar should already be available."""
    current = pd.Timestamp(now or pd.Timestamp.now())
    target = current.normalize()
    if current.weekday() >= 5:
        target = target - pd.offsets.BDay(1)
    elif current.hour < 16:
        target = target - pd.offsets.BDay(1)
    return target.strftime("%Y-%m-%d")


def refresh_backtest_inputs(args):
    """Refresh source data first, then rebuild every derived backtest artifact."""
    from apps import (
        rebuild_candidate_history,
        rebuild_formula_history,
        rebuild_mainline_history,
    )
    from stock_research.pipelines import formula33
    from stock_research.core.completion_manifest import CompletionManifest

    print(
        "[portfolio_backtest][refresh] first refresh full-market K-lines "
        f"through {args.end_date}",
        flush=True,
    )
    formula33.main([
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--lookback", "21",
        "--history-days", "420",
        "--workers", "8",
        "--maxtasksperchild", "1000",
        "--retries", "3",
        "--retry-delay", "1",
        "--capital-workers", "1",
        "--require-end-trade",
        "--price-source", "akshare",
        "--metadata-source", "akshare",
        "--missing-mktcap-policy", "exclude",
        "--market-cap-source", "auto",
    ])
    completion = CompletionManifest(formula33.FORMULA33_MANIFEST_FILE).read()
    observation_date = str(completion.get("observation_date") or "").strip()
    if completion.get("status") != "completed" or not observation_date:
        raise RuntimeError(
            "Formula33 refresh did not produce a completed observation-date manifest"
        )
    # The requested end can be a weekend/holiday.  All derived inputs must use
    # the market-confirmed observation date, not a guessed calendar date.
    args.end_date = pd.Timestamp(observation_date).strftime("%Y-%m-%d")
    print(
        "[portfolio_backtest][refresh] market-confirmed effective end "
        f"{args.end_date}",
        flush=True,
    )

    # Candidate dates come from the refreshed K-line calendar.  Build once to
    # establish that calendar, rebuild matching dated mainline snapshots, then
    # build candidates again so they consume the new mainline data.
    print("[portfolio_backtest][refresh] rebuild candidate calendar", flush=True)
    candidate_args = [
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output-directory", args.candidate_directory,
    ]
    rebuild_candidate_history.main(candidate_args)
    print(
        "[portfolio_backtest][refresh] rebuild missing dated mainline snapshots",
        flush=True,
    )
    rebuild_mainline_history.main([
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--candidate-directory", args.candidate_directory,
    ])
    print(
        "[portfolio_backtest][refresh] rebuild candidates with refreshed mainline",
        flush=True,
    )
    rebuild_candidate_history.main(candidate_args)
    print("[portfolio_backtest][refresh] rebuild Formula33 phase history", flush=True)
    rebuild_formula_history.main([
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output", args.formula_history,
    ])


def validate_backtest_input_coverage(snapshots, formula, requested_end):
    """Fail closed when derived inputs do not share the latest available session."""
    if not snapshots:
        raise RuntimeError("candidate snapshots are empty after input refresh")
    if formula.empty or "date" not in formula:
        raise RuntimeError("Formula33 phase history is empty after input refresh")
    candidate_end = max(pd.Timestamp(value).normalize() for value in snapshots)
    formula_dates = pd.to_datetime(formula["date"], errors="coerce").dropna()
    if formula_dates.empty:
        raise RuntimeError("Formula33 phase history contains no valid dates")
    formula_end = formula_dates.max().normalize()
    if formula_end != candidate_end:
        raise RuntimeError(
            "backtest input dates disagree: "
            f"candidate_end={candidate_end:%Y-%m-%d} "
            f"formula_end={formula_end:%Y-%m-%d}"
        )
    if candidate_end > pd.Timestamp(requested_end).normalize():
        raise RuntimeError("backtest input coverage unexpectedly exceeds requested end")
    return candidate_end.strftime("%Y-%m-%d")


def main(argv=None):
    parser = argparse.ArgumentParser(description="候选池、Formula33和三持仓组合回测")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-date", default="",
        help="default: latest session whose daily bar should be available",
    )
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="maximum symbols; 0 disables the symbol-count cap while exposure stays <=100%%",
    )
    parser.add_argument(
        "--codes",
        help="comma-separated explicit candidate universe; overrides snapshot members",
    )
    parser.add_argument(
        "--exit-tail-on-candidate-removal", action="store_true",
        help="exit right-side tails below 10%% at the next open after candidate removal",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("rolling", "fixed-first"),
        default="rolling",
        help="rolling逐日更新候选；fixed-first冻结首个交易日候选",
    )
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v3"),
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
    parser.add_argument(
        "--no-auto-price-structure", action="store_true",
        help="disable automatic structure anchors for a sensitivity comparison",
    )
    parser.add_argument(
        "--no-structure-pullback", action="store_true",
        help="allow structure breakout orders but disable structure support pullback orders",
    )
    parser.add_argument(
        "--close-confirmed-execution", choices=("close_proxy", "next_open"),
        default="close_proxy",
        help="14:55/close proxy follows the strategy; next_open is the conservative sensitivity case",
    )
    parser.add_argument("--commission-rate", type=float, default=0.000085)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--estimated-slippage-rate", type=float, default=0.0005)
    parser.add_argument(
        "--vectorbt-cross-check", action="store_true",
        help="replay emitted fills with vectorbt shared-cash accounting",
    )
    args = parser.parse_args(argv)
    if not args.end_date:
        args.end_date = default_data_end_date()
    requested_end_date = args.end_date
    if not args.no_refresh_inputs:
        refresh_backtest_inputs(args)
    else:
        print(
            "[portfolio_backtest][WARNING] input refresh explicitly disabled; "
            "results use frozen local data"
        )
    snapshots = load_candidate_snapshots(
        args.candidate_directory, args.start_date, args.end_date,
    )
    coverage_snapshots = dict(snapshots)
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
    trade_plans = load_trade_plans(args.trade_plans)
    price_frames = load_price_frames(
        codes,
        PATHS.cache / "formula33_kline" / "akshare",
        start_date=(pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=args.end_date,
    )
    formula = pd.read_csv(args.formula_history)
    input_coverage_end = validate_backtest_input_coverage(
        coverage_snapshots, formula, args.end_date,
    )
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
        trade_plans=trade_plans,
        max_positions=None if args.max_positions == 0 else args.max_positions,
        exit_tail_on_candidate_removal=args.exit_tail_on_candidate_removal,
        signals_effective_next_day=True,
        auto_price_structure=not args.no_auto_price_structure,
        allow_structure_pullback=not args.no_structure_pullback,
        close_confirmed_execution=args.close_confirmed_execution,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        initial_capital=args.initial_capital,
        sell_stamp_duty_rate=args.sell_stamp_duty_rate,
        estimated_slippage_rate=args.estimated_slippage_rate,
    )
    vectorbt_equity = []
    if args.vectorbt_cross_check:
        from stock_research.strategies.vectorbt_replay import run_vectorbt_cross_check

        cross_check = run_vectorbt_cross_check(
            price_frames,
            result,
            commission_rate=args.commission_rate,
            minimum_commission=args.minimum_commission,
            initial_capital=args.initial_capital,
            sell_stamp_duty_rate=args.sell_stamp_duty_rate,
            estimated_slippage_rate=args.estimated_slippage_rate,
        )
        vectorbt_equity = cross_check.pop("equity_curve")
        result["vectorbt_cross_check"] = cross_check
    database = Database(PATHS.database, code_version="portfolio-backtest-v3")
    database.initialize()
    research_repository = ResearchRepository(database)
    research_repository.persist_formula_history(formula, version="formula33-phase-v1")
    run_id = research_repository.persist_backtest_result(result)
    result["database_run_id"] = run_id
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{args.start_date}_{args.end_date}"
    pd.DataFrame(result["events"]).to_csv(
        output / f"{stem}_events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["trade_ledger"]).to_csv(
        output / f"{stem}_trades.csv", index=False, encoding="utf-8-sig",
    )
    build_readable_trade_frame(result["trade_ledger"]).to_csv(
        output / f"{stem}_买卖流水.csv", index=False, encoding="utf-8-sig",
    )
    (output / f"{stem}_买卖报告.md").write_text(
        render_trade_report_markdown(result), encoding="utf-8",
    )
    pd.DataFrame(result["equity_curve"]).to_csv(
        output / f"{stem}_equity.csv", index=False, encoding="utf-8-sig",
    )
    if args.vectorbt_cross_check:
        pd.DataFrame(vectorbt_equity).to_csv(
            output / f"{stem}_vectorbt_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {key: value for key, value in result.items() if key not in {"events", "equity_curve"}}
    summary["candidate_mode"] = args.candidate_mode
    if explicit_codes:
        summary["candidate_mode"] = "explicit_codes"
        summary["explicit_codes"] = explicit_codes
    summary["candidate_snapshot_dates"] = sorted(snapshots)
    summary["inputs_refreshed"] = not args.no_refresh_inputs
    summary["input_coverage_end"] = input_coverage_end
    summary["requested_end_date"] = requested_end_date
    summary["effective_end_date"] = args.end_date
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
