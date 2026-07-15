"""Run the point-in-time candidate portfolio backtest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from stock_research.core.financial_period import visible_report_periods
from stock_research.core.paths import PATHS
from stock_research.reporting.backtest_trade_report import (
    build_readable_trade_frame,
    render_trade_report_markdown,
)
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.storage import Database, KlineRepository, ResearchRepository
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


MIN_FINANCIAL_CACHE_FILES = 1_500


def _code_to_kline_cache_name(code):
    text = str(code).strip()
    if "." in text:
        market, symbol = text.split(".", 1)
    else:
        symbol = text.zfill(6)
        market = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{market}_{symbol}.csv"


def _latest_stock_basic_snapshot(snapshot_directory, end_date):
    target = pd.Timestamp(end_date).normalize()
    candidates = []
    for path in Path(snapshot_directory).glob("stock_basic_*.csv"):
        date = pd.to_datetime(path.stem.removeprefix("stock_basic_"), errors="coerce")
        if pd.notna(date) and date.normalize() <= target:
            candidates.append((date.normalize(), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _load_ipo_dates(stock_basic_path):
    if not stock_basic_path:
        return {}
    try:
        frame = pd.read_csv(stock_basic_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        return {}
    if not {"code", "ipoDate"}.issubset(frame.columns):
        return {}
    dates = pd.to_datetime(frame["ipoDate"], errors="coerce")
    return {
        str(code): date.normalize()
        for code, date in zip(frame["code"].astype(str), dates)
        if pd.notna(date)
    }


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


def summarize_kline_cache_coverage(
    kline_directory,
    universe_path,
    start_date,
    end_date,
    *,
    stock_basic_path=None,
):
    """Return a cheap full-universe K-line coverage summary for backtest inputs."""
    try:
        universe = pd.read_csv(universe_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        universe = pd.DataFrame()
    if universe.empty or "code" not in universe:
        return {
            "universe_count": 0,
            "missing_file_count": 0,
            "missing_start_count": 0,
            "missing_end_count": 0,
            "invalid_file_count": 0,
            "sample": [],
            "complete": False,
        }
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    summary = {
        "universe_count": int(universe["code"].dropna().nunique()),
        "missing_file_count": 0,
        "missing_start_count": 0,
        "post_ipo_start_count": 0,
        "missing_end_count": 0,
        "no_trade_end_count": 0,
        "invalid_file_count": 0,
        "sample": [],
        "complete": True,
    }
    directory = Path(kline_directory)
    ipo_dates = _load_ipo_dates(stock_basic_path)
    for code in universe["code"].dropna().astype(str).drop_duplicates():
        path = directory / _code_to_kline_cache_name(code)
        reason = ""
        if not path.is_file():
            summary["missing_file_count"] += 1
            reason = "missing_file"
        else:
            try:
                dates = pd.read_csv(path, usecols=["date"], low_memory=False)
                parsed = pd.to_datetime(dates["date"], errors="coerce").dropna()
            except (OSError, ValueError, EmptyDataError):
                parsed = pd.Series(dtype="datetime64[ns]")
            if parsed.empty:
                summary["invalid_file_count"] += 1
                reason = "invalid_file"
            else:
                min_date = parsed.min().normalize()
                max_date = parsed.max().normalize()
                if min_date > start:
                    ipo_date = ipo_dates.get(code)
                    if ipo_date is not None and ipo_date > start:
                        summary["post_ipo_start_count"] += 1
                    else:
                        summary["missing_start_count"] += 1
                        reason = f"starts_{min_date:%Y-%m-%d}"
                elif max_date < end:
                    marker = path.with_name(path.name + ".no-trade.json")
                    try:
                        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        marker_payload = {}
                    marker_date = pd.to_datetime(
                        marker_payload.get("observation_date"), errors="coerce",
                    )
                    if pd.notna(marker_date) and marker_date.normalize() == end:
                        summary["no_trade_end_count"] += 1
                    else:
                        summary["missing_end_count"] += 1
                        reason = f"ends_{max_date:%Y-%m-%d}"
        if reason:
            summary["complete"] = False
            if len(summary["sample"]) < 10:
                summary["sample"].append(f"{code}:{reason}")
    return summary


def invalidate_formula33_manifest_if_kline_cache_incomplete(
    *,
    manifest_path,
    kline_directory,
    universe_path,
    start_date,
    end_date,
):
    stock_basic_path = _latest_stock_basic_snapshot(
        Path(universe_path).parent / "formula33_snapshots",
        end_date,
    )
    coverage = summarize_kline_cache_coverage(
        kline_directory,
        universe_path,
        start_date,
        end_date,
        stock_basic_path=stock_basic_path,
    )
    if coverage["complete"]:
        print(
            "[portfolio_backtest][preflight] K-line cache covers requested "
            f"range for {coverage['universe_count']} symbols",
            flush=True,
        )
        return coverage
    path = Path(manifest_path)
    if path.exists():
        path.unlink()
    print(
        "[portfolio_backtest][preflight] K-line cache is incomplete; "
        "Formula33 manifest was invalidated so the refresh will auto-fetch "
        "missing bars. "
        f"missing_file={coverage['missing_file_count']} "
        f"missing_start={coverage['missing_start_count']} "
        f"missing_end={coverage['missing_end_count']} "
        f"invalid={coverage['invalid_file_count']} "
        f"sample={coverage['sample']}",
        flush=True,
    )
    return coverage


def repair_formula33_kline_metadata(source, universe_path):
    from stock_research.pipelines import formula33

    try:
        universe = pd.read_csv(universe_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        return {"checked": 0, "repaired": 0}
    checked = repaired = 0
    for code in universe.get("code", pd.Series(dtype=str)).dropna().astype(str).drop_duplicates():
        frame = formula33.load_cached_kline(source, code)
        if frame.empty:
            continue
        checked += 1
        if not formula33.kline_cache_metadata_matches(source, code, frame):
            formula33.save_kline_cache_metadata(source, code, frame)
            repaired += 1
    print(
        "[portfolio_backtest][preflight] Formula33 K-line metadata checked "
        f"{checked} symbols; repaired={repaired}",
        flush=True,
    )
    return {"checked": checked, "repaired": repaired}


def report_period_visible_date(report_period):
    period = pd.Timestamp(report_period).normalize()
    year = int(period.year)
    if period.month == 3 and period.day == 31:
        return pd.Timestamp(year=year, month=4, day=30)
    if period.month == 6 and period.day == 30:
        return pd.Timestamp(year=year, month=8, day=31)
    if period.month == 9 and period.day == 30:
        return pd.Timestamp(year=year, month=10, day=31)
    if period.month == 12 and period.day == 31:
        return pd.Timestamp(year=year + 1, month=4, day=30)
    return period


def financial_cache_file_count(directory, report_period):
    suffix = pd.Timestamp(report_period).strftime("%Y%m%d")
    return sum(1 for _path in Path(directory).glob(f"*_{suffix}.json"))


def ensure_financial_cache_for_backtest(args):
    from stock_research.pipelines import fundamental_update

    periods = visible_report_periods(args.start_date, args.end_date)
    if not periods:
        raise RuntimeError("no visible financial report periods found for backtest range")
    for period in periods:
        count = financial_cache_file_count(PATHS.cache / "q1_value", period)
        if count >= MIN_FINANCIAL_CACHE_FILES:
            print(
                "[portfolio_backtest][preflight] financial cache "
                f"{period} files={count}",
                flush=True,
            )
            continue
        visible_date = report_period_visible_date(period)
        as_of_date = max(pd.Timestamp(args.start_date), visible_date)
        if as_of_date > pd.Timestamp(args.end_date):
            as_of_date = pd.Timestamp(args.end_date)
        print(
            "[portfolio_backtest][preflight] financial cache missing or thin; "
            f"auto-fetch period={period} current_files={count} "
            f"as_of={as_of_date:%Y-%m-%d}",
            flush=True,
        )
        try:
            fundamental_update.main([
                "--report-period", period,
                "--as-of-date", as_of_date.strftime("%Y-%m-%d"),
                "--max-updates", "0",
                "--workers", "4",
                "--retries", "3",
            ])
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(
                    f"financial cache auto-fetch failed for {period} with exit {exc.code}"
                ) from exc
        refreshed_count = financial_cache_file_count(PATHS.cache / "q1_value", period)
        if refreshed_count < MIN_FINANCIAL_CACHE_FILES:
            raise RuntimeError(
                "financial cache remains incomplete after auto-fetch: "
                f"period={period} files={refreshed_count} "
                f"minimum={MIN_FINANCIAL_CACHE_FILES}"
            )
        print(
            "[portfolio_backtest][preflight] financial cache ready "
            f"{period} files={refreshed_count}",
            flush=True,
        )


def candidate_manifest_empty_dates(candidate_directory):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    rows = manifest.get("snapshots", [])
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("date"))
        for row in rows
        if int(row.get("candidate_count") or 0) <= 0
    ]


def default_data_end_date(now=None):
    """Return the latest session whose daily bar should already be available."""
    current = pd.Timestamp(now or pd.Timestamp.now())
    target = current.normalize()
    if current.weekday() >= 5:
        target = target - pd.offsets.BDay(1)
    elif current.hour < 16:
        target = target - pd.offsets.BDay(1)
    return target.strftime("%Y-%m-%d")


def formula33_refresh_window_args(start_date, end_date):
    """Size Formula33's calendar and history windows to cover a backtest range."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"invalid date range: {start_date}..{end_date}")
    calendar_days = max(0, (end - start).days)
    # Formula33 uses lookback for output rows and history-days for both raw
    # calendar depth and per-symbol K-line depth.  Keep both large enough for
    # the requested range while leaving the model's own indicator warmup intact.
    return {
        "lookback": max(21, calendar_days + 30),
        "history_days": 420,
    }


def refresh_backtest_inputs(args):
    """Refresh source data first, then rebuild every derived backtest artifact."""
    from apps import (
        rebuild_candidate_history,
        rebuild_formula_history,
        rebuild_mainline_history,
    )
    from stock_research.pipelines import formula33
    from stock_research.core.completion_manifest import CompletionManifest

    kline_coverage = invalidate_formula33_manifest_if_kline_cache_incomplete(
        manifest_path=formula33.FORMULA33_MANIFEST_FILE,
        kline_directory=PATHS.cache / "formula33_kline" / "akshare",
        universe_path=PATHS.cache / "stock_universe.csv",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if kline_coverage["complete"]:
        repair_formula33_kline_metadata("akshare", PATHS.cache / "stock_universe.csv")
    formula_window = formula33_refresh_window_args(args.start_date, args.end_date)
    print(
        "[portfolio_backtest][refresh] first refresh full-market K-lines "
        f"through {args.end_date} "
        f"lookback={formula_window['lookback']} "
        f"history_days={formula_window['history_days']}",
        flush=True,
    )
    formula33.main([
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--lookback", str(formula_window["lookback"]),
        "--history-days", str(formula_window["history_days"]),
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
    ensure_financial_cache_for_backtest(args)

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
    empty_candidate_dates = candidate_manifest_empty_dates(args.candidate_directory)
    if empty_candidate_dates:
        print(
            "[portfolio_backtest][refresh] candidate history has empty days; "
            f"rebuild mainline fallback count={len(empty_candidate_dates)} "
            f"sample={empty_candidate_dates[:10]}",
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
    else:
        print(
            "[portfolio_backtest][refresh] candidate history is non-empty "
            "for every trade day; skip slow mainline fallback rebuild",
            flush=True,
        )
    print("[portfolio_backtest][refresh] rebuild Formula33 phase history", flush=True)
    rebuild_formula_history.main([
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output", args.formula_history,
    ])


def validate_backtest_input_coverage(
    snapshots,
    formula,
    requested_start,
    requested_end,
):
    """Fail closed when every trading day lacks a fresh non-empty selection."""
    if not snapshots:
        raise RuntimeError("candidate snapshots are empty after input refresh")
    if formula.empty or "date" not in formula:
        raise RuntimeError("Formula33 phase history is empty after input refresh")
    formula_dates = pd.to_datetime(formula["date"], errors="coerce").dropna()
    if formula_dates.empty:
        raise RuntimeError("Formula33 phase history contains no valid dates")
    requested_start_date = pd.Timestamp(requested_start).normalize()
    requested_end_date = pd.Timestamp(requested_end).normalize()
    formula_trade_dates = {
        date.normalize()
        for date in formula_dates
        if requested_start_date <= date.normalize() <= requested_end_date
    }
    if not formula_trade_dates:
        raise RuntimeError("Formula33 phase history has no dates in requested backtest range")
    snapshot_trade_dates = {
        pd.Timestamp(value).normalize()
        for value in snapshots
        if requested_start_date <= pd.Timestamp(value).normalize() <= requested_end_date
    }
    missing_snapshot_dates = sorted(formula_trade_dates - snapshot_trade_dates)
    extra_snapshot_dates = sorted(snapshot_trade_dates - formula_trade_dates)
    if missing_snapshot_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in missing_snapshot_dates[:10])
        raise RuntimeError(
            "candidate snapshots do not cover every Formula33 trade date; "
            f"missing={preview}"
        )
    if extra_snapshot_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in extra_snapshot_dates[:10])
        raise RuntimeError(
            "candidate snapshots contain dates outside Formula33 trade calendar; "
            f"extra={preview}"
        )
    empty_candidate_dates = sorted(
        date for date in formula_trade_dates
        if not snapshots.get(date.strftime("%Y-%m-%d"))
    )
    if empty_candidate_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in empty_candidate_dates[:10])
        raise RuntimeError(
            "candidate snapshots contain empty selection days; "
            f"every backtest day must have a fresh non-empty selection result. empty={preview}"
        )
    candidate_end = max(snapshot_trade_dates)
    formula_end = max(formula_trade_dates)
    if formula_end != candidate_end:
        raise RuntimeError(
            "backtest input dates disagree: "
            f"candidate_end={candidate_end:%Y-%m-%d} "
            f"formula_end={formula_end:%Y-%m-%d}"
        )
    formula_start = min(formula_trade_dates)
    if min(snapshot_trade_dates) != formula_start:
        raise RuntimeError(
            "candidate snapshot coverage must start on the first Formula33 trade date: "
            f"candidate_start={min(snapshot_trade_dates):%Y-%m-%d} "
            f"formula_start={formula_start:%Y-%m-%d}"
        )
    if candidate_end > requested_end_date:
        raise RuntimeError("backtest input coverage unexpectedly exceeds requested end")
    return candidate_end.strftime("%Y-%m-%d")


def main(argv=None):
    parser = argparse.ArgumentParser(description="候选池、Formula33和四持仓组合回测")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-date", default="",
        help="default: latest session whose daily bar should be available",
    )
    parser.add_argument(
        "--max-positions", type=int, default=3,
        help="main capacity symbols; default 3, hard-capped at 5",
    )
    parser.add_argument(
        "--max-total-held-symbols", type=int, default=5,
        help="hard cap for all held symbols, including left cores and profit tails",
    )
    parser.add_argument(
        "--max-same-industry", type=int, default=2,
        help="maximum simultaneously held symbols sharing a dated industry/board tag",
    )
    parser.add_argument(
        "--same-theme-correlation", type=float, default=0.60,
        help="60-session return correlation used to group related exposure when tags are missing",
    )
    parser.add_argument(
        "--min-entry-evidence-score", type=float, default=8.0,
        help="minimum multi-signal technical evidence score for an executable entry",
    )
    parser.add_argument(
        "--profit-tail-min-return", type=float, default=0.50,
        help="minimum current return for the last tranche not to consume a slot",
    )
    parser.add_argument(
        "--profit-tranches", type=int, choices=(2, 3, 4, 5), default=5,
        help="number of profit-taking tranches; the last is reserved for maximum-profit-half",
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
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
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
        coverage_snapshots, formula, args.start_date, args.end_date,
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
        max_total_held_symbols=args.max_total_held_symbols,
        max_same_industry=args.max_same_industry,
        same_theme_correlation=args.same_theme_correlation,
        min_entry_evidence_score=args.min_entry_evidence_score,
        profit_tranches=args.profit_tranches,
        profit_tail_min_return=args.profit_tail_min_return,
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
