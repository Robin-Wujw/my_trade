"""Run the portfolio backtest with the MiniQMT execution profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from apps.portfolio_backtest import (
    load_candidate_snapshots,
    load_price_frames,
    validate_backtest_input_coverage,
)
from stock_research.core.paths import PATHS
from stock_research.market.miniqmt_data import load_miniqmt_price_frames
from stock_research.reporting.backtest_trade_report import (
    build_readable_trade_frame,
    render_trade_report_markdown,
)
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.miniqmt_backtest import (
    DEFAULT_MINIQMT_BACKTEST_PROFILE,
    MiniQmtBacktestProfile,
    run_miniqmt_backtest,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


def build_parser():
    parser = argparse.ArgumentParser(description="MiniQMT profile portfolio backtest")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--candidate-directory", default=str(
        PATHS.runtime_root / "backtests" / "candidate_snapshots" / SNAPSHOT_VERSION
    ))
    parser.add_argument("--formula-history", default=str(
        PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"
    ))
    parser.add_argument("--trade-plans", default=str(PATHS.project_root / "config" / "trade_plans.json"))
    parser.add_argument("--output-directory", default=str(PATHS.runtime_root / "backtests" / "miniqmt"))
    parser.add_argument("--price-source", choices=("akshare", "miniqmt"), default="miniqmt")
    parser.add_argument("--bar-period", default="1d")
    parser.add_argument("--miniqmt-dividend-type", default="front")
    parser.add_argument("--miniqmt-refresh", action="store_true")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-total-held-symbols", type=int, default=5)
    parser.add_argument("--profit-tranches", type=int, choices=(2, 3, 4, 5), default=5)
    parser.add_argument(
        "--min-entry-evidence-score",
        type=float,
        default=0.0,
        help="Deprecated explanation-only score floor; semantic high R/R gate controls entries.",
    )
    parser.add_argument("--max-symbol-exposure", type=float, default=0.70)
    parser.add_argument("--left-grid-unit", type=float, default=0.0)
    parser.add_argument("--left-grid-max-exposure", type=float, default=0.0)
    parser.add_argument("--disable-pullback-pilot", action="store_true")
    parser.add_argument("--allow-unsafe-financial", action="store_true")
    parser.add_argument("--allow-unsafe-industry", action="store_true")
    parser.add_argument("--commission-rate", type=float, default=DEFAULT_MINIQMT_BACKTEST_PROFILE.commission_rate)
    parser.add_argument("--minimum-commission", type=float, default=DEFAULT_MINIQMT_BACKTEST_PROFILE.minimum_commission)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=DEFAULT_MINIQMT_BACKTEST_PROFILE.sell_stamp_duty_rate)
    parser.add_argument("--estimated-slippage-rate", type=float, default=DEFAULT_MINIQMT_BACKTEST_PROFILE.estimated_slippage_rate)
    return parser


def miniqmt_lookback_summary(price_frames, snapshots, *, requested_start, min_prior_bars=60):
    first_candidate_dates = {}
    for date, rows in snapshots.items():
        candidate_date = pd.Timestamp(date).normalize()
        for row in rows:
            code = str(row.get("code") or "")
            if not code:
                continue
            first_candidate_dates[code] = min(
                first_candidate_dates.get(code, candidate_date),
                candidate_date,
            )
    insufficient = []
    new_listing_limited = []
    requested_start_ts = pd.Timestamp(requested_start).normalize()
    for code, first_date in sorted(first_candidate_dates.items()):
        frame = price_frames.get(code)
        if frame is None or frame.empty:
            insufficient.append({
                "code": code,
                "first_candidate_date": first_date.strftime("%Y-%m-%d"),
                "prior_bars": 0,
                "first_price_date": None,
            })
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
        prior_bars = int((dates < first_date).sum())
        first_price_date = dates.min()
        item = {
            "code": code,
            "first_candidate_date": first_date.strftime("%Y-%m-%d"),
            "prior_bars": prior_bars,
            "first_price_date": first_price_date.strftime("%Y-%m-%d"),
        }
        if prior_bars < int(min_prior_bars):
            new_listing_cutoff = first_date - pd.Timedelta(days=int(min_prior_bars) * 2)
            if first_price_date > requested_start_ts or first_price_date > new_listing_cutoff:
                new_listing_limited.append(item)
            else:
                insufficient.append(item)
    return {
        "min_prior_bars": int(min_prior_bars),
        "insufficient_count": len(insufficient),
        "insufficient_sample": insufficient[:10],
        "new_listing_limited_count": len(new_listing_limited),
        "new_listing_limited_sample": new_listing_limited[:10],
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    snapshots = load_candidate_snapshots(args.candidate_directory, args.start_date, args.end_date)
    codes = {str(row["code"]) for rows in snapshots.values() for row in rows}
    price_start_date = (pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d")
    if args.price_source == "miniqmt":
        price_frames, price_source_summary = load_miniqmt_price_frames(
            codes,
            start_date=price_start_date,
            end_date=args.end_date,
            period=args.bar_period,
            dividend_type=args.miniqmt_dividend_type,
            refresh=args.miniqmt_refresh,
            persist=args.bar_period == "1d",
        )
        if price_source_summary["missing_count"]:
            raise RuntimeError(
                "MiniQMT price cache is incomplete; rerun with --miniqmt-refresh. "
                f"missing={price_source_summary['missing_sample']}"
            )
        lookback_summary = miniqmt_lookback_summary(
            price_frames,
            snapshots,
            requested_start=args.start_date,
        )
        if lookback_summary["insufficient_count"]:
            raise RuntimeError(
                "MiniQMT price cache lacks pre-candidate lookback; rerun with --miniqmt-refresh. "
                f"insufficient={lookback_summary['insufficient_sample']}"
            )
        price_source_summary["lookback"] = lookback_summary
    else:
        price_frames = load_price_frames(
            codes,
            PATHS.cache / "formula33_kline" / "akshare",
            start_date=price_start_date,
            end_date=args.end_date,
        )
        price_source_summary = {
            "source": "akshare",
            "requested_count": len(codes),
            "loaded_count": len(price_frames),
            "missing_count": len(set(codes) - set(price_frames)),
        }
    formula = pd.read_csv(args.formula_history)
    input_coverage_end = validate_backtest_input_coverage(
        snapshots,
        formula,
        args.start_date,
        args.end_date,
        candidate_directory=args.candidate_directory,
        allow_unsafe_financial=args.allow_unsafe_financial,
        allow_unsafe_industry=args.allow_unsafe_industry,
    )
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    profile = MiniQmtBacktestProfile(
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        sell_stamp_duty_rate=args.sell_stamp_duty_rate,
        estimated_slippage_rate=args.estimated_slippage_rate,
    )
    result = run_miniqmt_backtest(
        price_frames,
        snapshots,
        phases,
        profile=profile,
        requested_start=args.start_date,
        end_date=args.end_date,
        trade_plans=load_trade_plans(args.trade_plans),
        max_positions=args.max_positions,
        max_total_held_symbols=args.max_total_held_symbols,
        max_symbol_exposure=args.max_symbol_exposure,
        min_entry_evidence_score=args.min_entry_evidence_score,
        profit_tranches=args.profit_tranches,
        left_grid_unit=args.left_grid_unit,
        left_grid_max_exposure=args.left_grid_max_exposure,
        allow_pullback_pilot=not args.disable_pullback_pilot,
        initial_capital=args.initial_capital,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"miniqmt_{args.start_date}_{args.end_date}"
    pd.DataFrame(result["events"]).to_csv(output / f"{stem}_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["trade_ledger"]).to_csv(output / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
    build_readable_trade_frame(result["trade_ledger"]).to_csv(
        output / f"{stem}_trade_flow.csv", index=False, encoding="utf-8-sig",
    )
    (output / f"{stem}_trade_report.md").write_text(
        render_trade_report_markdown(result), encoding="utf-8",
    )
    pd.DataFrame(result["equity_curve"]).to_csv(output / f"{stem}_equity.csv", index=False, encoding="utf-8-sig")
    summary = {key: value for key, value in result.items() if key not in {"events", "equity_curve"}}
    summary["candidate_snapshot_dates"] = sorted(snapshots)
    summary["input_coverage_end"] = input_coverage_end
    summary["read_only_connector"] = True
    summary["price_source"] = price_source_summary
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
