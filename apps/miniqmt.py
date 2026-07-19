"""MiniQMT read-only diagnostics and account queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from apps.portfolio_backtest import load_price_frames
from stock_research.api.miniqmt import (
    MiniQmtClient,
    check_sdk,
    detect_running_processes,
    load_miniqmt_config,
    probe_data_capabilities_via_qmt_python,
    query_financial_data_via_qmt_python,
    query_accounts_via_qmt_python,
)
from stock_research.core.paths import PATHS
from stock_research.market.miniqmt_data import (
    compare_price_frames,
    fetch_miniqmt_bars_via_qmt_python,
    load_miniqmt_price_frames,
    normalize_project_code,
)
from stock_research.market.miniqmt_financial import (
    build_miniqmt_financial_cache,
    default_financial_as_of_date,
    load_universe_codes,
    point_in_time_financial_cache_coverage,
)


def build_parser():
    parser = argparse.ArgumentParser(description="MiniQMT read-only integration tools")
    parser.add_argument("--config", help="JSON config path; default var/secrets/miniqmt.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check local SDK paths, imports, and running processes")
    subparsers.add_parser("probe-data", help="inspect read-only xtdata market/fundamental data APIs")

    query = subparsers.add_parser("query", help="connect and query assets/positions")
    query.add_argument("--account", action="append", help="account id; can be repeated")
    query.add_argument("--positions-sample-size", type=int, default=5)

    fetch = subparsers.add_parser("fetch-bars", help="download historical bars into MiniQMT cache")
    fetch.add_argument("--codes", required=True, help="comma-separated stock codes")
    fetch.add_argument("--start-date", required=True)
    fetch.add_argument("--end-date", required=True)
    fetch.add_argument("--period", default="1d")
    fetch.add_argument("--dividend-type", default="front")

    financial = subparsers.add_parser("fetch-financial", help="fetch a read-only MiniQMT financial data sample")
    financial.add_argument("--codes", required=True, help="comma-separated stock codes")
    financial.add_argument("--tables", default="Balance,Income,Capital,PershareIndex,PerShare")
    financial.add_argument("--start-time", required=True, help="YYYYMMDD")
    financial.add_argument("--end-time", required=True, help="YYYYMMDD")
    financial.add_argument("--report-type", default="announce_time", choices=("announce_time", "report_time"))
    financial.add_argument("--row-limit", type=int, default=5)

    financial_cache = subparsers.add_parser(
        "build-financial-cache",
        help="persist MiniQMT financial rows into q1_value value-line cache",
    )
    financial_cache.add_argument("--codes", default="", help="optional comma-separated stock codes; default universe")
    financial_cache.add_argument("--universe", default=str(PATHS.cache / "stock_universe.csv"))
    financial_cache.add_argument("--report-period", default="", help="YYYY-MM-DD")
    financial_cache.add_argument(
        "--periods",
        default="",
        help="comma-separated report periods; overrides --report-period",
    )
    financial_cache.add_argument(
        "--as-of-date",
        default="",
        help="announcement cutoff date; default statutory visibility deadline",
    )
    financial_cache.add_argument("--output-directory", default=str(PATHS.cache / "q1_value"))
    financial_cache.add_argument("--chunk-size", type=int, default=20)
    financial_cache.add_argument("--timeout", type=int, default=180)
    financial_cache.add_argument("--max-codes", type=int, default=0)
    financial_cache.add_argument("--overwrite", action="store_true")
    financial_cache.add_argument(
        "--missing-point-in-time-only",
        action="store_true",
        help="only fetch files missing strict announce_time metadata or core metrics",
    )
    financial_cache.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="fail if strict point-in-time cache coverage is below this ratio",
    )

    compare = subparsers.add_parser("compare-bars", help="compare MiniQMT cached bars against current backtest bars")
    compare.add_argument("--codes", required=True, help="comma-separated stock codes")
    compare.add_argument("--start-date", required=True)
    compare.add_argument("--end-date", required=True)
    compare.add_argument("--period", default="1d")
    compare.add_argument("--dividend-type", default="front")
    compare.add_argument("--refresh", action="store_true", help="download MiniQMT bars before comparing")
    compare.add_argument("--output", help="optional JSON output path")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_miniqmt_config(args.config)
    if args.command == "doctor":
        payload = {
            "sdk": check_sdk(config),
            "processes": detect_running_processes(),
            "configured_accounts": len(config.accounts),
            "read_only": True,
            "live_trading_enabled": False,
        }
    elif args.command == "probe-data":
        payload = probe_data_capabilities_via_qmt_python(config)
    elif args.command == "query":
        accounts = tuple(args.account or config.accounts)
        sdk_status = check_sdk(config)
        if sdk_status["xtquant_importable"]:
            with MiniQmtClient(config) as client:
                payload = client.query_accounts(
                    accounts,
                    positions_sample_size=max(0, args.positions_sample_size),
                )
        else:
            payload = query_accounts_via_qmt_python(
                config,
                accounts,
                positions_sample_size=max(0, args.positions_sample_size),
            )
            payload["bridge_reason"] = sdk_status["error"]
        payload["read_only"] = True
        payload["live_trading_enabled"] = False
    elif args.command == "fetch-bars":
        codes = [normalize_project_code(item) for item in args.codes.split(",") if item.strip()]
        payload = fetch_miniqmt_bars_via_qmt_python(
            codes,
            args.start_date,
            args.end_date,
            period=args.period,
            dividend_type=args.dividend_type,
            config=config,
        )
    elif args.command == "fetch-financial":
        codes = [normalize_project_code(item) for item in args.codes.split(",") if item.strip()]
        tables = [item.strip() for item in args.tables.split(",") if item.strip()]
        payload = query_financial_data_via_qmt_python(
            codes,
            tables,
            start_time=args.start_time,
            end_time=args.end_time,
            report_type=args.report_type,
            config=config,
            row_limit=args.row_limit,
        )
    elif args.command == "build-financial-cache":
        if args.codes:
            codes = [normalize_project_code(item) for item in args.codes.split(",") if item.strip()]
        else:
            codes = load_universe_codes(args.universe)
        if args.max_codes and args.max_codes > 0:
            codes = codes[:args.max_codes]
        periods = [
            item.strip() for item in (args.periods or "").split(",") if item.strip()
        ] or ([args.report_period] if args.report_period else [])
        if not periods:
            raise SystemExit("--report-period or --periods is required")
        results = []
        failed_periods = []
        for period in periods:
            as_of_date = args.as_of_date or default_financial_as_of_date(period)
            result = build_miniqmt_financial_cache(
                codes,
                report_period=period,
                as_of_date=as_of_date,
                output_directory=args.output_directory,
                chunk_size=args.chunk_size,
                config=config,
                overwrite=args.overwrite,
                missing_point_in_time_only=args.missing_point_in_time_only,
                timeout=args.timeout,
            )
            coverage = point_in_time_financial_cache_coverage(
                codes,
                report_period=period,
                as_of_date=as_of_date,
                output_directory=args.output_directory,
            )
            result["point_in_time_coverage"] = coverage
            results.append(result)
            if args.min_coverage and coverage["coverage"] < args.min_coverage:
                failed_periods.append({
                    "report_period": period,
                    "coverage": coverage["coverage"],
                    "required": args.min_coverage,
                })
        payload = {
            "results": results,
            "failed_periods": failed_periods,
            "period_count": len(periods),
        }
        if failed_periods:
            payload["ok"] = False
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        payload["ok"] = True
    elif args.command == "compare-bars":
        codes = [normalize_project_code(item) for item in args.codes.split(",") if item.strip()]
        start_date = (pd.Timestamp(args.start_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        miniqmt_frames, miniqmt_summary = load_miniqmt_price_frames(
            codes,
            start_date=start_date,
            end_date=args.end_date,
            period=args.period,
            dividend_type=args.dividend_type,
            refresh=args.refresh,
            persist=False,
        )
        baseline_frames = load_price_frames(
            codes,
            PATHS.cache / "formula33_kline" / "akshare",
            start_date=start_date,
            end_date=args.end_date,
        )
        payload = {
            "baseline_source": "akshare",
            "comparison_source": "miniqmt",
            "miniqmt": miniqmt_summary,
            "comparison": compare_price_frames(baseline_frames, miniqmt_frames),
        }
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
    else:  # pragma: no cover - argparse guards this branch.
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
