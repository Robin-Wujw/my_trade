"""Run portfolio backtests one natural year at a time with strict input refresh."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from apps import portfolio_backtest
from stock_research.core.paths import PATHS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐年运行组合回测；每个自然年单独补齐输入并单独输出收益。",
    )
    current_year = pd.Timestamp.now().year
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=current_year)
    parser.add_argument(
        "--through-date",
        default="",
        help="cap the final year at this date; default uses portfolio_backtest's latest available date",
    )
    parser.add_argument(
        "--output-root",
        default=str(PATHS.runtime_root / "backtests" / "portfolio_yearly"),
    )
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-total-held-symbols", type=int, default=5)
    parser.add_argument("--max-left-positions", type=int, default=2)
    parser.add_argument("--max-same-industry", type=int, default=2)
    parser.add_argument("--profit-tranches", type=int, choices=(2, 3, 4, 5), default=5)
    parser.add_argument("--profit-tail-min-return", type=float, default=0.50)
    parser.add_argument("--left-grid-unit", type=float, default=0.02)
    parser.add_argument("--left-grid-step", type=float, default=0.05)
    parser.add_argument("--left-grid-max-exposure", type=float, default=0.10)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument(
        "--close-confirmed-execution",
        choices=("close_proxy", "next_open"),
        default="close_proxy",
    )
    parser.add_argument("--disable-pullback-pilot", action="store_true")
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
    parser.add_argument("--price-kline-directory", default="")
    parser.add_argument(
        "--price-database-source",
        choices=("auto", "miniqmt", "akshare", "tushare"),
        default="auto",
    )
    parser.add_argument("--commission-rate", type=float, default=0.000085)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--estimated-slippage-rate", type=float, default=0.0005)
    return parser


def yearly_ranges(start_year: int, end_year: int, through_date: str = "") -> list[tuple[int, str, str]]:
    if int(start_year) > int(end_year):
        raise ValueError("start-year must be <= end-year")
    cap = pd.to_datetime(through_date, errors="coerce") if through_date else pd.NaT
    if pd.isna(cap):
        cap = pd.Timestamp(portfolio_backtest.default_data_end_date())
    cap = cap.normalize()
    ranges = []
    for year in range(int(start_year), int(end_year) + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = min(pd.Timestamp(year=year, month=12, day=31), cap)
        if end < start:
            continue
        ranges.append((year, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if not ranges:
        raise ValueError("no yearly ranges overlap the requested through-date")
    return ranges


def build_year_argv(args, year: int, start_date: str, end_date: str) -> tuple[list[str], Path]:
    output_root = Path(args.output_root)
    input_root = output_root / "inputs" / str(year)
    output_dir = output_root / str(year)
    candidate_dir = input_root / "candidate_snapshots"
    formula_history = input_root / "formula33_phase.csv"
    argv = [
        "--start-date", start_date,
        "--end-date", end_date,
        "--candidate-directory", str(candidate_dir),
        "--formula-history", str(formula_history),
        "--output-directory", str(output_dir),
        "--max-positions", str(args.max_positions),
        "--max-total-held-symbols", str(args.max_total_held_symbols),
        "--max-left-positions", str(args.max_left_positions),
        "--max-same-industry", str(args.max_same_industry),
        "--profit-tranches", str(args.profit_tranches),
        "--profit-tail-min-return", str(args.profit_tail_min_return),
        "--left-grid-unit", str(args.left_grid_unit),
        "--left-grid-step", str(args.left_grid_step),
        "--left-grid-max-exposure", str(args.left_grid_max_exposure),
        "--initial-capital", str(args.initial_capital),
        "--close-confirmed-execution", args.close_confirmed_execution,
        "--refresh-price-source", args.refresh_price_source,
        "--refresh-metadata-source", args.refresh_metadata_source,
        "--refresh-market-cap-source", args.refresh_market_cap_source,
        "--price-database-source", args.price_database_source,
        "--commission-rate", str(args.commission_rate),
        "--minimum-commission", str(args.minimum_commission),
        "--sell-stamp-duty-rate", str(args.sell_stamp_duty_rate),
        "--estimated-slippage-rate", str(args.estimated_slippage_rate),
    ]
    if args.price_kline_directory:
        argv.extend(["--price-kline-directory", args.price_kline_directory])
    if args.disable_pullback_pilot:
        argv.append("--disable-pullback-pilot")
    return argv, output_dir


def _latest_summary(output_dir: Path, start_date: str) -> Path:
    matches = sorted(
        output_dir.glob(f"portfolio_{start_date}_*_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RuntimeError(f"yearly backtest did not write a summary: {output_dir}")
    return matches[0]


def _summary_row(year: int, path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    r_summary = summary.get("r_multiple_summary") or {}
    concentration = summary.get("profit_concentration_summary") or {}
    top1 = concentration.get("top1_symbol") or {}
    trade_summary = summary.get("trade_summary") or {}
    return {
        "year": year,
        "actual_start": summary.get("actual_start"),
        "end_date": summary.get("end_date"),
        "final_return_pct": summary.get("final_return_pct"),
        "realized_return_pct": summary.get("realized_return_pct"),
        "unrealized_return_pct": summary.get("unrealized_return_pct"),
        "maximum_drawdown_pct": summary.get("maximum_drawdown_pct"),
        "buy_count": trade_summary.get("buy_count"),
        "sell_count": trade_summary.get("sell_count"),
        "r_average_realized": r_summary.get("average_realized_r"),
        "r_median_realized": r_summary.get("median_realized_r"),
        "r_profit_factor": r_summary.get("profit_factor_r"),
        "r_loss_beyond_one_count": r_summary.get("loss_beyond_one_r_count"),
        "top1_code": top1.get("code"),
        "top1_name": top1.get("name"),
        "top1_return_contribution_pct": concentration.get("top1_return_contribution_pct"),
        "inputs_refreshed": summary.get("inputs_refreshed"),
        "summary_path": str(path),
    }


def write_yearly_summary(rows: list[dict], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "yearly_summary.csv"
    columns = list(rows[0].keys()) if rows else ["year"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "yearly_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    output_root = Path(args.output_root)
    for year, start_date, end_date in yearly_ranges(args.start_year, args.end_year, args.through_date):
        year_argv, output_dir = build_year_argv(args, year, start_date, end_date)
        print(
            f"[portfolio_yearly_backtest] START year={year} "
            f"range={start_date}..{end_date}",
            flush=True,
        )
        portfolio_backtest.main(year_argv)
        summary_path = _latest_summary(output_dir, start_date)
        row = _summary_row(year, summary_path)
        rows.append(row)
        print(
            "[portfolio_yearly_backtest] DONE "
            f"year={year} return={row.get('final_return_pct')} "
            f"drawdown={row.get('maximum_drawdown_pct')} summary={summary_path}",
            flush=True,
        )
    summary_csv = write_yearly_summary(rows, output_root)
    print(f"[portfolio_yearly_backtest] yearly summary: {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
