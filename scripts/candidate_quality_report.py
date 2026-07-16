"""Diagnose forward behavior of dated candidate snapshots.

This is a research diagnostic only: forward returns are never used by the
selection model.  The report helps decide whether poor portfolio results come
from candidate quality or from entry/exit execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import load_candidate_snapshots
from stock_research.core.paths import PATHS
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots


def _cache_name(code: str) -> str:
    text = str(code)
    if "." in text:
        market, symbol = text.split(".", 1)
    else:
        symbol = text.zfill(6)
        market = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{market}_{symbol}.csv"


def load_close_series(codes: set[str], start_date: str, end_date: str) -> dict[str, pd.Series]:
    start = pd.Timestamp(start_date) - pd.Timedelta(days=30)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=60)
    result = {}
    for code in sorted(codes):
        path = PATHS.cache / "formula33_kline" / "akshare" / _cache_name(code)
        try:
            frame = pd.read_csv(path, usecols=["date", "close"])
        except (OSError, ValueError):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].dropna()
        if not frame.empty:
            result[code] = frame.drop_duplicates("date").set_index("date")["close"].sort_index()
    return result


def forward_return(series: pd.Series, date: str, days: int) -> float | None:
    current_date = pd.Timestamp(date)
    future = series[series.index > current_date]
    if len(future) < days:
        return None
    current = series[series.index <= current_date]
    if current.empty or current.iloc[-1] <= 0:
        return None
    return float(future.iloc[days - 1] / current.iloc[-1] - 1)


def bucket(value: float | None, edges: list[float], labels: list[str]) -> str:
    if value is None or pd.isna(value):
        return "missing"
    for edge, label in zip(edges, labels):
        if float(value) < edge:
            return label
    return labels[-1]


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    if frame.empty:
        return []
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: key for column, key in zip(group_columns, keys)}
        item["count"] = int(len(group))
        for horizon in (5, 10, 20):
            column = f"fwd_{horizon}d"
            valid = group[column].dropna()
            item[f"{column}_mean_pct"] = None if valid.empty else round(float(valid.mean()) * 100, 3)
            item[f"{column}_median_pct"] = None if valid.empty else round(float(valid.median()) * 100, 3)
            item[f"{column}_win_rate_pct"] = None if valid.empty else round(float((valid > 0).mean()) * 100, 3)
        rows.append(item)
    return sorted(rows, key=lambda item: (-item["count"], str(item)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
    )
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_quality"),
    )
    args = parser.parse_args(argv)

    raw = load_candidate_snapshots(args.candidate_directory, args.start_date, args.end_date)
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
    selected_rows = [
        row for date, rows in snapshots.items() for row in rows
        if row.get("selected_for_trading") and row.get("signal_eligible")
    ]
    codes = {str(row["code"]) for row in selected_rows}
    closes = load_close_series(codes, args.start_date, args.end_date)
    diagnostics = []
    for row in selected_rows:
        code = str(row["code"])
        series = closes.get(code)
        if series is None:
            continue
        item = {
            "date": row.get("date"),
            "code": code,
            "name": row.get("name"),
            "candidate_source": row.get("candidate_source") or "",
            "strategy_part": row.get("strategy_part") or "",
            "selection_rank": row.get("selection_rank"),
            "candidate_score": row.get("candidate_score"),
            "core_candidate_score": row.get("core_candidate_score"),
            "quality_score": row.get("quality_score"),
            "earnings_yoy": row.get("earnings_yoy"),
            "mktcap": row.get("mktcap"),
            "trade_basis_score": row.get("trade_basis_score"),
            "leadership_score": row.get("leadership_score"),
            "mainline_snapshot_fresh": row.get("mainline_snapshot_fresh"),
        }
        for horizon in (5, 10, 20):
            item[f"fwd_{horizon}d"] = forward_return(series, str(row.get("date")), horizon)
        item["rank_bucket"] = bucket(pd.to_numeric(item["selection_rank"], errors="coerce"), [4, 8, 11], ["1-3", "4-7", "8-10"])
        item["trade_basis_bucket"] = bucket(pd.to_numeric(item["trade_basis_score"], errors="coerce"), [4, 7, 10, 99], ["<4", "4-6", "7-9", "10+"])
        item["leadership_bucket"] = bucket(pd.to_numeric(item["leadership_score"], errors="coerce"), [15, 24, 99], ["<15", "15-23", "24+"])
        diagnostics.append(item)

    frame = pd.DataFrame(diagnostics)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "candidate_forward_returns.csv", index=False, encoding="utf-8-sig")
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "row_count": int(len(frame)),
        "overall": summarize(frame.assign(group="overall"), ["group"]),
        "by_source": summarize(frame, ["candidate_source"]),
        "by_rank_bucket": summarize(frame, ["rank_bucket"]),
        "by_trade_basis_bucket": summarize(frame, ["trade_basis_bucket"]),
        "by_leadership_bucket": summarize(frame, ["leadership_bucket"]),
    }
    (output / "candidate_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
