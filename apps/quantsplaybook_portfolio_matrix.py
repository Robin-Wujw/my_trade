"""Run factor/period portfolio replays concurrently with resumable validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from apps.quantsplaybook_compare import (
    _sha256,
    required_history_start_date,
)
from apps.quantsplaybook_results import write_results
from stock_research.core.paths import PATHS


DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)


def _periods(start_year: int, end_year: int, final_end_date: str):
    periods = []
    for year in range(int(start_year), int(end_year) + 1):
        end = (
            final_end_date
            if year == pd.Timestamp(final_end_date).year
            else f"{year}-12-31"
        )
        periods.append((str(year), f"{year}-01-01", end))
    periods.append((
        f"{start_year}_to_date",
        f"{start_year}-01-01",
        final_end_date,
    ))
    return periods


def _summary_path(root: Path, factor: str, start: str, end: str) -> Path:
    return (
        root / "portfolio" / factor
        / f"portfolio_{start}_{end}_summary.json"
    )


def _summary_reusable(
    root: Path,
    factor: str,
    start: str,
    end: str,
) -> bool:
    path = _summary_path(root, factor, start, end)
    manifest = root / "candidates" / factor / "manifest.json"
    if not path.is_file() or not manifest.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    audit = summary.get("trade_audit_summary") or {}
    fingerprints = summary.get("input_fingerprints") or {}
    return bool(
        summary.get("coverage_complete") is True
        and audit.get("violation_count") == 0
        and fingerprints.get("candidate_manifest_sha256") == _sha256(manifest)
        and str(summary.get("requested_start")) == start
        and str(summary.get("end_date")) == end
    )


def _run_task(
    *,
    root: Path,
    prepared_cache: Path,
    formula_history: Path,
    factor: str,
    label: str,
    start: str,
    end: str,
) -> tuple[str, str, str]:
    if _summary_reusable(root, factor, start, end):
        return factor, label, "reused"
    log_root = root / "portfolio_jobs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{label}_{factor}.log"
    metrics_file = Path("portfolio_jobs") / f"{label}_{factor}_metrics.csv"
    command = [
        sys.executable,
        "-m",
        "apps.quantsplaybook_compare",
        "--start-date",
        start,
        "--end-date",
        end,
        "--history-start-date",
        required_history_start_date(start).strftime("%Y-%m-%d"),
        "--formula-history",
        str(formula_history),
        "--output-root",
        str(root),
        "--reuse-candidates",
        "--prepared-price-cache",
        str(prepared_cache),
        "--run-portfolio",
        "--portfolio-factors",
        factor,
        "--portfolio-metrics-file",
        str(metrics_file),
        "--skip-portfolio-aggregate",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"portfolio task failed factor={factor} period={label}\n{tail}",
        )
    if not _summary_reusable(root, factor, start, end):
        raise RuntimeError(
            f"portfolio task produced no reusable summary: {factor} {label}",
        )
    return factor, label, "completed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all factor portfolios for full and independent yearly periods",
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--prepared-price-cache",
        default="",
    )
    parser.add_argument(
        "--formula-history", default=str(DEFAULT_FORMULA_HISTORY),
    )
    parser.add_argument(
        "--factors",
        default="",
        help="optional comma-separated subset",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root)
    prepared = (
        Path(args.prepared_price_cache)
        if args.prepared_price_cache
        else root / "prepared_prices"
    )
    requested = [
        value.strip() for value in args.factors.split(",") if value.strip()
    ]
    factors = requested or sorted(
        path.name
        for path in (root / "candidates").iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    periods = _periods(args.start_year, args.end_year, args.end_date)
    tasks = [
        {
            "root": root,
            "prepared_cache": prepared,
            "formula_history": Path(args.formula_history),
            "factor": factor,
            "label": label,
            "start": start,
            "end": end,
        }
        for label, start, end in periods
        for factor in factors
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        futures = [pool.submit(_run_task, **task) for task in tasks]
        for future in as_completed(futures):
            factor, label, status = future.result()
            completed += 1
            print(
                f"[portfolio-matrix] {completed}/{len(tasks)} "
                f"{label} {factor} {status}",
                flush=True,
            )
    comparison = write_results(
        root,
        start_year=args.start_year,
        end_year=args.end_year,
        end_date=args.end_date,
    )
    print(
        f"[portfolio-matrix] complete strategies={len(comparison)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
