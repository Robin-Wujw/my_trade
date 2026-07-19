"""Rebuild strict candidate history in month-sized chunks, then merge outputs."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.strategies.historical_candidates import (
    build_historical_candidate_snapshots,
    save_historical_candidate_snapshots,
)


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[candidate-monthly][{timestamp}] {message}", flush=True)


def _month_ranges(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    current = start
    while current <= end:
        month_end = min(current + pd.offsets.MonthEnd(0), end)
        yield current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        current = (month_end + pd.Timedelta(days=1)).normalize()


def _read_snapshots(directory: Path) -> dict[str, list[dict]]:
    snapshots: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("candidates_*.csv")):
        date = path.stem.removeprefix("candidates_")
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        snapshots[date] = frame.to_dict("records")
    return snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild candidate history by month")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--work-directory", default="")
    parser.add_argument("--price-source", choices=("akshare", "miniqmt"), default="miniqmt")
    parser.add_argument("--kline-directory", default="")
    parser.add_argument("--raw-kline-directory", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_directory)
    work = Path(args.work_directory) if args.work_directory else output.with_name(output.name + "_monthly")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    kline_directory = args.kline_directory or (
        PATHS.cache / "miniqmt_kline" / "1d" / "front"
        if args.price_source == "miniqmt"
        else PATHS.cache / "formula33_kline" / "akshare"
    )
    raw_kline_directory = args.raw_kline_directory or (
        PATHS.cache / "miniqmt_kline" / "1d" / "none"
        if args.price_source == "miniqmt"
        else PATHS.cache / "formula33_kline" / "akshare_raw"
    )

    os.environ.setdefault("CANDIDATE_HISTORY_PROGRESS", "1")
    all_snapshots: dict[str, list[dict]] = {}
    ranges = list(_month_ranges(args.start_date, args.end_date))
    _log(f"START months={len(ranges)} range={args.start_date}..{args.end_date}")
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        chunk_dir = work / f"{chunk_start}_{chunk_end}"
        manifest_path = chunk_dir / "manifest.json"
        if args.resume and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            if manifest.get("strict_financial_point_in_time") is True:
                _log(f"SKIP {index}/{len(ranges)} {chunk_start}..{chunk_end}")
                all_snapshots.update(_read_snapshots(chunk_dir))
                continue
        _log(f"BUILD {index}/{len(ranges)} {chunk_start}..{chunk_end}")
        snapshots = build_historical_candidate_snapshots(
            chunk_start,
            chunk_end,
            value_cache_directory=PATHS.cache / "q1_value",
            kline_directory=kline_directory,
            raw_kline_directory=raw_kline_directory,
            universe_path=PATHS.cache / "stock_universe.csv",
            mainline_directory=PATHS.cache,
            research_repository=None,
            price_source=args.price_source,
            strict_financial_point_in_time=True,
        )
        manifest = save_historical_candidate_snapshots(
            chunk_dir,
            snapshots,
            start_date=chunk_start,
            end_date=chunk_end,
        )
        _log(
            f"DONE {index}/{len(ranges)} snapshots={manifest['snapshot_count']} "
            f"strict={manifest['strict_financial_point_in_time']}"
        )
        all_snapshots.update(snapshots)

    final_manifest = save_historical_candidate_snapshots(
        output,
        all_snapshots,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    _log(
        f"MERGED snapshots={final_manifest['snapshot_count']} "
        f"strict={final_manifest['strict_financial_point_in_time']} "
        f"unsafe={final_manifest['unsafe_snapshot_count']}"
    )
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
