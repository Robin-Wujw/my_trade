"""Rebuild dated mainline rankings with resumable historical snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.pipelines import sector_watch


def _dates(candidate_directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    result = []
    for path in Path(candidate_directory).glob("candidates_*.csv"):
        date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
        if pd.notna(date) and start <= date <= end:
            result.append(date.normalize())
    return sorted(set(result))


def _snapshot_is_valid(date):
    stamp = date.strftime("%Y%m%d")
    path = PATHS.cache / f"sector_mainline_constituents_{stamp}.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return False
    return not frame.empty and {"code", "board", "board_date"}.issubset(frame.columns)


def main(argv=None):
    parser = argparse.ArgumentParser(description="断点续跑逐交易日主流板块历史")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-10")
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "mainline-left-manual-v2"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    dates = _dates(args.candidate_directory, args.start_date, args.end_date)
    completed = 0
    skipped = 0
    failed = []
    for index, date in enumerate(dates, 1):
        if not args.force and _snapshot_is_valid(date):
            skipped += 1
            print(f"[{index}/{len(dates)}] {date:%Y-%m-%d} 已有快照，跳过")
            continue
        print(f"[{index}/{len(dates)}] 回建 {date:%Y-%m-%d}")
        try:
            sector_watch.main([
                "--as-of-date", date.strftime("%Y-%m-%d"),
                "--days", "80", "--top", "30", "--workers", "8",
                "--sleep", "0.01", "--retries", "3", "--retry-delay", "1",
                "--allow-missing-limit-up",
            ])
            completed += 1
        except (Exception, SystemExit) as exc:
            failed.append({"date": date.strftime("%Y-%m-%d"), "error": str(exc)})
            print(f"{date:%Y-%m-%d} 回建失败: {exc}")
    print(f"主流历史完成={completed} 跳过={skipped} 失败={len(failed)}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
