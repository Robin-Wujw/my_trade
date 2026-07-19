"""Batch-refresh MiniQMT historical K-line CSV caches for the stock universe."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.market.miniqmt_data import load_miniqmt_price_frames, normalize_project_code


def _chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _load_codes(path: Path, limit: int) -> list[str]:
    frame = pd.read_csv(path, dtype={"code": str})
    if "code" not in frame:
        raise RuntimeError(f"universe file has no code column: {path}")
    codes = [
        normalize_project_code(code)
        for code in frame["code"].dropna().astype(str).drop_duplicates()
        if str(code).strip()
    ]
    if limit > 0:
        codes = codes[:limit]
    return codes


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[miniqmt-kline][{timestamp}] {message}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch refresh MiniQMT K-line cache")
    parser.add_argument("--universe", default=str(PATHS.cache / "stock_universe.csv"))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--period", default="1d")
    parser.add_argument("--dividend-type", default="front")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    codes = _load_codes(Path(args.universe), args.limit)
    if not codes:
        raise RuntimeError("no universe codes to fetch")
    chunk_size = max(1, int(args.chunk_size))
    totals = {
        "requested_count": len(codes),
        "loaded_count": 0,
        "missing_count": 0,
        "error_count": 0,
    }
    _log(
        "START "
        f"codes={len(codes)} dividend={args.dividend_type} "
        f"range={args.start_date}..{args.end_date} chunk={chunk_size}"
    )
    for batch_index, chunk in enumerate(_chunks(codes, chunk_size), start=1):
        _, summary = load_miniqmt_price_frames(
            chunk,
            start_date=args.start_date,
            end_date=args.end_date,
            period=args.period,
            dividend_type=args.dividend_type,
            refresh=True,
            persist=True,
        )
        fetch = summary.get("fetch") or {}
        errors = fetch.get("errors") or []
        totals["loaded_count"] += int(summary.get("loaded_count") or 0)
        totals["missing_count"] += int(summary.get("missing_count") or 0)
        totals["error_count"] += len(errors)
        _log(
            f"batch={batch_index} requested={summary.get('requested_count')} "
            f"loaded={summary.get('loaded_count')} missing={summary.get('missing_count')} "
            f"errors={len(errors)}"
        )
        if errors:
            _log("error_sample=" + json.dumps(errors[:2], ensure_ascii=False))
    _log("DONE " + json.dumps(totals, ensure_ascii=False))
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
