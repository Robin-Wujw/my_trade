"""Synchronize historical Tushare daily_basic rows by trading date."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_FIELDS = "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="19900101")
    parser.add_argument("--end-date", default="20211231")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--min-existing-coverage", type=float, default=0.8)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _state_path(state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / "tushare_daily_basic_history_sync.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {"done": [], "empty": [], "errors": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _normalized_date(value: str) -> str:
    return str(value).replace("-", "")


def _iso_date(value: str) -> str:
    converted = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(converted):
        raise ValueError(f"invalid date: {value}")
    return converted.strftime("%Y-%m-%d")


def _load_required_dates(
    database: Database,
    *,
    start_date: str,
    end_date: str,
    min_existing_coverage: float,
) -> list[tuple[str, int, int]]:
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            WITH kline AS (
                SELECT trade_date, COUNT(DISTINCT ts_code) AS kline_count
                FROM raw.tushare_dataset_rows
                WHERE dataset = 'daily_kline'
                  AND trade_date >= ?
                  AND trade_date <= ?
                GROUP BY trade_date
            ),
            basic AS (
                SELECT trade_date, COUNT(DISTINCT ts_code) AS basic_count
                FROM raw.tushare_dataset_rows
                WHERE dataset = 'daily_basic'
                  AND trade_date >= ?
                  AND trade_date <= ?
                GROUP BY trade_date
            )
            SELECT
                kline.trade_date,
                kline.kline_count,
                COALESCE(basic.basic_count, 0) AS basic_count
            FROM kline
            LEFT JOIN basic ON basic.trade_date = kline.trade_date
            WHERE COALESCE(basic.basic_count, 0) < MAX(50, CAST(kline.kline_count * ? AS INTEGER))
            ORDER BY kline.trade_date
            """,
            [start, end, start, end, float(min_existing_coverage)],
        ).fetchall()
    finally:
        connection.close()
    return [(str(date), int(kline_count), int(basic_count)) for date, kline_count, basic_count in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = Database(args.database, code_version="tushare-daily-basic-history-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    state_path = _state_path(args.state_file)
    state = _load_state(state_path)
    state.update({
        "dataset": "daily_basic",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fields": args.fields,
        "database": str(database.path),
        "min_existing_coverage": args.min_existing_coverage,
    })
    state.setdefault("done", [])
    state.setdefault("empty", [])
    state.setdefault("errors", [])
    done = set(state["done"])
    empty = set(state["empty"])

    required_dates = _load_required_dates(
        database,
        start_date=args.start_date,
        end_date=args.end_date,
        min_existing_coverage=args.min_existing_coverage,
    )
    state["required_date_count"] = len(required_dates)
    _save_state(state_path, state)

    request_count = 0
    rows_total = 0
    for index, (trade_date, kline_count, basic_count) in enumerate(required_dates, 1):
        compact_date = _normalized_date(trade_date)
        if compact_date in done or compact_date in empty:
            continue
        if args.max_requests > 0 and request_count >= args.max_requests:
            break
        try:
            frame = tushare_api.query(
                "daily_basic",
                fields=args.fields,
                trade_date=compact_date,
            )
            rows = repository.upsert_dataset(
                "daily_basic",
                frame,
                source="tushare/pro:daily_basic_history",
                params={"trade_date": compact_date, "fields": args.fields},
            )
            request_count += 1
            rows_total += rows
            (empty if rows == 0 else done).add(compact_date)
            state["done"] = sorted(done)
            state["empty"] = sorted(empty)
            state["last"] = {
                "trade_date": compact_date,
                "index": index,
                "required_date_count": len(required_dates),
                "previous_basic_count": basic_count,
                "kline_count": kline_count,
                "rows": rows,
            }
            if request_count % 20 == 0 or index == len(required_dates):
                _save_state(state_path, state)
            if not args.quiet or request_count % 100 == 0:
                print(
                    f"[daily_basic] {index}/{len(required_dates)} {compact_date} "
                    f"previous={basic_count}/{kline_count} rows={rows}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - persisted for resume.
            state["errors"].append({"trade_date": compact_date, "error": str(exc)})
            _save_state(state_path, state)
            print(f"[daily_basic] {index}/{len(required_dates)} {compact_date} error={exc}", flush=True)
            if args.fail_fast:
                raise
        if args.sleep > 0:
            time.sleep(args.sleep)

    _save_state(state_path, state)
    print(
        json.dumps(
            {
                "dataset": "daily_basic",
                "requests": request_count,
                "rows": rows_total,
                "done": len(done),
                "empty": len(empty),
                "errors": len(state.get("errors", [])),
                "required_date_count": len(required_dates),
                "state_file": str(state_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
