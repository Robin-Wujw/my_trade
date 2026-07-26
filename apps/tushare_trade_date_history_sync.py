"""Synchronize Tushare trade-date datasets into SQLite with coverage checks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_FIELDS = {
    "daily_kline": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "daily_basic": "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_FIELDS))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--fields", default="")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--min-existing-coverage", type=float, default=0.8)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _state_path(dataset: str, state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / f"tushare_{dataset}_trade_date_history_sync.json"


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


def _iso_date(value: str) -> str:
    converted = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(converted):
        raise ValueError(f"invalid date: {value}")
    return converted.strftime("%Y-%m-%d")


def _compact_date(value: str) -> str:
    return _iso_date(value).replace("-", "")


def _load_required_dates(
    database: Database,
    *,
    dataset: str,
    start_date: str,
    end_date: str,
    min_existing_coverage: float,
) -> list[tuple[str, int, int]]:
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    if dataset == "daily_kline":
        connection = database.connect(read_only=True)
        try:
            rows = connection.execute(
                """
                WITH calendar AS (
                    SELECT trade_date
                    FROM raw.tushare_dataset_rows
                    WHERE dataset = 'trade_cal'
                      AND trade_date >= ?
                      AND trade_date <= ?
                      AND json_extract(payload_json, '$.is_open') IN (1, '1')
                ),
                existing AS (
                    SELECT trade_date, COUNT(DISTINCT ts_code) AS existing_count
                    FROM raw.tushare_dataset_rows
                    WHERE dataset = 'daily_kline'
                      AND trade_date >= ?
                      AND trade_date <= ?
                    GROUP BY trade_date
                )
                SELECT
                    calendar.trade_date,
                    1 AS expected_count,
                    COALESCE(existing.existing_count, 0) AS existing_count
                FROM calendar
                LEFT JOIN existing ON existing.trade_date = calendar.trade_date
                WHERE COALESCE(existing.existing_count, 0) = 0
                ORDER BY calendar.trade_date
                """,
                [start, end, start, end],
            ).fetchall()
        finally:
            connection.close()
        return [(str(date), int(expected), int(existing)) for date, expected, existing in rows]

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
            existing AS (
                SELECT trade_date, COUNT(DISTINCT ts_code) AS existing_count
                FROM raw.tushare_dataset_rows
                WHERE dataset = ?
                  AND trade_date >= ?
                  AND trade_date <= ?
                GROUP BY trade_date
            )
            SELECT
                kline.trade_date,
                kline.kline_count,
                COALESCE(existing.existing_count, 0) AS existing_count
            FROM kline
            LEFT JOIN existing ON existing.trade_date = kline.trade_date
            WHERE COALESCE(existing.existing_count, 0) < MAX(50, CAST(kline.kline_count * ? AS INTEGER))
            ORDER BY kline.trade_date
            """,
            [start, end, dataset, start, end, float(min_existing_coverage)],
        ).fetchall()
    finally:
        connection.close()
    return [(str(date), int(expected), int(existing)) for date, expected, existing in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fields = args.fields or DEFAULT_FIELDS[args.dataset]
    database = Database(args.database, code_version=f"tushare-{args.dataset}-trade-date-history-sync-v1")
    database.initialize()
    repository = TushareRepository(database)

    state_path = _state_path(args.dataset, args.state_file)
    state = _load_state(state_path)
    state.update({
        "dataset": args.dataset,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fields": fields,
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
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        min_existing_coverage=args.min_existing_coverage,
    )
    state["required_date_count"] = len(required_dates)
    _save_state(state_path, state)

    request_count = 0
    rows_total = 0
    for index, (trade_date, expected_count, existing_count) in enumerate(required_dates, 1):
        compact = _compact_date(trade_date)
        if compact in done or compact in empty:
            continue
        if args.max_requests > 0 and request_count >= args.max_requests:
            break
        try:
            frame = tushare_api.query(args.dataset, fields=fields, trade_date=compact)
            rows = repository.upsert_dataset(
                args.dataset,
                frame,
                source=f"tushare/pro:{args.dataset}_trade_date_history",
                params={"trade_date": compact, "fields": fields},
            )
            request_count += 1
            rows_total += rows
            (empty if rows == 0 else done).add(compact)
            state["done"] = sorted(done)
            state["empty"] = sorted(empty)
            state["last"] = {
                "trade_date": compact,
                "index": index,
                "required_date_count": len(required_dates),
                "previous_existing_count": existing_count,
                "expected_count": expected_count,
                "rows": rows,
            }
            if request_count % 20 == 0 or index == len(required_dates):
                _save_state(state_path, state)
            if not args.quiet or request_count % 100 == 0:
                print(
                    f"[{args.dataset}] {index}/{len(required_dates)} {compact} "
                    f"previous={existing_count}/{expected_count} rows={rows}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - persisted for resume.
            state["errors"].append({"trade_date": compact, "error": str(exc)})
            _save_state(state_path, state)
            print(f"[{args.dataset}] {index}/{len(required_dates)} {compact} error={exc}", flush=True)
            if args.fail_fast:
                raise
        if args.sleep > 0:
            time.sleep(args.sleep)

    _save_state(state_path, state)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
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
