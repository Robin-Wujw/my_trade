"""Synchronize Tushare dividend implementation rows by ex-right date."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DIVIDEND_FIELDS = (
    "ts_code,ann_date,end_date,record_date,ex_date,div_listdate,div_proc,"
    "stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--state-file", default="")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _state_path(state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / "tushare_dividend_ex_date_sync.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"done": [], "errors": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _prune_resolved_errors(state: dict) -> None:
    done = set(str(item) for item in state.get("done", []))
    state["errors"] = [
        item for item in state.get("errors", [])
        if str(item.get("ex_date")) not in done
    ]


def _trade_dates(database: Database, start_date: str, end_date: str) -> list[str]:
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT trade_date
            FROM raw.tushare_dataset_rows
            WHERE dataset = 'daily_kline'
              AND trade_date >= ?
              AND (? = '' OR trade_date <= ?)
            ORDER BY trade_date
            """,
            [start_date, end_date, end_date],
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]).replace("-", "") for row in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = Database(args.database, code_version="tushare-dividend-ex-date-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    state_path = _state_path(args.state_file)
    state = _load_state(state_path)
    state.setdefault("done", [])
    state.setdefault("errors", [])
    _prune_resolved_errors(state)
    state["database"] = str(database.path)
    state["start_date"] = args.start_date
    state["end_date"] = args.end_date
    _save_state(state_path, state)

    dates = _trade_dates(database, args.start_date, args.end_date)
    done = set(str(item) for item in state["done"])
    synced_rows = 0
    request_count = 0
    for index, ex_date in enumerate(dates, 1):
        if ex_date in done:
            continue
        try:
            frame = tushare_api.query(
                "dividend",
                fields=DIVIDEND_FIELDS,
                ex_date=ex_date,
            )
            rows = repository.upsert_dataset(
                "dividend",
                frame,
                source="tushare/pro:dividend_ex_date",
                params={"ex_date": ex_date, "fields": DIVIDEND_FIELDS},
            )
            synced_rows += rows
            request_count += 1
            done.add(ex_date)
            state["done"] = sorted(done)
            _prune_resolved_errors(state)
            state["last"] = {
                "ex_date": ex_date,
                "index": index,
                "total_dates": len(dates),
                "rows": rows,
            }
            _save_state(state_path, state)
            print(f"[dividend_ex_date] {index}/{len(dates)} {ex_date} rows={rows}", flush=True)
        except Exception as exc:  # noqa: BLE001 - persist and resume provider failures.
            state["errors"].append({"ex_date": ex_date, "error": str(exc)})
            _save_state(state_path, state)
            print(f"[dividend_ex_date] {index}/{len(dates)} {ex_date} error={exc}", flush=True)
            if args.fail_fast:
                raise
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(
        f"dividend ex-date sync complete requests={request_count} rows={synced_rows} "
        f"state={state_path} database={database.path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
