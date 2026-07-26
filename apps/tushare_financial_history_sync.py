"""Synchronize per-stock Tushare financial history into SQLite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_FIELDS = (
    "ts_code,ann_date,end_date,eps,dt_eps,bps,roe,roe_dt,"
    "netprofit_yoy,dt_netprofit_yoy"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fina_indicator")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--start-period", default="20220930")
    parser.add_argument("--end-period", default="20260331")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _state_path(dataset: str, state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / f"tushare_{dataset}_history_sync.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"done": [], "empty": [], "errors": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_stock_codes(repository: TushareRepository) -> list[str]:
    frame = repository.load_dataset("stock_basic")
    if frame.empty:
        raise RuntimeError("stock_basic is not cached")
    work = frame.copy()
    if "list_status" in work:
        work = work[work["list_status"].fillna("L").astype(str).eq("L")]
    if "ts_code" not in work:
        raise RuntimeError("cached stock_basic has no ts_code column")
    return sorted(work["ts_code"].dropna().astype(str).unique())


def _filter_periods(frame: pd.DataFrame, start_period: str, end_period: str) -> pd.DataFrame:
    if frame is None or frame.empty or "end_date" not in frame:
        return pd.DataFrame()
    work = frame.copy()
    period = work["end_date"].fillna("").astype(str).str.replace("-", "", regex=False)
    return work.loc[period.ge(start_period) & period.le(end_period)].copy()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = Database(args.database, code_version="tushare-financial-history-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    codes = _load_stock_codes(repository)
    if args.max_codes > 0:
        codes = codes[: args.max_codes]

    state_path = _state_path(args.dataset, args.state_file)
    state = _load_state(state_path)
    state.update({
        "dataset": args.dataset,
        "fields": args.fields,
        "start_period": args.start_period,
        "end_period": args.end_period,
        "database": str(database.path),
    })
    state.setdefault("done", [])
    state.setdefault("empty", [])
    state.setdefault("errors", [])
    done = set(state["done"])
    empty = set(state["empty"])
    _save_state(state_path, state)

    request_count = 0
    rows_total = 0
    for index, ts_code in enumerate(codes, 1):
        if args.max_requests > 0 and request_count >= args.max_requests:
            break
        if ts_code in done or ts_code in empty:
            continue
        try:
            frame = tushare_api.query(
                args.dataset,
                fields=args.fields,
                ts_code=ts_code,
            )
            frame = _filter_periods(frame, args.start_period, args.end_period)
            rows = repository.upsert_dataset(
                args.dataset,
                frame,
                source="tushare/pro:per_stock_history",
                params={
                    "ts_code": ts_code,
                    "start_period": args.start_period,
                    "end_period": args.end_period,
                },
            )
            request_count += 1
            rows_total += rows
            (empty if rows == 0 else done).add(ts_code)
            state["done"] = sorted(done)
            state["empty"] = sorted(empty)
            state["last"] = {
                "ts_code": ts_code,
                "index": index,
                "total_codes": len(codes),
                "rows": rows,
            }
            if request_count % 20 == 0 or index == len(codes):
                _save_state(state_path, state)
            if not args.quiet or request_count % 100 == 0:
                print(
                    f"[{args.dataset}] {index}/{len(codes)} {ts_code} rows={rows}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - persisted for resume.
            state["errors"].append({"ts_code": ts_code, "error": str(exc)})
            _save_state(state_path, state)
            print(f"[{args.dataset}] {index}/{len(codes)} {ts_code} error={exc}", flush=True)
            if args.fail_fast:
                raise
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
                "state_file": str(state_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
