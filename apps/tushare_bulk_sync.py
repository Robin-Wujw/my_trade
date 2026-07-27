"""Bulk synchronize Tushare datasets that require per-stock requests."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_FINANCIAL_DATASETS = ("fina_indicator", "income", "balancesheet", "cashflow")
EVENT_DATASETS = ("dividend", "share_float")
BULK_DATASET_FIELDS = {
    "dividend": (
        "ts_code,ann_date,end_date,record_date,ex_date,div_listdate,div_proc,"
        "stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax"
    ),
    "share_float": (
        "ts_code,ann_date,float_date,float_share,float_ratio,holder_name,share_type"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_FINANCIAL_DATASETS))
    parser.add_argument(
        "--period",
        default="",
        help="Report period, for financial datasets, for example 20260331.",
    )
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--max-codes", type=int, default=0, help="Optional cap for smoke runs.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra sleep between calls.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--state-file", default="")
    return parser


def _load_stock_codes(repository: TushareRepository) -> list[str]:
    frame = repository.load_dataset("stock_basic")
    if frame.empty:
        raise RuntimeError("stock_basic is not cached; run `python -m apps.data_sync stock_basic` first")
    if "ts_code" not in frame.columns:
        raise RuntimeError("cached stock_basic has no ts_code column")
    if "list_status" in frame.columns:
        frame = frame[frame["list_status"].fillna("L").astype(str).eq("L")]
    return sorted(frame["ts_code"].dropna().astype(str).unique())


def _state_path(period: str, state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / f"tushare_bulk_sync_{period}.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"done": {}, "errors": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    financial = set(DEFAULT_FINANCIAL_DATASETS)
    missing_period = sorted(set(datasets) & financial) and not args.period
    if missing_period:
        raise SystemExit("--period is required for financial datasets")
    database = Database(args.database, code_version="tushare-bulk-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    codes = _load_stock_codes(repository)
    if args.max_codes > 0:
        codes = codes[: args.max_codes]

    state_label = args.period or "events"
    state_path = _state_path(state_label, args.state_file)
    state = _load_state(state_path)
    state.setdefault("done", {})
    state.setdefault("errors", [])
    state["period"] = args.period
    state["datasets"] = list(datasets)
    state["database"] = str(database.path)
    _save_state(state_path, state)

    synced_rows = 0
    request_count = 0
    for dataset in datasets:
        done_codes = set(state["done"].get(dataset, []))
        for index, ts_code in enumerate(codes, 1):
            if ts_code in done_codes:
                continue
            try:
                query_params = {"ts_code": ts_code}
                if args.period:
                    query_params["period"] = args.period
                fields = BULK_DATASET_FIELDS.get(dataset, "")
                frame = tushare_api.query(dataset, fields=fields, **query_params)
                rows = repository.upsert_dataset(
                    dataset,
                    frame,
                    source="tushare/pro",
                    params={**query_params, "fields": fields or "*"},
                )
                synced_rows += rows
                request_count += 1
                done_codes.add(ts_code)
                state["done"][dataset] = sorted(done_codes)
                state["last"] = {
                    "dataset": dataset,
                    "ts_code": ts_code,
                    "index": index,
                    "total_codes": len(codes),
                    "rows": rows,
                }
                _save_state(state_path, state)
                print(
                    f"[{dataset}] {index}/{len(codes)} {ts_code} rows={rows}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - errors are persisted for resume.
                item = {"dataset": dataset, "ts_code": ts_code, "error": str(exc)}
                state["errors"].append(item)
                _save_state(state_path, state)
                print(f"[{dataset}] {index}/{len(codes)} {ts_code} error={exc}", flush=True)
                if args.fail_fast:
                    raise
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(
        f"bulk sync complete requests={request_count} rows={synced_rows} state={state_path} database={database.path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
