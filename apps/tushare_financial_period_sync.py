"""Synchronize point-in-time Tushare financial datasets by report period."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_DATASETS = ("fina_indicator", "income", "balancesheet", "cashflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--start-period", default="19901231", help="First report period, YYYYMMDD.")
    parser.add_argument("--end-period", default="", help="Last report period, YYYYMMDD. Defaults to latest quarter <= as-of.")
    parser.add_argument("--as-of", required=True, help="Disclosure cutoff date, YYYYMMDD.")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _parse_yyyymmdd(text: str) -> date:
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _latest_quarter(as_of: str) -> str:
    value = _parse_yyyymmdd(as_of)
    quarters = ((3, 31), (6, 30), (9, 30), (12, 31))
    for month, day in reversed(quarters):
        if (value.month, value.day) >= (month, day):
            return f"{value.year}{month:02d}{day:02d}"
    return f"{value.year - 1}1231"


def _quarter_periods(start_period: str, end_period: str) -> list[str]:
    start = _parse_yyyymmdd(start_period)
    end = _parse_yyyymmdd(end_period)
    periods = []
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            text = f"{year}{month:02d}{day:02d}"
            if start_period <= text <= end_period:
                periods.append(text)
    return periods


def _state_path(as_of: str, state_file: str) -> Path:
    if state_file:
        return Path(state_file)
    return PATHS.state / f"tushare_financial_period_sync_to_{as_of}.json"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"done": {}, "empty": {}, "errors": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _query_paginated(dataset: str, *, period: str, page_size: int, max_pages: int) -> pd.DataFrame:
    frames = []
    offset = 0
    for _ in range(max_pages):
        frame = tushare_api.query(
            dataset,
            fields="",
            period=period,
            limit=page_size,
            offset=offset,
        )
        if frame is None or frame.empty:
            break
        frames.append(frame)
        if len(frame) < page_size:
            break
        offset += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter_point_in_time(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    date_columns = [column for column in ("ann_date", "f_ann_date", "pub_date") if column in frame.columns]
    if not date_columns:
        return frame
    any_present = pd.Series(False, index=frame.index)
    visible_date = pd.Series(False, index=frame.index)
    for column in date_columns:
        values = frame[column].fillna("").astype(str).str.replace("-", "", regex=False)
        present = values.ne("")
        any_present = any_present | present
        visible_date = visible_date | (present & values.le(as_of))
    return frame.loc[~any_present | visible_date].copy()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    end_period = args.end_period or _latest_quarter(args.as_of)
    periods = _quarter_periods(args.start_period, end_period)

    database = Database(args.database, code_version="tushare-financial-period-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    state_path = _state_path(args.as_of, args.state_file)
    state = _load_state(state_path)
    state.update({
        "as_of": args.as_of,
        "start_period": args.start_period,
        "end_period": end_period,
        "datasets": list(datasets),
        "database": str(database.path),
    })
    state.setdefault("done", {})
    state.setdefault("empty", {})
    state.setdefault("errors", [])
    _save_state(state_path, state)

    requests = rows_total = 0
    for dataset in datasets:
        done = set(state["done"].setdefault(dataset, []))
        empty = set(state["empty"].setdefault(dataset, []))
        for period in periods:
            if period in done or period in empty:
                continue
            try:
                frame = _filter_point_in_time(
                    _query_paginated(
                        dataset,
                        period=period,
                        page_size=max(1, args.page_size),
                        max_pages=max(1, args.max_pages),
                    ),
                    args.as_of,
                )
                rows = repository.upsert_dataset(
                    dataset,
                    frame,
                    source="tushare/pro:financial_period",
                    params={"period": period, "as_of": args.as_of},
                )
                (empty if rows == 0 else done).add(period)
                requests += 1
                rows_total += rows
                state["last"] = {"dataset": dataset, "period": period, "rows": rows}
                state["done"][dataset] = sorted(done)
                state["empty"][dataset] = sorted(empty)
                _save_state(state_path, state)
                print(f"[{dataset}] period={period} rows={rows}", flush=True)
            except Exception as exc:  # noqa: BLE001 - errors are persisted for resume.
                state["errors"].append({"dataset": dataset, "period": period, "error": str(exc)})
                _save_state(state_path, state)
                print(f"[{dataset}] period={period} error={exc}", flush=True)
                if args.fail_fast:
                    raise
    print(
        f"financial period sync complete requests={requests} rows={rows_total} "
        f"state={state_path} database={database.path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
