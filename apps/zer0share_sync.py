"""Synchronize the zer0share public data catalog into the unified SQLite cache."""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.data.zer0share_manifest import (
    ZER0SHARE_DATASET_MAP,
    ZER0SHARE_DATASETS,
    Zer0shareDataset,
    enabled_datasets,
)
from stock_research.storage import Database, TushareRepository


DEFAULT_SMOKE_DATE = "20260721"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Sync all enabled zer0share datasets.")
    parser.add_argument("--table", action="append", default=[], help="Sync one table; repeatable.")
    parser.add_argument("--group", action="append", default=[], help="Sync a group: stock, etf, industry, futures, options.")
    parser.add_argument("--start-date", default="", help="YYYYMMDD; defaults to --end-date for a one-day smoke.")
    parser.add_argument("--end-date", default=DEFAULT_SMOKE_DATE, help="YYYYMMDD; default is the current smoke date.")
    parser.add_argument("--full", action="store_true", help="Use each dataset's first_date as start_date.")
    parser.add_argument("--max-requests", type=int, default=0, help="Stop after this many not-yet-done request items.")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default="")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print the zer0share manifest and exit.")
    return parser


def _parse_date(text: str) -> date:
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _date_text(value: date) -> str:
    return value.strftime("%Y%m%d")


def _date_range(start: str, end: str, *, natural: bool = False, trading_days: set[str] | None = None) -> list[str]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date > end_date:
        raise ValueError("start-date must be on or before end-date")
    days = []
    current = start_date
    while current <= end_date:
        text = _date_text(current)
        if natural or trading_days is None or text in trading_days:
            days.append(text)
        current += timedelta(days=1)
    return days


def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    ranges = []
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        next_month = date(current.year + 1, 1, 1) if current.month == 12 else date(current.year, current.month + 1, 1)
        month_start = max(start_date, current)
        month_end = min(end_date, next_month - timedelta(days=1))
        ranges.append((_date_text(month_start), _date_text(month_end)))
        current = next_month
    return ranges


def _week_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    weeks = []
    seen = set()
    current = start_date
    while current <= end_date:
        iso_year, iso_week, _ = current.isocalendar()
        key = (iso_year, iso_week)
        if key not in seen:
            seen.add(key)
            monday = current - timedelta(days=current.weekday())
            weeks.append((f"{iso_year}{iso_week:02d}", _date_text(monday)))
        current += timedelta(days=7)
    return weeks


def _state_path(args) -> Path:
    if args.state_file:
        return Path(args.state_file)
    suffix = "full" if args.full else f"{args.start_date or args.end_date}_{args.end_date}"
    return PATHS.state / f"zer0share_sync_{suffix}.json"


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


def _load_trading_days(repository: TushareRepository, start: str, end: str) -> set[str] | None:
    frame = repository.load_dataset("trade_cal")
    if frame.empty or "cal_date" not in frame.columns:
        return None
    frame["cal_date"] = frame["cal_date"].astype(str).str.replace("-", "", regex=False)
    frame = frame[(frame["cal_date"] >= start) & (frame["cal_date"] <= end)]
    frame = frame[frame.get("exchange", "SSE").astype(str).eq("SSE")]
    if "is_open" in frame.columns:
        frame = frame[frame["is_open"].astype(str).isin(("1", "True", "true"))]
    return set(frame["cal_date"].dropna().astype(str).str.replace("-", "", regex=False))


def _loop_params(spec: Zer0shareDataset) -> Iterable[dict[str, str]]:
    if not spec.loops:
        yield {}
        return
    keys = list(spec.loops)
    for values in itertools.product(*(spec.loops[key] for key in keys)):
        yield dict(zip(keys, values))


def _requests_for(spec: Zer0shareDataset, start: str, end: str, trading_days: set[str] | None) -> Iterable[dict]:
    base_params = dict(spec.params)
    if spec.mode == "snapshot":
        for loop in _loop_params(spec):
            yield {"item_key": f"{spec.name}:snapshot:{json.dumps(loop, sort_keys=True)}", "params": {**base_params, **loop}}
    elif spec.mode == "range":
        for loop in _loop_params(spec):
            yield {
                "item_key": f"{spec.name}:range:{start}:{end}:{json.dumps(loop, sort_keys=True)}",
                "params": {**base_params, **loop, "start_date": start, "end_date": end},
            }
    elif spec.mode == "monthly":
        for loop in _loop_params(spec):
            for month_start, month_end in _month_ranges(start, end):
                yield {
                    "item_key": f"{spec.name}:month:{month_start}:{month_end}:{json.dumps(loop, sort_keys=True)}",
                    "params": {**base_params, **loop, "start_date": month_start, "end_date": month_end},
                }
    elif spec.mode == "weekly":
        for week, week_date in _week_ranges(start, end):
            yield {
                "item_key": f"{spec.name}:week:{week}",
                "params": {**base_params, spec.date_param: week},
                "context": {"week_date": week_date},
            }
    elif spec.mode == "daily":
        days = _date_range(start, end, natural=spec.calendar == "natural", trading_days=trading_days)
        for day in days:
            for loop in _loop_params(spec):
                yield {
                    "item_key": f"{spec.name}:day:{day}:{json.dumps(loop, sort_keys=True)}",
                    "params": {**base_params, **loop, spec.date_param: day},
                }
    else:
        raise ValueError(f"unsupported sync mode for {spec.name}: {spec.mode}")


def _query_frames(spec: Zer0shareDataset, params: dict) -> pd.DataFrame:
    if spec.name == "basic" and "," in str(params.get("list_status", "")):
        frames = []
        for status in str(params["list_status"]).split(","):
            frame = tushare_api.query(
                spec.api_name,
                fields=spec.fields_arg,
                **{**params, "list_status": status},
            )
            if frame is not None and not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(spec.fields))
    if not spec.pagination:
        return tushare_api.query(spec.api_name, fields=spec.fields_arg, **params)
    frames = []
    limit = 3000 if spec.name == "etf_sh_cons" else 1000
    offset = 0
    while True:
        frame = tushare_api.query(
            spec.api_name,
            fields=spec.fields_arg,
            **params,
            limit=limit,
            offset=offset,
        )
        if frame is None or frame.empty:
            break
        frames.append(frame)
        if len(frame) < limit:
            break
        offset += limit
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(spec.fields))


def _select_fields(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(fields))
    for column in fields:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(fields)]


def _sync_derived_member(spec: Zer0shareDataset) -> pd.DataFrame:
    if spec.name == "sw_member":
        frames = []
        for src in ("SW2014", "SW2021"):
            l1 = tushare_api.query("index_classify", fields="index_code", level="L1", src=src)
            for l1_code in l1.get("index_code", pd.Series(dtype=str)).dropna().astype(str).unique():
                for is_new in ("Y", "N"):
                    frame = tushare_api.query(
                        "index_member_all",
                        fields=spec.fields_arg,
                        l1_code=l1_code,
                        is_new=is_new,
                    )
                    if frame is not None and not frame.empty:
                        frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(spec.fields))
    if spec.name == "ci_member":
        first = tushare_api.query("ci_index_member", fields=spec.fields_arg)
        frames = [first] if first is not None and not first.empty else []
        l1_codes = first.get("l1_code", pd.Series(dtype=str)).dropna().astype(str).unique() if first is not None else []
        for l1_code in l1_codes:
            for is_new in ("Y", "N"):
                frame = tushare_api.query(
                    "ci_index_member",
                    fields=spec.fields_arg,
                    l1_code=l1_code,
                    is_new=is_new,
                )
                if frame is not None and not frame.empty:
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(spec.fields))
    raise ValueError(f"unsupported derived dataset: {spec.name}")


def _select_specs(args) -> tuple[Zer0shareDataset, ...]:
    if args.list:
        return tuple(ZER0SHARE_DATASETS)
    names = set(args.table)
    groups = set(args.group)
    if args.all or not (names or groups):
        return enabled_datasets()
    selected = []
    for spec in enabled_datasets():
        if spec.name in names or spec.group in groups:
            selected.append(spec)
    missing = sorted(names - {item.name for item in selected})
    if missing:
        raise ValueError(f"unknown or disabled zer0share tables: {', '.join(missing)}")
    return tuple(selected)


def _sync_spec(
    spec: Zer0shareDataset,
    repository: TushareRepository,
    state: dict,
    state_path: Path,
    *,
    start: str,
    end: str,
    trading_days: set[str] | None,
    max_requests: int,
    fail_fast: bool,
) -> tuple[int, int]:
    done = set(state.setdefault("done", {}).setdefault(spec.name, []))
    empty = set(state.setdefault("empty", {}).setdefault(spec.name, []))
    processed = rows_total = 0

    if spec.mode == "derived_snapshot":
        item_key = f"{spec.name}:derived_snapshot"
        if item_key in done or item_key in empty:
            return 0, 0
        try:
            frame = _select_fields(_sync_derived_member(spec), spec.fields)
            rows = repository.upsert_dataset(spec.name, frame, source="tushare/pro:zer0share", params={"api_name": spec.api_name})
            (empty if rows == 0 else done).add(item_key)
            processed += 1
            rows_total += rows
            print(f"[{spec.name}] derived rows={rows}", flush=True)
        except Exception as exc:  # noqa: BLE001
            state["errors"].append({"dataset": spec.name, "item": item_key, "error": str(exc)})
            print(f"[{spec.name}] derived error={exc}", flush=True)
            if fail_fast:
                raise
        finally:
            state["done"][spec.name] = sorted(done)
            state["empty"][spec.name] = sorted(empty)
            _save_state(state_path, state)
        return processed, rows_total

    for request in _requests_for(spec, start, end, trading_days):
        item_key = request["item_key"]
        if item_key in done or item_key in empty:
            continue
        if max_requests and processed >= max_requests:
            break
        params = request["params"]
        try:
            frame = _select_fields(_query_frames(spec, params), spec.fields)
            context = request.get("context") or {}
            for key, value in context.items():
                if key not in frame.columns:
                    frame[key] = value
            rows = repository.upsert_dataset(
                spec.name,
                frame,
                source="tushare/pro:zer0share",
                params={"api_name": spec.api_name, **params},
            )
            (empty if rows == 0 else done).add(item_key)
            processed += 1
            rows_total += rows
            state["last"] = {"dataset": spec.name, "item": item_key, "rows": rows}
            state["done"][spec.name] = sorted(done)
            state["empty"][spec.name] = sorted(empty)
            _save_state(state_path, state)
            print(f"[{spec.name}] {item_key} rows={rows}", flush=True)
        except Exception as exc:  # noqa: BLE001 - persisted for resumable sync.
            state["errors"].append({"dataset": spec.name, "item": item_key, "params": params, "error": str(exc)})
            _save_state(state_path, state)
            print(f"[{spec.name}] {item_key} error={exc}", flush=True)
            if fail_fast:
                raise
    return processed, rows_total


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = _select_specs(args)
    if args.list:
        for spec in specs:
            status = "enabled" if spec.enabled else "external"
            print(f"{spec.name}\t{spec.group}\t{spec.api_name}\t{spec.mode}\t{status}\t{spec.note}")
        return 0

    database = Database(args.database, code_version="zer0share-sync-v1")
    database.initialize()
    repository = TushareRepository(database)
    end = args.end_date
    state_path = _state_path(args)
    state = _load_state(state_path)
    state["database"] = str(database.path)
    state["end_date"] = end
    state["source_catalog"] = "github.com/zer0quant/zer0share"
    prior_tables = set(state.get("requested_tables", []))
    state["requested_tables"] = sorted(prior_tables | {item.name for item in specs})
    state.setdefault("runs", []).append({
        "tables": [item.name for item in specs],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "full": bool(args.full),
        "max_requests": int(args.max_requests),
    })
    _save_state(state_path, state)

    total_requests = total_rows = 0
    for spec in specs:
        start = spec.first_date if args.full and spec.first_date else (args.start_date or end)
        trading_days = None
        if spec.mode == "daily" and spec.calendar == "trading":
            trading_days = _load_trading_days(repository, start, end)
        processed, rows = _sync_spec(
            spec,
            repository,
            state,
            state_path,
            start=start,
            end=end,
            trading_days=trading_days,
            max_requests=max(0, args.max_requests - total_requests) if args.max_requests else 0,
            fail_fast=args.fail_fast,
        )
        total_requests += processed
        total_rows += rows
        if args.max_requests and total_requests >= args.max_requests:
            break

    print(
        f"zer0share sync complete requests={total_requests} rows={total_rows} "
        f"state={state_path} database={database.path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
