"""Incremental daily research capture with explicit receipts and fail-closed coverage.

Historical backfills are captured now. They never become historical forward
evidence merely because their trade or announcement date is in the past.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pandas as pd

from stock_research.api import tushare
from stock_research.core.financial_period import latest_visible_report_period
from stock_research.data.zer0share_manifest import ZER0SHARE_DATASET_MAP
from stock_research.storage import Database, TushareRepository


SCHEMA_VERSION = "forward-data-sync-v2"
# Obtain explicit suspension evidence before assessing a missing daily bar.
DAILY_DATASETS = ("stock_st", "suspend_d", "daily_kline", "adj_factor", "daily_basic", "stk_limit")
BENCHMARKS = ("000300.SH", "000905.SH", "000852.SH")
NUMERIC_FIELDS = {
    "daily_kline": {"open": True, "high": True, "low": True, "close": True, "vol": False, "amount": False},
    "adj_factor": {"adj_factor": True},
    "daily_basic": {"close": True, "turnover_rate": False, "total_mv": True, "circ_mv": True},
    "stk_limit": {"up_limit": True, "down_limit": True},
}
FINANCIAL_FIELDS = {"income": "n_income", "balancesheet": "total_assets",
                    "cashflow": "n_cashflow_act", "fina_indicator": "roe"}
DIVIDEND_FIELDS = ("ts_code", "ann_date", "imp_ann_date", "end_date", "record_date", "ex_date",
                   "pay_date", "div_listdate", "div_proc", "stk_div", "stk_bo_rate", "stk_co_rate",
                   "cash_div", "cash_div_tax")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def completed_session(calendar: pd.DataFrame, now=None) -> str:
    """Do not claim today's daily data are complete before 16:30 China time."""
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    timestamp = (timestamp.tz_localize("Asia/Shanghai") if timestamp.tzinfo is None
                 else timestamp.tz_convert("Asia/Shanghai"))
    cutoff = timestamp.tz_localize(None).normalize()
    if (timestamp.hour, timestamp.minute) < (16, 30):
        cutoff -= pd.Timedelta(days=1)
    dates = pd.to_datetime(calendar["cal_date"], errors="coerce")
    opened = calendar["is_open"].astype(str).isin(("1", "True", "true"))
    if "exchange" in calendar:
        opened &= calendar["exchange"].eq("SSE")
    eligible = dates[opened & dates.le(cutoff)]
    if eligible.empty:
        raise ValueError("No completed session in the cached calendar")
    return eligible.max().strftime("%Y-%m-%d")


def _dated_rows(frame: pd.DataFrame, day: str, *, allow_empty: bool, dataset=None) -> pd.DataFrame:
    if frame is None or not {"ts_code", "trade_date"}.issubset(frame.columns):
        raise ValueError("Provider response missing ts_code/trade_date schema")
    required = set(NUMERIC_FIELDS.get(dataset, {}))
    if dataset == "suspend_d":
        required |= {"suspend_type", "suspend_timing"}
    if dataset == "stock_st":
        required |= {"type"}
    if required - set(frame):
        raise ValueError(f"{dataset} response missing required fields: {sorted(required - set(frame))}")
    if frame.empty:
        if allow_empty:
            return frame
        raise ValueError("Empty daily response is retryable, not a no-trade declaration")
    dates = pd.to_datetime(frame["trade_date"].astype(str), errors="coerce")
    if not dates.eq(pd.Timestamp(day)).all() or not frame["ts_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
        raise ValueError("Provider response contains wrong date or missing stock code")
    # Sparse tables can carry several valid events for one stock. Core data
    # must contain one unambiguous observation, never arbitrary keep-last.
    frame = frame.drop_duplicates().copy()
    if dataset in NUMERIC_FIELDS and frame["ts_code"].duplicated().any():
        raise ValueError(f"Conflicting duplicate daily rows: {dataset}")
    for field, positive in NUMERIC_FIELDS.get(dataset, {}).items():
        values = pd.to_numeric(frame[field], errors="coerce")
        if not np.isfinite(values).all() or (values.le(0) if positive else values.lt(0)).any():
            raise ValueError(f"{dataset} has invalid {field}")
    if dataset == "daily_kline":
        values = frame[["open", "high", "low", "close"]].apply(pd.to_numeric)
        if (values["high"].lt(values[["open", "low", "close"]].max(axis=1))
                | values["low"].gt(values[["open", "high", "close"]].min(axis=1))).any():
            raise ValueError("Daily OHLC range is inconsistent")
    if dataset == "stk_limit" and pd.to_numeric(frame["up_limit"]).le(pd.to_numeric(frame["down_limit"])).any():
        raise ValueError("Invalid up/down price limit ordering")
    if dataset == "suspend_d" and not frame["suspend_type"].eq("S").all():
        raise ValueError("Suspension response did not honor suspend_type=S")
    if dataset == "stock_st" and (frame["type"].isna() | frame["type"].astype(str).str.strip().eq("")).any():
        raise ValueError("ST response has missing type")
    return frame


def database_identity(database: Database) -> str:
    """Persist an epoch inside SQLite so replacing the file invalidates receipts."""
    with database.connect() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS ops.forward_sync_identity (singleton INTEGER PRIMARY KEY, token TEXT NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO ops.forward_sync_identity(singleton, token) VALUES (1, ?)", [str(uuid.uuid4())])
        return connection.execute("SELECT token FROM ops.forward_sync_identity WHERE singleton=1").fetchone()[0]


def _scope_fingerprint(database, scope):
    columns = {"dataset": ("dataset", "="), "trade_date": ("trade_date", "="),
               "report_period": ("report_period", "="), "ts_code": ("ts_code", "="),
               "start_date": ("trade_date", ">="), "end_date": ("trade_date", "<=")}
    if not scope.get("dataset") or set(scope) - (set(columns) | {"event_date"}):
        raise ValueError("Invalid receipt database scope")
    fields = sorted(set(scope) - {"event_date"})
    clauses = [f"{columns[key][0]} {columns[key][1]} ?" for key in fields]
    params = [scope[key] for key in fields]
    if "event_date" in scope:
        clauses.append("(" + " OR ".join(
            f"replace(json_extract(payload_json, '$.{field}'), '-', '') = ?"
            for field in ("ex_date", "record_date", "ann_date")) + ")")
        params.extend([scope["event_date"].replace("-", "")] * 3)
    with database.connect(read_only=True) as connection:
        rows = connection.execute(
            "SELECT row_key, payload_json FROM raw.tushare_dataset_rows WHERE "
            + " AND ".join(clauses) + " ORDER BY row_key", params,
        ).fetchall()
    return {"row_count": len(rows), "sha256": hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}


def validate_receipt(database: Database, receipt: dict) -> bool:
    """Check the SQLite epoch and actual current rows before using a saved receipt."""
    try:
        with database.connect(read_only=True) as connection:
            row = connection.execute("SELECT token FROM ops.forward_sync_identity WHERE singleton=1").fetchone()
        return bool(row and row[0] == receipt.get("database_identity")
                    and _scope_fingerprint(database, receipt["scope"]) == receipt.get("database_fingerprint"))
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error):
        return False


def verify_completed_receipts(state: dict, repository: TushareRepository) -> dict:
    """Read-only guard for forward consumers; fail if SQLite differs from sync evidence."""
    database = repository.database
    receipts = state.get("completed", {})
    if (state.get("version") != SCHEMA_VERSION or state.get("status") != "completed"
            or state.get("database") != str(database.path.resolve()) or not receipts):
        raise ValueError("Forward synchronization is not complete for this database")
    if any(receipt.get("database_identity") != state.get("database_identity") for receipt in receipts.values()):
        raise ValueError("Sync receipt/database identity mismatch")
    invalid = [key for key, receipt in receipts.items() if not validate_receipt(database, receipt)]
    if invalid:
        raise ValueError(f"Completed sync receipts no longer match SQLite: {invalid[:10]}")
    digest = hashlib.sha256(json.dumps({key: value["database_fingerprint"]
                                       for key, value in sorted(receipts.items())}, sort_keys=True).encode()).hexdigest()
    if digest != state.get("database_fingerprint", {}).get("completed_receipts_sha256"):
        raise ValueError("Sync receipt-set fingerprint mismatch")
    return {"status": "verified", "receipt_count": len(receipts),
            "database_identity": state["database_identity"], "completed_receipts_sha256": digest}


def _full_day_suspensions(frame):
    timing = frame["suspend_timing"].fillna("unknown").astype(str).str.strip()
    return set(frame.loc[timing.isin({"全天", "全天停牌", "停牌一天", "09:30-15:00", "09:30:00-15:00:00"}), "ts_code"])


def _dividend_issues(frame, day):
    """Do not manufacture zero distributions or payment/listing dates."""
    issues = []
    for _, row in frame.iterrows():
        code = str(row.get("ts_code", ""))
        reason = []
        if not pd.Series([code]).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").iloc[0]:
            reason.append("invalid_stock_code")
        ann = pd.to_datetime(str(row.get("ann_date")), errors="coerce")
        if pd.isna(ann) or ann > pd.Timestamp(day):
            reason.append("announcement_not_visible")
        dates = {name: pd.to_datetime(str(row.get(name)), errors="coerce")
                 for name in ("record_date", "ex_date", "pay_date", "div_listdate")}
        if row.get("div_proc") in {"预案", "股东大会通过", "不分配", "取消分红"} and pd.isna(dates["ex_date"]) and pd.isna(dates["record_date"]):
            if reason:
                issues.append({"code": code, "reasons": reason})
            continue  # A proposal is captured, not an implemented cash/share event.
        if row.get("div_proc") != "实施":
            reason.append("implementation_not_confirmed")
        if pd.isna(dates["record_date"]) or pd.isna(dates["ex_date"]) or dates["record_date"] >= dates["ex_date"]:
            reason.append("record_or_ex_date_invalid")
        numbers = {name: pd.to_numeric(row.get(name), errors="coerce")
                   for name in ("cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate")}
        finite = {name: float(value) for name, value in numbers.items() if pd.notna(value) and np.isfinite(value)}
        if any(value < 0 for value in finite.values()):
            reason.append("negative_distribution")
        cash_values = [finite[name] for name in ("cash_div", "cash_div_tax") if name in finite]
        stock_values = [finite[name] for name in ("stk_div", "stk_bo_rate", "stk_co_rate") if name in finite]
        if not cash_values or not stock_values:
            reason.append("distribution_amount_unknown")
        if cash_values and max(cash_values) > 0 and (pd.isna(dates["pay_date"]) or dates["pay_date"] < dates["ex_date"]):
            reason.append("cash_pay_date_missing_or_invalid")
        if stock_values and max(stock_values) > 0 and (pd.isna(dates["div_listdate"]) or dates["div_listdate"] < dates["ex_date"]):
            reason.append("stock_listing_date_missing_or_invalid")
        if reason:
            issues.append({"code": code, "reasons": reason})
    return issues


def run_sync(
    *, database_path: str | Path, state_path: str | Path, end_date: str,
    start_date: str | None = None, max_requests: int = 0,
    capture_financials: bool = True, query=None, now=None,
) -> dict:
    """Capture missing daily batches plus current financial vintages.

    Each run refreshes the master and calendar. Successful daily receipts are
    reused only for the same database; failed/partial/empty core requests are
    never put in the completed set. Sparse ST/suspension responses require a
    successful provider call and an explicit schema even when empty.
    """
    query = query or tushare.query
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    current = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    current = current.tz_localize("Asia/Shanghai") if current.tzinfo is None else current.tz_convert("Asia/Shanghai")
    end = pd.Timestamp(end_date).normalize()
    cutoff = current.tz_localize(None).normalize() - pd.Timedelta(days=int((current.hour, current.minute) < (16, 30)))
    if pd.isna(end) or end.tzinfo is not None or end > cutoff:
        raise ValueError("Sync end date must be completed: today's data require 16:30 China time")
    database = Database(database_path, code_version=SCHEMA_VERSION)
    database.initialize()
    epoch = database_identity(database)
    repository = TushareRepository(database)
    state_path = Path(state_path)
    identity = str(Path(database_path).resolve())
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") not in {SCHEMA_VERSION, "forward-data-sync-v1"} or state.get("database") != identity:
            raise ValueError("Sync receipt belongs to a different schema or database")
    else:
        state = {"version": SCHEMA_VERSION, "database": identity, "completed": {}}
    invalidated = []
    for key, receipt in list(state["completed"].items()):
        if not validate_receipt(database, receipt):
            invalidated.append(key)
            state["completed"].pop(key)
    state.update(version=SCHEMA_VERSION, database_identity=epoch)
    if start_date is None:
        with database.connect(read_only=True) as connection:
            last = connection.execute(
                "SELECT MAX(trade_date) FROM raw.tushare_dataset_rows WHERE dataset='daily_kline'",
            ).fetchone()[0]
        start = pd.Timestamp(last) + pd.Timedelta(days=1) if last else end
        if state.get("status") != "completed" and state.get("requested_start"):
            start = min(start, pd.Timestamp(state["requested_start"]))
        # A restored/replaced database can still have a later MAX(date).
        # Revisit invalidated historical receipts instead of skipping holes.
        for key in invalidated:
            if key.split(":", 1)[0] in DAILY_DATASETS:
                start = min(start, pd.Timestamp(key.split(":", 1)[1]))
    else:
        start = pd.Timestamp(start_date).normalize()
        if pd.isna(start) or start > end:
            raise ValueError("Invalid requested sync interval")
    # Recheck the latest available session on an already-current database.
    start = min(start, end)
    state.update({"started_at": utc_now(), "requested_start": str(start.date()),
                  "requested_end": str(end.date()), "status": "running", "errors": [],
                  "requests": 0, "research_only": True,
                  "historical_backfill_is_forward_evidence": False,
                  "invalidated_receipts": invalidated, "incomplete": {}})
    write_json(state_path, state)
    required_receipt_keys = set()

    def fetch(api_name, *, fields="", **params):
        if max_requests and state["requests"] >= max_requests:
            raise RuntimeError("Request budget reached; synchronization is incomplete")
        state["requests"] += 1
        return query(api_name, fields=fields, **params)

    def store(dataset, frame, params):
        return repository.upsert_dataset(dataset, frame, source="tushare/forward_capture", params=params)

    def complete(key, scope, rows, **metadata):
        state["completed"][key] = {"rows": rows, "captured_at": utc_now(), "scope": scope,
                                   "database_identity": epoch,
                                   "database_fingerprint": _scope_fingerprint(database, scope), **metadata}
        state["incomplete"].pop(key, None)
        write_json(state_path, state)

    def incomplete(key, required, covered, **metadata):
        state["completed"].pop(key, None)
        state["incomplete"][key] = {"required_count": len(required), "covered_count": len(required & covered),
                                    "coverage_ratio": len(required & covered) / len(required) if required else 1.0,
                                    "missing_codes": sorted(required - covered), **metadata}
        write_json(state_path, state)

    try:
        calendar = fetch(
            "trade_cal", fields=ZER0SHARE_DATASET_MAP["trade_cal"].fields_arg,
            exchange="SSE", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        )
        if calendar.empty or not {"cal_date", "is_open", "exchange"}.issubset(calendar):
            raise ValueError("Calendar is empty or incomplete")
        calendar_dates = pd.to_datetime(calendar["cal_date"].astype(str), errors="coerce")
        expected_calendar = set(pd.date_range(start, end).strftime("%Y-%m-%d"))
        actual_calendar = set(calendar_dates[calendar["exchange"].eq("SSE")].dt.strftime("%Y-%m-%d"))
        if not expected_calendar.issubset(actual_calendar):
            raise ValueError("Calendar does not cover the entire requested interval")
        calendar_flags = calendar["is_open"].astype(str).str.lower()
        if (calendar_dates.isna().any() or not calendar_flags.isin({"1", "0", "true", "false"}).all()
                or calendar[["exchange", "cal_date"]].duplicated().any()):
            raise ValueError("Invalid or conflicting calendar rows")
        store("trade_cal", calendar, {"exchange": "SSE"})
        calendar_key = f"trade_cal:{end.date()}"
        required_receipt_keys.add(calendar_key)
        complete(calendar_key, {"dataset": "trade_cal"}, len(calendar),
                 coverage_basis="successful_calendar_interval_query")
        sessions = sorted(calendar_dates[
            calendar["is_open"].astype(str).isin(("1", "True", "true"))
            & calendar["exchange"].eq("SSE") & calendar_dates.between(start, end)
        ].dt.strftime("%Y-%m-%d").unique())
        master_frames = []
        for listing_status in ("L", "D", "P"):
            master = fetch("stock_basic", fields=ZER0SHARE_DATASET_MAP["basic"].fields_arg,
                           exchange="", list_status=listing_status)
            if not {"ts_code", "list_date", "list_status"}.issubset(master):
                raise ValueError(f"Invalid security master for status {listing_status}")
            if listing_status == "L" and master.empty:
                raise ValueError("Listed security master is empty")
            if not master.empty and not master["list_status"].eq(listing_status).all():
                raise ValueError(f"Provider did not honor listing status {listing_status}")
            master_frames.append(master)
        master = pd.concat(master_frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
        if not master["ts_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
            raise ValueError("Invalid code in security master")
        listing = pd.to_datetime(master["list_date"].astype(str), errors="coerce")
        delisting = pd.to_datetime(master.get("delist_date", pd.Series(index=master.index, dtype=str)).astype(str), errors="coerce")
        if listing.isna().any() or (master["list_status"].eq("D") & delisting.isna()).any():
            raise ValueError("Missing list/delist dates prevent historical universe coverage")
        for name in ("basic", "stock_basic"):
            store(name, master, {"list_status": "L,D,P"})
            master_key = f"{name}:{end.date()}"
            required_receipt_keys.add(master_key)
            complete(master_key, {"dataset": name}, len(master),
                     codes=sorted(set(master["ts_code"])), coverage_basis="all_listing_statuses")
        for day in sessions:
            expected = set(master.loc[
                listing.le(day) & (delisting.isna() | delisting.ge(day)), "ts_code",
            ].astype(str))
            if not expected:
                raise ValueError(f"No listed universe for {day}")
            daily_codes, full_day_suspensions = set(), set()
            for dataset in DAILY_DATASETS:
                key = f"{dataset}:{day}"
                required_receipt_keys.add(key)
                receipt = state["completed"].get(key)
                required = expected - full_day_suspensions if dataset == "daily_kline" else daily_codes
                if receipt:
                    if dataset in NUMERIC_FIELDS and not required.issubset(set(receipt.get("codes", []))):
                        state["completed"].pop(key)
                        receipt = None
                if receipt:
                    if dataset == "daily_kline":
                        daily_codes = set(receipt["codes"]) & expected
                    elif dataset == "suspend_d":
                        full_day_suspensions = set(receipt.get("full_day_codes", []))
                    continue
                spec = ZER0SHARE_DATASET_MAP[dataset]
                params = {**spec.params, "trade_date": day.replace("-", "")}
                frame = _dated_rows(fetch(spec.api_name, fields=spec.fields_arg, **params), day,
                                    allow_empty=dataset in {"stock_st", "suspend_d"}, dataset=dataset)
                codes = set(frame["ts_code"].astype(str))
                store(dataset, frame, params)
                if dataset == "daily_kline":
                    daily_codes = codes & expected
                elif dataset == "suspend_d":
                    full_day_suspensions = _full_day_suspensions(frame)
                if dataset in NUMERIC_FIELDS and not required.issubset(codes):
                    # Partial rows remain useful cached data; no full receipt.
                    # Re-running revisits this batch and records exact gaps.
                    incomplete(key, required, codes, reason="missing_required_security_rows")
                    continue
                complete(key, {"dataset": dataset, "trade_date": day}, len(frame), codes=sorted(codes),
                         full_day_codes=sorted(full_day_suspensions) if dataset == "suspend_d" else [],
                         coverage_basis="explicit_sparse_response" if dataset in {"stock_st", "suspend_d"} else "all_required_securities")
                print(f"captured {key} rows={len(frame)}", flush=True)
            dividend_key = f"dividend:{day}"
            required_receipt_keys.add(dividend_key)
            if dividend_key not in state["completed"]:
                chunks, issues = [], []
                for event_field in ("ex_date", "record_date", "ann_date"):
                    frame = fetch("dividend", fields=",".join(DIVIDEND_FIELDS), **{event_field: day.replace("-", "")})
                    if frame is None or not set(DIVIDEND_FIELDS).issubset(frame):
                        issues.append({"code": None, "reasons": [f"dividend_{event_field}_schema_incomplete"]})
                        continue
                    if not pd.to_datetime(frame[event_field].astype(str), errors="coerce").eq(pd.Timestamp(day)).all():
                        raise ValueError(f"Dividend API did not honor {event_field}={day}")
                    chunks.append(frame)
                    issues.extend(_dividend_issues(frame, day))
                combined = pd.concat(chunks, ignore_index=True).drop_duplicates() if chunks else pd.DataFrame(columns=DIVIDEND_FIELDS)
                if issues:
                    # Keep observation evidence without replacing previously
                    # valid current rows with a response missing payment dates.
                    repository.append_dataset_versions("dividend", combined, source="tushare/forward_capture_incomplete")
                    state["incomplete"][dividend_key] = {"reason": "corporate_action_data_incomplete", "issues": issues}
                    write_json(state_path, state)
                else:
                    store("dividend", combined, {"event_date": day, "queries": ["ex_date", "record_date", "ann_date"]})
                    complete(dividend_key, {"dataset": "dividend", "event_date": day}, len(combined),
                             coverage_basis="successful_ex_record_announcement_date_queries",
                             dividend_payment_dates_verified=True)
        for code in BENCHMARKS:
            key = f"index_daily:{code}:{start.date()}:{end.date()}"
            required_receipt_keys.add(key)
            if key in state["completed"]:
                continue
            params = {"ts_code": code, "start_date": start.strftime("%Y%m%d"),
                      "end_date": end.strftime("%Y%m%d")}
            frame = fetch("index_daily", fields=ZER0SHARE_DATASET_MAP["index_daily"].fields_arg, **params)
            if not {"ts_code", "trade_date", "close"}.issubset(frame):
                raise ValueError(f"Invalid benchmark response: {code}")
            dates = set(pd.to_datetime(frame["trade_date"].astype(str), errors="coerce").dt.strftime("%Y-%m-%d"))
            if not set(sessions).issubset(dates) or not frame["ts_code"].eq(code).all():
                raise ValueError(f"Incomplete benchmark response: {code}")
            closes = pd.to_numeric(frame["close"], errors="coerce")
            if not np.isfinite(closes).all() or closes.le(0).any() or frame["trade_date"].duplicated().any():
                raise ValueError(f"Invalid benchmark close/date: {code}")
            store("index_daily", frame, params)
            complete(key, {"dataset": "index_daily", "ts_code": code,
                           "start_date": str(start.date()), "end_date": str(end.date())}, len(frame))
        if capture_financials:
            annual_year = end.year - (1 if end >= pd.Timestamp(year=end.year, month=4, day=30) else 2)
            periods = sorted({f"{annual_year}1231", latest_visible_report_period(end).replace("-", "")})
            capture_day = utc_now()[:10]
            pagination = state.setdefault("financial_pagination", {})
            tasks = [(dataset, period, f"{dataset}:{pd.Timestamp(period).date()}:as_of:{end.date()}")
                     for dataset in FINANCIAL_FIELDS for period in periods]
            # Previously incomplete short responses go behind not-yet-attempted
            # tables, so a request cap cannot starve the remaining financials.
            tasks.sort(key=lambda item: bool(pagination.get(item[2], {}).get("finished")))
            for dataset, period, key in tasks:
                required_receipt_keys.add(key)
                required = set(master.loc[listing.le(pd.Timestamp(period)) &
                                          (delisting.isna() | delisting.ge(pd.Timestamp(period))), "ts_code"])
                receipt = state["completed"].get(key)
                if receipt and receipt.get("capture_day") == capture_day and required.issubset(set(receipt.get("codes", []))):
                    continue
                state["completed"].pop(key, None)
                scope = {"dataset": dataset, "report_period": str(pd.Timestamp(period).date())}
                checkpoint = pagination.get(key, {})
                if (checkpoint.get("capture_day") != capture_day or checkpoint.get("finished")
                        or not validate_receipt(database, checkpoint)):
                    checkpoint = {"capture_day": capture_day, "scope": scope, "database_identity": epoch,
                                  "offset": 0, "codes": [], "rows": 0, "page_sha256": [], "finished": False}
                pagination[key] = checkpoint
                while not checkpoint["finished"]:
                    if checkpoint["offset"] >= 250000:
                        raise ValueError(f"Financial pagination incomplete: {key}")
                    chunk = fetch(dataset, period=period, limit=5000, offset=checkpoint["offset"])
                    field = FINANCIAL_FIELDS[dataset]
                    if chunk is None or not {"ts_code", "end_date", "ann_date", field}.issubset(chunk):
                        raise ValueError(f"Invalid financial schema: {key}")
                    if chunk.empty:
                        if checkpoint["offset"] == 0:
                            raise ValueError(f"Empty financial response is retryable: {key}")
                        checkpoint["finished"] = True
                        break
                    page_hash = hashlib.sha256(json.dumps(sorted(
                        json.dumps(row, sort_keys=True, default=str) for row in chunk.to_dict("records")),
                        ensure_ascii=False).encode()).hexdigest()
                    if page_hash in checkpoint["page_sha256"]:
                        raise ValueError(f"Financial pagination repeated a page: {key}")
                    if not chunk["ts_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
                        raise ValueError(f"Invalid financial stock code: {key}")
                    reports = pd.to_datetime(chunk["end_date"].astype(str), errors="coerce")
                    if not reports.eq(pd.Timestamp(period)).all():
                        raise ValueError(f"Wrong financial period returned: {key}")
                    announcement = pd.to_datetime(chunk["ann_date"].astype(str), errors="coerce")
                    if announcement.isna().any():
                        raise ValueError(f"Invalid financial announcement date: {key}")
                    if "f_ann_date" in chunk:
                        raw_final = chunk["f_ann_date"].fillna("").astype(str).str.strip()
                        final_date = pd.to_datetime(raw_final, errors="coerce")
                        if (raw_final.ne("") & final_date.isna()).any():
                            raise ValueError(f"Invalid final announcement date: {key}")
                        announcement = pd.concat([announcement, final_date], axis=1).max(axis=1)
                    if announcement.lt(reports).any():
                        raise ValueError(f"Financial announcement precedes report period: {key}")
                    visible = chunk[announcement.le(end)].copy()
                    if not np.isfinite(pd.to_numeric(visible[field], errors="coerce")).all():
                        raise ValueError(f"Invalid financial value {field}: {key}")
                    store(dataset, visible, {"period": period, "as_of": str(end.date())})
                    checkpoint.update(offset=checkpoint["offset"] + 5000,
                                      rows=checkpoint["rows"] + len(visible),
                                      codes=sorted(set(checkpoint["codes"]) | set(visible["ts_code"])),
                                      page_sha256=checkpoint["page_sha256"] + [page_hash],
                                      database_fingerprint=_scope_fingerprint(database, scope),
                                      finished=len(chunk) < 5000, captured_at=utc_now())
                    write_json(state_path, state)
                covered = set(checkpoint["codes"])
                if not required or not required.issubset(covered):
                    incomplete(key, required, covered, reason="financial_universe_coverage_unverified")
                else:
                    complete(key, scope, checkpoint["rows"], codes=sorted(covered), capture_day=capture_day,
                             coverage_basis="listed_by_report_period_with_visible_announcements")
        # New captures can supersede an older scope (e.g. a financial revision).
        # Do not leave old receipts advertising hashes that no longer exist.
        for key, receipt in list(state["completed"].items()):
            if not validate_receipt(database, receipt):
                state["completed"].pop(key)
                state["invalidated_receipts"].append(key)
                if key in required_receipt_keys:
                    state["incomplete"][key] = {"reason": "database_changed_during_capture"}
        state["status"] = "data_limited" if state["incomplete"] else "completed"
        state["latest_session"] = sessions[-1] if sessions else None
        state["financial_capture_requested"] = capture_financials
    except Exception as exc:
        state["status"] = "failed"
        state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    state["finished_at"] = utc_now()
    state["database_fingerprint"] = {
        "identity": epoch,
        "completed_receipts_sha256": hashlib.sha256(json.dumps({
            key: value["database_fingerprint"] for key, value in sorted(state["completed"].items())
        }, sort_keys=True).encode()).hexdigest(),
        "method": "SQLite epoch plus per-scope row fingerprints; not a hash of the entire database file",
    }
    write_json(state_path, state)
    return state
