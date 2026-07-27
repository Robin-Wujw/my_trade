"""SQLite cache repository for Tushare Pro datasets."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pandas as pd

from .database import Database


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _date_value(row: dict, names: Iterable[str]):
    for name in names:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        converted = pd.to_datetime(str(value), errors="coerce")
        if pd.notna(converted):
            return converted.date()
    return None


def _row_key(dataset: str, row: dict, key_columns: tuple[str, ...]) -> str:
    values = [str(row.get(column) or "") for column in key_columns]
    if any(values):
        return "|".join([dataset, *values])
    digest = hashlib.sha256(_json(row).encode("utf-8")).hexdigest()
    return f"{dataset}|sha256:{digest}"


def normalize_tushare_code(code: str) -> str:
    """Normalize common local A-share code formats to Tushare ts_code."""
    text = str(code or "").strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ", "BJ"}:
            return f"{right.zfill(6)}.{left}"
        return f"{left.zfill(6)}.{right}"
    symbol = text.zfill(6)
    if symbol.startswith(("6", "9")):
        suffix = "SH"
    elif symbol.startswith(("4", "8")):
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{symbol}.{suffix}"


class TushareRepository:
    """Persist arbitrary Tushare tables with searchable point-in-time columns."""

    def __init__(self, database: Database):
        self.database = database

    def upsert_dataset(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        source: str = "tushare",
        key_columns: tuple[str, ...] = (
            "ts_code",
            "trade_date",
            "cal_date",
            "ann_date",
            "end_date",
            "f_ann_date",
            "index_code",
            "con_code",
            "exchange",
            "symbol",
            "broker",
            "warehouse",
            "week",
            "week_date",
            "prd",
            "l1_code",
            "l2_code",
            "l3_code",
            "industry_code",
            "in_date",
            "out_date",
            "opt_code",
            "mapping_ts_code",
        ),
        params: dict | None = None,
    ) -> int:
        if frame is None or frame.empty:
            return 0
        rows = []
        for raw_row in frame.to_dict("records"):
            row = {str(key): value for key, value in raw_row.items()}
            rows.append({
                "dataset": str(dataset),
                "row_key": _row_key(str(dataset), row, key_columns),
                "ts_code": row.get("ts_code"),
                "trade_date": _date_value(row, ("trade_date", "cal_date", "week_date")),
                "report_period": _date_value(row, ("end_date", "period", "report_period")),
                "ann_date": _date_value(row, ("ann_date", "f_ann_date", "pub_date")),
                "end_date": _date_value(row, ("end_date", "delist_date", "maturity_date")),
                "source": str(source),
                "payload_json": _json(row),
            })
        data = pd.DataFrame(rows).drop_duplicates(["dataset", "row_key"], keep="last")
        connection = self.database.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            columns = tuple(data.columns)
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"""
                INSERT OR REPLACE INTO raw.tushare_dataset_rows (
                    {", ".join(columns)}
                ) VALUES ({placeholders})
                """,
                (
                    [record[column] for column in columns]
                    for record in data.to_dict("records")
                ),
            )
            connection.execute(
                """
                DELETE FROM raw.tushare_sync_state WHERE dataset = ?
                """,
                [str(dataset)],
            )
            connection.execute(
                """
                INSERT INTO raw.tushare_sync_state (
                    dataset, source, last_params_json, last_row_count, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    str(dataset),
                    str(source),
                    _json(params or {}),
                    int(len(data)),
                    _json({"dataset": dataset, "params": params or {}, "rows": int(len(data))}),
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            connection.close()
        return int(len(data))

    def load_dataset(
        self,
        dataset: str,
        *,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        clauses = ["dataset = ?"]
        params = [str(dataset)]
        if ts_code:
            clauses.append("ts_code = ?")
            params.append(str(ts_code))
        if start_date:
            clauses.append("COALESCE(trade_date, ann_date, end_date, report_period) >= ?")
            params.append(str(start_date))
        if end_date:
            clauses.append("COALESCE(trade_date, ann_date, end_date, report_period) <= ?")
            params.append(str(end_date))
        connection = self.database.connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT payload_json
                FROM raw.tushare_dataset_rows
                WHERE {" AND ".join(clauses)}
                ORDER BY COALESCE(trade_date, ann_date, end_date, report_period), ts_code, row_key
                """,
                params,
            ).fetchall()
        finally:
            connection.close()
        return pd.DataFrame([json.loads(row[0]) for row in rows])

    def load_daily_kline_frames(
        self,
        codes,
        *,
        start_date: str,
        end_date: str,
        dataset: str = "daily_kline",
    ) -> pd.DataFrame:
        """Load Tushare daily OHLCV rows for a backtest without future bars."""
        requested = [str(code) for code in codes if str(code).strip()]
        ts_to_requested: dict[str, list[str]] = {}
        for code in requested:
            ts_code = normalize_tushare_code(code)
            if ts_code:
                ts_to_requested.setdefault(ts_code, []).append(code)
        if not ts_to_requested:
            return pd.DataFrame(columns=[
                "date", "code", "open", "high", "low", "close", "volume",
                "amount", "tradestatus", "raw_to_qfq_factor",
            ])
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid Tushare K-line date range: {start_date}..{end_date}")

        placeholders = ", ".join("?" for _ in ts_to_requested)
        params = [
            str(dataset),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            *sorted(ts_to_requested),
        ]
        connection = self.database.connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT ts_code, trade_date, payload_json
                FROM raw.tushare_dataset_rows
                WHERE dataset = ?
                  AND trade_date >= ?
                  AND trade_date <= ?
                  AND ts_code IN ({placeholders})
                ORDER BY ts_code, trade_date
                """,
                params,
            ).fetchall()
            factor_rows = connection.execute(
                f"""
                SELECT ts_code, trade_date, payload_json
                FROM raw.tushare_dataset_rows
                WHERE dataset = ?
                  AND trade_date >= ?
                  AND trade_date <= ?
                  AND ts_code IN ({placeholders})
                ORDER BY ts_code, trade_date
                """,
                ["adj_factor", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), *sorted(ts_to_requested)],
            ).fetchall()
        finally:
            connection.close()

        raw_to_qfq_by_key = {}
        for ts_code, trade_date, payload_json in factor_rows:
            payload = json.loads(payload_json)
            factor = pd.to_numeric(payload.get("adj_factor"), errors="coerce")
            if pd.notna(factor) and float(factor) > 0:
                raw_to_qfq_by_key[(str(ts_code), str(trade_date))] = 1.0 / float(factor)

        records = []
        for ts_code, trade_date, payload_json in rows:
            payload = json.loads(payload_json)
            for requested_code in ts_to_requested.get(str(ts_code), []):
                records.append({
                    "date": pd.to_datetime(trade_date, errors="coerce").strftime("%Y-%m-%d"),
                    "code": requested_code,
                    "open": payload.get("open"),
                    "high": payload.get("high"),
                    "low": payload.get("low"),
                    "close": payload.get("close"),
                    "volume": payload.get("vol", payload.get("volume")),
                    "amount": payload.get("amount"),
                    "tradestatus": payload.get("tradestatus", "1"),
                    "raw_to_qfq_factor": raw_to_qfq_by_key.get((str(ts_code), str(trade_date))),
                })
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=[
                "date", "code", "open", "high", "low", "close", "volume",
                "amount", "tradestatus", "raw_to_qfq_factor",
            ])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[(frame["date"] >= start.normalize()) & (frame["date"] <= end.normalize())]
        for column in ("open", "high", "low", "close", "volume", "amount", "raw_to_qfq_factor"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=["date", "high", "low", "close"])
            .drop_duplicates(["code", "date"], keep="last")
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame[[
            "date", "code", "open", "high", "low", "close", "volume",
            "amount", "tradestatus", "raw_to_qfq_factor",
        ]]

    def load_dividend_actions(
        self,
        codes,
        *,
        start_date: str,
        end_date: str,
    ) -> dict[str, list[dict]]:
        """Load local Tushare dividend/ex-rights rows keyed by project code."""
        requested = [str(code) for code in codes if str(code).strip()]
        ts_to_requested: dict[str, list[str]] = {}
        for code in requested:
            ts_code = normalize_tushare_code(code)
            if ts_code:
                ts_to_requested.setdefault(ts_code, []).append(code)
        if not ts_to_requested:
            return {}
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid Tushare dividend date range: {start_date}..{end_date}")

        placeholders = ", ".join("?" for _ in ts_to_requested)
        connection = self.database.connect(read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT ts_code, payload_json
                FROM raw.tushare_dataset_rows
                WHERE dataset = ?
                  AND ts_code IN ({placeholders})
                ORDER BY ts_code, row_key
                """,
                ["dividend", *sorted(ts_to_requested)],
            ).fetchall()
        finally:
            connection.close()

        actions: dict[str, list[dict]] = {}
        for ts_code, payload_json in rows:
            payload = json.loads(payload_json)
            ex_date = pd.to_datetime(
                payload.get("ex_date") or payload.get("div_listdate"),
                errors="coerce",
            )
            if pd.isna(ex_date):
                continue
            ex_date = pd.Timestamp(ex_date).normalize()
            if ex_date < start.normalize() or ex_date > end.normalize():
                continue
            for requested_code in ts_to_requested.get(str(ts_code), []):
                action = dict(payload)
                action["code"] = requested_code
                actions.setdefault(requested_code, []).append(action)
        return actions
