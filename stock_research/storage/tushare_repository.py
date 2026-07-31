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


DEFAULT_KEY_COLUMNS = (
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
)

DATASET_KEY_COLUMNS = {
    "dividend": (
        "ts_code",
        "ann_date",
        "end_date",
        "record_date",
        "ex_date",
        "div_listdate",
        "div_proc",
        "stk_div",
        "stk_bo_rate",
        "stk_co_rate",
        "cash_div",
        "cash_div_tax",
    ),
    "share_float": (
        "ts_code",
        "ann_date",
        "float_date",
        "float_share",
        "float_ratio",
        "holder_name",
        "share_type",
    ),
}


def dataset_key_columns(dataset: str) -> tuple[str, ...]:
    return DATASET_KEY_COLUMNS.get(str(dataset), DEFAULT_KEY_COLUMNS)


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


def tushare_to_project_code(code: str) -> str:
    """Convert a Tushare ts_code to the repository's exchange-first format."""
    text = str(code or "").strip().upper()
    if "." not in text:
        return text
    symbol, exchange = text.split(".", 1)
    if exchange not in {"SH", "SZ", "BJ"}:
        return text
    return f"{exchange.lower()}.{symbol.zfill(6)}"


def _chunks(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index:index + size]


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
        key_columns: tuple[str, ...] | None = None,
        params: dict | None = None,
    ) -> int:
        if frame is None or frame.empty:
            return 0
        key_columns = key_columns or dataset_key_columns(dataset)
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

    def load_dataset_for_codes(
        self,
        dataset: str,
        codes,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load indexed Tushare rows for several symbols without a table scan."""
        requested = [str(code) for code in codes if str(code).strip()]
        ts_to_requested: dict[str, list[str]] = {}
        for code in requested:
            ts_code = normalize_tushare_code(code)
            if ts_code:
                ts_to_requested.setdefault(ts_code, []).append(code)
        if not ts_to_requested:
            return pd.DataFrame()
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid Tushare date range: {start_date}..{end_date}")

        rows = []
        connection = self.database.connect(read_only=True)
        try:
            for chunk in _chunks(sorted(ts_to_requested), 80):
                placeholders = ", ".join("?" for _ in chunk)
                rows.extend(connection.execute(
                    f"""
                    SELECT ts_code, trade_date, payload_json
                    FROM raw.tushare_dataset_rows INDEXED BY idx_tushare_dataset_ts_code
                    WHERE dataset = ?
                      AND ts_code IN ({placeholders})
                      AND trade_date >= ?
                      AND trade_date <= ?
                    ORDER BY ts_code, trade_date, row_key
                    """,
                    [
                        str(dataset),
                        *chunk,
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    ],
                ).fetchall())
        finally:
            connection.close()

        records = []
        for ts_code, trade_date, payload_json in rows:
            payload = json.loads(payload_json)
            for requested_code in ts_to_requested.get(str(ts_code), []):
                record = dict(payload)
                record["code"] = requested_code
                record["date"] = pd.to_datetime(trade_date, errors="coerce")
                records.append(record)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records)
        return (
            frame.dropna(subset=["date", "code"])
            .drop_duplicates(["date", "code"], keep="last")
            .sort_values(["date", "code"])
            .reset_index(drop=True)
        )

    def load_market_daily_frame(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load a full-market daily slice through the indexed trade-date path.

        The generic payload table can contain more than one ``daily_basic`` row
        for a symbol/date when a later sync requested a wider field list.  The
        most complete row wins before it is joined to prices.
        """
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid Tushare market date range: {start_date}..{end_date}")
        start_text = start.strftime("%Y-%m-%d")
        end_text = end.strftime("%Y-%m-%d")

        connection = self.database.connect(read_only=True)
        try:
            prices = connection.execute(
                """
                SELECT
                    ts_code,
                    trade_date,
                    json_extract(payload_json, '$.open'),
                    json_extract(payload_json, '$.high'),
                    json_extract(payload_json, '$.low'),
                    json_extract(payload_json, '$.close'),
                    COALESCE(
                        json_extract(payload_json, '$.vol'),
                        json_extract(payload_json, '$.volume')
                    ),
                    json_extract(payload_json, '$.amount')
                FROM raw.tushare_dataset_rows
                     INDEXED BY idx_tushare_dataset_trade_date
                WHERE dataset = 'daily_kline'
                  AND trade_date >= ?
                  AND trade_date <= ?
                ORDER BY trade_date, ts_code, row_key
                """,
                [start_text, end_text],
            ).fetchall()
            columns = [
                "ts_code", "date", "open", "high", "low", "close", "volume", "amount",
            ]
            frame = pd.DataFrame(prices, columns=columns)
            del prices
            factors = connection.execute(
                """
                SELECT
                    ts_code,
                    trade_date,
                    json_extract(payload_json, '$.adj_factor')
                FROM raw.tushare_dataset_rows
                     INDEXED BY idx_tushare_dataset_trade_date
                WHERE dataset = 'adj_factor'
                  AND trade_date >= ?
                  AND trade_date <= ?
                ORDER BY trade_date, ts_code, row_key
                """,
                [start_text, end_text],
            ).fetchall()
            adjustment = pd.DataFrame(
                factors, columns=["ts_code", "date", "adj_factor"],
            ).drop_duplicates(["ts_code", "date"], keep="last")
            del factors
            daily_basic = connection.execute(
                """
                SELECT
                    ts_code,
                    trade_date,
                    json_extract(payload_json, '$.turnover_rate'),
                    json_extract(payload_json, '$.turnover_rate_f'),
                    json_extract(payload_json, '$.total_mv'),
                    json_extract(payload_json, '$.circ_mv'),
                    json_extract(payload_json, '$.pe_ttm'),
                    json_extract(payload_json, '$.pb'),
                    length(payload_json)
                FROM raw.tushare_dataset_rows
                     INDEXED BY idx_tushare_dataset_trade_date
                WHERE dataset = 'daily_basic'
                  AND trade_date >= ?
                  AND trade_date <= ?
                ORDER BY trade_date, ts_code, row_key
                """,
                [start_text, end_text],
            ).fetchall()
        finally:
            connection.close()

        if frame.empty:
            return pd.DataFrame(columns=[
                "date", "code", "open", "high", "low", "close", "volume",
                "amount", "tradestatus", "raw_to_qfq_factor", "turnover_rate",
                "turnover_rate_f", "total_mv", "circ_mv", "pe_ttm", "pb",
            ])

        basic = pd.DataFrame(
            daily_basic,
            columns=[
                "ts_code", "date", "turnover_rate", "turnover_rate_f",
                "total_mv", "circ_mv", "pe_ttm", "pb", "_payload_size",
            ],
        )
        del daily_basic
        if not basic.empty:
            basic["_non_null_fields"] = basic[
                [
                    "turnover_rate", "turnover_rate_f", "total_mv",
                    "circ_mv", "pe_ttm", "pb",
                ]
            ].notna().sum(axis=1)
            basic = (
                basic.sort_values(
                    ["ts_code", "date", "_non_null_fields", "_payload_size"],
                    kind="mergesort",
                )
                .drop_duplicates(["ts_code", "date"], keep="last")
                .drop(columns=["_non_null_fields", "_payload_size"])
            )

        frame = frame.merge(adjustment, on=["ts_code", "date"], how="left")
        if not basic.empty:
            frame = frame.merge(basic, on=["ts_code", "date"], how="left")
        else:
            for column in (
                "turnover_rate", "turnover_rate_f", "total_mv",
                "circ_mv", "pe_ttm", "pb",
            ):
                frame[column] = pd.NA
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["code"] = frame["ts_code"].map(tushare_to_project_code)
        numeric = [
            "open", "high", "low", "close", "volume", "amount", "adj_factor",
            "turnover_rate", "turnover_rate_f", "total_mv", "circ_mv",
            "pe_ttm", "pb",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["raw_to_qfq_factor"] = 1.0 / frame["adj_factor"].where(
            frame["adj_factor"] > 0,
        )
        frame["tradestatus"] = (
            frame["volume"].fillna(0).gt(0)
            & frame["close"].fillna(0).gt(0)
        ).astype(int).astype(str)
        return (
            frame.dropna(subset=["date", "code", "open", "high", "low", "close"])
            .drop_duplicates(["date", "code"], keep="last")
            .sort_values(["date", "code"])
            .reset_index(drop=True)[[
                "date", "code", "open", "high", "low", "close", "volume",
                "amount", "tradestatus", "raw_to_qfq_factor", "turnover_rate",
                "turnover_rate_f", "total_mv", "circ_mv", "pe_ttm", "pb",
            ]]
        )

    def load_stock_basic_universe(self) -> pd.DataFrame:
        """Load the broad A-share security master without current-status filters."""
        connection = self.database.connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM raw.tushare_dataset_rows
                WHERE dataset = 'basic'
                ORDER BY ts_code, row_key
                """
            ).fetchall()
        finally:
            connection.close()
        records = [json.loads(row[0]) for row in rows]
        if not records:
            return pd.DataFrame(columns=[
                "code", "ts_code", "name", "industry", "list_date",
                "delist_date", "list_status", "market",
            ])
        frame = pd.DataFrame(records)
        frame["code"] = frame.get("ts_code", pd.Series(dtype=str)).map(
            tushare_to_project_code,
        )
        for column in ("list_date", "delist_date"):
            frame[column] = pd.to_datetime(frame.get(column), errors="coerce")
        wanted = [
            "code", "ts_code", "name", "industry", "list_date",
            "delist_date", "list_status", "market",
        ]
        for column in wanted:
            if column not in frame:
                frame[column] = pd.NA
        return (
            frame[wanted]
            .dropna(subset=["code", "list_date"])
            .drop_duplicates("code", keep="last")
            .sort_values("code")
            .reset_index(drop=True)
        )

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

        connection = self.database.connect(read_only=True)
        try:
            rows = []
            factor_rows = []
            for chunk in _chunks(sorted(ts_to_requested), 80):
                placeholders = ", ".join("?" for _ in chunk)
                rows.extend(connection.execute(
                    f"""
                    SELECT ts_code, trade_date, payload_json
                    FROM raw.tushare_dataset_rows INDEXED BY idx_tushare_dataset_ts_code
                    WHERE dataset = ?
                      AND ts_code IN ({placeholders})
                      AND trade_date >= ?
                      AND trade_date <= ?
                    ORDER BY ts_code, trade_date
                    """,
                    [
                        str(dataset),
                        *chunk,
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    ],
                ).fetchall())
                factor_rows.extend(connection.execute(
                    f"""
                    SELECT ts_code, trade_date, payload_json
                    FROM raw.tushare_dataset_rows INDEXED BY idx_tushare_dataset_ts_code
                    WHERE dataset = ?
                      AND ts_code IN ({placeholders})
                      AND trade_date >= ?
                      AND trade_date <= ?
                    ORDER BY ts_code, trade_date
                    """,
                    [
                        "adj_factor",
                        *chunk,
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    ],
                ).fetchall())
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
