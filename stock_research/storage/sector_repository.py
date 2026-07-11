"""DuckDB persistence for sector boards and industry K-lines."""
from __future__ import annotations

import json
from threading import RLock
from typing import Any, Optional

import pandas as pd

from .database import Database


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _first_value(row, names: tuple[str, ...]) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(converted) else float(converted)


def _board_rows(
    frame: pd.DataFrame,
    source: str,
) -> list[tuple[str, Optional[str], Optional[str], str]]:
    if frame is None or frame.empty:
        return []
    name_columns = ("board_name", "board", "name", "板块名称", "名称")
    if not _first_existing(frame, name_columns):
        return []

    rows_by_name = {}
    for _, row in frame.iterrows():
        board_name = str(
            _first_value(row, name_columns) or ""
        ).strip()
        if not board_name:
            continue
        group_name = str(
            _first_value(row, ("group_name", "group")) or ""
        ).strip()
        board_code = str(
            _first_value(row, ("board_code", "code", "板块代码")) or ""
        ).strip()
        rows_by_name[board_name] = (
            board_name,
            group_name or None,
            board_code or None,
            str(source),
        )
    return list(rows_by_name.values())


def _source_filter(*, source: Optional[str], source_prefix: Optional[str]):
    if source is not None and source_prefix is not None:
        raise ValueError("source and source_prefix are mutually exclusive")
    if source is not None:
        return "source = ?", [str(source)]
    if source_prefix is not None:
        return "starts_with(source, ?)", [str(source_prefix)]
    return None, []


class SectorRepository:
    """Persist sector API results immediately and read them as a warm cache."""

    def __init__(self, database: Database):
        self.database = database
        self._lock = RLock()

    def upsert_boards(self, frame: pd.DataFrame, *, source: str) -> int:
        rows = _board_rows(frame, source)
        if not rows:
            return 0

        with self._lock:
            connection = self.database.connect()
            try:
                connection.execute("BEGIN TRANSACTION")
                for board_name, group_name, board_code, row_source in rows:
                    connection.execute(
                        "DELETE FROM raw.sector_boards WHERE board_name = ?",
                        [board_name],
                    )
                    connection.execute(
                        """
                        INSERT INTO raw.sector_boards (
                            board_name, group_name, board_code, source, updated_at
                        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        [board_name, group_name, board_code, row_source],
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        return len({item[0] for item in rows})

    def replace_boards(self, frame: pd.DataFrame, *, source: str) -> int:
        """Replace the complete active-board snapshot in one transaction."""
        rows = _board_rows(frame, source)
        if not rows:
            return 0

        with self._lock:
            connection = self.database.connect()
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute("DELETE FROM raw.sector_boards")
                for board_name, group_name, board_code, row_source in rows:
                    connection.execute(
                        """
                        INSERT INTO raw.sector_boards (
                            board_name, group_name, board_code, source, updated_at
                        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        [board_name, group_name, board_code, row_source],
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        return len(rows)

    def load_boards(
        self,
        *,
        max_age=None,
        source: Optional[str] = None,
        source_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        source_clause, parameters = _source_filter(
            source=source,
            source_prefix=source_prefix,
        )
        where_clause = f"WHERE {source_clause}" if source_clause else ""
        with self._lock:
            connection = self.database.connect()
            try:
                frame = connection.execute(
                    f"""
                    SELECT board_name, group_name, board_code, source, updated_at
                    FROM raw.sector_boards
                    {where_clause}
                    ORDER BY board_name
                    """,
                    parameters,
                ).fetchdf()
            finally:
                connection.close()
        if frame.empty or max_age is None:
            return frame
        updated_at = pd.to_datetime(frame["updated_at"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(max_age)
        if not (updated_at.notna() & (updated_at >= cutoff)).all():
            return frame.iloc[0:0].copy()
        return frame.reset_index(drop=True)

    def upsert_board_history(
        self,
        board_name: str,
        frame: pd.DataFrame,
        *,
        source: str,
    ) -> int:
        if frame is None or frame.empty:
            return 0
        date_col = _first_existing(frame, ("trade_date", "date", "日期"))
        if not date_col:
            return 0
        columns = {
            "open": _first_existing(frame, ("open", "开盘")),
            "close": _first_existing(frame, ("close", "收盘")),
            "high": _first_existing(frame, ("high", "最高")),
            "low": _first_existing(frame, ("low", "最低")),
            "amount": _first_existing(frame, ("amount", "成交额")),
            "volume": _first_existing(frame, ("volume", "成交量")),
            "pct_chg": _first_existing(frame, ("pct_chg", "涨跌幅")),
        }
        rows = []
        for _, row in frame.iterrows():
            trade_date = pd.to_datetime(row.get(date_col), errors="coerce")
            if pd.isna(trade_date):
                continue
            pct_chg = _number(row.get(columns["pct_chg"])) if columns["pct_chg"] else None
            if columns["pct_chg"] == "涨跌幅" and pct_chg is not None:
                pct_chg /= 100.0
            rows.append(
                (
                    str(board_name),
                    trade_date.date(),
                    _number(row.get(columns["open"])) if columns["open"] else None,
                    _number(row.get(columns["close"])) if columns["close"] else None,
                    _number(row.get(columns["high"])) if columns["high"] else None,
                    _number(row.get(columns["low"])) if columns["low"] else None,
                    _number(row.get(columns["amount"])) if columns["amount"] else None,
                    _number(row.get(columns["volume"])) if columns["volume"] else None,
                    pct_chg,
                    source,
                )
            )
        if not rows:
            return 0

        with self._lock:
            connection = self.database.connect()
            try:
                connection.execute("BEGIN TRANSACTION")
                for row in rows:
                    connection.execute(
                        """
                        DELETE FROM raw.sector_board_history
                        WHERE board_name = ? AND trade_date = ? AND source = ?
                        """,
                        [row[0], row[1], row[9]],
                    )
                    connection.execute(
                        """
                        INSERT INTO raw.sector_board_history (
                            board_name, trade_date, open, close, high, low,
                            amount, volume, pct_chg, source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        list(row),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        return len({item[1] for item in rows})

    def load_board_history(
        self,
        board_name: str,
        *,
        end_date: str | pd.Timestamp,
        days: int,
        date_column: str = "date",
        source: Optional[str] = None,
        source_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(end):
            return pd.DataFrame()
        source_clause, source_parameters = _source_filter(
            source=source,
            source_prefix=source_prefix,
        )
        source_sql = f" AND {source_clause}" if source_clause else ""
        with self._lock:
            connection = self.database.connect()
            try:
                frame = connection.execute(
                    f"""
                    SELECT trade_date, open, close, high, low, amount, volume,
                           pct_chg, source, updated_at
                    FROM raw.sector_board_history
                    WHERE board_name = ? AND trade_date <= ?{source_sql}
                    ORDER BY trade_date, source
                    """,
                    [str(board_name), end.date(), *source_parameters],
                ).fetchdf()
            finally:
                connection.close()
        if frame.empty:
            return frame
        frame = frame.tail(max(1, int(days))).reset_index(drop=True)
        frame[date_column] = pd.to_datetime(frame.pop("trade_date"), errors="coerce")
        ordered = [
            date_column,
            "open",
            "close",
            "high",
            "low",
            "amount",
            "volume",
            "pct_chg",
            "source",
            "updated_at",
        ]
        return frame[[column for column in ordered if column in frame.columns]]

    def log_event(
        self,
        *,
        step_name: str,
        part_name: str,
        event_type: str,
        status: str,
        message: str = "",
        rows: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        context: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            connection = self.database.connect()
            try:
                connection.execute(
                    """
                    INSERT INTO ops.pipeline_events (
                        run_id, step_name, part_name, event_type, status,
                        message, rows, elapsed_seconds, context_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id,
                        step_name,
                        part_name,
                        event_type,
                        status,
                        message,
                        rows,
                        elapsed_seconds,
                        json.dumps(context or {}, ensure_ascii=False, sort_keys=True),
                    ],
                )
            finally:
                connection.close()
