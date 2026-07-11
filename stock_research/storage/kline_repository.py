"""DuckDB persistence for stock daily K-lines."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
import time
from typing import Optional

import duckdb
import pandas as pd

from stock_research.core.paths import PATHS

from .database import Database

try:  # Windows production path.
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


def _lock_file(handle, *, max_attempts: int = 120) -> None:
    if msvcrt is None:
        return
    for attempt in range(max_attempts):
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError:
            if attempt >= max_attempts - 1:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1.0))


@contextmanager
def _process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    locked = False
    try:
        if msvcrt is not None:
            _lock_file(handle)
            locked = True
        yield
    finally:
        if msvcrt is not None and locked:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


_DUCKDB_LOCK_ERROR_MARKERS = (
    "could not set lock",
    "conflicting lock",
    "database is locked",
    "database is busy",
    "lock on file",
    "used by another process",
    "being used by another process",
)


def _is_transient_duckdb_lock_error(exc: Exception) -> bool:
    if not isinstance(exc, duckdb.IOException):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _DUCKDB_LOCK_ERROR_MARKERS)


class KlineRepository:
    """Persist stock K-lines with process-safe writes for multiprocessing runs."""

    def __init__(
        self,
        database: Database,
        *,
        lock_path: Optional[Path] = None,
        lock_retry_attempts: int = 5,
        lock_retry_delay: float = 0.05,
    ):
        self.database = database
        self.lock_path = Path(lock_path or PATHS.tmp / "duckdb_stock_kline.lock")
        self.lock_retry_attempts = max(1, int(lock_retry_attempts))
        self.lock_retry_delay = max(0.0, float(lock_retry_delay))
        self._thread_lock = RLock()

    def _with_lock_retry(self, operation):
        for attempt in range(self.lock_retry_attempts):
            try:
                with self._thread_lock:
                    with _process_lock(self.lock_path):
                        return operation()
            except Exception as exc:
                if (
                    not _is_transient_duckdb_lock_error(exc)
                    or attempt >= self.lock_retry_attempts - 1
                ):
                    raise
                delay = min(self.lock_retry_delay * (attempt + 1), 0.5)
                if delay:
                    time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    def upsert_stock_kline(self, source: str, code: str, frame: pd.DataFrame) -> int:
        if frame is None or frame.empty:
            return 0
        data = frame.copy()
        if "date" not in data.columns:
            return 0
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            else:
                data[column] = pd.NA
        if "tradestatus" not in data.columns:
            data["tradestatus"] = pd.NA
        rows = []
        for _, row in data.dropna(subset=["date", "high", "low", "close"]).iterrows():
            rows.append(
                [
                    str(source),
                    str(code),
                    row["date"].date(),
                    None if pd.isna(row["open"]) else float(row["open"]),
                    None if pd.isna(row["high"]) else float(row["high"]),
                    None if pd.isna(row["low"]) else float(row["low"]),
                    None if pd.isna(row["close"]) else float(row["close"]),
                    None if pd.isna(row["volume"]) else float(row["volume"]),
                    None if pd.isna(row["tradestatus"]) else str(row["tradestatus"]),
                ]
            )
        if not rows:
            return 0

        def write_rows():
            connection = None
            transaction_started = False
            try:
                connection = self.database.connect()
                connection.execute("BEGIN TRANSACTION")
                transaction_started = True
                for row in rows:
                    connection.execute(
                        """
                        DELETE FROM raw.stock_kline_daily
                        WHERE source = ? AND code = ? AND trade_date = ?
                        """,
                        row[:3],
                    )
                    connection.execute(
                        """
                        INSERT INTO raw.stock_kline_daily (
                            source, code, trade_date, open, high, low,
                            close, volume, tradestatus, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        row,
                    )
                connection.execute("COMMIT")
                transaction_started = False
            except Exception:
                if connection is not None and transaction_started:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                raise
            finally:
                if connection is not None:
                    connection.close()

        self._with_lock_retry(write_rows)
        return len({row[2] for row in rows})

    def replace_stock_kline_range(
        self,
        source: str,
        code: str,
        frame: pd.DataFrame,
        *,
        start_date: str,
        end_date: str,
    ) -> int:
        """Atomically replace one adjusted-price window for a stock."""
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid K-line replace range: {start_date}..{end_date}")

        data = frame.copy() if frame is not None else pd.DataFrame()
        if not data.empty and "date" not in data.columns:
            raise ValueError("K-line replacement frame has no date column")
        if not data.empty:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data[(data["date"] >= start) & (data["date"] <= end)]
            for column in ["open", "high", "low", "close", "volume"]:
                if column in data.columns:
                    data[column] = pd.to_numeric(data[column], errors="coerce")
                else:
                    data[column] = pd.NA
            if "tradestatus" not in data.columns:
                data["tradestatus"] = pd.NA

        rows = []
        for _, row in data.dropna(
            subset=["date", "high", "low", "close"]
        ).iterrows():
            rows.append(
                [
                    str(source),
                    str(code),
                    row["date"].date(),
                    None if pd.isna(row["open"]) else float(row["open"]),
                    None if pd.isna(row["high"]) else float(row["high"]),
                    None if pd.isna(row["low"]) else float(row["low"]),
                    None if pd.isna(row["close"]) else float(row["close"]),
                    None if pd.isna(row["volume"]) else float(row["volume"]),
                    None if pd.isna(row["tradestatus"]) else str(row["tradestatus"]),
                ]
            )

        def replace_rows():
            connection = None
            transaction_started = False
            try:
                connection = self.database.connect()
                connection.execute("BEGIN TRANSACTION")
                transaction_started = True
                connection.execute(
                    """
                    DELETE FROM raw.stock_kline_daily
                    WHERE source = ? AND code = ?
                      AND trade_date >= ? AND trade_date <= ?
                    """,
                    [str(source), str(code), start.date(), end.date()],
                )
                for row in rows:
                    connection.execute(
                        """
                        INSERT INTO raw.stock_kline_daily (
                            source, code, trade_date, open, high, low,
                            close, volume, tradestatus, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        row,
                    )
                connection.execute("COMMIT")
                transaction_started = False
            except Exception:
                if connection is not None and transaction_started:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                raise
            finally:
                if connection is not None:
                    connection.close()

        self._with_lock_retry(replace_rows)
        return len({row[2] for row in rows})

    def load_stock_kline(
        self,
        source: str,
        code: str,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        def read_rows():
            connection = self.database.connect(read_only=True)
            try:
                return connection.execute(
                    """
                    SELECT trade_date, code, open, high, low, close, volume, tradestatus
                    FROM raw.stock_kline_daily
                    WHERE source = ? AND code = ? AND trade_date >= ? AND trade_date <= ?
                    ORDER BY trade_date
                    """,
                    [str(source), str(code), start_date, end_date],
                ).fetchdf()
            finally:
                connection.close()

        frame = self._with_lock_retry(read_rows)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame.pop("trade_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        return frame[["date", "code", "open", "high", "low", "close", "volume", "tradestatus"]]
