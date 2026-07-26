"""Explicit SQLite connection and initialization boundary."""
from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from datetime import date
from typing import Any, Iterable, Union

import pandas as pd

from stock_research.core.paths import PATHS

from .migrations import apply_migrations


_SCHEMA_PREFIX = re.compile(r"\b(raw|core|derived|ops)\.")


def normalize_sql(sql: str) -> str:
    """Map the historical schema-qualified SQL dialect onto SQLite tables."""
    text = str(sql).strip()
    if not text:
        return text
    if re.fullmatch(r"CREATE\s+SCHEMA\s+IF\s+NOT\s+EXISTS\s+\w+", text, re.IGNORECASE):
        return ""
    text = re.sub(
        r"CURRENT_TIMESTAMP\s*-\s*INTERVAL\s+'(\d+)\s+days'",
        r"datetime(CURRENT_TIMESTAMP, '-\1 days')",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:DATE|TIMESTAMPTZ)\s+'([^']+)'",
        r"'\1'",
        text,
        flags=re.IGNORECASE,
    )
    return _SCHEMA_PREFIX.sub(lambda match: f"{match.group(1)}_", text)


def _coerce_parameters(parameters: Iterable[Any] | None) -> list[Any]:
    if parameters is None:
        return []
    values = []
    for value in parameters:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if hasattr(value, "isoformat"):
            value = value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
        values.append(value)
    return values


class SQLiteCursor:
    """Small cursor facade that preserves the repository-facing API."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    @property
    def description(self):
        return self._cursor.description

    def fetchone(self):
        row = self._cursor.fetchone()
        return None if row is None else self._convert_row(row)

    def fetchall(self):
        return [self._convert_row(row) for row in self._cursor.fetchall()]

    def fetchdf(self) -> pd.DataFrame:
        columns = [item[0] for item in self._cursor.description or []]
        return pd.DataFrame(self._cursor.fetchall(), columns=columns)

    def _convert_row(self, row):
        columns = [item[0] for item in self._cursor.description or []]
        values = []
        for column, value in zip(columns, row):
            if (
                isinstance(value, str)
                and column.endswith("date")
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            ):
                values.append(date.fromisoformat(value))
            else:
                values.append(value)
        return tuple(values)


class StaticCursor:
    """Cursor facade for compatibility metadata queries."""

    def __init__(self, columns: list[str], rows: list[tuple]):
        self.description = [(column,) for column in columns]
        self._rows = list(rows)

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self):
        rows = self._rows
        self._rows = []
        return rows

    def fetchdf(self) -> pd.DataFrame:
        return pd.DataFrame(self.fetchall(), columns=[item[0] for item in self.description])


class SQLiteConnection:
    """Tiny compatibility wrapper over sqlite3 used by storage repositories."""

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path)
        self._read_only = bool(read_only)
        uri = f"file:{self.path.as_posix()}?mode=ro" if read_only else str(self.path)
        self._connection = sqlite3.connect(
            uri,
            uri=read_only,
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        if not read_only:
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 5000")

    def __enter__(self) -> "SQLiteConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None and not self._read_only:
            self.commit()
        self.close()

    def execute(self, sql: str, parameters: Iterable[Any] | None = None) -> SQLiteCursor:
        normalized = normalize_sql(sql)
        if not normalized:
            return SQLiteCursor(self._connection.execute("SELECT 1 WHERE 0"))
        metadata = self._information_schema(normalized)
        if metadata is not None:
            return metadata
        try:
            cursor = self._connection.execute(normalized, _coerce_parameters(parameters))
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "UNIQUE constraint failed" in message:
                raise RuntimeError(f"Duplicate key: {message}") from exc
            raise
        return SQLiteCursor(cursor)

    def executemany(self, sql: str, seq_of_parameters: Iterable[Iterable[Any]]) -> SQLiteCursor:
        normalized = normalize_sql(sql)
        cursor = self._connection.executemany(
            normalized,
            [_coerce_parameters(parameters) for parameters in seq_of_parameters],
        )
        return SQLiteCursor(cursor)

    def register(self, name: str, frame: pd.DataFrame) -> None:
        data = frame.copy() if frame is not None else pd.DataFrame()
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                values = pd.to_datetime(data[column], errors="coerce")
                if (values.dt.time.astype(str) == "00:00:00").all():
                    data[column] = values.dt.strftime("%Y-%m-%d")
                else:
                    data[column] = values.dt.strftime("%Y-%m-%d %H:%M:%S")
        data.to_sql(str(name), self._connection, if_exists="replace", index=False)

    def unregister(self, name: str) -> None:
        safe_name = str(name).replace('"', '""')
        self._connection.execute(f'DROP TABLE IF EXISTS "{safe_name}"')

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        if not self._read_only:
            self._connection.commit()
        self._connection.close()

    def _information_schema(self, sql: str):
        lowered = " ".join(sql.lower().split())
        if "information_schema.schemata" in lowered:
            return StaticCursor(
                ["schema_name"],
                [("raw",), ("core",), ("derived",), ("ops",)],
            )
        if "information_schema.tables" in lowered:
            rows = []
            for (name,) in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall():
                if "_" not in name:
                    continue
                schema, table = name.split("_", 1)
                if schema in {"raw", "core", "derived", "ops"}:
                    rows.append((schema, table))
            return StaticCursor(["table_schema", "table_name"], sorted(rows))
        if "information_schema.columns" in lowered:
            schema_match = re.search(r"table_schema\s*=\s*'([^']+)'", sql, re.IGNORECASE)
            table_match = re.search(r"table_name\s*=\s*'([^']+)'", sql, re.IGNORECASE)
            if not (schema_match and table_match):
                return StaticCursor(["column_name"], [])
            table_name = f"{schema_match.group(1)}_{table_match.group(1)}"
            rows = [
                (row[1],)
                for row in self._connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            return StaticCursor(["column_name"], rows)
        return None


class Database:
    """Own a SQLite database path and initialize the unified cache schema."""

    def __init__(self, path: Union[str, Path] = PATHS.database, code_version: str = "unknown"):
        self.path = Path(path)
        self.code_version = str(code_version)

    def connect(self, *, read_only: bool = False) -> SQLiteConnection:
        """Open an independent connection, preserving read-only semantics."""
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"SQLite database does not exist: {self.path}")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteConnection(self.path, read_only=read_only)

    def initialize(self) -> None:
        """Create or migrate the database to the current schema version."""
        connection = self.connect()
        try:
            apply_migrations(connection, self.code_version)
        finally:
            connection.close()
