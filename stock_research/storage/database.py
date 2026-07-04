"""Explicit DuckDB connection and initialization boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import duckdb

from stock_research.core.paths import PATHS

from .migrations import apply_migrations


class Database:
    """Own a database path without sharing DuckDB's global connection."""

    def __init__(self, path: Union[str, Path] = PATHS.database, code_version: str = "unknown"):
        self.path = Path(path)
        self.code_version = str(code_version)

    def connect(self, *, read_only: bool = False):
        """Open an independent connection, preserving read-only semantics."""
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"DuckDB database does not exist: {self.path}")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path), read_only=read_only)

    def initialize(self) -> None:
        """Create or migrate the database to the current schema version."""
        connection = self.connect()
        try:
            apply_migrations(connection, self.code_version)
        finally:
            connection.close()
