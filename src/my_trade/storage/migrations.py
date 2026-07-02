"""Ordered DuckDB schema migrations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS = (
    Migration(
        version=1,
        name="initial_ops",
        statements=(
            "CREATE SCHEMA IF NOT EXISTS raw",
            "CREATE SCHEMA IF NOT EXISTS core",
            "CREATE SCHEMA IF NOT EXISTS derived",
            "CREATE SCHEMA IF NOT EXISTS ops",
            """
            CREATE TABLE IF NOT EXISTS ops.schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                code_version VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ops.runs (
                run_id VARCHAR PRIMARY KEY,
                observation_date DATE NOT NULL,
                market_cutoff DATE NOT NULL,
                financial_cutoff TIMESTAMPTZ NOT NULL,
                report_period DATE NOT NULL,
                mode VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                status VARCHAR NOT NULL DEFAULT 'running',
                gate_status VARCHAR NOT NULL DEFAULT 'pending',
                error_message VARCHAR,
                CHECK (market_cutoff <= observation_date),
                CHECK (report_period <= observation_date),
                CHECK (mode IN ('production', 'backtest', 'offline')),
                CHECK (status IN ('running', 'succeeded', 'failed')),
                CHECK (gate_status IN ('pending', 'passed', 'failed'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ops.run_steps (
                run_id VARCHAR NOT NULL,
                step_name VARCHAR NOT NULL,
                input_cutoff DATE,
                status VARCHAR NOT NULL DEFAULT 'running',
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                row_count BIGINT,
                coverage DOUBLE,
                elapsed_seconds DOUBLE,
                error_message VARCHAR,
                retry_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, step_name),
                FOREIGN KEY (run_id) REFERENCES ops.runs(run_id),
                CHECK (status IN ('running', 'succeeded', 'failed')),
                CHECK (row_count IS NULL OR row_count >= 0),
                CHECK (coverage IS NULL OR (coverage >= 0 AND coverage <= 1)),
                CHECK (elapsed_seconds IS NULL OR elapsed_seconds >= 0),
                CHECK (retry_count >= 0)
            )
            """,
        ),
    ),
)


def apply_migrations(connection, code_version: str, migrations: Iterable[Migration] = MIGRATIONS) -> None:
    """Apply unapplied migrations atomically in ascending version order."""
    ordered = tuple(sorted(migrations, key=lambda item: item.version))
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS ops")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ops.schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                code_version VARCHAR NOT NULL
            )
            """
        )
        applied = {
            row[0]
            for row in connection.execute(
                "SELECT version FROM ops.schema_migrations"
            ).fetchall()
        }
        for migration in ordered:
            if migration.version in applied:
                continue
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO ops.schema_migrations (version, name, code_version) VALUES (?, ?, ?)",
                [migration.version, migration.name, code_version],
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
