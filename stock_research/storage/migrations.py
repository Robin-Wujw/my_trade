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
    Migration(
        version=2,
        name="sector_persistence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS raw.sector_boards (
                board_name VARCHAR PRIMARY KEY,
                group_name VARCHAR,
                source VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw.sector_board_history (
                board_name VARCHAR NOT NULL,
                trade_date DATE NOT NULL,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                amount DOUBLE,
                volume DOUBLE,
                pct_chg DOUBLE,
                source VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (board_name, trade_date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ops.pipeline_events (
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                run_id VARCHAR,
                step_name VARCHAR NOT NULL,
                part_name VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                message VARCHAR,
                rows BIGINT,
                elapsed_seconds DOUBLE,
                context_json VARCHAR,
                CHECK (rows IS NULL OR rows >= 0),
                CHECK (elapsed_seconds IS NULL OR elapsed_seconds >= 0)
            )
            """,
        ),
    ),
    Migration(
        version=3,
        name="stock_kline_persistence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS raw.stock_kline_daily (
                source VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                trade_date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                tradestatus VARCHAR,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source, code, trade_date)
            )
            """,
        ),
    ),
    Migration(
        version=4,
        name="provider_aware_sector_storage",
        statements=(
            "ALTER TABLE raw.sector_boards ADD COLUMN board_code VARCHAR",
            """
            CREATE TABLE raw.sector_board_history_provider_aware (
                board_name VARCHAR NOT NULL,
                trade_date DATE NOT NULL,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                amount DOUBLE,
                volume DOUBLE,
                pct_chg DOUBLE,
                source VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (board_name, trade_date, source)
            )
            """,
            """
            INSERT INTO raw.sector_board_history_provider_aware (
                board_name, trade_date, open, close, high, low, amount,
                volume, pct_chg, source, updated_at
            )
            SELECT board_name, trade_date, open, close, high, low, amount,
                   volume, pct_chg, source, updated_at
            FROM raw.sector_board_history
            """,
            "DROP TABLE raw.sector_board_history",
            """
            ALTER TABLE raw.sector_board_history_provider_aware
            RENAME TO sector_board_history
            """,
        ),
    ),
    Migration(
        version=5,
        name="research_backtest_persistence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS raw.fundamental_metrics (
                code VARCHAR NOT NULL,
                report_period DATE NOT NULL,
                quality_score DOUBLE,
                earnings_yoy DOUBLE,
                market_cap DOUBLE,
                value_line DOUBLE,
                payload_json VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (code, report_period)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS derived.candidate_snapshots (
                observation_date DATE NOT NULL,
                snapshot_version VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                name VARCHAR,
                selection_rank INTEGER,
                candidate_score DOUBLE,
                candidate_source VARCHAR,
                selection_reason VARCHAR,
                report_period DATE,
                signal_eligible BOOLEAN,
                payload_json VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (observation_date, snapshot_version, code)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS derived.formula33_phase (
                observation_date DATE NOT NULL,
                version VARCHAR NOT NULL,
                phase VARCHAR NOT NULL,
                window_up_streak INTEGER,
                window_down_streak INTEGER,
                payload_json VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (observation_date, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS derived.backtest_runs (
                run_id VARCHAR PRIMARY KEY,
                requested_start DATE NOT NULL,
                actual_start DATE,
                end_date DATE NOT NULL,
                initial_capital DOUBLE NOT NULL,
                final_return_pct DOUBLE,
                maximum_drawdown_pct DOUBLE,
                final_cash DOUBLE,
                summary_json VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS derived.backtest_trades (
                run_id VARCHAR NOT NULL,
                sequence INTEGER NOT NULL,
                trade_date DATE NOT NULL,
                code VARCHAR NOT NULL,
                name VARCHAR,
                trade_side VARCHAR NOT NULL,
                quantity DOUBLE NOT NULL,
                execution_price DOUBLE,
                trade_amount DOUBLE,
                transaction_cost_amount DOUBLE,
                profit_loss_amount DOUBLE,
                reason VARCHAR,
                payload_json VARCHAR NOT NULL,
                PRIMARY KEY (run_id, sequence)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS derived.backtest_positions (
                run_id VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                name VARCHAR,
                quantity DOUBLE,
                cost DOUBLE,
                close DOUBLE,
                market_value DOUBLE,
                unrealized_pnl_amount DOUBLE,
                payload_json VARCHAR NOT NULL,
                PRIMARY KEY (run_id, code)
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
