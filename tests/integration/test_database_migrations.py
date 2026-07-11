from datetime import date

import pytest

from stock_research.storage import Database
from stock_research.storage.migrations import MIGRATIONS, apply_migrations


def test_initialize_creates_schemas_and_is_idempotent(tmp_path):
    database = Database(tmp_path / "nested" / "my_trade.duckdb", code_version="test")

    database.initialize()
    database.initialize()

    with database.connect(read_only=True) as connection:
        schemas = {
            row[0]
            for row in connection.execute(
                "select schema_name from information_schema.schemata"
            ).fetchall()
        }
        versions = connection.execute(
            "select version, name, code_version "
            "from ops.schema_migrations order by version"
        ).fetchall()
        tables = {
            tuple(row)
            for row in connection.execute(
                "select table_schema, table_name from information_schema.tables "
                "where table_schema in ('ops', 'raw')"
            ).fetchall()
        }

    assert {"raw", "core", "derived", "ops"} <= schemas
    assert versions == [
        (1, "initial_ops", "test"),
        (2, "sector_persistence", "test"),
        (3, "stock_kline_persistence", "test"),
        (4, "provider_aware_sector_storage", "test"),
    ]
    assert {
        ("ops", "schema_migrations"),
        ("ops", "runs"),
        ("ops", "run_steps"),
        ("ops", "pipeline_events"),
        ("raw", "sector_boards"),
        ("raw", "sector_board_history"),
        ("raw", "stock_kline_daily"),
    } <= tables


def test_read_only_connection_does_not_create_missing_database(tmp_path):
    path = tmp_path / "missing.duckdb"

    database = Database(path)

    try:
        database.connect(read_only=True)
    except FileNotFoundError as exc:
        assert str(path) in str(exc)
    else:
        raise AssertionError("read-only connection unexpectedly created a database")
    assert not path.exists()


def test_run_date_constraint_rejects_future_market_cutoff(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()

    with database.connect() as connection:
        with pytest.raises(Exception, match="CHECK constraint"):
            connection.execute(
                """
                INSERT INTO ops.runs (
                    run_id, observation_date, market_cutoff, financial_cutoff,
                    report_period, mode, code_version, started_at
                ) VALUES (
                    'invalid-date', DATE '2026-06-30', DATE '2026-07-01',
                    TIMESTAMPTZ '2026-06-30 15:00:00+00', DATE '2026-03-31',
                    'offline', 'test', CURRENT_TIMESTAMP
                )
                """
            )

    with database.connect(read_only=True) as connection:
        count = connection.execute(
            "SELECT count(*) FROM ops.runs WHERE run_id = 'invalid-date'"
        ).fetchone()[0]
    assert count == 0


def test_provider_history_migration_preserves_legacy_rows(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    with database.connect() as connection:
        apply_migrations(connection, "legacy", migrations=MIGRATIONS[:2])
        connection.execute(
            """
            INSERT INTO raw.sector_boards (board_name, group_name, source)
            VALUES ('semiconductor', 'legacy', 'eastmoney/industry')
            """
        )
        connection.execute(
            """
            INSERT INTO raw.sector_board_history (
                board_name, trade_date, close, source
            ) VALUES ('semiconductor', DATE '2026-07-09', 9.0, 'eastmoney/industry')
            """
        )

    database.initialize()

    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO raw.sector_board_history (
                board_name, trade_date, close, source
            ) VALUES ('semiconductor', DATE '2026-07-09', 10.0, 'ths/industry')
            """
        )
        rows = connection.execute(
            """
            SELECT board_name, trade_date, close, source
            FROM raw.sector_board_history
            ORDER BY source
            """
        ).fetchall()
        board = connection.execute(
            """
            SELECT board_name, group_name, source, board_code
            FROM raw.sector_boards
            """
        ).fetchone()

    assert rows == [
        ("semiconductor", date(2026, 7, 9), 9.0, "eastmoney/industry"),
        ("semiconductor", date(2026, 7, 9), 10.0, "ths/industry"),
    ]
    assert board == (
        "semiconductor",
        "legacy",
        "eastmoney/industry",
        None,
    )
