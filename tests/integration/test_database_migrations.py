from my_trade.storage import Database


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
                "where table_schema = 'ops'"
            ).fetchall()
        }

    assert {"raw", "core", "derived", "ops"} <= schemas
    assert versions == [(1, "initial_ops", "test")]
    assert {
        ("ops", "schema_migrations"),
        ("ops", "runs"),
        ("ops", "run_steps"),
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
