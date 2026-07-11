import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor

from stock_research.storage import Database, SectorRepository


def make_repository(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    return SectorRepository(database)


def test_sector_repository_upserts_boards_and_history(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)

    boards = pd.DataFrame(
        [
            {"board": "半导体", "group": "半导体"},
            {"board_name": "通信设备", "group": "通信设备"},
        ]
    )
    assert repository.upsert_boards(boards, source="unit") == 2

    loaded_boards = repository.load_boards()
    assert loaded_boards["board_name"].tolist() == ["半导体", "通信设备"]
    assert loaded_boards.loc[0, "source"] == "unit"

    history = pd.DataFrame(
        [
            {
                "date": "2026-07-06",
                "open": 10.0,
                "close": 11.0,
                "high": 12.0,
                "low": 9.5,
                "amount": 1000.0,
                "volume": 200.0,
                "pct_chg": 0.05,
            },
            {
                "date": "2026-07-07",
                "open": 11.0,
                "close": 12.0,
                "high": 12.5,
                "low": 10.5,
                "amount": 1200.0,
                "volume": 220.0,
                "pct_chg": 0.09,
            },
        ]
    )
    assert repository.upsert_board_history("半导体", history, source="unit") == 2

    loaded_history = repository.load_board_history(
        "半导体",
        end_date="2026-07-07",
        days=1,
        date_column="date",
    )
    assert loaded_history[["date", "close"]].to_dict("records") == [
        {"date": pd.Timestamp("2026-07-07"), "close": 12.0}
    ]


def test_sector_repository_logs_pipeline_events(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)

    repository.log_event(
        step_name="sector_stats",
        part_name="board_history",
        event_type="cache",
        status="hit",
        message="warm cache",
        rows=5,
        elapsed_seconds=0.2,
        context={"board": "半导体"},
    )

    with database.connect(read_only=True) as connection:
        row = connection.execute(
            """
            SELECT step_name, part_name, event_type, status, message, rows,
                   elapsed_seconds, context_json
            FROM ops.pipeline_events
            """
        ).fetchone()

    assert row[:7] == (
        "sector_stats",
        "board_history",
        "cache",
        "hit",
        "warm cache",
        5,
        0.2,
    )
    assert '"board": "半导体"' in row[7]


def test_sector_repository_normalizes_api_percentages_to_decimal(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)

    repository.upsert_board_history(
        "半导体",
        pd.DataFrame(
            [
                {
                    "日期": "2026-07-09",
                    "收盘": 10.5,
                    "最高": 11.0,
                    "最低": 10.0,
                    "涨跌幅": 1.2,
                }
            ]
        ),
        source="eastmoney",
    )

    loaded = repository.load_board_history(
        "半导体",
        end_date="2026-07-09",
        days=1,
    )

    assert loaded["pct_chg"].tolist() == pytest.approx([0.012])


def test_sector_repository_can_filter_stale_board_lists(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)
    repository.upsert_boards(
        pd.DataFrame([{"board": "旧板块", "group": "旧分类"}]),
        source="unit",
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE raw.sector_boards SET updated_at = CURRENT_TIMESTAMP - INTERVAL '2 days'"
        )

    assert repository.load_boards(max_age="24h").empty
    assert repository.load_boards()["board_name"].tolist() == ["旧板块"]


def test_sector_repository_treats_partially_stale_board_snapshot_as_stale(
    tmp_path,
):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)
    repository.upsert_boards(
        pd.DataFrame(
            [
                {"board": "新板块", "group": "新分类"},
                {"board": "旧板块", "group": "旧分类"},
            ]
        ),
        source="unit",
    )
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE raw.sector_boards
            SET updated_at = CURRENT_TIMESTAMP - INTERVAL '2 days'
            WHERE board_name = '旧板块'
            """
        )

    assert repository.load_boards(max_age="24h").empty


def test_replace_boards_reconciles_active_snapshot_without_deleting_history(tmp_path):
    repository = make_repository(tmp_path)
    repository.replace_boards(
        pd.DataFrame(
            [
                {"board_name": "old", "group_name": "legacy", "code": "880001"},
                {"board_name": "keep", "group_name": "legacy", "code": "880002"},
            ]
        ),
        source="eastmoney/industry",
    )
    repository.upsert_board_history(
        "old",
        pd.DataFrame([{"date": "2026-07-09", "close": 10.0}]),
        source="eastmoney/industry",
    )

    replaced = repository.replace_boards(
        pd.DataFrame(
            [
                {"name": "keep", "group_name": "current", "code": "881001"},
                {
                    "name": "new",
                    "group_name": "current",
                    "board_code": "881002",
                },
            ]
        ),
        source="ths/industry",
    )

    assert replaced == 2
    current = repository.load_boards(source="ths/industry")
    assert current[["board_name", "board_code"]].to_dict("records") == [
        {"board_name": "keep", "board_code": "881001"},
        {"board_name": "new", "board_code": "881002"},
    ]
    assert repository.load_boards(source_prefix="eastmoney/").empty
    retained = repository.load_board_history(
        "old",
        end_date="2026-07-10",
        days=10,
        source_prefix="eastmoney/",
    )
    assert retained[["date", "close"]].to_dict("records") == [
        {"date": pd.Timestamp("2026-07-09"), "close": 10.0}
    ]


def test_board_history_source_prefix_excludes_other_providers(tmp_path):
    repository = make_repository(tmp_path)
    repository.upsert_board_history(
        "semiconductor",
        pd.DataFrame([{"date": "2026-07-09", "close": 9.0}]),
        source="eastmoney/industry",
    )
    repository.upsert_board_history(
        "semiconductor",
        pd.DataFrame([{"date": "2026-07-09", "close": 10.0}]),
        source="ths/industry",
    )

    loaded = repository.load_board_history(
        "semiconductor",
        end_date="2026-07-10",
        days=10,
        source_prefix="ths/",
    )

    assert loaded["date"].tolist() == [pd.Timestamp("2026-07-09")]
    assert loaded["source"].tolist() == ["ths/industry"]

    legacy = repository.load_board_history(
        "semiconductor",
        end_date="2026-07-10",
        days=10,
        source_prefix="eastmoney/",
    )
    assert legacy["date"].tolist() == [pd.Timestamp("2026-07-09")]
    assert legacy["close"].tolist() == [9.0]


def test_concurrent_sector_reads_can_log_events_on_same_database(tmp_path):
    repository = make_repository(tmp_path)
    repository.upsert_board_history(
        "semiconductor",
        pd.DataFrame([{"date": "2026-07-10", "close": 10.0}]),
        source="ths/history",
    )

    def read_and_log(index):
        frame = repository.load_board_history(
            "semiconductor",
            end_date="2026-07-10",
            days=1,
            source="ths/history",
        )
        repository.log_event(
            step_name="sector_watch",
            part_name="board_history",
            event_type="duckdb",
            status="hit",
            rows=len(frame),
            context={"worker": index},
        )
        return len(frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(read_and_log, range(20)))

    assert results == [1] * 20
