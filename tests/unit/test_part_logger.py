import pytest

from stock_research.core.part_logger import PartLogger
from stock_research.storage import Database, SectorRepository


def test_part_logger_prints_and_persists_events(tmp_path, capsys):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = SectorRepository(database)
    logger = PartLogger("sector_stats", repository=repository)

    with logger.part("board_history"):
        logger.event(
            "board_history",
            "cache",
            "hit",
            message="warm cache",
            rows=3,
            context={"board": "半导体"},
        )

    output = capsys.readouterr().out
    assert "[sector_stats][board_history][cache][hit]" in output
    assert "warm cache" in output

    with database.connect(read_only=True) as connection:
        events = connection.execute(
            """
            SELECT part_name, event_type, status, message, rows
            FROM ops.pipeline_events
            ORDER BY created_at
            """
        ).fetchall()

    assert ("board_history", "part", "start", "start board_history", None) in events
    assert ("board_history", "cache", "hit", "warm cache", 3) in events
    assert any(event[:3] == ("board_history", "part", "finish") for event in events)


def test_part_logger_storage_failure_does_not_block_or_mask_business(capsys):
    class BrokenRepository:
        def log_event(self, **kwargs):
            raise RuntimeError("database locked")

    logger = PartLogger("sector_stats", repository=BrokenRepository())
    executed = []

    with logger.part("board_names"):
        executed.append(True)

    assert executed == [True]
    assert "[logger][write_failed]" in capsys.readouterr().out

    with pytest.raises(ValueError, match="business failed"):
        with logger.part("board_history"):
            raise ValueError("business failed")
