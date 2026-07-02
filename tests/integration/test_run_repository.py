from datetime import date, datetime, timedelta, timezone

import pytest

from my_trade.domain import RunContext, RunMode
from my_trade.storage import Database, RunRepository


def make_context():
    return RunContext.create(
        observation_date=date(2026, 6, 30),
        market_cutoff=date(2026, 6, 30),
        financial_cutoff=datetime(2026, 6, 30, 15, tzinfo=timezone.utc),
        report_period=date(2026, 3, 31),
        mode=RunMode.OFFLINE,
        code_version="test",
        now=datetime(2026, 7, 2, 1, 2, 3, tzinfo=timezone.utc),
    )


def test_run_and_step_lifecycle_is_persisted(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    context = make_context()
    times = iter(
        [
            context.created_at + timedelta(seconds=1),
            context.created_at + timedelta(seconds=4),
            context.created_at + timedelta(seconds=5),
        ]
    )
    repository = RunRepository(database, clock=lambda: next(times))

    repository.start_run(context)
    repository.start_step(context.run_id, "market", context.market_cutoff)
    repository.finish_step(
        context.run_id,
        "market",
        status="succeeded",
        row_count=42,
        coverage=0.99,
    )
    repository.finish_run(context.run_id, gate_status="passed")

    run = repository.get_run(context.run_id)
    assert run["status"] == "succeeded"
    assert run["gate_status"] == "passed"
    assert run["steps"][0]["row_count"] == 42
    assert run["steps"][0]["elapsed_seconds"] == pytest.approx(3.0)


def test_rejects_invalid_terminal_status_before_writing(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    repository = RunRepository(database)

    with pytest.raises(ValueError, match="step status"):
        repository.finish_step("missing", "market", status="running")
    with pytest.raises(ValueError, match="gate_status"):
        repository.finish_run("missing", gate_status="pending")


def test_unknown_run_update_raises_lookup_error(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    repository = RunRepository(database)

    with pytest.raises(LookupError, match="missing"):
        repository.finish_run("missing", gate_status="failed")


@pytest.mark.parametrize("step_status", ["running", "failed"])
def test_run_cannot_pass_with_incomplete_step(tmp_path, step_status):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    context = make_context()
    repository = RunRepository(database)
    repository.start_run(context)
    repository.start_step(context.run_id, "market", context.market_cutoff)
    if step_status == "failed":
        repository.finish_step(context.run_id, "market", status="failed")

    with pytest.raises(ValueError, match="cannot pass run"):
        repository.finish_run(context.run_id, gate_status="passed")
