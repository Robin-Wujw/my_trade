"""Transactional persistence for pipeline run and step lifecycles."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from stock_research.core import RunContext

from .database import Database


def _rows_as_dicts(cursor) -> list[dict]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


class RunRepository:
    """Persist one run identity and its observable step state."""

    def __init__(
        self,
        database: Database,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.database = database
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def start_run(self, context: RunContext) -> None:
        record = context.to_record()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO ops.runs (
                    run_id, observation_date, market_cutoff, financial_cutoff,
                    report_period, mode, code_version, started_at, status, gate_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', 'pending')
                """,
                [
                    record["run_id"],
                    record["observation_date"],
                    record["market_cutoff"],
                    record["financial_cutoff"],
                    record["report_period"],
                    record["mode"],
                    record["code_version"],
                    record["created_at"],
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def start_step(self, run_id: str, step_name: str, input_cutoff) -> None:
        started_at = self.clock()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO ops.run_steps (
                    run_id, step_name, input_cutoff, status, started_at
                ) VALUES (?, ?, ?, 'running', ?)
                """,
                [run_id, step_name, input_cutoff, started_at],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_step(
        self,
        run_id: str,
        step_name: str,
        *,
        status: str,
        row_count: Optional[int] = None,
        coverage: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("step status must be 'succeeded' or 'failed'")
        finished_at = self.clock()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                "SELECT started_at FROM ops.run_steps WHERE run_id = ? AND step_name = ?",
                [run_id, step_name],
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown run step: {run_id}/{step_name}")
            elapsed_seconds = max(0.0, (finished_at - row[0]).total_seconds())
            connection.execute(
                """
                UPDATE ops.run_steps
                SET status = ?, finished_at = ?, row_count = ?, coverage = ?,
                    elapsed_seconds = ?, error_message = ?
                WHERE run_id = ? AND step_name = ?
                """,
                [
                    status,
                    finished_at,
                    row_count,
                    coverage,
                    elapsed_seconds,
                    error_message,
                    run_id,
                    step_name,
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_run(
        self,
        run_id: str,
        *,
        gate_status: str,
        error_message: Optional[str] = None,
    ) -> None:
        if gate_status not in {"passed", "failed"}:
            raise ValueError("gate_status must be 'passed' or 'failed'")
        status = "succeeded" if gate_status == "passed" else "failed"
        finished_at = self.clock()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN TRANSACTION")
            if connection.execute(
                "SELECT 1 FROM ops.runs WHERE run_id = ?", [run_id]
            ).fetchone() is None:
                raise LookupError(f"unknown run: {run_id}")
            if gate_status == "passed":
                incomplete_steps = connection.execute(
                    """
                    SELECT step_name, status
                    FROM ops.run_steps
                    WHERE run_id = ? AND status <> 'succeeded'
                    ORDER BY step_name
                    """,
                    [run_id],
                ).fetchall()
                if incomplete_steps:
                    detail = ", ".join(
                        f"{step_name}={step_status}"
                        for step_name, step_status in incomplete_steps
                    )
                    raise ValueError(f"cannot pass run with incomplete steps: {detail}")
            connection.execute(
                """
                UPDATE ops.runs
                SET status = ?, gate_status = ?, finished_at = ?, error_message = ?
                WHERE run_id = ?
                """,
                [status, gate_status, finished_at, error_message, run_id],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_run(self, run_id: str) -> dict:
        connection = self.database.connect(read_only=True)
        try:
            runs = _rows_as_dicts(
                connection.execute("SELECT * FROM ops.runs WHERE run_id = ?", [run_id])
            )
            if not runs:
                raise LookupError(f"unknown run: {run_id}")
            steps = _rows_as_dicts(
                connection.execute(
                    "SELECT * FROM ops.run_steps WHERE run_id = ? ORDER BY started_at, step_name",
                    [run_id],
                )
            )
            run = runs[0]
            run["steps"] = steps
            return run
        finally:
            connection.close()
