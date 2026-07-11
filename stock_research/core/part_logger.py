"""Small structured logger for pipeline parts."""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Optional


class PartLogger:
    """Print readable part logs and optionally persist structured events."""

    def __init__(self, step_name: str, *, repository=None, run_id: Optional[str] = None):
        self.step_name = str(step_name)
        self.repository = repository
        self.run_id = run_id

    def event(
        self,
        part_name: str,
        event_type: str,
        status: str,
        *,
        message: str = "",
        rows: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        pieces = [
            f"[{self.step_name}]",
            f"[{part_name}]",
            f"[{event_type}]",
            f"[{status}]",
        ]
        detail = "".join(pieces)
        if rows is not None:
            detail += f" rows={rows}"
        if elapsed_seconds is not None:
            detail += f" elapsed={elapsed_seconds:.2f}s"
        if message:
            detail += f" {message}"
        print(detail.strip())
        if self.repository is not None:
            try:
                self.repository.log_event(
                    run_id=self.run_id,
                    step_name=self.step_name,
                    part_name=str(part_name),
                    event_type=str(event_type),
                    status=str(status),
                    message=str(message),
                    rows=rows,
                    elapsed_seconds=elapsed_seconds,
                    context=context,
                )
            except Exception as exc:
                print(
                    f"[{self.step_name}][{part_name}][logger][write_failed] "
                    f"{type(exc).__name__}: {exc}"
                )

    @contextmanager
    def part(self, part_name: str):
        start = time.monotonic()
        self.event(part_name, "part", "start", message=f"start {part_name}")
        try:
            yield
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.event(
                part_name,
                "part",
                "failed",
                message=str(exc),
                elapsed_seconds=elapsed,
            )
            raise
        else:
            elapsed = time.monotonic() - start
            self.event(
                part_name,
                "part",
                "finish",
                message=f"finish {part_name}",
                elapsed_seconds=elapsed,
            )
