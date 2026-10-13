"""Small background runner for the optional local dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional


@dataclass
class BacktestJob:
    command: List[str]
    cwd: Path
    process: Optional[subprocess.Popen] = None
    lines: List[str] = field(default_factory=list)
    returncode: Optional[int] = None
    _queue: queue.Queue = field(default_factory=queue.Queue, repr=False)

    @property
    def running(self) -> bool:
        return self.process is not None and self.returncode is None

    def poll(self) -> list[str]:
        """Drain output captured by the reader thread."""

        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                break
            self.lines.append(line.rstrip())
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.returncode = code
        return self.lines


def _read_output(job: BacktestJob) -> None:
    assert job.process is not None and job.process.stdout is not None
    for line in job.process.stdout:
        job._queue.put(line)
    job._queue.put(None)


def start_backtest(
    project_root: Path,
    *,
    start_date: str,
    end_date: str,
    output_directory: Path,
    allow_frozen_inputs: bool,
    close_confirmed_execution: str = "next_open",
    initial_capital: float = 1_000_000.0,
    commission_rate: float = 0.000085,
    minimum_commission: float = 5.0,
    sell_stamp_duty_rate: float = 0.0005,
    estimated_slippage_rate: float = 0.0005,
) -> BacktestJob:
    """Start the existing backtest CLI without invoking a shell."""

    command = [
        sys.executable,
        "-m",
        "apps.portfolio_backtest",
        "--start-date",
        start_date,
        "--output-directory",
        str(output_directory),
        "--close-confirmed-execution",
        close_confirmed_execution,
        "--initial-capital",
        str(float(initial_capital)),
        "--commission-rate",
        str(float(commission_rate)),
        "--minimum-commission",
        str(float(minimum_commission)),
        "--sell-stamp-duty-rate",
        str(float(sell_stamp_duty_rate)),
        "--estimated-slippage-rate",
        str(float(estimated_slippage_rate)),
    ]
    if end_date.strip():
        command.extend(["--end-date", end_date.strip()])
    if allow_frozen_inputs:
        command.extend(["--no-refresh-inputs", "--allow-frozen-inputs"])
    output_directory.mkdir(parents=True, exist_ok=True)
    job = BacktestJob(command=command, cwd=Path(project_root).resolve())
    job.process = subprocess.Popen(
        command,
        cwd=str(job.cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    threading.Thread(target=_read_output, args=(job,), daemon=True).start()
    return job


__all__ = ["BacktestJob", "start_backtest"]
