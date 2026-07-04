"""Daily report input gate and reporting coordination."""
from __future__ import annotations

from stock_research.reporting import daily_report


def ensure_same_observation_date(inputs):
    dates = {str(value) for value in inputs.values() if value}
    if len(dates) > 1:
        raise ValueError(f"observation date mismatch: {sorted(dates)}")
    return next(iter(dates), "")


def run(argv=None) -> int:
    return int(daily_report.main(argv) or 0)
