"""Daily report input gate and reporting coordination."""
from __future__ import annotations

from stock_research.reporting import daily_report


ensure_same_observation_date = daily_report.ensure_same_observation_date


def run(argv=None) -> int:
    return int(daily_report.main(argv) or 0)
