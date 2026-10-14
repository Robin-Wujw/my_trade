"""Point-in-time financial report period rules shared by live and backtest flows."""
from __future__ import annotations

import pandas as pd


def latest_visible_report_period(as_of_date) -> str:
    """Return the newest regular A-share report period visible on *as_of_date*.

    The conservative cutoffs are the statutory disclosure deadlines.  This
    does not pretend to know each company's exact announcement timestamp, but
    it prevents a backtest from using a quarter before that quarter could have
    been public.
    """
    date = pd.Timestamp(as_of_date).normalize()
    year = int(date.year)
    if date >= pd.Timestamp(year=year, month=10, day=31):
        return f"{year}-09-30"
    if date >= pd.Timestamp(year=year, month=8, day=31):
        return f"{year}-06-30"
    if date >= pd.Timestamp(year=year, month=4, day=30):
        return f"{year}-03-31"
    return f"{year - 1}-09-30"


def visible_report_periods(start_date, end_date) -> list[str]:
    """Enumerate every report period that can become visible in a date range."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"invalid date range: {start_date}..{end_date}")
    boundary_dates = [start, end]
    for year in range(start.year, end.year + 1):
        boundary_dates.extend([
            pd.Timestamp(year=year, month=4, day=30),
            pd.Timestamp(year=year, month=8, day=31),
            pd.Timestamp(year=year, month=10, day=31),
        ])
    return sorted({
        latest_visible_report_period(date)
        for date in boundary_dates
        if start <= date <= end
    })


def resolve_report_period(as_of_date, report_period: str = "") -> str:
    """Resolve a daily-production period without accepting a future override.

    Statutory deadlines are a conservative selection rule, not evidence of
    per-company announcement times or historical revision vintages.
    """
    date = pd.Timestamp(as_of_date)
    if pd.isna(date):
        raise ValueError(f"invalid observation date: {as_of_date!r}")
    latest = latest_visible_report_period(date)
    if not report_period or not report_period.strip():
        return latest
    period = pd.Timestamp(report_period)
    if pd.isna(period) or not period.is_quarter_end:
        raise ValueError(f"invalid quarterly report period: {report_period!r}")
    resolved = period.strftime("%Y-%m-%d")
    if resolved > latest:
        raise ValueError(
            f"report period {resolved} is not visible on {date:%Y-%m-%d}; "
            f"latest permitted period is {latest}"
        )
    return resolved
