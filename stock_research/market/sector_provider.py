"""Provider-independent sector freshness and coverage validation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral

import pandas as pd


@dataclass(frozen=True)
class CoverageResult:
    expected: int
    fresh: int
    stale: int
    missing: int
    coverage: float
    passed: bool


def _count(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return converted


def minimum_fresh_count(expected: int, minimum: float = 0.95) -> int:
    """Return the smallest integer fresh count that satisfies coverage."""
    expected_count = _count("expected", expected)
    threshold = float(minimum)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum must be between 0 and 1")
    return math.ceil(expected_count * threshold)


def coverage_can_still_pass(
    *,
    expected: int,
    completed: int,
    fresh: int,
    minimum: float = 0.95,
) -> bool:
    """Return whether all remaining items succeeding could still pass the gate."""
    expected_count = _count("expected", expected)
    completed_count = _count("completed", completed)
    fresh_count = _count("fresh", fresh)
    if completed_count > expected_count:
        raise ValueError("completed cannot exceed expected")
    if fresh_count > completed_count:
        raise ValueError("fresh cannot exceed completed")
    possible_fresh = fresh_count + expected_count - completed_count
    return possible_fresh >= minimum_fresh_count(expected_count, minimum)


def effective_pipeline_retries(provider, requested: int) -> int:
    """Avoid multiplying pipeline retries with an adapter's own request retries."""
    requested_count = max(1, _count("requested", requested))
    try:
        adapter_attempts = int(getattr(provider, "REQUEST_ATTEMPTS", 1))
    except (TypeError, ValueError):
        adapter_attempts = 1
    return 1 if adapter_attempts > 1 else requested_count


def _eligible_history_dates(frame, observation) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DatetimeIndex([])
    date_column = next(
        (
            candidate
            for candidate in ("date", "trade_date", "日期")
            if candidate in frame.columns
        ),
        None,
    )
    if date_column is None:
        return pd.DatetimeIndex([])
    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    dates = dates[dates.notna() & (dates <= observation)]
    if dates.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates).normalize().unique().sort_values()


def sector_history_is_fresh(
    frame,
    *,
    observation_date,
    max_stale_days: int = 7,
    minimum_rows: int = 1,
) -> bool:
    """Return whether history has enough distinct dates and reaches the cutoff."""
    observation = pd.to_datetime(observation_date, errors="coerce", utc=True)
    if pd.isna(observation):
        raise ValueError("observation_date must be a valid date")
    observation = observation.normalize()
    stale_days = _count("max_stale_days", max_stale_days)
    required_rows = max(1, _count("minimum_rows", minimum_rows))
    eligible_dates = _eligible_history_dates(frame, observation)
    if len(eligible_dates) < required_rows:
        return False
    freshness_cutoff = observation - pd.Timedelta(days=stale_days)
    return eligible_dates.max() >= freshness_cutoff


def validate_coverage(
    *,
    expected: int,
    fresh: int,
    stale: int = 0,
    missing: int | None = None,
    minimum: float = 0.95,
) -> CoverageResult:
    """Calculate a fail-closed coverage result from mutually exclusive counts."""
    expected_count = _count("expected", expected)
    fresh_count = _count("fresh", fresh)
    stale_count = _count("stale", stale)
    if fresh_count + stale_count > expected_count:
        raise ValueError("fresh and stale cannot exceed expected")

    if missing is None:
        missing_count = expected_count - fresh_count - stale_count
    else:
        missing_count = _count("missing", missing)
        if fresh_count + stale_count + missing_count != expected_count:
            raise ValueError("fresh, stale, and missing must add up to expected")

    threshold = float(minimum)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum must be between 0 and 1")

    coverage = fresh_count / expected_count if expected_count else 0.0
    return CoverageResult(
        expected=expected_count,
        fresh=fresh_count,
        stale=stale_count,
        missing=missing_count,
        coverage=coverage,
        passed=expected_count > 0 and coverage >= threshold,
    )


def validate_sector_histories(
    expected_boards: Iterable[str],
    histories: Mapping[str, pd.DataFrame],
    *,
    observation_date,
    max_stale_days: int = 7,
    minimum_rows: int = 1,
    minimum: float = 0.95,
) -> tuple[dict[str, pd.DataFrame], CoverageResult]:
    """Return histories meeting date and depth requirements plus diagnostics."""
    observation = pd.to_datetime(observation_date, errors="coerce", utc=True)
    if pd.isna(observation):
        raise ValueError("observation_date must be a valid date")
    observation = observation.normalize()

    stale_days = _count("max_stale_days", max_stale_days)
    required_rows = max(1, _count("minimum_rows", minimum_rows))
    freshness_cutoff = observation - pd.Timedelta(days=stale_days)
    board_names = list(
        dict.fromkeys(
            name
            for name in (str(value).strip() for value in expected_boards)
            if name
        )
    )

    fresh_histories = {}
    stale_count = 0
    missing_count = 0
    for board_name in board_names:
        frame = histories.get(board_name)
        eligible_dates = _eligible_history_dates(frame, observation)
        if len(eligible_dates) < required_rows:
            missing_count += 1
            continue
        if eligible_dates.max() < freshness_cutoff:
            stale_count += 1
            continue
        fresh_histories[board_name] = frame

    coverage = validate_coverage(
        expected=len(board_names),
        fresh=len(fresh_histories),
        stale=stale_count,
        missing=missing_count,
        minimum=minimum,
    )
    return fresh_histories, coverage
