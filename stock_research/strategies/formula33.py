"""Observation-date eligibility rules for Formula33 results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_count_direction_streaks(values):
    """Build changes and directional streaks for consecutive rolling nodes."""
    rows = []
    previous = None
    up_streak = 0
    down_streak = 0
    for value in values:
        current = int(value)
        change = 0 if previous is None else current - previous
        if change > 0:
            up_streak += 1
            down_streak = 0
        elif change < 0:
            up_streak = 0
            down_streak += 1
        else:
            up_streak = 0
            down_streak = 0
        rows.append(
            {
                "window_change": change,
                "window_up_streak": up_streak,
                "window_down_streak": down_streak,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def _build_trade_coverage(trade_coverage):
    if trade_coverage is None:
        return None
    if isinstance(trade_coverage, dict):
        records = trade_coverage.items()
    elif isinstance(trade_coverage, pd.DataFrame):
        if not {"code", "covered_dates"}.issubset(trade_coverage.columns):
            return {}
        records = trade_coverage[["code", "covered_dates"]].itertuples(
            index=False,
            name=None,
        )
    else:
        records = trade_coverage

    coverage = {}
    for code, covered_dates in records:
        if covered_dates is None:
            dates = set()
        elif isinstance(covered_dates, str):
            dates = {covered_dates}
        else:
            dates = {str(value) for value in covered_dates if pd.notna(value)}
        coverage[str(code)] = dates
    return coverage


def _build_current_statuses(current_statuses):
    if current_statuses is None:
        return None
    if isinstance(current_statuses, dict):
        records = current_statuses.items()
    elif isinstance(current_statuses, pd.DataFrame):
        if not {"code", "observation_status"}.issubset(current_statuses.columns):
            return {}
        records = (
            current_statuses.drop_duplicates("code", keep="last")
            [["code", "observation_status"]]
            .itertuples(index=False, name=None)
        )
    else:
        records = current_statuses
    return {str(code): str(status) for code, status in records}


def build_window_trend(
    xg_hits,
    trade_dates,
    window=21,
    output_days=21,
    trade_coverage=None,
    current_statuses=None,
):
    """Build rolling unique-XG breadth and its rolling linear trend."""
    dates = [str(value) for value in trade_dates]
    columns = [
        "date",
        "window_unique_count",
        "technical_unique_count",
        "tradable_unique_count",
        "window_change",
        "window_up_streak",
        "window_down_streak",
        "window_trend_slope",
        "trend_up_streak",
        "trend_down_streak",
        "trend_signal",
    ]
    required_dates = window
    if len(dates) < required_dates:
        return pd.DataFrame(columns=columns)

    hit_codes = {}
    if xg_hits is not None and not xg_hits.empty:
        normalized = xg_hits[["date", "code"]].dropna().copy()
        normalized["date"] = normalized["date"].astype(str)
        for date, group in normalized.groupby("date"):
            hit_codes[date] = set(group["code"].astype(str))

    coverage_by_code = _build_trade_coverage(trade_coverage)
    status_by_code = _build_current_statuses(current_statuses)
    latest_date = dates[-1]
    unique_rows = []
    for end_index in range(window - 1, len(dates)):
        codes = set()
        for date in dates[end_index - window + 1 : end_index + 1]:
            codes.update(hit_codes.get(date, set()))
        observation_date = dates[end_index]
        technical_count = len(codes)
        if coverage_by_code is not None:
            codes = {
                code
                for code in codes
                if observation_date in coverage_by_code.get(code, set())
            }
        if observation_date == latest_date and status_by_code is not None:
            codes = {
                code for code in codes if status_by_code.get(code) == "traded"
            }
        formal_count = len(codes)
        unique_rows.append(
            {
                "date": observation_date,
                "window_unique_count": formal_count,
                "technical_unique_count": technical_count,
                "tradable_unique_count": formal_count,
            }
        )

    rolling = pd.DataFrame(unique_rows)
    values = rolling["window_unique_count"].astype(float)
    directions = build_count_direction_streaks(values)
    for column in directions.columns:
        rolling[column] = directions[column].to_numpy()
    x_axis = np.arange(window, dtype=float)
    rolling["window_trend_slope"] = values.rolling(window).apply(
        lambda sample: float(np.polyfit(x_axis, sample, 1)[0]), raw=True
    )

    up_streak = 0
    down_streak = 0
    up_values = []
    down_values = []
    signals = []
    for slope in rolling["window_trend_slope"]:
        if pd.notna(slope) and slope > 1e-12:
            up_streak += 1
            down_streak = 0
        elif pd.notna(slope) and slope < -1e-12:
            up_streak = 0
            down_streak += 1
        else:
            up_streak = 0
            down_streak = 0
        up_values.append(up_streak)
        down_values.append(down_streak)
        signals.append(
            "up" if up_streak else "down" if down_streak else "neutral"
        )

    rolling["trend_up_streak"] = up_values
    rolling["trend_down_streak"] = down_values
    rolling["trend_signal"] = signals
    return rolling.tail(output_days).reset_index(drop=True)[columns]


def classify_observation_status(
    latest_data_date, observation_date, fetch_error=None
):
    if fetch_error:
        return "data_unavailable"
    latest = pd.to_datetime(latest_data_date, errors="coerce")
    observation = pd.to_datetime(observation_date, errors="coerce")
    if pd.isna(latest) or pd.isna(observation):
        return "data_unavailable"
    if latest.normalize() >= observation.normalize():
        return "traded"
    return "suspended_or_no_trade"


def select_window_unique_hits(xg_hits, statuses):
    if xg_hits is None or xg_hits.empty:
        empty = pd.DataFrame(columns=list(getattr(xg_hits, "columns", [])))
        return empty, empty.copy()
    technical = (
        xg_hits.sort_values(["code", "date"])
        .drop_duplicates("code", keep="last")
        .sort_values("code")
        .reset_index(drop=True)
    )
    if statuses is None or statuses.empty:
        return technical, technical.iloc[0:0].copy()
    latest_status = statuses.drop_duplicates("code", keep="last").set_index("code")[
        "observation_status"
    ]
    formal = technical[
        technical["code"].map(latest_status).eq("traded")
    ].reset_index(drop=True)
    return technical, formal
