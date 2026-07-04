"""Observation-date eligibility rules for Formula33 results."""
from __future__ import annotations

import pandas as pd


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
