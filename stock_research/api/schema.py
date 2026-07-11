"""Strict schema adaptation for external tabular APIs."""
from __future__ import annotations

import re

import pandas as pd


def normalize_column_name(value) -> str:
    """Normalize harmless presentation differences without guessing positions."""
    return re.sub(r"[\s_\-./（）()]+", "", str(value).strip().lower())


def rename_columns_strict(
    frame: pd.DataFrame,
    aliases: dict[str, tuple[str, ...] | list[str]],
    *,
    label: str,
) -> pd.DataFrame:
    """Rename required columns by aliases and fail clearly on ambiguity/missing data."""
    if frame is None or frame.empty:
        raise ValueError(f"{label} returned no rows")
    normalized: dict[str, list[object]] = {}
    for column in frame.columns:
        normalized.setdefault(normalize_column_name(column), []).append(column)

    rename = {}
    missing = []
    for target, candidates in aliases.items():
        matches = []
        for candidate in (target, *candidates):
            matches.extend(normalized.get(normalize_column_name(candidate), []))
        matches = list(dict.fromkeys(matches))
        if not matches:
            missing.append(f"{target}<-{list(candidates)}")
        elif len(matches) > 1:
            raise ValueError(
                f"{label} column {target!r} is ambiguous: {matches}; "
                f"actual columns={list(frame.columns)}"
            )
        else:
            rename[matches[0]] = target
    if missing:
        raise KeyError(
            f"{label} missing required columns: {', '.join(missing)}; "
            f"actual columns={list(frame.columns)}"
        )
    return frame.rename(columns=rename).copy()
