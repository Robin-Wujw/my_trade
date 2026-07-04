"""Sector mainline scoring rules."""

import pandas as pd


def score_direct(value, low, high):
    if value is None or pd.isna(value) or high == low:
        return 0.0
    return max(0.0, min(100.0, (float(value) - low) / (high - low) * 100.0))
