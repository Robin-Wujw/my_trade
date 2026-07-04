"""Pure parsing and scaling helpers used by factor selection."""

import numpy as np
import pandas as pd


def parse_yi(value):
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in {"--", "False"}:
            return None
        if "亿" in text:
            return float(text.replace("亿", "")) * 1e8
        if "万" in text:
            return float(text.replace("万", "")) * 1e4
        return float(text)
    except Exception:
        return None


def parse_pct(value):
    try:
        text = str(value).replace("%", "").strip()
        if not text or text.lower() == "nan" or text == "--":
            return None
        return float(text) / 100
    except Exception:
        return None


def parse_float(value):
    try:
        text = str(value).replace(",", "").strip()
        if not text or text.lower() == "nan" or text in {"--", "False"}:
            return None
        return float(text)
    except Exception:
        return None


def clamp(value, low=0, high=100):
    if value is None or pd.isna(value):
        return 0
    return max(low, min(high, value))


def score_direct(value, worst, best):
    if value is None or pd.isna(value) or best == worst:
        return 0
    return clamp((value - worst) / (best - worst) * 100)


def score_inverse(value, best, worst):
    if value is None or pd.isna(value) or best == worst:
        return 0
    return clamp((worst - value) / (worst - best) * 100)


def remove_outliers(values):
    if len(values) < 3:
        return values
    median = np.median(values)
    if median <= 0:
        return values
    filtered = [value for value in values if 0.5 * median <= value <= 2.0 * median]
    return filtered if len(filtered) >= 3 else values
