"""Pure Formula33 indicator calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tdx_sma(series, n, m=1):
    prev = np.nan
    values = []
    for raw in pd.to_numeric(series, errors="coerce"):
        if pd.isna(raw):
            values.append(np.nan)
            continue
        if pd.isna(prev):
            prev = raw
        else:
            prev = (m * raw + (n - m) * prev) / n
        values.append(prev)
    return pd.Series(values, index=series.index)


def calc_kdj_k(df, n=9):
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    kd = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    return tdx_sma(kd, 3, 1)


def calc_wr(df, n):
    high_n = df["high"].rolling(n, min_periods=n).max()
    low_n = df["low"].rolling(n, min_periods=n).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * 100


def calc_rsi(series, n=9):
    diff = series.diff()
    up = diff.clip(lower=0)
    absolute = diff.abs()
    return tdx_sma(up, n, 1) / tdx_sma(absolute, n, 1).replace(0, np.nan) * 100
