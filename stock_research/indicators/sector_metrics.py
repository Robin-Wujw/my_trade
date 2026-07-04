"""Pure sector metric calculations."""

import numpy as np
import pandas as pd


def pct_change(series, days):
    if len(series) <= days:
        return np.nan
    base = series.iloc[-days - 1]
    return series.iloc[-1] / base - 1 if base else np.nan


def candle_label(row):
    pct = row.get("pct_chg", np.nan)
    if pd.isna(pct):
        return "-"
    direction = "阳" if pct >= 0 else "阴"
    absolute = abs(pct)
    if absolute >= 0.05:
        level = "长"
    elif absolute >= 0.02:
        level = "中"
    else:
        level = "小"
    return f"{level}{direction}"
