"""Quantified technical snapshot used by the daily report."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _sma_cn(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """TongdaXin SMA(X,N,M), seeded with the first valid value."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    previous = np.nan
    for index, value in enumerate(values):
        if math.isnan(value):
            continue
        previous = value if math.isnan(previous) else (m * value + (n - m) * previous) / n
        out[index] = previous
    return pd.Series(out, index=series.index)


def _rsv(price: pd.Series, low: pd.Series, high: pd.Series, period: int = 9) -> pd.Series:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    width = highest - lowest
    return ((price - lowest) / width.replace(0, np.nan) * 100).clip(0, 100)


def _last(series: pd.Series):
    value = pd.to_numeric(series, errors="coerce").iloc[-1]
    return None if pd.isna(value) else float(value)


def _divergence(price: pd.Series, indicator: pd.Series, reset: pd.Series | None = None, lookback: int = 60):
    """Compare the latest point with the previous local extreme in the active cycle."""
    work = pd.DataFrame({"price": price, "indicator": indicator}).dropna().tail(lookback)
    if reset is not None and not work.empty:
        reset_values = reset.reindex(work.index).fillna(False)
        reset_indexes = np.flatnonzero(reset_values.to_numpy())
        if len(reset_indexes):
            work = work.iloc[int(reset_indexes[-1]) + 1 :]
    if len(work) < 5:
        return 0
    previous = work.iloc[:-1]
    current = work.iloc[-1]
    high_index = previous["price"].idxmax()
    low_index = previous["price"].idxmin()
    if current["price"] > previous.loc[high_index, "price"] and current["indicator"] < previous.loc[high_index, "indicator"]:
        return -1
    if current["price"] < previous.loc[low_index, "price"] and current["indicator"] > previous.loc[low_index, "indicator"]:
        return 1
    return 0


def technical_snapshot(frame: pd.DataFrame) -> dict:
    """Return auditable values plus normalized opportunity/risk scores."""
    if frame is None or frame.empty:
        return {"technical_available": False, "technical_reason": "no kline data"}
    data = frame.copy().sort_values("date").drop_duplicates("date")
    for column in ("high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["high", "low", "close"])
    if len(data) < 20:
        return {"technical_available": False, "technical_reason": f"only {len(data)} bars; need 20"}

    close, high, low = data["close"], data["high"], data["low"]
    volume = data["volume"].fillna(0)
    close_rsv = _rsv(close, low, high)
    k = _sma_cn(close_rsv, 3, 1)
    d = _sma_cn(k, 3, 1)
    high_rsv = _rsv(high, low, high)
    low_rsv = _rsv(low, low, high)
    k_high = k.shift(1) * 2 / 3 + high_rsv / 3
    d_high = d.shift(1) * 2 / 3 + k_high / 3
    k_low = k.shift(1) * 2 / 3 + low_rsv / 3
    d_low = d.shift(1) * 2 / 3 + k_low / 3
    kd_reset = (k_low < 20) & (d_low < 20)

    delta = close.diff()
    rsi_up = _sma_cn(delta.clip(lower=0), 999, 1)
    rsi_abs = _sma_cn(delta.abs(), 999, 1)
    rsi = (rsi_up / rsi_abs.replace(0, np.nan) * 100).clip(0, 100)
    rsi_reset = rsi < 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    ma10 = close.rolling(10).mean()
    ene_upper, ene_lower = ma10 * 1.11, ma10 * 0.91
    hh10, ll10 = high.rolling(10).max(), low.rolling(10).min()
    hh20, ll20 = high.rolling(20).max(), low.rolling(20).min()
    wr10 = 100 * (hh10 - close) / (hh10 - ll10).replace(0, np.nan)
    wr20 = 100 * (hh20 - close) / (hh20 - ll20).replace(0, np.nan)
    bias10 = (close / ma10 - 1) * 100
    ma20 = close.rolling(20).mean()
    bias20 = (close / ma20 - 1) * 100

    current_volume = float(volume.iloc[-1])
    volume_ma5 = float(volume.tail(5).mean())
    volume_ma10 = float(volume.tail(10).mean())
    volume_ref5 = float(volume.iloc[-6]) if len(volume) >= 6 else np.nan
    volume_ref10 = float(volume.iloc[-11]) if len(volume) >= 11 else np.nan
    base_volume_ratio = current_volume / max(volume_ma5, volume_ma10) if max(volume_ma5, volume_ma10) > 0 else np.nan
    volume_baseline_ok = bool(
        current_volume > volume_ma5 and current_volume > volume_ma10
        and current_volume > volume_ref5 and current_volume > volume_ref10
    )
    volume_checks = {
        "volume_above_ma5": current_volume > volume_ma5,
        "volume_above_ma10": current_volume > volume_ma10,
        "volume_above_ref5": current_volume > volume_ref5,
        "volume_above_ref10": current_volume > volume_ref10,
    }
    volume_baseline_count = sum(volume_checks.values())

    kd_gap = _last(k - d)
    kd_gap_extreme = abs(kd_gap) >= 20 if kd_gap is not None else False
    kd_div = _divergence(high, (k_high + d_high) / 2, kd_reset)
    rsi_div = _divergence(high, rsi, rsi_reset)
    macd_div = _divergence(high, macd)
    bearish_divergences = sum(value == -1 for value in (kd_div, rsi_div, macd_div))
    bullish_divergences = sum(value == 1 for value in (kd_div, rsi_div, macd_div))
    price = float(close.iloc[-1])
    ene_position = (price - float(ene_lower.iloc[-1])) / max(float(ene_upper.iloc[-1] - ene_lower.iloc[-1]), 1e-12) * 100

    opportunity = 50.0
    risk = 25.0
    opportunity += 12 if volume_baseline_ok else -5
    opportunity += bullish_divergences * 9 - bearish_divergences * 7
    risk += bearish_divergences * 14 - bullish_divergences * 5
    if kd_gap_extreme:
        risk += 15 if kd_gap > 0 else -5
        opportunity += 10 if kd_gap < 0 else -8
    if ene_position >= 100:
        risk += 12
    elif ene_position <= 0:
        opportunity += 10
    if _last(bias10) is not None and abs(_last(bias10)) >= 10:
        risk += 12 if _last(bias10) > 0 else -5
        opportunity += 8 if _last(bias10) < 0 else -5
    wr_extreme = "oversold" if _last(wr10) >= 90 and _last(wr20) >= 90 else "overbought" if _last(wr10) <= 10 and _last(wr20) <= 10 else "neutral"
    risk += 8 if wr_extreme == "overbought" else -3 if wr_extreme == "oversold" else 0
    opportunity += 6 if wr_extreme == "oversold" else -3 if wr_extreme == "overbought" else 0
    opportunity = round(min(100, max(0, opportunity)), 1)
    risk = round(min(100, max(0, risk)), 1)
    confidence = round(min(100, 45 + len(data) / 5 + 8 * sum(v != 0 for v in (kd_div, rsi_div, macd_div))), 1)
    action_score = round(min(100, max(0, opportunity * 0.6 + (100 - risk) * 0.4)), 1)

    return {
        "technical_available": True,
        "kd_k_close": _last(k), "kd_d_close": _last(d),
        "kd_k_high": _last(k_high), "kd_d_high": _last(d_high),
        "kd_k_low": _last(k_low), "kd_d_low": _last(d_low),
        "kd_gap": kd_gap, "kd_gap_extreme": kd_gap_extreme,
        "kd_divergence": kd_div, "rsi999": _last(rsi), "rsi_divergence": rsi_div,
        "macd_dif": _last(dif), "macd_dea": _last(dea), "macd_hist": _last(macd), "macd_divergence": macd_div,
        "ene_upper": _last(ene_upper), "ene_mid": _last(ma10), "ene_lower": _last(ene_lower), "ene_position": round(ene_position, 1),
        "wr10": _last(wr10), "wr20": _last(wr20), "wr_extreme": wr_extreme,
        "bias10": _last(bias10), "bias20": _last(bias20),
        "volume_ma5": volume_ma5, "volume_ma10": volume_ma10,
        "volume_ref5": volume_ref5, "volume_ref10": volume_ref10,
        "base_volume_ratio": None if pd.isna(base_volume_ratio) else round(base_volume_ratio, 3),
        "volume_baseline_ok": volume_baseline_ok,
        "volume_baseline_count": volume_baseline_count,
        **volume_checks,
        "volume_ratio_ma5": round(current_volume / volume_ma5, 3) if volume_ma5 > 0 else None,
        "volume_ratio_ma10": round(current_volume / volume_ma10, 3) if volume_ma10 > 0 else None,
        "volume_ratio_ref5": round(current_volume / volume_ref5, 3) if volume_ref5 > 0 else None,
        "volume_ratio_ref10": round(current_volume / volume_ref10, 3) if volume_ref10 > 0 else None,
        "technical_opportunity_score": opportunity, "technical_risk_score": risk,
        "technical_confidence": confidence, "technical_action_score": action_score,
    }
