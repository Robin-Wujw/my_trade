"""Causal price preparation; raw execution prices remain distinct from signals."""
from __future__ import annotations
import math
import pandas as pd
from stock_research.indicators.technical_quant import _sma_cn


def _technical_rsv(price: pd.Series, low: pd.Series, high: pd.Series, period: int = 9) -> pd.Series:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    width = highest.sub(lowest)
    return price.sub(lowest).div(width.where(width > 0)).mul(100).clip(0, 100)



def _divergence_series(
    price_high: pd.Series,
    price_low: pd.Series,
    indicator: pd.Series,
    reset: pd.Series | None = None,
    *,
    lookback: int = 60,
) -> pd.Series:
    high_values = pd.to_numeric(
        price_high.reindex(indicator.index), errors="coerce",
    ).to_numpy(dtype=float)
    low_values = pd.to_numeric(
        price_low.reindex(indicator.index), errors="coerce",
    ).to_numpy(dtype=float)
    indicator_values = pd.to_numeric(
        indicator, errors="coerce",
    ).to_numpy(dtype=float)
    reset_flags = (
        reset.reindex(indicator.index).fillna(False).astype(bool).to_numpy()
        if reset is not None else None
    )
    values = [0] * len(indicator_values)
    active_rows = 0
    highest_price = highest_indicator = None
    lowest_price = lowest_indicator = None
    lookback = int(lookback)
    for position in range(len(indicator_values)):
        if reset_flags is not None and bool(reset_flags[position]):
            active_rows = 0
            highest_price = highest_indicator = None
            lowest_price = lowest_indicator = None
            continue
        current_high = high_values[position]
        current_low = low_values[position]
        current_indicator = indicator_values[position]
        if not (
            math.isfinite(current_high)
            and math.isfinite(current_low)
            and math.isfinite(current_indicator)
        ):
            continue
        signal = 0
        if active_rows >= 4:
            if (
                highest_price is not None
                and current_high > highest_price
                and current_indicator < highest_indicator
            ):
                signal = -1
            elif (
                lowest_price is not None
                and current_low < lowest_price
                and current_indicator > lowest_indicator
            ):
                signal = 1
        values[position] = signal
        if highest_price is None or current_high >= highest_price:
            highest_price = current_high
            highest_indicator = current_indicator
        if lowest_price is None or current_low <= lowest_price:
            lowest_price = current_low
            lowest_indicator = current_indicator
        active_rows = min(lookback, active_rows + 1)
        if active_rows >= lookback:
            window_start = max(0, position - lookback + 2)
            window_high = high_values[window_start:position + 1]
            window_low = low_values[window_start:position + 1]
            window_indicator = indicator_values[window_start:position + 1]
            valid = (
                pd.notna(window_high)
                & pd.notna(window_low)
                & pd.notna(window_indicator)
            )
            if valid.any():
                valid_positions = valid.nonzero()[0]
                high_position = valid_positions[
                    window_high[valid].argmax()
                ]
                low_position = valid_positions[
                    window_low[valid].argmin()
                ]
                highest_price = float(window_high[high_position])
                highest_indicator = float(window_indicator[high_position])
                lowest_price = float(window_low[low_position])
                lowest_indicator = float(window_indicator[low_position])
    return pd.Series(values, index=indicator.index)



def _structure_proximity_score_series(data):
    """Rebuild the frozen candidate structure score from visible price history."""
    close = pd.to_numeric(data["close"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma20_rising = ma20.notna() & ma20.shift(5).notna() & (ma20 > ma20.shift(5))
    ma60_rising = ma60.notna() & ma60.shift(5).notna() & (ma60 > ma60.shift(5))
    above_ma20 = ma20.notna() & close.ge(ma20)

    prior_high = close.shift(1).rolling(21).max()
    near_breakout = prior_high.notna() & prior_high.gt(0) & close.ge(
        prior_high * 0.97
    )
    high_21 = close.rolling(21, min_periods=10).max()
    low_21 = close.rolling(21, min_periods=10).min()
    range_21 = high_21.div(low_21.where(low_21 > 0)).sub(1)
    close_position = close.sub(low_21).div(
        high_21.sub(low_21).where(high_21 > low_21)
    )

    volume_base = pd.concat([
        volume.rolling(5).mean().shift(1),
        volume.rolling(10).mean().shift(1),
    ], axis=1).max(axis=1).fillna(0.0)
    volume_ratio = volume.div(volume_base.where(volume_base > 0)).fillna(0.0)
    volume_node = volume_ratio.ge(1.2) & ma20.notna() & close.ge(ma20)
    volume_node_count = volume_node.rolling(60, min_periods=1).sum()
    volume_node_close = close.where(volume_node).ffill()
    volume_node_distance = close.div(
        volume_node_close.where(volume_node_close > 0)
    ).sub(1)

    score = pd.Series(0.0, index=data.index)
    tight_platform = range_21.le(0.18) & close_position.ge(0.72)
    broad_platform = (
        ~tight_platform & range_21.le(0.25) & close_position.ge(0.65)
    )
    score = score.add(tight_platform.astype(float) * 24.0)
    score = score.add(broad_platform.astype(float) * 14.0)
    score = score.add(near_breakout.astype(float) * 18.0)
    score = score.add(volume_node_count.ge(3).astype(float) * 18.0)
    score = score.add(
        (volume_node_count.ge(1) & volume_node_count.lt(3)).astype(float) * 10.0
    )
    score = score.add(
        (
            volume_node_distance.ge(-0.03)
            & volume_node_distance.le(0.12)
        ).astype(float) * 14.0
    )
    score = score.add(
        (
            volume_node_distance.gt(0.12)
            & volume_node_distance.le(0.25)
        ).astype(float) * 6.0
    )
    score = score.add((ma20_rising & above_ma20).astype(float) * 14.0)
    score = score.add((ma20_rising & ma60_rising).astype(float) * 8.0)
    score = score.add(volume_ratio.ge(1.0).astype(float) * 4.0)
    return score.clip(upper=100.0)



def _prepare_frame(frame):
    data = frame.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    if "raw_to_qfq_factor" in data:
        data["raw_to_qfq_factor"] = pd.to_numeric(
            data["raw_to_qfq_factor"], errors="coerce",
        )
    for column in ("open", "high", "low", "close"):
        data[f"raw_{column}"] = pd.to_numeric(
            data.get(f"raw_{column}", data[column]), errors="coerce",
        )
    data = data.dropna(subset=["date", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if "raw_to_qfq_factor" in data:
        valid_factor = data["raw_to_qfq_factor"].notna() & data[
            "raw_to_qfq_factor"
        ].gt(0)
    else:
        valid_factor = pd.Series(dtype=bool)
    if not valid_factor.empty:
        if not bool(valid_factor.all()):
            invalid_dates = data.loc[
                ~valid_factor, "date",
            ].dt.strftime("%Y-%m-%d").head(5).tolist()
            raise ValueError(
                "raw_to_qfq_factor must be complete and positive; "
                f"invalid_dates={invalid_dates}"
            )
        anchor_factor = float(data["raw_to_qfq_factor"].iloc[-1])
        signal_multiplier = anchor_factor / data["raw_to_qfq_factor"]
    else:
        signal_multiplier = pd.Series(1.0, index=data.index)
    data["qfq_to_raw_factor"] = 1.0 / signal_multiplier.where(signal_multiplier > 0)
    for column in ("open", "high", "low", "close"):
        signal = data[f"raw_{column}"].mul(signal_multiplier)
        data[f"signal_{column}"] = signal
        data[column] = signal.where(signal.notna(), data[column])
    for period in (5, 10, 20, 60, 120):
        data[f"ma{period}"] = data["close"].rolling(period).mean()
    data["structure_proximity_score"] = _structure_proximity_score_series(data)
    close = data["close"]
    high = data["high"]
    low = data["low"]
    close_rsv = _technical_rsv(close, low, high, 9)
    kd_k_close = _sma_cn(close_rsv, 3, 1)
    kd_d_close = _sma_cn(kd_k_close, 3, 1)
    high_rsv = _technical_rsv(high, low, high, 9)
    low_rsv = _technical_rsv(low, low, high, 9)
    kd_k_high = kd_k_close.shift(1).mul(2 / 3).add(high_rsv.div(3))
    kd_d_high = kd_d_close.shift(1).mul(2 / 3).add(kd_k_high.div(3))
    kd_k_low = kd_k_close.shift(1).mul(2 / 3).add(low_rsv.div(3))
    kd_d_low = kd_d_close.shift(1).mul(2 / 3).add(kd_k_low.div(3))
    delta = close.diff()
    rsi_up = _sma_cn(delta.clip(lower=0), 999, 1)
    rsi_abs = _sma_cn(delta.abs(), 999, 1)
    rsi999 = rsi_up.div(rsi_abs.where(rsi_abs > 0)).mul(100).clip(0, 100)
    hh10 = high.rolling(10, min_periods=10).max()
    ll10 = low.rolling(10, min_periods=10).min()
    hh20 = high.rolling(20, min_periods=20).max()
    ll20 = low.rolling(20, min_periods=20).min()
    wr10 = hh10.sub(close).div(hh10.sub(ll10).where(hh10 > ll10)).mul(100).clip(0, 100)
    wr20 = hh20.sub(close).div(hh20.sub(ll20).where(hh20 > ll20)).mul(100).clip(0, 100)
    kd_divergence = _divergence_series(
        high, low, kd_k_high.add(kd_d_high).div(2), (kd_k_low < 20) & (kd_d_low < 20)
    )
    rsi_divergence = _divergence_series(high, low, rsi999, rsi999 < 50)
    wr_divergence = _divergence_series(
        high, low, (100 - wr10).add(100 - wr20).div(2), (wr10 >= 80) & (wr20 >= 80)
    )
    data["kd_divergence"] = kd_divergence
    data["rsi_divergence"] = rsi_divergence
    data["wr_divergence"] = wr_divergence
    data["bearish_divergence_count"] = (
        kd_divergence.eq(-1).astype(int)
        + rsi_divergence.eq(-1).astype(int)
        + wr_divergence.eq(-1).astype(int)
    )
    return data



def prepare_portfolio_price_frames(price_frames):
    """Prepare signal frames once for repeated portfolio comparisons."""
    return {
        code: _prepare_frame(price_frames[code])
        for code in sorted(price_frames)
    }

