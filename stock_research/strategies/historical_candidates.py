"""Build conservative dated research candidates from locally cached data."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from stock_research.indicators.technical_quant import _sma_cn
from stock_research.core.financial_period import (
    latest_visible_report_period,
    visible_report_periods,
)
from stock_research.strategies.candidate_interface import (
    left_value_safety_reasons,
    normalize_candidate,
    normalize_candidate_snapshots,
)


SNAPSHOT_VERSION = "unified-selection-v4"
MAX_MAINLINE_AGE_DAYS = 31
CANDIDATE_SNAPSHOT_COLUMNS = [
    "date", "code", "name", "close", "value_line", "quality_score",
    "earnings_yoy", "mktcap", "report_period", "snapshot_version",
    "financial_point_in_time", "financial_point_in_time_source",
    "announcement_date", "price_to_value", "mainline_snapshot_date",
    "mainline_snapshot_fresh", "mainline_boards", "trade_basis_score",
    "trade_basis_reason", "technical_alignment", "ma20_rising",
    "ma60_rising", "above_ma20", "near_ma20", "near_21d_close_high",
    "known_volume_ratio", "volume_deduction_periods", "ima_web_validation",
    "return_5d", "return_10d", "return_20d", "return_60d", "return_120d", "distance_120d_high",
    "leadership_score", "leadership_reason", "long_term_structure_favorable",
    "right_strength_score", "right_strength_reason",
    "right_quant_score", "right_quant_reason", "right_momentum_rank",
    "right_trend_rank", "right_volume_rank", "right_risk_rank",
    "right_acceleration_rank", "right_mainline_rank", "right_structure_rank",
    "right_volume_node_rank", "right_quant_rank",
    "quant_quality_rank", "quant_growth_rank", "quant_liquidity_rank",
    "quant_momentum_rank", "quant_low_risk_rank", "quant_trend_stability_rank",
    "quant_overheat_control_rank", "quant_alpha_rank",
    "quant_trend_efficiency_rank", "quant_payoff_rank",
    "quant_structure_rank", "quant_volume_confirm_rank",
    "quant_market_regime",
    "volatility_20", "drawdown_60", "ma20_slope", "ma60_slope",
    "right_acceleration", "range_21_pct", "close_position_21",
    "volume_node_count_60", "volume_node_distance",
    "avg_amount_20", "momentum_60_ex5", "momentum_120_ex20",
    "positive_day_ratio_60", "downside_volatility_60",
    "data_status", "tradestatus", "is_traded_bar", "raw_amount",
    "amount_source", "price_source", "valid_price_bar", "alpha_volume_price_corr_20",
    "alpha_turnover_expansion_20", "alpha_close_position_60",
    "alpha_close_position_120", "alpha_intraday_strength_20",
    "alpha_gap_5d_count",
    "kd_k_high", "kd_d_high", "kd_k_low", "kd_d_low",
    "kd_divergence", "rsi999", "rsi_divergence", "wr10", "wr20",
    "wr_divergence", "bearish_divergence_count", "bullish_divergence_count",
    "divergence_score", "divergence_reason", "quant_divergence_rank",
    "structure_proximity_score", "structure_proximity_reason",
    "validation_sources", "strategy_part", "candidate_score",
    "historical_adjustment_check", "qfq_anchor_date", "candidate_source", "signal_eligible",
    "selected_for_trading", "candidate_failure_reason", "value_falsified",
    "value_falsification_reason",
    "selection_reason", "selection_rank", "right_quant_setup",
    "allow_left", "allow_right",
]


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
    work = pd.DataFrame({
        "price_high": pd.to_numeric(price_high, errors="coerce"),
        "price_low": pd.to_numeric(price_low, errors="coerce"),
        "indicator": pd.to_numeric(indicator, errors="coerce"),
    }, index=indicator.index)
    reset_flags = (
        reset.reindex(work.index).fillna(False).astype(bool)
        if reset is not None else pd.Series(False, index=work.index)
    )
    values = []
    active_rows = 0
    highest_price = highest_indicator = None
    lowest_price = lowest_indicator = None
    for position, (_, row) in enumerate(work.iterrows()):
        if bool(reset_flags.iloc[position]):
            active_rows = 0
            highest_price = highest_indicator = None
            lowest_price = lowest_indicator = None
            values.append(0)
            continue
        if row.isna().any():
            values.append(0)
            continue
        current_high = float(row["price_high"])
        current_low = float(row["price_low"])
        current_indicator = float(row["indicator"])
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
        values.append(signal)
        if highest_price is None or current_high >= highest_price:
            highest_price = current_high
            highest_indicator = current_indicator
        if lowest_price is None or current_low <= lowest_price:
            lowest_price = current_low
            lowest_indicator = current_indicator
        active_rows = min(int(lookback), active_rows + 1)
        if active_rows >= int(lookback):
            active = work.iloc[max(0, position - int(lookback) + 2):position + 1].dropna()
            if not active.empty:
                high_index = active["price_high"].idxmax()
                low_index = active["price_low"].idxmin()
                highest_price = float(active.loc[high_index, "price_high"])
                highest_indicator = float(active.loc[high_index, "indicator"])
                lowest_price = float(active.loc[low_index, "price_low"])
                lowest_indicator = float(active.loc[low_index, "indicator"])
        else:
            pass
    return pd.Series(values, index=work.index)


IMA_WEB_VALIDATION_SOURCES = [
    {
        "source_type": "ima",
        "title": "均线均量扣抵思想",
        "rule": "扣抵方向、均线支撑和量能确认只能作为结构证据，不能单独生成买卖信号",
    },
    {
        "source_type": "web",
        "title": "东方财富作者页：白白胖胖0",
        "url": "https://i.eastmoney.com/2920015446601888",
        "rule": "市场结构、量、价格、技术指标按优先级互相验证",
    },
    {
        "source_type": "web",
        "title": "均线均量扣抵基础公式",
        "url": "https://caifuhao.eastmoney.com/news/20220502203427525523430",
        "rule": "移动平均方向由新值和扣抵值关系决定",
    },
]


def _load_mainline_snapshots(directory):
    snapshots = {}
    if not directory or not Path(directory).exists():
        return snapshots
    for path in Path(directory).glob("sector_mainline_constituents*.csv"):
        match = re.search(r"_(\d{8})$", path.stem)
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except (OSError, ValueError):
            continue
        if frame.empty or "code" not in frame:
            continue
        if match:
            snapshot_date = pd.Timestamp(match.group(1))
        elif "board_date" in frame and frame["board_date"].notna().any():
            snapshot_date = pd.to_datetime(frame["board_date"], errors="coerce").max()
        else:
            continue
        if pd.isna(snapshot_date):
            continue
        members = {}
        for code, group in frame.groupby(frame["code"].astype(str).str.split(".").str[-1].str.zfill(6)):
            members[code] = "、".join(group.get("board", pd.Series(dtype=str)).astype(str).drop_duplicates())
        snapshots[pd.Timestamp(snapshot_date).normalize()] = members
    return snapshots


def report_period_for(date) -> str:
    return latest_visible_report_period(date)


def _load_financial_cache(directory, report_period):
    suffix = pd.Timestamp(report_period).strftime("%Y%m%d")
    rows = {}
    for path in Path(directory).glob(f"*_{suffix}.json"):
        code = path.stem.split("_", 1)[0].zfill(6)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        rows[code] = payload
    return rows


def _financial_point_in_time_status(metrics, observation_date):
    source = str(metrics.get("financial_point_in_time_source") or "").strip()
    announcement = pd.to_datetime(
        metrics.get("announcement_date"), errors="coerce",
    )
    observation = pd.Timestamp(observation_date).normalize()
    visible = (
        source == "announce_time"
        and pd.notna(announcement)
        and announcement.normalize() <= observation
    )
    for key in ("annual_announcement_date", "capital_announcement_date"):
        value = pd.to_datetime(metrics.get(key), errors="coerce")
        if pd.notna(value) and value.normalize() > observation:
            visible = False
    return {
        "financial_point_in_time": bool(visible),
        "financial_point_in_time_source": source or None,
        "announcement_date": (
            None if pd.isna(announcement) else announcement.strftime("%Y-%m-%d")
        ),
    }


def _latest_point_in_time_financial(financial_by_period, code, observation_date, eligible_periods):
    """Return the newest per-company financial cache that was visible on date."""
    for period in sorted(eligible_periods, key=pd.Timestamp, reverse=True):
        metrics = financial_by_period.get(period, {}).get(code)
        if not metrics:
            continue
        status = _financial_point_in_time_status(metrics, observation_date)
        if status["financial_point_in_time"]:
            return period, metrics, status
    return None, None, None


def _validate_required_financial_periods(financial_by_period):
    missing = [
        period for period, rows in sorted(financial_by_period.items())
        if not rows
    ]
    if missing:
        raise RuntimeError(
            "missing financial cache for required point-in-time report periods: "
            + ", ".join(missing)
            + ". Run fundamental_update for these periods before rebuilding backtest candidates."
        )


def _load_raw_price_frame(raw_kline_directory, market, code, start_date, end_date):
    if not raw_kline_directory:
        return pd.DataFrame()
    path = Path(raw_kline_directory) / f"{market}_{code}.csv"
    try:
        header = pd.read_csv(path, nrows=0)
        columns = [
            column for column in (
                "date", "open", "high", "low", "close", "volume",
                "amount", "turnover", "tradestatus",
            ) if column in header.columns
        ]
        frame = pd.read_csv(path, usecols=columns)
    except (OSError, ValueError):
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover", "tradestatus"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        (frame["date"] >= pd.Timestamp(start_date))
        & (frame["date"] <= pd.Timestamp(end_date))
    ].dropna(subset=["date", "close"])
    if frame.empty:
        return frame
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def _attach_akshare_asof_prices(frame, raw_frame, *, source_label="AkShare"):
    """Add observation-day anchored prices from raw + current-qfq data.

    Current AkShare qfq prices are anchored to the provider's latest adjustment
    factor.  For a historical observation date, the visible qfq series can be
    re-anchored by multiplying the visible qfq values by:

        raw_close(observation_date) / current_qfq_close(observation_date)

    The current row's re-anchored close equals the raw close visible that day,
    which is the correct absolute price for value-line comparisons.
    """
    result = frame.copy()
    result["_asof_price_available"] = False
    result["_asof_adjustment_method"] = f"缺少{source_label}不复权价，沿用当前锚点前复权价"
    if raw_frame is None or raw_frame.empty:
        return result
    aligned_raw = raw_frame.reindex(result.index)
    qfq_close = pd.to_numeric(result["close"], errors="coerce")
    raw_close = pd.to_numeric(aligned_raw.get("close"), errors="coerce")
    scale = raw_close.div(qfq_close.where(qfq_close > 0))
    valid = scale.gt(0) & scale.notna()
    for column in ("open", "high", "low", "close"):
        raw_column = pd.to_numeric(aligned_raw.get(column), errors="coerce")
        result[f"_asof_{column}"] = raw_column.where(
            raw_column.gt(0),
            pd.to_numeric(result[column], errors="coerce").mul(scale),
        )
    result["_asof_rebase_scale"] = scale
    result["_asof_price_available"] = valid
    result.loc[valid, "_asof_adjustment_method"] = f"{source_label}不复权价反推观察日锚定前复权价"
    return result


def _load_prices(
    kline_directory,
    codes,
    start_date,
    end_date,
    *,
    raw_kline_directory=None,
    price_source="akshare",
):
    result = {}
    source_label = "MiniQMT" if price_source == "miniqmt" else "AkShare"
    for code in codes:
        market = "sh" if code.startswith(("6", "9")) else "sz"
        path = Path(kline_directory) / f"{market}_{code}.csv"
        meta_path = Path(f"{path}.meta.json")
        qfq_anchor_date = pd.NaT
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            qfq_anchor_date = pd.to_datetime(
                payload.get("qfq_anchor_date"),
                errors="coerce",
            )
        except (OSError, ValueError, TypeError):
            qfq_anchor_date = pd.NaT
        if price_source == "miniqmt" and pd.isna(qfq_anchor_date):
            qfq_anchor_date = pd.Timestamp(end_date).normalize()
        try:
            header = pd.read_csv(path, nrows=0)
            columns = [
                column for column in (
                    "date", "open", "high", "low", "close", "volume",
                    "amount", "turnover", "tradestatus",
                ) if column in header.columns
            ]
            frame = pd.read_csv(path, usecols=columns)
        except (OSError, ValueError):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount", "turnover", "tradestatus"):
            if column not in frame:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
        ].dropna(subset=["date", "close"])
        if not frame.empty:
            frame["_qfq_anchor_date"] = qfq_anchor_date
            frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
            raw_frame = _load_raw_price_frame(
                raw_kline_directory,
                market,
                code,
                start_date,
                end_date,
            )
            frame["_price_source"] = price_source
            result[code] = _attach_akshare_asof_prices(
                frame,
                raw_frame,
                source_label=source_label,
            )
    return result


def _candidate_feature_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Precompute rolling candidate features once per symbol."""
    frame = price_frame.copy()
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    if "amount" in frame:
        raw_amount = pd.to_numeric(frame["amount"], errors="coerce")
    else:
        raw_amount = pd.Series(float("nan"), index=frame.index)
    if "tradestatus" in frame:
        tradestatus = pd.to_numeric(frame["tradestatus"], errors="coerce")
    else:
        tradestatus = pd.Series(float("nan"), index=frame.index)
    valid_price_bar = (
        open_.gt(0)
        & high.gt(0)
        & low.gt(0)
        & close.gt(0)
        & volume.gt(0)
        & high.ge(low)
        & high.ge(open_)
        & high.ge(close)
        & low.le(open_)
        & low.le(close)
    )
    tradestatus_missing = tradestatus.isna()
    is_traded_bar = valid_price_bar & (tradestatus_missing | tradestatus.eq(1))
    frame["_valid_price_bar"] = valid_price_bar
    frame["_tradestatus"] = tradestatus
    frame["_is_traded_bar"] = is_traded_bar
    frame["_data_status"] = "traded"
    frame.loc[~valid_price_bar, "_data_status"] = "invalid_price_bar"
    frame.loc[valid_price_bar & ~tradestatus_missing & ~tradestatus.eq(1), "_data_status"] = "suspended_or_no_trade"
    clean_volume = volume.where(is_traded_bar)
    daily_return = close.pct_change().where(valid_price_bar)
    volume_change = clean_volume.pct_change()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    frame["_ma20_rising"] = ma20.notna() & ma20.shift(5).notna() & (ma20 > ma20.shift(5))
    frame["_ma60_rising"] = ma60.notna() & ma60.shift(5).notna() & (ma60 > ma60.shift(5))
    frame["_above_ma20"] = ma20.notna() & (close >= ma20)
    frame["_near_ma20"] = ma20.notna() & ma20.gt(0) & (close.div(ma20).sub(1).abs() <= 0.05)
    prior_high = close.shift(1).rolling(21).max()
    frame["_near_breakout"] = prior_high.notna() & prior_high.gt(0) & (close >= prior_high * 0.97)
    high_21 = close.rolling(21, min_periods=10).max()
    low_21 = close.rolling(21, min_periods=10).min()
    range_21 = high_21.div(low_21.where(low_21 > 0)).sub(1)
    frame["_range_21_pct"] = range_21
    frame["_close_position_21"] = close.sub(low_21).div(high_21.sub(low_21).where(high_21 > low_21))
    volume_base = pd.concat([
        volume.rolling(5).mean().shift(1),
        volume.rolling(10).mean().shift(1),
    ], axis=1).max(axis=1).fillna(0.0)
    frame["_volume_ratio"] = volume.div(volume_base.where(volume_base > 0)).fillna(0.0)
    proxy_amount = close * clean_volume
    amount = raw_amount.where(raw_amount.gt(0), proxy_amount)
    frame["_raw_amount"] = raw_amount
    frame["_amount_source"] = "raw"
    frame.loc[raw_amount.isna() | raw_amount.le(0), "_amount_source"] = "close_x_volume_proxy"
    frame.loc[amount.isna() | amount.le(0), "_amount_source"] = "missing"
    frame["_avg_amount_20"] = amount.rolling(20, min_periods=5).mean()
    volume_node = frame["_volume_ratio"].ge(1.2) & ma20.notna() & close.ge(ma20)
    frame["_volume_node_count_60"] = volume_node.rolling(60, min_periods=1).sum()
    node_close = close.where(volume_node).ffill()
    frame["_volume_node_distance"] = close.div(node_close.where(node_close > 0)).sub(1)
    for period in (5, 10, 20):
        frame[f"_deduction_{period}"] = volume.shift(period).notna() & (
            volume > volume.shift(period)
        )
    for period in (5, 10, 20, 60, 120):
        base = close.shift(period)
        frame[f"_return_{period}"] = close.div(base.where(base > 0)).sub(1)
    frame["_momentum_60_ex5"] = close.shift(5).div(close.shift(65).where(close.shift(65) > 0)).sub(1)
    frame["_momentum_120_ex20"] = close.shift(20).div(close.shift(140).where(close.shift(140) > 0)).sub(1)
    frame["_positive_day_ratio_60"] = daily_return.gt(0).rolling(60, min_periods=20).mean()
    downside_return = daily_return.where(daily_return < 0, 0.0)
    frame["_downside_volatility_60"] = downside_return.rolling(60, min_periods=20).std()
    high_60 = high.rolling(60, min_periods=20).max()
    low_60 = low.rolling(60, min_periods=20).min()
    high_120 = high.rolling(120, min_periods=40).max()
    low_120 = low.rolling(120, min_periods=40).min()
    frame["_alpha_close_position_60"] = close.sub(low_60).div(high_60.sub(low_60).where(high_60 > low_60)).clip(0.0, 1.0)
    frame["_alpha_close_position_120"] = close.sub(low_120).div(high_120.sub(low_120).where(high_120 > low_120)).clip(0.0, 1.0)
    frame["_alpha_volume_price_corr_20"] = daily_return.rolling(20, min_periods=10).corr(volume_change)
    frame["_alpha_turnover_expansion_20"] = clean_volume.rolling(5, min_periods=3).mean().div(
        clean_volume.rolling(20, min_periods=10).mean().where(clean_volume.rolling(20, min_periods=10).mean() > 0)
    ).sub(1.0)
    intraday_range = high.sub(low)
    intraday_strength = close.mul(2).sub(high).sub(low).div(intraday_range.where(intraday_range > 0))
    frame["_alpha_intraday_strength_20"] = intraday_strength.rolling(20, min_periods=10).mean()
    gap_up = open_.div(close.shift(1).where(close.shift(1) > 0)).sub(1.0).gt(0.03)
    frame["_alpha_gap_5d_count"] = gap_up.rolling(5, min_periods=1).sum()
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
    kd_strength = kd_k_high.add(kd_d_high).div(2)
    wr_strength = (100 - wr10).add(100 - wr20).div(2)
    kd_divergence = _divergence_series(
        high, low, kd_strength, (kd_k_low < 20) & (kd_d_low < 20)
    )
    rsi_divergence = _divergence_series(high, low, rsi999, rsi999 < 50)
    wr_divergence = _divergence_series(high, low, wr_strength, (wr10 >= 80) & (wr20 >= 80))
    bearish_divergence_count = (
        kd_divergence.eq(-1).astype(int)
        + rsi_divergence.eq(-1).astype(int)
        + wr_divergence.eq(-1).astype(int)
    )
    bullish_divergence_count = (
        kd_divergence.eq(1).astype(int)
        + rsi_divergence.eq(1).astype(int)
        + wr_divergence.eq(1).astype(int)
    )
    frame["_kd_k_high"] = kd_k_high
    frame["_kd_d_high"] = kd_d_high
    frame["_kd_k_low"] = kd_k_low
    frame["_kd_d_low"] = kd_d_low
    frame["_kd_divergence"] = kd_divergence
    frame["_rsi999"] = rsi999
    frame["_rsi_divergence"] = rsi_divergence
    frame["_wr10"] = wr10
    frame["_wr20"] = wr20
    frame["_wr_divergence"] = wr_divergence
    frame["_bearish_divergence_count"] = bearish_divergence_count
    frame["_bullish_divergence_count"] = bullish_divergence_count
    frame["_divergence_score"] = bullish_divergence_count.mul(8).sub(bearish_divergence_count.mul(10))
    rolling_high = close.rolling(120, min_periods=1).max()
    frame["_distance_120d_high"] = close.div(rolling_high.where(rolling_high > 0)).sub(1)
    frame["_volatility_20"] = daily_return.rolling(20).std()
    frame["_drawdown_60"] = close.div(close.rolling(60, min_periods=20).max()).sub(1)
    frame["_ma20_slope"] = ma20.div(ma20.shift(10).where(ma20.shift(10) > 0)).sub(1)
    frame["_ma60_slope"] = ma60.div(ma60.shift(20).where(ma60.shift(20) > 0)).sub(1)
    frame["_right_acceleration"] = frame["_return_20"].sub(frame["_return_60"].div(3))
    return frame


def _trade_basis_from_feature_row(row) -> dict:
    ma20_rising = bool(row.get("_ma20_rising", False))
    ma60_rising = bool(row.get("_ma60_rising", False))
    above_ma20 = bool(row.get("_above_ma20", False))
    near_ma20 = bool(row.get("_near_ma20", False))
    near_breakout = bool(row.get("_near_breakout", False))
    volume_ratio = float(row.get("_volume_ratio") or 0.0)
    deduction_periods = [
        period for period in (5, 10, 20)
        if bool(row.get(f"_deduction_{period}", False))
    ]

    score = 0.0
    reasons = []
    if ma20_rising and ma60_rising:
        score += 4.0
        reasons.append("MA20/MA60同步上扬")
    elif ma20_rising:
        score += 2.0
        reasons.append("MA20上扬")
    if above_ma20:
        score += 2.0
        reasons.append("收盘站上MA20")
    elif near_ma20 and ma20_rising:
        score += 2.0
        reasons.append("贴近上扬MA20支撑")
    if near_breakout:
        reasons.append("候选观察：接近21日收盘高点（非买点）")
    if volume_ratio >= 1.2:
        score += 2.0
        reasons.append(f"量能高于5/10日基准{volume_ratio:.2f}倍")
    if len(deduction_periods) >= 2:
        score += 1.0
        reasons.append("多周期均量扣低走高")

    alignment = "trade_ready" if score >= 7 else "watch" if score >= 4 else "fundamental_only"
    return {
        "trade_basis_score": round(score, 3),
        "trade_basis_reason": "；".join(reasons) or "基本面入选，等待价格/量能买点",
        "technical_alignment": alignment,
        "ma20_rising": bool(ma20_rising),
        "ma60_rising": bool(ma60_rising),
        "above_ma20": bool(above_ma20),
        "near_ma20": bool(near_ma20),
        "near_21d_close_high": bool(near_breakout),
        "known_volume_ratio": round(volume_ratio, 4),
        "volume_deduction_periods": ",".join(map(str, deduction_periods)),
        "ima_web_validation": "aligned" if score >= 4 else "needs_price_confirmation",
        "validation_sources": IMA_WEB_VALIDATION_SOURCES,
    }


def _trade_basis_snapshot(price_frame: pd.DataFrame, date) -> dict:
    """Score model candidates with only information visible at observation close."""
    history = _candidate_feature_frame(price_frame.loc[:date])
    if history.empty:
        return {
            "trade_basis_score": 0.0,
            "trade_basis_reason": "缺少观察日行情，等待补数",
            "technical_alignment": "missing_price",
        }
    return _trade_basis_from_feature_row(history.iloc[-1])


def _structure_proximity_from_feature_row(row) -> dict:
    range_21 = pd.to_numeric(row.get("_range_21_pct"), errors="coerce")
    close_position = pd.to_numeric(row.get("_close_position_21"), errors="coerce")
    volume_node_count = pd.to_numeric(row.get("_volume_node_count_60"), errors="coerce")
    volume_node_distance = pd.to_numeric(row.get("_volume_node_distance"), errors="coerce")
    near_breakout = bool(row.get("_near_breakout", False))
    ma20_rising = bool(row.get("_ma20_rising", False))
    ma60_rising = bool(row.get("_ma60_rising", False))
    above_ma20 = bool(row.get("_above_ma20", False))
    volume_ratio = float(row.get("_volume_ratio") or 0.0)

    score = 0.0
    reasons = []
    if pd.notna(range_21) and pd.notna(close_position):
        if float(range_21) <= 0.18 and float(close_position) >= 0.72:
            score += 24.0
            reasons.append("21日平台收紧，收盘靠近平台上沿")
        elif float(range_21) <= 0.25 and float(close_position) >= 0.65:
            score += 14.0
            reasons.append("21日平台基本收敛，收盘处在区间上半部")
    if near_breakout:
        score += 18.0
        reasons.append("接近21日收盘高点，候选层提前观察")
    if pd.notna(volume_node_count):
        if float(volume_node_count) >= 3:
            score += 18.0
            reasons.append("近60日有多次放量站上均线节点")
        elif float(volume_node_count) >= 1:
            score += 10.0
            reasons.append("近60日出现过放量站上均线节点")
    if pd.notna(volume_node_distance):
        if -0.03 <= float(volume_node_distance) <= 0.12:
            score += 14.0
            reasons.append("现价仍贴近最近有效量价节点")
        elif 0.12 < float(volume_node_distance) <= 0.25:
            score += 6.0
            reasons.append("现价已离开放量节点但趋势仍可观察")
    if ma20_rising and above_ma20:
        score += 14.0
        reasons.append("收盘站在上扬MA20上方")
    if ma20_rising and ma60_rising:
        score += 8.0
        reasons.append("MA20和MA60同步上行")
    if volume_ratio >= 1.0:
        score += 4.0
        reasons.append("观察日量能不低于短期基准")

    return {
        "range_21_pct": None if pd.isna(range_21) else round(float(range_21), 6),
        "close_position_21": None if pd.isna(close_position) else round(float(close_position), 6),
        "volume_node_count_60": None if pd.isna(volume_node_count) else int(float(volume_node_count)),
        "volume_node_distance": None if pd.isna(volume_node_distance) else round(float(volume_node_distance), 6),
        "structure_proximity_score": round(min(score, 100.0), 3),
        "structure_proximity_reason": "；".join(reasons) or "结构买点尚未靠近，只保留基础观察",
    }


def _divergence_from_feature_row(row) -> dict:
    kd_divergence = pd.to_numeric(row.get("_kd_divergence"), errors="coerce")
    rsi_divergence = pd.to_numeric(row.get("_rsi_divergence"), errors="coerce")
    wr_divergence = pd.to_numeric(row.get("_wr_divergence"), errors="coerce")
    bearish_count = int(pd.to_numeric(row.get("_bearish_divergence_count"), errors="coerce") or 0)
    bullish_count = int(pd.to_numeric(row.get("_bullish_divergence_count"), errors="coerce") or 0)
    score = pd.to_numeric(row.get("_divergence_score"), errors="coerce")
    labels = []
    for name, value in (("高低KD", kd_divergence), ("RSI", rsi_divergence), ("WR", wr_divergence)):
        if pd.isna(value) or int(value) == 0:
            continue
        labels.append(f"{name}{'底背离' if int(value) > 0 else '顶背离'}")
    if not labels:
        labels.append("暂无明显指标背离")
    if bearish_count >= 2:
        labels.append("多指标顶背离，右侧追高需要降权")
    elif bullish_count >= 2:
        labels.append("多指标底背离，右侧观察价值提高")
    return {
        "kd_k_high": None if pd.isna(row.get("_kd_k_high")) else float(row.get("_kd_k_high")),
        "kd_d_high": None if pd.isna(row.get("_kd_d_high")) else float(row.get("_kd_d_high")),
        "kd_k_low": None if pd.isna(row.get("_kd_k_low")) else float(row.get("_kd_k_low")),
        "kd_d_low": None if pd.isna(row.get("_kd_d_low")) else float(row.get("_kd_d_low")),
        "kd_divergence": None if pd.isna(kd_divergence) else int(kd_divergence),
        "rsi999": None if pd.isna(row.get("_rsi999")) else float(row.get("_rsi999")),
        "rsi_divergence": None if pd.isna(rsi_divergence) else int(rsi_divergence),
        "wr10": None if pd.isna(row.get("_wr10")) else float(row.get("_wr10")),
        "wr20": None if pd.isna(row.get("_wr20")) else float(row.get("_wr20")),
        "wr_divergence": None if pd.isna(wr_divergence) else int(wr_divergence),
        "bearish_divergence_count": bearish_count,
        "bullish_divergence_count": bullish_count,
        "divergence_score": 0.0 if pd.isna(score) else float(score),
        "divergence_reason": "；".join(labels),
    }


def _leadership_from_feature_row(row) -> dict:
    def number(value):
        return None if pd.isna(value) else float(value)

    return_20d = number(row.get("_return_20"))
    return_60d = number(row.get("_return_60"))
    return_120d = number(row.get("_return_120"))
    distance_high = number(row.get("_distance_120d_high"))

    def scaled(value, worst: float, best: float, points: float) -> float:
        if value is None:
            return 0.0
        ratio = (float(value) - worst) / (best - worst)
        return min(points, max(0.0, ratio * points))

    score_20 = scaled(return_20d, -0.05, 0.30, 10.0)
    score_60 = scaled(return_60d, 0.00, 0.50, 10.0)
    score_120 = scaled(return_120d, 0.00, 0.60, 6.0)
    high_score = 4.0 if distance_high is not None and distance_high >= -0.05 else (
        2.0 if distance_high is not None and distance_high >= -0.12 else 0.0
    )
    score = score_20 + score_60 + score_120 + high_score
    reasons = []
    for label, value in (("20日", return_20d), ("60日", return_60d), ("120日", return_120d)):
        if value is not None:
            reasons.append(f"{label}强度{value:+.1%}")
    if distance_high is not None:
        reasons.append(f"距120日高点{distance_high:+.1%}")
    return {
        "return_20d": None if return_20d is None else round(return_20d, 6),
        "return_60d": None if return_60d is None else round(return_60d, 6),
        "return_120d": None if return_120d is None else round(return_120d, 6),
        "distance_120d_high": None if distance_high is None else round(distance_high, 6),
        "leadership_score": round(score, 3),
        "leadership_reason": "；".join(reasons),
        "long_term_structure_favorable": bool(score >= 15.0),
    }


def _leadership_snapshot(price_frame: pd.DataFrame, date) -> dict:
    """Rank durable price leadership using only bars visible at observation close."""
    history = _candidate_feature_frame(price_frame.loc[:date])
    if history.empty:
        return {
            "return_20d": None,
            "return_60d": None,
            "return_120d": None,
            "distance_120d_high": None,
            "leadership_score": 0.0,
            "leadership_reason": "缺少观察日行情",
            "long_term_structure_favorable": False,
        }
    return _leadership_from_feature_row(history.iloc[-1])


def _right_strength_from_feature_row(row) -> dict:
    """Capture early right-side turns without changing the fundamental gate."""
    return_20d = pd.to_numeric(row.get("_return_20"), errors="coerce")
    return_60d = pd.to_numeric(row.get("_return_60"), errors="coerce")
    return_120d = pd.to_numeric(row.get("_return_120"), errors="coerce")
    distance_high = pd.to_numeric(row.get("_distance_120d_high"), errors="coerce")
    trade_basis = _trade_basis_from_feature_row(row)
    trade_basis_score = float(trade_basis["trade_basis_score"])
    volume_ratio = float(row.get("_volume_ratio") or 0.0)
    ma20_rising = bool(row.get("_ma20_rising", False))
    ma60_rising = bool(row.get("_ma60_rising", False))
    above_ma20 = bool(row.get("_above_ma20", False))
    near_breakout = bool(row.get("_near_breakout", False))

    score = 0.0
    reasons = []
    score += min(max(trade_basis_score, 0.0), 12.0) * 1.5
    if pd.notna(return_20d):
        if float(return_20d) >= 0.30:
            score += 10.0
            reasons.append(f"20日转强{float(return_20d):+.1%}")
        elif float(return_20d) >= 0.18:
            score += 6.0
            reasons.append(f"20日转强{float(return_20d):+.1%}")
    if pd.notna(return_60d):
        if float(return_60d) >= 0.20:
            score += 6.0
            reasons.append(f"60日确认{float(return_60d):+.1%}")
        elif float(return_60d) >= 0.0:
            score += 3.0
            reasons.append(f"60日止跌转正{float(return_60d):+.1%}")
    if pd.notna(return_120d) and float(return_120d) >= 0:
        score += 3.0
        reasons.append(f"120日不再下行{float(return_120d):+.1%}")
    if pd.notna(distance_high):
        if float(distance_high) >= -0.05:
            score += 6.0
            reasons.append("距离120日高点5%内")
        elif float(distance_high) >= -0.15:
            score += 3.0
            reasons.append("距离120日高点15%内")
    if ma20_rising and above_ma20:
        score += 4.0
        reasons.append("站上上扬MA20")
    if ma20_rising and ma60_rising:
        score += 4.0
        reasons.append("MA20/MA60同步上扬")
    if near_breakout:
        score += 2.0
        reasons.append("接近21日收盘高点")
    if volume_ratio >= 1.2:
        score += 3.0
        reasons.append(f"量能放大{volume_ratio:.2f}倍")
    return {
        "right_strength_score": round(score, 3),
        "right_strength_reason": "；".join(reasons) or "右侧强度尚未形成",
    }


def _rank_right_side_candidates(rows: list[dict]) -> list[dict]:
    """Rank candidates with a visible-data multi-factor quant model."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)

    def number_column(name):
        if name not in frame:
            return pd.Series(float("nan"), index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce")

    def pct_rank(series, *, ascending=True):
        series = pd.to_numeric(series, errors="coerce")
        return series.rank(pct=True, ascending=ascending).fillna(0.5)

    quality = number_column("quality_score")
    growth = number_column("earnings_yoy")
    market_cap = number_column("mktcap")
    return_5 = number_column("return_5d")
    return_20 = number_column("return_20d")
    return_60 = number_column("return_60d")
    return_120 = number_column("return_120d")
    distance_high = number_column("distance_120d_high")
    momentum_60_ex5 = number_column("momentum_60_ex5")
    momentum_120_ex20 = number_column("momentum_120_ex20")
    avg_amount_20 = number_column("avg_amount_20")
    volatility = number_column("volatility_20")
    downside_volatility = number_column("downside_volatility_60")
    drawdown = number_column("drawdown_60")
    positive_ratio = number_column("positive_day_ratio_60")
    ma20_slope = number_column("ma20_slope")
    ma60_slope = number_column("ma60_slope")
    alpha_volume_price_corr = number_column("alpha_volume_price_corr_20")
    alpha_turnover_expansion = number_column("alpha_turnover_expansion_20")
    alpha_close_position_60 = number_column("alpha_close_position_60")
    alpha_close_position_120 = number_column("alpha_close_position_120")
    alpha_intraday_strength = number_column("alpha_intraday_strength_20")
    alpha_gap_count = number_column("alpha_gap_5d_count")
    divergence_score = number_column("divergence_score")
    bearish_divergence_count = number_column("bearish_divergence_count")
    structure_proximity_score = number_column("structure_proximity_score")
    volume_node_count = number_column("volume_node_count_60")
    volume_node_distance = number_column("volume_node_distance")
    close_position_21 = number_column("close_position_21")
    range_21 = number_column("range_21_pct")

    quality_rank = pct_rank(quality)
    growth_rank = pct_rank(growth.clip(lower=-0.5, upper=2.0))
    liquidity_rank = pct_rank(market_cap) * 0.40 + pct_rank(avg_amount_20) * 0.60
    momentum_rank = (
        pct_rank(momentum_60_ex5) * 0.45
        + pct_rank(momentum_120_ex20) * 0.30
        + pct_rank(return_20) * 0.25
    )
    low_risk_rank = (
        pct_rank(drawdown) * 0.45
        + pct_rank(-volatility) * 0.30
        + pct_rank(-downside_volatility) * 0.25
    )
    trend_stability_rank = (
        pct_rank(positive_ratio) * 0.45
        + pct_rank(ma20_slope) * 0.30
        + pct_rank(ma60_slope) * 0.25
    )
    relative_strength_rank = (
        pct_rank(return_60) * 0.35
        + pct_rank(return_120) * 0.25
        + pct_rank(distance_high) * 0.25
        + pct_rank(return_20) * 0.15
    )
    alpha_rank = (
        pct_rank(alpha_close_position_60) * 0.28
        + pct_rank(alpha_close_position_120) * 0.20
        + pct_rank(alpha_volume_price_corr.clip(lower=-1.0, upper=1.0)) * 0.20
        + pct_rank(alpha_turnover_expansion.clip(lower=-0.5, upper=1.5)) * 0.17
        + pct_rank(alpha_intraday_strength.clip(lower=-1.0, upper=1.0)) * 0.15
    )
    overheat_control_rank = (
        pct_rank(-return_5.abs()) * 0.35
        + pct_rank(-return_20.clip(lower=0.0)) * 0.25
        + pct_rank(drawdown) * 0.20
        + pct_rank(-alpha_gap_count) * 0.20
    )
    divergence_rank = (
        pct_rank(divergence_score) * 0.70
        + pct_rank(-bearish_divergence_count) * 0.30
    )
    trend_efficiency = return_60.clip(lower=-0.50, upper=1.50).div(
        downside_volatility.mul(8.0).abs()
        .add(volatility.mul(3.0).abs())
        .add(drawdown.abs())
        .add(0.05)
    )
    trend_efficiency_rank = (
        pct_rank(trend_efficiency) * 0.55
        + trend_stability_rank * 0.25
        + low_risk_rank * 0.20
    )
    controlled_pullback = (
        (1.0 - (distance_high.abs().sub(0.08).abs().div(0.24))).clip(0.0, 1.0)
        .where(return_60.fillna(-1.0).ge(0), 0.0)
    )
    structure_rank = (
        pct_rank(structure_proximity_score) * 0.35
        + pct_rank(volume_node_count) * 0.20
        + pct_rank(-volume_node_distance.abs()) * 0.15
        + pct_rank(close_position_21) * 0.15
        + pct_rank(-range_21) * 0.15
    )
    volume_confirm_rank = (
        pct_rank(alpha_volume_price_corr.clip(lower=-1.0, upper=1.0)) * 0.35
        + pct_rank(alpha_turnover_expansion.clip(lower=-0.5, upper=1.5)) * 0.25
        + pct_rank(volume_node_count) * 0.25
        + pct_rank(alpha_intraday_strength.clip(lower=-1.0, upper=1.0)) * 0.15
    )
    payoff_rank = (
        trend_efficiency_rank * 0.30
        + pct_rank(controlled_pullback) * 0.20
        + structure_rank * 0.20
        + volume_confirm_rank * 0.15
        + divergence_rank * 0.10
        + growth_rank * 0.05
    )
    attack_score = (
        quality_rank * 10.0
        + growth_rank * 12.0
        + liquidity_rank * 8.0
        + momentum_rank * 24.0
        + relative_strength_rank * 12.0
        + alpha_rank * 10.0
        + trend_stability_rank * 9.0
        + payoff_rank * 18.0
        + structure_rank * 8.0
        + volume_confirm_rank * 7.0
        + low_risk_rank * 6.0
        + overheat_control_rank * 6.0
        + divergence_rank * 4.0
    )
    defense_score = (
        quality_rank * 15.0
        + growth_rank * 15.0
        + liquidity_rank * 10.0
        + momentum_rank * 22.0
        + payoff_rank * 15.0
        + trend_efficiency_rank * 10.0
        + low_risk_rank * 15.0
        + trend_stability_rank * 10.0
        + overheat_control_rank * 5.0
        + divergence_rank * 3.0
    )
    quant_score = (
        attack_score * 0.75
        + defense_score * 0.25
    )
    median_return_20 = return_20.median(skipna=True)
    median_return_60 = return_60.median(skipna=True)
    positive_20_ratio = return_20.gt(0).mean()
    if (
        pd.notna(median_return_20)
        and pd.notna(median_return_60)
        and median_return_20 >= 0.08
        and median_return_60 >= 0.12
        and positive_20_ratio >= 0.60
    ):
        market_regime = "进攻"
        regime_score = (
            momentum_rank * 5.0
            + relative_strength_rank * 4.0
            + volume_confirm_rank * 3.0
            + payoff_rank * 2.0
        )
    elif (
        pd.notna(median_return_20)
        and pd.notna(median_return_60)
        and (
            median_return_20 < 0.0
            or median_return_60 < 0.0
            or positive_20_ratio < 0.45
        )
    ):
        market_regime = "防守"
        regime_score = (
            low_risk_rank * 5.0
            + overheat_control_rank * 3.0
            + structure_rank * 3.0
            + payoff_rank * 3.0
        )
    else:
        market_regime = "平衡"
        regime_score = (
            payoff_rank * 5.0
            + trend_efficiency_rank * 4.0
            + structure_rank * 3.0
            + volume_confirm_rank * 2.0
        )
    quant_score = quant_score + regime_score
    if "mainline_snapshot_fresh" in frame:
        mainline_fresh = frame["mainline_snapshot_fresh"].fillna(False).astype(bool)
    else:
        mainline_fresh = pd.Series(False, index=frame.index)
    if "mainline_boards" in frame:
        mainline_boards = frame["mainline_boards"].fillna("").astype(str).str.strip()
    else:
        mainline_boards = pd.Series("", index=frame.index)
    ranked = []
    for index, item in enumerate(rows):
        row = dict(item)
        row["quant_quality_rank"] = round(float(quality_rank.iloc[index]) * 100.0, 3)
        row["quant_growth_rank"] = round(float(growth_rank.iloc[index]) * 100.0, 3)
        row["quant_liquidity_rank"] = round(float(liquidity_rank.iloc[index]) * 100.0, 3)
        row["quant_momentum_rank"] = round(float(momentum_rank.iloc[index]) * 100.0, 3)
        row["quant_low_risk_rank"] = round(float(low_risk_rank.iloc[index]) * 100.0, 3)
        row["quant_trend_stability_rank"] = round(float(trend_stability_rank.iloc[index]) * 100.0, 3)
        row["quant_overheat_control_rank"] = round(float(overheat_control_rank.iloc[index]) * 100.0, 3)
        row["quant_alpha_rank"] = round(float(alpha_rank.iloc[index]) * 100.0, 3)
        row["quant_divergence_rank"] = round(float(divergence_rank.iloc[index]) * 100.0, 3)
        row["quant_trend_efficiency_rank"] = round(float(trend_efficiency_rank.iloc[index]) * 100.0, 3)
        row["quant_payoff_rank"] = round(float(payoff_rank.iloc[index]) * 100.0, 3)
        row["quant_structure_rank"] = round(float(structure_rank.iloc[index]) * 100.0, 3)
        row["quant_volume_confirm_rank"] = round(float(volume_confirm_rank.iloc[index]) * 100.0, 3)
        row["quant_market_regime"] = market_regime
        row["right_momentum_rank"] = row["quant_momentum_rank"]
        row["right_trend_rank"] = row["quant_trend_stability_rank"]
        row["right_volume_rank"] = row["quant_liquidity_rank"]
        row["right_risk_rank"] = row["quant_low_risk_rank"]
        row["right_acceleration_rank"] = row["quant_overheat_control_rank"]
        row["right_mainline_rank"] = 100.0 if bool(mainline_fresh.iloc[index] and mainline_boards.iloc[index]) else 0.0
        row["right_structure_rank"] = row["quant_structure_rank"]
        row["right_volume_node_rank"] = row["quant_volume_confirm_rank"]
        row["right_quant_score"] = round(float(quant_score.iloc[index]), 3)
        row["right_quant_reason"] = (
            f"质量排名{row['quant_quality_rank']:.1f}；"
            f"成长排名{row['quant_growth_rank']:.1f}；"
            f"流动性排名{row['quant_liquidity_rank']:.1f}；"
            f"动量强度排名{row['quant_momentum_rank']:.1f}；"
            f"相对强度排名{float(relative_strength_rank.iloc[index]) * 100.0:.1f}；"
            f"趋势效率排名{row['quant_trend_efficiency_rank']:.1f}；"
            f"盈亏比代理排名{row['quant_payoff_rank']:.1f}；"
            f"结构位置排名{row['quant_structure_rank']:.1f}；"
            f"量价确认排名{row['quant_volume_confirm_rank']:.1f}；"
            f"市场状态{market_regime}；"
            f"60日回撤风险排名{row['quant_low_risk_rank']:.1f}；"
            f"趋势稳定排名{row['quant_trend_stability_rank']:.1f}；"
            f"短期不过热排名{row['quant_overheat_control_rank']:.1f}；"
            f"背离环境排名{row['quant_divergence_rank']:.1f}"
        )
        ranked.append(row)
    ranked.sort(key=lambda item: (-float(item["right_quant_score"]), item["code"]))
    for rank, row in enumerate(ranked, start=1):
        row["right_quant_rank"] = rank
    return ranked


def _passes_fundamental_gate(quality, yoy, market_cap) -> bool:
    """Keep the right-side lane behind the same fundamental hard gate."""
    quality = pd.to_numeric(quality, errors="coerce")
    yoy = pd.to_numeric(yoy, errors="coerce")
    market_cap = pd.to_numeric(market_cap, errors="coerce")
    return bool(
        pd.notna(quality)
        and pd.notna(yoy)
        and pd.notna(market_cap)
        and float(quality) >= 70.0
        and float(yoy) >= 0.10
        and float(market_cap) >= 100.0
    )


def _right_quant_selection_rows(rows: list[dict]) -> list[dict]:
    """Convert the ranked right-side pool into executable quant-right rows."""
    selected = []
    for ranked in _rank_right_side_candidates(rows):
        rank = int(ranked.get("right_quant_rank") or 999999)
        def ranked_number(key, default=0.0):
            value = pd.to_numeric(ranked.get(key), errors="coerce")
            return float(value) if pd.notna(value) else float(default)

        score = ranked_number("right_quant_score")
        return_20d = ranked_number("return_20d")
        return_60d = ranked_number("return_60d")
        avg_amount_value = pd.to_numeric(ranked.get("avg_amount_20"), errors="coerce")
        avg_amount_20 = (
            float(avg_amount_value)
            if pd.notna(avg_amount_value)
            else 0.0
        )
        momentum_rank = ranked_number("quant_momentum_rank")
        low_risk_rank = ranked_number("quant_low_risk_rank")
        trend_stability_rank = ranked_number("quant_trend_stability_rank")
        overheat_control_rank = ranked_number("quant_overheat_control_rank")
        alpha_rank = ranked_number("quant_alpha_rank")
        payoff_rank = ranked_number("quant_payoff_rank")
        trend_efficiency_rank = ranked_number("quant_trend_efficiency_rank")
        structure_rank = ranked_number("quant_structure_rank")
        volume_confirm_rank = ranked_number("quant_volume_confirm_rank")
        drawdown_60 = ranked_number("drawdown_60", -1.0)

        qualified = (
            rank <= 90
            and score >= 68.0
            and avg_amount_20 >= 500_000_000.0
            and momentum_rank >= 62.0
            and alpha_rank >= 50.0
            and trend_stability_rank >= 52.0
            and low_risk_rank >= 30.0
            and overheat_control_rank >= 30.0
            and payoff_rank >= 55.0
            and trend_efficiency_rank >= 45.0
            and return_20d >= 0.0
            and drawdown_60 >= -0.32
        )
        strong_trend_exception = (
            rank <= 60
            and score >= 72.0
            and avg_amount_20 >= 1_000_000_000.0
            and momentum_rank >= 78.0
            and alpha_rank >= 45.0
            and trend_stability_rank >= 50.0
            and low_risk_rank >= 25.0
            and overheat_control_rank >= 30.0
            and payoff_rank >= 45.0
            and return_20d <= 0.70
            and drawdown_60 >= -0.38
        )
        high_payoff_setup = (
            rank <= 85
            and score >= 66.0
            and avg_amount_20 >= 1_000_000_000.0
            and momentum_rank >= 52.0
            and alpha_rank >= 45.0
            and low_risk_rank >= 60.0
            and payoff_rank >= 68.0
            and structure_rank >= 55.0
            and volume_confirm_rank >= 45.0
            and overheat_control_rank >= 45.0
            and return_20d >= 0.04
            and drawdown_60 >= -0.16
        )
        asymmetric_pivot_watch = (
            rank <= 100
            and score >= 74.0
            and avg_amount_20 >= 1_000_000_000.0
            and alpha_rank >= 70.0
            and structure_rank >= 55.0
            and volume_confirm_rank >= 60.0
            and low_risk_rank >= 40.0
            and overheat_control_rank >= 25.0
            and 0.08 <= return_20d <= 0.25
            and return_60d >= 0.0
            and drawdown_60 >= -0.12
        )
        compact_attack_core = (
            rank <= 50
            and score >= 85.0
            and avg_amount_20 >= 450_000_000.0
            and alpha_rank >= 70.0
            and structure_rank >= 80.0
            and volume_confirm_rank >= 60.0
            and low_risk_rank >= 60.0
            and trend_stability_rank >= 60.0
            and overheat_control_rank >= 40.0
            and return_20d >= 0.04
            and return_60d >= 0.15
            and drawdown_60 >= -0.08
        )
        if (
            not qualified
            and not strong_trend_exception
            and not high_payoff_setup
            and not asymmetric_pivot_watch
            and not compact_attack_core
        ):
            continue
        right_quant_setup = (
            "高盈亏比" if high_payoff_setup
            else "强趋势" if strong_trend_exception
            else "标准量化"
        )
        if asymmetric_pivot_watch:
            right_quant_setup = "asymmetric_pivot_watch"
        if compact_attack_core:
            right_quant_setup = "compact_attack_core"
        selected.append({
            **ranked,
            "strategy_part": "3.多因子量化选股候选",
            "right_quant_setup": right_quant_setup,
            "candidate_score": float(ranked.get("right_quant_score") or 0.0),
            "candidate_source": "factor_quant",
            "signal_eligible": True,
            "selection_reason": (
                "财务硬门槛通过；按时点可见的多因子横截面模型入选；"
                + f"综合排名第{rank}名；因子分{score:.1f}；"
                + f"设置={right_quant_setup}；"
                + f"{ranked['right_quant_reason']}"
            ),
        })
    return selected


def _value_falsification_reasons(value_line, quality, yoy, market_cap) -> list[str]:
    reasons = []
    if pd.isna(value_line) or float(value_line) <= 0:
        reasons.append("value_line_missing_or_nonpositive")
    if pd.isna(quality) or float(quality) < 70:
        reasons.append("quality_score_below_70")
    if pd.isna(yoy) or float(yoy) < 0.10:
        reasons.append("earnings_yoy_below_10pct")
    if pd.isna(market_cap) or float(market_cap) < 100:
        reasons.append("mktcap_below_100")
    return reasons


def _value_nonselection_reasons(price_to_value, financial_reasons) -> list[str]:
    reasons = list(financial_reasons)
    if not reasons:
        if price_to_value is None or pd.isna(price_to_value):
            reasons.append("price_to_value_unavailable")
        elif float(price_to_value) < 0.80:
            reasons.append("price_below_value_band_0_80")
        elif float(price_to_value) > 1.08:
            reasons.append("price_above_value_band_1_08")
    return reasons


def build_historical_candidate_snapshots(
    start_date,
    end_date,
    *,
    value_cache_directory,
    kline_directory,
    universe_path,
    mainline_directory=None,
    raw_kline_directory=None,
    price_source="akshare",
    max_mainline_age_days=MAX_MAINLINE_AGE_DAYS,
    research_repository=None,
    strict_financial_point_in_time=True,
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    universe = pd.read_csv(universe_path, dtype={"code": str})
    names = {
        str(row["code"]).split(".")[-1]: str(row.get("code_name") or row["code"])
        for _, row in universe.iterrows()
    }
    periods = visible_report_periods(start, end)
    financial = {
        period: _load_financial_cache(value_cache_directory, period)
        for period in periods
    }
    _validate_required_financial_periods(financial)
    if research_repository is not None:
        research_repository.persist_fundamentals(financial)
    mainline_snapshots = _load_mainline_snapshots(mainline_directory)
    codes = set().union(*(rows.keys() for rows in financial.values()))
    price_start = start - pd.Timedelta(days=420)
    prices = _load_prices(
        kline_directory,
        codes,
        price_start,
        end,
        raw_kline_directory=raw_kline_directory,
        price_source=price_source,
    )
    prices = {
        code: _candidate_feature_frame(frame)
        for code, frame in prices.items()
    }
    calendar = sorted({
        date.normalize()
        for frame in prices.values()
        for date in frame.index
        if start <= date.normalize() <= end
    })
    snapshots = {}
    tracked_value_codes = set()
    progress_enabled = os.environ.get("CANDIDATE_HISTORY_PROGRESS") == "1"
    total_dates = len(calendar)
    for date_index, date in enumerate(calendar, start=1):
        period = report_period_for(date)
        eligible_mainline_dates = [item for item in mainline_snapshots if item <= date]
        mainline_date = max(eligible_mainline_dates) if eligible_mainline_dates else None
        mainline_fresh = bool(
            mainline_date is not None
            and (date - mainline_date).days <= int(max_mainline_age_days)
        )
        mainline_members = mainline_snapshots.get(mainline_date, {}) if mainline_fresh else {}
        value_rows = []
        normal_rows = []
        leadership_rows = []
        right_side_pool_rows = []
        diagnostic_rows = []
        eligible_periods = [
            item for item in periods
            if pd.Timestamp(item) <= pd.Timestamp(period)
        ]
        if strict_financial_point_in_time:
            financial_items = []
            for code in sorted(codes):
                selected_period, metrics, pit_status = _latest_point_in_time_financial(
                    financial, code, date, eligible_periods,
                )
                if metrics is None:
                    continue
                financial_items.append((code, selected_period, metrics, pit_status))
        else:
            financial_items = [
                (
                    code,
                    period,
                    metrics,
                    _financial_point_in_time_status(metrics, date),
                )
                for code, metrics in financial.get(period, {}).items()
            ]
        for code, selected_period, metrics, financial_point_in_time in financial_items:
            price_frame = prices.get(code)
            if price_frame is None or date not in price_frame.index:
                continue
            market_row = price_frame.loc[date]
            if not bool(market_row.get("_is_traded_bar", False)):
                continue
            asof_close = pd.to_numeric(market_row.get("_asof_close"), errors="coerce")
            close = float(asof_close if pd.notna(asof_close) and asof_close > 0 else market_row["close"])
            volume = pd.to_numeric(market_row.get("volume"), errors="coerce")
            if close <= 0 or pd.isna(volume) or volume <= 0:
                continue
            value_line = pd.to_numeric(metrics.get("value_line"), errors="coerce")
            quality = pd.to_numeric(metrics.get("quality_score"), errors="coerce")
            yoy = pd.to_numeric(metrics.get("yoy"), errors="coerce")
            total_share = pd.to_numeric(metrics.get("total_share"), errors="coerce")
            cached_market_cap = pd.to_numeric(metrics.get("mktcap"), errors="coerce")
            market_cap = (
                close * float(total_share) / 1e8
                if pd.notna(total_share) and float(total_share) > 0
                else cached_market_cap
            )
            if any(pd.isna(item) for item in (quality, yoy, market_cap)):
                continue
            full_code = ("sh." if code.startswith(("6", "9")) else "sz.") + code
            price_to_value = (
                None if pd.isna(value_line) or value_line <= 0
                else close / float(value_line)
            )
            qfq_anchor_date = pd.to_datetime(
                market_row.get("_qfq_anchor_date"),
                errors="coerce",
            )
            if pd.isna(qfq_anchor_date):
                adjustment_check = "前复权锚点缺失：历史回测不满足严格时点"
                qfq_anchor_text = ""
            elif bool(market_row.get("_asof_price_available", False)):
                adjustment_check = str(
                    market_row.get("_asof_adjustment_method")
                    or "AkShare不复权价反推观察日锚定前复权价"
                )
                qfq_anchor_text = date.strftime("%Y-%m-%d")
            elif qfq_anchor_date.normalize() > date.normalize():
                adjustment_check = "缺少AkShare不复权价，绝对价格仍含未来复权因子"
                qfq_anchor_text = qfq_anchor_date.strftime("%Y-%m-%d")
            else:
                adjustment_check = "前复权锚点不晚于观察日"
                qfq_anchor_text = qfq_anchor_date.strftime("%Y-%m-%d")
            value_falsification_reasons = _value_falsification_reasons(
                value_line, quality, yoy, market_cap,
            )
            value_nonselection_reasons = _value_nonselection_reasons(
                price_to_value, value_falsification_reasons,
            )
            base = {
                "date": date.strftime("%Y-%m-%d"),
                "code": full_code,
                "name": names.get(code, code),
                "close": close,
                "value_line": None if pd.isna(value_line) else float(value_line),
                "quality_score": float(quality),
                "earnings_yoy": float(yoy),
                "mktcap": float(market_cap),
                "report_period": selected_period,
                "snapshot_version": SNAPSHOT_VERSION,
                **financial_point_in_time,
                "price_to_value": price_to_value,
                "historical_adjustment_check": adjustment_check,
                "qfq_anchor_date": qfq_anchor_text,
                "mainline_snapshot_date": None if mainline_date is None else mainline_date.strftime("%Y-%m-%d"),
                "mainline_snapshot_fresh": mainline_fresh,
                "mainline_boards": mainline_members.get(code, ""),
                "selected_for_trading": True,
                "candidate_failure_reason": "",
                "value_falsified": False,
                "value_falsification_reason": "",
            }
            value_safety_reasons = left_value_safety_reasons(base)
            trade_basis = _trade_basis_from_feature_row(market_row)
            base.update(trade_basis)
            structure_proximity = _structure_proximity_from_feature_row(market_row)
            base.update(structure_proximity)
            divergence = _divergence_from_feature_row(market_row)
            base.update(divergence)
            leadership = _leadership_from_feature_row(market_row)
            base.update(leadership)
            right_strength = _right_strength_from_feature_row(market_row)
            base.update(right_strength)
            base.update({
                "volatility_20": None if pd.isna(market_row.get("_volatility_20")) else float(market_row.get("_volatility_20")),
                "drawdown_60": None if pd.isna(market_row.get("_drawdown_60")) else float(market_row.get("_drawdown_60")),
                "return_5d": None if pd.isna(market_row.get("_return_5")) else float(market_row.get("_return_5")),
                "return_10d": None if pd.isna(market_row.get("_return_10")) else float(market_row.get("_return_10")),
                "ma20_slope": None if pd.isna(market_row.get("_ma20_slope")) else float(market_row.get("_ma20_slope")),
                "ma60_slope": None if pd.isna(market_row.get("_ma60_slope")) else float(market_row.get("_ma60_slope")),
                "right_acceleration": None if pd.isna(market_row.get("_right_acceleration")) else float(market_row.get("_right_acceleration")),
                "avg_amount_20": None if pd.isna(market_row.get("_avg_amount_20")) else float(market_row.get("_avg_amount_20")),
                "momentum_60_ex5": None if pd.isna(market_row.get("_momentum_60_ex5")) else float(market_row.get("_momentum_60_ex5")),
                "momentum_120_ex20": None if pd.isna(market_row.get("_momentum_120_ex20")) else float(market_row.get("_momentum_120_ex20")),
                "positive_day_ratio_60": None if pd.isna(market_row.get("_positive_day_ratio_60")) else float(market_row.get("_positive_day_ratio_60")),
                "downside_volatility_60": None if pd.isna(market_row.get("_downside_volatility_60")) else float(market_row.get("_downside_volatility_60")),
                "data_status": str(market_row.get("_data_status") or "unknown"),
                "tradestatus": None if pd.isna(market_row.get("_tradestatus")) else float(market_row.get("_tradestatus")),
                "is_traded_bar": bool(market_row.get("_is_traded_bar", False)),
                "raw_amount": None if pd.isna(market_row.get("_raw_amount")) else float(market_row.get("_raw_amount")),
                "amount_source": str(market_row.get("_amount_source") or "missing"),
                "price_source": (
                    f"{market_row.get('_price_source')}_raw_plus_qfq_asof"
                    if bool(market_row.get("_asof_price_available", False))
                    else f"{market_row.get('_price_source')}_current_qfq"
                ),
                "valid_price_bar": bool(market_row.get("_valid_price_bar", False)),
                "alpha_volume_price_corr_20": None if pd.isna(market_row.get("_alpha_volume_price_corr_20")) else float(market_row.get("_alpha_volume_price_corr_20")),
                "alpha_turnover_expansion_20": None if pd.isna(market_row.get("_alpha_turnover_expansion_20")) else float(market_row.get("_alpha_turnover_expansion_20")),
                "alpha_close_position_60": None if pd.isna(market_row.get("_alpha_close_position_60")) else float(market_row.get("_alpha_close_position_60")),
                "alpha_close_position_120": None if pd.isna(market_row.get("_alpha_close_position_120")) else float(market_row.get("_alpha_close_position_120")),
                "alpha_intraday_strength_20": None if pd.isna(market_row.get("_alpha_intraday_strength_20")) else float(market_row.get("_alpha_intraday_strength_20")),
                "alpha_gap_5d_count": None if pd.isna(market_row.get("_alpha_gap_5d_count")) else float(market_row.get("_alpha_gap_5d_count")),
            })
            passes_fundamental_gate = _passes_fundamental_gate(quality, yoy, market_cap)
            if passes_fundamental_gate:
                right_side_pool_rows.append(base)
            if full_code in tracked_value_codes and value_falsification_reasons:
                diagnostic_rows.append({
                    **base,
                    "strategy_part": "value_thesis_failed_diagnostic",
                    "candidate_score": 0.0,
                    "historical_adjustment_check": (
                        f"{adjustment_check}；financial_falsification"
                    ),
                    "candidate_source": "value_model",
                    "signal_eligible": False,
                    "selected_for_trading": False,
                    "candidate_failure_reason": (
                        "value_financial_falsification: "
                        + ";".join(value_falsification_reasons)
                    ),
                    "value_falsified": True,
                    "value_falsification_reason": ";".join(value_falsification_reasons),
                    "selection_reason": (
                        "diagnostic row only; value thesis failed current "
                        f"report_period={period}"
                    ),
                })
            if (
                not value_falsification_reasons
                and value_safety_reasons
                and price_to_value is not None
                and pd.notna(price_to_value)
                and 0.80 <= float(price_to_value) <= 1.08
            ):
                diagnostic_rows.append({
                    **base,
                    "strategy_part": "value_safety_rejected_diagnostic",
                    "candidate_score": 0.0,
                    "historical_adjustment_check": (
                        f"{adjustment_check}；left_value_safety_rejected"
                    ),
                    "candidate_source": "value_model",
                    "signal_eligible": False,
                    "selected_for_trading": False,
                    "candidate_failure_reason": (
                        "value_safety_rejected: "
                        + ";".join(value_safety_reasons)
                    ),
                    "value_falsified": False,
                    "value_falsification_reason": "",
                    "selection_reason": (
                        "diagnostic row only; value-line entry lacks left-side "
                        "safety margin"
                    ),
                })
            if (
                not pd.isna(value_line)
                and value_line > 0
                and 0.80 <= close / value_line <= 1.08
                and passes_fundamental_gate
                and not value_safety_reasons
            ):
                value_rows.append({
                    **base,
                    "strategy_part": "1.基本价值线或附近",
                    "candidate_score": (
                        float(quality)
                        + min(max(float(yoy), 0.0), 1.0) * 20
                        + float(trade_basis["trade_basis_score"])
                    ),
                    "historical_adjustment_check": (
                        f"{adjustment_check}；price_to_value_between_0.80_and_1.08"
                    ),
                    "candidate_source": "value_model",
                    "signal_eligible": True,
                    "selection_reason": (
                        "基本价值线模型入选；"
                        f"{trade_basis['trade_basis_reason']}"
                    ),
                })
            if code in mainline_members and passes_fundamental_gate:
                right_trade_basis = float(trade_basis["trade_basis_score"])
                right_leadership = float(leadership["leadership_score"])
                right_return_20 = base.get("return_20d")
                right_return_60 = base.get("return_60d")
                right_drawdown_60 = base.get("drawdown_60")
                right_avg_amount_20 = base.get("avg_amount_20")
                mainline_right_ready = (
                    pd.notna(right_return_20)
                    and pd.notna(right_return_60)
                    and pd.notna(right_drawdown_60)
                    and pd.notna(right_avg_amount_20)
                    and float(right_avg_amount_20) >= 500_000_000.0
                    and float(right_drawdown_60) >= -0.18
                    and (
                        (
                            right_leadership >= 22.0
                            and right_trade_basis >= 6.0
                            and float(right_return_60) >= 0.12
                        )
                        or (
                            right_leadership >= 18.0
                            and right_trade_basis >= 8.0
                            and float(right_return_20) >= 0.05
                            and float(right_return_60) >= 0.20
                            and float(right_drawdown_60) >= -0.12
                        )
                    )
                )
                normal_rows.append({
                    **base,
                    "strategy_part": "2.正常基本面选股",
                    "candidate_score": (
                        float(quality)
                        + min(max(float(yoy), 0.0), 1.0) * 20
                        + 15
                        + float(trade_basis["trade_basis_score"])
                    ),
                    "candidate_source": (
                        "standard_mainline"
                        if mainline_right_ready else "standard_mainline_watch"
                    ),
                    "signal_eligible": bool(mainline_right_ready),
                    "selected_for_trading": bool(mainline_right_ready),
                    "candidate_failure_reason": (
                        "" if mainline_right_ready else "right_side_permission_not_ready"
                    ),
                    "selection_reason": (
                        "主流标准基本面模型入选；"
                        f"{trade_basis['trade_basis_reason']}"
                    ),
                })
        leadership_rows.extend(_right_quant_selection_rows(right_side_pool_rows))
        normal_rows.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        by_code = {}
        for item in leadership_rows:
            existing = by_code.get(item["code"])
            if existing:
                sources = {
                    source for source in (
                        str(existing.get("candidate_source") or "").split("+")
                        + str(item.get("candidate_source") or "").split("+")
                    ) if source
                }
                existing.update({
                    "strategy_part": " + ".join(dict.fromkeys([
                        str(existing.get("strategy_part") or ""),
                        str(item.get("strategy_part") or ""),
                    ])),
                    "candidate_source": "+".join(sorted(sources)),
                    "signal_eligible": True,
                    "mainline_boards": item["mainline_boards"] or existing.get("mainline_boards", ""),
                    "candidate_score": max(existing["candidate_score"], item["candidate_score"]),
                    "selection_reason": (
                        f"{existing.get('selection_reason', '')}；"
                        f"{item.get('selection_reason', '')}"
                    ).strip("；"),
                })
            else:
                by_code[item["code"]] = item
        pool_rows = list(by_code.values())
        selected = normalize_candidate_snapshots(
            {date.strftime("%Y-%m-%d"): pool_rows}
        )[date.strftime("%Y-%m-%d")]
        selected_codes = {item["code"] for item in selected}
        for item in value_rows + normal_rows + pool_rows:
            normalized = normalize_candidate(item)
            if normalized["code"] in selected_codes:
                continue
            diagnostic = dict(normalized)
            diagnostic["signal_eligible"] = False
            diagnostic["selected_for_trading"] = False
            diagnostic["selection_rank"] = None
            diagnostic["candidate_failure_reason"] = (
                "not_selected_for_trading: factor_rank_not_in_observation_pool"
            )
            if "value_model" in str(diagnostic.get("candidate_source") or "").split("+"):
                price_to_value = diagnostic.get("price_to_value")
                reasons = _value_nonselection_reasons(
                    price_to_value,
                    _value_falsification_reasons(
                        diagnostic.get("value_line"),
                        diagnostic.get("quality_score"),
                        diagnostic.get("earnings_yoy"),
                        diagnostic.get("mktcap"),
                    ),
                )
                diagnostic["candidate_failure_reason"] += (
                    "; value_nonselection=" + ",".join(reasons)
                )
            diagnostic_rows.append(diagnostic)
        selected_by_code = {item["code"]: item for item in selected}
        diagnostics_by_code = {}
        for diagnostic in diagnostic_rows:
            code = diagnostic["code"]
            if code in selected_by_code:
                if diagnostic.get("value_falsified"):
                    selected_by_code[code]["value_falsified"] = True
                    selected_by_code[code]["value_falsification_reason"] = diagnostic.get(
                        "value_falsification_reason", "",
                    )
                    selected_by_code[code]["candidate_failure_reason"] = diagnostic.get(
                        "candidate_failure_reason", "",
                    )
                elif "value_safety_rejected" in str(diagnostic.get("candidate_failure_reason") or ""):
                    selected_row = selected_by_code[code]
                    selected_row["allow_left"] = False
                    existing_reason = str(selected_row.get("candidate_failure_reason") or "")
                    new_reason = str(diagnostic.get("candidate_failure_reason") or "")
                    if new_reason and new_reason not in existing_reason:
                        selected_row["candidate_failure_reason"] = (
                            f"{existing_reason}; {new_reason}"
                            if existing_reason else new_reason
                        )
                continue
            existing = diagnostics_by_code.get(code)
            if existing is None or diagnostic.get("value_falsified"):
                diagnostics_by_code[code] = diagnostic
            elif diagnostic.get("candidate_failure_reason"):
                existing_reason = str(existing.get("candidate_failure_reason") or "")
                new_reason = str(diagnostic.get("candidate_failure_reason") or "")
                if new_reason and new_reason not in existing_reason:
                    existing["candidate_failure_reason"] = (
                        f"{existing_reason}; {new_reason}" if existing_reason else new_reason
                    )
        diagnostics = normalize_candidate_snapshots(
            {date.strftime("%Y-%m-%d"): list(diagnostics_by_code.values())},
            include_diagnostics=True,
        )[date.strftime("%Y-%m-%d")]
        snapshots[date.strftime("%Y-%m-%d")] = selected + diagnostics
        if progress_enabled and (
            date_index == 1 or date_index == total_dates or date_index % 10 == 0
        ):
            print(
                "[candidate-history] "
                f"{date_index}/{total_dates} {date:%Y-%m-%d} "
                f"selected={len(selected)} diagnostics={len(diagnostics)}",
                flush=True,
            )
    return snapshots


def save_historical_candidate_snapshots(output_directory, snapshots, *, start_date, end_date):
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for date, rows in sorted(snapshots.items()):
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=CANDIDATE_SNAPSHOT_COLUMNS)
        else:
            ordered_columns = CANDIDATE_SNAPSHOT_COLUMNS + [
                column for column in frame.columns
                if column not in CANDIDATE_SNAPSHOT_COLUMNS
            ]
            frame = frame.reindex(columns=ordered_columns)
        path = target / f"candidates_{date}.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
        report_period = rows[0]["report_period"] if rows else report_period_for(date)
        eligible_count = sum(bool(row.get("signal_eligible", True)) for row in rows)
        mainline_date = next((row.get("mainline_snapshot_date") for row in rows if row.get("mainline_snapshot_date")), None)
        mainline_fresh = any(bool(row.get("mainline_snapshot_fresh")) for row in rows)
        financial_point_in_time = all(
            row.get("financial_point_in_time") is True for row in rows
        )
        manifest_rows.append({
            "date": date,
            "report_period": report_period,
            "candidate_count": len(rows),
            "signal_eligible_count": eligible_count,
            "mainline_snapshot_date": mainline_date,
            "mainline_snapshot_fresh": mainline_fresh,
            "financial_point_in_time": financial_point_in_time,
            "file": path.name,
        })
    financial_point_in_time = bool(manifest_rows) and all(
        row["financial_point_in_time"] for row in manifest_rows
    )
    unsafe_snapshot_dates = [
        row["date"] for row in manifest_rows
        if row["financial_point_in_time"] is not True
    ]
    point_in_time_note = (
        "Financial metrics are selected per stock from announce_time rows with "
        "announcement_date, annual_announcement_date, and capital_announcement_date "
        "not later than the observation date; no future financial report is admitted."
        if financial_point_in_time
        else (
            "Price history is cut at the observation date, but some candidate rows "
            "lack strict per-company announce_time visibility; this is research-only."
        )
    )
    manifest = {
        "version": SNAPSHOT_VERSION,
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "snapshot_count": len(manifest_rows),
        "financial_point_in_time": financial_point_in_time,
        "strict_financial_point_in_time": financial_point_in_time,
        "unsafe_snapshot_count": len(unsafe_snapshot_dates),
        "unsafe_snapshot_sample": unsafe_snapshot_dates[:10],
        "candidate_pool_formula": "所有模型统一输出候选接口；人工观察清单不得注入候选",
        "selection_standard": {
            "value": (
                "旧价值线模型仅作诊断：0.80 <= price/value_line <= 1.08，"
                "quality >= 70，yoy >= 0.10，mktcap >= 100；"
                "高增长小市值安全边际不足时不允许左侧执行"
            ),
            "normal": "旧主线模型仅作诊断：不再享有候选保留名额，也不直接决定入选",
            "factor_quant": (
                "最终执行候选只来自时点可见的多因子横截面排序；"
                "硬门槛为 quality >= 70、yoy >= 0.10、mktcap >= 100；"
                "因子包括质量、成长、流动性、20/60/120日动量、相对强度、"
                "趋势效率、波动回撤、短期过热控制、结构位置、量价确认和背离环境；"
                "主线标签只作为诊断字段，不给加分和保留名额"
            ),
            "ranking": (
                "按 factor_quant 的 candidate_score 统一排序；"
                "不设置 value/mainline 核心保留名额；候选池上限只是观察宽度，不是交易数量"
            ),
            "execution": "所有 signal_eligible 候选进入同一套结构、仓位和退出引擎",
            "manual": "观察清单和交易计划不能直接注入候选",
            "mainline_max_age_days": MAX_MAINLINE_AGE_DAYS,
        },
        "point_in_time_note": (
            "行情严格按观察日截断；财务报告期保守选择，但缺少逐公告修订历史，"
            "不得声明为严格财务时点回测。"
        ),
        "point_in_time_note": point_in_time_note,
        "snapshots": manifest_rows,
    }
    manifest_path = target / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
