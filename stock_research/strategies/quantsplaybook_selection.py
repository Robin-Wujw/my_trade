"""Point-in-time stock-selection factors reproduced from QuantsPlaybook.

The module deliberately separates strategy inventory from executable factors.
An inventory item is executable only when the source formula and its required
historical data are both available. Intraday, analyst, and fund-holding models
must not be replaced by daily-bar proxies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd


QUANTSPLAYBOOK_COMMIT = "87163521c75629a3466564c017ac734a236a9ce4"


@dataclass(frozen=True)
class SelectionStrategySpec:
    key: str
    name: str
    category: str
    frequency: str
    required_data: tuple[str, ...]
    factor_column: str | None
    status: str
    source_path: str
    note: str = ""

    @property
    def executable(self) -> bool:
        return self.status == "source_exact" and bool(self.factor_column)


def _spec(
    key,
    name,
    category,
    frequency,
    required_data,
    factor_column,
    status,
    source_path,
    note="",
):
    return SelectionStrategySpec(
        key=key,
        name=name,
        category=category,
        frequency=frequency,
        required_data=tuple(required_data),
        factor_column=factor_column,
        status=status,
        source_path=source_path,
        note=note,
    )


# One row per stock-selection directory in QuantsPlaybook A/B sections.
STRATEGY_SPECS = (
    _spec(
        "ffscore",
        "华泰FFScore",
        "fundamental",
        "quarterly",
        ("point_in_time_financials",),
        None,
        "missing_historical_fields",
        "A-量化基本面/华泰FFScore/FFScore.ipynb",
    ),
    _spec(
        "rob_reck_cashflow",
        "罗伯·瑞克超额现金流",
        "fundamental",
        "quarterly",
        ("point_in_time_financials", "daily_basic"),
        None,
        "missing_historical_fields",
        "A-量化基本面/申万大师系列十三/py/系列十三：罗伯·瑞克超额现金流选股法则.ipynb",
    ),
    _spec(
        "apm",
        "APM因子",
        "intraday",
        "monthly",
        ("30m_bars", "benchmark_30m"),
        None,
        "missing_intraday_data",
        "B-因子构建类/APM因子模型/py/APM因子模型.ipynb",
    ),
    _spec(
        "a_share_momentum",
        "A股切割动量",
        "price_volume",
        "daily",
        ("qfq_daily",),
        None,
        "missing_source_formula",
        "B-因子构建类/A股市场中如何构造动量因子？/notebook/A股市场中如何构造动量因子.ipynb",
        "Uses the robust r60-3000*variance implementation shared by the follow-up report.",
    ),
    _spec(
        "shadow_line",
        "上下影线因子",
        "price_volume",
        "daily",
        ("qfq_daily",),
        "shadow_reversal",
        "source_exact",
        "B-因子构建类/上下影线因子/py/上下引线因子.ipynb",
    ),
    _spec(
        "coin_team",
        "球队硬币因子",
        "price_volume",
        "daily",
        ("qfq_daily",),
        "coin_team",
        "source_exact",
        "B-因子构建类/个股动量效应的识别及球队硬币因子/FactorZoo/SportBetting.py",
    ),
    _spec(
        "corporate_lifecycle",
        "企业生命周期因子",
        "fundamental",
        "quarterly",
        ("point_in_time_cashflow", "factor_library"),
        None,
        "missing_historical_fields",
        "B-因子构建类/企业生命周期/企业生命周期测试.ipynb",
    ),
    _spec(
        "momentum_revisited",
        "再论动量因子",
        "price_volume",
        "daily",
        ("qfq_daily",),
        None,
        "missing_source_formula",
        "B-因子构建类/再论动量因子/py/再论动量因子.ipynb",
        "Same source formula as high_quality_momentum; deduplicated in comparisons.",
    ),
    _spec(
        "salience_str",
        "凸显理论STR因子",
        "behavioral",
        "daily",
        ("qfq_daily",),
        "salience_str",
        "source_exact",
        "B-因子构建类/凸显理论STR因子/凸显度因子.ipynb",
    ),
    _spec(
        "pure_idiosyncratic_volatility",
        "纯真特质波动率",
        "risk",
        "daily",
        ("qfq_daily",),
        "low_idiosyncratic_volatility",
        "source_exact",
        "B-因子构建类/剔除跨期截面相关性的纯真波动率因子/py/波动率选股因子_特质波动率.ipynb",
        "Market residual volatility is executable; six-month cross-sectional decorrelation is a later extension.",
    ),
    _spec(
        "factor_timing",
        "因子择时",
        "meta",
        "monthly",
        ("macro_history", "factor_return_history"),
        None,
        "not_a_stock_selector",
        "B-因子构建类/因子择时/因子择时研究.ipynb",
    ),
    _spec(
        "buy_sell_pressure",
        "量价买卖压力APB",
        "price_volume",
        "daily",
        ("qfq_daily", "amount"),
        "buying_pressure",
        "source_exact",
        "B-因子构建类/基于量价关系度量股票的买卖压力/py/基于量价关系度量股票的买卖压力.ipynb",
    ),
    _spec(
        "overnight_day_network",
        "隔夜与日间网络关系",
        "network",
        "daily",
        ("qfq_daily", "trained_delta_lag_model"),
        None,
        "missing_trained_model",
        "B-因子构建类/基于隔夜与日间的网络关系因子/factor_pipeline.py",
    ),
    _spec(
        "fund_overweight",
        "基金重仓超配因子",
        "fund_holdings",
        "quarterly",
        ("point_in_time_fund_holdings", "historical_index_members"),
        None,
        "missing_historical_holdings",
        "B-因子构建类/基金重仓超配因子及其对指数增强组合的影响/py/基金持股比列.ipynb",
    ),
    _spec(
        "disposition_cgo",
        "处置效应CGO",
        "behavioral",
        "daily",
        ("qfq_daily", "turnover_rate"),
        "disposition_reversal",
        "source_exact",
        "B-因子构建类/处置效应因子/py/资本利得突出量CGO与风险偏好_重置.ipynb",
    ),
    _spec(
        "multifactor_enhancement",
        "多因子指数增强",
        "meta",
        "daily",
        ("executable_factor_scores",),
        None,
        "missing_trained_model",
        "B-因子构建类/多因子指数增强/py/基于自适应风险控制的指数增强策略.ipynb",
        "Equal-weight robust rank ensemble; our portfolio engine keeps sizing and exits.",
    ),
    _spec(
        "w_reversal",
        "A股反转W因子",
        "microstructure",
        "intraday",
        ("intraday_returns",),
        None,
        "missing_intraday_data",
        "B-因子构建类/开源证券-市场微观结构研究系列（1）：A股反转之力的微观来源/py/W因子构建.ipynb",
    ),
    _spec(
        "ma_convergence",
        "均线收敛与发散",
        "price_volume",
        "daily",
        ("qfq_daily",),
        "ma_convergence",
        "source_exact",
        "B-因子构建类/开源证券-开源量化评论（91）：形态识别，均线的收敛与发散/FactorArithmetic/convergence_factor.py",
    ),
    _spec(
        "amplitude_structure",
        "振幅隐藏结构",
        "price_volume",
        "daily",
        ("qfq_daily",),
        "amplitude_structure",
        "source_exact",
        "B-因子构建类/振幅因子的隐藏结构/notebook/振幅因子的隐藏结构.ipynb",
    ),
    _spec(
        "excellent_fund_manager",
        "优秀基金经理超额收益",
        "fund_holdings",
        "quarterly",
        ("point_in_time_fund_nav", "point_in_time_fund_holdings"),
        None,
        "missing_historical_holdings",
        "B-因子构建类/来自优秀基金经理的超额收益/py/来自优秀基金经理的超额收益.ipynb",
    ),
    _spec(
        "chip_distribution",
        "筹码分布因子",
        "behavioral",
        "daily",
        ("qfq_daily", "turnover_rate"),
        "chip_loss_overhang",
        "source_exact",
        "B-因子构建类/筹码因子/scr/turnover_coefficient_ops.py",
    ),
    _spec(
        "smart_money",
        "聪明钱因子2.0",
        "microstructure",
        "30m",
        ("30m_bars",),
        None,
        "missing_intraday_data",
        "B-因子构建类/聪明钱因子模型的2.0版本/notebook/聪明钱因子模型的 2.0 版本.ipynb",
    ),
    _spec(
        "network_centrality",
        "股票网络中心度",
        "network",
        "monthly",
        ("qfq_daily",),
        "network_cc",
        "source_exact",
        "B-因子构建类/股票网络与网络中心度因子研究/src/factor_algo.py",
        "The source directory exposes SCC, TCC and their 1:1 composite CC.",
    ),
    _spec(
        "industry_volume_price",
        "行业有效量价轮动",
        "industry",
        "daily",
        ("point_in_time_industry_members", "industry_index_bars"),
        None,
        "not_a_stock_selector",
        "B-因子构建类/行业有效量价因子与行业轮动策略/行业有效量价因子与行业轮动策略ETF.ipynb",
    ),
    _spec(
        "gold_stock_enhancement",
        "金股增强策略",
        "analyst",
        "monthly",
        ("point_in_time_analyst_recommendations",),
        None,
        "missing_recommendation_history",
        "B-因子构建类/金股增强策略/金股增强策略.ipynb",
    ),
    _spec(
        "high_quality_momentum",
        "高质量动量",
        "price_volume",
        "daily",
        ("qfq_daily",),
        "high_quality_momentum",
        "source_exact",
        "B-因子构建类/高质量动量因子选股/高质量动量选股.ipynb",
    ),
    _spec(
        "high_frequency_price_volume",
        "高频价量相关性CPV",
        "microstructure",
        "intraday",
        ("intraday_price_volume",),
        None,
        "missing_intraday_data",
        "B-因子构建类/高频价量相关性，意想不到的选股因子/CPV因子.ipynb",
    ),
)


def strategy_inventory_frame() -> pd.DataFrame:
    rows = []
    for spec in STRATEGY_SPECS:
        row = asdict(spec)
        row["required_data"] = ",".join(spec.required_data)
        row["executable"] = spec.executable
        row["source_commit"] = QUANTSPLAYBOOK_COMMIT
        rows.append(row)
    return pd.DataFrame(rows)


def executable_factor_columns() -> list[str]:
    source_factors = [
        spec.factor_column for spec in STRATEGY_SPECS if spec.executable
    ]
    source_variants = ["network_scc", "network_tcc"]
    return list(dict.fromkeys([
        *source_factors,
        *source_variants,
        "playbook_ensemble",
    ]))


def _clean_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily price frame missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["code"] = result["code"].fillna("").astype(str).str.strip()
    numeric = [
        column for column in (
            "open", "high", "low", "close", "volume", "amount",
            "raw_to_qfq_factor", "turnover_rate",
        )
        if column in result
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "code", "open", "high", "low", "close"])
    result = (
        result.drop_duplicates(["date", "code"], keep="last")
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )
    inverse_adjustment = result.get("raw_to_qfq_factor")
    if inverse_adjustment is None:
        inverse_adjustment = pd.Series(1.0, index=result.index)
    inverse_adjustment = pd.to_numeric(
        inverse_adjustment, errors="coerce",
    ).where(lambda value: value > 0, 1.0)
    # TushareRepository stores 1 / adj_factor because the execution engine
    # anchors that inverse series at the requested endpoint.  Factor returns
    # need the forward adjustment path itself.  Multiplying by adj_factor is
    # point-in-time safe and avoids anchoring historical signals to a future
    # corporate action.
    result["adjustment_factor"] = 1.0 / inverse_adjustment
    for column in ("open", "high", "low", "close"):
        result[f"qfq_{column}"] = result[column] * result["adjustment_factor"]
    return result


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="code", values=column).sort_index()


def _turnover_fraction(turnover_rate: pd.DataFrame) -> pd.DataFrame:
    # Tushare and the reproduced source notebooks both expose turnover as a
    # percentage (1.0 means one percent), while the decay formula needs 0.01.
    return (turnover_rate.astype(float) / 100.0).clip(
        lower=0.0, upper=0.999999,
    )


def _finite_turnover_reference_price(
    price: pd.DataFrame,
    turnover_rate: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Finite-window turnover-decay reference price used by CGO and ARC."""
    values = price.to_numpy(dtype=float)
    turnover = _turnover_fraction(
        turnover_rate.reindex_like(price),
    ).fillna(0.0).to_numpy(dtype=float)
    decay = 1.0 - turnover
    log_decay = pd.DataFrame(
        np.log(decay), index=price.index, columns=price.columns,
    )
    survival = np.exp(
        log_decay.rolling(int(window), min_periods=int(window)).sum()
    ).to_numpy(dtype=float)
    numerator = np.zeros(values.shape[1], dtype=float)
    denominator = np.zeros(values.shape[1], dtype=float)
    output = np.full_like(values, np.nan)
    for position in range(values.shape[0]):
        current_price = np.where(np.isfinite(values[position]), values[position], 0.0)
        current_turnover = np.where(
            np.isfinite(values[position]), turnover[position], 0.0,
        )
        numerator = current_turnover * current_price + decay[position] * numerator
        denominator = current_turnover + decay[position] * denominator
        if position >= int(window):
            old = position - int(window)
            old_price = np.where(np.isfinite(values[old]), values[old], 0.0)
            old_turnover = np.where(
                np.isfinite(values[old]), turnover[old], 0.0,
            )
            old_survival = np.nan_to_num(survival[position], nan=0.0)
            numerator -= old_turnover * old_price * old_survival
            denominator -= old_turnover * old_survival
        if position >= int(window) - 1:
            valid = denominator > 1e-12
            output[position, valid] = numerator[valid] / denominator[valid]
    return pd.DataFrame(output, index=price.index, columns=price.columns)


def _rolling_multifactor_residual_std(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling multivariate OLS residual volatility with shared regressors."""
    y_values = returns.to_numpy(dtype=float)
    factor_values = factors.reindex(returns.index).to_numpy(dtype=float)
    output = np.full_like(y_values, np.nan)
    for position in range(int(window) - 1, len(returns)):
        x_window = factor_values[position - int(window) + 1:position + 1]
        if not np.isfinite(x_window).all():
            continue
        x_window = np.column_stack([np.ones(len(x_window)), x_window])
        y_window = y_values[position - int(window) + 1:position + 1]
        valid = np.isfinite(y_window).all(axis=0)
        if valid.sum() < 3:
            continue
        beta = np.linalg.lstsq(x_window, y_window[:, valid], rcond=None)[0]
        residual = y_window[:, valid] - x_window @ beta
        output[position, valid] = residual.std(axis=0, ddof=1)
    return pd.DataFrame(output, index=returns.index, columns=returns.columns)


def _pure_idiosyncratic_volatility(
    returns: pd.DataFrame,
    market_cap: pd.DataFrame,
    pb: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce ID_Vol and its six-period cross-sectional de-correlation."""
    cap = market_cap.where(market_cap > 0)
    book_to_market = 1.0 / pb.where(pb > 0)
    cap_rank = cap.rank(axis=1, pct=True)
    bm_rank = book_to_market.rank(axis=1, pct=True)

    def weighted_group(mask: pd.DataFrame) -> pd.Series:
        weights = cap.where(mask)
        weights = weights.div(weights.sum(axis=1), axis=0)
        return weights.mul(returns).sum(axis=1, min_count=1)

    small = weighted_group(cap_rank <= 0.10)
    large = weighted_group(cap_rank > 0.90)
    high_bm = weighted_group(bm_rank > 0.90)
    low_bm = weighted_group(bm_rank <= 0.10)
    market_weights = cap.div(cap.sum(axis=1), axis=0)
    market = market_weights.mul(returns).sum(axis=1, min_count=1)
    ff3 = pd.concat(
        [market.rename("mkt"), (small - large).rename("smb"), (high_bm - low_bm).rename("hml")],
        axis=1,
    )
    id_vol = _rolling_multifactor_residual_std(returns, ff3, window=20)

    # A truncated input must not turn its final row into a fake month end.
    # Completed month ends are known when the following trading row belongs to
    # a new month; the current incomplete month carries the prior observation.
    periods = id_vol.index.to_period("M")
    completed = periods[:-1] != periods[1:]
    month_ends = id_vol.index[:-1][completed]
    monthly = id_vol.reindex(pd.DatetimeIndex(month_ends))
    pure = pd.DataFrame(np.nan, index=monthly.index, columns=monthly.columns)
    for position in range(6, len(monthly)):
        y = monthly.iloc[position]
        valid_y = y.notna()
        if valid_y.sum() < 10:
            continue
        x = monthly.iloc[position - 6:position].T.fillna(0.0)
        x = x.loc[valid_y]
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        values = y.loc[valid_y].to_numpy(dtype=float)
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        pure.loc[monthly.index[position], valid_y] = values - design @ beta
    return pure.reindex(id_vol.index).ffill()


def _network_scc(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """SCC without materializing a 5,000 by 5,000 correlation matrix."""
    values = returns.to_numpy(dtype=float)
    output = np.full_like(values, np.nan)
    for position in range(int(window) - 1, len(returns)):
        sample = values[position - int(window) + 1:position + 1]
        valid = np.isfinite(sample).all(axis=0)
        sample = sample[:, valid]
        if sample.shape[1] < 3:
            continue
        std = sample.std(axis=0, ddof=1)
        usable = std > 1e-12
        if usable.sum() < 3:
            continue
        normalized = (
            sample[:, usable] - sample[:, usable].mean(axis=0)
        ) / std[usable]
        market_sum = normalized.sum(axis=1)
        sum_correlation = (
            normalized.T @ market_sum / (int(window) - 1) - 1.0
        )
        average = sum_correlation / (usable.sum() - 1)
        scc = 1.0 / (2.0 * (1.0 - np.minimum(average, 0.999999)))
        valid_positions = np.flatnonzero(valid)[usable]
        output[position, valid_positions] = scc
    return pd.DataFrame(output, index=returns.index, columns=returns.columns)


def _network_tcc(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Reproduce the source TCC rolling mean-square standardized distance."""
    market_mean = returns.mean(axis=1)
    market_std = returns.std(axis=1, ddof=0).replace(0, np.nan)
    standardized = returns.sub(market_mean, axis=0).div(market_std, axis=0)
    mean_square = standardized.pow(2).rolling(
        int(window), min_periods=int(window),
    ).mean()
    return 1.0 / mean_square.replace(0, np.nan)


def _cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    mean = values.mean(axis=1)
    std = values.std(axis=1).replace(0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0)


def _cross_sectional_size_residual_zscore(
    values: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectionally neutralize a factor to market cap, then standardize."""
    y = values.astype(float)
    x = market_cap.reindex_like(y).where(lambda item: item > 0).astype(float)
    valid = y.notna() & x.notna()
    count = valid.sum(axis=1).replace(0, np.nan)
    x_mean = x.where(valid).sum(axis=1).div(count)
    y_mean = y.where(valid).sum(axis=1).div(count)
    x_centered = x.sub(x_mean, axis=0).where(valid)
    y_centered = y.sub(y_mean, axis=0).where(valid)
    denominator = x_centered.pow(2).sum(axis=1).replace(0, np.nan)
    beta = x_centered.mul(y_centered).sum(axis=1).div(denominator)
    residual = y_centered.sub(x_centered.mul(beta, axis=0))
    return _cross_sectional_zscore(residual)


def _amplitude_hidden_structure(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    valid_after_prior_day: pd.DataFrame,
    *,
    rank_window: int = 22,
    factor_window: int = 20,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """Reproduce AF_high - AF_low with each observation's fixed lookback."""
    close_values = close.to_numpy(dtype=float)
    amplitude = high.div(low.replace(0, np.nan)).sub(1.0).to_numpy(dtype=float)
    valid_values = valid_after_prior_day.reindex_like(close).fillna(False).to_numpy(
        dtype=bool,
    )
    output = np.full_like(close_values, np.nan)
    for position in range(int(rank_window) - 1, len(close)):
        start = position - int(rank_window) + 1
        close_window = close_values[start:position + 1]
        amplitude_window = amplitude[
            position - int(factor_window) + 1:position + 1
        ]
        valid_window = valid_values[
            position - int(factor_window) + 1:position + 1
        ]
        complete = np.isfinite(close_window).all(axis=0)
        complete &= np.isfinite(amplitude_window).all(axis=0)
        if not complete.any():
            continue
        sample = close_window[:, complete]
        order = np.argsort(sample, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        columns = np.arange(sample.shape[1])
        ranks[order, columns] = np.arange(1, int(rank_window) + 1)[:, None]
        high_price = (
            ranks[-int(factor_window):] / float(rank_window)
        ) >= float(threshold)
        signed = np.where(high_price, amplitude_window[:, complete], -amplitude_window[:, complete])
        signed = np.where(valid_window[:, complete], signed, 0.0)
        output[position, complete] = signed.mean(axis=0)
    return pd.DataFrame(output, index=close.index, columns=close.columns)


def calculate_daily_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate source-aligned factors using bars no later than each row date."""
    data = _clean_price_frame(frame)
    close = _pivot(data, "qfq_close")
    open_ = _pivot(data, "qfq_open").reindex_like(close)
    high = _pivot(data, "qfq_high").reindex_like(close)
    low = _pivot(data, "qfq_low").reindex_like(close)
    returns = close.pct_change(fill_method=None)

    factors: dict[str, pd.DataFrame] = {}

    prior_returns = returns.shift(1)
    momentum = close.div(close.shift(61)) - 1.0
    momentum -= 3000.0 * prior_returns.rolling(60, min_periods=60).var()
    max_return = prior_returns.rolling(20, min_periods=20).max()
    if "turnover_rate" in data:
        turnover = _pivot(data, "turnover_rate").reindex_like(close)
    else:
        turnover = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    turnover_std = turnover.shift(1).rolling(60, min_periods=60).std()
    factors["high_quality_momentum"] = pd.concat(
        [
            max_return.rank(axis=1, ascending=False, pct=True),
            turnover_std.rank(axis=1, ascending=False, pct=True),
            momentum.rank(axis=1, ascending=True, pct=True),
        ],
        axis=0,
        keys=["max", "turnover", "momentum"],
    ).groupby(level=1).mean()

    upper_shadow = high - np.maximum(open_, close)
    upper_baseline = upper_shadow.rolling(5, min_periods=5).mean().shift(1)
    normalized_upper = upper_shadow / upper_baseline.replace(0, np.nan)
    williams_lower = close - low
    williams_lower_baseline = williams_lower.rolling(
        5, min_periods=5,
    ).mean().shift(1)
    normalized_williams_lower = (
        williams_lower / williams_lower_baseline.replace(0, np.nan)
    )
    if "total_mv" in data:
        market_cap = _pivot(data, "total_mv").reindex_like(close)
        upper_std = normalized_upper.rolling(20, min_periods=20).std()
        williams_lower_mean = normalized_williams_lower.rolling(
            20, min_periods=20,
        ).mean()
        ubl = (
            _cross_sectional_size_residual_zscore(upper_std, market_cap)
            + _cross_sectional_size_residual_zscore(
                williams_lower_mean, market_cap,
            )
        )
        # The source portfolio is long the lowest UBL group.
        factors["shadow_reversal"] = -ubl

    intraday_returns = close.div(open_) - 1.0
    overnight_returns = open_.div(close.shift(1)) - 1.0
    turnover_change = turnover.diff()

    def coin_component(return_frame: pd.DataFrame) -> pd.DataFrame:
        average_return = return_frame.rolling(20, min_periods=15).mean()
        volatility = return_frame.rolling(20, min_periods=15).std()
        volatility_flip = average_return * np.sign(
            volatility.sub(volatility.mean(axis=1), axis=0)
        )
        turnover_flip = (
            return_frame
            * np.sign(turnover_change.sub(turnover_change.mean(axis=1), axis=0))
        ).rolling(20, min_periods=15).mean()
        return (volatility_flip + turnover_flip) * 0.5

    factors["coin_team"] = -sum(
        coin_component(item)
        for item in (returns, intraday_returns, overnight_returns)
    )

    market_return = returns.mean(axis=1)
    sigma = returns.sub(market_return, axis=0).abs().div(
        returns.abs().add(market_return.abs(), axis=0) + 0.1
    )
    salience_rank = sigma.rank(axis=1, ascending=False)
    salience_weight = np.power(0.7, salience_rank)
    salience_weight = salience_weight.div(salience_weight.mean(axis=1), axis=0)
    salience_covariance = (
        salience_weight.mul(returns)
        .rolling(20, min_periods=20)
        .mean()
        - salience_weight.rolling(20, min_periods=20).mean()
        * returns.rolling(20, min_periods=20).mean()
    )
    factors["salience_str"] = salience_covariance

    if {"total_mv", "pb"}.issubset(data.columns):
        market_cap = _pivot(data, "total_mv").reindex_like(close)
        pb = _pivot(data, "pb").reindex_like(close)
        pure_id_vol = _pure_idiosyncratic_volatility(
            returns, market_cap, pb,
        )
        factors["low_idiosyncratic_volatility"] = -pure_id_vol

    if {"amount", "volume"}.issubset(data.columns):
        volume = _pivot(data, "volume").reindex_like(close)
        vwap_30 = close.mul(volume).rolling(30, min_periods=15).sum().div(
            volume.rolling(30, min_periods=15).sum().replace(0, np.nan)
        )
        simple_vwap = vwap_30.rolling(30, min_periods=15).mean()
        volume_vwap = vwap_30.mul(volume).rolling(
            30, min_periods=15,
        ).sum().div(
            volume.rolling(30, min_periods=15).sum().replace(0, np.nan)
        )
        apb = simple_vwap.div(volume_vwap)
        factors["buying_pressure"] = np.log(apb.where(apb > 0)).rolling(
            30, min_periods=15,
        ).mean()

    moving_averages = [close]
    moving_averages.extend(
        close.rolling(window, min_periods=window).mean()
        for window in (5, 10, 20, 60, 120)
    )
    stacked = np.stack([item.to_numpy(dtype=float) for item in moving_averages], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        convergence_std = np.nanstd(stacked, axis=2, ddof=1)
    adjustment = _pivot(data, "adjustment_factor").reindex_like(close)
    price_convergence = pd.DataFrame(
        -np.log1p(convergence_std / adjustment),
        index=close.index,
        columns=close.columns,
    )
    if "volume" in data:
        volume_panel = _pivot(data, "volume").reindex_like(close)
        volume_moving_averages = [volume_panel]
        volume_moving_averages.extend(
            volume_panel.rolling(window, min_periods=window).mean()
            for window in (5, 10, 20, 60, 120)
        )
        volume_stacked = np.stack(
            [item.to_numpy(dtype=float) for item in volume_moving_averages],
            axis=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            volume_std = np.nanstd(volume_stacked, axis=2, ddof=1)
        volume_convergence = pd.DataFrame(
            -np.log1p(volume_std), index=close.index, columns=close.columns,
        )
        factors["ma_convergence"] = (
            _cross_sectional_zscore(price_convergence)
            + _cross_sectional_zscore(volume_convergence)
        ) * 0.5
    else:
        factors["ma_convergence"] = price_convergence

    if "tradestatus" in data:
        status = _pivot(data, "tradestatus").reindex_like(close)
        paused = ~status.astype(str).eq("1")
    else:
        paused = close.isna()
    one_price_limit_down = (
        close.div(close.shift(1)).sub(1.0).lt(-0.09)
        & high.eq(low)
    )
    prior_day_invalid = (paused | one_price_limit_down).shift(
        1, fill_value=True,
    )
    valid_after_prior_day = ~prior_day_invalid
    factors["amplitude_structure"] = _amplitude_hidden_structure(
        close,
        high,
        low,
        valid_after_prior_day,
    )

    if "turnover_rate" in data:
        average_price = (
            _pivot(data, "amount").reindex_like(close).mul(10.0)
            .div(_pivot(data, "volume").reindex_like(close).replace(0, np.nan))
            .mul(adjustment)
        )
        cgo_reference = _finite_turnover_reference_price(
            average_price, turnover, window=100,
        )
        cgo = close.div(cgo_reference) - 1.0
        factors["disposition_reversal"] = -cgo
        chip_reference = _finite_turnover_reference_price(
            close, turnover, window=60,
        )
        arc = 1.0 - chip_reference.div(close)
        factors["chip_loss_overhang"] = -arc

    network_scc = _network_scc(returns, window=20)
    network_tcc = _network_tcc(returns, window=20)
    factors["network_scc"] = network_scc
    factors["network_tcc"] = network_tcc
    factors["network_cc"] = (network_scc + network_tcc) * 0.5

    for name in (
        spec.factor_column for spec in STRATEGY_SPECS if spec.executable
    ):
        if name not in factors:
            factors[name] = pd.DataFrame(
                np.nan, index=close.index, columns=close.columns,
            )

    long_frames = []
    for name, values in factors.items():
        series = values.stack()
        series.index.names = ["date", "code"]
        long_frames.append(series.rename(name))
    panel = pd.concat(long_frames, axis=1).sort_index()

    executable_base = [
        column for column in executable_factor_columns()
        if column != "playbook_ensemble" and column in panel
    ]
    percentile = panel[executable_base].groupby(level="date").rank(
        method="average", pct=True
    )
    panel["playbook_ensemble"] = percentile.mean(axis=1, skipna=True)
    return panel.replace([np.inf, -np.inf], np.nan)
