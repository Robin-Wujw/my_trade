"""Build factor-only A-share selections and compare them in one execution engine."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.backtesting import workflow as portfolio_backtest
from stock_research.core.paths import PATHS
from stock_research.reporting.backtest_trade_report import (
    apply_stock_basic_display_names,
)
from stock_research.storage import Database, TushareRepository
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import (
    SNAPSHOT_VERSION,
    _candidate_feature_frame,
    _divergence_from_feature_row,
    _leadership_from_feature_row,
    _right_strength_from_feature_row,
    _structure_proximity_from_feature_row,
    _trade_basis_from_feature_row,
)
from stock_research.strategies.quantsplaybook_selection import (
    QUANTSPLAYBOOK_COMMIT,
    calculate_daily_factor_panel,
    executable_factor_columns,
    low_market_residual_volatility_factor_columns,
    low_market_residual_volatility_factor_metadata,
    market_residual_momentum_factor_columns,
    market_residual_momentum_factor_metadata,
    strategy_inventory_frame,
)


DEFAULT_PRICE_DIRECTORY = (
    PATHS.runtime_root / "backtests" / "tushare_price_frames_2021_to_20260721_404"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests" / "quantsplaybook_factor_only_2022_to_20260721"
)
REQUIRED_HISTORY_CALENDAR_DAYS = 700
LOW_RESIDUAL_VOLATILITY_HISTORY_CALENDAR_DAYS = 420
BEIJING_STOCK_EXCHANGE_OPEN_DATE = pd.Timestamp("2021-11-15")


CANDIDATE_PROFILE_EXTRA_FIELDS = (
    "volatility_20",
    "drawdown_60",
    "return_5d",
    "return_10d",
    "ma20_slope",
    "ma60_slope",
    "right_acceleration",
    "avg_amount_20",
    "momentum_60_ex5",
    "momentum_120_ex20",
    "positive_day_ratio_60",
    "downside_volatility_60",
    "data_status",
    "tradestatus",
    "is_traded_bar",
    "raw_amount",
    "amount_source",
    "price_source",
    "valid_price_bar",
    "alpha_volume_price_corr_20",
    "alpha_turnover_expansion_20",
    "alpha_close_position_60",
    "alpha_close_position_120",
    "alpha_intraday_strength_20",
    "alpha_gap_5d_count",
)


def _market_residual_factor_configuration(
    include_market_residual_momentum: bool,
    include_low_market_residual_volatility: bool = False,
) -> tuple[list[str], dict[str, dict]]:
    factors = []
    metadata = {}
    if include_market_residual_momentum:
        momentum_factors = market_residual_momentum_factor_columns()
        factors.extend(momentum_factors)
        metadata.update({
            factor: market_residual_momentum_factor_metadata(factor)
            for factor in momentum_factors
        })
    if include_low_market_residual_volatility:
        volatility_factors = low_market_residual_volatility_factor_columns()
        factors.extend(volatility_factors)
        metadata.update({
            factor: low_market_residual_volatility_factor_metadata()
            for factor in volatility_factors
        })
    return factors, metadata


def _drop_ineligible_invalid_adjustment_rows(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Drop invalid adjustment rows only when they are not PIT-tradable."""
    if "raw_to_qfq_factor" not in prices:
        return prices
    adjustment = pd.to_numeric(
        prices["raw_to_qfq_factor"], errors="coerce",
    )
    invalid = adjustment.isna() | adjustment.le(0)
    if not bool(invalid.any()):
        return prices
    invalid_keys = prices.loc[invalid, ["date", "code"]].copy()
    invalid_keys["date"] = pd.to_datetime(
        invalid_keys["date"], errors="coerce",
    )
    invalid_keys["code"] = invalid_keys["code"].astype(str)
    eligible_keys = membership[["date", "code"]].copy()
    eligible_keys["date"] = pd.to_datetime(
        eligible_keys["date"], errors="coerce",
    )
    eligible_keys["code"] = eligible_keys["code"].astype(str)
    invalid_index = pd.MultiIndex.from_frame(invalid_keys)
    eligible_index = pd.MultiIndex.from_frame(eligible_keys.drop_duplicates())
    eligible_invalid = invalid_index.isin(eligible_index)
    if bool(eligible_invalid.any()):
        sample = invalid_keys.loc[eligible_invalid].head(5).to_dict("records")
        raise ValueError(
            "PIT-tradable rows require complete positive raw_to_qfq_factor; "
            f"invalid_rows={sample}"
        )
    return prices.loc[~invalid].copy()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qfq_close_panel(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    if "raw_to_qfq_factor" in frame:
        inverse = pd.to_numeric(
            frame["raw_to_qfq_factor"], errors="coerce",
        ).where(lambda value: value > 0, 1.0)
    else:
        inverse = pd.Series(1.0, index=frame.index)
    frame["qfq_close"] = close / inverse
    return frame.pivot(index="date", columns="code", values="qfq_close").sort_index()


def _spearman_rank_corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
    ).dropna()
    if len(frame) < 2:
        return None
    value = frame["left"].rank(method="average").corr(
        frame["right"].rank(method="average"),
    )
    return None if pd.isna(value) else float(value)


def point_in_time_tradable_universe(
    prices: pd.DataFrame,
    stock_basic: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    minimum_history: int = 60,
    stock_st: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return only time-valid, listed and traded A-share rows.

    No quality, growth, valuation, market-cap, industry, Formula33 or previous
    candidate-pool rule is applied here.
    """
    frame = prices[
        ["date", "code", "close", "high", "low", "volume", "amount", "tradestatus"]
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["code", "date"], kind="mergesort")
    frame["history_count"] = frame.groupby("code", sort=False).cumcount() + 1

    basic = stock_basic[
        ["code", "list_date", "delist_date"]
    ].drop_duplicates("code", keep="last")
    frame = frame.merge(basic, on="code", how="inner")
    numeric = ["close", "high", "low", "volume", "amount"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    traded = (
        frame["tradestatus"].astype(str).eq("1")
        & frame["close"].gt(0)
        & frame["high"].gt(0)
        & frame["low"].gt(0)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
    )
    listed = (
        frame["list_date"].notna()
        & frame["date"].ge(frame["list_date"])
        & (
            frame["delist_date"].isna()
            | frame["date"].le(frame["delist_date"])
        )
    )
    beijing_open = (
        ~frame["code"].astype(str).str.startswith("bj.")
        | frame["date"].ge(BEIJING_STOCK_EXCHANGE_OPEN_DATE)
    )
    seasoned = (
        frame["history_count"].ge(int(minimum_history))
        & (frame["date"] - frame["list_date"]).dt.days.ge(60)
    )
    in_range = frame["date"].between(
        pd.Timestamp(start_date), pd.Timestamp(end_date),
    )
    result = (
        frame.loc[
            traded & listed & beijing_open & seasoned & in_range,
            ["date", "code"],
        ]
        .drop_duplicates(["date", "code"])
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )
    if stock_st is not None and not stock_st.empty:
        risk = stock_st[["date", "code"]].copy()
        risk["date"] = pd.to_datetime(risk["date"], errors="coerce")
        risk["code"] = risk["code"].astype(str)
        risk = risk.dropna(subset=["date", "code"]).drop_duplicates()
        result = (
            result.merge(
                risk.assign(_risk_warning=True),
                on=["date", "code"],
                how="left",
            )
            .loc[lambda value: value["_risk_warning"].ne(True), ["date", "code"]]
            .reset_index(drop=True)
        )
    return result


def evaluate_factor_panel(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    holding_days: int = 20,
    start_date: str,
    end_date: str,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate factors with future returns used only as labels."""
    data = build_labeled_factor_sample(
        factor_panel,
        prices,
        membership,
        holding_days=holding_days,
        start_date=start_date,
        end_date=end_date,
    )
    return evaluate_labeled_factor_sample(
        data,
        factor_columns=factor_columns or executable_factor_columns(),
        label_end_date=end_date,
    )


def build_labeled_factor_sample(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    holding_days: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Attach research labels while retaining the label realization date."""
    close = _qfq_close_panel(prices)
    forward = close.shift(-int(holding_days)).div(close) - 1.0
    forward_long = forward.stack().rename("forward_return")
    forward_long.index.names = ["date", "code"]
    label_dates = pd.Series(close.index, index=close.index).shift(
        -int(holding_days),
    )
    sample = membership[
        membership["date"].between(
            pd.Timestamp(start_date), pd.Timestamp(end_date),
        )
    ]
    index = pd.MultiIndex.from_frame(sample[["date", "code"]])
    data = factor_panel.reindex(index).join(forward_long, how="left")
    data["label_date"] = data.index.get_level_values("date").map(label_dates)
    return data


def evaluate_labeled_factor_sample(
    data: pd.DataFrame,
    *,
    factor_columns: list[str],
    label_end_date: str | None = None,
) -> pd.DataFrame:
    """Evaluate a cached PIT panel; labels never enter candidate scores."""
    if label_end_date is not None:
        data = data[
            pd.to_datetime(data["label_date"], errors="coerce").le(
                pd.Timestamp(label_end_date),
            )
        ]
    metrics = []
    for factor in factor_columns:
        if factor not in data:
            continue
        valid = data[[factor, "forward_return"]].dropna()
        daily_rows = []
        for date, group in valid.groupby(level="date", sort=True):
            if len(group) < 50 or group[factor].nunique() < 10:
                continue
            top_count = max(1, int(np.ceil(len(group) * 0.2)))
            ordered = group.sort_values(factor, ascending=False, kind="mergesort")
            top_return = ordered.head(top_count)["forward_return"].mean()
            universe_return = ordered["forward_return"].mean()
            daily_rows.append({
                "date": date,
                "ic": _spearman_rank_corr(
                    ordered[factor],
                    ordered["forward_return"],
                ),
                "top_return": top_return,
                "universe_return": universe_return,
                "top_positive": top_return > universe_return,
                "count": len(ordered),
            })
        daily = pd.DataFrame(daily_rows).dropna(subset=["ic"])
        if daily.empty:
            continue
        spread = daily["top_return"] - daily["universe_return"]
        ic_std = daily["ic"].std()
        metrics.append({
            "factor": factor,
            "observation_count": len(valid),
            "date_count": len(daily),
            "mean_universe_size": round(daily["count"].mean(), 3),
            "mean_rank_ic": round(daily["ic"].mean(), 6),
            "median_rank_ic": round(daily["ic"].median(), 6),
            "rank_ic_ir": (
                None if not pd.notna(ic_std) or ic_std == 0
                else round(daily["ic"].mean() / ic_std, 6)
            ),
            "top20_mean_forward_return": round(daily["top_return"].mean(), 6),
            "universe_mean_forward_return": round(
                daily["universe_return"].mean(), 6,
            ),
            "top20_excess_forward_return": round(spread.mean(), 6),
            "top20_excess_hit_rate": round(daily["top_positive"].mean(), 6),
        })
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics).sort_values(
        ["top20_excess_forward_return", "mean_rank_ic"],
        ascending=False,
    ).reset_index(drop=True)


def rank_weighted_factor_combination(
    factor_panel: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Combine dated cross-sectional ranks without filling unavailable factors."""
    clean_weights = {
        str(column): float(weight)
        for column, weight in weights.items()
        if column in factor_panel
        and np.isfinite(float(weight))
        and float(weight) > 0
    }
    if not clean_weights:
        raise ValueError("factor combination has no positive available weights")
    values = factor_panel[list(clean_weights)].groupby(
        level="date", sort=False,
    ).rank(method="average", pct=True)
    weighted = values.mul(pd.Series(clean_weights), axis=1)
    available_weight = values.notna().mul(
        pd.Series(clean_weights), axis=1,
    ).sum(axis=1)
    return weighted.sum(axis=1, min_count=1).div(
        available_weight.where(available_weight > 0),
    )


def fit_walk_forward_combinations(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    holding_days: int,
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    positive_ic_top_n: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Fit declared combinations before OOS and freeze one on validation."""
    source_factors = [
        factor for factor in executable_factor_columns()
        if factor != "playbook_ensemble"
        and factor in factor_panel
        and factor_panel[factor].notna().any()
    ]
    train = evaluate_factor_panel(
        factor_panel,
        prices,
        membership,
        holding_days=holding_days,
        start_date=train_start_date,
        end_date=train_end_date,
        factor_columns=source_factors,
    ).set_index("factor")
    validation = evaluate_factor_panel(
        factor_panel,
        prices,
        membership,
        holding_days=holding_days,
        start_date=validation_start_date,
        end_date=validation_end_date,
        factor_columns=source_factors,
    ).set_index("factor")
    if train.empty or validation.empty:
        raise RuntimeError("training or validation factor metrics are empty")

    equal_weight = {
        factor: 1.0 / len(source_factors) for factor in source_factors
    }
    positive_train = train[
        pd.to_numeric(train["mean_rank_ic"], errors="coerce").gt(0)
    ].sort_values(
        ["mean_rank_ic", "top20_excess_forward_return"],
        ascending=False,
    )
    top_factors = positive_train.head(int(positive_ic_top_n)).index.tolist()
    if not top_factors:
        raise RuntimeError("no positive training IC factor is available")
    positive_top_weights = {
        factor: 1.0 / len(top_factors) for factor in top_factors
    }
    positive_ic = pd.to_numeric(
        positive_train["mean_rank_ic"], errors="coerce",
    ).clip(lower=0)
    train_weighted = (
        positive_ic / positive_ic.sum()
    ).to_dict()

    specs = {
        "playbook_ensemble": {
            "method": "equal_weight_all_source_factor_streams",
            "weights": equal_weight,
        },
        "playbook_positive_ic_top5": {
            "method": "equal_weight_top_positive_training_rank_ic",
            "weights": positive_top_weights,
        },
        "playbook_train_ic_weighted": {
            "method": "positive_training_rank_ic_weighted",
            "weights": train_weighted,
        },
    }
    result = factor_panel.copy()
    for factor, spec in specs.items():
        result[factor] = rank_weighted_factor_combination(
            result,
            spec["weights"],
        )

    combination_names = list(specs)
    train_combination = evaluate_factor_panel(
        result,
        prices,
        membership,
        holding_days=holding_days,
        start_date=train_start_date,
        end_date=train_end_date,
        factor_columns=combination_names,
    ).set_index("factor")
    validation_combination = evaluate_factor_panel(
        result,
        prices,
        membership,
        holding_days=holding_days,
        start_date=validation_start_date,
        end_date=validation_end_date,
        factor_columns=combination_names,
    ).set_index("factor")
    ordered = validation_combination.sort_values(
        ["top20_excess_forward_return", "mean_rank_ic"],
        ascending=False,
    )
    selected = str(ordered.index[0])
    for factor, spec in specs.items():
        spec["weights"] = {
            key: round(float(value), 12)
            for key, value in spec["weights"].items()
        }
        spec["training_metrics"] = (
            train_combination.loc[factor].to_dict()
        )
        spec["validation_metrics"] = (
            validation_combination.loc[factor].to_dict()
        )
        spec["selected_before_oos"] = factor == selected
    manifest = {
        "version": 1,
        "source_commit": QUANTSPLAYBOOK_COMMIT,
        "holding_days": int(holding_days),
        "train_start_date": str(train_start_date),
        "train_end_date": str(train_end_date),
        "validation_start_date": str(validation_start_date),
        "validation_end_date": str(validation_end_date),
        "oos_start_date": (
            pd.Timestamp(validation_end_date) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "future_return_role": "training_and_validation_label_only",
        "selection_rule": (
            "highest_validation_top20_excess_then_validation_mean_rank_ic"
        ),
        "selected_combination": selected,
        "oos_used_for_fitting_or_selection": False,
        "combinations": specs,
    }
    return result, manifest


def _optional_float(value) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _optional_bool(value) -> bool:
    return bool(value) if pd.notna(value) else False


def _candidate_profile_from_feature_row(row) -> dict:
    profile: dict = {}
    profile.update(_trade_basis_from_feature_row(row))
    profile.update(_structure_proximity_from_feature_row(row))
    profile.update(_divergence_from_feature_row(row))
    profile.update(_leadership_from_feature_row(row))
    profile.update(_right_strength_from_feature_row(row))

    raw_amount = _optional_float(row.get("_raw_amount"))
    tradestatus = _optional_float(row.get("_tradestatus"))
    price_source = str(row.get("_price_source") or "unknown")
    if bool(row.get("_asof_price_available", False)):
        price_source = f"{price_source}_raw_plus_qfq_asof"
    else:
        price_source = f"{price_source}_current_qfq"

    profile.update({
        "volatility_20": _optional_float(row.get("_volatility_20")),
        "drawdown_60": _optional_float(row.get("_drawdown_60")),
        "return_5d": _optional_float(row.get("_return_5")),
        "return_10d": _optional_float(row.get("_return_10")),
        "ma20_slope": _optional_float(row.get("_ma20_slope")),
        "ma60_slope": _optional_float(row.get("_ma60_slope")),
        "right_acceleration": _optional_float(row.get("_right_acceleration")),
        "avg_amount_20": _optional_float(row.get("_avg_amount_20")),
        "momentum_60_ex5": _optional_float(row.get("_momentum_60_ex5")),
        "momentum_120_ex20": _optional_float(row.get("_momentum_120_ex20")),
        "positive_day_ratio_60": _optional_float(
            row.get("_positive_day_ratio_60"),
        ),
        "downside_volatility_60": _optional_float(
            row.get("_downside_volatility_60"),
        ),
        "data_status": str(row.get("_data_status") or "unknown"),
        "tradestatus": tradestatus,
        "is_traded_bar": _optional_bool(row.get("_is_traded_bar")),
        "raw_amount": raw_amount,
        "amount_source": str(row.get("_amount_source") or "missing"),
        "price_source": price_source,
        "valid_price_bar": _optional_bool(row.get("_valid_price_bar")),
        "alpha_volume_price_corr_20": _optional_float(
            row.get("_alpha_volume_price_corr_20"),
        ),
        "alpha_turnover_expansion_20": _optional_float(
            row.get("_alpha_turnover_expansion_20"),
        ),
        "alpha_close_position_60": _optional_float(
            row.get("_alpha_close_position_60"),
        ),
        "alpha_close_position_120": _optional_float(
            row.get("_alpha_close_position_120"),
        ),
        "alpha_intraday_strength_20": _optional_float(
            row.get("_alpha_intraday_strength_20"),
        ),
        "alpha_gap_5d_count": _optional_float(row.get("_alpha_gap_5d_count")),
    })
    return profile


def _iter_price_frames(
    prices: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    codes: set[str] | None = None,
):
    if isinstance(prices, Mapping):
        for code, frame in prices.items():
            code = str(code)
            if codes is not None and code not in codes:
                continue
            yield code, frame.copy()
        return

    required = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(
            "candidate profile price frame misses required columns: "
            f"{sorted(missing)}",
        )
    frame = prices.copy()
    frame["code"] = frame["code"].astype(str)
    if codes is not None:
        frame = frame[frame["code"].isin(codes)]
    for code, group in frame.groupby("code", sort=True):
        yield str(code), group.drop(columns=["code"]).reset_index(drop=True)


def _candidate_qfq_close_panel(
    prices: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    codes: set[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for code, frame in _iter_price_frames(prices, codes=codes):
        if frame.empty or "date" not in frame or "close" not in frame:
            continue
        columns = ["date", "close"]
        if "raw_to_qfq_factor" in frame:
            columns.append("raw_to_qfq_factor")
        item = frame[columns].copy()
        item["code"] = code
        frames.append(item)
    if not frames:
        return pd.DataFrame()
    return _qfq_close_panel(pd.concat(frames, ignore_index=True))


def build_candidate_trading_profile_enrichment(
    prices: pd.DataFrame | Mapping[str, pd.DataFrame],
    snapshot_dates: list[pd.Timestamp],
    *,
    codes=None,
    candidate_codes_by_date: Mapping[str, object] | None = None,
    amount_multiplier: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """Build visible-at-close trading profiles for candidate snapshot rows."""
    dates = sorted({
        pd.Timestamp(value).normalize()
        for value in snapshot_dates
        if pd.notna(value)
    })
    enrichment = {
        date.strftime("%Y-%m-%d"): {}
        for date in dates
    }
    if not dates:
        return enrichment

    code_filter = None if codes is None else {str(value) for value in codes}
    dates_by_code: dict[str, list[pd.Timestamp]] | None = None
    if candidate_codes_by_date is not None:
        valid_dates = set(dates)
        dates_by_code = {}
        for date_value, date_codes in candidate_codes_by_date.items():
            date = pd.Timestamp(date_value).normalize()
            if date not in valid_dates:
                continue
            for code in date_codes or ():
                dates_by_code.setdefault(str(code), []).append(date)
        for code in dates_by_code:
            dates_by_code[code] = sorted(set(dates_by_code[code]))
        if code_filter is None:
            code_filter = set(dates_by_code)
        else:
            code_filter &= set(dates_by_code)
    for code, frame in _iter_price_frames(prices, codes=code_filter):
        if frame.empty or "date" not in frame:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values(
            "date", kind="mergesort",
        )
        frame["date"] = frame["date"].dt.normalize()
        frame = frame.drop_duplicates("date", keep="last").set_index("date")
        if frame.empty:
            continue
        adjustment = pd.to_numeric(
            frame.get(
                "raw_to_qfq_factor",
                pd.Series(1.0, index=frame.index),
            ),
            errors="coerce",
        ).where(lambda value: value > 0, 1.0)
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce",
            ).div(adjustment)
        if "amount" in frame and float(amount_multiplier) != 1.0:
            frame["amount"] = pd.to_numeric(
                frame["amount"], errors="coerce",
            ).mul(float(amount_multiplier))
        features = _candidate_feature_frame(frame)
        features = features[~features.index.duplicated(keep="last")]
        feature_dates = pd.DatetimeIndex(features.index).sort_values()
        features = features.reindex(feature_dates)
        code_dates = (
            dates if dates_by_code is None
            else dates_by_code.get(code, [])
        )
        if not code_dates:
            continue
        date_index = pd.DatetimeIndex(code_dates)
        positions = feature_dates.searchsorted(date_index, side="right") - 1
        for snapshot_date, position in zip(date_index, positions):
            if position < 0:
                continue
            profile_date = feature_dates[position]
            profile = _candidate_profile_from_feature_row(
                features.iloc[int(position)],
            )
            profile["candidate_profile_price_date"] = profile_date.strftime(
                "%Y-%m-%d",
            )
            profile["candidate_profile_stale_calendar_days"] = int(
                (snapshot_date - profile_date).days,
            )
            profile["candidate_profile_status"] = (
                "aligned_observation_bar"
                if profile_date == snapshot_date
                else "prior_observation_bar"
            )
            profile["candidate_profile_rule"] = (
                "price_features_use_bars_on_or_before_snapshot_date"
            )
            enrichment[snapshot_date.strftime("%Y-%m-%d")][code] = profile
    return enrichment


def merge_candidate_snapshot_enrichment(
    snapshots: dict[str, list[dict]],
    candidate_enrichment_by_date: dict[str, dict[str, dict]] | None,
) -> dict[str, list[dict]]:
    if candidate_enrichment_by_date is None:
        return snapshots
    merged: dict[str, list[dict]] = {}
    for date, rows in snapshots.items():
        by_code = candidate_enrichment_by_date.get(str(date), {})
        merged_rows = []
        for row in rows:
            enriched = dict(row)
            profile = by_code.get(str(row.get("code")))
            if profile:
                enriched.update(profile)
            else:
                enriched["candidate_profile_status"] = "missing_price_profile"
            merged_rows.append(enriched)
        merged[str(date)] = merged_rows
    return merged


def _outcome_label_cn(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "标签缺失"
    if float(number) >= 0.10:
        return "强盈利"
    if float(number) > 0:
        return "盈利"
    if float(number) >= -0.05:
        return "小亏"
    return "亏损"


def build_candidate_outcome_labels_from_close_panel(
    snapshots: dict[str, list[dict]],
    close: pd.DataFrame,
    *,
    holding_days: int,
    label_end_date: str | None = None,
) -> pd.DataFrame:
    """Create post-hoc research labels without feeding them back into candidates."""
    if close.empty:
        return pd.DataFrame()
    close = close.sort_index()
    close.index = pd.to_datetime(close.index, errors="coerce").normalize()
    close = close[~close.index.duplicated(keep="last")]
    horizon = int(holding_days)
    forward = close.shift(-horizon).div(close) - 1.0
    realized_dates = pd.Series(close.index, index=close.index).shift(-horizon)
    label_end = None if label_end_date is None else pd.Timestamp(label_end_date)
    rows = []
    for snapshot_date, candidate_rows in sorted(snapshots.items()):
        date = pd.Timestamp(snapshot_date).normalize()
        for row in candidate_rows:
            code = str(row.get("code") or "")
            entry_close = exit_close = forward_return = None
            label_date = pd.NaT
            if date in close.index and code in close.columns:
                entry_close = _optional_float(close.at[date, code])
                label_date = realized_dates.get(date, pd.NaT)
                if pd.notna(label_date) and code in close.columns:
                    exit_close = _optional_float(close.at[label_date, code])
                    forward_return = _optional_float(forward.at[date, code])
            label_available = (
                pd.notna(label_date)
                and forward_return is not None
                and (label_end is None or pd.Timestamp(label_date) <= label_end)
            )
            if not label_available:
                forward_return = None
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "playbook_factor": row.get("playbook_factor"),
                "playbook_factor_rank": row.get("playbook_factor_rank"),
                "candidate_score": row.get("candidate_score"),
                "holding_days": horizon,
                "label_date": (
                    None if pd.isna(label_date)
                    else pd.Timestamp(label_date).strftime("%Y-%m-%d")
                ),
                "entry_close_qfq": entry_close,
                "exit_close_qfq": exit_close if label_available else None,
                "forward_return": forward_return,
                "outcome_label_cn": _outcome_label_cn(forward_return),
                "label_available": bool(label_available),
                "label_role": "research_label_only_not_candidate_input",
                "trade_basis_score": row.get("trade_basis_score"),
                "structure_proximity_score": row.get(
                    "structure_proximity_score",
                ),
                "right_strength_score": row.get("right_strength_score"),
                "leadership_score": row.get("leadership_score"),
                "data_status": row.get("data_status"),
            })
    return pd.DataFrame(rows)


def build_candidate_outcome_labels(
    snapshots: dict[str, list[dict]],
    prices: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    holding_days: int,
    label_end_date: str | None = None,
) -> pd.DataFrame:
    codes = {
        str(row.get("code"))
        for rows in snapshots.values()
        for row in rows
        if row.get("code")
    }
    close = _candidate_qfq_close_panel(prices, codes=codes)
    return build_candidate_outcome_labels_from_close_panel(
        snapshots,
        close,
        holding_days=holding_days,
        label_end_date=label_end_date,
    )


def factor_candidate_snapshots(
    factor_panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    factor_column: str,
    snapshot_dates: list[pd.Timestamp],
    top_n: int,
    trading_top_n: int | None = None,
    candidate_enrichment_by_date: dict[str, dict[str, dict]] | None = None,
) -> dict[str, list[dict]]:
    """Select Top-N directly from each point-in-time full-market factor cross-section."""
    if factor_column not in factor_panel:
        raise ValueError(f"factor column is unavailable: {factor_column}")
    eligible_index = pd.MultiIndex.from_frame(membership[["date", "code"]])
    scores = pd.to_numeric(
        factor_panel[factor_column].reindex(eligible_index), errors="coerce",
    ).dropna()
    by_date = {
        pd.Timestamp(date).normalize(): group.droplevel("date")
        for date, group in scores.groupby(level="date", sort=True)
    }
    snapshots: dict[str, list[dict]] = {}
    for date in snapshot_dates:
        date = pd.Timestamp(date).normalize()
        cross_section = by_date.get(date, pd.Series(dtype=float))
        cross_section = cross_section[
            ~cross_section.index.duplicated(keep="last")
        ]
        ranked = cross_section.sort_values(
            ascending=False, kind="mergesort",
        )
        percentile = cross_section.rank(
            method="average", ascending=True, pct=True,
        )
        rows = []
        for rank, (code, score) in enumerate(
            ranked.head(int(top_n)).items(), 1,
        ):
            trading_eligible = (
                trading_top_n is None or rank <= int(trading_top_n)
            )
            rows.append({
                "code": str(code),
                # Current names and industries are deliberately not fed into
                # historical execution because their current snapshots can
                # leak later ST or classification changes.
                "name": str(code),
                "strategy_part": f"quantsplaybook:{factor_column}",
                "selection_engine": "quantsplaybook_factor_only",
                "selection_profile": "quantsplaybook_factor_only",
                "playbook_factor": factor_column,
                "playbook_factor_score": float(score),
                "playbook_factor_rank": int(rank),
                "candidate_score": round(float(percentile.loc[code]) * 100.0, 6),
                "selected_for_trading": trading_eligible,
                "signal_eligible": trading_eligible,
                "allow_right": trading_eligible,
                "allow_left": False,
                "candidate_failure_reason": (
                    "" if trading_eligible
                    else "not_selected_for_trading: factor_rank_not_in_observation_pool"
                ),
            })
        snapshots[date.strftime("%Y-%m-%d")] = rows
    return merge_candidate_snapshot_enrichment(
        snapshots,
        candidate_enrichment_by_date,
    )


def save_factor_snapshots(
    snapshots: dict[str, list[dict]],
    *,
    output_directory: str | Path,
    factor_column: str,
    top_n: int,
    start_date: str,
    end_date: str,
    factor_metadata: dict | None = None,
    candidate_outcomes: pd.DataFrame | None = None,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    declared_factors = {str(factor_column)}
    if factor_metadata:
        declared_factors.update(
            str(value)
            for value in (factor_metadata.get("selected_factors") or [])
        )
        declared_factors.update(
            str(value)
            for value in (factor_metadata.get("weights") or {})
        )
    inferred_financial_data = (
        "low_idiosyncratic_volatility" in declared_factors
    )
    metadata = factor_metadata or {}
    uses_financial_data = bool(metadata.get(
        "uses_financial_data", inferred_financial_data,
    ))
    financial_point_in_time = bool(metadata.get(
        "financial_point_in_time", not uses_financial_data,
    ))
    strict_financial_point_in_time = bool(metadata.get(
        "strict_financial_point_in_time", not uses_financial_data,
    ))
    industry_point_in_time = bool(metadata.get(
        "industry_point_in_time", False,
    ))
    manifest_rows = []
    for date, rows in sorted(snapshots.items()):
        path = output / f"candidates_{date}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(rows),
            "signal_eligible_count": sum(
                bool(row.get("signal_eligible", True)) for row in rows
            ),
            "financial_point_in_time": financial_point_in_time,
            "strict_financial_point_in_time": strict_financial_point_in_time,
            "industry_point_in_time": industry_point_in_time,
        })
    profile_enriched = any(
        "candidate_profile_status" in row
        for rows in snapshots.values()
        for row in rows
    )
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(manifest_rows),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "financial_point_in_time": financial_point_in_time,
        "strict_financial_point_in_time": strict_financial_point_in_time,
        "uses_financial_data": uses_financial_data,
        "industry_point_in_time": industry_point_in_time,
        "selection_engine": "quantsplaybook_factor_only",
        "old_candidate_pool_used": False,
        "source_commit": QUANTSPLAYBOOK_COMMIT,
        "factor": factor_column,
        "top_n": int(top_n),
        "trading_top_n": metadata.get("trading_top_n"),
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "universe_rule": (
            "listed_by_date;not_past_delist_date;valid_traded_bar;"
            "beijing_exchange_not_before_2021_11_15;minimum_history;"
            "no_old_selection_gate"
        ),
        "candidate_profile_enriched": profile_enriched,
        "candidate_profile_rule": (
            "price_features_use_bars_on_or_before_snapshot_date"
        ),
        "candidate_profile_extra_fields": (
            list(CANDIDATE_PROFILE_EXTRA_FIELDS) if profile_enriched else []
        ),
        "snapshots": manifest_rows,
    }
    if uses_financial_data and not strict_financial_point_in_time:
        manifest.update({
            "research_only": True,
            "financial_point_in_time_reason": (
                metadata.get("financial_point_in_time_reason")
                or "low_idiosyncratic_volatility uses historical daily PB for "
                "HML; the local provider cache does not preserve all "
                "historical valuation revisions"
            ),
        })
    if factor_metadata:
        manifest["factor_metadata"] = factor_metadata
        for field in (
            "research_only",
            "research_only_reason",
            "financial_point_in_time_reason",
            "industry_point_in_time_status",
            "industry_data_source",
            "industry_source_path",
            "industry_source_sha256",
            "industry_source_modified_at",
            "industry_as_of_date",
            "industry_mapping_count",
            "industry_universe_count",
            "industry_coverage_ratio",
        ):
            if field in factor_metadata:
                manifest[field] = factor_metadata[field]
    if candidate_outcomes is not None:
        outcome_path = output / "candidate_outcomes.csv"
        candidate_outcomes.to_csv(
            outcome_path, index=False, encoding="utf-8-sig",
        )
        available = (
            candidate_outcomes["label_available"]
            if "label_available" in candidate_outcomes
            else pd.Series(dtype=bool)
        )
        manifest["candidate_outcome_labels"] = {
            "file": outcome_path.name,
            "sha256": _sha256(outcome_path),
            "row_count": int(len(candidate_outcomes)),
            "available_count": int(available.fillna(False).sum()),
            "label_role": "research_label_only_not_candidate_input",
            "future_return_used_for_candidate_selection": False,
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def prepare_factor_candidates(
    factor_panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    factors: list[str],
    snapshot_dates: list[pd.Timestamp],
    output_root: str | Path,
    top_n: int,
    start_date: str,
    end_date: str,
    trading_top_n: int | None = None,
    factor_metadata_by_name: dict[str, dict] | None = None,
    candidate_enrichment_by_date: dict[str, dict[str, dict]] | None = None,
    candidate_enrichment_prices: (
        pd.DataFrame | Mapping[str, pd.DataFrame] | None
    ) = None,
    candidate_enrichment_price_loader=None,
    candidate_outcome_holding_days: int | None = None,
    candidate_enrichment_amount_multiplier: float = 1.0,
) -> pd.DataFrame:
    rows = []
    pending = []
    selected_codes: set[str] = set()
    selected_codes_by_date: dict[str, set[str]] = {}
    for factor in factors:
        snapshots = factor_candidate_snapshots(
            factor_panel,
            membership,
            factor_column=factor,
            snapshot_dates=snapshot_dates,
            top_n=top_n,
            trading_top_n=trading_top_n,
        )
        selected_codes.update(
            str(row["code"])
            for factor_rows in snapshots.values()
            for row in factor_rows
        )
        for date, factor_rows in snapshots.items():
            selected_codes_by_date.setdefault(str(date), set()).update(
                str(row["code"]) for row in factor_rows
            )
        pending.append((factor, snapshots))

    if (
        candidate_enrichment_by_date is None
        and candidate_enrichment_prices is not None
    ):
        candidate_enrichment_by_date = build_candidate_trading_profile_enrichment(
            candidate_enrichment_prices,
            snapshot_dates,
            codes=selected_codes,
            candidate_codes_by_date=selected_codes_by_date,
            amount_multiplier=candidate_enrichment_amount_multiplier,
        )
    if (
        candidate_enrichment_by_date is None
        and candidate_enrichment_prices is None
        and candidate_enrichment_price_loader is not None
        and selected_codes
    ):
        candidate_enrichment_prices = candidate_enrichment_price_loader(
            selected_codes,
        )
        candidate_enrichment_by_date = build_candidate_trading_profile_enrichment(
            candidate_enrichment_prices,
            snapshot_dates,
            codes=selected_codes,
            candidate_codes_by_date=selected_codes_by_date,
            amount_multiplier=candidate_enrichment_amount_multiplier,
        )
    outcome_close = None
    if (
        candidate_outcome_holding_days is not None
        and candidate_enrichment_prices is not None
    ):
        outcome_close = _candidate_qfq_close_panel(
            candidate_enrichment_prices,
            codes=selected_codes,
        )

    for factor, snapshots in pending:
        snapshots = merge_candidate_snapshot_enrichment(
            snapshots,
            candidate_enrichment_by_date,
        )
        candidate_outcomes = None
        if outcome_close is not None:
            candidate_outcomes = build_candidate_outcome_labels_from_close_panel(
                snapshots,
                outcome_close,
                holding_days=int(candidate_outcome_holding_days),
                label_end_date=end_date,
            )
        directory = save_factor_snapshots(
            snapshots,
            output_directory=Path(output_root) / "candidates" / factor,
            factor_column=factor,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
            factor_metadata=(
                (factor_metadata_by_name or {}).get(factor)
            ),
            candidate_outcomes=candidate_outcomes,
        )
        counts = [len(value) for value in snapshots.values()]
        rows.append({
            "factor": factor,
            "candidate_directory": str(directory),
            "snapshot_count": len(snapshots),
            "minimum_candidates": min(counts) if counts else 0,
            "maximum_candidates": max(counts) if counts else 0,
            "mean_candidates": round(float(np.mean(counts)), 3) if counts else 0,
        })
    return pd.DataFrame(rows)


def _portfolio_summary_row(factor: str, summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    concentration = summary.get("profit_concentration_summary") or {}
    trades = summary.get("trade_summary") or {}
    sell_rows = [
        row for row in summary.get("trade_ledger") or []
        if row.get("trade_side") == "卖出"
        and pd.notna(pd.to_numeric(row.get("profit_loss_amount"), errors="coerce"))
    ]
    profitable = [
        float(row["profit_loss_amount"]) for row in sell_rows
        if float(row["profit_loss_amount"]) > 0
    ]
    losing = [
        float(row["profit_loss_amount"]) for row in sell_rows
        if float(row["profit_loss_amount"]) < 0
    ]
    profit_loss_ratio = (
        float(np.mean(profitable)) / abs(float(np.mean(losing)))
        if profitable and losing else None
    )
    return {
        "factor": factor,
        "final_return_pct": summary.get("final_return_pct"),
        "maximum_drawdown_pct": summary.get("maximum_drawdown_pct"),
        "realized_return_pct": summary.get("realized_return_pct"),
        "unrealized_return_pct": summary.get("unrealized_return_pct"),
        "transaction_cost_pct": summary.get("transaction_cost_pct"),
        "sell_count": trades.get("sell_count"),
        "win_rate_pct": trades.get("sell_win_rate_pct"),
        "profit_loss_ratio": (
            None if profit_loss_ratio is None
            else round(profit_loss_ratio, 6)
        ),
        "top1_return_contribution_pct": concentration.get(
            "top1_return_contribution_pct",
        ),
        "top1_positive_profit_share_pct": concentration.get(
            "top1_positive_profit_share_pct",
        ),
        "exclude_top1_approx_final_return_pct": concentration.get(
            "exclude_top1_approx_final_return_pct",
        ),
        "top3_return_contribution_pct": concentration.get(
            "top3_return_contribution_pct",
        ),
        "top3_positive_profit_share_pct": concentration.get(
            "top3_positive_profit_share_pct",
        ),
        "exclude_top3_approx_final_return_pct": concentration.get(
            "exclude_top3_approx_final_return_pct",
        ),
        "concentration_warning": concentration.get("concentration_warning"),
        "summary_path": str(summary_path),
    }


def audit_portfolio_trades(
    result: dict,
    prepared_price_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    """Audit every fill against raw bars and every new entry against PIT timing."""
    rows = []
    for trade in result.get("trade_ledger") or []:
        side = str(trade.get("trade_side") or "")
        if side not in {"买入", "卖出"}:
            continue
        code = str(trade.get("code") or "")
        date = pd.Timestamp(trade.get("date")).normalize()
        frame = prepared_price_frames.get(code)
        matched = (
            pd.DataFrame()
            if frame is None
            else frame[frame["date"].dt.normalize().eq(date)]
        )
        raw_low = raw_high = None
        if not matched.empty:
            raw_low = pd.to_numeric(
                matched.iloc[-1].get("raw_low", matched.iloc[-1].get("low")),
                errors="coerce",
            )
            raw_high = pd.to_numeric(
                matched.iloc[-1].get("raw_high", matched.iloc[-1].get("high")),
                errors="coerce",
            )
        execution_price = pd.to_numeric(
            trade.get("execution_price", trade.get("price")),
            errors="coerce",
        )
        price_inside_bar = bool(
            pd.notna(execution_price)
            and pd.notna(raw_low)
            and pd.notna(raw_high)
            and float(raw_low) - 1e-6
            <= float(execution_price)
            <= float(raw_high) + 1e-6
        )
        snapshot_date = pd.to_datetime(
            trade.get("candidate_snapshot_date"), errors="coerce",
        )
        candidate_pit_ok = (
            None if side != "买入"
            else bool(pd.notna(snapshot_date) and snapshot_date.normalize() < date)
        )
        technical_date = pd.to_datetime(
            trade.get("technical_signal_date"), errors="coerce",
        )
        requires_technical_signal = (
            side == "买入"
            and str(trade.get("account_mode") or "").strip() != "left"
        )
        technical_timing_ok = (
            None if not requires_technical_signal
            else bool(pd.notna(technical_date) and technical_date.normalize() <= date)
        )
        trigger = pd.to_numeric(trade.get("trigger"), errors="coerce")
        trigger_raw = pd.to_numeric(
            trade.get("trigger_raw_price"), errors="coerce",
        )
        conversion_ok = None
        if side == "买入" and pd.notna(trigger) and pd.notna(trigger_raw):
            conversion_ok = bool(float(trigger) > 0 and float(trigger_raw) > 0)
        violations = []
        if not price_inside_bar:
            violations.append("execution_outside_raw_bar")
        if candidate_pit_ok is False:
            violations.append("candidate_not_strictly_before_execution")
        if technical_timing_ok is False:
            violations.append("technical_signal_after_execution")
        if conversion_ok is False:
            violations.append("invalid_qfq_raw_conversion")
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "code": code,
            "trade_side": side,
            "execution_price": execution_price,
            "raw_low": raw_low,
            "raw_high": raw_high,
            "price_inside_raw_bar": price_inside_bar,
            "candidate_snapshot_date": (
                None if pd.isna(snapshot_date)
                else snapshot_date.strftime("%Y-%m-%d")
            ),
            "candidate_strictly_before_execution": candidate_pit_ok,
            "technical_signal_date": (
                None if pd.isna(technical_date)
                else technical_date.strftime("%Y-%m-%d")
            ),
            "technical_signal_no_later_than_execution": technical_timing_ok,
            "technical_signal_timing": trade.get("technical_signal_timing"),
            "signal_price_basis": trade.get("signal_price_basis"),
            "execution_price_basis": trade.get("execution_price_basis"),
            "qfq_raw_conversion_positive": conversion_ok,
            "violations": ";".join(violations),
        })
    audit = pd.DataFrame(rows)
    violation_rows = (
        audit[audit["violations"].astype(str).ne("")]
        if not audit.empty else audit
    )
    summary = {
        "audited_trade_count": int(len(audit)),
        "audited_buy_count": int((audit.get("trade_side") == "买入").sum())
        if not audit.empty else 0,
        "violation_count": int(len(violation_rows)),
        "all_execution_prices_inside_raw_daily_bar": bool(
            not audit.empty and audit["price_inside_raw_bar"].all()
        ),
        "all_buy_candidates_strictly_before_execution": bool(
            not audit.empty
            and audit.loc[
                audit["trade_side"].eq("买入"),
                "candidate_strictly_before_execution",
            ].eq(True).all()
        ),
        "close_proxy_caveat": (
            "same-day close-confirmed signals execute at a 14:55/close proxy; "
            "this is not a future-data leak but is an optimistic execution assumption"
        ),
    }
    return audit, summary


def _save_shared_portfolio_result(
    result: dict,
    *,
    factor: str,
    snapshots: dict,
    output_root: str | Path,
    start_date: str,
    end_date: str,
    candidate_directory: Path,
    formula_history: str | Path,
) -> Path:
    output = Path(output_root) / "portfolio" / factor
    output.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{start_date}_{end_date}"
    pd.DataFrame(result["events"]).to_csv(
        output / f"{stem}_events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["trade_ledger"]).to_csv(
        output / f"{stem}_trades.csv", index=False, encoding="utf-8-sig",
    )
    portfolio_backtest.build_readable_trade_frame(result["trade_ledger"]).to_csv(
        output / f"{stem}_买卖流水.csv", index=False, encoding="utf-8-sig",
    )
    (output / f"{stem}_买卖报告.md").write_text(
        portfolio_backtest.render_trade_report_markdown(result),
        encoding="utf-8",
    )
    pd.DataFrame(result["equity_curve"]).to_csv(
        output / f"{stem}_equity.csv", index=False, encoding="utf-8-sig",
    )
    trade_audit = pd.DataFrame(result.pop("_trade_audit_rows", []))
    trade_audit.to_csv(
        output / f"{stem}_trade_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        key: value for key, value in result.items()
        if key not in {"events", "equity_curve"}
    }
    manifest_path = candidate_directory / "manifest.json"
    summary.update({
        "candidate_mode": "rolling",
        "candidate_snapshot_dates": sorted(snapshots),
        "input_fingerprints": {
            "candidate_directory": str(candidate_directory),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": (
                _sha256(manifest_path) if manifest_path.exists() else None
            ),
            "formula_history": str(Path(formula_history)),
            "formula_history_sha256": _sha256(Path(formula_history)),
            "price_source": "shared_full_market_tushare_sqlite_frame",
        },
        "inputs_refreshed": False,
        "frozen_inputs_allowed": True,
        "input_coverage_end": str(end_date),
        "requested_end_date": str(end_date),
        "effective_end_date": str(end_date),
    })
    summary_path = output / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def run_portfolio_comparison(
    factors: list[str],
    *,
    output_root: str | Path,
    prices: pd.DataFrame,
    stock_basic: pd.DataFrame,
    repository: TushareRepository,
    formula_history: str | Path,
    start_date: str,
    end_date: str,
    prepared_price_frames: dict[str, pd.DataFrame] | None = None,
    max_left_positions: int = 2,
    factor_only_breadth_reversal_only: bool = False,
    breadth_reversal_max_unique_count: int = 50,
    breadth_reversal_risk_budget: float = 0.006,
    breadth_reversal_max_position: float = 0.20,
    breadth_reversal_time_limit_days: int = 20,
    factor_only_compressed_structure_only: bool = False,
    compressed_structure_risk_budget: float = 0.015,
    compressed_structure_stop_loss_pct: float = 0.08,
    compressed_structure_max_position: float = 0.20,
    compressed_structure_holding_days: int = 20,
    compressed_structure_min_entry_evidence_score: float = 9.0,
    compressed_structure_min_proximity_score: float = 40.0,
) -> pd.DataFrame:
    snapshots_by_factor = {}
    all_codes = set()
    for factor in factors:
        candidate_directory = Path(output_root) / "candidates" / factor
        snapshots = portfolio_backtest.load_candidate_snapshots(
            candidate_directory, start_date, end_date,
        )
        snapshots_by_factor[factor] = snapshots
        all_codes.update(
            str(row["code"])
            for rows in snapshots.values()
            for row in rows
        )

    if prepared_price_frames is None:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.dropna(subset=["date", "code"])
        selected_prices = prices[
            prices["code"].astype(str).isin(all_codes)
        ].copy()
        limit_prices = repository.load_dataset_for_codes(
            "stk_limit",
            all_codes,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        if not limit_prices.empty:
            limit_prices = limit_prices[
                ["date", "code", "up_limit", "down_limit"]
            ].copy()
            limit_prices["date"] = pd.to_datetime(
                limit_prices["date"], errors="coerce",
            )
            for column in ("up_limit", "down_limit"):
                limit_prices[column] = pd.to_numeric(
                    limit_prices[column], errors="coerce",
                )
            selected_prices = selected_prices.merge(
                limit_prices.drop_duplicates(["date", "code"], keep="last"),
                on=["date", "code"],
                how="left",
            )
        price_frames = {
            str(code): group.drop(columns=["code"]).reset_index(drop=True)
            for code, group in selected_prices.groupby("code", sort=True)
        }
        del selected_prices
        price_frames = portfolio_backtest.prepare_portfolio_price_frames(
            price_frames,
        )
        price_mode = "SQLite统一缓存一次加载，多因子共享"
        daily_limit_source = (
            "local_tushare_stk_limit"
            if not limit_prices.empty else "board_rule_fallback"
        )
        daily_limit_row_count = int(len(limit_prices))
    else:
        missing = sorted(all_codes - set(prepared_price_frames))
        if missing:
            raise RuntimeError(
                "prepared price cache misses candidate codes: "
                f"count={len(missing)} sample={missing[:10]}",
            )
        price_frames = {
            code: prepared_price_frames[code]
            for code in sorted(all_codes)
        }
        price_mode = "按代码加载已校验的Tushare预处理执行行情缓存"
        daily_limit_source = "prepared_local_tushare_stk_limit"
        daily_limit_row_count = None
    trade_calendar = sorted({
        pd.Timestamp(value).normalize()
        for frame in price_frames.values()
        for value in frame.loc[
            frame["date"].between(
                pd.Timestamp(start_date), pd.Timestamp(end_date),
            ),
            "date",
        ].dropna()
    })
    formula = pd.read_csv(formula_history)
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
            "window_unique_count": (
                None
                if pd.isna(pd.to_numeric(
                    row.get("window_unique_count"), errors="coerce",
                ))
                else int(pd.to_numeric(
                    row.get("window_unique_count"), errors="coerce",
                ))
            ),
        }
        for _, row in formula.iterrows()
    }
    corporate_actions = repository.load_dividend_actions(
        all_codes,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    basic = stock_basic[
        stock_basic["code"].astype(str).isin(all_codes)
        & stock_basic["delist_date"].notna()
    ]
    security_end_dates = {
        str(row["code"]): pd.Timestamp(row["delist_date"]).normalize()
        for _, row in basic.iterrows()
    }
    suspension_frame = repository.load_dataset_for_codes(
        "suspend_d",
        all_codes,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    security_suspension_dates = {
        str(code): {
            pd.Timestamp(value).normalize()
            for value in group["date"].dropna()
        }
        for code, group in suspension_frame.groupby("code", sort=False)
    } if not suspension_frame.empty else {}
    coverage_by_factor = {}
    for factor, snapshots in snapshots_by_factor.items():
        codes = {
            str(row["code"])
            for snapshot_rows in snapshots.values()
            for row in snapshot_rows
        }
        coverage_by_factor[factor] = (
            portfolio_backtest.validate_price_frame_coverage(
                price_frames,
                codes,
                start_date,
                end_date,
                code_start_dates=portfolio_backtest.first_candidate_dates(
                    snapshots,
                ),
                security_end_dates=security_end_dates,
                security_suspension_dates=security_suspension_dates,
                trade_calendar=trade_calendar,
            )
        )
    rows = []
    for index, factor in enumerate(factors, 1):
        print(
            f"[quantsplaybook] portfolio {index}/{len(factors)} factor={factor}",
            flush=True,
        )
        candidate_directory = Path(output_root) / "candidates" / factor
        snapshots = snapshots_by_factor[factor]
        codes = {
            str(row["code"])
            for snapshot_rows in snapshots.values()
            for row in snapshot_rows
        }
        coverage = coverage_by_factor[factor]
        result = portfolio_backtest.run_portfolio_backtest(
            {
                code: price_frames[code]
                for code in sorted(codes)
                if code in price_frames
            },
            snapshots,
            phases,
            requested_start=start_date,
            end_date=end_date,
            max_positions=5,
            max_total_held_symbols=5,
            max_left_positions=max_left_positions,
            max_same_industry=2,
            same_theme_correlation=0.60,
            min_entry_evidence_score=0.0,
            profit_tranches=5,
            profit_tail_min_return=0.50,
            left_grid_unit=0.02,
            left_grid_step=0.05,
            left_grid_max_exposure=0.10,
            signals_effective_next_day=True,
            auto_price_structure=True,
            allow_structure_pullback=True,
            allow_pullback_pilot=True,
            enable_breadth_reversal_half_pullback=(
                factor_only_breadth_reversal_only
            ),
            factor_only_breadth_reversal_only=(
                factor_only_breadth_reversal_only
            ),
            breadth_reversal_max_unique_count=(
                breadth_reversal_max_unique_count
            ),
            breadth_reversal_risk_budget=breadth_reversal_risk_budget,
            breadth_reversal_max_position=breadth_reversal_max_position,
            breadth_reversal_time_limit_days=(
                breadth_reversal_time_limit_days
            ),
            factor_only_compressed_structure_only=(
                factor_only_compressed_structure_only
            ),
            compressed_structure_risk_budget=(
                compressed_structure_risk_budget
            ),
            compressed_structure_stop_loss_pct=(
                compressed_structure_stop_loss_pct
            ),
            compressed_structure_max_position=(
                compressed_structure_max_position
            ),
            compressed_structure_holding_days=(
                compressed_structure_holding_days
            ),
            compressed_structure_min_entry_evidence_score=(
                compressed_structure_min_entry_evidence_score
            ),
            compressed_structure_min_proximity_score=(
                compressed_structure_min_proximity_score
            ),
            close_confirmed_execution="close_proxy",
            commission_rate=0.000085,
            minimum_commission=5.0,
            initial_capital=1_000_000.0,
            sell_stamp_duty_rate=0.0005,
            estimated_slippage_rate=0.0005,
            corporate_actions=corporate_actions,
            security_end_dates=security_end_dates,
            price_frames_prepared=True,
        )
        result["price_source"] = {
            "source": "tushare",
            "mode": price_mode,
            "coverage": coverage,
            "daily_limit_source": daily_limit_source,
            "daily_limit_row_count": daily_limit_row_count,
        }
        result["corporate_action_source"] = {
            "source": "local_tushare_dividend",
            "loaded_symbol_count": len(corporate_actions),
            "loaded_event_count": sum(
                len(action_rows) for action_rows in corporate_actions.values()
            ),
        }
        apply_stock_basic_display_names(result, stock_basic)
        trade_audit, trade_audit_summary = audit_portfolio_trades(
            result,
            {
                code: price_frames[code]
                for code in sorted(codes)
                if code in price_frames
            },
        )
        if trade_audit_summary["violation_count"]:
            sample = trade_audit.loc[
                trade_audit["violations"].astype(str).ne("")
            ].head(10)
            raise RuntimeError(
                "trade audit failed:\n" + sample.to_string(index=False),
            )
        result["trade_audit_summary"] = trade_audit_summary
        result["_trade_audit_rows"] = trade_audit.to_dict("records")
        summary_path = _save_shared_portfolio_result(
            result,
            factor=factor,
            snapshots=snapshots,
            output_root=output_root,
            start_date=start_date,
            end_date=end_date,
            candidate_directory=candidate_directory,
            formula_history=formula_history,
        )
        rows.append(_portfolio_summary_row(factor, summary_path))
    return pd.DataFrame(rows).sort_values(
        ["exclude_top1_approx_final_return_pct", "final_return_pct"],
        ascending=False,
    ).reset_index(drop=True)


def aggregate_portfolio_summaries(output_root: str | Path) -> pd.DataFrame:
    """Collect every completed factor/period summary without trusting stale CSVs."""
    rows = []
    root = Path(output_root)
    for path in sorted((root / "portfolio").glob("*/*_summary.json")):
        factor = path.parent.name
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = _portfolio_summary_row(factor, path)
        row.update({
            "requested_start": summary.get("requested_start"),
            "actual_start": summary.get("actual_start"),
            "end_date": summary.get("end_date"),
            "coverage_complete": summary.get("coverage_complete"),
            "trade_audit_violation_count": (
                (summary.get("trade_audit_summary") or {}).get(
                    "violation_count",
                )
            ),
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "requested_start",
            "end_date",
            "exclude_top1_approx_final_return_pct",
            "final_return_pct",
        ],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def render_portfolio_comparison_markdown(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "# QuantsPlaybook 回测比较\n\n暂无完整回测结果。\n"
    columns = [
        "requested_start", "end_date", "factor", "final_return_pct",
        "maximum_drawdown_pct", "sell_count", "win_rate_pct",
        "profit_loss_ratio", "exclude_top1_approx_final_return_pct",
        "trade_audit_violation_count",
    ]
    display = metrics.reindex(columns=columns).copy()
    display.columns = [
        "开始", "结束", "因子/组合", "收益率%", "最大回撤%", "卖出次数",
        "胜率%", "盈亏比", "剔除头部1近似收益%", "成交审计违规",
    ]
    return "\n".join([
        "# QuantsPlaybook 回测比较",
        "",
        "所有行使用同一买卖、仓位、T+1、整手、费用、涨跌停和分红送转口径。",
        "剔除头部贡献是诊断值，不是重新分配现金后的独立回测。",
        "",
        display.to_markdown(index=False),
        "",
    ])


def prepared_price_cache_path(
    cache_directory: str | Path,
    code: str,
) -> Path:
    return Path(cache_directory) / f"{str(code).replace('.', '_')}.pkl"


def load_prepared_price_cache(
    cache_directory: str | Path,
    codes,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    frames = {}
    missing = []
    for code in sorted({str(value) for value in codes}):
        path = prepared_price_cache_path(cache_directory, code)
        if not path.is_file():
            missing.append(code)
            continue
        frame = pd.read_pickle(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if start_date is not None:
            frame = frame[frame["date"].ge(pd.Timestamp(start_date))]
        if end_date is not None:
            frame = frame[frame["date"].le(pd.Timestamp(end_date))]
        frame = frame.reset_index(drop=True)
        frames[code] = frame
    if missing:
        raise RuntimeError(
            "prepared price cache is incomplete: "
            f"count={len(missing)} sample={missing[:10]}",
        )
    return frames


def _formula_snapshot_dates(
    path: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    formula = pd.read_csv(path)
    dates = pd.to_datetime(formula.get("date"), errors="coerce").dropna()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return sorted({
        value.normalize() for value in dates
        if start <= value.normalize() <= end
    })


def required_history_start_date(
    start_date: str,
    *,
    calendar_days: int = REQUIRED_HISTORY_CALENDAR_DAYS,
) -> pd.Timestamp:
    return (
        pd.Timestamp(start_date).normalize()
        - pd.Timedelta(days=int(calendar_days))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "QuantsPlaybook因子直接从时点全A股截面选股；"
            "不使用本项目旧候选池和旧选股门槛"
        ),
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument(
        "--history-start-date",
        default="",
        help="default and hard maximum: start-date minus 700 calendar days",
    )
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--minimum-history", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--include-market-residual-momentum",
        action="store_true",
        help=(
            "research-only: calculate, evaluate, and emit the four fixed "
            "market residual momentum factors"
        ),
    )
    parser.add_argument(
        "--include-low-market-residual-volatility",
        action="store_true",
        help=(
            "research-only: calculate, evaluate, and emit the fixed "
            "price-only 20-day low market-residual-volatility factor"
        ),
    )
    parser.add_argument(
        "--skip-candidate-generation",
        action="store_true",
        help=(
            "research pipeline optimization: write factor metrics and the "
            "optional factor-panel cache without emitting single-factor "
            "candidate snapshots"
        ),
    )
    parser.add_argument(
        "--max-left-positions", type=int, default=2,
        help="maximum simultaneous left-side symbols inside the five-symbol cap",
    )
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--price-directory", default=str(DEFAULT_PRICE_DIRECTORY))
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-portfolio", action="store_true",
        help="run the existing entry/exit/position engine for every prepared factor",
    )
    parser.add_argument(
        "--factor-only-breadth-reversal-only", action="store_true",
        help="纯因子组合只回测低广度连阳一半拉回通道",
    )
    parser.add_argument(
        "--breadth-reversal-max-unique-count", type=int, default=50,
        help="信号日Formula33最近21日去重数必须严格低于该值",
    )
    parser.add_argument(
        "--breadth-reversal-risk-budget", type=float, default=0.006,
        help="每笔低广度反转交易的账户计划风险比例",
    )
    parser.add_argument(
        "--breadth-reversal-max-position", type=float, default=0.20,
        help="低广度反转单股最大账户仓位比例",
    )
    parser.add_argument(
        "--breadth-reversal-time-limit-days", type=int, default=20,
        help="低广度反转批次的时间退出交易日数",
    )
    parser.add_argument(
        "--factor-only-compressed-structure-only", action="store_true",
        help="纯因子组合只回测紧结构平台延续通道",
    )
    parser.add_argument(
        "--compressed-structure-risk-budget", type=float, default=0.015,
        help="每笔紧结构平台延续交易的账户计划风险比例",
    )
    parser.add_argument(
        "--compressed-structure-stop-loss-pct", type=float, default=0.08,
        help="紧结构平台延续交易的固定保护止损距离",
    )
    parser.add_argument(
        "--compressed-structure-max-position", type=float, default=0.20,
        help="紧结构平台延续交易的单股最大账户仓位比例",
    )
    parser.add_argument(
        "--compressed-structure-holding-days", type=int, default=20,
        help="紧结构平台延续交易的固定持有交易日数",
    )
    parser.add_argument(
        "--compressed-structure-min-entry-evidence-score",
        type=float,
        default=9.0,
        help="紧结构平台延续信号的最低技术证据分",
    )
    parser.add_argument(
        "--compressed-structure-min-proximity-score",
        type=float,
        default=40.0,
        help="候选结构接近分必须严格高于该值",
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="reuse existing manifest-verified factor candidates and skip factor recomputation",
    )
    parser.add_argument(
        "--portfolio-factors", default="",
        help="comma-separated subset; default runs every executable factor",
    )
    parser.add_argument(
        "--fit-combinations",
        action="store_true",
        help="fit declared combinations on train/validation only and freeze before OOS",
    )
    parser.add_argument("--train-start-date", default="2021-01-01")
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--validation-start-date", default="2024-01-01")
    parser.add_argument("--validation-end-date", default="2024-12-31")
    parser.add_argument("--positive-ic-top-n", type=int, default=5)
    parser.add_argument(
        "--factor-panel-cache",
        default="",
        help="optional pickle path for the PIT eligible factor panel and research labels",
    )
    parser.add_argument(
        "--prepared-price-cache",
        default="",
        help="optional directory of per-code prepared execution frames",
    )
    parser.add_argument(
        "--portfolio-metrics-file",
        default="portfolio_metrics.csv",
        help="relative or absolute metrics path for this portfolio invocation",
    )
    parser.add_argument(
        "--skip-portfolio-aggregate",
        action="store_true",
        help="leave root-wide aggregation to a coordinating process",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_candidate_generation and args.run_portfolio:
        raise ValueError(
            "--skip-candidate-generation cannot be combined with --run-portfolio"
        )
    history_calendar_days = (
        LOW_RESIDUAL_VOLATILITY_HISTORY_CALENDAR_DAYS
        if (
            args.include_low_market_residual_volatility
            and not args.include_market_residual_momentum
        )
        else REQUIRED_HISTORY_CALENDAR_DAYS
    )
    required_history_start = required_history_start_date(
        args.start_date,
        calendar_days=history_calendar_days,
    )
    if args.history_start_date:
        supplied_history_start = pd.Timestamp(
            args.history_start_date,
        ).normalize()
        if supplied_history_start > required_history_start:
            raise RuntimeError(
                "factor and execution history is incomplete: "
                f"history_start_date={supplied_history_start:%Y-%m-%d} "
                f"must be on or before {required_history_start:%Y-%m-%d} "
                f"({history_calendar_days} calendar-day warm-up)"
            )
    else:
        args.history_start_date = required_history_start.strftime("%Y-%m-%d")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    inventory = strategy_inventory_frame()
    inventory.to_csv(
        output / "strategy_inventory.csv", index=False, encoding="utf-8-sig",
    )

    repository = TushareRepository(Database(args.database_path))
    stock_basic = repository.load_stock_basic_universe()
    prepared_price_frames = None
    reusable_factors = None
    if args.reuse_candidates:
        candidate_root = output / "candidates"
        reusable_factors = sorted(
            path.name for path in candidate_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ) if candidate_root.is_dir() else []
        if not reusable_factors:
            raise RuntimeError(
                f"no reusable factor candidate manifests found: {candidate_root}",
            )
        requested_for_load = [
            value.strip() for value in args.portfolio_factors.split(",")
            if value.strip()
        ] or reusable_factors
        unknown_for_load = sorted(
            set(requested_for_load) - set(reusable_factors),
        )
        if unknown_for_load:
            raise ValueError(
                f"unprepared factor streams: {unknown_for_load}",
            )
        requested_codes = set()
        for factor in requested_for_load:
            snapshots = portfolio_backtest.load_candidate_snapshots(
                candidate_root / factor,
                args.start_date,
                args.end_date,
            )
            requested_codes.update(
                str(row["code"])
                for rows in snapshots.values()
                for row in rows
            )
        if args.prepared_price_cache:
            prepared_price_frames = load_prepared_price_cache(
                args.prepared_price_cache,
                requested_codes,
                start_date=args.history_start_date,
                end_date=args.end_date,
            )
            prices = pd.DataFrame()
        else:
            prices = repository.load_daily_kline_frames(
                requested_codes,
                start_date=args.history_start_date,
                end_date=args.end_date,
            )
    else:
        prices = repository.load_market_daily_frame(
            start_date=args.history_start_date,
            end_date=args.end_date,
        )
    if args.reuse_candidates:
        factors = reusable_factors
        metrics_path = output / "factor_metrics.csv"
        if metrics_path.is_file():
            print(pd.read_csv(metrics_path).to_string(index=False), flush=True)
    else:
        residual_factors, residual_metadata = (
            _market_residual_factor_configuration(
                args.include_market_residual_momentum,
                args.include_low_market_residual_volatility,
            )
        )
        stock_st_start_date = (
            args.history_start_date
            if (
                args.include_market_residual_momentum
                or args.include_low_market_residual_volatility
            )
            else args.start_date
        )
        stock_st = repository.load_dataset_for_codes(
            "stock_st",
            stock_basic["code"].dropna().astype(str),
            start_date=stock_st_start_date,
            end_date=args.end_date,
        )
        membership = point_in_time_tradable_universe(
            prices,
            stock_basic,
            start_date=args.start_date,
            end_date=args.end_date,
            minimum_history=args.minimum_history,
            stock_st=stock_st,
        )
        factor_membership = None
        if (
            args.include_market_residual_momentum
            or args.include_low_market_residual_volatility
        ):
            factor_membership = point_in_time_tradable_universe(
                prices,
                stock_basic,
                start_date=args.history_start_date,
                end_date=args.end_date,
                minimum_history=args.minimum_history,
                stock_st=stock_st,
            )
        snapshot_dates = _formula_snapshot_dates(
            args.formula_history,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        missing_dates = sorted(
            set(snapshot_dates) - set(membership["date"].drop_duplicates()),
        )
        if missing_dates:
            raise RuntimeError(
                "full-market membership misses formula trade dates: "
                f"count={len(missing_dates)} sample={missing_dates[:10]}",
            )

        coverage = {
            "database_path": str(Path(args.database_path)),
            "history_start_date": str(args.history_start_date),
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "price_rows": int(len(prices)),
            "price_codes": int(prices["code"].nunique()),
            "price_first_date": prices["date"].min().strftime("%Y-%m-%d"),
            "price_last_date": prices["date"].max().strftime("%Y-%m-%d"),
            "eligible_rows": int(len(membership)),
            "eligible_codes": int(membership["code"].nunique()),
            "snapshot_dates": len(snapshot_dates),
            "old_candidate_pool_used": False,
            "risk_warning_source": "local_tushare_stock_st",
            "risk_warning_first_date": (
                None if stock_st.empty
                else stock_st["date"].min().strftime("%Y-%m-%d")
            ),
            "risk_warning_last_date": (
                None if stock_st.empty
                else stock_st["date"].max().strftime("%Y-%m-%d")
            ),
            "risk_warning_rows": int(len(stock_st)),
        }
        if factor_membership is not None:
            coverage.update({
                "factor_warmup_eligible_rows": int(len(factor_membership)),
                "factor_warmup_eligible_codes": int(
                    factor_membership["code"].nunique()
                ),
            })
        (output / "data_coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        factor_prices = (
            _drop_ineligible_invalid_adjustment_rows(
                prices,
                factor_membership,
            )
            if factor_membership is not None
            else prices
        )
        factor_panel = calculate_daily_factor_panel(
            factor_prices,
            tradable_membership=factor_membership,
            include_market_residual_momentum=(
                args.include_market_residual_momentum
            ),
            include_low_market_residual_volatility=(
                args.include_low_market_residual_volatility
            ),
        )
        del factor_prices
        metrics = evaluate_factor_panel(
            factor_panel,
            prices,
            membership,
            holding_days=args.holding_days,
            start_date=args.start_date,
            end_date=args.end_date,
            factor_columns=[
                *executable_factor_columns(),
                *residual_factors,
            ],
        )
        metrics.to_csv(
            output / "factor_metrics.csv", index=False, encoding="utf-8-sig",
        )
        if args.factor_panel_cache:
            panel_cache = build_labeled_factor_sample(
                factor_panel,
                prices,
                membership,
                holding_days=args.holding_days,
                start_date=args.start_date,
                end_date=args.end_date,
            ).reset_index()
            panel_cache["code"] = panel_cache["code"].astype("category")
            cache_path = Path(args.factor_panel_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            panel_cache.to_pickle(cache_path)
            del panel_cache
            gc.collect()
        factor_metadata_by_name = dict(residual_metadata)
        if args.fit_combinations:
            if pd.Timestamp(args.start_date) > pd.Timestamp(args.train_start_date):
                raise RuntimeError(
                    "combination fitting requires start_date no later than "
                    f"train_start_date={args.train_start_date}",
                )
            if pd.Timestamp(args.end_date) < pd.Timestamp(args.validation_end_date):
                raise RuntimeError(
                    "combination fitting requires end_date no earlier than "
                    f"validation_end_date={args.validation_end_date}",
                )
            factor_panel, combination_manifest = fit_walk_forward_combinations(
                factor_panel,
                prices,
                membership,
                holding_days=args.holding_days,
                train_start_date=args.train_start_date,
                train_end_date=args.train_end_date,
                validation_start_date=args.validation_start_date,
                validation_end_date=args.validation_end_date,
                positive_ic_top_n=args.positive_ic_top_n,
            )
            (output / "combination_manifest.json").write_text(
                json.dumps(
                    combination_manifest,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            factor_metadata_by_name.update(
                combination_manifest["combinations"]
            )
        factors = [
            factor for factor in executable_factor_columns()
            if factor in factor_panel and factor_panel[factor].notna().any()
        ]
        factors.extend(
            factor for factor in residual_factors
            if factor in factor_panel
            and factor_panel[factor].notna().any()
            and factor not in factors
        )
        factors.extend(
            factor for factor in factor_metadata_by_name
            if factor not in factors
        )
        if not args.skip_candidate_generation:
            candidate_index = prepare_factor_candidates(
                factor_panel,
                membership,
                factors=factors,
                snapshot_dates=snapshot_dates,
                output_root=output,
                top_n=args.top_n,
                start_date=args.start_date,
                end_date=args.end_date,
                factor_metadata_by_name=factor_metadata_by_name,
                candidate_enrichment_prices=prices,
                candidate_outcome_holding_days=args.holding_days,
                candidate_enrichment_amount_multiplier=1000.0,
            )
            candidate_index.to_csv(
                output / "candidate_streams.csv",
                index=False,
                encoding="utf-8-sig",
            )
        print(metrics.to_string(index=False), flush=True)

    if args.run_portfolio:
        requested = [
            value.strip() for value in args.portfolio_factors.split(",")
            if value.strip()
        ]
        portfolio_factors = requested or factors
        unknown = sorted(set(portfolio_factors) - set(factors))
        if unknown:
            raise ValueError(f"unprepared factor streams: {unknown}")
        if not args.reuse_candidates:
            del factor_panel, membership
        gc.collect()
        portfolio_metrics = run_portfolio_comparison(
            portfolio_factors,
            output_root=output,
            prices=prices,
            stock_basic=stock_basic,
            repository=repository,
            formula_history=args.formula_history,
            start_date=args.start_date,
            end_date=args.end_date,
            prepared_price_frames=prepared_price_frames,
            max_left_positions=args.max_left_positions,
            factor_only_breadth_reversal_only=(
                args.factor_only_breadth_reversal_only
            ),
            breadth_reversal_max_unique_count=(
                args.breadth_reversal_max_unique_count
            ),
            breadth_reversal_risk_budget=args.breadth_reversal_risk_budget,
            breadth_reversal_max_position=args.breadth_reversal_max_position,
            breadth_reversal_time_limit_days=(
                args.breadth_reversal_time_limit_days
            ),
            factor_only_compressed_structure_only=(
                args.factor_only_compressed_structure_only
            ),
            compressed_structure_risk_budget=(
                args.compressed_structure_risk_budget
            ),
            compressed_structure_stop_loss_pct=(
                args.compressed_structure_stop_loss_pct
            ),
            compressed_structure_max_position=(
                args.compressed_structure_max_position
            ),
            compressed_structure_holding_days=(
                args.compressed_structure_holding_days
            ),
            compressed_structure_min_entry_evidence_score=(
                args.compressed_structure_min_entry_evidence_score
            ),
            compressed_structure_min_proximity_score=(
                args.compressed_structure_min_proximity_score
            ),
        )
        metrics_path = Path(args.portfolio_metrics_file)
        if not metrics_path.is_absolute():
            metrics_path = output / metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio_metrics.to_csv(
            metrics_path, index=False, encoding="utf-8-sig",
        )
        if not args.skip_portfolio_aggregate:
            all_metrics = aggregate_portfolio_summaries(output)
            all_metrics.to_csv(
                output / "portfolio_metrics_all_periods.csv",
                index=False,
                encoding="utf-8-sig",
            )
            (output / "portfolio_comparison.md").write_text(
                render_portfolio_comparison_markdown(all_metrics),
                encoding="utf-8",
            )
        print(portfolio_metrics.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
