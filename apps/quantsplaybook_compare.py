"""Build factor-only A-share selections and compare them in one execution engine."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apps import portfolio_backtest
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.quantsplaybook_selection import (
    QUANTSPLAYBOOK_COMMIT,
    calculate_daily_factor_panel,
    executable_factor_columns,
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
BEIJING_STOCK_EXCHANGE_OPEN_DATE = pd.Timestamp("2021-11-15")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qfq_close_panel(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    inverse = pd.to_numeric(
        frame.get("raw_to_qfq_factor", 1.0), errors="coerce",
    ).where(lambda value: value > 0, 1.0)
    frame["qfq_close"] = close / inverse
    return frame.pivot(index="date", columns="code", values="qfq_close").sort_index()


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
                "ic": ordered[factor].corr(
                    ordered["forward_return"], method="spearman",
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


def factor_candidate_snapshots(
    factor_panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    factor_column: str,
    snapshot_dates: list[pd.Timestamp],
    top_n: int,
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
                "selected_for_trading": True,
                "signal_eligible": True,
                "allow_right": True,
                "allow_left": False,
                "candidate_failure_reason": "",
            })
        snapshots[date.strftime("%Y-%m-%d")] = rows
    return snapshots


def save_factor_snapshots(
    snapshots: dict[str, list[dict]],
    *,
    output_directory: str | Path,
    factor_column: str,
    top_n: int,
    start_date: str,
    end_date: str,
    factor_metadata: dict | None = None,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for date, rows in sorted(snapshots.items()):
        path = output / f"candidates_{date}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(rows),
            "signal_eligible_count": len(rows),
            "financial_point_in_time": True,
            "industry_point_in_time": False,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(manifest_rows),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "financial_point_in_time": True,
        "industry_point_in_time": False,
        "selection_engine": "quantsplaybook_factor_only",
        "old_candidate_pool_used": False,
        "source_commit": QUANTSPLAYBOOK_COMMIT,
        "factor": factor_column,
        "top_n": int(top_n),
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "universe_rule": (
            "listed_by_date;not_past_delist_date;valid_traded_bar;"
            "beijing_exchange_not_before_2021_11_15;minimum_history;"
            "no_old_selection_gate"
        ),
        "snapshots": manifest_rows,
    }
    if factor_metadata:
        manifest["factor_metadata"] = factor_metadata
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
    factor_metadata_by_name: dict[str, dict] | None = None,
) -> pd.DataFrame:
    rows = []
    for factor in factors:
        snapshots = factor_candidate_snapshots(
            factor_panel,
            membership,
            factor_column=factor,
            snapshot_dates=snapshot_dates,
            top_n=top_n,
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
            max_left_positions=1,
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


def required_history_start_date(start_date: str) -> pd.Timestamp:
    return (
        pd.Timestamp(start_date).normalize()
        - pd.Timedelta(days=REQUIRED_HISTORY_CALENDAR_DAYS)
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
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--price-directory", default=str(DEFAULT_PRICE_DIRECTORY))
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-portfolio", action="store_true",
        help="run the existing entry/exit/position engine for every prepared factor",
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
    required_history_start = required_history_start_date(args.start_date)
    if args.history_start_date:
        supplied_history_start = pd.Timestamp(
            args.history_start_date,
        ).normalize()
        if supplied_history_start > required_history_start:
            raise RuntimeError(
                "factor and execution history is incomplete: "
                f"history_start_date={supplied_history_start:%Y-%m-%d} "
                f"must be on or before {required_history_start:%Y-%m-%d} "
                f"({REQUIRED_HISTORY_CALENDAR_DAYS} calendar-day warm-up)"
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
        stock_st = repository.load_dataset_for_codes(
            "stock_st",
            stock_basic["code"].dropna().astype(str),
            start_date=args.start_date,
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
        (output / "data_coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        factor_panel = calculate_daily_factor_panel(prices)
        metrics = evaluate_factor_panel(
            factor_panel,
            prices,
            membership,
            holding_days=args.holding_days,
            start_date=args.start_date,
            end_date=args.end_date,
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
        factor_metadata_by_name = {}
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
            factor_metadata_by_name = (
                combination_manifest["combinations"]
            )
        factors = [
            factor for factor in executable_factor_columns()
            if factor in factor_panel and factor_panel[factor].notna().any()
        ]
        factors.extend(
            factor for factor in factor_metadata_by_name
            if factor not in factors
        )
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
        )
        candidate_index.to_csv(
            output / "candidate_streams.csv", index=False, encoding="utf-8-sig",
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
