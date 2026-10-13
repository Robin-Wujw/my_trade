"""Evaluate frozen factors inside a PIT high-liquidity stock universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_chunked import _load_cached_panel, _year_periods
from stock_research.research.quantsplaybook_compare import (
    _formula_snapshot_dates,
    _spearman_rank_corr,
    required_history_start_date,
)
from stock_research.research.quantsplaybook_liquidity_universe import _rolling_average_amount
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_SOURCE_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_liquidity_factor_research_2021_2023"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_FACTORS = (
    "high_quality_momentum",
    "shadow_reversal",
    "coin_team",
    "salience_str",
    "low_idiosyncratic_volatility",
    "buying_pressure",
    "ma_convergence",
    "amplitude_structure",
    "disposition_reversal",
    "chip_loss_overhang",
    "network_scc",
    "network_tcc",
    "network_cc",
    "playbook_ensemble",
)


def select_liquidity_factor_top_n(
    panel: pd.DataFrame,
    average_amount_yuan: pd.Series,
    *,
    factors,
    snapshot_dates,
    minimum_average_amount_yuan: float,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select by frozen score only; forward returns remain research labels."""
    amount = pd.to_numeric(
        average_amount_yuan.reindex(panel.index), errors="coerce",
    )
    eligible = amount.ge(float(minimum_average_amount_yuan))
    labels = pd.to_numeric(panel.get("forward_return"), errors="coerce")
    selected_rows = []
    daily_rows = []
    for factor in factors:
        if factor not in panel:
            raise RuntimeError(f"factor is unavailable: {factor}")
        scores = pd.to_numeric(panel[factor], errors="coerce")
        for value in snapshot_dates:
            date = pd.Timestamp(value).normalize()
            try:
                date_scores = scores.xs(date, level="date")
                date_eligible = eligible.xs(date, level="date")
                date_labels = labels.xs(date, level="date")
            except KeyError:
                continue
            available_scores = date_scores[date_eligible].dropna()
            ranked = available_scores.sort_values(
                ascending=False, kind="mergesort",
            ).head(int(top_n))
            valid = pd.concat(
                [available_scores.rename("score"), date_labels.rename("label")],
                axis=1,
            ).dropna()
            rank_ic = _spearman_rank_corr(
                valid["score"],
                valid["label"],
            )
            selected_labels = date_labels.reindex(ranked.index)
            daily_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "factor": factor,
                "eligible_count": int(len(available_scores)),
                "selected_count": int(len(ranked)),
                "selected_label_count": int(selected_labels.notna().sum()),
                "selected_mean_forward_return": (
                    float(selected_labels.mean())
                    if selected_labels.notna().any() else None
                ),
                "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
            })
            for code, score in ranked.items():
                label = date_labels.get(code)
                selected_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "code": str(code),
                    "factor": factor,
                    "factor_score": float(score),
                    "forward_return": (
                        None if pd.isna(label) else float(label)
                    ),
                })
    return pd.DataFrame(selected_rows), pd.DataFrame(daily_rows)


def summarize_factor_research(
    selected: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    periods = [
        (str(year), str(year), str(year))
        for year in sorted(pd.to_datetime(selected["date"]).dt.year.unique())
    ]
    periods.append(("all", "0000", "9999"))
    for factor in sorted(selected["factor"].unique()):
        factor_selected = selected[selected["factor"].eq(factor)].copy()
        factor_daily = daily[daily["factor"].eq(factor)].copy()
        for period, start_year, end_year in periods:
            selected_year = pd.to_datetime(factor_selected["date"]).dt.year.astype(str)
            daily_year = pd.to_datetime(factor_daily["date"]).dt.year.astype(str)
            chosen = factor_selected[
                selected_year.between(start_year, end_year)
            ]
            chosen_daily = factor_daily[
                daily_year.between(start_year, end_year)
            ]
            returns = pd.to_numeric(
                chosen["forward_return"], errors="coerce",
            ).dropna()
            rank_ic = pd.to_numeric(
                chosen_daily["rank_ic"], errors="coerce",
            ).dropna()
            rows.append({
                "factor": factor,
                "period": period,
                "snapshot_count": int(chosen_daily["date"].nunique()),
                "mean_eligible_symbols": float(
                    chosen_daily["eligible_count"].mean()
                ),
                "minimum_eligible_symbols": int(
                    chosen_daily["eligible_count"].min()
                ),
                "selected_label_count": int(len(returns)),
                "mean_forward_return": float(returns.mean()),
                "median_forward_return": float(returns.median()),
                "win_rate": float(returns.gt(0).mean()),
                "loss_below_minus_5pct_rate": float(returns.lt(-0.05).mean()),
                "gain_at_least_10pct_rate": float(returns.ge(0.10).mean()),
                "mean_rank_ic": float(rank_ic.mean()) if len(rank_ic) else None,
                "positive_rank_ic_day_rate": (
                    float(rank_ic.gt(0).mean()) if len(rank_ic) else None
                ),
            })
    return pd.DataFrame(rows)


def run_liquidity_factor_research(
    *,
    source_root: str | Path,
    output_root: str | Path,
    formula_history: str | Path,
    database_path: str | Path,
    start_date: str,
    end_date: str,
    factors,
    minimum_average_amount_yuan: float,
    top_n: int,
) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repository = TushareRepository(Database(Path(database_path)))
    selected_frames = []
    daily_frames = []
    for year, annual_start, annual_end in _year_periods(start_date, end_date):
        print(
            f"[liquidity-factor-research] year={year} "
            f"period={annual_start}..{annual_end}",
            flush=True,
        )
        panel = _load_cached_panel(
            source_root / "research_panels" / f"factor_panel_{year}.pkl",
        )
        codes = panel.index.get_level_values("code").unique().astype(str)
        prices = repository.load_daily_kline_frames(
            codes,
            start_date=required_history_start_date(annual_start).strftime(
                "%Y-%m-%d",
            ),
            end_date=annual_end,
        )
        average_amount = _rolling_average_amount(prices)
        selected, daily = select_liquidity_factor_top_n(
            panel,
            average_amount,
            factors=factors,
            snapshot_dates=_formula_snapshot_dates(
                formula_history,
                start_date=annual_start,
                end_date=annual_end,
            ),
            minimum_average_amount_yuan=minimum_average_amount_yuan,
            top_n=top_n,
        )
        selected_frames.append(selected)
        daily_frames.append(daily)

    selected = pd.concat(selected_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    metrics = summarize_factor_research(selected, daily)
    metrics.to_csv(
        output_root / "factor_metrics.csv", index=False, encoding="utf-8-sig",
    )
    daily.to_csv(
        output_root / "factor_metrics_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "version": 1,
        "start_date": start_date,
        "end_date": end_date,
        "minimum_average_amount_yuan": float(minimum_average_amount_yuan),
        "top_n": int(top_n),
        "factors": list(factors),
        "selection_fields": ["factor_score", "average_amount_20_yuan"],
        "future_return_used_for_candidate_selection": False,
        "label_role": "research_label_only_not_candidate_input",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "factor_count": len(factors),
        "selected_row_count": int(len(selected)),
        "daily_row_count": int(len(daily)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--minimum-average-amount-yuan", type=float, default=5e8)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    factors = tuple(
        value.strip() for value in str(args.factors).split(",") if value.strip()
    )
    result = run_liquidity_factor_research(
        source_root=args.source_root,
        output_root=args.output_root,
        formula_history=args.formula_history,
        database_path=args.database_path,
        start_date=args.start_date,
        end_date=args.end_date,
        factors=factors,
        minimum_average_amount_yuan=args.minimum_average_amount_yuan,
        top_n=args.top_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
