"""Generate frozen low-correlation candidates after a PIT liquidity prefilter."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_chunked import (
    _load_cached_panel,
    _merge_candidate_chunks,
    _year_periods,
)
from stock_research.research.quantsplaybook_compare import (
    _formula_snapshot_dates,
    prepare_factor_candidates,
    rank_weighted_factor_combination,
    required_history_start_date,
)
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


SOURCE_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_liquidity_universe_2021_to_20260721"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_THRESHOLDS = (200_000_000.0, 500_000_000.0, 1_000_000_000.0)
NAMED_FROZEN_COMBINATIONS = {
    "playbook_liquid_balanced": {
        "disposition_reversal": 0.25,
        "network_scc": 0.25,
        "chip_loss_overhang": 0.25,
        "high_quality_momentum": 0.25,
    },
}


def liquidity_factor_name(
    threshold: float,
    source_factor: str = "playbook_low_corr",
) -> str:
    value = float(threshold)
    if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
        suffix = f"{int(value / 1_000_000_000)}b"
    else:
        suffix = f"{int(value / 1_000_000)}m"
    return f"{source_factor}_liquidity_{suffix}"


def apply_liquidity_universe(
    panel: pd.DataFrame,
    source_score: pd.Series,
    average_amount_yuan: pd.Series,
    thresholds,
    *,
    source_factor: str = "playbook_low_corr",
) -> tuple[pd.DataFrame, list[dict]]:
    result = panel.copy()
    average_amount = pd.to_numeric(
        average_amount_yuan.reindex(result.index), errors="coerce",
    )
    diagnostics = []
    for threshold in thresholds:
        factor = liquidity_factor_name(
            float(threshold), source_factor=source_factor,
        )
        eligible = average_amount.ge(float(threshold))
        result[factor] = source_score.reindex(result.index).where(eligible)
        counts = eligible.groupby(level="date", sort=False).sum()
        diagnostics.append({
            "factor": factor,
            "minimum_average_amount_yuan": float(threshold),
            "minimum_eligible_symbols": int(counts.min()),
            "mean_eligible_symbols": round(float(counts.mean()), 3),
            "maximum_eligible_symbols": int(counts.max()),
        })
    return result, diagnostics


def _rolling_average_amount(prices: pd.DataFrame) -> pd.Series:
    frame = prices[["date", "code", "amount"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = frame["code"].astype(str)
    frame["amount_yuan"] = pd.to_numeric(
        frame["amount"], errors="coerce",
    ).mul(1000.0)
    frame["amount_yuan"] = frame["amount_yuan"].where(
        frame["amount_yuan"] > 0,
    )
    frame = frame.dropna(subset=["date", "code"]).sort_values(
        ["code", "date"], kind="mergesort",
    )
    frame["average_amount_20_yuan"] = frame.groupby(
        "code", sort=False,
    )["amount_yuan"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    return frame.set_index(["date", "code"])["average_amount_20_yuan"]


def frozen_source_score(
    panel: pd.DataFrame,
    source_factor: str,
    low_corr_weights: dict[str, float],
    combination_specs: dict[str, dict] | None = None,
) -> pd.Series:
    combination_specs = combination_specs or {}
    combination_spec = combination_specs.get(source_factor, {})
    combination_weights = combination_spec.get("weights")
    if isinstance(combination_weights, dict) and combination_weights:
        return rank_weighted_factor_combination(
            panel,
            {
                str(key): float(value)
                for key, value in combination_weights.items()
            },
        )
    if source_factor == "playbook_low_corr":
        return rank_weighted_factor_combination(panel, low_corr_weights)
    if source_factor in NAMED_FROZEN_COMBINATIONS:
        return rank_weighted_factor_combination(
            panel, NAMED_FROZEN_COMBINATIONS[source_factor],
        )
    if source_factor in panel:
        return pd.to_numeric(panel[source_factor], errors="coerce")
    raise RuntimeError(f"source factor is unavailable: {source_factor}")


def generate_liquidity_universe_candidates(
    *,
    source_root: str | Path,
    output_root: str | Path,
    formula_history: str | Path,
    database_path: str | Path,
    start_date: str,
    end_date: str,
    holding_days: int,
    top_n: int,
    thresholds,
    source_factors=("playbook_low_corr",),
    enrich_candidate_profiles: bool = True,
) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    combination_path = source_root / "combination_manifest.json"
    combination = json.loads(combination_path.read_text(encoding="utf-8"))
    combination_specs = combination["combinations"]
    low_corr_spec = combination_specs["playbook_low_corr"]
    weights = {
        str(key): float(value)
        for key, value in low_corr_spec["weights"].items()
    }
    source_factors = tuple(str(value) for value in source_factors)
    factor_pairs = [
        (
            liquidity_factor_name(threshold, source_factor=source_factor),
            source_factor,
            float(threshold),
        )
        for source_factor in source_factors
        for threshold in thresholds
    ]
    factors = [factor for factor, _, _ in factor_pairs]
    specs = {}
    source_weights_by_factor = {}
    for factor, source_factor, threshold in factor_pairs:
        source_spec = combination_specs.get(source_factor, {})
        source_weights = source_spec.get("weights")
        if not isinstance(source_weights, dict) or not source_weights:
            source_weights = NAMED_FROZEN_COMBINATIONS.get(source_factor)
        if source_weights:
            source_weights = {
                str(key): float(value)
                for key, value in source_weights.items()
            }
            source_weights_by_factor[source_factor] = source_weights
        spec = {
            "method": "PIT_liquidity_prefilter_then_frozen_factor_rank",
            "minimum_average_amount_yuan": threshold,
            "liquidity_window_trading_days": 20,
            "liquidity_minimum_observations": 20,
            "source_factor": source_factor,
            "source_combination_manifest": str(combination_path),
            "future_return_used_for_candidate_selection": False,
            "uses_financial_data": bool(
                source_spec.get("uses_financial_data") is True
            ),
            "financial_point_in_time": (
                source_spec.get("financial_point_in_time") is not False
            ),
            "strict_financial_point_in_time": (
                source_spec.get("strict_financial_point_in_time") is not False
            ),
            "research_only": bool(source_spec.get("research_only") is True),
        }
        if source_weights:
            spec["weights"] = source_weights
        specs[factor] = spec
    periods = _year_periods(start_date, end_date)
    repository = TushareRepository(Database(Path(database_path)))
    coverage_rows = []
    for year, annual_start, annual_end in periods:
        print(
            f"[liquidity-universe] year={year} period={annual_start}..{annual_end}",
            flush=True,
        )
        panel = _load_cached_panel(
            source_root / "research_panels" / f"factor_panel_{year}.pkl",
        )
        source_scores = {}
        for source_factor in source_factors:
            source_scores[source_factor] = frozen_source_score(
                panel,
                source_factor,
                weights,
                combination_specs=combination_specs,
            )
        codes = panel.index.get_level_values("code").unique().astype(str)
        history_start = required_history_start_date(annual_start).strftime(
            "%Y-%m-%d",
        )
        prices = repository.load_daily_kline_frames(
            codes,
            start_date=history_start,
            end_date=annual_end,
        )
        average_amount = _rolling_average_amount(prices)
        for source_factor, source_score in source_scores.items():
            panel, annual_coverage = apply_liquidity_universe(
                panel,
                source_score,
                average_amount,
                thresholds,
                source_factor=source_factor,
            )
            for row in annual_coverage:
                row["year"] = int(year)
            coverage_rows.extend(annual_coverage)
        membership = panel.index.to_frame(index=False)
        snapshot_dates = _formula_snapshot_dates(
            formula_history,
            start_date=annual_start,
            end_date=annual_end,
        )
        prepare_factor_candidates(
            panel,
            membership,
            factors=factors,
            snapshot_dates=snapshot_dates,
            output_root=output_root / "chunks" / str(year),
            top_n=top_n,
            start_date=annual_start,
            end_date=annual_end,
            factor_metadata_by_name=specs,
            candidate_enrichment_prices=(
                prices if enrich_candidate_profiles else None
            ),
            candidate_outcome_holding_days=holding_days,
            candidate_enrichment_amount_multiplier=1000.0,
        )
        del panel, prices, average_amount, membership, source_scores
        gc.collect()

    merge_manifest = {
        "annual_label_boundary": (
            "labels crossing a calendar-year boundary are excluded"
        ),
        "combinations": specs,
    }
    candidate_index = _merge_candidate_chunks(
        periods=periods,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        combination_manifest=merge_manifest,
    )
    candidate_index.to_csv(
        output_root / "candidate_streams.csv", index=False, encoding="utf-8-sig",
    )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        output_root / "liquidity_universe_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "version": 1,
        "start_date": start_date,
        "end_date": end_date,
        "holding_days": int(holding_days),
        "top_n": int(top_n),
        "thresholds_yuan": [float(value) for value in thresholds],
        "source_combination_manifest": str(combination_path),
        "source_weights": weights,
        "source_weights_by_factor": source_weights_by_factor,
        "source_factors": list(source_factors),
        "named_frozen_combinations": {
            factor: NAMED_FROZEN_COMBINATIONS[factor]
            for factor in source_factors
            if factor in NAMED_FROZEN_COMBINATIONS
        },
        "future_return_used_for_candidate_selection": False,
        "candidate_factors": factors,
    }
    (output_root / "liquidity_universe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "candidate_factors": factors,
        "coverage": coverage_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--liquidity-thresholds",
        default=",".join(str(int(value)) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--source-factors",
        default="playbook_low_corr",
        help="comma-separated frozen factor columns; playbook_low_corr uses its frozen weights",
    )
    parser.add_argument(
        "--skip-candidate-profile-enrichment",
        action="store_true",
        help=(
            "skip candidate trading-profile fields; liquidity filtering still "
            "uses the full point-in-time amount history"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = tuple(
        float(value.strip())
        for value in str(args.liquidity_thresholds).split(",")
        if value.strip()
    )
    source_factors = tuple(
        value.strip()
        for value in str(args.source_factors).split(",")
        if value.strip()
    )
    result = generate_liquidity_universe_candidates(
        source_root=args.source_root,
        output_root=args.output_root,
        formula_history=args.formula_history,
        database_path=args.database_path,
        start_date=args.start_date,
        end_date=args.end_date,
        holding_days=args.holding_days,
        top_n=args.top_n,
        thresholds=thresholds,
        source_factors=source_factors,
        enrich_candidate_profiles=(
            not args.skip_candidate_profile_enrichment
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
