"""Build the single pre-registered residual-momentum/low-volatility candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_chunked import (
    ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR,
    PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR,
    _apply_combination,
    _formula_snapshot_dates,
    _load_cached_panel,
    _merge_candidate_chunks,
    _one_network_price_only_residual_momentum_low_volatility_spec,
    _year_periods,
)
from stock_research.research.quantsplaybook_compare import prepare_factor_candidates
from stock_research.core.paths import PATHS


DEFAULT_RESIDUAL_PANEL_ROOT = (
    PATHS.runtime_root / "backtests" / "one_network_price_only_experiment"
    / "research_panels"
)
DEFAULT_LOW_VOLATILITY_PANEL_ROOT = (
    PATHS.runtime_root / "backtests"
    / "one_network_price_only_low_residual_volatility_experiment"
    / "research_panels"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "right_residual_momentum_252_skip20_lowvol_train"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_source_factor_panels(
    residual_panel: pd.DataFrame,
    low_volatility_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    spec = _one_network_price_only_residual_momentum_low_volatility_spec()
    if not isinstance(residual_panel.index, pd.MultiIndex):
        raise ValueError("residual momentum panel requires a MultiIndex")
    if not isinstance(low_volatility_panel.index, pd.MultiIndex):
        raise ValueError("low-volatility panel requires a MultiIndex")
    if list(residual_panel.index.names) != ["date", "code"]:
        raise ValueError("residual momentum panel index must be date/code")
    if list(low_volatility_panel.index.names) != ["date", "code"]:
        raise ValueError("low-volatility panel index must be date/code")
    if residual_panel.index.has_duplicates or low_volatility_panel.index.has_duplicates:
        raise ValueError("source factor panels contain duplicate date/code rows")
    if PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR not in residual_panel:
        raise ValueError("residual momentum source factor is missing")
    missing = sorted(
        set(spec["selected_factors"])
        - {PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR}
        - set(low_volatility_panel.columns)
    )
    if missing:
        raise ValueError(f"low-volatility source factors are missing: {missing}")

    panel = low_volatility_panel.copy()
    residual = pd.to_numeric(
        residual_panel[PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR],
        errors="coerce",
    ).reindex(panel.index)
    panel[PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR] = residual
    factor = ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR
    panel[factor] = _apply_combination(panel, spec)
    available = int(panel[factor].notna().sum())
    if available == 0:
        raise RuntimeError("fixed residual-momentum combination has no values")
    return panel, {
        "low_volatility_panel_rows": int(len(low_volatility_panel)),
        "residual_momentum_panel_rows": int(len(residual_panel)),
        "aligned_panel_rows": int(len(panel)),
        "residual_momentum_available_rows": int(residual.notna().sum()),
        "combination_available_rows": available,
        "alignment_rule": (
            "low-volatility panel owns the PIT date/code universe; residual "
            "momentum is exact-index reindexed with no fill"
        ),
    }


def build_candidates(
    *,
    residual_panel_root: str | Path,
    low_volatility_panel_root: str | Path,
    formula_history: str | Path,
    output_root: str | Path,
    start_date: str,
    end_date: str,
    top_n: int = 100,
) -> Path:
    residual_root = Path(residual_panel_root)
    low_volatility_root = Path(low_volatility_panel_root)
    formula_path = Path(formula_history)
    output = Path(output_root)
    periods = _year_periods(start_date, end_date)
    spec = _one_network_price_only_residual_momentum_low_volatility_spec()
    factor = ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR
    source_panels = []

    for year, period_start, period_end in periods:
        residual_path = residual_root / f"factor_panel_{year}.pkl"
        low_volatility_path = low_volatility_root / f"factor_panel_{year}.pkl"
        if not residual_path.is_file() or not low_volatility_path.is_file():
            raise RuntimeError(f"cached source factor panel is missing for {year}")
        residual_panel = _load_cached_panel(residual_path)
        low_volatility_panel = _load_cached_panel(low_volatility_path)
        panel, alignment = merge_source_factor_panels(
            residual_panel, low_volatility_panel,
        )
        snapshot_dates = _formula_snapshot_dates(
            formula_path,
            start_date=period_start,
            end_date=period_end,
        )
        prepare_factor_candidates(
            panel,
            panel.index.to_frame(index=False),
            factors=[factor],
            snapshot_dates=snapshot_dates,
            output_root=output / "chunks" / str(year),
            top_n=int(top_n),
            start_date=period_start,
            end_date=period_end,
            factor_metadata_by_name={factor: spec},
        )
        source_panels.append({
            "year": int(year),
            "residual_momentum_panel": str(residual_path),
            "residual_momentum_panel_sha256": _sha256(residual_path),
            "low_volatility_panel": str(low_volatility_path),
            "low_volatility_panel_sha256": _sha256(low_volatility_path),
            **alignment,
        })

    manifest = {
        "version": 1,
        "selection_rule": "single_pre_registered_fixed_combination",
        "future_return_used_for_candidate_selection": False,
        "full_history_previously_observed": True,
        "research_only": True,
        "start_date": start_date,
        "end_date": end_date,
        "top_n": int(top_n),
        "formula_history": str(formula_path),
        "formula_history_sha256": _sha256(formula_path),
        "combinations": {factor: spec},
        "source_panels": source_panels,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "combination_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    candidate_index = _merge_candidate_chunks(
        periods=periods,
        output_root=output,
        start_date=start_date,
        end_date=end_date,
        combination_manifest=manifest,
        factors_to_merge=[factor],
    )
    candidate_index.to_csv(
        output / "candidate_streams.csv", index=False, encoding="utf-8-sig",
    )
    return output / "candidates" / factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-panel-root", default=str(DEFAULT_RESIDUAL_PANEL_ROOT),
    )
    parser.add_argument(
        "--low-volatility-panel-root",
        default=str(DEFAULT_LOW_VOLATILITY_PANEL_ROOT),
    )
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-29")
    parser.add_argument("--top-n", type=int, default=100)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = build_candidates(
        residual_panel_root=args.residual_panel_root,
        low_volatility_panel_root=args.low_volatility_panel_root,
        formula_history=args.formula_history,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
