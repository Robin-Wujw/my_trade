"""Build the pre-registered China-size/low-volatility research candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_chunked import (
    _apply_combination,
    _formula_snapshot_dates,
    _load_cached_panel,
    _merge_candidate_chunks,
    _one_network_price_only_low_volatility_spec,
    _year_periods,
)
from stock_research.research.quantsplaybook_compare import prepare_factor_candidates
from stock_research.research.quantsplaybook_liquidity_universe import (
    _rolling_average_amount,
    apply_liquidity_universe,
    liquidity_factor_name,
)
from stock_research.core.paths import PATHS
from stock_research.storage import Database
from stock_research.storage.tushare_repository import tushare_to_project_code


SIZE_FACTOR = "china_size_ex_shell_smallest_30pct"
SOURCE_FACTOR = (
    "playbook_low_corr_network_cap1_price_only_residual_volatility_"
    "china_size_ex_shell30"
)
FINAL_FACTOR = liquidity_factor_name(30_000_000.0, source_factor=SOURCE_FACTOR)
SIZE_EXCLUSION_PERCENTILE = 0.30
LIQUIDITY_THRESHOLD_YUAN = 30_000_000.0
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
    PATHS.runtime_root / "backtests" / "right_china_size_ex_shell30_train"
    / "factor"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def china_size_factor_metadata() -> dict:
    return {
        "factor_family": "china_size_ex_shell_value",
        "formula": (
            "within each observation-date market-cap cross-section, remove "
            "total_mv percentile <= 30%; score remaining stocks by -ln(total_mv)"
        ),
        "market_cap_field": "Tushare daily_basic.total_mv",
        "market_cap_timing": "same observation date",
        "smallest_market_cap_exclusion_percentile": SIZE_EXCLUSION_PERCENTILE,
        "invalid_market_cap_rule": "non-positive or missing total_mv fails closed",
        "financial_statement_data_used": False,
        "industry_data_used": False,
        "literature": {
            "title": "Size and Value in China",
            "authors": "Liu, Stambaugh, and Yuan",
            "doi": "10.1016/j.jfineco.2019.03.008",
            "nber": "10.3386/w24458",
        },
    }


def fixed_size_combination_spec() -> dict:
    base = _one_network_price_only_low_volatility_spec()
    selected = [*base["selected_factors"], SIZE_FACTOR]
    weight = 1.0 / len(selected)
    return {
        "method": (
            "fixed_equal_weight_seven_factor_low_residual_volatility_"
            "plus_china_size_ex_shell30"
        ),
        "weights": {factor: weight for factor in selected},
        "selected_factors": selected,
        "require_complete_factors": True,
        "missing_factor_rule": (
            "all seven selected factors must be available before ranking"
        ),
        "future_return_used_for_candidate_selection": False,
        "uses_financial_data": False,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "research_only": True,
        "research_only_reason": (
            "the full historical period was observed before this fixed "
            "literature-motivated factor was evaluated"
        ),
        "china_size_factor": china_size_factor_metadata(),
        "base_factor_spec": base,
    }


def calculate_china_size_factor(market_cap: pd.Series) -> pd.Series:
    if not isinstance(market_cap.index, pd.MultiIndex):
        raise ValueError("market cap series requires a MultiIndex")
    if list(market_cap.index.names) != ["date", "code"]:
        raise ValueError("market cap series index must be date/code")
    values = pd.to_numeric(market_cap, errors="coerce")
    values = values.where(values > 0)
    percentile = values.groupby(level="date", sort=False).rank(
        method="average", pct=True,
    )
    return -np.log(values).where(percentile > SIZE_EXCLUSION_PERCENTILE)


def add_size_combination(
    low_volatility_panel: pd.DataFrame,
    market_cap: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    if not isinstance(low_volatility_panel.index, pd.MultiIndex):
        raise ValueError("low-volatility panel requires a MultiIndex")
    if list(low_volatility_panel.index.names) != ["date", "code"]:
        raise ValueError("low-volatility panel index must be date/code")
    if low_volatility_panel.index.has_duplicates:
        raise ValueError("low-volatility panel contains duplicate date/code rows")
    spec = fixed_size_combination_spec()
    missing = sorted(set(spec["selected_factors"]) - {SIZE_FACTOR} - set(
        low_volatility_panel.columns
    ))
    if missing:
        raise ValueError(f"low-volatility source factors are missing: {missing}")

    panel = low_volatility_panel.copy()
    aligned_market_cap = pd.to_numeric(
        market_cap.reindex(panel.index), errors="coerce",
    )
    panel[SIZE_FACTOR] = calculate_china_size_factor(aligned_market_cap)
    panel[SOURCE_FACTOR] = _apply_combination(panel, spec)
    available = int(panel[SOURCE_FACTOR].notna().sum())
    if available == 0:
        raise RuntimeError("fixed China-size combination has no values")
    return panel, {
        "panel_rows": int(len(panel)),
        "market_cap_available_rows": int(aligned_market_cap.notna().sum()),
        "size_factor_available_rows": int(panel[SIZE_FACTOR].notna().sum()),
        "combination_available_rows": available,
    }


def _market_cap_series(frame: pd.DataFrame) -> pd.Series:
    required = {"date", "code", "total_mv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"daily_basic market-cap fields are missing: {missing}")
    data = frame[["date", "code", "total_mv"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str)
    data["total_mv"] = pd.to_numeric(data["total_mv"], errors="coerce")
    data = (
        data.dropna(subset=["date", "code"])
        .drop_duplicates(["date", "code"], keep="last")
        .set_index(["date", "code"])
        .sort_index()
    )
    return data["total_mv"]


def _load_market_caps(
    database: Database,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                ts_code,
                trade_date,
                json_extract(payload_json, '$.total_mv'),
                length(payload_json)
            FROM raw.tushare_dataset_rows
                 INDEXED BY idx_tushare_dataset_trade_date
            WHERE dataset = 'daily_basic'
              AND trade_date >= ?
              AND trade_date <= ?
            ORDER BY trade_date, ts_code, row_key
            """,
            [start_date, end_date],
        ).fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame(
        rows, columns=["ts_code", "date", "total_mv", "_payload_size"],
    )
    if frame.empty:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "code"]),
            name="total_mv",
        )
    frame["code"] = frame["ts_code"].map(tushare_to_project_code)
    frame["_has_market_cap"] = pd.to_numeric(
        frame["total_mv"], errors="coerce",
    ).notna()
    frame = (
        frame.sort_values(
            ["date", "code", "_has_market_cap", "_payload_size"],
            kind="mergesort",
        )
        .drop_duplicates(["date", "code"], keep="last")
    )
    return _market_cap_series(frame)


def _load_average_amount(
    database: Database,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                ts_code,
                trade_date,
                json_extract(payload_json, '$.amount')
            FROM raw.tushare_dataset_rows
                 INDEXED BY idx_tushare_dataset_trade_date
            WHERE dataset = 'daily_kline'
              AND trade_date >= ?
              AND trade_date <= ?
            ORDER BY trade_date, ts_code, row_key
            """,
            [start_date, end_date],
        ).fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame(rows, columns=["ts_code", "date", "amount"])
    if frame.empty:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "code"]),
            name="average_amount_20_yuan",
        )
    frame["code"] = frame["ts_code"].map(tushare_to_project_code)
    frame = frame.drop_duplicates(["date", "code"], keep="last")
    return _rolling_average_amount(frame)


def build_candidates(
    *,
    low_volatility_panel_root: str | Path,
    formula_history: str | Path,
    database_path: str | Path,
    output_root: str | Path,
    start_date: str,
    end_date: str,
    top_n: int = 100,
) -> Path:
    panel_root = Path(low_volatility_panel_root)
    formula_path = Path(formula_history)
    database = Path(database_path)
    output = Path(output_root)
    periods = _year_periods(start_date, end_date)
    market_database = Database(database)
    combination_spec = fixed_size_combination_spec()
    final_spec = {
        **combination_spec,
        "method": "PIT_liquidity_prefilter_then_fixed_seven_factor_rank",
        "source_factor": SOURCE_FACTOR,
        "minimum_average_amount_yuan": LIQUIDITY_THRESHOLD_YUAN,
        "liquidity_window_trading_days": 20,
        "liquidity_minimum_observations": 20,
    }
    source_panels = []
    coverage_rows = []

    for year, period_start, period_end in periods:
        panel_path = panel_root / f"factor_panel_{year}.pkl"
        if not panel_path.is_file():
            raise RuntimeError(f"cached low-volatility panel is missing for {year}")
        print(
            f"[china-size] year={year} period={period_start}..{period_end}",
            flush=True,
        )
        low_volatility_panel = _load_cached_panel(panel_path)
        market_caps = _load_market_caps(
            market_database,
            start_date=period_start,
            end_date=period_end,
        )
        panel, coverage = add_size_combination(
            low_volatility_panel,
            market_caps,
        )
        history_start = (
            pd.Timestamp(period_start) - pd.Timedelta(days=60)
        ).strftime(
            "%Y-%m-%d"
        )
        average_amount = _load_average_amount(
            market_database,
            start_date=history_start,
            end_date=period_end,
        )
        panel, liquidity_coverage = apply_liquidity_universe(
            panel,
            panel[SOURCE_FACTOR],
            average_amount,
            (LIQUIDITY_THRESHOLD_YUAN,),
            source_factor=SOURCE_FACTOR,
        )
        if liquidity_coverage[0]["factor"] != FINAL_FACTOR:
            raise RuntimeError("unexpected fixed liquidity factor name")
        coverage_rows.append({
            "year": int(year),
            **coverage,
            **liquidity_coverage[0],
        })
        snapshot_dates = _formula_snapshot_dates(
            formula_path,
            start_date=period_start,
            end_date=period_end,
        )
        prepare_factor_candidates(
            panel,
            panel.index.to_frame(index=False),
            factors=[FINAL_FACTOR],
            snapshot_dates=snapshot_dates,
            output_root=output / "chunks" / str(year),
            top_n=int(top_n),
            start_date=period_start,
            end_date=period_end,
            factor_metadata_by_name={FINAL_FACTOR: final_spec},
        )
        source_panels.append({
            "year": int(year),
            "low_volatility_panel": str(panel_path),
            "low_volatility_panel_sha256": _sha256(panel_path),
            **coverage,
            **liquidity_coverage[0],
        })
        del low_volatility_panel, market_caps, panel, average_amount
        gc.collect()

    manifest = {
        "version": 1,
        "selection_rule": "single_pre_registered_fixed_china_size_combination",
        "future_return_used_for_candidate_selection": False,
        "full_history_previously_observed": True,
        "research_only": True,
        "start_date": start_date,
        "end_date": end_date,
        "top_n": int(top_n),
        "formula_history": str(formula_path),
        "formula_history_sha256": _sha256(formula_path),
        "database_path": str(database),
        "database_size_bytes": database.stat().st_size,
        "combinations": {FINAL_FACTOR: final_spec},
        "source_panels": source_panels,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "combination_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(coverage_rows).to_csv(
        output / "size_liquidity_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_index = _merge_candidate_chunks(
        periods=periods,
        output_root=output,
        start_date=start_date,
        end_date=end_date,
        combination_manifest=manifest,
        factors_to_merge=[FINAL_FACTOR],
    )
    candidate_index.to_csv(
        output / "candidate_streams.csv", index=False, encoding="utf-8-sig",
    )
    return output / "candidates" / FINAL_FACTOR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low-volatility-panel-root",
        default=str(DEFAULT_LOW_VOLATILITY_PANEL_ROOT),
    )
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-29")
    parser.add_argument("--top-n", type=int, default=100)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = build_candidates(
        low_volatility_panel_root=args.low_volatility_panel_root,
        formula_history=args.formula_history,
        database_path=args.database_path,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
