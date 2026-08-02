"""Generate announcement-PIT three-lane candidates from frozen local data."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import pandas as pd

from apps.quantsplaybook_compare import _formula_snapshot_dates
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository
from stock_research.storage.tushare_repository import tushare_to_project_code
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.three_lane_selection import (
    LANE_ORDER,
    THREE_LANE_MODEL,
    build_fundamental_events,
    calculate_technical_lanes,
    map_fundamental_lane,
    quota_union_candidates,
)


DEFAULT_SOURCE_ROOT = (
    PATHS.runtime_root / "backtests" / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests" / "three_lane_2021_to_20260721"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests" / "formula33_tushare_2021_to_20260721.csv"
)
STATEMENT_FIELDS = {
    "fina_indicator": (
        "dt_netprofit_yoy", "netprofit_yoy", "roe_dt", "roe", "dt_eps", "bps",
    ),
    "income": ("total_revenue", "n_income_attr_p", "basic_eps"),
    "balancesheet": (
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
    ),
    "cashflow": (
        "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_statement_history(
    database: Database,
    *,
    start_period: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    result = {}
    connection = database.connect(read_only=True)
    try:
        for dataset, fields in STATEMENT_FIELDS.items():
            extracted = ",\n".join(
                f"json_extract(payload_json, '$.{field}') AS {field}"
                for field in fields
            )
            rows = connection.execute(
                f"""
                SELECT ts_code AS code, report_period, ann_date, {extracted}
                FROM raw.tushare_dataset_rows
                WHERE dataset = ?
                  AND report_period >= ?
                  AND ann_date IS NOT NULL
                  AND ann_date <= ?
                ORDER BY ts_code, ann_date, report_period, row_key
                """,
                [dataset, start_period, end_date],
            ).fetchdf()
            rows["code"] = rows["code"].map(tushare_to_project_code)
            result[dataset] = rows
    finally:
        connection.close()
    return result


def _annual_periods(start_date: str, end_date: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for year in range(start.year, end.year + 1):
        yield (
            year,
            max(start, pd.Timestamp(year=year, month=1, day=1)),
            min(end, pd.Timestamp(year=year, month=12, day=31)),
        )


def generate_candidates(
    *,
    database_path: str | Path,
    source_root: str | Path,
    formula_history: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
    lane_top_n: int = 20,
    maximum_candidates: int = 50,
) -> Path:
    database = Database(database_path)
    repository = TushareRepository(database)
    source_root = Path(source_root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    statements = load_statement_history(
        database,
        start_period=(pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
        end_date=end_date,
    )
    print("[three-lane] build announcement event stream", flush=True)
    fundamental_events = build_fundamental_events(statements)
    del statements
    gc.collect()
    if fundamental_events.empty:
        raise RuntimeError("fundamental event stream is empty")

    snapshots = []
    lane_totals = {lane: 0 for lane in LANE_ORDER}
    fundamental_available = 0
    eligible_rows = 0
    for year, period_start, period_end in _annual_periods(start_date, end_date):
        panel_path = source_root / "research_panels" / f"factor_panel_{year}.pkl"
        if not panel_path.is_file():
            raise RuntimeError(f"eligible PIT panel is missing: {panel_path}")
        print(f"[three-lane] year={year} load eligible membership", flush=True)
        panel = pd.read_pickle(panel_path)[["date", "code"]]
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel["code"] = panel["code"].astype(str)
        membership = panel[
            panel["date"].between(period_start, period_end)
        ].drop_duplicates(["date", "code"])
        del panel
        eligible_rows += len(membership)

        history_start = (period_start - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
        print(f"[three-lane] year={year} load local prices {history_start}..{period_end:%Y-%m-%d}", flush=True)
        prices = repository.load_market_daily_frame(
            start_date=history_start,
            end_date=period_end.strftime("%Y-%m-%d"),
        )
        print(f"[three-lane] year={year} calculate technical lanes", flush=True)
        technical = calculate_technical_lanes(prices)
        technical = technical[
            technical["date"].between(period_start, period_end)
        ]
        del prices
        gc.collect()
        print(f"[three-lane] year={year} map financial announcements", flush=True)
        fundamental = map_fundamental_lane(fundamental_events, membership)
        fundamental_available += int(fundamental["fundamental_momentum"].notna().sum())
        scores = (
            membership.merge(technical, on=["date", "code"], how="left")
            .merge(fundamental, on=["date", "code"], how="left")
        )
        snapshot_dates = _formula_snapshot_dates(
            formula_history,
            start_date=period_start.strftime("%Y-%m-%d"),
            end_date=period_end.strftime("%Y-%m-%d"),
        )
        for date in snapshot_dates:
            daily = scores[scores["date"].eq(date)].copy()
            rows = quota_union_candidates(
                daily,
                lane_top_n=lane_top_n,
                maximum_candidates=maximum_candidates,
            )
            details = daily.set_index("code")
            for row in rows:
                source = details.loc[row["code"]]
                if isinstance(source, pd.DataFrame):
                    source = source.iloc[-1]
                lanes = set(row["three_lane_membership"].split("+"))
                for lane in lanes:
                    lane_totals[lane] += 1
                if "fundamental_momentum" in lanes:
                    row.update({
                        "financial_report_period": pd.Timestamp(source["report_period"]).strftime("%Y-%m-%d"),
                        "financial_effective_date": pd.Timestamp(source["effective_date"]).strftime("%Y-%m-%d"),
                        "financial_component_count": int(source["financial_component_count"]),
                        "financial_point_in_time": True,
                        "strict_financial_point_in_time": False,
                    })
            path = output / f"candidates_{date:%Y-%m-%d}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            snapshots.append({
                "date": date.strftime("%Y-%m-%d"),
                "file": path.name,
                "sha256": _sha256(path),
                "candidate_count": len(rows),
                "signal_eligible_count": len(rows),
                "financial_point_in_time": True,
                "strict_financial_point_in_time": False,
                "industry_point_in_time": False,
            })
        del membership, technical, fundamental, scores
        gc.collect()

    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": snapshots[0]["date"],
        "end_date": snapshots[-1]["date"],
        "selection_engine": "three_lane_factor_only",
        "selection_profile": "three_lane_factor_only",
        "factor": THREE_LANE_MODEL,
        "uses_financial_data": True,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": False,
        "financial_point_in_time_note": (
            "Every statement row is activated no earlier than ann_date. The local "
            "cache does not prove a complete vendor revision history, so this is "
            "announcement-PIT research rather than strict revision-complete PIT."
        ),
        "industry_point_in_time": False,
        "industry_data_used": False,
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "universe_source": str(source_root / "research_panels"),
        "universe_rule": (
            "reuse frozen listed/traded/non-ST/minimum-history PIT membership;"
            "no legacy candidate gate"
        ),
        "lane_order": list(LANE_ORDER),
        "lane_top_n": int(lane_top_n),
        "maximum_candidates": int(maximum_candidates),
        "lane_policy": (
            "top20 per lane then deterministic round-robin deduplication;"
            "no cross-lane score addition"
        ),
        "penalties": [
            "very_high_turnover", "turnover_instability", "single_day_volume_spike",
            "single_day_return_concentration", "late_stage_acceleration",
        ],
        "lane_selected_rows": lane_totals,
        "eligible_rows": int(eligible_rows),
        "fundamental_score_available_rows": int(fundamental_available),
        "fundamental_event_count": int(len(fundamental_events)),
        "snapshots": snapshots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--factor", default=THREE_LANE_MODEL)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--lane-top-n", type=int, default=20)
    parser.add_argument("--maximum-candidates", type=int, default=50)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / "candidates" / args.factor
    generate_candidates(
        database_path=args.database,
        source_root=args.source_root,
        formula_history=args.formula_history,
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
        lane_top_n=args.lane_top_n,
        maximum_candidates=args.maximum_candidates,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
