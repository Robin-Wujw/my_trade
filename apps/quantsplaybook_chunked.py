"""Memory-bounded QuantsPlaybook factor generation and combination fitting."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from apps.quantsplaybook_compare import (
    _formula_snapshot_dates,
    _sha256,
    evaluate_labeled_factor_sample,
    prepare_factor_candidates,
    rank_weighted_factor_combination,
    required_history_start_date,
)
from stock_research.core.paths import PATHS
from stock_research.strategies.quantsplaybook_selection import (
    QUANTSPLAYBOOK_COMMIT,
    executable_factor_columns,
)


DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
COMBINATION_FACTORS = (
    "playbook_ensemble",
    "playbook_positive_ic_top5",
    "playbook_train_ic_weighted",
)


def _year_periods(start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    periods = []
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        periods.append((
            year,
            period_start.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"),
        ))
    return periods


def _run_year_source_generation(
    *,
    year: int,
    start_date: str,
    end_date: str,
    output_root: Path,
    formula_history: Path,
    database_path: Path,
    holding_days: int,
    minimum_history: int,
    top_n: int,
    force: bool = False,
) -> None:
    chunk = output_root / "chunks" / str(year)
    cache = output_root / "research_panels" / f"factor_panel_{year}.pkl"
    expected = chunk / "candidates" / "high_quality_momentum" / "manifest.json"
    if not force and cache.is_file() and expected.is_file():
        print(f"[chunked] reuse source factor panel year={year}", flush=True)
        return
    history_start = required_history_start_date(start_date).strftime("%Y-%m-%d")
    command = [
        sys.executable,
        "-m",
        "apps.quantsplaybook_compare",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--history-start-date",
        history_start,
        "--holding-days",
        str(int(holding_days)),
        "--minimum-history",
        str(int(minimum_history)),
        "--top-n",
        str(int(top_n)),
        "--database-path",
        str(database_path),
        "--formula-history",
        str(formula_history),
        "--output-root",
        str(chunk),
        "--factor-panel-cache",
        str(cache),
    ]
    print(
        f"[chunked] generate source factors year={year} "
        f"period={start_date}..{end_date} history={history_start}",
        flush=True,
    )
    subprocess.run(command, check=True)


def _weighted_training_metrics(
    yearly_metrics: pd.DataFrame,
    *,
    train_start_date: str,
    train_end_date: str,
) -> pd.DataFrame:
    start_year = pd.Timestamp(train_start_date).year
    end_year = pd.Timestamp(train_end_date).year
    source = yearly_metrics[
        yearly_metrics["year"].between(start_year, end_year)
        & yearly_metrics["factor"].ne("playbook_ensemble")
    ].copy()
    if source.empty:
        raise RuntimeError("training factor metrics are empty")
    rows = []
    for factor, group in source.groupby("factor", sort=True):
        weights = pd.to_numeric(group["date_count"], errors="coerce").fillna(0)
        if weights.sum() <= 0:
            continue
        rows.append({
            "factor": factor,
            "mean_rank_ic": float(np.average(
                pd.to_numeric(group["mean_rank_ic"], errors="coerce"),
                weights=weights,
            )),
            "top20_excess_forward_return": float(np.average(
                pd.to_numeric(
                    group["top20_excess_forward_return"], errors="coerce",
                ),
                weights=weights,
            )),
            "date_count": int(weights.sum()),
        })
    return pd.DataFrame(rows).set_index("factor")


def _declared_combination_specs(
    training: pd.DataFrame,
    *,
    positive_ic_top_n: int,
) -> dict[str, dict]:
    source_factors = [
        factor for factor in executable_factor_columns()
        if factor != "playbook_ensemble" and factor in training.index
    ]
    equal = {factor: 1.0 / len(source_factors) for factor in source_factors}
    positive = training[
        pd.to_numeric(training["mean_rank_ic"], errors="coerce").gt(0)
    ].sort_values(
        ["mean_rank_ic", "top20_excess_forward_return"],
        ascending=False,
    )
    top = positive.head(int(positive_ic_top_n)).index.tolist()
    if not top:
        raise RuntimeError("no positive training IC factor is available")
    positive_ic = pd.to_numeric(
        positive["mean_rank_ic"], errors="coerce",
    ).clip(lower=0)
    return {
        "playbook_ensemble": {
            "method": "equal_weight_all_source_factor_streams",
            "weights": equal,
        },
        "playbook_positive_ic_top5": {
            "method": "equal_weight_top_positive_training_rank_ic",
            "weights": {factor: 1.0 / len(top) for factor in top},
        },
        "playbook_train_ic_weighted": {
            "method": "positive_training_rank_ic_weighted",
            "weights": (positive_ic / positive_ic.sum()).to_dict(),
        },
    }


def _load_cached_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = frame["code"].astype(str)
    return (
        frame.dropna(subset=["date", "code"])
        .drop_duplicates(["date", "code"], keep="last")
        .set_index(["date", "code"])
        .sort_index()
    )


def _fit_and_generate_combinations(
    *,
    periods: list[tuple[int, str, str]],
    output_root: Path,
    formula_history: Path,
    yearly_metrics: pd.DataFrame,
    holding_days: int,
    top_n: int,
    positive_ic_top_n: int,
    train_start_date: str,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
) -> dict:
    training = _weighted_training_metrics(
        yearly_metrics,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )
    specs = _declared_combination_specs(
        training,
        positive_ic_top_n=positive_ic_top_n,
    )
    validation_year = pd.Timestamp(validation_start_date).year
    validation_path = (
        output_root / "research_panels"
        / f"factor_panel_{validation_year}.pkl"
    )
    validation_panel = _load_cached_panel(validation_path)
    for factor, spec in specs.items():
        validation_panel[factor] = rank_weighted_factor_combination(
            validation_panel,
            spec["weights"],
        )
    validation_metrics = evaluate_labeled_factor_sample(
        validation_panel,
        factor_columns=list(specs),
        label_end_date=validation_end_date,
    ).set_index("factor")
    selected = str(validation_metrics.sort_values(
        ["top20_excess_forward_return", "mean_rank_ic"],
        ascending=False,
    ).index[0])
    del validation_panel
    gc.collect()

    for factor, spec in specs.items():
        spec["weights"] = {
            key: round(float(value), 12)
            for key, value in spec["weights"].items()
        }
        spec["validation_metrics"] = validation_metrics.loc[factor].to_dict()
        spec["selected_before_oos"] = factor == selected

    manifest = {
        "version": 1,
        "source_commit": QUANTSPLAYBOOK_COMMIT,
        "holding_days": int(holding_days),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "validation_start_date": validation_start_date,
        "validation_end_date": validation_end_date,
        "oos_start_date": (
            pd.Timestamp(validation_end_date) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "future_return_role": "training_and_validation_label_only",
        "selection_rule": (
            "highest_validation_top20_excess_then_validation_mean_rank_ic"
        ),
        "selected_combination": selected,
        "oos_used_for_fitting_or_selection": False,
        "annual_label_boundary": (
            "labels crossing a calendar-year boundary are excluded"
        ),
        "combinations": specs,
    }

    for year, start_date, end_date in periods:
        print(f"[chunked] generate combinations year={year}", flush=True)
        panel = _load_cached_panel(
            output_root / "research_panels" / f"factor_panel_{year}.pkl",
        )
        panel["playbook_ensemble"] = rank_weighted_factor_combination(
            panel,
            specs["playbook_ensemble"]["weights"],
        )
        for factor in COMBINATION_FACTORS[1:]:
            panel[factor] = rank_weighted_factor_combination(
                panel,
                specs[factor]["weights"],
            )
        membership = panel.index.to_frame(index=False)
        snapshot_dates = _formula_snapshot_dates(
            formula_history,
            start_date=start_date,
            end_date=end_date,
        )
        prepare_factor_candidates(
            panel,
            membership,
            factors=list(COMBINATION_FACTORS),
            snapshot_dates=snapshot_dates,
            output_root=output_root / "chunks" / str(year),
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
            factor_metadata_by_name=specs,
        )
        del panel, membership
        gc.collect()
    return manifest


def _merge_candidate_chunks(
    *,
    periods: list[tuple[int, str, str]],
    output_root: Path,
    start_date: str,
    end_date: str,
    combination_manifest: dict,
) -> pd.DataFrame:
    target_root = output_root / "candidates"
    target_root.mkdir(parents=True, exist_ok=True)
    factors = sorted({
        path.name
        for year, _, _ in periods
        for path in (output_root / "chunks" / str(year) / "candidates").glob("*")
        if path.is_dir() and (path / "manifest.json").is_file()
    })
    index_rows = []
    for factor in factors:
        target = target_root / factor
        target.mkdir(parents=True, exist_ok=True)
        snapshots = []
        template = None
        seen_dates = set()
        for year, _, _ in periods:
            source = output_root / "chunks" / str(year) / "candidates" / factor
            manifest_path = source / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"candidate factor missing from year {year}: {factor}",
                )
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            template = template or current
            for row in current.get("snapshots") or []:
                date = str(row["date"])
                if date in seen_dates:
                    raise RuntimeError(
                        f"duplicate candidate snapshot factor={factor} date={date}",
                    )
                seen_dates.add(date)
                source_file = source / str(row["file"])
                target_file = target / str(row["file"])
                shutil.copy2(source_file, target_file)
                copied = dict(row)
                copied["sha256"] = _sha256(target_file)
                snapshots.append(copied)
        snapshots.sort(key=lambda row: row["date"])
        manifest = dict(template or {})
        manifest.update({
            "snapshot_count": len(snapshots),
            "start_date": start_date,
            "end_date": end_date,
            "snapshots": snapshots,
        })
        if factor in combination_manifest["combinations"]:
            manifest["factor_metadata"] = (
                combination_manifest["combinations"][factor]
            )
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        counts = [int(row.get("candidate_count") or 0) for row in snapshots]
        index_rows.append({
            "factor": factor,
            "candidate_directory": str(target),
            "snapshot_count": len(snapshots),
            "minimum_candidates": min(counts) if counts else 0,
            "maximum_candidates": max(counts) if counts else 0,
            "mean_candidates": (
                round(float(np.mean(counts)), 3) if counts else 0
            ),
        })
    return pd.DataFrame(index_rows).sort_values("factor").reset_index(drop=True)


def _combined_data_coverage(
    *,
    periods: list[tuple[int, str, str]],
    output_root: Path,
) -> dict:
    slices = []
    for year, _, _ in periods:
        path = output_root / "chunks" / str(year) / "data_coverage.json"
        if not path.is_file():
            raise RuntimeError(f"annual data coverage is missing: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        item["year"] = int(year)
        slices.append(item)
    risk_first = [
        item.get("risk_warning_first_date")
        for item in slices if item.get("risk_warning_first_date")
    ]
    risk_last = [
        item.get("risk_warning_last_date")
        for item in slices if item.get("risk_warning_last_date")
    ]
    return {
        "version": 2,
        "database_path": slices[0].get("database_path"),
        "history_start_date": min(
            str(item["history_start_date"]) for item in slices
        ),
        "start_date": periods[0][1],
        "end_date": periods[-1][2],
        "eligible_rows": sum(int(item.get("eligible_rows") or 0) for item in slices),
        "eligible_codes_max_annual": max(
            int(item.get("eligible_codes") or 0) for item in slices
        ),
        "snapshot_dates": sum(
            int(item.get("snapshot_dates") or 0) for item in slices
        ),
        "old_candidate_pool_used": False,
        "beijing_exchange_open_date": "2021-11-15",
        "risk_warning_sources": [
            "baostock_isST_2021",
            "local_tushare_stock_st_2022_onward",
        ],
        "risk_warning_first_date": min(risk_first) if risk_first else None,
        "risk_warning_last_date": max(risk_last) if risk_last else None,
        "risk_warning_rows": sum(
            int(item.get("risk_warning_rows") or 0) for item in slices
        ),
        "annual_slices": slices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QuantsPlaybook factor candidates in annual memory-bounded chunks",
    )
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--minimum-history", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--positive-ic-top-n", type=int, default=5)
    parser.add_argument("--train-start-date", default="2021-01-01")
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--validation-start-date", default="2024-01-01")
    parser.add_argument("--validation-end-date", default="2024-12-31")
    parser.add_argument(
        "--force-years",
        default="",
        help="comma-separated years whose source factor panels are rebuilt",
    )
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument(
        "--formula-history", default=str(DEFAULT_FORMULA_HISTORY),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    formula_history = Path(args.formula_history)
    periods = _year_periods(args.start_date, args.end_date)
    force_years = {
        int(value.strip())
        for value in str(args.force_years).split(",")
        if value.strip()
    }
    for year, start_date, end_date in periods:
        _run_year_source_generation(
            year=year,
            start_date=start_date,
            end_date=end_date,
            output_root=output_root,
            formula_history=formula_history,
            database_path=Path(args.database_path),
            holding_days=args.holding_days,
            minimum_history=args.minimum_history,
            top_n=args.top_n,
            force=year in force_years,
        )

    metric_frames = []
    for year, _, _ in periods:
        path = output_root / "chunks" / str(year) / "factor_metrics.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "year", year)
        metric_frames.append(frame)
    yearly_metrics = pd.concat(metric_frames, ignore_index=True)
    yearly_metrics.to_csv(
        output_root / "factor_metrics_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combination_manifest = _fit_and_generate_combinations(
        periods=periods,
        output_root=output_root,
        formula_history=formula_history,
        yearly_metrics=yearly_metrics,
        holding_days=args.holding_days,
        top_n=args.top_n,
        positive_ic_top_n=args.positive_ic_top_n,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
    )
    (output_root / "combination_manifest.json").write_text(
        json.dumps(
            combination_manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    candidate_index = _merge_candidate_chunks(
        periods=periods,
        output_root=output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        combination_manifest=combination_manifest,
    )
    candidate_index.to_csv(
        output_root / "candidate_streams.csv",
        index=False,
        encoding="utf-8-sig",
    )
    first_inventory = (
        output_root / "chunks" / str(periods[0][0])
        / "strategy_inventory.csv"
    )
    shutil.copy2(first_inventory, output_root / "strategy_inventory.csv")
    (output_root / "data_coverage.json").write_text(
        json.dumps(
            _combined_data_coverage(
                periods=periods,
                output_root=output_root,
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(candidate_index.to_string(index=False), flush=True)
    print(
        "[chunked] selected before OOS="
        f"{combination_manifest['selected_combination']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
