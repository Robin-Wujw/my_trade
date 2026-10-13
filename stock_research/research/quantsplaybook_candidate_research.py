"""Enrich frozen QuantsPlaybook candidates without refitting factor models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_compare import (
    CANDIDATE_PROFILE_EXTRA_FIELDS,
    _sha256,
    build_candidate_outcome_labels,
    build_candidate_trading_profile_enrichment,
    merge_candidate_snapshot_enrichment,
    required_history_start_date,
)
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_CANDIDATE_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
    / "candidates" / "playbook_low_corr"
)
FROZEN_SELECTION_FIELDS = (
    "code",
    "name",
    "strategy_part",
    "selection_engine",
    "selection_profile",
    "playbook_factor",
    "playbook_factor_score",
    "playbook_factor_rank",
    "candidate_score",
    "selected_for_trading",
    "signal_eligible",
    "allow_right",
    "allow_left",
    "candidate_failure_reason",
)
FACTOR_HORIZON_EXTRA_FIELDS = (
    "factor_horizon_channel_eligible",
    "factor_horizon_structure_proximity_score",
    "factor_horizon_avg_amount_20",
)


def _to_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _with_factor_horizon_channel(row: dict) -> dict:
    """标记 factor_horizon 通道资格，全部只用观察日可见数据。

    与引擎 factor_horizon 分支一致：factor 前 10 名且结构接近分 > 40。
    结构分与 20 日均额来自候选画像（观察日及之前的 bar），显式写出使
    引擎读取的数值与标记依据完全一致，不依赖引擎内部回退。
    """
    updated = dict(row)
    rank = _to_number(row.get("playbook_factor_rank"))
    structure = _to_number(row.get("structure_proximity_score"))
    eligible = (
        rank is not None and float(rank) <= 10
        and structure is not None and float(structure) > 40.0
    )
    updated["factor_horizon_channel_eligible"] = bool(eligible)
    if structure is not None:
        updated["factor_horizon_structure_proximity_score"] = float(structure)
    avg_amount = _to_number(row.get("avg_amount_20"))
    if avg_amount is not None:
        updated["factor_horizon_avg_amount_20"] = float(avg_amount)
    return updated


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_snapshot_rows(
    candidate_directory: Path,
    manifest_rows: list[dict],
) -> dict[str, list[dict]]:
    snapshots = {}
    for item in manifest_rows:
        path = candidate_directory / str(item["file"])
        if not path.is_file():
            raise RuntimeError(f"candidate snapshot is missing: {path}")
        expected = str(item.get("sha256") or "")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"candidate snapshot hash mismatch: {path}")
        frame = pd.read_csv(path, dtype={"code": str})
        snapshots[str(item["date"])] = frame.to_dict("records")
    return snapshots


def enrich_frozen_candidate_directory(
    candidate_directory: str | Path,
    *,
    repository: TushareRepository,
    output_directory: str | Path | None = None,
    holding_days: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
    generate_outcome_labels: bool = True,
) -> dict:
    directory = Path(candidate_directory)
    output = Path(output_directory or directory)
    output.mkdir(parents=True, exist_ok=True)
    in_place = output.resolve() == directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = list(manifest.get("snapshots") or [])
    if not manifest_rows:
        raise RuntimeError(f"candidate manifest has no snapshots: {manifest_path}")

    start = pd.Timestamp(start_date or manifest_rows[0]["date"])
    end = pd.Timestamp(end_date or manifest_rows[-1]["date"])
    selected_rows = [
        row for row in manifest_rows
        if start <= pd.Timestamp(row["date"]) <= end
    ]
    selected_dates = {str(row["date"]) for row in selected_rows}
    preserved_rows = (
        [
            row for row in manifest_rows
            if str(row["date"]) not in selected_dates
        ]
        if in_place else []
    )

    updated_rows = []
    outcome_frames = []
    coverage_rows = []
    years = sorted({pd.Timestamp(row["date"]).year for row in selected_rows})
    for year in years:
        print(f"[候选画像] 年度={year}", flush=True)
        annual_rows = [
            row for row in selected_rows
            if pd.Timestamp(row["date"]).year == year
        ]
        snapshots = _load_snapshot_rows(directory, annual_rows)
        codes_by_date = {
            date: {str(row.get("code")) for row in rows if row.get("code")}
            for date, rows in snapshots.items()
        }
        codes = set().union(*codes_by_date.values()) if codes_by_date else set()
        annual_start = min(snapshots)
        annual_end = max(snapshots)
        history_start = required_history_start_date(annual_start).strftime(
            "%Y-%m-%d",
        )
        prices = repository.load_daily_kline_frames(
            sorted(codes),
            start_date=history_start,
            end_date=annual_end,
        )
        enrichment = build_candidate_trading_profile_enrichment(
            prices,
            [pd.Timestamp(value) for value in snapshots],
            codes=codes,
            candidate_codes_by_date=codes_by_date,
            amount_multiplier=1000.0,
        )
        snapshots = merge_candidate_snapshot_enrichment(snapshots, enrichment)
        outcomes = (
            build_candidate_outcome_labels(
                snapshots,
                prices,
                holding_days=int(holding_days),
                label_end_date=annual_end,
            )
            if generate_outcome_labels else pd.DataFrame()
        )
        if generate_outcome_labels:
            outcome_frames.append(outcomes)

        profile_count = 0
        candidate_count = 0
        for item in annual_rows:
            date = str(item["date"])
            rows = [_with_factor_horizon_channel(row) for row in snapshots[date]]
            path = output / str(item["file"])
            _atomic_write_csv(pd.DataFrame(rows), path)
            copied = dict(item)
            copied["sha256"] = _sha256(path)
            copied["candidate_count"] = len(rows)
            copied["signal_eligible_count"] = len(rows)
            updated_rows.append(copied)
            candidate_count += len(rows)
            profile_count += sum(
                row.get("candidate_profile_status")
                != "missing_price_profile"
                for row in rows
            )
        coverage_rows.append({
            "year": year,
            "candidate_count": candidate_count,
            "profile_count": profile_count,
            "profile_coverage": (
                profile_count / candidate_count if candidate_count else 0.0
            ),
            "outcome_count": int(len(outcomes)),
            "available_outcome_count": int(
                outcomes.get(
                    "label_available",
                    pd.Series(False, index=outcomes.index),
                ).fillna(False).astype(bool).sum()
            ),
        })

    all_manifest_rows = sorted(
        preserved_rows + updated_rows,
        key=lambda row: str(row["date"]),
    )
    if not all_manifest_rows:
        raise RuntimeError("selected candidate snapshot range is empty")
    outcomes = (
        pd.concat(outcome_frames, ignore_index=True)
        if outcome_frames else pd.DataFrame()
    )
    outcome_path = output / "candidate_outcomes.csv"
    if generate_outcome_labels:
        outcomes = outcomes.sort_values(
            ["date", "playbook_factor_rank", "code"], kind="mergesort",
        ).reset_index(drop=True)
        _atomic_write_csv(outcomes, outcome_path)

    manifest.update({
        "snapshot_count": len(all_manifest_rows),
        "start_date": str(all_manifest_rows[0]["date"]),
        "end_date": str(all_manifest_rows[-1]["date"]),
        "snapshots": all_manifest_rows,
        "candidate_profile_enriched": True,
        "candidate_profile_rule": (
            "price_features_use_bars_on_or_before_snapshot_date"
        ),
        "candidate_profile_extra_fields": list(CANDIDATE_PROFILE_EXTRA_FIELDS),
        "factor_horizon_channel_rule": (
            "factor_rank<=10 and structure_proximity_score>40; "
            "visible_at_close_features_only"
        ),
        "factor_horizon_extra_fields": list(FACTOR_HORIZON_EXTRA_FIELDS),
    })
    if generate_outcome_labels:
        manifest["candidate_outcome_labels"] = {
            "file": outcome_path.name,
            "sha256": _sha256(outcome_path),
            "row_count": int(len(outcomes)),
            "available_count": int(
                outcomes["label_available"].fillna(False).astype(bool).sum()
            ),
            "holding_days": int(holding_days),
            "label_role": "research_label_only_not_candidate_input",
            "future_return_used_for_candidate_selection": False,
            "annual_label_boundary": (
                "labels crossing a calendar-year boundary are excluded"
            ),
        }
    else:
        manifest.pop("candidate_outcome_labels", None)
        manifest["candidate_profiles_only"] = True
    output_manifest_path = output / "manifest.json"
    _atomic_write_json(manifest, output_manifest_path)
    return {
        "source_candidate_directory": str(directory),
        "candidate_directory": str(output),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "snapshot_count": len(selected_rows),
        "outcome_row_count": int(len(outcomes)),
        "annual_coverage": coverage_rows,
    }


def restore_frozen_candidate_schema(candidate_directory: str | Path) -> dict:
    """Remove research-only columns from a frozen candidate stream."""
    directory = Path(candidate_directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_rows = []
    row_count = 0
    for item in manifest.get("snapshots") or []:
        path = directory / str(item["file"])
        frame = pd.read_csv(path, dtype={"code": str})
        missing = set(FROZEN_SELECTION_FIELDS) - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"candidate snapshot misses frozen fields: {path} {sorted(missing)}",
            )
        _atomic_write_csv(frame.loc[:, FROZEN_SELECTION_FIELDS], path)
        copied = dict(item)
        copied["sha256"] = _sha256(path)
        copied["candidate_count"] = int(len(frame))
        copied["signal_eligible_count"] = int(len(frame))
        updated_rows.append(copied)
        row_count += len(frame)
    manifest.update({
        "snapshot_count": len(updated_rows),
        "snapshots": updated_rows,
        "candidate_profile_enriched": False,
        "candidate_profile_extra_fields": [],
    })
    manifest.pop("candidate_profile_rule", None)
    manifest.pop("candidate_outcome_labels", None)
    _atomic_write_json(manifest, manifest_path)
    return {
        "candidate_directory": str(directory),
        "snapshot_count": len(updated_rows),
        "candidate_row_count": int(row_count),
        "frozen_field_count": len(FROZEN_SELECTION_FIELDS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "为冻结的QuantsPlaybook候选补充交易画像与独立研究结果标签"
        ),
    )
    parser.add_argument(
        "--candidate-directory", default=str(DEFAULT_CANDIDATE_DIRECTORY),
    )
    parser.add_argument("--output-directory")
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--restore-frozen-schema",
        action="store_true",
        help="移除研究画像列并恢复冻结候选字段，不生成结果标签",
    )
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="只补充观察日前交易画像，不生成旧收盘到收盘结果标签",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.restore_frozen_schema:
        result = restore_frozen_candidate_schema(args.candidate_directory)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    repository = TushareRepository(Database(Path(args.database_path)))
    result = enrich_frozen_candidate_directory(
        args.candidate_directory,
        repository=repository,
        output_directory=args.output_directory,
        holding_days=args.holding_days,
        start_date=args.start_date,
        end_date=args.end_date,
        generate_outcome_labels=not args.profiles_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
