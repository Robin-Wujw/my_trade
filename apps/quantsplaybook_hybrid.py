"""Build a PIT hybrid stream: QuantsPlaybook right lane plus value left lane."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from apps.portfolio_backtest import (
    load_candidate_snapshots,
    validate_candidate_manifest_financial_point_in_time,
    validate_candidate_manifest_industry_point_in_time,
)
from stock_research.core.paths import PATHS
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


DEFAULT_FACTOR_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_LEFT_CANDIDATES = (
    PATHS.runtime_root / "backtests"
    / "tushare_official_candidates_2021_to_20260721_merged"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_hybrid_low_corr_value_2021_to_20260721"
)
HYBRID_FACTOR = "hybrid_low_corr_value"
INDUSTRY_MANIFEST_FIELDS = (
    "industry_source_path",
    "industry_source_sha256",
    "industry_source_modified_at",
    "industry_as_of_date",
    "industry_data_source",
    "industry_point_in_time_status",
    "industry_mapping_count",
    "industry_universe_count",
    "industry_coverage_ratio",
    "industry_point_in_time_note",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _sources(row: dict) -> set[str]:
    return {
        value.strip()
        for value in str(row.get("candidate_source") or "").split("+")
        if value.strip()
    }


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _left_rows(rows: list[dict]) -> dict[str, dict]:
    selected = {}
    for source in rows:
        if "value_model" not in _sources(source):
            continue
        if not all(
            _truthy(source.get(field))
            for field in (
                "allow_left", "selected_for_trading", "signal_eligible",
                "value_industry_allowed", "financial_point_in_time",
                "industry_point_in_time",
            )
        ):
            continue
        if _truthy(source.get("value_falsified")) or _clean_text(
            source.get("value_falsification_reason")
        ):
            continue
        row = dict(source)
        row.update({
            "candidate_source": "value_model",
            "selection_profile": "hybrid_value_left",
            "hybrid_lane": "left_value",
            "allow_left": True,
            "allow_right": False,
        })
        selected[str(row["code"])] = row
    return selected


def _right_rows(
    rows: list[dict],
    *,
    right_factor: str,
) -> dict[str, dict]:
    lane = f"right_{right_factor.removeprefix('playbook_')}"
    selected = {}
    for source in rows:
        if not all(
            _truthy(source.get(field))
            for field in (
                "selected_for_trading", "signal_eligible", "allow_right",
            )
        ):
            continue
        row = dict(source)
        row.update({
            "candidate_source": "quantsplaybook_factor",
            "selection_profile": "quantsplaybook_factor_only",
            "hybrid_lane": lane,
            "allow_left": False,
            "allow_right": True,
        })
        selected[str(row["code"])] = row
    return selected


def _candidate_score(row: dict) -> float:
    value = pd.to_numeric(row.get("candidate_score"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def merge_hybrid_rows(
    right_rows: list[dict],
    left_rows: list[dict],
    *,
    right_factor: str = "playbook_low_corr",
) -> list[dict]:
    right = _right_rows(right_rows, right_factor=right_factor)
    left = _left_rows(left_rows)
    right_lane = f"right_{right_factor.removeprefix('playbook_')}"
    result = []
    for code in sorted(set(right) | set(left)):
        if code in right and code in left:
            row = dict(left[code])
            factor = right[code]
            for field in (
                "playbook_factor", "playbook_factor_score",
                "playbook_factor_rank", "candidate_score",
                "strategy_part", "selection_engine", "selection_profile",
            ):
                if field in factor:
                    row[field] = factor[field]
            row.update({
                "candidate_source": "value_model+quantsplaybook_factor",
                "hybrid_lane": f"left_value+{right_lane}",
                "allow_left": True,
                "allow_right": True,
            })
        elif code in right:
            row = right[code]
        else:
            row = left[code]
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            -_candidate_score(row),
            str(row.get("code") or ""),
        ),
    )


def _read_manifest(path: Path, *, lane: str) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != SNAPSHOT_VERSION:
        raise RuntimeError(f"{lane} candidate version is stale: {manifest_path}")
    if manifest.get("value_industry_rule_version") != VALUE_INDUSTRY_RULE_VERSION:
        raise RuntimeError(f"{lane} value-industry rule is stale: {manifest_path}")
    if manifest.get("financial_point_in_time") is not True:
        raise RuntimeError(f"{lane} candidates are not financial PIT: {manifest_path}")
    return manifest


def build_hybrid_candidates(
    *,
    right_directory: str | Path,
    left_directory: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
) -> Path:
    right_directory = Path(right_directory)
    left_directory = Path(left_directory)
    output = Path(output_directory)
    right_manifest = _read_manifest(right_directory, lane="right")
    left_manifest = _read_manifest(left_directory, lane="left")
    right_factor = str(
        right_manifest.get("factor") or right_directory.name
    ).strip()
    if not right_factor:
        raise RuntimeError(f"right factor is missing: {right_directory}")
    validate_candidate_manifest_financial_point_in_time(left_directory)
    validate_candidate_manifest_industry_point_in_time(left_directory)

    right = load_candidate_snapshots(right_directory, start_date, end_date)
    left = load_candidate_snapshots(left_directory, start_date, end_date)
    dates = sorted(set(right) | set(left))
    if not dates:
        raise RuntimeError("hybrid candidate date range is empty")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    total_left = 0
    total_right = 0
    total_overlap = 0
    for date in dates:
        rows = merge_hybrid_rows(
            right.get(date, []),
            left.get(date, []),
            right_factor=right_factor,
        )
        frame = pd.DataFrame(rows)
        path = output / f"candidates_{date}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        lanes = frame.get("hybrid_lane", pd.Series(dtype=str)).fillna("")
        left_count = int(lanes.str.contains("left_value", regex=False).sum())
        right_count = int(lanes.str.contains("right_", regex=False).sum())
        overlap_count = int(lanes.str.contains("+", regex=False).sum())
        total_left += left_count
        total_right += right_count
        total_overlap += overlap_count
        snapshots.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(frame),
            "signal_eligible_count": len(frame),
            "left_candidate_count": left_count,
            "right_candidate_count": right_count,
            "overlap_candidate_count": overlap_count,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": dates[0],
        "end_date": dates[-1],
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "unsafe_snapshot_count": 0,
        "industry_point_in_time": True,
        **{
            field: left_manifest.get(field)
            for field in INDUSTRY_MANIFEST_FIELDS
        },
        "selection_engine": "quantsplaybook_right_plus_value_left",
        "selection_profile": output.name,
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "right_lane": {
            "factor": right_factor,
            "candidate_directory": str(right_directory),
            "manifest_sha256": _sha256(right_directory / "manifest.json"),
            "industry_data_used": False,
        },
        "left_lane": {
            "factor": "basic_value_line",
            "candidate_directory": str(left_directory),
            "manifest_sha256": _sha256(left_directory / "manifest.json"),
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
        },
        "lane_policy": (
            f"right candidates only from {right_factor}; left candidates "
            "only from executable value_model; scores are not added across lanes"
        ),
        "total_left_candidate_rows": total_left,
        "total_right_candidate_rows": total_right,
        "total_overlap_candidate_rows": total_overlap,
        "snapshots": snapshots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--right-directory",
        default=str(DEFAULT_FACTOR_ROOT / "candidates" / "playbook_low_corr"),
    )
    parser.add_argument("--left-directory", default=str(DEFAULT_LEFT_CANDIDATES))
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--factor", default=HYBRID_FACTOR)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / "candidates" / args.factor
    build_hybrid_candidates(
        right_directory=args.right_directory,
        left_directory=args.left_directory,
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
