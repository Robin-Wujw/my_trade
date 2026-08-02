"""Append a small low-priority technical supplement to a primary factor stream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from apps.portfolio_backtest import load_candidate_snapshots
from stock_research.core.paths import PATHS
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


DEFAULT_PRIMARY = (
    PATHS.runtime_root / "backtests" / "quantsplaybook_factor_only_2021_to_20260721"
    / "candidates" / "playbook_low_corr"
)
DEFAULT_SUPPLEMENT = (
    PATHS.runtime_root / "backtests" / "three_lane_2021_to_20260721"
    / "candidates" / "fundamental_smooth_high_stage2_vcp"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests" / "three_lane_2021_to_20260721"
)
BLEND_FACTOR = "playbook_low_corr_plus_smooth5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(row: dict) -> float:
    value = pd.to_numeric(row.get("candidate_score"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def _lane_score(row: dict, lane: str) -> float | None:
    for item in str(row.get("three_lane_scores") or "").split(";"):
        key, separator, value = item.partition("=")
        if separator and key == lane:
            converted = pd.to_numeric(value, errors="coerce")
            return float(converted) if pd.notna(converted) else None
    return None


def merge_primary_with_supplement(
    primary_rows: list[dict],
    supplement_rows: list[dict],
    *,
    supplement_lane: str = "smooth_52week_high",
    supplement_count: int = 5,
) -> list[dict]:
    primary = sorted(
        (dict(row) for row in primary_rows),
        key=lambda row: (-_score(row), str(row.get("code") or "")),
    )
    primary_codes = {str(row.get("code") or "") for row in primary}
    for row in primary:
        row["blend_lane"] = "primary_low_corr"
        row["playbook_factor"] = BLEND_FACTOR
    ranked_supplement = []
    for source in supplement_rows:
        code = str(source.get("code") or "")
        lane_score = _lane_score(source, supplement_lane)
        lanes = set(str(source.get("three_lane_membership") or "").split("+"))
        if not code or code in primary_codes or supplement_lane not in lanes or lane_score is None:
            continue
        ranked_supplement.append((code, lane_score, dict(source)))
    ranked_supplement.sort(key=lambda item: (-item[1], item[0]))
    floor = min((_score(row) for row in primary), default=100.0)
    supplement = []
    for rank, (code, lane_score, row) in enumerate(
        ranked_supplement[: int(supplement_count)], 1,
    ):
        row.update({
            "strategy_part": f"supplement:{supplement_lane}",
            "selection_engine": "quantsplaybook_primary_plus_supplement",
            "selection_profile": "quantsplaybook_factor_only",
            "playbook_factor": BLEND_FACTOR,
            "playbook_factor_score": lane_score,
            "playbook_factor_rank": rank,
            "candidate_score": round(floor - rank * 0.001, 6),
            "blend_lane": f"supplement_{supplement_lane}",
            "selected_for_trading": True,
            "signal_eligible": True,
            "allow_right": True,
            "allow_left": False,
        })
        supplement.append(row)
    return primary + supplement


def build_supplement_candidates(
    *,
    primary_directory: str | Path,
    supplement_directory: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
    supplement_lane: str = "smooth_52week_high",
    supplement_count: int = 5,
) -> Path:
    primary_directory = Path(primary_directory)
    supplement_directory = Path(supplement_directory)
    output = Path(output_directory)
    primary_manifest = json.loads(
        (primary_directory / "manifest.json").read_text(encoding="utf-8")
    )
    supplement_manifest = json.loads(
        (supplement_directory / "manifest.json").read_text(encoding="utf-8")
    )
    for name, manifest in (("primary", primary_manifest), ("supplement", supplement_manifest)):
        if manifest.get("version") != SNAPSHOT_VERSION:
            raise RuntimeError(f"{name} candidate snapshot version is stale")
    primary = load_candidate_snapshots(primary_directory, start_date, end_date)
    supplement = load_candidate_snapshots(supplement_directory, start_date, end_date)
    dates = sorted(set(primary) | set(supplement))
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    supplement_total = 0
    for date in dates:
        rows = merge_primary_with_supplement(
            primary.get(date, []),
            supplement.get(date, []),
            supplement_lane=supplement_lane,
            supplement_count=supplement_count,
        )
        supplement_rows = sum(
            str(row.get("blend_lane") or "").startswith("supplement_")
            for row in rows
        )
        supplement_total += supplement_rows
        path = output / f"candidates_{date}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        snapshots.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(rows),
            "primary_count": len(rows) - supplement_rows,
            "supplement_count": supplement_rows,
            "signal_eligible_count": len(rows),
            "financial_point_in_time": True,
            "strict_financial_point_in_time": True,
            "industry_point_in_time": False,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": dates[0],
        "end_date": dates[-1],
        "factor": BLEND_FACTOR,
        "selection_engine": "quantsplaybook_primary_plus_supplement",
        "selection_profile": "quantsplaybook_factor_only",
        "uses_financial_data": False,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "industry_point_in_time": False,
        "industry_data_used": False,
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "primary": {
            "factor": primary_manifest.get("factor"),
            "directory": str(primary_directory),
            "manifest_sha256": _sha256(primary_directory / "manifest.json"),
            "count_policy": "retain_every_primary_candidate",
        },
        "supplement": {
            "lane": supplement_lane,
            "maximum_daily_count": int(supplement_count),
            "directory": str(supplement_directory),
            "manifest_sha256": _sha256(supplement_directory / "manifest.json"),
            "priority_policy": "score_below_daily_primary_floor",
        },
        "total_supplement_rows": int(supplement_total),
        "snapshots": snapshots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-directory", default=str(DEFAULT_PRIMARY))
    parser.add_argument("--supplement-directory", default=str(DEFAULT_SUPPLEMENT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--factor", default=BLEND_FACTOR)
    parser.add_argument("--supplement-lane", default="smooth_52week_high")
    parser.add_argument("--supplement-count", type=int, default=5)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    args = parser.parse_args(argv)
    output = Path(args.output_root) / "candidates" / args.factor
    build_supplement_candidates(
        primary_directory=args.primary_directory,
        supplement_directory=args.supplement_directory,
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
        supplement_lane=args.supplement_lane,
        supplement_count=args.supplement_count,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
