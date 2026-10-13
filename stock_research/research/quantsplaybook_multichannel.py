"""构建紧结构、因子周期、低广度反转与左侧价值的逐日多通道候选。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.backtesting.workflow import (
    load_candidate_snapshots,
    validate_candidate_manifest_financial_point_in_time,
    validate_candidate_manifest_industry_point_in_time,
)
from stock_research.research.quantsplaybook_hybrid import (
    INDUSTRY_MANIFEST_FIELDS,
    _left_rows,
    _read_manifest,
    _sha256,
    _truthy,
)
from stock_research.core.paths import PATHS
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


DEFAULT_PROFILE_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_candidate_profile_research_2021_to_20260721"
)
DEFAULT_FACTOR_HORIZON_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_horizon_train_2021_2023"
)
DEFAULT_BREADTH_REVERSAL_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_breadth_reversal_liquid_balanced_500m_train_2021_2023"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_multichannel_train_2021_2023"
)
MULTICHANNEL_FACTOR = "hybrid_multichannel_value"
CHANNEL_FIELDS = (
    "candidate_score",
    "playbook_factor_rank",
    "playbook_factor_score",
    "structure_proximity_score",
    "avg_amount_20",
    "return_60d",
    "known_volume_ratio",
)
CHANNEL_LABELS_CN = {
    "compressed_structure": "紧结构平台延续候选",
    "factor_horizon": "因子周期匹配候选",
    "breadth_reversal": "低广度反转候选",
}


def _right_rows(rows: list[dict], *, eligibility_field: str | None = None):
    result = {}
    for source in rows:
        if not all(
            _truthy(source.get(field))
            for field in ("selected_for_trading", "signal_eligible", "allow_right")
        ):
            continue
        if eligibility_field and not _truthy(source.get(eligibility_field)):
            continue
        result[str(source.get("code") or "")] = dict(source)
    return {code: row for code, row in result.items() if code}


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _candidate_score(row) -> float:
    value = pd.to_numeric(row.get("candidate_score"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def _merge_channel_row(result, source, channel):
    code = str(source.get("code") or "")
    current = result.get(code, {})
    row = {**source, **current}
    for field in CHANNEL_FIELDS:
        value = source.get(field)
        if not _missing(value):
            row[f"{channel}_{field}"] = value
    row[f"{channel}_channel_eligible"] = True
    channels = set(row.pop("_eligible_channels", []))
    channels.add(channel)
    row["_eligible_channels"] = sorted(channels)
    result[code] = row


def merge_multichannel_rows(
    compressed_rows: list[dict],
    factor_horizon_rows: list[dict],
    breadth_reversal_rows: list[dict],
) -> list[dict]:
    result = {}
    left = {
        **_left_rows(factor_horizon_rows),
        **_left_rows(breadth_reversal_rows),
    }
    for code, row in left.items():
        result[code] = dict(row)

    channel_rows = (
        (
            "compressed_structure",
            _right_rows(compressed_rows),
        ),
        (
            "factor_horizon",
            _right_rows(
                factor_horizon_rows,
                eligibility_field="factor_horizon_channel_eligible",
            ),
        ),
        (
            "breadth_reversal",
            _right_rows(breadth_reversal_rows),
        ),
    )
    for channel, rows in channel_rows:
        for row in rows.values():
            _merge_channel_row(result, row, channel)

    merged = []
    for code, row in result.items():
        channels = row.pop("_eligible_channels", [])
        has_left = code in left
        scores = [
            pd.to_numeric(row.get(f"{channel}_candidate_score"), errors="coerce")
            for channel in channels
        ]
        scores = [float(score) for score in scores if pd.notna(score)]
        if scores:
            row["candidate_score"] = max(scores)
        row.update({
            "candidate_source": (
                "value_model+quantsplaybook_factor"
                if has_left and channels else
                "value_model" if has_left else "quantsplaybook_factor"
            ),
            "selection_engine": "三通道共享资金研究",
            "selection_profile": "quantsplaybook_multichannel",
            "hybrid_lane": "+".join(
                (["left_value"] if has_left else []) + channels
            ),
            "allow_left": has_left,
            "allow_right": bool(channels),
            "selected_for_trading": True,
            "signal_eligible": True,
            "selection_reason": "；".join(
                (["左侧价值候选"] if has_left else [])
                + [CHANNEL_LABELS_CN[channel] for channel in channels]
            ),
        })
        merged.append(row)
    return sorted(
        merged,
        key=lambda row: (
            -_candidate_score(row),
            str(row.get("code") or ""),
        ),
    )


def _validate_channel_manifest(directory: Path, *, lane: str) -> dict:
    manifest = _read_manifest(directory, lane=lane)
    if manifest.get("financial_point_in_time") is not True:
        raise RuntimeError(f"{lane}候选未通过财务时点校验: {directory}")
    return manifest


def build_multichannel_candidates(
    *,
    compressed_directory: str | Path,
    factor_horizon_directory: str | Path,
    breadth_reversal_directory: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
) -> Path:
    compressed_directory = Path(compressed_directory)
    factor_horizon_directory = Path(factor_horizon_directory)
    breadth_reversal_directory = Path(breadth_reversal_directory)
    output = Path(output_directory)
    manifests = {
        "compressed_structure": _validate_channel_manifest(
            compressed_directory, lane="紧结构",
        ),
        "factor_horizon": _validate_channel_manifest(
            factor_horizon_directory, lane="因子周期",
        ),
        "breadth_reversal": _validate_channel_manifest(
            breadth_reversal_directory, lane="低广度反转",
        ),
    }
    for directory in (factor_horizon_directory, breadth_reversal_directory):
        validate_candidate_manifest_financial_point_in_time(directory)
        validate_candidate_manifest_industry_point_in_time(directory)

    snapshots_by_channel = {
        "compressed_structure": load_candidate_snapshots(
            compressed_directory, start_date, end_date,
        ),
        "factor_horizon": load_candidate_snapshots(
            factor_horizon_directory, start_date, end_date,
        ),
        "breadth_reversal": load_candidate_snapshots(
            breadth_reversal_directory, start_date, end_date,
        ),
    }
    date_sets = {
        channel: set(snapshots) for channel, snapshots in snapshots_by_channel.items()
    }
    expected_dates = date_sets["compressed_structure"]
    mismatched = {
        channel: sorted(expected_dates.symmetric_difference(dates))
        for channel, dates in date_sets.items()
        if dates != expected_dates
    }
    if mismatched:
        preview = {
            channel: dates[:5] for channel, dates in mismatched.items()
        }
        raise RuntimeError(f"三通道候选日期不一致: {preview}")
    dates = sorted(expected_dates)
    if not dates:
        raise RuntimeError("三通道候选日期为空")

    output.mkdir(parents=True, exist_ok=True)
    snapshot_manifest = []
    channel_row_totals = {channel: 0 for channel in snapshots_by_channel}
    left_total = 0
    maximum_candidates = 0
    for date in dates:
        rows = merge_multichannel_rows(
            snapshots_by_channel["compressed_structure"].get(date, []),
            snapshots_by_channel["factor_horizon"].get(date, []),
            snapshots_by_channel["breadth_reversal"].get(date, []),
        )
        frame = pd.DataFrame(rows)
        path = output / f"candidates_{date}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        counts = {
            channel: int(
                frame.get(
                    f"{channel}_channel_eligible",
                    pd.Series(dtype=bool),
                ).fillna(False).astype(bool).sum()
            )
            for channel in snapshots_by_channel
        }
        for channel, count in counts.items():
            channel_row_totals[channel] += count
        left_count = int(
            frame.get("allow_left", pd.Series(dtype=bool))
            .fillna(False).astype(bool).sum()
        )
        left_total += left_count
        maximum_candidates = max(maximum_candidates, len(frame))
        snapshot_manifest.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(frame),
            "signal_eligible_count": len(frame),
            "left_candidate_count": left_count,
            **{f"{channel}_candidate_count": count for channel, count in counts.items()},
            "financial_point_in_time": True,
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
        })

    industry_manifest = manifests["factor_horizon"]
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshot_manifest),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": dates[0],
        "end_date": dates[-1],
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "unsafe_snapshot_count": 0,
        "industry_point_in_time": True,
        **{
            field: industry_manifest.get(field)
            for field in INDUSTRY_MANIFEST_FIELDS
        },
        "selection_engine": "三通道共享资金研究",
        "selection_profile": output.name,
        "signal_timing": "T日收盘冻结候选，最早T+1执行",
        "channel_policy": "三类右侧候选按独立资格标签路由；左侧价值候选只保留一份",
        "recommended_max_daily_candidates": 120,
        "maximum_merged_candidates": maximum_candidates,
        "total_left_candidate_rows": left_total,
        "channel_candidate_row_totals": channel_row_totals,
        "research_only": False,
        "channels": {
            channel: {
                "candidate_directory": str(directory),
                "manifest_sha256": _sha256(directory / "manifest.json"),
            }
            for channel, directory in (
                ("compressed_structure", compressed_directory),
                ("factor_horizon", factor_horizon_directory),
                ("breadth_reversal", breadth_reversal_directory),
            )
        },
        "snapshots": snapshot_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compressed-directory",
        default=str(DEFAULT_PROFILE_ROOT / "candidates" / "playbook_low_corr"),
    )
    parser.add_argument(
        "--factor-horizon-directory",
        default=str(
            DEFAULT_FACTOR_HORIZON_ROOT / "candidates"
            / "hybrid_factor_horizon_value"
        ),
    )
    parser.add_argument(
        "--breadth-reversal-directory",
        default=str(
            DEFAULT_BREADTH_REVERSAL_ROOT / "candidates"
            / "hybrid_liquid_balanced_500m_value"
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--factor", default=MULTICHANNEL_FACTOR)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / "candidates" / args.factor
    build_multichannel_candidates(
        compressed_directory=args.compressed_directory,
        factor_horizon_directory=args.factor_horizon_directory,
        breadth_reversal_directory=args.breadth_reversal_directory,
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
