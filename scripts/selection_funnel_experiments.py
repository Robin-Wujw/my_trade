"""Offline selection-funnel diagnostics without changing fundamentals."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import load_candidate_snapshots, load_price_frames
from stock_research.core.paths import PATHS
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots


WATCH_CODES = {
    "sz.300308": "中际旭创",
    "sz.300502": "新易盛",
    "sh.601138": "工业富联",
    "sh.601869": "长飞光纤",
    "sh.603993": "洛阳钼业",
}


def _number(value, default=0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _sources(row: dict) -> set[str]:
    return {item for item in str(row.get("candidate_source") or "").split("+") if item}


def visible_strength_score(row: dict) -> float:
    """Score only observation-day visible price/volume leadership."""
    trade_basis = min(max(_number(row.get("trade_basis_score")), 0.0), 12.0)
    leadership = min(max(_number(row.get("leadership_score")), 0.0), 30.0)
    ret20 = _number(row.get("return_20d"))
    ret60 = _number(row.get("return_60d"))
    ret120 = _number(row.get("return_120d"))
    distance_high = _number(row.get("distance_120d_high"), default=-1.0)
    volume_ratio = _number(row.get("known_volume_ratio"))

    score = 0.0
    score += trade_basis * 2.0
    score += leadership * 1.5
    score += min(max(ret20, -0.10), 0.50) * 30.0
    score += min(max(ret60, -0.10), 0.80) * 20.0
    score += min(max(ret120, -0.10), 1.00) * 10.0
    if distance_high >= -0.05:
        score += 8.0
    elif distance_high >= -0.12:
        score += 4.0
    if volume_ratio >= 1.2:
        score += 4.0
    return round(score, 6)


def baida_growth_candidate(row: dict) -> bool:
    """Keep the fundamental gate unchanged; only add a right-side growth lane."""
    return (
        _number(row.get("quality_score")) >= 70.0
        and _number(row.get("earnings_yoy")) >= 0.10
        and _number(row.get("mktcap")) >= 100.0
        and _number(row.get("trade_basis_score")) >= 7.0
        and _number(row.get("leadership_score")) >= 15.0
        and _number(row.get("return_20d")) >= 0.15
        and _number(row.get("distance_120d_high"), default=-1.0) >= -0.12
    )


def experiment_selection(rows: list[dict], mode: str, limit: int) -> list[dict]:
    normalized = normalize_candidate_snapshots({"x": rows}, include_diagnostics=True)["x"]
    tradeable = [
        dict(row)
        for row in normalized
        if bool(row.get("signal_eligible")) and bool(row.get("selected_for_trading"))
    ]
    diagnostics = [
        dict(row)
        for row in normalized
        if not (bool(row.get("signal_eligible")) and bool(row.get("selected_for_trading")))
    ]
    if mode == "current":
        selected = sorted(
            tradeable,
            key=lambda item: (
                _number(item.get("selection_rank"), default=9999),
                -_number(item.get("candidate_score")),
                item.get("code", ""),
            ),
        )[:limit]
    elif mode == "strength_top10":
        selected = sorted(
            tradeable + diagnostics,
            key=lambda item: (-visible_strength_score(item), item.get("code", "")),
        )[:limit]
    elif mode == "core5_growth5":
        core = [
            item for item in tradeable
            if _sources(item) & {"value_model", "standard_mainline"}
        ]
        core = sorted(
            core,
            key=lambda item: (
                -_number(item.get("core_candidate_score"), _number(item.get("candidate_score"))),
                item.get("code", ""),
            ),
        )[:5]
        core_codes = {item["code"] for item in core}
        growth_pool = [
            item for item in tradeable + diagnostics
            if item.get("code") not in core_codes and baida_growth_candidate(item)
        ]
        growth = sorted(
            growth_pool,
            key=lambda item: (-visible_strength_score(item), item.get("code", "")),
        )[: max(0, limit - len(core))]
        selected = core + growth
    else:
        raise ValueError(f"unknown mode: {mode}")
    for rank, item in enumerate(selected, start=1):
        item["experiment_rank"] = rank
        item["visible_strength_score"] = visible_strength_score(item)
        item["baida_growth_candidate"] = baida_growth_candidate(item)
        item["experiment_scope"] = (
            "production_tradeable"
            if bool(item.get("signal_eligible")) and bool(item.get("selected_for_trading"))
            else "diagnostic_nontradable_not_for_production"
        )
    return selected


def forward_returns(price_frames, code: str, date: str) -> dict:
    frame = price_frames.get(code)
    if frame is None or frame.empty:
        return {}
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    current = data[data["date"] >= pd.Timestamp(date)]
    if current.empty:
        return {}
    entry = float(current.iloc[0]["close"])
    result = {}
    for horizon in (20, 60, 120):
        window = current.head(horizon + 1)
        if len(window) < 2 or entry <= 0:
            continue
        result[f"forward_{horizon}d_close_return_pct"] = round(
            (float(window.iloc[-1]["close"]) / entry - 1.0) * 100.0, 3,
        )
        result[f"forward_{horizon}d_max_return_pct"] = round(
            (float(window["high"].max()) / entry - 1.0) * 100.0, 3,
        )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2025-04-01")
    parser.add_argument("--end-date", default="2025-06-30")
    parser.add_argument("--forward-end-date", default="2026-07-14")
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
    )
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "selection_funnel_experiments"),
    )
    args = parser.parse_args(argv)

    raw = load_candidate_snapshots(
        args.candidate_directory, args.start_date, args.end_date,
    )
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
    all_codes = {
        str(row["code"])
        for rows in snapshots.values()
        for row in rows
    } | set(WATCH_CODES)
    price_frames = load_price_frames(
        all_codes,
        PATHS.cache / "formula33_kline" / "akshare",
        start_date=(pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=args.forward_end_date,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    watch_rows = []
    hit_counts = defaultdict(int)
    modes = ("current", "strength_top10", "core5_growth5")
    for date, candidates in sorted(snapshots.items()):
        mode_selected = {
            mode: experiment_selection(candidates, mode, 10)
            for mode in modes
        }
        for mode, selected in mode_selected.items():
            for item in selected:
                row = {
                    "date": date,
                    "mode": mode,
                    "experiment_rank": item.get("experiment_rank"),
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "candidate_source": item.get("candidate_source"),
                    "candidate_score": item.get("candidate_score"),
                    "core_candidate_score": item.get("core_candidate_score"),
                    "visible_strength_score": item.get("visible_strength_score"),
                    "baida_growth_candidate": item.get("baida_growth_candidate"),
                    "selection_rank": item.get("selection_rank"),
                    "selected_for_trading": item.get("selected_for_trading"),
                    "allow_right": item.get("allow_right"),
                    "trade_basis_score": item.get("trade_basis_score"),
                    "leadership_score": item.get("leadership_score"),
                    "return_20d": item.get("return_20d"),
                    "return_60d": item.get("return_60d"),
                    "return_120d": item.get("return_120d"),
                    "distance_120d_high": item.get("distance_120d_high"),
                    "candidate_failure_reason": item.get("candidate_failure_reason"),
                    "experiment_scope": item.get("experiment_scope"),
                }
                row.update(forward_returns(price_frames, str(item.get("code")), date))
                rows.append(row)
                if row["code"] in WATCH_CODES:
                    hit_counts[(mode, row["code"])] += 1

        by_code = {str(item.get("code")): item for item in candidates}
        for code, name in WATCH_CODES.items():
            item = by_code.get(code)
            if item is None:
                watch_rows.append({
                    "date": date,
                    "code": code,
                    "name": name,
                    "present": False,
                    "reason": "not_in_candidate_snapshot",
                })
                continue
            row = {
                "date": date,
                "code": code,
                "name": item.get("name") or name,
                "present": True,
                "candidate_source": item.get("candidate_source"),
                "selection_rank": item.get("selection_rank"),
                "selected_for_trading": item.get("selected_for_trading"),
                "signal_eligible": item.get("signal_eligible"),
                "allow_left": item.get("allow_left"),
                "allow_right": item.get("allow_right"),
                "candidate_score": item.get("candidate_score"),
                "core_candidate_score": item.get("core_candidate_score"),
                "visible_strength_score": visible_strength_score(item),
                "baida_growth_candidate": baida_growth_candidate(item),
                "quality_score": item.get("quality_score"),
                "earnings_yoy": item.get("earnings_yoy"),
                "mktcap": item.get("mktcap"),
                "trade_basis_score": item.get("trade_basis_score"),
                "leadership_score": item.get("leadership_score"),
                "return_20d": item.get("return_20d"),
                "return_60d": item.get("return_60d"),
                "return_120d": item.get("return_120d"),
                "distance_120d_high": item.get("distance_120d_high"),
                "candidate_failure_reason": item.get("candidate_failure_reason"),
            }
            for mode, selected in mode_selected.items():
                ranks = {
                    str(entry.get("code")): entry.get("experiment_rank")
                    for entry in selected
                }
                row[f"{mode}_rank"] = ranks.get(code)
            row.update(forward_returns(price_frames, code, date))
            watch_rows.append(row)

    pd.DataFrame(rows).to_csv(
        output / "selection_experiment_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(watch_rows).to_csv(
        output / "watch_code_funnel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "forward_end_date": args.forward_end_date,
        "mode_count": {
            mode: sum(1 for row in rows if row["mode"] == mode)
            for mode in modes
        },
        "watch_code_mode_hits": {
            f"{mode}:{code}": count
            for (mode, code), count in sorted(hit_counts.items())
        },
        "diagnostic_note": (
            "strength_top10/core5_growth5 may include diagnostic_nontradable_not_for_production "
            "rows to study missed names. Forward returns are labels for review only, not "
            "production ranking inputs or tuning gold standards."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
