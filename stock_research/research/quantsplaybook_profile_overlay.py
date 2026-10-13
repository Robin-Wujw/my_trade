"""Research a trading-profile overlay inside the frozen low-correlation Top50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.core.paths import PATHS


DEFAULT_PROFILE_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_candidate_profile_research_2021_to_20260721"
    / "candidates" / "playbook_low_corr"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_profile_overlay_2021_to_20260721"
)
OVERLAY_WEIGHTS = {
    "low_corr": 0.50,
    "trade_basis": 0.20,
    "structure_proximity": 0.15,
    "right_strength": 0.10,
    "liquidity": 0.05,
}
VARIANTS = {
    "playbook_low_corr_profiled": 50,
    "playbook_low_corr_profile_top40": 40,
    "playbook_low_corr_profile_top30": 30,
}


def _read_profile_snapshots(candidate_directory: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = candidate_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for item in manifest.get("snapshots") or []:
        path = candidate_directory / str(item["file"])
        expected = str(item.get("sha256") or "")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"candidate snapshot hash mismatch: {path}")
        frame = pd.read_csv(path, dtype={"code": str})
        frame["date"] = str(item["date"])
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"candidate profile snapshots are empty: {manifest_path}")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "code"])
    return panel, manifest


def add_profile_overlay_score(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    numeric_fields = (
        "playbook_factor_rank",
        "candidate_score",
        "trade_basis_score",
        "structure_proximity_score",
        "right_strength_score",
        "avg_amount_20",
    )
    for field in numeric_fields:
        result[field] = pd.to_numeric(result.get(field), errors="coerce")
    result["original_candidate_score"] = result["candidate_score"]
    result["low_corr_component"] = (
        51.0 - result["playbook_factor_rank"]
    ).div(50.0).clip(0.0, 1.0)

    ranked_fields = {
        "trade_basis_component": "trade_basis_score",
        "structure_proximity_component": "structure_proximity_score",
        "right_strength_component": "right_strength_score",
        "liquidity_component": "avg_amount_20",
    }
    for output, source in ranked_fields.items():
        result[output] = result.groupby("date", sort=False)[source].rank(
            method="average", pct=True,
        ).fillna(0.0)
    result["profile_overlay_score"] = 100.0 * (
        result["low_corr_component"] * OVERLAY_WEIGHTS["low_corr"]
        + result["trade_basis_component"] * OVERLAY_WEIGHTS["trade_basis"]
        + result["structure_proximity_component"]
        * OVERLAY_WEIGHTS["structure_proximity"]
        + result["right_strength_component"]
        * OVERLAY_WEIGHTS["right_strength"]
        + result["liquidity_component"] * OVERLAY_WEIGHTS["liquidity"]
    )
    result["profile_overlay_rank"] = result.groupby(
        "date", sort=False,
    )["profile_overlay_score"].rank(
        method="first", ascending=False,
    ).astype(int)
    return result


def _period_name(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp("2023-12-31"):
        return "训练期2021-2023"
    if date <= pd.Timestamp("2024-12-31"):
        return "验证期2024"
    return "样本外2025-2026"


def _selection_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    source = panel.copy()
    source["阶段"] = source["date"].map(_period_name)
    source["forward_return"] = pd.to_numeric(
        source.get("forward_return"), errors="coerce",
    )
    baseline_daily = source.dropna(subset=["forward_return"]).groupby(
        "date", sort=False,
    )["forward_return"].mean()
    rows = []
    selections = {"原始Top50": pd.Series(True, index=source.index)}
    selections.update({
        f"画像重排Top{top_n}": source["profile_overlay_rank"].le(top_n)
        for top_n in (40, 30)
    })
    for name, selected_mask in selections.items():
        selected = source[selected_mask].copy()
        for period, group in selected.groupby("阶段", sort=False):
            valid = group.dropna(subset=["forward_return"])
            daily = valid.groupby("date", sort=False)["forward_return"].mean()
            baseline = baseline_daily.reindex(daily.index)
            excess = daily - baseline
            counts = group.groupby("date", sort=False).size()
            rows.append({
                "方案": name,
                "阶段": period,
                "日均候选数": round(float(counts.mean()), 3),
                "标签行数": int(len(valid)),
                "20日平均收益": float(valid["forward_return"].mean()),
                "20日中位收益": float(valid["forward_return"].median()),
                "20日胜率": float(valid["forward_return"].gt(0).mean()),
                "强盈利率_不低于10%": float(
                    valid["forward_return"].ge(0.10).mean()
                ),
                "尾部亏损率_低于负5%": float(
                    valid["forward_return"].lt(-0.05).mean()
                ),
                "日均超额收益": float(excess.mean()),
                "超额为正交易日占比": float(excess.gt(0).mean()),
            })
    return pd.DataFrame(rows)


def _bucket_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    source = panel[
        panel["date"].le(pd.Timestamp("2023-12-31"))
    ].dropna(subset=["forward_return"]).copy()
    definitions = {
        "原始因子名次": (
            "playbook_factor_rank",
            [0, 10, 20, 30, 40, 50],
            ["1-10", "11-20", "21-30", "31-40", "41-50"],
        ),
        "交易基础分": (
            "trade_basis_score",
            [-np.inf, 4.0, 6.5, 8.0, np.inf],
            ["低于4", "4至6.5", "6.5至8", "不低于8"],
        ),
        "结构接近分": (
            "structure_proximity_score",
            [-np.inf, 20.0, 40.0, 60.0, np.inf],
            ["低于20", "20至40", "40至60", "不低于60"],
        ),
        "右侧强度分": (
            "right_strength_score",
            [-np.inf, 10.0, 20.0, 30.0, np.inf],
            ["低于10", "10至20", "20至30", "不低于30"],
        ),
        "领导力分": (
            "leadership_score",
            [-np.inf, 10.0, 18.0, 25.0, np.inf],
            ["低于10", "10至18", "18至25", "不低于25"],
        ),
        "60日回撤": (
            "drawdown_60",
            [-np.inf, -0.25, -0.15, -0.08, np.inf],
            ["低于负25%", "负25%至负15%", "负15%至负8%", "不低于负8%"],
        ),
        "20日波动": (
            "volatility_20",
            [-np.inf, 0.02, 0.03, 0.05, np.inf],
            ["低于2%", "2%至3%", "3%至5%", "不低于5%"],
        ),
        "20日均成交额": (
            "avg_amount_20",
            [-np.inf, 50e6, 200e6, 500e6, np.inf],
            ["低于0.5亿", "0.5亿至2亿", "2亿至5亿", "不低于5亿"],
        ),
    }
    rows = []
    for name, (field, bins, labels) in definitions.items():
        values = pd.to_numeric(source.get(field), errors="coerce")
        bucket = pd.cut(
            values, bins=bins, labels=labels, include_lowest=True, right=True,
        )
        for label in labels:
            group = source[bucket.eq(label)]
            returns = pd.to_numeric(group["forward_return"], errors="coerce")
            rows.append({
                "字段": name,
                "分桶": label,
                "样本数": int(len(group)),
                "20日平均收益": float(returns.mean()) if len(group) else np.nan,
                "20日中位收益": float(returns.median()) if len(group) else np.nan,
                "20日胜率": float(returns.gt(0).mean()) if len(group) else np.nan,
                "强盈利率_不低于10%": (
                    float(returns.ge(0.10).mean()) if len(group) else np.nan
                ),
                "尾部亏损率_低于负5%": (
                    float(returns.lt(-0.05).mean()) if len(group) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _write_variant(
    panel: pd.DataFrame,
    source_manifest: dict,
    *,
    source_manifest_path: Path,
    output_directory: Path,
    factor: str,
    top_n: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    snapshots = []
    decision_columns = [
        column for column in panel.columns
        if column not in {"forward_return", "label_available", "label_date"}
    ]
    for date, group in panel.groupby("date", sort=True):
        selected = group[
            group["profile_overlay_rank"].le(int(top_n))
        ].copy()
        selected["profile_overlay_rule"] = (
            "50%冻结低相关因子+20%交易基础+15%结构接近+"
            "10%右侧强度+5%流动性"
        )
        if top_n < 50:
            selected["candidate_score"] = selected["profile_overlay_score"]
        selected["playbook_factor"] = factor
        selected["strategy_part"] = f"quantsplaybook:{factor}"
        selected["selection_engine"] = "quantsplaybook_profile_overlay_research"
        selected = selected.sort_values(
            ["candidate_score", "code"],
            ascending=[False, True],
            kind="mergesort",
        )
        path = output_directory / f"candidates_{date:%Y-%m-%d}.csv"
        selected.loc[:, [
            column for column in decision_columns
            if column in selected.columns
        ]].to_csv(path, index=False, encoding="utf-8-sig")
        snapshots.append({
            "date": date.strftime("%Y-%m-%d"),
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": int(len(selected)),
            "signal_eligible_count": int(len(selected)),
            "financial_point_in_time": True,
            "industry_point_in_time": False,
        })
    manifest = dict(source_manifest)
    manifest.update({
        "factor": factor,
        "top_n": int(top_n),
        "snapshot_count": len(snapshots),
        "selection_engine": "quantsplaybook_profile_overlay_research",
        "candidate_profile_enriched": True,
        "profile_overlay_weights": OVERLAY_WEIGHTS,
        "profile_overlay_selected_on": "daily_frozen_low_corr_top50_only",
        "source_candidate_manifest": str(source_manifest_path),
        "source_candidate_manifest_sha256": _sha256(source_manifest_path),
        "future_return_used_for_candidate_selection": False,
        "research_only": True,
        "snapshots": snapshots,
    })
    manifest.pop("candidate_outcome_labels", None)
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_profile_overlay_research(
    candidate_directory: str | Path,
    output_root: str | Path,
) -> dict:
    candidate_directory = Path(candidate_directory)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    panel, manifest = _read_profile_snapshots(candidate_directory)
    outcomes = pd.read_csv(
        candidate_directory / "candidate_outcomes.csv",
        dtype={"code": str},
    )
    outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce")
    outcome_columns = [
        "date", "code", "forward_return", "label_available", "label_date",
    ]
    panel = panel.merge(
        outcomes[outcome_columns], on=["date", "code"], how="left",
        validate="one_to_one",
    )
    panel = add_profile_overlay_score(panel)

    research = output_root / "research"
    research.mkdir(parents=True, exist_ok=True)
    selection_metrics = _selection_metrics(panel)
    bucket_metrics = _bucket_metrics(panel)
    selection_metrics.to_csv(
        research / "候选方案20日标签对照.csv", index=False, encoding="utf-8-sig",
    )
    bucket_metrics.to_csv(
        research / "训练期候选画像分桶.csv", index=False, encoding="utf-8-sig",
    )

    decision_panel = panel.drop(
        columns=["forward_return", "label_available", "label_date"],
    )
    source_manifest_path = candidate_directory / "manifest.json"
    for factor, top_n in VARIANTS.items():
        _write_variant(
            decision_panel,
            manifest,
            source_manifest_path=source_manifest_path,
            output_directory=output_root / "candidates" / factor,
            factor=factor,
            top_n=top_n,
        )
    return {
        "candidate_rows": int(len(panel)),
        "snapshot_count": int(panel["date"].nunique()),
        "variants": VARIANTS,
        "selection_metrics": selection_metrics.to_dict("records"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-directory", default=str(DEFAULT_PROFILE_DIRECTORY),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_profile_overlay_research(
        args.candidate_directory,
        args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
