"""用训练期标签冻结因子周期匹配规则，并生成不含未来标签的候选流。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.core.paths import PATHS


DEFAULT_CANDIDATE_DIRECTORIES = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_liquid_balanced_500m_train_2021_2023"
    / "candidates" / "playbook_liquid_balanced_liquidity_500m",
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_liquid_balanced_500m_validation_2024"
    / "candidates" / "playbook_liquid_balanced_liquidity_500m",
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_liquid_balanced_500m_forward_2025_to_20260721"
    / "candidates" / "playbook_liquid_balanced_liquidity_500m",
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_horizon_research_2021_to_20260721"
)
FROZEN_FACTOR = "playbook_liquid_balanced_top10_structure_mid"
FROZEN_RULE_ID = "top10_structure_40_exclusive_60_inclusive"
FROZEN_RULE_CN = "流动性均衡因子前10名且结构接近分大于40、不高于60"
TRAINING_END = pd.Timestamp("2023-12-31")
VALIDATION_END = pd.Timestamp("2024-12-31")
MIN_TRAINING_LABELS = 120
MIN_TRAINING_DATES = 90
PROFILE_FIELDS = (
    "avg_amount_20",
    "drawdown_60",
    "return_10d",
    "return_20d",
    "momentum_60_ex5",
    "momentum_120_ex20",
    "trade_basis_score",
    "structure_proximity_score",
    "right_strength_score",
)
RESEARCH_ONLY_FIELDS = {
    "date",
    "forward_return",
    "label_available",
    "label_date",
    "holding_days",
    "outcome_label_cn",
    "label_role",
    "entry_close_qfq",
    "exit_close_qfq",
    "window_unique_count",
    "window_change",
    "window_up_streak",
    "window_down_streak",
    "phase",
    "reconstruction_version",
    "研究阶段",
    "市场阶段",
    "广度变化方向",
    "广度区间",
}


def _atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _period_name(date: pd.Timestamp) -> str:
    if date <= TRAINING_END:
        return "训练期2021-2023"
    if date <= VALIDATION_END:
        return "验证期2024"
    return "样本外2025-2026"


def _truthy_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def _read_candidate_directory(
    directory: Path,
) -> tuple[pd.DataFrame, dict, Path]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for item in manifest.get("snapshots") or []:
        path = directory / str(item["file"])
        if not path.is_file():
            raise RuntimeError(f"candidate snapshot is missing: {path}")
        expected = str(item.get("sha256") or "")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"candidate snapshot hash mismatch: {path}")
        frame = pd.read_csv(path, dtype={"code": str})
        frame["date"] = str(item["date"])
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"candidate snapshots are empty: {manifest_path}")

    outcome_spec = dict(manifest.get("candidate_outcome_labels") or {})
    outcome_path = directory / str(
        outcome_spec.get("file") or "candidate_outcomes.csv"
    )
    if not outcome_path.is_file():
        raise RuntimeError(f"candidate outcomes are missing: {outcome_path}")
    expected = str(outcome_spec.get("sha256") or "")
    if expected and _sha256(outcome_path) != expected:
        raise RuntimeError(f"candidate outcome hash mismatch: {outcome_path}")
    outcomes = pd.read_csv(outcome_path, dtype={"code": str})
    outcome_fields = [
        field for field in (
            "date", "code", "forward_return", "label_available",
            "label_date", "holding_days", "outcome_label_cn", "label_role",
            "entry_close_qfq", "exit_close_qfq",
        )
        if field in outcomes
    ]
    panel = pd.concat(frames, ignore_index=True).merge(
        outcomes.loc[:, outcome_fields],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    return panel, manifest, manifest_path


def load_research_panel(
    candidate_directories,
    formula_history: str | Path,
) -> tuple[pd.DataFrame, list[dict]]:
    panels = []
    sources = []
    for value in candidate_directories:
        directory = Path(value)
        panel, manifest, manifest_path = _read_candidate_directory(directory)
        panels.append(panel)
        sources.append({
            "candidate_directory": str(directory),
            "manifest_sha256": _sha256(manifest_path),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "factor": manifest.get("factor"),
        })
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date", "code"])
    duplicates = panel.duplicated(["date", "code"], keep=False)
    if duplicates.any():
        sample = panel.loc[duplicates, ["date", "code"]].head().to_dict("records")
        raise RuntimeError(f"candidate directories overlap: {sample}")

    formula = pd.read_csv(formula_history)
    formula["date"] = pd.to_datetime(
        formula["date"], errors="coerce",
    ).dt.normalize()
    formula = formula.dropna(subset=["date"]).drop_duplicates(
        "date", keep="last",
    )
    panel = panel.merge(formula, on="date", how="left", validate="many_to_one")
    if panel["phase"].isna().any():
        missing = panel.loc[panel["phase"].isna(), "date"].drop_duplicates()
        raise RuntimeError(
            "Formula33 history misses candidate dates: "
            f"{missing.dt.strftime('%Y-%m-%d').head().tolist()}"
        )
    panel["研究阶段"] = panel["date"].map(_period_name)
    panel["市场阶段"] = panel["phase"].map({
        "waiting": "等待确认",
        "watch": "观察",
        "active": "右侧激活",
        "exited": "已退出右侧",
    }).fillna("未知")
    change = pd.to_numeric(panel["window_change"], errors="coerce")
    panel["广度变化方向"] = np.select(
        [change.gt(0), change.lt(0)], ["上升", "下降"], default="平稳",
    )
    panel["广度区间"] = pd.cut(
        pd.to_numeric(panel["window_unique_count"], errors="coerce"),
        [-np.inf, 49, 99, 149, np.inf],
        labels=["低于50", "50至99", "100至149", "不低于150"],
    )
    for field in ("playbook_factor_rank", "forward_return", *PROFILE_FIELDS):
        panel[field] = pd.to_numeric(panel.get(field), errors="coerce")
    return panel.sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    ).reset_index(drop=True), sources


def _variant_definitions(panel: pd.DataFrame) -> dict[str, pd.Series]:
    rank = panel["playbook_factor_rank"]
    structure = panel["structure_proximity_score"]
    middle_structure = structure.gt(40.0) & structure.le(60.0)
    variants = {}
    for top_n in (5, 10, 20):
        variants[f"因子前{top_n}名"] = rank.le(top_n)
        variants[f"因子前{top_n}名+结构接近分中档"] = (
            rank.le(top_n) & middle_structure
        )
    return variants


def _metric_row(
    group: pd.DataFrame,
    *,
    rule_name: str,
    period_name: str,
) -> dict:
    valid = group[
        _truthy_series(group["label_available"])
    ].dropna(subset=["forward_return"])
    returns = valid["forward_return"]
    daily = valid.groupby("date", sort=False)["forward_return"].mean()
    return {
        "规则": rule_name,
        "研究阶段": period_name,
        "标签数": int(len(valid)),
        "不同信号日数": int(valid["date"].nunique()),
        "日均候选数": (
            float(valid.groupby("date", sort=False).size().mean())
            if len(valid) else None
        ),
        "20日平均收益": float(returns.mean()) if len(valid) else None,
        "20日中位收益": float(returns.median()) if len(valid) else None,
        "20日胜率": float(returns.gt(0).mean()) if len(valid) else None,
        "强盈利率_不低于10%": (
            float(returns.ge(0.10).mean()) if len(valid) else None
        ),
        "尾部亏损率_低于负5%": (
            float(returns.lt(-0.05).mean()) if len(valid) else None
        ),
        "信号日等权平均收益": float(daily.mean()) if len(daily) else None,
    }


def summarize_variants(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule_name, mask in _variant_definitions(panel).items():
        selected = panel[mask]
        for period_name, group in selected.groupby("研究阶段", sort=False):
            rows.append(_metric_row(
                group, rule_name=rule_name, period_name=str(period_name),
            ))
        training = selected[selected["date"].le(TRAINING_END)].copy()
        training["训练年度"] = training["date"].dt.year.astype(str)
        for year, group in training.groupby("训练年度", sort=True):
            rows.append(_metric_row(
                group, rule_name=rule_name, period_name=f"训练年度{year}",
            ))
    return pd.DataFrame(rows)


def summarize_condition_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    training = panel[
        panel["date"].le(TRAINING_END)
        & panel["playbook_factor_rank"].le(10)
    ].copy()
    definitions = {
        "市场阶段": ("市场阶段", None, None),
        "广度变化方向": ("广度变化方向", None, None),
        "广度区间": ("广度区间", None, None),
        "20日均成交额": (
            "avg_amount_20",
            [-np.inf, 1e9, 2e9, np.inf],
            ["5亿至10亿", "10亿至20亿", "不低于20亿"],
        ),
        "60日回撤": (
            "drawdown_60",
            [-np.inf, -0.25, -0.15, -0.08, np.inf],
            ["低于负25%", "负25%至负15%", "负15%至负8%", "不低于负8%"],
        ),
        "10日收益": (
            "return_10d",
            [-np.inf, -0.10, 0.0, 0.10, np.inf],
            ["低于负10%", "负10%至0", "0至10%", "不低于10%"],
        ),
        "剔除近5日的60日动量": (
            "momentum_60_ex5",
            [-np.inf, -0.10, 0.0, 0.10, 0.30, np.inf],
            ["低于负10%", "负10%至0", "0至10%", "10%至30%", "不低于30%"],
        ),
        "剔除近20日的120日动量": (
            "momentum_120_ex20",
            [-np.inf, -0.10, 0.0, 0.10, 0.30, np.inf],
            ["低于负10%", "负10%至0", "0至10%", "10%至30%", "不低于30%"],
        ),
        "结构接近分": (
            "structure_proximity_score",
            [-np.inf, 20.0, 40.0, 60.0, np.inf],
            ["低于20", "20至40", "40至60", "高于60"],
        ),
    }
    rows = []
    for dimension, (field, bins, labels) in definitions.items():
        bucket = (
            training[field]
            if bins is None else pd.cut(
                pd.to_numeric(training[field], errors="coerce"),
                bins,
                labels=labels,
                include_lowest=True,
            )
        )
        for label, group in training.groupby(bucket, observed=True, sort=False):
            row = _metric_row(
                group,
                rule_name=f"因子前10名/{dimension}/{label}",
                period_name="训练期2021-2023",
            )
            row.update({"条件维度": dimension, "条件分桶": str(label)})
            yearly = group[
                _truthy_series(group["label_available"])
            ].dropna(subset=["forward_return"]).copy()
            yearly["年份"] = yearly["date"].dt.year
            annual = yearly.groupby("年份")["forward_return"].mean()
            row["训练期最差年度平均收益"] = (
                float(annual.min()) if len(annual) else None
            )
            row["训练期正收益年度数"] = int(annual.gt(0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def select_frozen_rule(
    metrics: pd.DataFrame,
    *,
    minimum_labels: int = MIN_TRAINING_LABELS,
    minimum_dates: int = MIN_TRAINING_DATES,
) -> dict:
    training = metrics[
        metrics["研究阶段"].eq("训练期2021-2023")
    ].copy()
    annual = metrics[
        metrics["研究阶段"].astype(str).str.startswith("训练年度")
    ].copy()
    annual_min = annual.groupby("规则")["20日平均收益"].min()
    annual_count = annual.groupby("规则")["研究阶段"].nunique()
    training["训练期最差年度平均收益"] = training["规则"].map(annual_min)
    training["训练年度覆盖数"] = training["规则"].map(annual_count).fillna(0)
    eligible = training[
        training["标签数"].ge(int(minimum_labels))
        & training["不同信号日数"].ge(int(minimum_dates))
        & training["训练年度覆盖数"].ge(3)
        & training["训练期最差年度平均收益"].gt(0)
        & training["20日平均收益"].gt(0)
        & training["20日中位收益"].gt(0)
        & training["20日胜率"].gt(0.50)
    ].copy()
    if eligible.empty:
        raise RuntimeError("没有预设规则通过训练期跨年度稳定性门槛")
    eligible = eligible.sort_values(
        [
            "训练期最差年度平均收益", "标签数", "20日中位收益", "规则",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = eligible.iloc[0].to_dict()
    expected = "因子前10名+结构接近分中档"
    if selected["规则"] != expected:
        raise RuntimeError(
            f"冻结规则和已声明规则不一致: {selected['规则']} != {expected}"
        )
    return {
        "规则编号": FROZEN_RULE_ID,
        "规则中文": FROZEN_RULE_CN,
        "训练期": "2021-01-01至2023-12-31",
        "因子名次上限": 10,
        "结构接近分下限": 40.0,
        "结构接近分下限是否包含": False,
        "结构接近分上限": 60.0,
        "结构接近分上限是否包含": True,
        "20日均成交额下限元": 500_000_000.0,
        "计划持有交易日": 20,
        "训练标签用于冻结规则": True,
        "每日候选选择读取未来收益": False,
        "验证期和样本外用于调参": False,
        "训练期指标": {
            key: selected[key]
            for key in (
                "标签数", "不同信号日数", "日均候选数", "20日平均收益",
                "20日中位收益", "20日胜率", "强盈利率_不低于10%",
                "尾部亏损率_低于负5%", "信号日等权平均收益",
                "训练期最差年度平均收益",
            )
        },
    }


def frozen_rule_mask(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["playbook_factor_rank"].le(10)
        & panel["structure_proximity_score"].gt(40.0)
        & panel["structure_proximity_score"].le(60.0)
    )


def write_frozen_candidate_stream(
    panel: pd.DataFrame,
    *,
    output_directory: Path,
    source_manifests: list[dict],
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = panel[frozen_rule_mask(panel)].copy()
    decision_columns = [
        field for field in panel.columns if field not in RESEARCH_ONLY_FIELDS
    ]
    snapshots = []
    for date, source in panel.groupby("date", sort=True):
        chosen = selected[selected["date"].eq(date)].copy()
        chosen["playbook_factor"] = FROZEN_FACTOR
        chosen["strategy_part"] = f"quantsplaybook:{FROZEN_FACTOR}"
        chosen["selection_engine"] = "因子周期匹配研究"
        chosen["selection_profile"] = "20日因子周期匹配"
        chosen["factor_horizon_rule_cn"] = FROZEN_RULE_CN
        chosen["factor_horizon_holding_days"] = 20
        chosen["factor_horizon_channel_eligible"] = True
        chosen["selected_for_trading"] = True
        chosen["signal_eligible"] = True
        chosen["allow_right"] = True
        chosen["allow_left"] = False
        columns = list(decision_columns)
        for field in (
            "factor_horizon_rule_cn",
            "factor_horizon_holding_days",
            "factor_horizon_channel_eligible",
        ):
            if field not in columns:
                columns.append(field)
        chosen = chosen.loc[:, [field for field in columns if field in chosen]]
        chosen = chosen.sort_values(
            ["playbook_factor_rank", "code"], kind="mergesort",
        )
        path = output_directory / f"candidates_{date:%Y-%m-%d}.csv"
        chosen.to_csv(path, index=False, encoding="utf-8-sig")
        snapshots.append({
            "date": date.strftime("%Y-%m-%d"),
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": int(len(chosen)),
            "signal_eligible_count": int(len(chosen)),
            "financial_point_in_time": True,
            "industry_point_in_time": False,
        })
    source_versions = []
    for value in source_manifests:
        directory = Path(value["candidate_directory"])
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        source_versions.append(manifest)
    manifest = {
        "version": source_versions[0].get("version"),
        "value_industry_rule_version": source_versions[0].get(
            "value_industry_rule_version"
        ),
        "snapshot_count": len(snapshots),
        "requested_start": snapshots[0]["date"],
        "requested_end": snapshots[-1]["date"],
        "start_date": snapshots[0]["date"],
        "end_date": snapshots[-1]["date"],
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "uses_financial_data": False,
        "industry_point_in_time": False,
        "factor": FROZEN_FACTOR,
        "top_n": 10,
        "selection_engine": "quantsplaybook_factor_horizon_research",
        "selection_profile": "20日因子周期匹配",
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "factor_horizon_holding_days": 20,
        "frozen_rule_id": FROZEN_RULE_ID,
        "frozen_rule_cn": FROZEN_RULE_CN,
        "training_period": "2021-01-01_to_2023-12-31",
        "training_labels_used_to_freeze_rule": True,
        "future_return_used_for_daily_candidate_selection": False,
        "validation_or_forward_used_for_rule_selection": False,
        "minimum_average_amount_yuan": 500_000_000.0,
        "source_manifests": source_manifests,
        "research_only": True,
        "snapshots": snapshots,
    }
    _atomic_write_json(manifest, output_directory / "manifest.json")
    return manifest


def run_conditional_return_research(
    *,
    candidate_directories,
    formula_history: str | Path,
    output_root: str | Path,
) -> dict:
    output = Path(output_root)
    research = output / "research"
    candidate_output = output / "candidates" / FROZEN_FACTOR
    research.mkdir(parents=True, exist_ok=True)
    panel, sources = load_research_panel(
        candidate_directories, formula_history,
    )
    metrics = summarize_variants(panel)
    buckets = summarize_condition_buckets(panel)
    frozen = select_frozen_rule(metrics)
    metrics.to_csv(
        research / "预设规则20日表现.csv", index=False, encoding="utf-8-sig",
    )
    buckets.to_csv(
        research / "训练期条件分桶.csv", index=False, encoding="utf-8-sig",
    )
    _atomic_write_json(frozen, research / "冻结规则.json")
    candidate_manifest = write_frozen_candidate_stream(
        panel,
        output_directory=candidate_output,
        source_manifests=sources,
    )
    manifest = {
        "版本": 1,
        "候选行数": int(len(panel)),
        "候选快照数": int(panel["date"].nunique()),
        "标签可用数": int(_truthy_series(panel["label_available"]).sum()),
        "训练期截止": TRAINING_END.strftime("%Y-%m-%d"),
        "验证期截止": VALIDATION_END.strftime("%Y-%m-%d"),
        "冻结规则": frozen,
        "输入候选清单": sources,
        "Formula33文件": str(formula_history),
        "Formula33文件摘要": _sha256(Path(formula_history)),
        "未来收益用途": "仅用于训练期规则研究和样本外验收",
        "每日候选文件包含未来收益": False,
        "候选输出目录": str(candidate_output),
        "候选输出快照数": candidate_manifest["snapshot_count"],
    }
    _atomic_write_json(manifest, research / "研究清单.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-directories",
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_DIRECTORIES),
        help="逗号分隔的训练期、验证期和样本外候选目录",
    )
    parser.add_argument(
        "--formula-history", default=str(DEFAULT_FORMULA_HISTORY),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    directories = tuple(
        value.strip()
        for value in str(args.candidate_directories).split(",")
        if value.strip()
    )
    result = run_conditional_return_research(
        candidate_directories=directories,
        formula_history=args.formula_history,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
