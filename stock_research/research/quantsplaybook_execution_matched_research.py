"""用执行一致标签冻结因子与交易周期匹配规则。"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.research.quantsplaybook_conditional_return_research import (
    DEFAULT_CANDIDATE_DIRECTORIES,
    DEFAULT_FORMULA_HISTORY,
    RESEARCH_ONLY_FIELDS,
)
from stock_research.research.quantsplaybook_execution_aligned_research import (
    EXECUTION_LABEL_FIELDS,
    load_candidate_snapshot_panel,
)
from stock_research.core.paths import PATHS


DEFAULT_EXECUTION_LABELS = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_execution_aligned_research_2021_to_20260721"
    / "执行一致候选结果.csv"
)
DEFAULT_INDUSTRY_MANIFEST = (
    PATHS.runtime_root / "backtests"
    / "tushare_official_candidates_2021_to_20260721_merged"
    / "manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_execution_matched_rule_2021_to_20260721"
)
FROZEN_FACTOR = "playbook_liquid_balanced_top10_structure_ready_event"
FROZEN_RULE_ID = "top10_structure_above_40_event_20_industry_2"
FROZEN_RULE_CN = (
    "流动性均衡因子前10名、结构接近分大于40，"
    "单股20日冷却且同一行业20日内最多2个事件"
)
TRAINING_START = pd.Timestamp("2021-01-01")
TRAINING_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31")
FORWARD_START = pd.Timestamp("2025-01-01")
FORWARD_END = pd.Timestamp("2026-07-21")
STOCK_COOLDOWN_DAYS = 20
INDUSTRY_WINDOW_DAYS = 20
MAX_INDUSTRY_EVENTS = 2
MAX_DAILY_EVENTS = 5
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


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _number(value):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted) or not math.isfinite(float(converted)):
        return None
    return float(converted)


def load_execution_research_panel(
    candidate_directories,
    formula_history: str | Path,
    execution_labels: str | Path,
) -> tuple[pd.DataFrame, list[dict]]:
    del formula_history
    panel, sources = load_candidate_snapshot_panel(candidate_directories)
    labels = pd.read_csv(execution_labels, dtype={"code": str})
    labels["date"] = pd.to_datetime(
        labels["date"], errors="coerce",
    ).dt.normalize()
    label_columns = [
        field for field in EXECUTION_LABEL_FIELDS
        if field not in {"playbook_factor_rank", "candidate_score"}
    ]
    panel = panel.merge(
        labels.loc[:, label_columns],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    return panel, sources


def load_industry_history(
    industry_manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    manifest_path = Path(industry_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(str(manifest.get("industry_source_path") or ""))
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    if not source_path.is_file():
        raise RuntimeError(f"行业时点来源不存在: {source_path}")
    expected = str(manifest.get("industry_source_sha256") or "")
    if expected and _sha256(source_path) != expected:
        raise RuntimeError(f"行业时点来源哈希不一致: {source_path}")
    history = pd.read_csv(source_path, dtype=str)
    history["in_date_ts"] = pd.to_datetime(
        history.get("in_date"), format="%Y%m%d", errors="coerce",
    )
    history["out_date_ts"] = pd.to_datetime(
        history.get("out_date"), format="%Y%m%d", errors="coerce",
    )
    history["industry"] = history.get(
        "industry", pd.Series("", index=history.index),
    ).fillna("").astype(str).str.strip()
    history = history.dropna(subset=["code", "in_date_ts"])
    return history, manifest


def attach_point_in_time_industry(
    panel: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.copy()
    groups = {
        str(code): group.sort_values(
            ["in_date_ts", "out_date_ts"], na_position="last",
        )
        for code, group in history.groupby("code", sort=False)
    }
    industries = []
    for code, date in zip(result["code"], result["date"]):
        group = groups.get(str(code))
        if group is None or group.empty:
            industries.append("")
            continue
        active = group[
            group["in_date_ts"].le(date)
            & (group["out_date_ts"].isna() | group["out_date_ts"].ge(date))
        ]
        industries.append(
            "" if active.empty else str(active.iloc[-1]["industry"])
        )
    result["industry"] = industries
    result["industry_point_in_time"] = True
    return result


def rule_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    rank = pd.to_numeric(panel["playbook_factor_rank"], errors="coerce")
    structure = pd.to_numeric(
        panel["structure_proximity_score"], errors="coerce",
    )
    return {
        "原规则_前10名且结构分大于40不高于60": (
            rank.le(10) & structure.gt(40) & structure.le(60)
        ),
        "前5名且结构分大于40": rank.le(5) & structure.gt(40),
        "前10名且结构分大于40": rank.le(10) & structure.gt(40),
        "前20名且结构分大于40": rank.le(20) & structure.gt(40),
    }


def select_independent_events(
    panel: pd.DataFrame,
    eligible: pd.Series,
    *,
    stock_cooldown_days: int = STOCK_COOLDOWN_DAYS,
    industry_window_days: int = INDUSTRY_WINDOW_DAYS,
    max_industry_events: int = MAX_INDUSTRY_EVENTS,
    max_daily_events: int = MAX_DAILY_EVENTS,
) -> pd.DataFrame:
    """Select events using observation-date fields only."""
    dates = pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values()
    date_positions = {
        pd.Timestamp(date): index for index, date in enumerate(dates)
    }
    source = panel[eligible.fillna(False)].copy()
    sort_spec = [
        ("date", True),
        ("playbook_factor_rank", True),
        ("structure_proximity_score", False),
        ("candidate_score", False),
        ("code", True),
    ]
    sort_spec = [item for item in sort_spec if item[0] in source]
    source = source.sort_values(
        [field for field, _ in sort_spec],
        ascending=[ascending for _, ascending in sort_spec],
        kind="mergesort",
    )
    selected_indices = []
    last_symbol_event: dict[str, int] = {}
    industry_events: dict[str, deque[int]] = defaultdict(deque)
    for date, daily in source.groupby("date", sort=True):
        date_position = date_positions[pd.Timestamp(date)]
        daily_count = 0
        for row in daily.itertuples():
            code = str(row.code)
            if (
                date_position - last_symbol_event.get(code, -10_000)
                < int(stock_cooldown_days)
            ):
                continue
            industry = str(getattr(row, "industry", "") or "").strip()
            recent = industry_events[industry]
            while recent and (
                date_position - recent[0] >= int(industry_window_days)
            ):
                recent.popleft()
            if industry and len(recent) >= int(max_industry_events):
                continue
            selected_indices.append(row.Index)
            last_symbol_event[code] = date_position
            if industry:
                recent.append(date_position)
            daily_count += 1
            if daily_count >= int(max_daily_events):
                break
    events = panel.loc[selected_indices].copy()
    events = events.sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    ).reset_index(drop=True)
    events["selection_event_id"] = [
        f"{date:%Y%m%d}-{code.replace('.', '')}"
        for date, code in zip(events["date"], events["code"])
    ]
    events["selection_reason_cn"] = FROZEN_RULE_CN
    events["stock_cooldown_days"] = int(stock_cooldown_days)
    events["industry_window_days"] = int(industry_window_days)
    events["max_industry_events"] = int(max_industry_events)
    return events


def _valid_returns(events: pd.DataFrame) -> pd.DataFrame:
    available = events["execution_label_available"].fillna(False).astype(bool)
    result = events[available].copy()
    result["net_return"] = pd.to_numeric(
        result["net_return"], errors="coerce",
    )
    return result.dropna(subset=["net_return"])


def summarize_events(
    events: pd.DataFrame,
    *,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    selected = events[events["date"].between(start, end)].copy()
    valid = _valid_returns(selected)
    returns = valid["net_return"]
    gains = returns[returns.gt(0)].sort_values(ascending=False)
    losses = returns[returns.lt(0)]
    gain_sum = float(gains.sum())
    loss_sum = float(losses.sum())
    net_sum = float(returns.sum())
    code_contribution = valid.groupby("code")["net_return"].sum()
    industry_contribution = valid.groupby("industry")["net_return"].sum()
    positive_code = code_contribution[code_contribution.gt(0)].sort_values(
        ascending=False,
    )
    positive_industry = industry_contribution[
        industry_contribution.gt(0)
    ].sort_values(ascending=False)
    return {
        "阶段": period_name,
        "起始日": start.strftime("%Y-%m-%d"),
        "结束日": end.strftime("%Y-%m-%d"),
        "独立事件数": int(len(valid)),
        "不同股票数": int(valid["code"].nunique()),
        "不同行业数": int(valid["industry"].replace("", np.nan).nunique()),
        "平均净收益": float(returns.mean()) if len(returns) else None,
        "中位净收益": float(returns.median()) if len(returns) else None,
        "胜率": float(returns.gt(0).mean()) if len(returns) else None,
        "利润因子": (
            gain_sum / abs(loss_sum) if gain_sum > 0 and loss_sum < 0 else None
        ),
        "盈亏比": (
            float(gains.mean()) / abs(float(losses.mean()))
            if len(gains) and len(losses) else None
        ),
        "净收益简单求和": net_sum,
        "剔除Top1后净收益简单求和": (
            net_sum - float(gains.head(1).sum())
        ),
        "剔除Top3后净收益简单求和": (
            net_sum - float(gains.head(3).sum())
        ),
        "Top1占正收益比例": (
            float(gains.head(1).sum()) / gain_sum if gain_sum > 0 else None
        ),
        "Top3占正收益比例": (
            float(gains.head(3).sum()) / gain_sum if gain_sum > 0 else None
        ),
        "最大正贡献股票占比": (
            float(positive_code.head(1).sum()) / float(positive_code.sum())
            if len(positive_code) else None
        ),
        "最大正贡献行业占比": (
            float(positive_industry.head(1).sum())
            / float(positive_industry.sum())
            if len(positive_industry) else None
        ),
    }


def training_rule_passes(
    overall: dict,
    annual: list[dict],
) -> bool:
    return bool(
        overall["独立事件数"] >= 90
        and (overall["平均净收益"] or 0) > 0
        and (overall["利润因子"] or 0) >= 1.5
        and (overall["盈亏比"] or 0) >= 1.5
        and overall["剔除Top1后净收益简单求和"] > 0
        and overall["剔除Top3后净收益简单求和"] > 0
        and (overall["Top3占正收益比例"] or 1) < 0.5
        and len(annual) == 3
        and all((row["平均净收益"] or 0) > 0 for row in annual)
    )


def validation_rule_passes(metrics: dict) -> bool:
    return bool(
        metrics["独立事件数"] >= 50
        and (metrics["平均净收益"] or 0) > 0
        and (metrics["利润因子"] or 0) >= 1.2
        and (metrics["盈亏比"] or 0) >= 1.5
        and metrics["剔除Top1后净收益简单求和"] > 0
        and metrics["剔除Top3后净收益简单求和"] > 0
        and (metrics["Top3占正收益比例"] or 1) < 0.5
    )


def forward_rule_passes(metrics: dict) -> bool:
    return bool(
        metrics["独立事件数"] >= 50
        and (metrics["平均净收益"] or 0) > 0
        and (metrics["利润因子"] or 0) >= 1.5
        and (metrics["盈亏比"] or 0) >= 1.5
        and metrics["剔除Top1后净收益简单求和"] > 0
        and metrics["剔除Top3后净收益简单求和"] > 0
        and (metrics["Top3占正收益比例"] or 1) < 0.5
    )


def write_candidate_stream(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    output_directory: Path,
    sources: list[dict],
    industry_manifest: dict,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    execution_only_fields = set(EXECUTION_LABEL_FIELDS) - {
        "code", "playbook_factor_rank", "candidate_score",
    }
    excluded = set(RESEARCH_ONLY_FIELDS) | execution_only_fields
    decision_columns = [
        field for field in panel.columns if field not in excluded
    ]
    required_decision_fields = {"code", "playbook_factor_rank", "candidate_score"}
    missing = required_decision_fields - set(decision_columns)
    if missing:
        raise RuntimeError(f"候选决策字段被错误排除: {sorted(missing)}")
    event_fields = (
        "selection_event_id", "selection_reason_cn", "stock_cooldown_days",
        "industry_window_days", "max_industry_events",
    )
    snapshots = []
    for date in pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values():
        chosen = events[events["date"].eq(date)].copy()
        chosen["playbook_factor"] = FROZEN_FACTOR
        chosen["strategy_part"] = f"quantsplaybook:{FROZEN_FACTOR}"
        chosen["selection_engine"] = "执行一致因子周期研究"
        chosen["selection_profile"] = "20日独立事件与行业冷却"
        chosen["factor_horizon_rule_cn"] = FROZEN_RULE_CN
        chosen["factor_horizon_holding_days"] = 20
        chosen["factor_horizon_channel_eligible"] = True
        chosen["selected_for_trading"] = True
        chosen["signal_eligible"] = True
        chosen["allow_right"] = True
        chosen["allow_left"] = False
        columns = list(decision_columns)
        for field in (*event_fields, "factor_horizon_rule_cn",
                      "factor_horizon_holding_days",
                      "factor_horizon_channel_eligible"):
            if field not in columns:
                columns.append(field)
        chosen = chosen.reindex(columns=columns)
        chosen = chosen.sort_values(
            ["playbook_factor_rank", "code"], kind="mergesort",
        )
        path = output_directory / f"candidates_{date:%Y-%m-%d}.csv"
        _atomic_write_csv(chosen, path)
        snapshots.append({
            "date": date.strftime("%Y-%m-%d"),
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": int(len(chosen)),
            "signal_eligible_count": int(len(chosen)),
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        })
    source_manifest = json.loads(
        (Path(sources[0]["candidate_directory"]) / "manifest.json")
        .read_text(encoding="utf-8")
    )
    manifest = {
        "version": source_manifest.get("version"),
        "value_industry_rule_version": source_manifest.get(
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
        "industry_point_in_time": True,
        "unsafe_snapshot_count": 0,
        **{
            field: industry_manifest.get(field)
            for field in INDUSTRY_MANIFEST_FIELDS
        },
        "factor": FROZEN_FACTOR,
        "selection_engine": "quantsplaybook_execution_matched_research",
        "selection_profile": "20日独立事件与行业冷却",
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "factor_horizon_holding_days": 20,
        "frozen_rule_id": FROZEN_RULE_ID,
        "frozen_rule_cn": FROZEN_RULE_CN,
        "stock_cooldown_days": STOCK_COOLDOWN_DAYS,
        "industry_window_days": INDUSTRY_WINDOW_DAYS,
        "max_industry_events": MAX_INDUSTRY_EVENTS,
        "max_daily_events": MAX_DAILY_EVENTS,
        "training_period": "2021-01-01_to_2023-12-31",
        "validation_period": "2024-01-01_to_2024-12-31",
        "future_return_used_for_daily_candidate_selection": False,
        "execution_labels_written_to_candidates": False,
        "research_only": True,
        "source_manifests": sources,
        "snapshots": snapshots,
    }
    _atomic_write_json(manifest, output_directory / "manifest.json")
    return manifest


def _prepare(
    candidate_directories,
    formula_history,
    execution_labels,
    industry_manifest_path,
):
    panel, sources = load_execution_research_panel(
        candidate_directories, formula_history, execution_labels,
    )
    industry_history, industry_manifest = load_industry_history(
        industry_manifest_path,
    )
    panel = attach_point_in_time_industry(panel, industry_history)
    masks = rule_masks(panel)
    events = {
        name: select_independent_events(panel, mask)
        for name, mask in masks.items()
    }
    return panel, sources, industry_manifest, events


def freeze_and_validate(
    *,
    candidate_directories,
    formula_history,
    execution_labels,
    industry_manifest_path,
    output_root,
) -> dict:
    output = Path(output_root)
    research = output / "research"
    panel, sources, industry_manifest, events_by_rule = _prepare(
        candidate_directories, formula_history, execution_labels,
        industry_manifest_path,
    )
    rows = []
    passing = []
    for rule_name, events in events_by_rule.items():
        overall = summarize_events(
            events, period_name="训练期2021-2023",
            start=TRAINING_START, end=TRAINING_END,
        )
        annual = [
            summarize_events(
                events, period_name=f"训练年度{year}",
                start=pd.Timestamp(f"{year}-01-01"),
                end=pd.Timestamp(f"{year}-12-31"),
            )
            for year in (2021, 2022, 2023)
        ]
        passed = training_rule_passes(overall, annual)
        rows.append({"规则": rule_name, "通过训练门禁": passed, **overall})
        rows.extend({"规则": rule_name, "通过训练门禁": passed, **row}
                    for row in annual)
        if passed:
            passing.append(rule_name)
    expected_rule = "前10名且结构分大于40"
    if passing != [expected_rule]:
        raise RuntimeError(
            f"训练期未唯一冻结预声明规则: {passing} != [{expected_rule}]"
        )
    frozen_events = events_by_rule[expected_rule]
    validation = summarize_events(
        frozen_events, period_name="验证期2024",
        start=VALIDATION_START, end=VALIDATION_END,
    )
    validation_passed = validation_rule_passes(validation)
    _atomic_write_csv(pd.DataFrame(rows), research / "训练期预声明规则.csv")
    _atomic_write_json({
        "规则编号": FROZEN_RULE_ID,
        "规则中文": FROZEN_RULE_CN,
        "训练期唯一通过规则": expected_rule,
        "训练期使用执行一致标签": True,
        "验证期和样本外参与选参": False,
        "标签年度边界": "跨越信号自然年的开仓或退出样本均剔除",
    }, research / "冻结规则.json")
    _atomic_write_json({
        "验证通过": validation_passed,
        "验证指标": validation,
        "样本外是否已揭盲": False,
    }, research / "验证门禁.json")
    candidate_manifest = write_candidate_stream(
        panel, frozen_events,
        output_directory=output / "candidates" / FROZEN_FACTOR,
        sources=sources,
        industry_manifest=industry_manifest,
    )
    result = {
        "冻结规则": FROZEN_RULE_CN,
        "验证通过": validation_passed,
        "验证指标": validation,
        "候选事件数": int(sum(
            item["candidate_count"]
            for item in candidate_manifest["snapshots"]
        )),
        "候选目录": str(output / "candidates" / FROZEN_FACTOR),
    }
    _atomic_write_json(result, research / "冻结与验证摘要.json")
    return result


def reveal_forward(
    *,
    candidate_directories,
    formula_history,
    execution_labels,
    industry_manifest_path,
    output_root,
) -> dict:
    output = Path(output_root)
    gate_path = output / "research" / "验证门禁.json"
    frozen_path = output / "research" / "冻结规则.json"
    if not gate_path.is_file() or not frozen_path.is_file():
        raise RuntimeError("必须先完成训练冻结和2024验证")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("验证通过") is not True:
        raise RuntimeError("2024验证未通过，禁止揭盲样本外")
    panel, _, _, events_by_rule = _prepare(
        candidate_directories, formula_history, execution_labels,
        industry_manifest_path,
    )
    del panel
    metrics = summarize_events(
        events_by_rule["前10名且结构分大于40"],
        period_name="样本外2025-2026",
        start=FORWARD_START, end=FORWARD_END,
    )
    result = {
        "规则编号": FROZEN_RULE_ID,
        "规则中文": FROZEN_RULE_CN,
        "样本外通过候选标签门禁": forward_rule_passes(metrics),
        "样本外指标": metrics,
        "说明": "这是独立候选事件标签，不是完整组合收益率",
    }
    _atomic_write_json(result, output / "research" / "样本外揭盲.json")
    gate["样本外是否已揭盲"] = True
    _atomic_write_json(gate, gate_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "forward"), default="freeze")
    parser.add_argument(
        "--candidate-directory", action="append", dest="candidate_directories",
    )
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--execution-labels", default=str(DEFAULT_EXECUTION_LABELS))
    parser.add_argument(
        "--industry-manifest", default=str(DEFAULT_INDUSTRY_MANIFEST),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    call = freeze_and_validate if args.mode == "freeze" else reveal_forward
    result = call(
        candidate_directories=(
            args.candidate_directories or DEFAULT_CANDIDATE_DIRECTORIES
        ),
        formula_history=args.formula_history,
        execution_labels=args.execution_labels,
        industry_manifest_path=args.industry_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
