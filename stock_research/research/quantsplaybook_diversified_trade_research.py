"""拆解纯因子分散仓的真实批次现金流与训练期质量门。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.core.paths import PATHS


DEFAULT_RUN_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_hybrid_low_corr_value_diversified_intraday_stop_train_2021_2023"
    / "portfolio" / "hybrid_low_corr_value"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_diversified_trade_research_train_2021_2023"
)
BATCH_PATTERN = re.compile(r"^(R\d+)(?:\([^)]*\))?;")


def _number(value):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted) or not math.isfinite(float(converted)):
        return None
    return float(converted)


def _batch_id(reason) -> str:
    matched = BATCH_PATTERN.match(str(reason or "").strip())
    return matched.group(1) if matched else ""


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


def _event_frame(events_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(events_csv, dtype={"code": str})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    return frame


def _allocate_by_quantity(
    amount: float,
    allocations: list[tuple[dict, float]],
) -> None:
    total = sum(quantity for _, quantity in allocations)
    if total <= 0:
        return
    for batch, quantity in allocations:
        batch["net_pnl_amount"] += amount * quantity / total


def attribute_right_batch_cashflows(
    events: pd.DataFrame,
    final_positions: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Rebuild right-batch economics from fills, distributions and marks."""
    batches: dict[str, dict] = {}
    active: dict[str, list[dict]] = defaultdict(list)
    diagnostics = {
        "unattributed_sell_count": 0,
        "unattributed_sell_quantity": 0.0,
        "multi_batch_fifo_sell_count": 0,
        "corporate_action_adjustment_count": 0,
        "cash_distribution_count": 0,
    }

    for row in events.to_dict("records"):
        code = str(row.get("code") or "")
        side = str(row.get("trade_side") or "")
        action = str(row.get("action") or "")
        quantity = abs(_number(row.get("execution_quantity")) or 0.0)
        batch_id = _batch_id(row.get("reason"))

        if side == "买入" and action == "右侧买入":
            if not batch_id:
                raise RuntimeError(f"右侧买入缺少批次号: {row}")
            if batch_id in batches:
                raise RuntimeError(f"右侧批次号重复: {batch_id}")
            record = {
                "batch_id": batch_id,
                "entry_date": row["date"],
                "exit_date": pd.NaT,
                "entry_year": int(row["date"].year),
                "code": code,
                "name": str(row.get("name") or code),
                "signal_type": str(row.get("signal_type") or ""),
                "signal_type_cn": str(row.get("signal_type_cn") or "未标注"),
                "market_phase": str(row.get("phase") or "unknown"),
                "entry_risk_pct": _number(row.get("entry_risk_pct")),
                "planned_reward_r_multiple": _number(
                    row.get("planned_reward_r_multiple"),
                ),
                "entry_evidence_score": _number(
                    row.get("entry_evidence_score"),
                ),
                "candidate_snapshot_date": str(
                    row.get("candidate_snapshot_date") or ""
                ),
                "entry_quantity": quantity,
                "remaining_quantity": quantity,
                "buy_cashflow_amount": _number(
                    row.get("cash_change_amount"),
                ) or 0.0,
                "sell_cashflow_amount": 0.0,
                "cash_distribution_amount": 0.0,
                "ending_market_value": 0.0,
                "ending_unrealized_pnl": 0.0,
                "net_pnl_amount": _number(row.get("cash_change_amount")) or 0.0,
                "sell_fill_count": 0,
                "is_open": False,
            }
            batches[batch_id] = record
            active[code].append(record)
            continue

        if action == "除权持仓调整":
            before = _number(row.get("corporate_action_quantity_before"))
            after = _number(row.get("corporate_action_quantity_after"))
            lots = active.get(code, [])
            if before and after is not None and lots:
                scale = after / before
                for item in lots:
                    item["remaining_quantity"] *= scale
                scaled = sum(float(item["remaining_quantity"]) for item in lots)
                if scaled > 0 and abs(scaled - after) > 1e-6:
                    for item in lots:
                        item["remaining_quantity"] *= after / scaled
                diagnostics["corporate_action_adjustment_count"] += 1
            continue

        if action == "现金分红到账":
            amount = _number(row.get("corporate_action_cash_amount")) or 0.0
            lots = [
                item for item in active.get(code, [])
                if item["remaining_quantity"] > 1e-9
            ]
            allocations = [
                (item, float(item["remaining_quantity"])) for item in lots
            ]
            _allocate_by_quantity(amount, allocations)
            for item, quantity_part in allocations:
                total = sum(quantity for _, quantity in allocations)
                if total > 0:
                    item["cash_distribution_amount"] += (
                        amount * quantity_part / total
                    )
            diagnostics["cash_distribution_count"] += 1
            continue

        if side != "卖出" or quantity <= 0:
            continue

        lots = [
            item for item in active.get(code, [])
            if item["remaining_quantity"] > 1e-9
        ]
        if not lots and batch_id not in batches:
            diagnostics["ignored_non_factor_sell_count"] = (
                diagnostics.get("ignored_non_factor_sell_count", 0) + 1
            )
            continue
        if batch_id and batch_id in batches:
            target = batches[batch_id]
            lots = [target] if target["remaining_quantity"] > 1e-9 else []
        remaining = quantity
        allocations = []
        for item in lots:
            take = min(float(item["remaining_quantity"]), remaining)
            if take <= 0:
                continue
            allocations.append((item, take))
            item["remaining_quantity"] -= take
            item["sell_fill_count"] += 1
            item["exit_date"] = row["date"]
            remaining -= take
            if remaining <= 1e-6:
                break
        if len(allocations) > 1:
            diagnostics["multi_batch_fifo_sell_count"] += 1
        if remaining > 1e-6:
            diagnostics["unattributed_sell_count"] += 1
            diagnostics["unattributed_sell_quantity"] += remaining
        cashflow = _number(row.get("cash_change_amount")) or 0.0
        _allocate_by_quantity(cashflow, allocations)
        allocated_quantity = sum(value for _, value in allocations)
        for item, quantity_part in allocations:
            if allocated_quantity > 0:
                item["sell_cashflow_amount"] += (
                    cashflow * quantity_part / allocated_quantity
                )
        active[code] = [
            item for item in active.get(code, [])
            if item["remaining_quantity"] > 1e-6
        ]

    for position in final_positions or []:
        code = str(position.get("code") or "")
        open_batches = position.get("batches") or []
        if not open_batches:
            continue
        unrealized = _number(position.get("unrealized_pnl_amount")) or 0.0
        market_value = _number(position.get("market_value")) or 0.0
        quantities = [
            (_batch_id(f"{item.get('batch')};"), _number(item.get("quantity")) or 0.0)
            for item in open_batches
        ]
        left_quantity = sum(
            _number(item.get("quantity")) or 0.0
            for item in position.get("left_batches") or []
        )
        total_quantity = (
            sum(quantity for _, quantity in quantities) + left_quantity
        )
        for batch_id, quantity in quantities:
            item = batches.get(batch_id)
            if item is None or total_quantity <= 0:
                continue
            share = quantity / total_quantity
            item["ending_market_value"] += market_value * share
            item["ending_unrealized_pnl"] += unrealized * share
            item["net_pnl_amount"] += market_value * share
            item["is_open"] = True
            item["remaining_quantity"] = quantity
        if code not in active:
            diagnostics["unattributed_open_position_count"] = (
                diagnostics.get("unattributed_open_position_count", 0) + 1
            )

    frame = pd.DataFrame(batches.values())
    if frame.empty:
        return frame, diagnostics
    frame["net_pnl_amount"] = frame["net_pnl_amount"].round(2)
    frame["holding_calendar_days"] = (
        frame["exit_date"].fillna(events["date"].max()) - frame["entry_date"]
    ).dt.days
    frame["win"] = frame["net_pnl_amount"].gt(0)
    return frame.sort_values("entry_date").reset_index(drop=True), diagnostics


def attach_candidate_ranks(
    batches: pd.DataFrame,
    candidate_manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    manifest_path = Path(candidate_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_dates = set(batches["candidate_snapshot_date"].dropna().astype(str))
    rows = []
    files_read = 0
    for item in manifest.get("snapshots") or []:
        date = str(item.get("date") or "")
        if date not in requested_dates:
            continue
        path = manifest_path.parent / str(item["file"])
        expected = str(item.get("sha256") or "")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"候选快照哈希不一致: {path}")
        source_columns = [
            "code", "playbook_factor_rank", "playbook_factor_score",
            "candidate_score",
        ]
        frame = pd.read_csv(
            path, dtype={"code": str}, usecols=source_columns,
        ).copy()
        frame["candidate_snapshot_date"] = date
        rows.append(frame)
        files_read += 1
    candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    columns = [
        "candidate_snapshot_date", "code", "playbook_factor_rank",
        "playbook_factor_score", "candidate_score",
    ]
    missing = set(columns) - set(candidates.columns)
    if missing:
        raise RuntimeError(f"候选快照缺少研究字段: {sorted(missing)}")
    candidates = candidates.loc[:, columns].drop_duplicates(
        ["candidate_snapshot_date", "code"], keep="last",
    )
    result = batches.merge(
        candidates,
        on=["candidate_snapshot_date", "code"],
        how="left",
        validate="many_to_one",
    )
    coverage = int(result["playbook_factor_rank"].notna().sum())
    return result, {
        "candidate_snapshot_files_read": files_read,
        "factor_rank_coverage_count": coverage,
        "factor_rank_coverage_pct": round(
            coverage / len(result) * 100.0, 3,
        ) if len(result) else 0.0,
    }


def _bucket_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["factor_rank_bucket_cn"] = pd.cut(
        pd.to_numeric(result["playbook_factor_rank"], errors="coerce"),
        bins=[0, 10, 20, 30, 40, np.inf],
        labels=["因子前10名", "因子11至20名", "因子21至30名", "因子31至40名", "因子41名以后"],
    ).astype("string").fillna("因子排名缺失")
    result["risk_bucket_cn"] = pd.cut(
        pd.to_numeric(result["entry_risk_pct"], errors="coerce"),
        bins=[-np.inf, 2, 3, 4, 5, np.inf],
        labels=["风险距离不高于2%", "风险距离2%至3%", "风险距离3%至4%", "风险距离4%至5%", "风险距离高于5%"],
    ).astype("string").fillna("风险距离缺失")
    result["planned_rr_bucket_cn"] = pd.cut(
        pd.to_numeric(result["planned_reward_r_multiple"], errors="coerce"),
        bins=[-np.inf, 4, 6, 10, 15, np.inf],
        labels=["计划盈亏比不高于4", "计划盈亏比4至6", "计划盈亏比6至10", "计划盈亏比10至15", "计划盈亏比高于15"],
    ).astype("string").fillna("计划盈亏比缺失")
    result["market_phase_cn"] = result["market_phase"].map({
        "active": "右侧阶段活跃",
        "watch": "右侧阶段观察",
        "exited": "右侧阶段已退出",
        "waiting": "右侧阶段等待",
    }).fillna("市场阶段未知")
    return result


def summarize_group(frame: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(frame["net_pnl_amount"], errors="coerce").dropna()
    wins = pnl[pnl.gt(0)]
    losses = pnl[pnl.lt(0)]
    top1 = float(wins.max()) if len(wins) else 0.0
    return {
        "批次数": int(len(pnl)),
        "股票数": int(frame.loc[pnl.index, "code"].nunique()) if len(pnl) else 0,
        "净盈亏金额": round(float(pnl.sum()), 2),
        "平均每批盈亏": round(float(pnl.mean()), 2) if len(pnl) else None,
        "中位每批盈亏": round(float(pnl.median()), 2) if len(pnl) else None,
        "胜率_pct": round(float(pnl.gt(0).mean() * 100.0), 3) if len(pnl) else 0.0,
        "平均盈利": round(float(wins.mean()), 2) if len(wins) else None,
        "平均亏损": round(float(losses.mean()), 2) if len(losses) else None,
        "现金盈亏比": (
            round(float(wins.mean() / abs(losses.mean())), 3)
            if len(wins) and len(losses) and abs(float(losses.mean())) > 1e-12
            else None
        ),
        "现金利润因子": (
            round(float(wins.sum() / abs(losses.sum())), 3)
            if len(losses) and abs(float(losses.sum())) > 1e-12
            else None
        ),
        "第一盈利批次占正利润_pct": (
            round(top1 / float(wins.sum()) * 100.0, 3)
            if float(wins.sum()) > 0 else 0.0
        ),
        "剔除第一盈利批次后净盈亏": round(float(pnl.sum()) - top1, 2),
    }


def build_group_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = {
        "买入信号": "signal_type_cn",
        "因子排名": "factor_rank_bucket_cn",
        "风险距离": "risk_bucket_cn",
        "计划盈亏比": "planned_rr_bucket_cn",
        "市场阶段": "market_phase_cn",
    }
    rows = []
    periods = [("训练期合计", frame)] + [
        (f"{year}年", frame[frame["entry_year"].eq(year)])
        for year in sorted(frame["entry_year"].unique())
    ]
    for period_name, period in periods:
        for dimension_name, column in dimensions.items():
            for value, group in period.groupby(column, dropna=False, sort=True):
                rows.append({
                    "阶段": period_name,
                    "分组维度": dimension_name,
                    "分组值": str(value),
                    **summarize_group(group),
                })
    return pd.DataFrame(rows)


def _quality_gate_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    rank = pd.to_numeric(frame["playbook_factor_rank"], errors="coerce")
    risk = pd.to_numeric(frame["entry_risk_pct"], errors="coerce")
    planned = pd.to_numeric(
        frame["planned_reward_r_multiple"], errors="coerce",
    )
    platform = frame["signal_type_cn"].eq("整理平台突破")
    active = frame["market_phase"].eq("active")
    return {
        "全部分散仓买点": pd.Series(True, index=frame.index),
        "仅整理平台突破": platform,
        "排除右侧阶段已退出": frame["market_phase"].ne("exited"),
        "仅右侧阶段活跃": active,
        "仅因子前10名": rank.le(10),
        "仅因子前20名": rank.le(20),
        "风险距离不高于3%": risk.le(3),
        "风险距离2%至4%": risk.gt(2) & risk.le(4),
        "计划盈亏比4至10": planned.gt(4) & planned.le(10),
        "计划盈亏比不高于10": planned.le(10),
        "整理平台突破且因子前20名": platform & rank.le(20),
        "整理平台突破且右侧阶段活跃": platform & active,
        "右侧阶段活跃且因子前20名": active & rank.le(20),
    }


def build_quality_gate_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate_name, mask in _quality_gate_masks(frame).items():
        selected = frame[mask.fillna(False)]
        all_summary = summarize_group(selected)
        annual = {
            int(year): summarize_group(group)
            for year, group in selected.groupby("entry_year", sort=True)
        }
        annual_pnls = [
            float(annual.get(year, {}).get("净盈亏金额") or 0.0)
            for year in sorted(frame["entry_year"].unique())
        ]
        annual_counts = [
            int(annual.get(year, {}).get("批次数") or 0)
            for year in sorted(frame["entry_year"].unique())
        ]
        rows.append({
            "质量门": gate_name,
            **all_summary,
            "年度全为正": all(value > 0 for value in annual_pnls),
            "最差年度净盈亏": min(annual_pnls) if annual_pnls else None,
            "最少年度批次数": min(annual_counts) if annual_counts else 0,
            **{
                f"{year}年批次数": annual.get(year, {}).get("批次数", 0)
                for year in sorted(frame["entry_year"].unique())
            },
            **{
                f"{year}年净盈亏": annual.get(year, {}).get("净盈亏金额", 0.0)
                for year in sorted(frame["entry_year"].unique())
            },
        })
    return pd.DataFrame(rows).sort_values(
        ["年度全为正", "最差年度净盈亏", "净盈亏金额"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def run_research(
    summary_json: str | Path,
    events_csv: str | Path,
    candidate_manifest: str | Path,
    output_directory: str | Path,
) -> dict:
    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("diversified_factor_sleeve_enabled"):
        raise RuntimeError("输入回测未开启纯因子分散仓")
    events = _event_frame(events_csv)
    batches, attribution = attribute_right_batch_cashflows(
        events, summary.get("final_positions") or [],
    )
    batches, candidate_coverage = attach_candidate_ranks(
        batches, candidate_manifest,
    )
    batches = _bucket_columns(batches)
    profiles = build_group_profiles(batches)
    gates = build_quality_gate_table(batches)
    output = Path(output_directory)
    batch_path = output / "纯因子分散仓批次归因.csv"
    profile_path = output / "训练期逐年分组表现.csv"
    gate_path = output / "训练期简单质量门.csv"
    _atomic_write_csv(batches, batch_path)
    _atomic_write_csv(profiles, profile_path)
    _atomic_write_csv(gates, gate_path)
    payload = {
        "研究口径": "仅使用2021至2023训练期成交，不读取后续验证区间",
        "输入回测摘要": str(summary_path),
        "输入成交事件": str(events_csv),
        "输入候选清单": str(candidate_manifest),
        "回测最终收益_pct": summary.get("final_return_pct"),
        "回测最大回撤_pct": summary.get("maximum_drawdown_pct"),
        "右侧买入批次数": int(len(batches)),
        "批次净盈亏合计": round(float(batches["net_pnl_amount"].sum()), 2),
        "归因诊断": attribution,
        "候选字段覆盖": candidate_coverage,
        "年度全为正的简单质量门": gates.loc[
            gates["年度全为正"].fillna(False).astype(bool), "质量门"
        ].tolist(),
        "产物": {
            "批次归因": str(batch_path),
            "逐年分组表现": str(profile_path),
            "简单质量门": str(gate_path),
        },
    }
    _atomic_write_json(payload, output / "研究摘要.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="拆解纯因子分散仓训练成交并评估简单质量门",
    )
    parser.add_argument("--run-directory", default=str(DEFAULT_RUN_DIRECTORY))
    parser.add_argument("--summary-json")
    parser.add_argument("--events-csv")
    parser.add_argument("--candidate-manifest")
    parser.add_argument(
        "--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_directory = Path(args.run_directory)
    summaries = sorted(run_directory.glob("*_summary.json"))
    events = sorted(run_directory.glob("*_events.csv"))
    summary_path = Path(args.summary_json) if args.summary_json else (
        summaries[-1] if summaries else None
    )
    events_path = Path(args.events_csv) if args.events_csv else (
        events[-1] if events else None
    )
    if summary_path is None or events_path is None:
        raise RuntimeError(f"回测目录缺少摘要或事件文件: {run_directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_manifest = args.candidate_manifest or (
        summary.get("input_fingerprints") or {}
    ).get("candidate_manifest")
    if not candidate_manifest:
        raise RuntimeError("回测摘要缺少候选manifest路径")
    result = run_research(
        summary_path,
        events_path,
        candidate_manifest,
        args.output_directory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
