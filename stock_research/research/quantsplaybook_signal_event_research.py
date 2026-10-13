"""研究候选画像与技术买点匹配后的未来价格路径。"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.core.paths import PATHS
from stock_research.reporting.signal_text import (
    block_reason_text,
    signal_type_text,
)


DEFAULT_BACKTEST_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_hybrid_liquid_balanced_500m_train_2021_2023"
)
DEFAULT_FACTOR = "hybrid_liquid_balanced_500m_value"
DEFAULT_PRICE_CACHE = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721" / "prepared_prices"
)

PROFILE_FIELDS = (
    "playbook_factor",
    "playbook_factor_rank",
    "candidate_score",
    "trade_basis_score",
    "structure_proximity_score",
    "right_strength_score",
    "leadership_score",
    "return_20d",
    "return_60d",
    "return_120d",
    "distance_120d_high",
    "volatility_20",
    "drawdown_60",
    "ma20_slope",
    "ma60_slope",
    "right_acceleration",
    "avg_amount_20",
)


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_date(value) -> pd.Timestamp | None:
    date = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(date) else pd.Timestamp(date).normalize()


def prior_snapshot_date(
    event_date,
    snapshot_dates,
    explicit_date=None,
) -> pd.Timestamp | None:
    explicit = _normalize_date(explicit_date)
    if explicit is not None:
        return explicit
    event = _normalize_date(event_date)
    if event is None:
        return None
    dates = sorted(pd.Timestamp(value).normalize() for value in snapshot_dates)
    position = bisect_left(dates, event) - 1
    return dates[position] if position >= 0 else None


def label_future_path(
    prices: pd.DataFrame,
    event_date,
    *,
    trigger_price=None,
    stop_price=None,
    target_price=None,
) -> dict:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    date = _normalize_date(event_date)
    matches = frame.index[frame["date"].eq(date)]
    if date is None or matches.empty:
        return {"价格路径状态": "信号日无价格"}
    index = int(matches[-1])
    row = frame.iloc[index]
    entry = pd.to_numeric(row.get("raw_close", row.get("close")), errors="coerce")
    if pd.isna(entry) or float(entry) <= 0:
        return {"价格路径状态": "信号日价格无效"}
    entry = float(entry)
    result = {
        "价格路径状态": "可用",
        "信号日原始收盘价": entry,
        "信号日原始最低价": float(pd.to_numeric(
            row.get("raw_low", row.get("low")), errors="coerce",
        )),
    }
    future = frame.iloc[index + 1:].copy()
    raw_close = pd.to_numeric(
        future.get("raw_close", future.get("close")), errors="coerce",
    )
    raw_high = pd.to_numeric(
        future.get("raw_high", future.get("high")), errors="coerce",
    )
    raw_low = pd.to_numeric(
        future.get("raw_low", future.get("low")), errors="coerce",
    )
    trigger = pd.to_numeric(trigger_price, errors="coerce")
    if len(future) >= 1 and pd.notna(trigger):
        result["次日收盘仍站上触发价"] = bool(
            float(raw_close.iloc[0]) >= float(trigger)
        )
    else:
        result["次日收盘仍站上触发价"] = None
    for days in (5, 10, 20):
        if len(future) < days:
            result[f"未来{days}日收益"] = None
            result[f"未来{days}日最大有利波动"] = None
            result[f"未来{days}日最大不利波动"] = None
            continue
        window_close = raw_close.iloc[:days]
        window_high = raw_high.iloc[:days]
        window_low = raw_low.iloc[:days]
        result[f"未来{days}日收益"] = float(window_close.iloc[-1]) / entry - 1.0
        result[f"未来{days}日最大有利波动"] = float(window_high.max()) / entry - 1.0
        result[f"未来{days}日最大不利波动"] = float(window_low.min()) / entry - 1.0
    first_twenty_low = raw_low.iloc[:20]
    first_twenty_high = raw_high.iloc[:20]
    stop = pd.to_numeric(stop_price, errors="coerce")
    target = pd.to_numeric(target_price, errors="coerce")
    result["20日内触及初始止损"] = (
        bool(first_twenty_low.le(float(stop)).any())
        if pd.notna(stop) and len(first_twenty_low) else None
    )
    result["20日内触及计划目标"] = (
        bool(first_twenty_high.ge(float(target)).any())
        if pd.notna(target) and len(first_twenty_high) else None
    )
    signal_low = pd.to_numeric(
        row.get("raw_low", row.get("low")), errors="coerce",
    )
    proposed_stop = None
    if pd.notna(signal_low):
        proposed_stop = float(signal_low) * 0.995
        if pd.notna(trigger):
            proposed_stop = min(proposed_stop, float(trigger) * 0.995)
    proposed_risk = (
        entry - proposed_stop
        if proposed_stop is not None and proposed_stop < entry else None
    )
    result["信号日低点保护止损"] = proposed_stop
    result["信号日低点止损风险距离"] = (
        proposed_risk / entry if proposed_risk is not None else None
    )
    if proposed_risk is None or len(first_twenty_low) < 20:
        result["低点止损与3R目标20日结果"] = None
    else:
        proposed_target = entry + proposed_risk * 3.0
        outcome = "20日未触发"
        for low, high in zip(first_twenty_low, first_twenty_high):
            if float(low) <= proposed_stop:
                outcome = "先触及止损"
                break
            if float(high) >= proposed_target:
                outcome = "先达到3R目标"
                break
        result["低点止损与3R目标20日结果"] = outcome

    confirmed = result.get("次日收盘仍站上触发价") is True
    if not confirmed:
        result["次日收盘确认成交状态"] = "未确认"
        return result
    confirmed_entry = float(raw_close.iloc[0])
    result["次日收盘确认成交状态"] = "确认"
    result["次日确认原始收盘价"] = confirmed_entry
    result["次日确认相对触发价距离"] = (
        confirmed_entry / float(trigger) - 1.0 if pd.notna(trigger) else None
    )
    confirmed_risk = (
        confirmed_entry - proposed_stop
        if proposed_stop is not None and proposed_stop < confirmed_entry else None
    )
    result["次日确认止损风险距离"] = (
        confirmed_risk / confirmed_entry if confirmed_risk is not None else None
    )
    confirmed_future_close = raw_close.iloc[1:]
    confirmed_future_high = raw_high.iloc[1:]
    confirmed_future_low = raw_low.iloc[1:]
    for days in (5, 10, 20):
        if len(confirmed_future_close) < days:
            result[f"确认后未来{days}日收益"] = None
            continue
        result[f"确认后未来{days}日收益"] = (
            float(confirmed_future_close.iloc[days - 1]) / confirmed_entry - 1.0
        )
    if confirmed_risk is None or len(confirmed_future_low) < 20:
        result["次日确认后低点止损与3R目标20日结果"] = None
    else:
        confirmed_target = confirmed_entry + confirmed_risk * 3.0
        outcome = "20日未触发"
        for low, high in zip(
            confirmed_future_low.iloc[:20], confirmed_future_high.iloc[:20],
        ):
            if float(low) <= proposed_stop:
                outcome = "先触及止损"
                break
            if float(high) >= confirmed_target:
                outcome = "先达到3R目标"
                break
        result["次日确认后低点止损与3R目标20日结果"] = outcome
    return result


def _load_signal_events(summary_path: Path, trades_path: Path) -> pd.DataFrame:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for item in summary.get("entry_blocks") or []:
        row = dict(item)
        row.update({
            "review_status": "阻断",
            "review_reason_cn": block_reason_text(item.get("reason")),
            "accepted_priority": 0,
        })
        rows.append(row)
    trades = pd.read_csv(trades_path, dtype={"code": str})
    accepted = trades[
        trades["trade_side"].eq("买入")
        & trades["signal_type"].fillna("").ne("")
    ]
    for item in accepted.to_dict("records"):
        row = dict(item)
        row.update({
            "review_status": "已成交",
            "review_reason_cn": "通过",
            "accepted_priority": 1,
            "signal_trigger_raw_price": item.get("trigger_raw_price"),
            "signal_stop_raw_price": item.get("stop_raw_price"),
        })
        rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("回测没有可研究的技术信号事件")
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
    events = events.dropna(subset=["date", "code", "signal_type"])
    events = events.sort_values(
        ["date", "code", "signal_type", "accepted_priority"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(["date", "code", "signal_type"], keep="first")
    events["signal_cn"] = events["signal_type"].map(signal_type_text)
    return events.reset_index(drop=True)


def _snapshot_index(candidate_directory: Path) -> tuple[dict, dict]:
    manifest_path = candidate_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = {}
    for item in manifest.get("snapshots") or []:
        date = _normalize_date(item.get("date"))
        if date is not None:
            items[date] = item
    if not items:
        raise RuntimeError(f"候选清单没有快照: {manifest_path}")
    return manifest, items


def _attach_candidate_profiles(
    events: pd.DataFrame,
    candidate_directory: Path,
) -> pd.DataFrame:
    _, snapshots = _snapshot_index(candidate_directory)
    dates = sorted(snapshots)
    result = events.copy()
    result["snapshot_date"] = [
        prior_snapshot_date(
            row.date,
            dates,
            getattr(row, "candidate_snapshot_date", None),
        )
        for row in result.itertuples(index=False)
    ]
    joined = []
    for snapshot_date, group in result.groupby("snapshot_date", dropna=False):
        if pd.isna(snapshot_date) or snapshot_date not in snapshots:
            continue
        item = snapshots[snapshot_date]
        path = candidate_directory / str(item["file"])
        expected = str(item.get("sha256") or "")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"候选快照哈希不一致: {path}")
        candidates = pd.read_csv(path, dtype={"code": str})
        candidates = candidates[candidates.get("allow_right").map(_truthy)]
        columns = ["code"] + [field for field in PROFILE_FIELDS if field in candidates]
        merged = group.merge(
            candidates[columns], on="code", how="inner", suffixes=("_event", ""),
            validate="many_to_one",
        )
        joined.append(merged)
    if not joined:
        raise RuntimeError("技术信号事件无法匹配右侧候选快照")
    return pd.concat(joined, ignore_index=True)


def _attach_future_paths(events: pd.DataFrame, price_cache: Path) -> pd.DataFrame:
    frames = {}
    rows = []
    for item in events.to_dict("records"):
        code = str(item["code"])
        if code not in frames:
            path = price_cache / f"{code.replace('.', '_')}.pkl"
            frames[code] = pd.read_pickle(path) if path.exists() else pd.DataFrame()
        if frames[code].empty:
            path_metrics = {"价格路径状态": "缺少价格缓存"}
        else:
            path_metrics = label_future_path(
                frames[code],
                item["date"],
                trigger_price=item.get("signal_trigger_raw_price"),
                stop_price=item.get("signal_stop_raw_price"),
                target_price=item.get("entry_target_raw_price"),
            )
        rows.append({**item, **path_metrics})
    return pd.DataFrame(rows)


def _performance_row(name: str, group: pd.DataFrame) -> dict:
    returns = pd.to_numeric(group.get("未来20日收益"), errors="coerce").dropna()
    favorable = pd.to_numeric(
        group.get("未来20日最大有利波动"), errors="coerce",
    ).dropna()
    adverse = pd.to_numeric(
        group.get("未来20日最大不利波动"), errors="coerce",
    ).dropna()
    hold = group.get("次日收盘仍站上触发价", pd.Series(dtype=object)).dropna()
    path_outcomes = group.get(
        "低点止损与3R目标20日结果", pd.Series(dtype=object),
    ).dropna()
    return {
        "分组": name,
        "事件数": int(len(group)),
        "有效20日样本": int(len(returns)),
        "20日平均收益": float(returns.mean()) if len(returns) else None,
        "20日中位收益": float(returns.median()) if len(returns) else None,
        "20日胜率": float(returns.gt(0).mean()) if len(returns) else None,
        "20日强盈利率_不低于10%": (
            float(returns.ge(0.10).mean()) if len(returns) else None
        ),
        "20日尾部亏损率_低于负5%": (
            float(returns.lt(-0.05).mean()) if len(returns) else None
        ),
        "平均最大有利波动": float(favorable.mean()) if len(favorable) else None,
        "平均最大不利波动": float(adverse.mean()) if len(adverse) else None,
        "次日收盘确认率": float(hold.astype(bool).mean()) if len(hold) else None,
        "低点止损先达到3R率": (
            float(path_outcomes.eq("先达到3R目标").mean())
            if len(path_outcomes) else None
        ),
        "低点止损先止损率": (
            float(path_outcomes.eq("先触及止损").mean())
            if len(path_outcomes) else None
        ),
    }


def _group_metrics(events: pd.DataFrame, field: str) -> pd.DataFrame:
    return pd.DataFrame([
        _performance_row(str(name), group)
        for name, group in events.groupby(field, dropna=False, sort=False)
    ])


def _bucket_metrics(events: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "因子名次": (
            "playbook_factor_rank", [0, 10, 20, 30, 40, np.inf],
            ["1至10", "11至20", "21至30", "31至40", "41以后"],
        ),
        "候选日60日收益": (
            "return_60d", [-np.inf, -0.20, 0.0, 0.20, np.inf],
            ["低于负20%", "负20%至0", "0至20%", "不低于20%"],
        ),
        "候选日60日回撤": (
            "drawdown_60", [-np.inf, -0.25, -0.15, -0.08, np.inf],
            ["低于负25%", "负25%至负15%", "负15%至负8%", "不低于负8%"],
        ),
        "候选日20日波动": (
            "volatility_20", [-np.inf, 0.02, 0.03, 0.05, np.inf],
            ["低于2%", "2%至3%", "3%至5%", "不低于5%"],
        ),
        "候选日MA60斜率": (
            "ma60_slope", [-np.inf, 0.0, np.inf], ["下降", "上升"],
        ),
    }
    rows = []
    for label, (field, bins, names) in definitions.items():
        values = pd.to_numeric(events.get(field), errors="coerce")
        buckets = pd.cut(
            values, bins=bins, labels=names, include_lowest=True, right=True,
        )
        for name in names:
            group = events[buckets.eq(name)]
            row = _performance_row(str(name), group)
            row.update({"字段": label, "分桶": name})
            rows.append(row)
    return pd.DataFrame(rows)


def _detail_for_output(events: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "date": "日期",
        "code": "代码",
        "name": "名称",
        "signal_cn": "信号",
        "review_status": "回测处理",
        "review_reason_cn": "通过或阻断原因",
        "snapshot_date": "候选快照日期",
        "playbook_factor": "候选模型",
        "playbook_factor_rank": "因子名次",
        "candidate_score": "候选分",
        "trade_basis_score_event": "信号日交易基础分",
        "trade_basis_score": "候选日交易基础分",
        "return_20d": "候选日20日收益",
        "return_60d": "候选日60日收益",
        "return_120d": "候选日120日收益",
        "distance_120d_high": "距120日高点",
        "volatility_20": "候选日20日波动",
        "drawdown_60": "候选日60日回撤",
        "ma20_slope": "候选日MA20斜率",
        "ma60_slope": "候选日MA60斜率",
        "right_acceleration": "候选日右侧加速度",
        "avg_amount_20": "候选日20日均成交额",
        "entry_risk_pct": "计划风险距离_pct",
        "entry_reward_risk": "计划盈亏比",
        "信号日原始收盘价": "信号日原始收盘价",
        "信号日原始最低价": "信号日原始最低价",
        "signal_trigger_raw_price": "信号触发价",
        "signal_stop_raw_price": "原计划止损价",
        "次日收盘仍站上触发价": "次日收盘仍站上触发价",
        "未来5日收益": "未来5日收益",
        "未来10日收益": "未来10日收益",
        "未来20日收益": "未来20日收益",
        "未来20日最大有利波动": "未来20日最大有利波动",
        "未来20日最大不利波动": "未来20日最大不利波动",
        "20日内触及初始止损": "20日内触及初始止损",
        "20日内触及计划目标": "20日内触及计划目标",
        "信号日低点保护止损": "信号日低点保护止损",
        "信号日低点止损风险距离": "信号日低点止损风险距离",
        "低点止损与3R目标20日结果": "低点止损与3R目标20日结果",
        "次日收盘确认成交状态": "次日收盘确认成交状态",
        "次日确认原始收盘价": "次日确认原始收盘价",
        "次日确认相对触发价距离": "次日确认相对触发价距离",
        "次日确认止损风险距离": "次日确认止损风险距离",
        "确认后未来5日收益": "确认后未来5日收益",
        "确认后未来10日收益": "确认后未来10日收益",
        "确认后未来20日收益": "确认后未来20日收益",
        "次日确认后低点止损与3R目标20日结果": (
            "次日确认后低点止损与3R目标20日结果"
        ),
        "价格路径状态": "价格路径状态",
    }
    available = [column for column in columns if column in events]
    output = events[available].rename(columns=columns).copy()
    if "日期" in output:
        output["日期"] = pd.to_datetime(output["日期"]).dt.strftime("%Y-%m-%d")
    if "候选快照日期" in output:
        output["候选快照日期"] = pd.to_datetime(
            output["候选快照日期"], errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    return output


def run_signal_event_research(
    *,
    backtest_root: str | Path,
    factor: str,
    candidate_directory: str | Path,
    price_cache: str | Path,
    output_root: str | Path,
) -> dict:
    backtest_root = Path(backtest_root)
    portfolio = backtest_root / "portfolio" / factor
    summaries = sorted(portfolio.glob("*_summary.json"))
    trades = sorted(portfolio.glob("*_trades.csv"))
    if len(summaries) != 1 or len(trades) != 1:
        raise RuntimeError(f"组合回测摘要或成交文件不唯一: {portfolio}")
    events = _load_signal_events(summaries[0], trades[0])
    events = _attach_candidate_profiles(events, Path(candidate_directory))
    events = _attach_future_paths(events, Path(price_cache))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    signal_metrics = _group_metrics(events, "signal_cn")
    block_metrics = _group_metrics(events, "review_reason_cn")
    bucket_metrics = _bucket_metrics(events)
    _detail_for_output(events).to_csv(
        output_root / "信号事件明细.csv", index=False, encoding="utf-8-sig",
    )
    signal_metrics.to_csv(
        output_root / "信号类型统计.csv", index=False, encoding="utf-8-sig",
    )
    block_metrics.to_csv(
        output_root / "通过与阻断原因统计.csv", index=False, encoding="utf-8-sig",
    )
    bucket_metrics.to_csv(
        output_root / "候选画像分桶统计.csv", index=False, encoding="utf-8-sig",
    )
    manifest = {
        "version": 1,
        "factor": factor,
        "event_count": int(len(events)),
        "accepted_event_count": int(events["review_status"].eq("已成交").sum()),
        "blocked_event_count": int(events["review_status"].eq("阻断").sum()),
        "candidate_directory": str(candidate_directory),
        "price_cache": str(price_cache),
        "decision_fields_use_only_candidate_snapshot_or_signal_date": True,
        "future_path_fields_are_research_labels_only": True,
        "future_path_fields_used_for_historical_trade_decisions": False,
        "display_signal_language": "中文",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-root", default=str(DEFAULT_BACKTEST_ROOT))
    parser.add_argument("--factor", default=DEFAULT_FACTOR)
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--price-cache", default=str(DEFAULT_PRICE_CACHE))
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_signal_event_research(
        backtest_root=args.backtest_root,
        factor=args.factor,
        candidate_directory=args.candidate_directory,
        price_cache=args.price_cache,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
