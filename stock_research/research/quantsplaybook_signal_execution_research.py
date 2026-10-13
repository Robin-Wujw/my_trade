"""用真实执行口径检验技术信号与候选画像的匹配关系。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_execution_aligned_research import (
    label_execution_path,
    prepare_execution_frame,
)
from stock_research.research.quantsplaybook_signal_event_research import (
    DEFAULT_PRICE_CACHE,
    _attach_candidate_profiles,
    _load_signal_events,
)
from stock_research.core.paths import PATHS


DEFAULT_BACKTEST_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_hybrid_low_corr_value_diversified_intraday_stop_train_2021_2023"
)
DEFAULT_FACTOR = "hybrid_low_corr_value"
DEFAULT_CANDIDATE_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_candidate_profile_research_2021_to_20260721"
    / "candidates" / "playbook_low_corr"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_signal_execution_research_low_corr_train_2021_2023"
)
STOP_VARIANTS = (
    ("固定5%保护止损", 0.05),
    ("固定8%保护止损", 0.08),
)
NARROW_REASONS = {
    "止损距离过窄",
    "收盘确认买点的止损距离过窄",
}


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


def _market_dates(formula_history: str | Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(formula_history)
    date_column = "date" if "date" in frame else "trade_date"
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame[date_column], errors="coerce").dropna(),
    ).normalize()
    return dates.drop_duplicates().sort_values()


def attach_execution_labels(
    events: pd.DataFrame,
    price_cache: str | Path,
    market_dates,
    *,
    holding_days: int = 20,
    stop_variants=STOP_VARIANTS,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(pd.to_datetime(market_dates)).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    next_dates = {
        pd.Timestamp(date): pd.Timestamp(calendar[index + 1])
        for index, date in enumerate(calendar[:-1])
    }
    cache = Path(price_cache)
    frames = {}
    indices = {}
    rows = []
    for event in events.sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    ).to_dict("records"):
        code = str(event["code"])
        if code not in frames:
            path = cache / f"{code.replace('.', '_')}.pkl"
            raw = pd.read_pickle(path) if path.exists() else pd.DataFrame()
            prepared = (
                prepare_execution_frame(raw) if not raw.empty else raw
            )
            frames[code] = prepared
            indices[code] = {
                pd.Timestamp(value).normalize(): int(index)
                for index, value in enumerate(prepared.get("date", []))
            }
        signal_date = pd.Timestamp(event["date"]).normalize()
        for variant_name, stop_loss_pct in stop_variants:
            if frames[code].empty:
                label = {
                    "execution_label_available": False,
                    "execution_status_cn": "缺少股票价格序列",
                }
            elif signal_date not in next_dates:
                label = {
                    "execution_label_available": False,
                    "execution_status_cn": "缺少下一市场交易日",
                }
            else:
                label = label_execution_path(
                    frames[code],
                    signal_date,
                    expected_entry_date=next_dates[signal_date],
                    code=code,
                    frame_is_prepared=True,
                    date_to_index=indices[code],
                    holding_days=int(holding_days),
                    stop_loss_pct=float(stop_loss_pct),
                    same_calendar_year_only=True,
                )
            rows.append({
                **event,
                "执行变体": variant_name,
                "固定止损距离": float(stop_loss_pct),
                **label,
            })
    return pd.DataFrame(rows)


def select_independent_events(
    events: pd.DataFrame,
    market_dates,
    *,
    stock_cooldown_days: int = 20,
    max_daily_events: int = 5,
) -> pd.DataFrame:
    """Apply only observation-time ordering and trading-day cooldowns."""
    dates = pd.DatetimeIndex(pd.to_datetime(market_dates)).normalize()
    dates = dates.drop_duplicates().sort_values()
    positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    source = events.copy()
    source["playbook_factor_rank"] = pd.to_numeric(
        source["playbook_factor_rank"], errors="coerce",
    )
    source = source.sort_values(
        ["date", "playbook_factor_rank", "candidate_score", "code"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    last_by_code: dict[str, int] = {}
    selected = []
    for date, daily in source.groupby("date", sort=True):
        position = positions.get(pd.Timestamp(date).normalize())
        if position is None:
            continue
        daily_count = 0
        for row in daily.itertuples():
            code = str(row.code)
            if (
                position - last_by_code.get(code, -10_000)
                < int(stock_cooldown_days)
            ):
                continue
            selected.append(row.Index)
            last_by_code[code] = position
            daily_count += 1
            if daily_count >= int(max_daily_events):
                break
    return source.loc[selected].sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    ).reset_index(drop=True)


def _number(value):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted) or not math.isfinite(float(converted)):
        return None
    return float(converted)


def summarize_returns(events: pd.DataFrame) -> dict:
    valid = events[
        events["execution_label_available"].fillna(False).astype(bool)
    ].copy()
    returns = pd.to_numeric(valid["net_return"], errors="coerce").dropna()
    valid = valid.loc[returns.index]
    gains = returns[returns.gt(0)]
    losses = returns[returns.lt(0)]
    code_returns = valid.assign(_net=returns).groupby("code")["_net"].sum()
    positive_codes = code_returns[code_returns.gt(0)].sort_values(
        ascending=False,
    )
    top1 = float(positive_codes.iloc[0]) if len(positive_codes) else 0.0
    positive_total = float(positive_codes.sum())
    return {
        "独立事件数": int(len(returns)),
        "不同股票数": int(valid["code"].nunique()),
        "平均净收益": float(returns.mean()) if len(returns) else None,
        "中位净收益": float(returns.median()) if len(returns) else None,
        "胜率": float(returns.gt(0).mean()) if len(returns) else None,
        "平均盈利": float(gains.mean()) if len(gains) else None,
        "平均亏损": float(losses.mean()) if len(losses) else None,
        "盈亏比": (
            float(gains.mean() / abs(losses.mean()))
            if len(gains) and len(losses) and abs(float(losses.mean())) > 1e-12
            else None
        ),
        "利润因子": (
            float(gains.sum() / abs(losses.sum()))
            if len(losses) and abs(float(losses.sum())) > 1e-12
            else None
        ),
        "净收益合计": float(returns.sum()),
        "第一股票占正收益": (
            top1 / positive_total if positive_total > 0 else 0.0
        ),
        "剔除第一股票后净收益合计": float(returns.sum()) - top1,
    }


def _gate_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    narrow = events["review_reason_cn"].isin(NARROW_REASONS)
    return_60d = pd.to_numeric(events["return_60d"], errors="coerce")
    volatility = pd.to_numeric(events["volatility_20"], errors="coerce")
    return {
        "全部技术信号": pd.Series(True, index=events.index),
        "止损距离过窄信号": narrow,
        "止损距离过窄的整理平台突破": (
            narrow & events["signal_cn"].eq("整理平台突破")
        ),
        "止损距离过窄且60日跌幅超过20%": (
            narrow & return_60d.lt(-0.20)
        ),
        "止损距离过窄且20日波动为2%至5%": (
            narrow & volatility.ge(0.02) & volatility.lt(0.05)
        ),
    }


def build_gate_summary(
    labels: pd.DataFrame,
    market_dates,
) -> pd.DataFrame:
    rows = []
    years = sorted(pd.to_datetime(labels["date"]).dt.year.unique())
    for variant, variant_rows in labels.groupby("执行变体", sort=True):
        for gate_name, mask in _gate_masks(variant_rows).items():
            eligible = variant_rows[mask.fillna(False)]
            selected = select_independent_events(eligible, market_dates)
            total = summarize_returns(selected)
            annual = {}
            for year in years:
                annual[year] = summarize_returns(
                    selected[pd.to_datetime(selected["date"]).dt.year.eq(year)],
                )
            annual_means = [
                _number(annual[year].get("平均净收益")) for year in years
            ]
            annual_counts = [
                int(annual[year].get("独立事件数") or 0) for year in years
            ]
            rows.append({
                "执行变体": variant,
                "质量门": gate_name,
                **total,
                "逐年平均净收益均为正": all(
                    value is not None and value > 0 for value in annual_means
                ),
                "最差年度平均净收益": (
                    min(value for value in annual_means if value is not None)
                    if any(value is not None for value in annual_means) else None
                ),
                "最少年度独立事件数": min(annual_counts) if annual_counts else 0,
                **{
                    f"{year}年独立事件数": annual[year]["独立事件数"]
                    for year in years
                },
                **{
                    f"{year}年平均净收益": annual[year]["平均净收益"]
                    for year in years
                },
                **{
                    f"{year}年利润因子": annual[year]["利润因子"]
                    for year in years
                },
            })
    return pd.DataFrame(rows).sort_values(
        ["逐年平均净收益均为正", "最差年度平均净收益", "平均净收益"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def run_research(
    *,
    backtest_root: str | Path,
    factor: str,
    candidate_directory: str | Path,
    price_cache: str | Path,
    formula_history: str | Path,
    output_root: str | Path,
) -> dict:
    portfolio = Path(backtest_root) / "portfolio" / factor
    summaries = sorted(portfolio.glob("*_summary.json"))
    trades = sorted(portfolio.glob("*_trades.csv"))
    if len(summaries) != 1 or len(trades) != 1:
        raise RuntimeError(f"组合回测摘要或成交文件不唯一: {portfolio}")
    events = _load_signal_events(summaries[0], trades[0])
    events = _attach_candidate_profiles(events, Path(candidate_directory))
    market_dates = _market_dates(formula_history)
    labels = attach_execution_labels(events, price_cache, market_dates)
    summary = build_gate_summary(labels, market_dates)
    output = Path(output_root)
    _atomic_write_csv(labels, output / "执行一致信号标签.csv")
    _atomic_write_csv(summary, output / "训练期质量门逐年表现.csv")
    valid = summary[
        summary["逐年平均净收益均为正"].fillna(False).astype(bool)
        & summary["最少年度独立事件数"].ge(8)
    ]
    payload = {
        "研究口径": (
            "T日收盘信号、T+1开盘成交、买入当日不可卖、"
            "固定保护止损、20个交易日退出、跨年标签剔除"
        ),
        "事件总数": int(events.shape[0]),
        "执行标签总数": int(labels.shape[0]),
        "固定止损变体": [name for name, _ in STOP_VARIANTS],
        "独立事件规则": "单股20个市场交易日冷却，每日最多5个事件",
        "逐年为正且每年不少于8个事件的质量门": valid[
            ["执行变体", "质量门"]
        ].to_dict("records"),
        "未来标签用于历史决策": False,
        "展示信号语言": "中文",
    }
    _atomic_write_json(payload, output / "研究摘要.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-root", default=str(DEFAULT_BACKTEST_ROOT))
    parser.add_argument("--factor", default=DEFAULT_FACTOR)
    parser.add_argument(
        "--candidate-directory", default=str(DEFAULT_CANDIDATE_DIRECTORY),
    )
    parser.add_argument("--price-cache", default=str(DEFAULT_PRICE_CACHE))
    parser.add_argument(
        "--formula-history", default=str(DEFAULT_FORMULA_HISTORY),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_research(
        backtest_root=args.backtest_root,
        factor=args.factor,
        candidate_directory=args.candidate_directory,
        price_cache=args.price_cache,
        formula_history=args.formula_history,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
