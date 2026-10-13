"""Generate execution-aligned research labels for frozen factor candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from stock_research.research.quantsplaybook_compare import _sha256
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository
from stock_research.strategies.ohlc_execution import (
    fill_limit_order,
    fill_sell_stop,
    locked_limit_direction,
)


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
    / "quantsplaybook_execution_aligned_research_2021_to_20260721"
)
EXECUTION_LABEL_FIELDS = (
    "date",
    "code",
    "playbook_factor_rank",
    "candidate_score",
    "execution_label_available",
    "execution_status_cn",
    "entry_date",
    "entry_raw_price",
    "exit_date",
    "exit_raw_price",
    "exit_reason_cn",
    "holding_bars",
    "gross_return",
    "net_return",
    "realized_r_multiple",
    "label_role",
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


def load_candidate_snapshot_panel(
    candidate_directories,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load frozen decision snapshots without requiring legacy outcome labels."""
    panels = []
    sources = []
    for value in candidate_directories:
        directory = Path(value)
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
        panels.append(pd.concat(frames, ignore_index=True))
        sources.append({
            "candidate_directory": str(directory),
            "manifest_sha256": _sha256(manifest_path),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "factor": manifest.get("factor"),
        })
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce",
    ).dt.normalize()
    panel = panel.dropna(subset=["date", "code"])
    duplicates = panel.duplicated(["date", "code"], keep=False)
    if duplicates.any():
        sample = panel.loc[
            duplicates, ["date", "code"],
        ].head().to_dict("records")
        raise RuntimeError(f"candidate directories overlap: {sample}")
    for field in (
        "playbook_factor_rank", "candidate_score", "playbook_factor_score",
        "structure_proximity_score", "avg_amount_20",
    ):
        if field in panel:
            panel[field] = pd.to_numeric(panel[field], errors="coerce")
    return panel.sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    ).reset_index(drop=True), sources


def prepare_execution_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the portfolio engine's end-anchored qfq price convention."""
    data = frame.copy()
    factor_supplied = "raw_to_qfq_factor" in data.columns
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "raw_to_qfq_factor"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    data = data.reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        data[f"raw_{column}"] = data[column]
    factors = data["raw_to_qfq_factor"]
    data["_qfq_factor_observed"] = (
        factors.notna() & factors.gt(0)
        if factor_supplied
        else pd.Series(True, index=data.index)
    )
    valid_factors = factors.dropna()
    if not valid_factors.empty and float(valid_factors.iloc[-1]) > 0:
        anchor = float(valid_factors.iloc[-1])
        multiplier = anchor / factors
        # 复权因子只能从过去向未来延续。禁止 bfill，否则首段缺失会
        # 使用未来首个有效因子补齐历史，形成前视数据。
        multiplier = multiplier.where(multiplier > 0).ffill()
    else:
        multiplier = pd.Series(1.0, index=data.index)
    data["qfq_to_raw_factor"] = 1.0 / multiplier.where(multiplier > 0)
    for column in ("open", "high", "low", "close"):
        data[column] = data[f"raw_{column}"] * multiplier
    return data


def _raw_execution_bar(row: pd.Series) -> pd.Series:
    result = row.copy()
    for column in ("open", "high", "low", "close"):
        result[column] = row[f"raw_{column}"]
    return result


def _raw_price(row: pd.Series, column: str) -> float:
    return float(row[f"raw_{column}"])


def _signal_price(row: pd.Series, raw_price: float) -> float:
    factor = _number(row.get("qfq_to_raw_factor")) or 1.0
    return float(raw_price) / factor


def _raw_from_signal(row: pd.Series, signal_price: float) -> float:
    factor = _number(row.get("qfq_to_raw_factor")) or 1.0
    return float(signal_price) * factor


def _net_return(
    entry_signal_price: float,
    exit_signal_price: float,
    *,
    commission_rate: float,
    sell_stamp_duty_rate: float,
    slippage_rate: float,
) -> float:
    entry_cost = entry_signal_price * (commission_rate + slippage_rate)
    exit_cost = exit_signal_price * (
        commission_rate + sell_stamp_duty_rate + slippage_rate
    )
    return (
        exit_signal_price - entry_signal_price - entry_cost - exit_cost
    ) / entry_signal_price


def label_execution_path(
    frame: pd.DataFrame,
    signal_date,
    *,
    expected_entry_date=None,
    code: str = "",
    frame_is_prepared: bool = False,
    date_to_index: dict[pd.Timestamp, int] | None = None,
    holding_days: int = 20,
    stop_loss_pct: float = 0.08,
    commission_rate: float = 8.5e-5,
    sell_stamp_duty_rate: float = 5e-4,
    slippage_rate: float = 5e-4,
    same_calendar_year_only: bool = True,
) -> dict:
    """Label one T-close signal using the factor-horizon execution contract."""
    data = frame if frame_is_prepared else prepare_execution_frame(frame)
    signal = pd.to_datetime(signal_date, errors="coerce")
    if pd.isna(signal):
        return {
            "execution_label_available": False,
            "execution_status_cn": "信号日期无效",
        }
    signal = pd.Timestamp(signal).normalize()
    signal_index = (
        date_to_index.get(signal)
        if date_to_index is not None else None
    )
    if signal_index is None:
        signal_matches = data.index[data["date"].eq(signal)]
        signal_index = (
            int(signal_matches[-1]) if not signal_matches.empty else None
        )
    if signal_index is None:
        return {
            "execution_label_available": False,
            "execution_status_cn": "信号日无价格",
        }
    if expected_entry_date is None:
        entry_index = signal_index + 1
        if entry_index >= len(data):
            return {
                "execution_label_available": False,
                "execution_status_cn": "缺少下一交易日",
            }
    else:
        expected = pd.to_datetime(expected_entry_date, errors="coerce")
        if pd.isna(expected):
            return {
                "execution_label_available": False,
                "execution_status_cn": "预期成交日期无效",
            }
        expected = pd.Timestamp(expected).normalize()
        entry_index = (
            date_to_index.get(expected)
            if date_to_index is not None else None
        )
        if entry_index is None:
            entry_matches = data.index[data["date"].eq(expected)]
            entry_index = (
                int(entry_matches[-1]) if not entry_matches.empty else None
            )
        if entry_index is None:
            return {
                "execution_label_available": False,
                "execution_status_cn": "下一交易日无可用价格",
            }
        if entry_index <= signal_index:
            return {
                "execution_label_available": False,
                "execution_status_cn": "成交日未晚于信号日",
            }

    signal_row = data.iloc[signal_index]
    entry_row = data.iloc[entry_index]
    if (
        same_calendar_year_only
        and pd.Timestamp(entry_row["date"]).year != signal.year
    ):
        return {
            "execution_label_available": False,
            "execution_status_cn": "开仓跨越信号年度边界",
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
        }
    previous_close = _raw_price(signal_row, "close")
    entry_limit = _raw_price(entry_row, "open")
    entry_fill = fill_limit_order(
        _raw_execution_bar(entry_row),
        side="buy",
        limit_price=entry_limit,
        previous_close=previous_close,
        code=str(code),
        is_st=False,
    )
    if not entry_fill["filled"]:
        return {
            "execution_label_available": False,
            "execution_status_cn": "下一交易日开盘未成交",
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
            "entry_fill_status": entry_fill["status"],
        }

    entry_raw = float(entry_fill["price"])
    if (
        not bool(entry_row.get("_qfq_factor_observed", True))
        or _number(entry_row.get("qfq_to_raw_factor")) is None
    ):
        return {
            "execution_label_available": False,
            "execution_status_cn": "开仓日复权因子不完整",
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
            "entry_raw_price": entry_raw,
        }
    entry_signal = _signal_price(entry_row, entry_raw)
    stop_signal = entry_signal * (1.0 - float(stop_loss_pct))
    pending_exit_reason = None
    exit_row = None
    exit_raw = None
    exit_reason = None
    exit_status = None
    holding_bars = 0
    for index in range(entry_index, len(data)):
        row = data.iloc[index]
        if (
            same_calendar_year_only
            and pd.Timestamp(row["date"]).year != signal.year
        ):
            return {
                "execution_label_available": False,
                "execution_status_cn": "退出跨越信号年度边界",
                "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
                "entry_raw_price": entry_raw,
            }
        if (
            not bool(row.get("_qfq_factor_observed", True))
            or _number(row.get("qfq_to_raw_factor")) is None
        ):
            return {
                "execution_label_available": False,
                "execution_status_cn": "持有期间复权因子不完整",
                "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
                "entry_raw_price": entry_raw,
                "factor_missing_date": row["date"].strftime("%Y-%m-%d"),
            }
        holding_bars = index - entry_index + 1
        previous_raw_close = (
            _raw_price(data.iloc[index - 1], "close")
            if index > 0 else None
        )
        if index > entry_index and pending_exit_reason:
            pending_fill = fill_sell_stop(
                _raw_execution_bar(row),
                stop_price=float("inf"),
                previous_close=previous_raw_close,
                code=str(code),
                is_st=False,
            )
            if pending_fill["filled"]:
                exit_row = row
                exit_raw = float(pending_fill["price"])
                exit_reason = pending_exit_reason
                exit_status = pending_fill["status"]
                break
            continue
        if index > entry_index:
            stop_raw = _raw_from_signal(row, stop_signal)
            stop_fill = fill_sell_stop(
                _raw_execution_bar(row),
                stop_price=stop_raw,
                previous_close=previous_raw_close,
                code=str(code),
                is_st=False,
            )
            if stop_fill["filled"]:
                exit_row = row
                exit_raw = float(stop_fill["price"])
                exit_reason = "固定失效距离保护止损"
                exit_status = stop_fill["status"]
                break
            if _raw_price(row, "low") <= stop_raw:
                pending_exit_reason = "固定失效距离保护止损"
                continue
        if holding_bars >= int(holding_days):
            if locked_limit_direction(
                _raw_execution_bar(row),
                previous_raw_close,
                str(code),
                is_st=False,
            ) == "down":
                pending_exit_reason = f"固定持有{int(holding_days)}个交易日"
                continue
            exit_row = row
            exit_raw = _raw_price(row, "close")
            exit_reason = f"固定持有{int(holding_days)}个交易日"
            exit_status = "close_proxy"
            break

    if exit_row is None or exit_raw is None:
        return {
            "execution_label_available": False,
            "execution_status_cn": "标签窗口未完整结束",
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
            "entry_raw_price": entry_raw,
        }

    if (
        not bool(exit_row.get("_qfq_factor_observed", True))
        or _number(exit_row.get("qfq_to_raw_factor")) is None
    ):
        return {
            "execution_label_available": False,
            "execution_status_cn": "退出日复权因子不完整",
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
            "entry_raw_price": entry_raw,
            "exit_date": exit_row["date"].strftime("%Y-%m-%d"),
            "exit_raw_price": exit_raw,
        }
    exit_signal = _signal_price(exit_row, exit_raw)
    gross_return = exit_signal / entry_signal - 1.0
    net_return = _net_return(
        entry_signal,
        exit_signal,
        commission_rate=float(commission_rate),
        sell_stamp_duty_rate=float(sell_stamp_duty_rate),
        slippage_rate=float(slippage_rate),
    )
    return {
        "execution_label_available": True,
        "execution_status_cn": "可用",
        "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
        "entry_raw_price": entry_raw,
        "entry_fill_status": entry_fill["status"],
        "exit_date": exit_row["date"].strftime("%Y-%m-%d"),
        "exit_raw_price": exit_raw,
        "exit_fill_status": exit_status,
        "exit_reason_cn": exit_reason,
        "holding_bars": int(holding_bars),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "realized_r_multiple": float(net_return) / float(stop_loss_pct),
        "label_role": "仅供研究的执行一致结果标签，不得写入候选输入",
    }


def build_execution_aligned_labels(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    calendar_dates,
    *,
    holding_days: int = 20,
    stop_loss_pct: float = 0.08,
    same_calendar_year_only: bool = True,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_dates)).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    next_dates = {
        date: calendar[index + 1]
        for index, date in enumerate(calendar[:-1])
    }
    frames = {
        str(code): prepare_execution_frame(group)
        for code, group in prices.groupby("code", sort=False)
    }
    date_indices = {
        code: {
            pd.Timestamp(value).normalize(): int(index)
            for index, value in enumerate(frame["date"])
        }
        for code, frame in frames.items()
    }
    rows = []
    ordered = panel.sort_values(
        ["date", "playbook_factor_rank", "code"], kind="mergesort",
    )
    for item in ordered.to_dict("records"):
        date = pd.Timestamp(item["date"]).normalize()
        code = str(item["code"])
        frame = frames.get(code)
        if frame is None or frame.empty:
            label = {
                "execution_label_available": False,
                "execution_status_cn": "缺少股票价格序列",
            }
        elif date not in next_dates:
            label = {
                "execution_label_available": False,
                "execution_status_cn": "缺少下一市场交易日",
            }
        else:
            label = label_execution_path(
                frame,
                date,
                expected_entry_date=next_dates[date],
                code=code,
                frame_is_prepared=True,
                date_to_index=date_indices[code],
                holding_days=int(holding_days),
                stop_loss_pct=float(stop_loss_pct),
                same_calendar_year_only=bool(same_calendar_year_only),
            )
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "code": code,
            "playbook_factor_rank": item.get("playbook_factor_rank"),
            "candidate_score": item.get("candidate_score"),
            **label,
        })
    return pd.DataFrame(rows).reindex(columns=EXECUTION_LABEL_FIELDS)


def run_execution_aligned_research(
    candidate_directories,
    formula_history: str | Path,
    *,
    database_path: str | Path,
    output_root: str | Path,
    holding_days: int = 20,
    stop_loss_pct: float = 0.08,
) -> dict:
    panel, sources = load_candidate_snapshot_panel(candidate_directories)
    start = panel["date"].min().strftime("%Y-%m-%d")
    end = panel["date"].max().strftime("%Y-%m-%d")
    codes = sorted(panel["code"].astype(str).unique())
    repository = TushareRepository(Database(Path(database_path)))
    prices = repository.load_daily_kline_frames(
        codes, start_date=start, end_date=end,
    )
    formula = pd.read_csv(formula_history)
    calendar_dates = pd.to_datetime(
        formula["date"], errors="coerce",
    ).dropna()
    labels = build_execution_aligned_labels(
        panel,
        prices,
        calendar_dates,
        holding_days=int(holding_days),
        stop_loss_pct=float(stop_loss_pct),
    )
    output_root = Path(output_root)
    label_path = output_root / "执行一致候选结果.csv"
    _atomic_write_csv(labels, label_path)
    available = labels["execution_label_available"].fillna(False).astype(bool)
    status_counts = (
        labels.groupby("execution_status_cn", dropna=False).size()
        .sort_values(ascending=False).to_dict()
    )
    manifest = {
        "research_only": True,
        "selection_inputs_contain_future_returns": False,
        "execution_labels_written_back_to_candidates": False,
        "signal_timing": "T日收盘冻结，T+1开盘成交",
        "t_plus_one": True,
        "same_day_stop_disabled": True,
        "annual_label_boundary": "跨越信号自然年的开仓或退出标签均剔除",
        "holding_days": int(holding_days),
        "stop_loss_pct": float(stop_loss_pct),
        "commission_rate": 8.5e-5,
        "sell_stamp_duty_rate": 5e-4,
        "slippage_rate": 5e-4,
        "label_file": label_path.name,
        "label_sha256": _sha256(label_path),
        "label_count": int(len(labels)),
        "available_label_count": int(available.sum()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "candidate_sources": sources,
    }
    manifest_path = output_root / "研究清单.json"
    _atomic_write_json(manifest, manifest_path)
    return {
        "label_path": str(label_path),
        "manifest_path": str(manifest_path),
        "label_count": int(len(labels)),
        "available_label_count": int(available.sum()),
        "status_counts": manifest["status_counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-directory",
        action="append",
        dest="candidate_directories",
    )
    parser.add_argument(
        "--formula-history", default=str(DEFAULT_FORMULA_HISTORY),
    )
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--stop-loss-pct", type=float, default=0.08)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_execution_aligned_research(
        args.candidate_directories or DEFAULT_CANDIDATE_DIRECTORIES,
        args.formula_history,
        database_path=args.database_path,
        output_root=args.output_root,
        holding_days=args.holding_days,
        stop_loss_pct=args.stop_loss_pct,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
