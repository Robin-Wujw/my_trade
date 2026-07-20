"""Independent monthly QVR selection and trading backtest.

This is a research-only control strategy.  It deliberately does not use the
production Formula33 phase, sector mainline, right-side structure entries, value
line grids, or the portfolio_backtest trade engine.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import load_candidate_snapshots, load_price_frames
from stock_research.core.paths import PATHS


DEFAULT_CANDIDATE_DIR = (
    PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"
)
DEFAULT_PRICE_DIR = PATHS.cache / "formula33_kline" / "akshare"


@dataclass
class QvrPosition:
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    stop_price: float
    target_weight: float
    highest_close: float
    rank: int
    trailing_half_done: bool = False
    peak_return: float = 0.0
    realized_cost: float = 0.0
    notes: list[str] = field(default_factory=list)


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _rank_pct(series: pd.Series, *, ascending: bool) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([math.nan] * len(series), index=series.index)
    return numeric.rank(pct=True, ascending=ascending) * 100.0


def _mean_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series([math.nan] * len(frame), index=frame.index)
    return frame[available].mean(axis=1, skipna=True)


def compute_qvr_scores(rows: list[dict]) -> list[dict]:
    """Score observation-day rows with an independent QVR factor recipe."""
    if not rows:
        return []
    frame = pd.DataFrame([dict(row) for row in rows])
    for column in (
        "quality_score", "earnings_yoy", "mktcap", "avg_amount_20",
        "price_to_value", "return_20d", "return_60d", "drawdown_60",
        "volatility_20", "downside_volatility_60", "range_21_pct",
        "known_volume_ratio", "right_acceleration", "momentum_60_ex5",
        "momentum_120_ex20",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "quality_score", "earnings_yoy", "mktcap", "avg_amount_20",
        "price_to_value", "return_20d", "return_60d", "drawdown_60",
        "volatility_20", "downside_volatility_60", "range_21_pct",
        "known_volume_ratio", "right_acceleration", "momentum_60_ex5",
        "momentum_120_ex20",
    ):
        if column not in frame:
            frame[column] = math.nan

    quality_component = pd.concat([
        _rank_pct(frame.get("quality_score"), ascending=True),
        _rank_pct(frame.get("earnings_yoy").clip(upper=1.0), ascending=True),
    ], axis=1).mean(axis=1, skipna=True)
    valuation_component = _rank_pct(frame.get("price_to_value"), ascending=False)
    risk_component = pd.concat([
        _rank_pct(frame.get("volatility_20"), ascending=False),
        _rank_pct(frame.get("downside_volatility_60"), ascending=False),
        _rank_pct(frame.get("drawdown_60"), ascending=True),
    ], axis=1).mean(axis=1, skipna=True)
    discipline_component = pd.concat([
        _rank_pct(frame.get("range_21_pct"), ascending=False),
        _rank_pct(frame.get("known_volume_ratio"), ascending=False),
        _rank_pct(frame.get("right_acceleration").abs(), ascending=False),
    ], axis=1).mean(axis=1, skipna=True)
    delayed_momentum_component = pd.concat([
        _rank_pct(frame.get("momentum_60_ex5"), ascending=True),
        _rank_pct(frame.get("momentum_120_ex20"), ascending=True),
    ], axis=1).mean(axis=1, skipna=True)

    frame["qvr_quality_score"] = quality_component
    frame["qvr_valuation_score"] = valuation_component
    frame["qvr_low_risk_score"] = risk_component
    frame["qvr_discipline_score"] = discipline_component
    frame["qvr_delayed_momentum_score"] = delayed_momentum_component
    frame["qvr_score"] = (
        quality_component * 0.30
        + valuation_component * 0.25
        + risk_component * 0.20
        + discipline_component * 0.15
        + delayed_momentum_component * 0.10
    )

    valid = (
        frame.get("code").astype(str).str.len().gt(0)
        & (frame.get("mktcap") >= 100.0)
        & (frame.get("avg_amount_20") >= 300_000_000.0)
        & (frame.get("quality_score") >= 55.0)
        & (frame.get("earnings_yoy") >= 0.0)
        & frame.get("price_to_value").gt(0.0)
        & frame.get("volatility_20").gt(0.0)
        & frame.get("drawdown_60").ge(-0.30)
        & frame.get("qvr_score").notna()
    )
    if "data_status" in frame:
        valid &= frame["data_status"].astype(str).eq("traded")
    if "valid_price_bar" in frame:
        valid &= frame["valid_price_bar"].map(_truthy)
    if "is_traded_bar" in frame:
        valid &= frame["is_traded_bar"].map(_truthy)
    overheat = (frame.get("return_20d") > 0.45) & (frame.get("known_volume_ratio") > 3.0)
    valid &= ~overheat.fillna(False)

    frame["qvr_selected_universe"] = valid
    frame["qvr_reject_reason"] = ""
    frame.loc[~valid, "qvr_reject_reason"] = "failed_qvr_hard_gate_or_missing_fields"
    scored = frame.to_dict("records")
    selected = [row for row in scored if bool(row.get("qvr_selected_universe"))]
    selected.sort(key=lambda row: (-float(row["qvr_score"]), str(row.get("code"))))
    for rank, row in enumerate(selected, start=1):
        row["qvr_rank"] = rank
    return scored


def select_monthly_qvr_snapshots(snapshots: dict, *, top_n: int = 20) -> dict[str, list[dict]]:
    """Return the last available snapshot in each month, scored and ranked."""
    by_month: dict[tuple[int, int], tuple[pd.Timestamp, list[dict]]] = {}
    for date, rows in snapshots.items():
        day = pd.Timestamp(date).normalize()
        key = (day.year, day.month)
        if key not in by_month or day > by_month[key][0]:
            by_month[key] = (day, rows)
    result = {}
    for day, rows in sorted(by_month.values(), key=lambda item: item[0]):
        scored = compute_qvr_scores(rows)
        selected = [
            row for row in scored
            if bool(row.get("qvr_selected_universe")) and int(row.get("qvr_rank", 999999)) <= top_n
        ]
        result[day.strftime("%Y-%m-%d")] = selected
    return result


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    previous_close = data["close"].shift(1)
    true_range = pd.concat([
        data["high"] - data["low"],
        (data["high"] - previous_close).abs(),
        (data["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data["atr20"] = true_range.rolling(20, min_periods=5).mean()
    return data


def _calendar(price_frames: dict[str, pd.DataFrame], start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = set()
    for frame in price_frames.values():
        if frame is None or frame.empty:
            continue
        series = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        dates.update(day for day in series if start <= day <= end)
    return sorted(dates)


def _bar_on_or_after(frame: pd.DataFrame, date: pd.Timestamp):
    data = frame[frame["date"] >= date]
    return None if data.empty else data.iloc[0]


def _bar_on(frame: pd.DataFrame, date: pd.Timestamp):
    data = frame[frame["date"] == date]
    return None if data.empty else data.iloc[0]


def _trade_fee(amount: float, *, rate: float, minimum: float) -> float:
    return max(float(minimum), abs(amount) * float(rate)) if amount else 0.0


def _portfolio_value(cash: float, positions: dict[str, QvrPosition], price_frames, date) -> float:
    value = cash
    for code, position in positions.items():
        bar = _bar_on(price_frames[code], date)
        if bar is not None:
            value += position.shares * float(bar["close"])
    return value


def _market_exposure_cap(rows: list[dict]) -> float:
    returns = pd.to_numeric(pd.Series([row.get("return_20d") for row in rows]), errors="coerce")
    median = returns.median()
    if pd.isna(median):
        return 0.80
    if median <= -0.12:
        return 0.0
    if median <= -0.08:
        return 0.40
    return 0.80


def _target_weights(selected: list[dict], max_total_exposure: float) -> dict[str, float]:
    top = selected[:5]
    if not top or max_total_exposure <= 0:
        return {}
    inverse_vol = {}
    for row in top:
        vol = max(_number(row.get("volatility_20"), 0.03), 0.005)
        inverse_vol[str(row["code"])] = 1.0 / vol
    total = sum(inverse_vol.values())
    weights = {
        code: min(0.20, max_total_exposure * value / total)
        for code, value in inverse_vol.items()
    }
    scale = min(1.0, max_total_exposure / sum(weights.values())) if sum(weights.values()) else 1.0
    return {code: weight * scale for code, weight in weights.items()}


def run_qvr_backtest(
    price_frames: dict[str, pd.DataFrame],
    monthly_snapshots: dict[str, list[dict]],
    *,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000.0,
    commission_rate: float = 0.000085,
    minimum_commission: float = 5.0,
    sell_stamp_duty_rate: float = 0.0005,
    estimated_slippage_rate: float = 0.0005,
) -> dict:
    frames = {code: _prepare_frame(frame) for code, frame in price_frames.items() if frame is not None}
    dates = _calendar(frames, start_date, end_date)
    if not dates:
        raise ValueError("no price dates available for QVR backtest")

    snapshots = {pd.Timestamp(date).normalize(): rows for date, rows in monthly_snapshots.items()}
    cash = float(initial_capital)
    positions: dict[str, QvrPosition] = {}
    pending_rebalance = None
    events = []
    equity_rows = []
    transaction_costs = 0.0

    def sell(code: str, date: pd.Timestamp, reason: str, fraction: float = 1.0):
        nonlocal cash, transaction_costs
        position = positions.get(code)
        if position is None:
            return
        frame = frames.get(code)
        bar = _bar_on_or_after(frame, date) if frame is not None else None
        if bar is None:
            events.append({"date": date.strftime("%Y-%m-%d"), "code": code, "action": "sell_blocked", "reason": "missing_bar"})
            return
        shares = int(position.shares * fraction // 100 * 100)
        if fraction >= 0.999:
            shares = position.shares
        if shares <= 0:
            return
        price = float(bar["open"]) * (1 - estimated_slippage_rate)
        amount = shares * price
        fee = _trade_fee(amount, rate=commission_rate, minimum=minimum_commission)
        stamp = amount * sell_stamp_duty_rate
        cash += amount - fee - stamp
        transaction_costs += fee + stamp
        position.shares -= shares
        pnl = (price - position.entry_price) * shares - fee - stamp
        events.append({
            "date": pd.Timestamp(bar["date"]).strftime("%Y-%m-%d"),
            "code": code,
            "name": position.name,
            "action": "sell",
            "shares": shares,
            "price": round(price, 4),
            "reason": reason,
            "pnl": round(pnl, 2),
        })
        if position.shares <= 0:
            positions.pop(code, None)

    def buy(code: str, row: dict, date: pd.Timestamp, target_weight: float, portfolio_value: float):
        nonlocal cash, transaction_costs
        frame = frames.get(code)
        bar = _bar_on_or_after(frame, date) if frame is not None else None
        if bar is None:
            events.append({"date": date.strftime("%Y-%m-%d"), "code": code, "action": "buy_blocked", "reason": "missing_bar"})
            return
        price = float(bar["open"]) * (1 + estimated_slippage_rate)
        current_value = 0.0
        if code in positions:
            current_value = positions[code].shares * price
        budget = max(0.0, portfolio_value * target_weight - current_value)
        affordable = max(0.0, cash - minimum_commission)
        shares = int(min(budget, affordable) / price // 100 * 100)
        if shares <= 0:
            return
        amount = shares * price
        fee = _trade_fee(amount, rate=commission_rate, minimum=minimum_commission)
        if amount + fee > cash:
            shares = int((cash - fee) / price // 100 * 100)
            amount = shares * price
            fee = _trade_fee(amount, rate=commission_rate, minimum=minimum_commission)
        if shares <= 0:
            return
        atr = _number(bar.get("atr20"), 0.0) or 0.0
        atr_pct = atr / float(bar["close"]) if float(bar["close"]) > 0 else 0.0
        stop_distance = max(0.12, 2.5 * atr_pct)
        cash -= amount + fee
        transaction_costs += fee
        if code in positions:
            position = positions[code]
            old_value = position.shares * position.entry_price
            new_value = shares * price
            position.entry_price = (old_value + new_value) / (position.shares + shares)
            position.shares += shares
            position.stop_price = max(position.stop_price, position.entry_price * (1 - stop_distance))
            position.target_weight = target_weight
            action = "rebalance_buy"
        else:
            positions[code] = QvrPosition(
                code=code,
                name=str(row.get("name") or code),
                shares=shares,
                entry_date=pd.Timestamp(bar["date"]).strftime("%Y-%m-%d"),
                entry_price=price,
                stop_price=price * (1 - stop_distance),
                target_weight=target_weight,
                highest_close=float(bar["close"]),
                rank=int(row.get("qvr_rank", 999999)),
            )
            action = "buy"
        events.append({
            "date": pd.Timestamp(bar["date"]).strftime("%Y-%m-%d"),
            "code": code,
            "name": str(row.get("name") or code),
            "action": action,
            "shares": shares,
            "price": round(price, 4),
            "target_weight": round(target_weight, 6),
            "qvr_rank": int(row.get("qvr_rank", 999999)),
            "qvr_score": round(float(row.get("qvr_score") or 0.0), 6),
        })

    for date in dates:
        if pending_rebalance is not None and date > pending_rebalance["snapshot_date"]:
            selected = pending_rebalance["selected"]
            top_by_code = {str(row["code"]): row for row in selected[:5]}
            target_weights = _target_weights(selected, pending_rebalance["exposure_cap"])
            before_value = _portfolio_value(cash, positions, frames, date)
            for code in list(positions):
                position = positions[code]
                target = target_weights.get(code, 0.0)
                current_bar = _bar_on_or_after(frames[code], date)
                current_price = float(current_bar["open"]) if current_bar is not None else position.entry_price
                current_weight = (position.shares * current_price / before_value) if before_value > 0 else 0.0
                if code not in target_weights:
                    sell(code, date, "monthly_top5_exit")
                elif target > 0 and current_weight > target * 1.5:
                    excess_fraction = max(0.0, min(1.0, (current_weight - target) / current_weight))
                    sell(code, date, "monthly_overweight_trim", excess_fraction)
            after_sells_value = _portfolio_value(cash, positions, frames, date)
            for code, row in top_by_code.items():
                buy(code, row, date, target_weights.get(code, 0.0), after_sells_value)
            pending_rebalance = None

        for code, position in list(positions.items()):
            bar = _bar_on(frames[code], date)
            if bar is None:
                continue
            close = float(bar["close"])
            position.highest_close = max(position.highest_close, close)
            position.peak_return = max(position.peak_return, position.highest_close / position.entry_price - 1.0)
            atr = _number(bar.get("atr20"), 0.0) or 0.0
            atr_pct = atr / close if close > 0 else 0.0
            if close <= position.stop_price:
                sell(code, date + pd.Timedelta(days=1), "initial_stop_next_open")
                continue
            if position.peak_return >= 0.40:
                trail = max(0.15, 4.0 * atr_pct)
                if close <= position.highest_close * (1 - trail):
                    sell(code, date + pd.Timedelta(days=1), "trailing_stop_final")
                    continue
            if position.peak_return >= 0.25 and not position.trailing_half_done:
                trail = max(0.10, 3.0 * atr_pct)
                if close <= position.highest_close * (1 - trail):
                    sell(code, date + pd.Timedelta(days=1), "trailing_stop_half", 0.5)
                    if code in positions:
                        positions[code].trailing_half_done = True

        if date in snapshots:
            selected = sorted(
                snapshots[date],
                key=lambda row: (int(row.get("qvr_rank", 999999)), str(row.get("code"))),
            )
            pending_rebalance = {
                "snapshot_date": date,
                "selected": selected,
                "exposure_cap": _market_exposure_cap(selected),
            }

        total_value = _portfolio_value(cash, positions, frames, date)
        equity_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": round(total_value, 2),
            "cash": round(cash, 2),
            "position_count": len(positions),
        })

    equity = pd.DataFrame(equity_rows)
    if equity.empty:
        raise ValueError("no equity rows produced")
    equity["running_max"] = equity["equity"].cummax()
    equity["drawdown_pct"] = (equity["equity"] / equity["running_max"] - 1.0) * 100.0
    final_equity = float(equity.iloc[-1]["equity"])
    sells = [event for event in events if event.get("action") == "sell"]
    winning_sells = [event for event in sells if _number(event.get("pnl"), 0.0) > 0]
    summary = {
        "strategy": "independent_qvr_monthly",
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "final_return_pct": round((final_equity / initial_capital - 1.0) * 100.0, 3),
        "maximum_drawdown_pct": round(float(equity["drawdown_pct"].min()), 3),
        "transaction_cost": round(transaction_costs, 2),
        "transaction_cost_pct": round(transaction_costs / initial_capital * 100.0, 3),
        "event_count": len(events),
        "buy_count": sum(1 for event in events if event.get("action") in {"buy", "rebalance_buy"}),
        "sell_count": len(sells),
        "sell_win_rate_pct": round(len(winning_sells) / len(sells) * 100.0, 3) if sells else None,
        "final_positions": [
            {
                "code": position.code,
                "name": position.name,
                "shares": position.shares,
                "entry_price": round(position.entry_price, 4),
                "stop_price": round(position.stop_price, 4),
                "peak_return_pct": round(position.peak_return * 100.0, 3),
            }
            for position in positions.values()
        ],
        "research_limitations": [
            "candidate snapshots are frozen point-in-time inputs, but they are not a full-market raw cross-section",
            "market risk filter uses candidate-median 20-day return when index history is not supplied",
            "daily bars execute close-confirmed exits at the next available open and cannot prove intraday queue priority",
        ],
    }
    return {"summary": summary, "events": events, "equity": equity.to_dict("records")}


def write_outputs(result: dict, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(result["events"]).to_csv(
        output_directory / "events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["equity"]).to_csv(
        output_directory / "equity.csv", index=False, encoding="utf-8-sig",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Independent QVR monthly strategy backtest")
    parser.add_argument("--start-date", default="2024-09-24")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--candidate-directory", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--price-kline-directory", default=str(DEFAULT_PRICE_DIR))
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "independent_qvr_monthly"),
    )
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.000085)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--estimated-slippage-rate", type=float, default=0.0005)
    parser.add_argument("--no-price-database", action="store_true")
    args = parser.parse_args(argv)

    raw_snapshots = load_candidate_snapshots(args.candidate_directory, args.start_date, args.end_date)
    monthly = select_monthly_qvr_snapshots(raw_snapshots)
    codes = {str(row["code"]) for rows in monthly.values() for row in rows}
    price_frames = load_price_frames(
        codes,
        Path(args.price_kline_directory),
        start_date=(pd.Timestamp(args.start_date) - pd.Timedelta(days=320)).strftime("%Y-%m-%d"),
        end_date=args.end_date,
        prefer_database=not args.no_price_database,
    )
    result = run_qvr_backtest(
        price_frames,
        monthly,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        sell_stamp_duty_rate=args.sell_stamp_duty_rate,
        estimated_slippage_rate=args.estimated_slippage_rate,
    )
    result["summary"]["candidate_directory"] = str(Path(args.candidate_directory).resolve())
    result["summary"]["price_kline_directory"] = str(Path(args.price_kline_directory).resolve())
    result["summary"]["monthly_snapshot_count"] = len(monthly)
    result["summary"]["candidate_code_count"] = len(codes)
    write_outputs(result, Path(args.output_directory))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
