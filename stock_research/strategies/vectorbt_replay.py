"""Independent vectorbt replay of orders emitted by the strategy engine.

The strategy engine remains responsible for point-in-time candidate selection,
price structures, batch state, and OHLC trigger semantics.  This module uses a
separate broker/accounting implementation to expose cash, fee, sizing, and
mark-to-market differences without duplicating those decisions.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _load_vectorbt():
    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "vectorbt cross-check requires: pip install -e .[vectorbt]"
        ) from exc
    return vbt


def _daily_closes(price_frames, dates, codes):
    closes = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=codes, dtype=float)
    for code in codes:
        frame = pd.DataFrame(price_frames[code]).copy()
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
        frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
        series = (
            frame.dropna(subset=["date", "close"])
            .drop_duplicates("date", keep="last")
            .set_index("date")["close"]
        )
        closes[code] = series.reindex(closes.index)
    # Missing bars usually mean suspension.  Carry the last tradable close for
    # valuation; bfill only affects dates before a symbol has any position.
    return closes.ffill().bfill()


def _build_timeline(price_frames, events, equity_curve):
    trade_events = [
        dict(event) for event in events
        if abs(float(event.get("execution_quantity") or 0.0)) > 0
    ]
    codes = sorted({str(event["code"]) for event in trade_events})
    missing = sorted(set(codes) - set(price_frames))
    if missing:
        raise ValueError(f"price frames missing vectorbt order symbols: {missing}")

    dates = {
        pd.Timestamp(row["date"]).normalize() for row in equity_curve
    }
    dates.update(pd.Timestamp(event["date"]).normalize() for event in trade_events)
    dates = sorted(dates)
    if not dates:
        return None

    daily_close = _daily_closes(price_frames, dates, codes)
    events_by_date = defaultdict(list)
    for event in trade_events:
        events_by_date[pd.Timestamp(event["date"]).normalize()].append(event)

    rows = []
    event_at = {}
    eod_rows = []
    for date in dates:
        for sequence, event in enumerate(events_by_date.get(date, [])):
            timestamp = date + pd.Timedelta(hours=9, minutes=30, microseconds=sequence)
            rows.append(timestamp)
            event_at[timestamp] = event
        timestamp = date + pd.Timedelta(hours=15)
        rows.append(timestamp)
        eod_rows.append(timestamp)

    index = pd.DatetimeIndex(rows)
    close = pd.DataFrame(index=index, columns=codes, dtype=float)
    for timestamp in index:
        close.loc[timestamp] = daily_close.loc[timestamp.normalize()].to_numpy()
    return close, event_at, pd.DatetimeIndex(eod_rows)


def run_vectorbt_cross_check(
    price_frames,
    result,
    *,
    commission_rate,
    minimum_commission,
    initial_capital,
    sell_stamp_duty_rate,
    estimated_slippage_rate,
):
    """Replay emitted fills in vectorbt and compare daily account values.

    This intentionally enforces shared cash.  Consequently, a strategy order
    can be partially filled when the original normalized-exposure engine has
    allocated 100% before paying fees.  Such a delta is a finding rather than
    something to hide in the adapter.
    """
    vbt = _load_vectorbt()
    built = _build_timeline(
        price_frames, result.get("events") or [], result.get("equity_curve") or [],
    )
    if built is None:
        return {
            "engine": "vectorbt",
            "vectorbt_version": vbt.__version__,
            "requested_order_count": 0,
            "filled_order_count": 0,
            "partial_fill_count": 0,
            "rejected_order_count": 0,
            "final_return_pct": 0.0,
            "maximum_drawdown_pct": 0.0,
            "total_fees": 0.0,
            "equity_curve": [],
        }

    close, event_at, eod_rows = built
    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    price = size.copy()
    fees = size.copy()
    fixed_fees = size.copy()
    log = pd.DataFrame(False, index=close.index, columns=close.columns)

    for timestamp, event in event_at.items():
        code = str(event["code"])
        quantity = float(event["execution_quantity"])
        execution_price = float(event.get("execution_price") or event["price"])
        turnover = abs(quantity * execution_price)
        # vectorbt computes percentage fee + fixed fee.  The dynamic fixed
        # component below reproduces max(turnover * 万0.85, 5), not 万0.85+5.
        minimum_top_up = max(
            0.0, float(minimum_commission) - turnover * float(commission_rate),
        )
        variable_rate = float(commission_rate) + float(estimated_slippage_rate)
        if quantity < 0:
            variable_rate += float(sell_stamp_duty_rate)
        size.loc[timestamp, code] = quantity
        price.loc[timestamp, code] = execution_price
        fees.loc[timestamp, code] = variable_rate
        fixed_fees.loc[timestamp, code] = minimum_top_up
        log.loc[timestamp, code] = True

    portfolio = vbt.Portfolio.from_orders(
        close,
        size=size,
        size_type="amount",
        direction="longonly",
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        init_cash=float(initial_capital),
        cash_sharing=True,
        group_by=True,
        allow_partial=True,
        log=log,
        freq="1D",
    )

    value = portfolio.value().reindex(eod_rows)
    cash = portfolio.cash().reindex(eod_rows)
    vector_equity = value / float(initial_capital)
    vector_drawdown = vector_equity / vector_equity.cummax() - 1.0
    current = pd.Series(
        {
            pd.Timestamp(row["date"]).normalize(): float(row["equity"])
            for row in result.get("equity_curve") or []
        },
        dtype=float,
    )

    comparison = []
    for timestamp in eod_rows:
        date = timestamp.normalize()
        current_equity = current.get(date, np.nan)
        checked_equity = float(vector_equity.loc[timestamp])
        comparison.append({
            "date": date.strftime("%Y-%m-%d"),
            "current_equity": None if pd.isna(current_equity) else float(current_equity),
            "vectorbt_equity": checked_equity,
            "equity_delta_pct": (
                None if pd.isna(current_equity)
                else (checked_equity - float(current_equity)) * 100.0
            ),
            "vectorbt_cash": float(cash.loc[timestamp]),
        })

    logs = portfolio.logs.records_readable
    order_issues = []
    if logs.empty:
        rejected = partial = 0
    else:
        rejected = int((logs["Result Status"] != "Filled").sum())
        filled_logs = logs[logs["Result Status"] == "Filled"]
        partial_mask = (
            (filled_logs["Result Size"].abs() - filled_logs["Request Size"].abs()).abs()
            > 1e-7
        )
        partial = int(partial_mask.sum())
        issue_logs = pd.concat([
            logs[logs["Result Status"] != "Filled"],
            filled_logs[partial_mask],
        ]).sort_values(["Timestamp", "Column"])
        for _, row in issue_logs.iterrows():
            order_issues.append({
                "timestamp": pd.Timestamp(row["Timestamp"]).isoformat(),
                "code": str(row["Column"]),
                "request_size": float(row["Request Size"]),
                "result_size": (
                    None if pd.isna(row["Result Size"])
                    else float(row["Result Size"])
                ),
                "status": str(row["Result Status"]),
                "status_info": str(row["Result Status Info"]),
            })
    deltas = [abs(row["equity_delta_pct"]) for row in comparison if row["equity_delta_pct"] is not None]
    final_equity = float(vector_equity.iloc[-1])
    final_timestamp = eod_rows[-1]
    final_assets = portfolio.assets().loc[final_timestamp]
    final_asset_values = portfolio.asset_value(group_by=False).loc[final_timestamp]
    original_rows = {
        str(row["code"]): dict(row) for row in result.get("final_positions") or []
    }
    final_positions = []
    for code in close.columns:
        quantity = float(final_assets.get(code, 0.0))
        asset_value = float(final_asset_values.get(code, 0.0))
        if abs(quantity) <= 1e-9 and code not in original_rows:
            continue
        weight_pct = asset_value / (final_equity * float(initial_capital)) * 100.0
        original_row = original_rows.get(code, {})
        direct_quantity = original_row.get("quantity")
        original_quantity = (
            float(direct_quantity) if direct_quantity is not None else 0.0
        )
        for lot in ([] if direct_quantity is not None else (original_row.get("right_batches") or [])):
            if lot.get("quantity") is not None:
                original_quantity += float(lot["quantity"])
            else:
                original_quantity += (
                    float(initial_capital) * float(lot.get("position_pct") or 0.0)
                    / 100.0 / float(lot["cost"])
                )
        left_batches = [] if direct_quantity is not None else (original_row.get("left_batches") or [])
        if not left_batches:
            levels = original_row.get("left_levels") or []
            left_pct = float(original_row.get("left_position_pct") or 0.0)
            if len(levels) == 1 and left_pct:
                left_batches = [{"position_pct": left_pct, "cost": levels[0]}]
        for lot in left_batches:
            if lot.get("quantity") is not None:
                original_quantity += float(lot["quantity"])
            else:
                original_quantity += (
                    float(initial_capital) * float(lot.get("position_pct") or 0.0)
                    / 100.0 / float(lot["cost"])
                )
        original_market_value = original_quantity * float(close.loc[final_timestamp, code])
        original_weight_pct = (
            original_market_value
            / ((1.0 + float(result.get("final_return_pct") or 0.0) / 100.0) * float(initial_capital))
            * 100.0
        )
        final_positions.append({
            "code": str(code),
            "quantity": round(quantity, 8),
            "close": round(float(close.loc[final_timestamp, code]), 6),
            "market_value": round(asset_value, 6),
            "account_weight_pct": round(weight_pct, 6),
            "original_model_quantity": round(original_quantity, 8),
            "quantity_delta": round(quantity - original_quantity, 8),
            "original_model_market_value": round(original_market_value, 6),
            "market_value_delta": round(asset_value - original_market_value, 6),
            "original_model_account_weight_pct": round(original_weight_pct, 6),
            "weight_delta_pct": round(weight_pct - original_weight_pct, 6),
        })
    return {
        "engine": "vectorbt",
        "vectorbt_version": vbt.__version__,
        "requested_order_count": len(event_at),
        "filled_order_count": int(portfolio.orders.count()),
        "partial_fill_count": partial,
        "rejected_order_count": rejected,
        "order_issues": order_issues,
        "final_return_pct": round((final_equity - 1.0) * 100.0, 6),
        "current_final_return_pct": float(result.get("final_return_pct") or 0.0),
        "final_return_delta_pct": round(
            (final_equity - 1.0) * 100.0 - float(result.get("final_return_pct") or 0.0),
            6,
        ),
        "maximum_drawdown_pct": round(float(vector_drawdown.min()) * 100.0, 6),
        "current_maximum_drawdown_pct": float(result.get("maximum_drawdown_pct") or 0.0),
        "max_abs_daily_equity_delta_pct": round(max(deltas, default=0.0), 6),
        "max_abs_daily_equity_delta_date": (
            max(
                (row for row in comparison if row["equity_delta_pct"] is not None),
                key=lambda row: abs(row["equity_delta_pct"]),
                default={"date": None},
            )["date"]
        ),
        "total_fees": round(float(portfolio.orders.fees.sum()), 6),
        "final_cash": round(float(cash.iloc[-1]), 6),
        "final_cash_pct": round(float(cash.iloc[-1]) / (final_equity * float(initial_capital)) * 100.0, 6),
        "final_positions": final_positions,
        "equity_curve": comparison,
    }
