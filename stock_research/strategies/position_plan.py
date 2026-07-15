"""Point-in-time entry/exit state machine for a single stock position."""
from __future__ import annotations

import pandas as pd

from stock_research.indicators.position_risk import position_exit_snapshot


def backtest_position_plan(
    frame: pd.DataFrame,
    start_date,
    *,
    value_line: float | None,
    left_levels: list[float] | None = None,
    left_part_pct: float = 0.03,
    left_grid_plan: list[dict] | None = None,
    right_position_pct: float = 0.15,
) -> dict:
    """Replay deterministic left/right rules without future-bar access.

    ``left_grid_plan`` is the authoritative grid input. Each row must provide
    ``buy_price``, ``sell_price`` and ``position_pct``; ``core`` is optional.
    The older value-line/level arguments remain as a derived compatibility
    mode and must not be presented as an author's prescribed grid.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["date", "high", "low", "close"]).sort_values("date").drop_duplicates("date")
    for period in (5, 10, 20, 60):
        data[f"ma{period}"] = data["close"].rolling(period).mean()
    start = pd.Timestamp(start_date).normalize()
    if left_grid_plan:
        grid_plan = []
        for item in left_grid_plan:
            buy = float(item["buy_price"])
            sell = float(item["sell_price"])
            size = float(item["position_pct"])
            if buy <= 0 or sell <= buy or size <= 0:
                raise ValueError("grid rows require sell_price > buy_price > 0 and position_pct > 0")
            grid_plan.append({
                "buy_price": buy,
                "sell_price": sell,
                "position_pct": size,
                "core": bool(item.get("core", False)),
            })
        grid_plan.sort(key=lambda item: item["buy_price"], reverse=True)
        if len({item["buy_price"] for item in grid_plan}) != len(grid_plan):
            raise ValueError("grid buy prices must be unique")
        grid_plan_source = "explicit"
    else:
        levels = (
            left_levels or [round(value_line - offset, 2) for offset in range(5)]
            if value_line is not None else []
        )
        step = abs(levels[0] - levels[1]) if len(levels) >= 2 else 1.0
        grid_plan = [
            {
                "buy_price": float(level),
                "sell_price": float(level + step),
                "position_pct": float(left_part_pct),
                "core": index == 0,
            }
            for index, level in enumerate(levels)
        ]
        grid_plan_source = "derived_compatibility"
    levels = [item["buy_price"] for item in grid_plan]
    plan_by_level = {item["buy_price"]: item for item in grid_plan}
    target_right_position_pct = float(right_position_pct)
    events = []
    position_pct = 0.0
    left_lots: dict[float, float] = {}
    grid_round_trips = {level: 0 for level in levels}
    right_entry_date = None
    right_cost = 0.0
    right_position = 0.0
    right_parts_remaining = 0
    right_condition_stop = None
    right_triggers_sold: set[str] = set()
    previous_left_confirmation = False

    def current_mode() -> str:
        if left_lots and right_position > 0:
            return "left_to_right"
        if left_lots:
            return "left"
        if right_position > 0:
            return "right"
        return "flat"

    def event(row, action, reason, price, change, lot_type):
        nonlocal position_pct
        position_pct = round(max(0.0, position_pct + change), 6)
        events.append({
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "action": action, "price": round(float(price), 3),
            "position_change_pct": round(change * 100, 2),
            "position_pct": round(position_pct * 100, 2),
            "mode": current_mode(), "lot_type": lot_type, "reason": reason,
        })

    for index, row in data.iterrows():
        if pd.Timestamp(row["date"]).normalize() < start:
            continue
        history = data.loc[:index]
        sold_right_today = False
        if right_position > 0:
            risk = position_exit_snapshot(
                history, right_cost, right_entry_date, entry_mode="right",
                condition_stop=right_condition_stop,
            )
            if risk["hard_stop_triggered"] or risk["entry_time_stop"]:
                action = risk["position_action"]
                event(row, action, action, row["close"], -right_position, "right")
                right_entry_date, right_cost = None, 0.0
                right_position, right_parts_remaining = 0.0, 0
                right_condition_stop, right_triggers_sold = None, set()
                sold_right_today = True
            else:
                active_trigger_ids = set(risk.get("take_profit_trigger_ids") or [])
                new_trigger_ids = active_trigger_ids - right_triggers_sold
                sell_parts = len(new_trigger_ids)
                sell_parts = min(sell_parts, right_parts_remaining)
                if sell_parts and right_parts_remaining:
                    change = -right_position * sell_parts / right_parts_remaining
                    event(row, f"分仓止盈{sell_parts}份", "浮盈回撤条件触发", row["close"], change, "right")
                    right_position = round(max(0.0, right_position + change), 6)
                    right_parts_remaining -= sell_parts
                    right_triggers_sold.update(sorted(new_trigger_ids)[:sell_parts])
                    sold_right_today = True
                    if right_parts_remaining == 0:
                        right_entry_date, right_cost = None, 0.0
                        right_position, right_condition_stop = 0.0, None
                        right_triggers_sold = set()

        sold_grid_levels = set()
        if left_lots:
            # Grid orders are planned in advance, so touching the limit price
            # is sufficient. A level sold today cannot be rebought from the
            # same OHLC bar because its intraday ordering is unknowable.
            for level in sorted(list(left_lots), reverse=True):
                plan = plan_by_level[level]
                if plan["core"]:
                    continue
                target = plan["sell_price"]
                if row["high"] >= target:
                    size = left_lots.pop(level)
                    event(row, "左侧网格卖出一层", f"触及预设卖价{target:.2f}", target, -size, "left_grid")
                    grid_round_trips[level] += 1
                    sold_grid_levels.add(level)

            for level in levels:
                if level not in left_lots and level not in sold_grid_levels and row["low"] <= level:
                    size = plan_by_level[level]["position_pct"]
                    left_lots[level] = size
                    event(row, "左侧买入一份", f"预设网格{level:.2f}", level, size, "left_grid")
            baseline = max(history["volume"].iloc[:-1].tail(5).mean(), history["volume"].iloc[:-1].tail(10).mean()) if len(history) >= 11 else None
            right_confirmed = (
                pd.notna(row["ma20"]) and row["close"] > row["ma20"]
                and history.iloc[-2]["close"] <= history.iloc[-2]["ma20"]
                and row["ma20"] > history.iloc[-6]["ma20"]
                and row["ma60"] > history.iloc[-6]["ma60"]
                and baseline is not None and row["volume"] > baseline
            )
            if right_confirmed and not previous_left_confirmation and right_position == 0 and not sold_right_today:
                left_position = sum(left_lots.values())
                add_pct = min(left_position / 2, 0.30 - position_pct)
                if add_pct > 0:
                    right_entry_date, right_cost = row["date"], float(row["close"])
                    right_position = add_pct
                    right_parts_remaining, right_triggers_sold = 3, set()
                    right_condition_stop = float(row["ma20"])
                    event(row, "左转右加仓", "放量收复MA20；新增批次独立止盈止损", row["close"], add_pct, "right")
            previous_left_confirmation = bool(right_confirmed)
            continue
        previous_left_confirmation = False

        if right_position > 0:
            continue

        # Flat: use objective trend pullbacks; generic 21-day high breakouts are
        # too aggressive to act as standalone entries.
        previous = history.iloc[:-1]
        if len(previous) < 60:
            for level in levels:
                if row["low"] <= level:
                    size = plan_by_level[level]["position_pct"]
                    left_lots[level] = size
                    event(row, "左侧买入一份", f"进入预设左侧网格{level:.2f}", level, size, "left_grid")
                    break
            continue
        pullback_level = float(previous.iloc[-1]["ma20"])
        right_pullback = (
            pd.notna(pullback_level) and previous.iloc[-1]["ma20"] > previous.iloc[-1]["ma60"]
            and previous.iloc[-1]["ma20"] > previous.iloc[-6]["ma20"]
            and previous.iloc[-1]["ma60"] > previous.iloc[-6]["ma60"]
            and previous.iloc[-1]["close"] > pullback_level
            and row["low"] <= pullback_level and row["close"] >= pullback_level
        )
        if right_pullback:
            right_entry_date, right_cost = row["date"], float(row["close"])
            right_position = target_right_position_pct
            right_parts_remaining, right_triggers_sold = 3, set()
            right_condition_stop = pullback_level
            reason = "上扬MA20/MA60结构回踩MA20"
            event(row, "右侧买入", reason, row["close"], right_position, "right")
            continue
        for level in levels:
            if row["low"] <= level:
                size = plan_by_level[level]["position_pct"]
                left_lots[level] = size
                event(row, "左侧买入一份", f"进入预设左侧网格{level:.2f}", level, size, "left_grid")
                break

    invested_value = sum(level * size for level, size in left_lots.items())
    invested_value += right_cost * right_position
    final_right_risk = (
        position_exit_snapshot(
            data, right_cost, right_entry_date, entry_mode="right",
            condition_stop=right_condition_stop,
        )
        if right_position > 0 else None
    )
    return {
        "events": events,
        "final_mode": current_mode(),
        "final_position_pct": round(position_pct * 100, 2),
        "final_cost": round(invested_value / position_pct, 3) if position_pct else None,
        "filled_left_levels": sorted(left_lots, reverse=True),
        "grid_round_trips": grid_round_trips,
        "grid_plan_source": grid_plan_source,
        "grid_plan": grid_plan,
        "right_position_pct": round(right_position * 100, 2),
        "right_risk": final_right_risk,
    }
