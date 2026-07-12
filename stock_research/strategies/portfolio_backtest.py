"""Point-in-time portfolio replay over dated selection snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stock_research.indicators.position_risk import position_exit_snapshot
from stock_research.strategies.ohlc_execution import fill_buy_stop, fill_limit_order, fill_sell_stop


@dataclass
class PositionState:
    right: list[dict] = field(default_factory=list)
    right_parts: int = 5
    right_sold: set[str] = field(default_factory=set)
    right_plan_date: object | None = None
    left_lots: dict[float, float] = field(default_factory=dict)
    left_peaks: dict[float, float] = field(default_factory=dict)
    left_plan: dict | None = None
    previous_left_confirmation: bool = False


def build_formula_phase_history(formula_rows) -> dict[str, str]:
    frame = pd.DataFrame(formula_rows).copy()
    if frame.empty or "date" not in frame:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    phase = "waiting"
    result = {}

    def streak(value):
        number = pd.to_numeric(value, errors="coerce")
        return 0 if pd.isna(number) else int(number)

    for _, row in frame.iterrows():
        up = streak(row.get("window_up_streak"))
        down = streak(row.get("window_down_streak"))
        if down >= 5:
            phase = "exited"
        elif up >= 5:
            phase = "active"
        elif up >= 3 and phase != "active":
            phase = "watch"
        result[row["date"].strftime("%Y-%m-%d")] = phase
    return result


def _prepare_frame(frame):
    data = frame.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["date", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for period in (5, 10, 20, 60):
        data[f"ma{period}"] = data["close"].rolling(period).mean()
    return data


def _enabled(value, *, default=True):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _right_signal(data, index):
    if index < 60:
        return None
    row = data.iloc[index]
    previous = data.iloc[:index]
    baseline = max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )
    local_trigger = float(previous["close"].tail(21).max())
    local_breakout = bool(
        previous.iloc[-1]["close"] <= local_trigger
        and row["high"] > local_trigger
        and row["volume"] >= baseline
        and local_trigger >= previous.iloc[-1]["ma20"]
    )
    pullback_level = float(previous.iloc[-1]["ma20"])
    pullback = bool(
        previous.iloc[-1]["ma20"] > previous.iloc[-1]["ma60"]
        and previous.iloc[-1]["ma20"] > previous.iloc[-6]["ma20"]
        and previous.iloc[-1]["ma60"] > previous.iloc[-6]["ma60"]
        and previous.iloc[-1]["close"] > pullback_level
        and row["low"] < pullback_level
    )
    if local_breakout:
        return {"rank": 2, "stop": local_trigger, "trigger": local_trigger, "order_type": "stop", "reason": "放量突破21日收盘高点", "volume_ratio": float(row["volume"]) / baseline}
    if pullback:
        return {"rank": 1, "stop": pullback_level, "trigger": pullback_level, "order_type": "limit", "reason": "上扬MA20/MA60结构回踩20日均线", "volume_ratio": float(row["volume"]) / baseline}
    return None


def run_portfolio_backtest(
    price_frames,
    candidate_snapshots,
    formula_phases,
    *,
    requested_start,
    end_date,
    trade_plans=None,
    max_positions=3,
    exit_tail_on_candidate_removal=False,
    signals_effective_next_day=False,
):
    """Replay a portfolio without filling dates before candidate coverage."""
    snapshots = {
        pd.Timestamp(date).normalize(): {str(item["code"]): dict(item) for item in rows}
        for date, rows in candidate_snapshots.items()
    }
    if not snapshots:
        raise ValueError("candidate snapshot history is empty")
    coverage_start = min(snapshots)
    requested = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(end_date).normalize()
    start = max(requested, coverage_start)
    frames = {code: _prepare_frame(frame) for code, frame in price_frames.items()}
    plans = dict((trade_plans or {}).get("plans") or {})
    states = {code: PositionState() for code in frames}
    events = []
    realized = 0.0
    equity_curve = []
    current_candidates = {}
    current_phase = "waiting"
    current_down_streak = 0
    current_up_streak = 0
    next_batch_id = 1
    previous_candidate_codes = set()
    pending_candidate_exits = set()
    pending_weak_exits = set()

    candidate_names = {
        str(item["code"]): str(item.get("name") or "")
        for rows in candidate_snapshots.values()
        for item in rows
    }

    def is_st(code):
        return "ST" in candidate_names.get(code, "").upper()

    def occupied_codes():
        return {code for code, state in states.items() if state.right or state.left_lots}

    def gross_exposure():
        return sum(
            sum(float(lot["size"]) for lot in state.right)
            + sum(state.left_lots.values())
            for state in states.values()
        )

    def symbol_limit_reached():
        configured = 5 if max_positions is None or int(max_positions) <= 0 else min(5, int(max_positions))
        return len(occupied_codes()) >= configured

    def right_codes():
        return {code for code, state in states.items() if state.right}

    def has_twenty_percent_float(date):
        for code in occupied_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty:
                continue
            close = float(visible.iloc[-1]["close"])
            if any(close / lot["cost"] - 1 >= 0.20 for lot in state.right):
                return True
            if any(close / buy - 1 >= 0.20 for buy in state.left_lots):
                return True
        return False

    def right_symbol_returns(date):
        returns = []
        for code in right_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty or not state.right:
                continue
            close = float(visible.iloc[-1]["close"])
            cost = sum(lot["size"] * lot["cost"] for lot in state.right) / sum(
                lot["size"] for lot in state.right
            )
            returns.append(close / cost - 1)
        return sorted(returns, reverse=True)

    def can_open_new_right_symbol(date):
        returns = right_symbol_returns(date)
        if not returns:
            return True
        if current_up_streak >= 3:
            return len(returns) < 3 and all(value >= 0.10 for value in returns)
        if len(returns) == 1:
            return returns[0] >= 0.20
        return len(returns) < 3 and returns[0] >= 0.20 and returns[1] >= 0.10

    def add_event(date, code, action, price, size, reason, pnl=0.0):
        nonlocal realized
        realized += pnl
        events.append({
            "date": date.strftime("%Y-%m-%d"), "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "action": action, "price": round(float(price), 3),
            "position_change_pct": round(size * 100, 2),
            "reason": reason, "phase": current_phase,
            "realized_account_pct": round(pnl * 100, 4),
        })

    calendar = sorted({
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[(frame["date"] >= start) & (frame["date"] <= end), "date"]
    })
    snapshot_dates = sorted(snapshots)
    formula_by_date = {
        pd.Timestamp(date).normalize(): value
        for date, value in formula_phases.items()
    }
    formula_dates = sorted(formula_by_date)
    for date in calendar:
        exited_today = set()
        for code, state in states.items():
            pending_lots = [
                lot for lot in state.right
                if lot.get("reconfirm_on_next_day") and pd.Timestamp(lot["date"]).normalize() < date
            ]
            if not pending_lots:
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            for lot in pending_lots:
                level = float(lot["reconfirm_level"])
                if float(row["open"]) >= level:
                    lot["reconfirm_on_next_day"] = False
                    add_event(date, code, "突破次日确认持有", row["open"], 0.0, f"{lot['batch']}; 开盘仍在{level:.3f}上方")
                    continue
                fill = fill_sell_stop(
                    row, stop_price=float("inf"), previous_close=previous_close,
                    code=code, is_st=is_st(code),
                )
                if not fill["filled"]:
                    continue
                size = float(lot["size"])
                sell = float(fill["price"])
                pnl = size * (sell / float(lot["cost"]) - 1)
                state.right.remove(lot)
                add_event(date, code, "突破次日未确认退出", sell, -size, f"{lot['batch']}; 开盘未站上{level:.3f}; {fill['status']}", pnl)
            if not state.right:
                state.right_parts = 5
                state.right_sold.clear()
                state.right_plan_date = None
        for code in list(pending_candidate_exits):
            state = states.get(code)
            right_size = 0.0 if state is None else sum(lot["size"] for lot in state.right)
            if state is None or not state.right or right_size >= 0.10 - 1e-9:
                pending_candidate_exits.discard(code)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                size = lot["size"]
                pnl = size * (sell / lot["cost"] - 1)
                state.right.remove(lot)
                add_event(date, code, "落选尾仓退出", sell, -size, f"{lot['batch']}; 右侧总仓低于10%; 次日开盘退出", pnl)
            state.right_parts = 5
            state.right_sold.clear()
            state.right_plan_date = None
            pending_candidate_exits.discard(code)
            exited_today.add(code)
        for code in list(pending_weak_exits):
            state = states.get(code)
            if state is None or not state.right:
                pending_weak_exits.discard(code)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                size = lot["size"]
                pnl = size * (sell / lot["cost"] - 1)
                state.right.remove(lot)
                add_event(date, code, "汰弱退出", sell, -size, f"{lot['batch']}; 低效率仓次日开盘退出", pnl)
            state.right_parts = 5
            state.right_sold.clear()
            state.right_plan_date = None
            pending_weak_exits.discard(code)
            exited_today.add(code)
        eligible_snapshots = [
            item for item in snapshot_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        if eligible_snapshots:
            current_candidates = snapshots[eligible_snapshots[-1]]
        current_candidate_codes = set(current_candidates)
        if exit_tail_on_candidate_removal and previous_candidate_codes:
            for code in set(states) - current_candidate_codes:
                state = states.get(code)
                if state and state.right and sum(lot["size"] for lot in state.right) < 0.10 - 1e-9:
                    pending_candidate_exits.add(code)
        previous_candidate_codes = current_candidate_codes
        eligible_formula_dates = [
            item for item in formula_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        formula_state = (
            formula_by_date[eligible_formula_dates[-1]]
            if eligible_formula_dates else current_phase
        )
        if isinstance(formula_state, dict):
            current_phase = str(formula_state.get("phase") or current_phase)
            current_down_streak = int(formula_state.get("window_down_streak") or 0)
            current_up_streak = int(formula_state.get("window_up_streak") or 0)
        else:
            current_phase = str(formula_state)
            current_down_streak = 0
            current_up_streak = 0

        for code, state in states.items():
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            history = data.iloc[: index + 1]
            if state.right:
                for lot in list(state.right):
                    if float(row["close"]) / lot["cost"] - 1 >= 0.10:
                        lot["proven"] = True
                    time_limit = 5 if current_phase == "exited" else 8 if current_phase == "watch" else 13
                    risk = position_exit_snapshot(
                        history, lot["cost"], lot["date"], entry_mode="right",
                        condition_stop=lot["stop"], time_limit_days=time_limit,
                    )
                    if risk["hard_stop_triggered"] or risk["entry_time_stop"]:
                        size = lot["size"]
                        execution_price = float(row["close"])
                        execution_reason = risk["position_action"]
                        if risk.get("space_stop_triggered"):
                            stop_price = float(risk["space_stop"])
                            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
                            fill = fill_sell_stop(
                                row, stop_price=stop_price,
                                previous_close=previous_close, code=code, is_st=is_st(code),
                            )
                            if not fill["filled"]:
                                continue
                            execution_price = float(fill["price"])
                            execution_reason = f"hard space stop {stop_price:.3f}; {fill['status']}"
                        pnl = size * (execution_price / lot["cost"] - 1)
                        state.right.remove(lot)
                        exited_today.add(code)
                        add_event(date, code, risk["position_action"], execution_price, -size, f"{lot['batch']}; {execution_reason}", pnl)
                if state.right:
                    if not any(lot["merged"] for lot in state.right):
                        state.right_parts = 5
                        state.right_sold.clear()
                    merge_threshold = 0.10 if current_up_streak >= 3 else 0.20
                    merged_new_batch = False
                    for lot in state.right:
                        if not lot["merged"] and float(row["close"]) / lot["cost"] - 1 >= merge_threshold:
                            lot["merged"] = True
                            merged_new_batch = True
                            add_event(date, code, "加仓批次合并", row["close"], 0.0, f"{lot['batch']}; 浮盈达到{merge_threshold:.0%}")

                    if merged_new_batch:
                        state.right_parts = 5
                        state.right_sold.clear()
                        state.right_plan_date = row["date"]

                    merged = [lot for lot in state.right if lot["merged"]]
                    if merged and state.right_parts > 0:
                        merged_size = sum(lot["size"] for lot in merged)
                        merged_cost = sum(lot["size"] * lot["cost"] for lot in merged) / merged_size
                        merged_entry = state.right_plan_date or min(lot["date"] for lot in merged)
                        profit = position_exit_snapshot(
                            history, merged_cost, merged_entry, entry_mode="right",
                            condition_stop=None, time_limit_days=9999, exit_tranches=5,
                        )
                        maximum_return = float(profit.get("maximum_return_pct") or 0.0)
                        if maximum_return < 10:
                            active_profit_ids = set()
                        elif maximum_return < 20:
                            active_profit_ids = {"half_profit"} if profit.get("half_profit_triggered") else set()
                        else:
                            active_profit_ids = set(profit.get("take_profit_trigger_ids") or [])
                        new_ids = active_profit_ids - state.right_sold
                        parts = min(len(new_ids), state.right_parts)
                        if parts:
                            liquidate_tail = merged_size < 0.10 - 1e-9
                            size = merged_size if liquidate_tail else merged_size * parts / state.right_parts
                            pnl = size * (float(row["close"]) / merged_cost - 1)
                            remaining_ratio = max(0.0, (merged_size - size) / merged_size)
                            for lot in merged:
                                lot["size"] *= remaining_ratio
                                if lot["size"] <= 1e-9:
                                    state.right.remove(lot)
                            state.right_parts = 0 if liquidate_tail else state.right_parts - parts
                            state.right_sold.update(sorted(new_ids)[:parts])
                            action = "统一尾仓退出" if liquidate_tail else f"统一分仓止盈{parts}份"
                            add_event(date, code, action, row["close"], -size, "合并后总仓独立止盈条件", pnl)
                            if state.right_parts == 0:
                                for lot in list(state.right):
                                    if lot["merged"]:
                                        state.right.remove(lot)
                                state.right_parts = 5
                                state.right_sold.clear()
                                state.right_plan_date = None
                                exited_today.add(code)
                    if not state.right:
                        state.right_parts = 5
                        state.right_sold.clear()
                        state.right_plan_date = None

            plan = state.left_plan or plans.get(code)
            if plan and state.left_lots:
                sold_today = set()
                previous_close = data.iloc[index - 1]["close"] if index > 0 else None
                for item in plan.get("grid") or []:
                    buy = float(item["buy_price"])
                    if buy not in state.left_lots or not item.get("core"):
                        continue
                    prior_peak = state.left_peaks.get(buy, buy)
                    if prior_peak / buy - 1 < 0.20:
                        continue
                    trailing_stop = prior_peak * 0.90
                    core_exit = fill_sell_stop(
                        row, stop_price=trailing_stop,
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                    if core_exit["filled"]:
                        size = state.left_lots.pop(buy)
                        state.left_peaks.pop(buy, None)
                        sell = float(core_exit["price"])
                        pnl = size * (sell / buy - 1)
                        sold_today.add(buy)
                        exited_today.add(code)
                        add_event(date, code, "左侧核心止盈", sell, -size, f"峰值回撤10%; {core_exit['status']}", pnl)
                for item in plan.get("grid") or []:
                    buy = float(item["buy_price"])
                    sell_order = fill_limit_order(
                        row, side="sell", limit_price=float(item["sell_price"]),
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                    if buy in state.left_lots and not item.get("core") and sell_order["filled"]:
                        size = state.left_lots.pop(buy)
                        state.left_peaks.pop(buy, None)
                        sell = float(sell_order["price"])
                        pnl = size * (sell / buy - 1)
                        sold_today.add(buy)
                        exited_today.add(code)
                        add_event(date, code, "左侧网格卖出", sell, -size, f"{buy:.2f}层到达卖价", pnl)
                for item in plan.get("grid") or []:
                    buy = float(item["buy_price"])
                    buy_order = fill_limit_order(
                        row, side="buy", limit_price=buy,
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                    if (
                        buy not in state.left_lots and buy not in sold_today
                        and buy_order["filled"]
                        and gross_exposure() + float(item["position_pct"]) <= 1.0 + 1e-9
                        and sum(state.left_lots.values()) + float(item["position_pct"]) <= 0.30 + 1e-9
                    ):
                        size = float(item["position_pct"])
                        state.left_lots[buy] = size
                        state.left_peaks[buy] = float(buy_order["price"])
                        add_event(date, code, "左侧网格买入", buy_order["price"], size, buy_order["status"])
                for buy in state.left_lots:
                    state.left_peaks[buy] = max(state.left_peaks.get(buy, buy), float(row["high"]))

        candidates = []
        entry_allowed = (
            current_down_streak < 3
            or not occupied_codes()
            or has_twenty_percent_float(date)
        )
        if entry_allowed:
            for code in current_candidates:
                if code not in states or code in exited_today:
                    continue
                if not _enabled(current_candidates[code].get("allow_right"), default=True):
                    continue
                data = frames[code]
                indexes = data.index[data["date"].dt.normalize() == date]
                if indexes.empty:
                    continue
                index = int(indexes[0])
                signal = _right_signal(data, index)
                if signal:
                    fundamental_score = float(current_candidates[code].get("candidate_score") or 0.0)
                    candidates.append((signal["rank"], signal["volume_ratio"], fundamental_score, code, index, signal))
            for _, __, fundamental_score, code, index, signal in sorted(candidates, reverse=True):
                if code not in occupied_codes() and symbol_limit_reached() and len(right_codes()) < 3:
                    continue
                opening_right_symbol = not states[code].right
                if opening_right_symbol:
                    if not can_open_new_right_symbol(date):
                        continue
                    if len(right_codes()) >= 3:
                        incoming_stop_pct = max(0.0, 1 - float(signal["stop"]) / float(frames[code].iloc[index]["close"]))
                        incoming_score = fundamental_score + signal["rank"] * 10 + min(float(signal["volume_ratio"]), 5.0) * 2 - incoming_stop_pct * 20
                        weak = []
                        for held_code in right_codes() - pending_weak_exits:
                            held_state = states[held_code]
                            visible = frames[held_code][frames[held_code]["date"] <= date]
                            if visible.empty:
                                continue
                            held_row = visible.iloc[-1]
                            held_size = sum(lot["size"] for lot in held_state.right)
                            held_cost = sum(lot["size"] * lot["cost"] for lot in held_state.right) / held_size
                            held_return = float(held_row["close"]) / held_cost - 1
                            holding_days = max(
                                len(visible[visible["date"] >= min(lot["date"] for lot in held_state.right)]), 1,
                            )
                            below_ma20 = pd.notna(held_row["ma20"]) and float(held_row["close"]) < float(held_row["ma20"])
                            removed = held_code not in current_candidates
                            eligible = removed and below_ma20 and holding_days >= 5 and held_return < 0.10
                            if not eligible:
                                continue
                            held_fundamental = float(current_candidates.get(held_code, {}).get("candidate_score") or 0.0)
                            held_score = held_fundamental + held_return * 100 + (20 if any(lot.get("proven") for lot in held_state.right) else 0) + (-10 if below_ma20 else 10)
                            if removed or incoming_score >= held_score + 5:
                                weak.append((held_score, held_code))
                        if weak:
                            pending_weak_exits.add(min(weak)[1])
                        continue
                row = frames[code].iloc[index]
                previous_close = frames[code].iloc[index - 1]["close"] if index > 0 else None
                if signal["order_type"] == "stop":
                    fill = fill_buy_stop(
                        row, trigger_price=signal["trigger"],
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                else:
                    fill = fill_limit_order(
                        row, side="buy", limit_price=signal["trigger"],
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                if not fill["filled"]:
                    continue
                existing_size = sum(lot["size"] for lot in states[code].right)
                if not states[code].right:
                    direct_breakout = signal["order_type"] == "stop"
                    size = 0.30 if current_up_streak >= 3 and not direct_breakout else 0.20
                    if direct_breakout:
                        entry_kind = "直接突破首仓"
                    else:
                        entry_kind = "33上行首仓" if current_up_streak >= 3 else "33未上行首仓"
                else:
                    threshold = 0.10 if current_up_streak >= 3 else 0.20
                    if not any(float(row["close"]) / lot["cost"] - 1 >= threshold for lot in states[code].right):
                        continue
                    ratio = 0.50 if signal["order_type"] == "limit" else 1 / 3
                    size = existing_size * ratio
                    entry_kind = f"浮盈加仓{ratio:.0%}"
                if gross_exposure() + size > 1.0 + 1e-9:
                    continue
                batch = f"R{next_batch_id}"
                next_batch_id += 1
                states[code].right.append({
                    "cost": float(fill["price"]), "date": row["date"],
                    "stop": float(signal["stop"]), "size": size,
                    "batch": batch, "merged": not bool(states[code].right),
                    "proven": float(row["close"]) / float(fill["price"]) - 1 >= 0.10,
                    "reconfirm_level": float(signal["trigger"]),
                    "reconfirm_on_next_day": float(row["close"]) < float(signal["stop"]),
                })
                if not states[code].right_plan_date:
                    states[code].right_plan_date = row["date"]
                add_event(date, code, "右侧买入", fill["price"], size, f"{batch}; {entry_kind}; {signal['reason']}; {fill['status']}")

        for code in current_candidates:
            if code not in states or code in occupied_codes() or symbol_limit_reached():
                continue
            if sum(bool(state.left_lots) for state in states.values()) >= 2:
                continue
            plan = plans.get(code)
            if not plan:
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            for item in plan.get("grid") or []:
                buy = float(item["buy_price"])
                buy_order = fill_limit_order(
                    row, side="buy", limit_price=buy,
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
                if (
                    buy_order["filled"]
                    and gross_exposure() + float(item["position_pct"]) <= 1.0 + 1e-9
                    and sum(states[code].left_lots.values()) + float(item["position_pct"]) <= 0.30 + 1e-9
                ):
                    size = float(item["position_pct"])
                    states[code].left_plan = plan
                    states[code].left_lots[buy] = size
                    states[code].left_peaks[buy] = float(buy_order["price"])
                    add_event(date, code, "左侧网格买入", buy_order["price"], size, buy_order["status"])
                    break

        unrealized = 0.0
        for code, state in states.items():
            data = frames[code]
            visible = data[data["date"] <= date]
            if visible.empty:
                continue
            close = float(visible.iloc[-1]["close"])
            unrealized += sum(size * (close / buy - 1) for buy, size in state.left_lots.items())
            if state.right:
                unrealized += sum(lot["size"] * (close / lot["cost"] - 1) for lot in state.right)
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": 1 + realized + unrealized,
            "gross_exposure_pct": round(gross_exposure() * 100, 2),
            "candidate_count": len(current_candidates),
        })

    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        maximum_drawdown = 0.0
        final_equity = 1.0
    else:
        drawdown = curve["equity"] / curve["equity"].cummax() - 1
        maximum_drawdown = float(drawdown.min())
        final_equity = float(curve.iloc[-1]["equity"])
    precoverage_dates = {
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[
            (frame["date"] >= requested) & (frame["date"] < coverage_start),
            "date",
        ]
    }
    final_position_details = []
    for code in sorted(occupied_codes()):
        state = states[code]
        frame = frames[code]
        visible = frame[frame["date"] <= end]
        close = None if visible.empty else float(visible.iloc[-1]["close"])
        left_size = sum(state.left_lots.values())
        right_size = sum(float(lot["size"]) for lot in state.right)
        right_value = sum(float(lot["size"]) * float(lot["cost"]) for lot in state.right)
        final_position_details.append({
            "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "close": close,
            "left_position_pct": round(left_size * 100, 2),
            "right_position_pct": round(right_size * 100, 2),
            "total_position_pct": round((left_size + right_size) * 100, 2),
            "right_cost": None if not right_size else round(right_value / right_size, 3),
            "right_batches": [
                {"batch": lot["batch"], "position_pct": round(lot["size"] * 100, 2), "cost": round(lot["cost"], 3), "stop": round(lot["stop"], 3), "merged": lot["merged"], "proven": lot.get("proven", False)}
                for lot in state.right
            ],
            "left_levels": sorted(state.left_lots, reverse=True),
        })
    return {
        "requested_start": requested.strftime("%Y-%m-%d"),
        "actual_start": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "candidate_coverage_start": coverage_start.strftime("%Y-%m-%d"),
        "coverage_complete": not precoverage_dates,
        "missing_candidate_trade_dates": sorted(date.strftime("%Y-%m-%d") for date in precoverage_dates),
        "events": events,
        "equity_curve": equity_curve,
        "event_count": len(events),
        "realized_return_pct": round(realized * 100, 3),
        "unrealized_return_pct": round((final_equity - 1 - realized) * 100, 3),
        "final_return_pct": round((final_equity - 1) * 100, 3),
        "maximum_drawdown_pct": round(maximum_drawdown * 100, 3),
        "maximum_gross_exposure_pct": round(
            max((row["gross_exposure_pct"] for row in equity_curve), default=0.0), 2,
        ),
        "exit_tail_on_candidate_removal": bool(exit_tail_on_candidate_removal),
        "signals_effective_next_day": bool(signals_effective_next_day),
        "pending_candidate_exit_codes": sorted(pending_candidate_exits),
        "final_positions": final_position_details,
    }
