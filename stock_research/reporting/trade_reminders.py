"""Advance reminders for explicit trade plans and selected-stock candidates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_research.strategies.position_plan import backtest_position_plan


def load_trade_plans(path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": 1, "proximity_pct": 0.02, "plans": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("plans"), dict):
        raise ValueError("trade plan file must contain a plans object")
    return payload


def _number(value):
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _near_above(close: float, target: float, threshold: float) -> float | None:
    distance = close / target - 1
    return distance if 0 <= distance <= threshold else None


def build_trade_reminders(stocks, observation_date, kline_loader, plan_config) -> list[dict]:
    """Build reminders without turning unplanned candidates into orders."""
    threshold = float(plan_config.get("proximity_pct", 0.02))
    plans = dict(plan_config.get("plans") or {})
    rows = {str(row.get("code")): row for row in stocks.to_dict("records")}
    reminders = []

    for code, plan in sorted(plans.items()):
        frame = kline_loader(code, observation_date)
        if frame is None or frame.empty:
            continue
        result = backtest_position_plan(
            frame,
            plan.get("start_date", observation_date),
            value_line=plan.get("value_line"),
            left_grid_plan=plan.get("grid"),
            right_position_pct=float(plan.get("right_position_pct", 0.15)),
        )
        close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[-1])
        filled = {float(level) for level in result["filled_left_levels"]}
        for item in plan.get("grid") or []:
            buy = float(item["buy_price"])
            sell = float(item["sell_price"])
            size = float(item["position_pct"])
            if buy not in filled:
                distance = _near_above(close, buy, threshold)
                if distance is not None:
                    reminders.append({
                        "code": code, "name": plan.get("name", code), "kind": "计划买入",
                        "close": close, "target": buy, "position_pct": size * 100,
                        "distance_pct": distance * 100,
                        "message": f"接近网格买价{buy:.2f}，计划买入{size * 100:.1f}%",
                    })
            elif not item.get("core"):
                distance = _near_above(sell, close, threshold)
                if distance is not None:
                    reminders.append({
                        "code": code, "name": plan.get("name", code), "kind": "计划卖出",
                        "close": close, "target": sell, "position_pct": size * 100,
                        "distance_pct": distance * 100,
                        "message": f"接近网格卖价{sell:.2f}，计划卖出{size * 100:.1f}%",
                    })
        risk = result.get("right_risk") or {}
        if risk.get("position_risk_available"):
            protection_levels = {
                "条件/空间止损": risk.get("initial_stop"),
                "保本保护": risk.get("profit_floor"),
            }
            if float(risk.get("maximum_return_pct") or 0) >= 10:
                protection_levels["浮盈回撤一半"] = risk.get("half_profit_stop")
            if risk.get("trailing_10_stop") is not None:
                protection_levels["峰值回撤10%"] = risk.get("trailing_10_stop")
            seen_targets = set()
            for label, target_value in protection_levels.items():
                target = _number(target_value)
                if target is None or target in seen_targets:
                    continue
                seen_targets.add(target)
                distance = _near_above(close, target, threshold)
                if distance is not None:
                    reminders.append({
                        "code": code, "name": plan.get("name", code), "kind": "计划卖出",
                        "close": close, "target": target,
                        "position_pct": float(result.get("right_position_pct") or 0),
                        "distance_pct": distance * 100,
                        "message": f"接近右侧{label}{target:.2f}，准备按批次退出",
                    })
            days_left = int(risk.get("time_limit_days") or 0) - int(risk.get("holding_days") or 0)
            if days_left == 1 and float(risk.get("maximum_return_pct") or 0) < 10:
                reminders.append({
                    "code": code, "name": plan.get("name", code), "kind": "计划卖出",
                    "close": close, "target": None,
                    "position_pct": float(result.get("right_position_pct") or 0),
                    "distance_pct": 0.0,
                    "message": "右侧时间止损剩1个交易日，若最大浮盈仍不足10%则退出",
                })

    for code, row in rows.items():
        if code in plans:
            continue
        close = _number(row.get("close"))
        if close is None or close <= 0:
            continue
        strategy_part = str(row.get("strategy_part") or "")
        value_line = _number(row.get("value_line"))
        if strategy_part.startswith("1.") and value_line:
            distance = _near_above(close, value_line, threshold)
            if distance is not None:
                reminders.append({
                    "code": code, "name": row.get("name", code), "kind": "左侧候选",
                    "close": close, "target": value_line, "position_pct": None,
                    "distance_pct": distance * 100,
                    "message": f"距离价值线{value_line:.2f}仅{distance * 100:.1f}%，待制定显式网格",
                })
    order = {"计划卖出": 0, "计划买入": 1, "左侧候选": 2}
    return sorted(reminders, key=lambda item: (order[item["kind"]], item["distance_pct"], item["code"]))
