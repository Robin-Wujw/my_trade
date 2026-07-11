"""Persistent two-month pullback-breakout watch list."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_research.indicators.waves import infer_downtrend_recovery


def load_watch_state(path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": 1, "stocks": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("stocks"), dict):
        return {"version": 1, "stocks": {}}
    return payload


def save_watch_state(path, state):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def recent_pool(history, current_rows, observation_date, months=2):
    observation = pd.Timestamp(observation_date).normalize()
    cutoff = observation - pd.DateOffset(months=months)
    pool = {}
    for date_text, rows in getattr(history, "snapshots", {}).items():
        date = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(date) or date.normalize() < cutoff:
            continue
        for row in rows:
            code = str(row.get("code") or "").strip()
            if code:
                pool[code] = {"code": code, "name": str(row.get("name") or code), "first_seen": str(date_text)}
    for row in current_rows:
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        previous = pool.get(code, {})
        pool[code] = {
            "code": code,
            "name": str(row.get("name") or previous.get("name") or code),
            "first_seen": previous.get("first_seen") or observation.strftime("%Y-%m-%d"),
        }
    return pool


def update_breakout_watch(pool, current_codes, observation_date, kline_loader, previous_state=None):
    previous = dict((previous_state or {}).get("stocks") or {})
    observation = pd.Timestamp(observation_date).strftime("%Y-%m-%d")
    current_codes = {str(code) for code in current_codes}
    next_stocks = {}
    alerts = []
    for code, identity in sorted(pool.items()):
        old = dict(previous.get(code) or {})
        frame = kline_loader(code, observation_date)
        wave = infer_downtrend_recovery(frame, lookback=500) if frame is not None and not frame.empty else None
        if not wave:
            closes = pd.to_numeric(frame.get("close"), errors="coerce").dropna() if frame is not None and not frame.empty else pd.Series(dtype=float)
            close = float(closes.iloc[-1]) if not closes.empty else None
            completed = bool(close is not None and old.get("prior_high") and close >= float(old["prior_high"]))
            if completed and code not in current_codes:
                continue
            next_stocks[code] = {
                **old, **identity, "last_observation": observation,
                "data_available": False, "close": close,
                "completed": completed or bool(old.get("completed")),
                "in_current_selection": code in current_codes,
            }
            continue
        close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[-1])
        recovery = float(wave["recovery_progress_pct"])
        above_50 = recovery >= 50
        completed = recovery >= 100
        is_new_day = str(old.get("last_observation") or "") < observation
        crossing_count = int(old.get("crossing_count") or 0)
        if is_new_day and above_50 and old.get("above_50") is False:
            crossing_count += 1
        elif not old and above_50:
            crossing_count = 1
        in_current_selection = code in current_codes
        state = {
            **old,
            **identity,
            "last_observation": observation,
            "data_available": True,
            "close": close,
            "recovery_pct": round(recovery, 2),
            "pullback_level_50": wave["recovery_level_50"],
            "uptrend_level_50": wave["uptrend_level_50"],
            "prior_high": wave["downtrend_high"],
            "above_50": above_50,
            "crossing_count": crossing_count,
            "completed": completed,
            "in_current_selection": in_current_selection,
        }
        # Removal requires both conditions. Completing the breakout alone is
        # not enough while the stock remains in any current selection.
        if completed and not in_current_selection:
            continue
        next_stocks[code] = state
        if 45 <= recovery <= 60:
            if completed:
                alert_level = "突破前高完成，仍在当前筛选"
            elif above_50:
                alert_level = f"突破跟踪：第{crossing_count}次突破回调50%"
            else:
                alert_level = "强提醒：进入回调45%-50%突破区"
            alerts.append({**state, "alert_level": alert_level})
    alerts.sort(key=lambda item: (not item["completed"], -item["recovery_pct"], item["code"]))
    return {"version": 1, "observation_date": observation, "stocks": next_stocks}, alerts
