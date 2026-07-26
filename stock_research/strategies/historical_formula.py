"""Reconstruct daily Formula33 breadth and phase from local QFQ caches."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.indicators.formula33 import calc_kdj_k, calc_rsi, calc_wr
from stock_research.storage import Database


def rebuild_formula_history(kline_directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    warmup = start - pd.DateOffset(days=70)
    hits = defaultdict(set)
    traded = defaultdict(set)
    calendar = set()
    for path in Path(kline_directory).glob("*.csv"):
        try:
            frame = pd.read_csv(path, usecols=["date", "high", "low", "close"])
        except (OSError, ValueError):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[(frame["date"] >= warmup) & (frame["date"] <= end)].dropna()
        if len(frame) < 25:
            continue
        for column in ("high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna().sort_values("date").reset_index(drop=True)
        base = (
            (calc_kdj_k(frame) > 80)
            & (calc_wr(frame, 10) < 20)
            & (calc_wr(frame, 20) < 20)
            & (calc_rsi(frame["close"], 9) > 70)
        )
        xg = base.rolling(5, min_periods=5).sum().eq(5)
        code = path.stem.replace("_", ".", 1)
        for date in frame.loc[xg, "date"]:
            hits[date.normalize()].add(code)
        for date in frame.loc[frame["date"] >= start, "date"]:
            normalized = date.normalize()
            traded[normalized].add(code)
            calendar.add(normalized)

    dates = sorted(calendar)
    rows = []
    phase = "waiting"
    up_streak = 0
    down_streak = 0
    previous_count = None
    for index, date in enumerate(dates):
        pool = set()
        for window_date in dates[max(0, index - 20) : index + 1]:
            pool.update(hits[window_date])
        count = len(pool & traded[date])
        change = 0 if previous_count is None else count - previous_count
        up_streak = up_streak + 1 if change > 0 else 0
        down_streak = down_streak + 1 if change < 0 else 0
        if down_streak >= 5:
            phase = "exited"
        elif up_streak >= 5:
            phase = "active"
        elif up_streak >= 3 and phase != "active":
            phase = "watch"
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "window_unique_count": count,
            "window_change": change,
            "window_up_streak": up_streak,
            "window_down_streak": down_streak,
            "phase": phase,
            "reconstruction_version": "formula33-research-v1",
        })
        previous_count = count
    return pd.DataFrame(rows)


def _append_formula_rows(frame, code, hits, traded, calendar, start):
    if frame.empty:
        return
    frame = frame.dropna(subset=["date", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if len(frame) < 25:
        return
    for column in ("high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["high", "low", "close"])
    if len(frame) < 25:
        return
    base = (
        (calc_kdj_k(frame) > 80)
        & (calc_wr(frame, 10) < 20)
        & (calc_wr(frame, 20) < 20)
        & (calc_rsi(frame["close"], 9) > 70)
    )
    xg = base.rolling(5, min_periods=5).sum().eq(5)
    for date in frame.loc[xg, "date"]:
        hits[date.normalize()].add(code)
    for date in frame.loc[frame["date"] >= start, "date"]:
        normalized = date.normalize()
        traded[normalized].add(code)
        calendar.add(normalized)


def _phase_rows(hits, traded, calendar):
    dates = sorted(calendar)
    rows = []
    phase = "waiting"
    up_streak = 0
    down_streak = 0
    previous_count = None
    for index, date in enumerate(dates):
        pool = set()
        for window_date in dates[max(0, index - 20) : index + 1]:
            pool.update(hits[window_date])
        count = len(pool & traded[date])
        change = 0 if previous_count is None else count - previous_count
        up_streak = up_streak + 1 if change > 0 else 0
        down_streak = down_streak + 1 if change < 0 else 0
        if down_streak >= 5:
            phase = "exited"
        elif up_streak >= 5:
            phase = "active"
        elif up_streak >= 3 and phase != "active":
            phase = "watch"
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "window_unique_count": count,
            "window_change": change,
            "window_up_streak": up_streak,
            "window_down_streak": down_streak,
            "phase": phase,
            "reconstruction_version": "formula33-research-v1",
        })
        previous_count = count
    return pd.DataFrame(rows)


def rebuild_formula_history_from_tushare(start_date, end_date, *, database_path=None):
    """Reconstruct Formula33 phase from Tushare daily_kline and adj_factor rows."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    warmup = start - pd.DateOffset(days=70)
    database = Database(database_path or PATHS.database, code_version="formula33-tushare-history")
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT dataset, ts_code, trade_date, payload_json
            FROM raw.tushare_dataset_rows
            WHERE dataset IN ('daily_kline', 'adj_factor')
              AND trade_date >= ?
              AND trade_date <= ?
            ORDER BY ts_code, trade_date, dataset
            """,
            [warmup.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        ).fetchall()
    finally:
        connection.close()
    daily_rows = []
    adj_rows = []
    for dataset, ts_code, trade_date, payload_json in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            continue
        item = {
            "code": str(ts_code),
            "date": pd.to_datetime(trade_date, errors="coerce"),
        }
        if dataset == "daily_kline":
            item.update({
                "high": payload.get("high"),
                "low": payload.get("low"),
                "close": payload.get("close"),
            })
            daily_rows.append(item)
        elif dataset == "adj_factor":
            item["adj_factor"] = payload.get("adj_factor")
            adj_rows.append(item)
    daily = pd.DataFrame(daily_rows)
    adj = pd.DataFrame(adj_rows)
    if daily.empty or adj.empty:
        return pd.DataFrame()
    merged = daily.merge(adj, on=["code", "date"], how="inner").dropna(subset=["date"])
    for column in ("high", "low", "close", "adj_factor"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["high", "low", "close", "adj_factor"])
    for column in ("high", "low", "close"):
        merged[column] = merged[column] * merged["adj_factor"]

    hits = defaultdict(set)
    traded = defaultdict(set)
    calendar = set()
    for code, group in merged.groupby("code", sort=False):
        _append_formula_rows(
            group[["date", "high", "low", "close"]].copy(),
            code,
            hits,
            traded,
            calendar,
            start,
        )
    return _phase_rows(hits, traded, calendar)
