"""Fast analysis for a small user-maintained stock watch list."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_research.pipelines.formula33 import load_cached_kline, load_kline_with_cache


def expected_market_data_date(now=None) -> pd.Timestamp:
    current = pd.Timestamp(now or pd.Timestamp.now())
    date = current.normalize()
    if current.weekday() >= 5:
        return date - pd.offsets.BDay(1)
    if current.hour < 16:
        return date - pd.offsets.BDay(1)
    return date


def load_or_refresh_watch_kline(
    code,
    now=None,
    fetcher=load_kline_with_cache,
) -> tuple[pd.DataFrame, dict]:
    """Return fresh-enough QFQ bars, fetching only this stock when needed."""
    expected = expected_market_data_date(now)
    cached = load_cached_kline("akshare", code)
    cached_dates = (
        pd.to_datetime(cached.get("date"), errors="coerce")
        if not cached.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    cached_latest = cached_dates.max() if not cached_dates.dropna().empty else None
    if cached_latest is not None and cached_latest.normalize() >= expected:
        return cached, {
            "fresh": True,
            "fetched": False,
            "latest_date": cached_latest.strftime("%Y-%m-%d"),
        }
    start = (expected - pd.DateOffset(days=800)).strftime("%Y-%m-%d")
    end = expected.strftime("%Y-%m-%d")
    try:
        frame = fetcher(
            "akshare", code, start, end,
            retries=3, retry_delay=1.0, minimum_history_rows=120,
        )
    except Exception as exc:
        return cached, {
            "fresh": False, "fetched": True,
            "latest_date": cached_latest.strftime("%Y-%m-%d") if cached_latest is not None else None,
            "error": str(exc),
        }
    dates = (
        pd.to_datetime(frame.get("date"), errors="coerce")
        if not frame.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    latest = dates.max() if not dates.dropna().empty else None
    # A successful through-date request can legitimately return no newer bar
    # on a market holiday or while the stock is suspended.
    fresh = latest is not None
    return frame, {
        "fresh": fresh, "fetched": True,
        "latest_date": latest.strftime("%Y-%m-%d") if latest is not None else None,
        "confirmed_through": end,
        "error": None if fresh else f"截至{end}仍没有可用日线",
    }


def load_watch_stocks(path) -> list[dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    stocks = payload.get("stocks") if isinstance(payload, dict) else None
    if not isinstance(stocks, list):
        raise ValueError("watch stock file must contain a stocks array")
    return [item for item in stocks if isinstance(item, dict) and item.get("code")]


def analyze_watch_stock(identity, frame, reminders=(), data_status=None) -> dict:
    data_status = data_status or {"fresh": True, "fetched": False}
    if frame is None or frame.empty:
        return {**identity, "available": False, "opinion": "本地行情不可用，先更新该股日线"}
    data = frame.copy()
    for column in ("close", "high", "low", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["close", "high", "low"]).sort_values("date")
    if data.empty:
        return {**identity, "available": False, "opinion": "本地行情不可用，先更新该股日线"}
    for period in (5, 10, 20, 60):
        data[f"ma{period}"] = data["close"].rolling(period).mean()
    last = data.iloc[-1]
    close = float(last["close"])
    ma20 = float(last["ma20"]) if pd.notna(last["ma20"]) else None
    ma60 = float(last["ma60"]) if pd.notna(last["ma60"]) else None
    volume = float(last["volume"]) if pd.notna(last.get("volume")) else None
    ref5 = (
        float(data.iloc[-6]["volume"])
        if len(data) >= 6 and pd.notna(data.iloc[-6]["volume"])
        else None
    )
    ref10 = (
        float(data.iloc[-11]["volume"])
        if len(data) >= 11 and pd.notna(data.iloc[-11]["volume"])
        else None
    )
    stock_reminders = [
        item for item in reminders if item.get("code") == identity.get("code")
    ]
    if not data_status.get("fresh"):
        opinion = f"行情过期，补抓失败：{data_status.get('error') or '未知错误'}；暂不执行买卖提醒"
    elif stock_reminders:
        opinion = "；".join(item["message"] for item in stock_reminders)
    elif ma20 is not None and ma60 is not None and close < ma20 < ma60:
        opinion = "均线结构偏弱，暂不做右侧；只执行已预设左侧网格"
    elif ma20 is not None and close >= ma20:
        ma20_rising = len(data) >= 25 and float(data.iloc[-1]["ma20"]) > float(data.iloc[-6]["ma20"])
        ma60_rising = len(data) >= 65 and float(data.iloc[-1]["ma60"]) > float(data.iloc[-6]["ma60"])
        opinion = "均线向上，继续等待明确结构买点或上扬均线拉回" if ma20_rising and ma60_rising else "价格在20日线上方，但中长期均线方向未确认，继续观察"
    else:
        opinion = "暂无临近交易条件，继续观察"
    return {
        **identity, "available": True, "data_fresh": bool(data_status.get("fresh")),
        "data_fetched": bool(data_status.get("fetched")),
        "data_error": data_status.get("error"),
        "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "close": close, "ma5": last["ma5"], "ma10": last["ma10"],
        "ma20": ma20, "ma60": ma60,
        "volume": volume, "volume_ref5": ref5, "volume_ref10": ref10,
        "volume_ready": bool(volume and ref5 and ref10 and volume > max(ref5, ref10)),
        "opinion": opinion,
    }
