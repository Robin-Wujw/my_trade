import pandas as pd

from apps import quick_watch as quick_watch_app
from stock_research.reporting.quick_watch import (
    analyze_watch_stock,
    load_or_refresh_watch_kline,
)


def test_quick_watch_uses_trade_plan_opinion_without_automatic_wave_levels():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=80),
        "open": [20.0] * 80, "high": [20.5] * 80, "low": [19.5] * 80,
        "close": [20.0] * 80, "volume": [1000] * 80,
    })
    reminder = {"code": "A", "message": "接近网格买价19.30，计划买入6.0%"}

    result = analyze_watch_stock({"code": "A", "name": "甲"}, frame, [reminder])

    assert result["available"] is True
    assert result["opinion"] == reminder["message"]
    assert "uptrend_level_50" not in result
    assert "pullback_level_50" not in result


def test_watch_loader_uses_fresh_cache_without_network(monkeypatch):
    cached = pd.DataFrame({"date": ["2026-07-10"], "close": [10]})
    monkeypatch.setattr("stock_research.reporting.quick_watch.load_cached_kline", lambda *_: cached)

    frame, status = load_or_refresh_watch_kline(
        "A", now="2026-07-12 10:00",
        fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    assert frame.equals(cached)
    assert status == {"fresh": True, "fetched": False, "latest_date": "2026-07-10"}


def test_watch_loader_fetches_stale_cache(monkeypatch):
    cached = pd.DataFrame({"date": ["2026-07-09"], "close": [10]})
    fresh = pd.DataFrame({"date": ["2026-07-10"], "close": [11]})
    calls = []
    monkeypatch.setattr("stock_research.reporting.quick_watch.load_cached_kline", lambda *_: cached)

    frame, status = load_or_refresh_watch_kline(
        "A", now="2026-07-12 10:00",
        fetcher=lambda *args, **kwargs: calls.append((args, kwargs)) or fresh,
    )

    assert frame.equals(fresh)
    assert status["fresh"] is True
    assert status["fetched"] is True
    assert calls[0][0][0:2] == ("akshare", "A")


def test_successful_fetch_with_no_new_bar_is_latest_available(monkeypatch):
    cached = pd.DataFrame({"date": ["2026-07-09"], "close": [10]})
    monkeypatch.setattr("stock_research.reporting.quick_watch.load_cached_kline", lambda *_: cached)

    _, status = load_or_refresh_watch_kline(
        "A", now="2026-07-12 10:00", fetcher=lambda *_args, **_kwargs: cached,
    )

    assert status["fresh"] is True
    assert status["latest_date"] == "2026-07-09"
    assert status["confirmed_through"] == "2026-07-10"


def test_stale_fetch_failure_blocks_trade_opinion(monkeypatch):
    cached = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=60),
        "open": [10] * 60, "high": [10] * 60, "low": [10] * 60,
        "close": [10] * 60, "volume": [1000] * 60,
    })
    monkeypatch.setattr("stock_research.reporting.quick_watch.load_cached_kline", lambda *_: cached)
    frame, status = load_or_refresh_watch_kline(
        "A", now="2026-07-12 10:00",
        fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    result = analyze_watch_stock(
        {"code": "A", "name": "甲"}, frame,
        [{"code": "A", "message": "计划买入"}], status,
    )

    assert status["fresh"] is False
    assert "行情过期" in result["opinion"]
    assert "暂不执行买卖提醒" in result["opinion"]


def test_quick_watch_includes_explicit_plan_not_repeated_in_watch_list(monkeypatch):
    monkeypatch.setattr(quick_watch_app, "load_watch_stocks", lambda _path: [])
    monkeypatch.setattr(
        quick_watch_app,
        "load_trade_plans",
        lambda _path: {"plans": {"A": {"name": "甲", "grid": []}}},
    )
    monkeypatch.setattr(
        quick_watch_app,
        "load_or_refresh_watch_kline",
        lambda _code: (pd.DataFrame(), {"fresh": False}),
    )
    monkeypatch.setattr(quick_watch_app, "build_trade_reminders", lambda *_args: [])

    analyses, _ = quick_watch_app.build_quick_watch("watch.json", "plans.json")

    assert analyses[0]["code"] == "A"
    assert analyses[0]["name"] == "甲"
