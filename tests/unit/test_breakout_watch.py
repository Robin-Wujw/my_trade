import pandas as pd

from stock_research.reporting.breakout_watch import recent_pool, update_breakout_watch
from stock_research.reporting.diff import SelectionHistory


def bars(last_close):
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "high": [20.0] * 10 + [18.0] * 50,
            "low": [19.0] * 10 + [10.0] * 50,
            "close": [19.5] * 10 + [12.0] * 49 + [last_close],
        }
    )


def test_recent_pool_keeps_prior_two_month_stocks_outside_current_selection():
    history = SelectionHistory(
        {
            "2026-06-01": [{"code": "OLD", "name": "旧股", "strategy_part": "1"}],
            "2026-04-01": [{"code": "STALE", "name": "过期", "strategy_part": "1"}],
        }
    )

    pool = recent_pool(history, [{"code": "NOW", "name": "当前"}], "2026-07-10")

    assert set(pool) == {"OLD", "NOW"}


def test_watch_repeats_crossings_and_only_removes_completed_non_selection():
    pool = {"A": {"code": "A", "name": "甲", "first_seen": "2026-06-01"}}
    state1, alerts1 = update_breakout_watch(pool, {"A"}, "2026-07-08", lambda *_: bars(15.0))
    assert alerts1[0]["crossing_count"] == 1
    state2, _ = update_breakout_watch(pool, {"A"}, "2026-07-09", lambda *_: bars(14.0), state1)
    state3, alerts3 = update_breakout_watch(pool, {"A"}, "2026-07-10", lambda *_: bars(15.1), state2)
    assert alerts3[0]["crossing_count"] == 2

    kept, completed_alerts = update_breakout_watch(pool, {"A"}, "2026-07-11", lambda *_: bars(21.0), state3)
    assert "A" in kept["stocks"] and completed_alerts == []
    removed, alerts = update_breakout_watch(pool, set(), "2026-07-12", lambda *_: bars(21.0), kept)
    assert "A" not in removed["stocks"] and alerts == []


def test_watch_strong_alert_starts_at_45_and_stops_after_60():
    pool = {"A": {"code": "A", "name": "甲", "first_seen": "2026-06-01"}}

    _, below = update_breakout_watch(pool, {"A"}, "2026-07-08", lambda *_: bars(14.4))
    _, strong = update_breakout_watch(pool, {"A"}, "2026-07-08", lambda *_: bars(14.5))
    _, detached = update_breakout_watch(pool, {"A"}, "2026-07-08", lambda *_: bars(16.1))

    assert below == []
    assert strong[0]["alert_level"].startswith("强提醒")
    assert detached == []
