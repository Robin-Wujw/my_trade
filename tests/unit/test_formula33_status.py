import pandas as pd
from stock_research.pipelines import formula33
from stock_research.strategies.formula33 import (
    build_window_trend,
    classify_observation_status,
    select_window_unique_hits,
)


def test_build_window_trend_uses_distinct_codes_in_each_21_day_window():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [
            {"date": date, "code": f"sz.{index:06d}"}
            for index, date in enumerate(dates)
        ]
        + [{"date": dates[-1], "code": "sz.000060"}]
    )

    result = build_window_trend(hits, dates, window=21, output_days=21)

    assert len(result) == 21
    assert result["window_unique_count"].tolist() == [21] * 21
    assert result["window_trend_slope"].abs().max() < 1e-12
    assert result.iloc[-1]["trend_up_streak"] == 0
    assert result.iloc[-1]["trend_down_streak"] == 0


def test_build_window_trend_counts_consecutive_positive_slopes():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [
            {"date": date, "code": f"sz.{code_index:06d}"}
            for index, date in enumerate(dates)
            for code_index in range(index + 1)
        ]
    )

    result = build_window_trend(hits, dates, window=21, output_days=21)

    assert result.iloc[-1]["window_trend_slope"] > 0
    assert result.iloc[-1]["trend_up_streak"] == 21
    assert result.iloc[-1]["trend_down_streak"] == 0


def test_build_window_trend_counts_consecutive_negative_slopes():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [
            {"date": date, "code": f"sz.{code_index:06d}"}
            for index, date in enumerate(dates)
            for code_index in range(index, len(dates))
        ]
    )

    result = build_window_trend(hits, dates, window=21, output_days=21)

    assert result.iloc[-1]["window_trend_slope"] < 0
    assert result.iloc[-1]["trend_up_streak"] == 0
    assert result.iloc[-1]["trend_down_streak"] == 21


def test_formula_summary_populates_all_21_rolling_rows():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [
            {
                "signal_type": "XG",
                "date": date,
                "code": f"sz.{index:06d}",
            }
            for index, date in enumerate(dates)
        ]
    )

    summary = formula33.build_formula_summary(hits, dates, output_days=21)

    assert len(summary) == 21
    assert summary["window_unique_count"].notna().all()
    assert summary["window_trend_slope"].notna().all()


def test_status_distinguishes_traded_suspended_and_unavailable():
    assert classify_observation_status("2026-07-02", "2026-07-02") == "traded"
    assert (
        classify_observation_status("2026-06-26", "2026-07-02")
        == "suspended_or_no_trade"
    )
    assert (
        classify_observation_status("2026-06-26", "2026-07-02", fetch_error="timeout")
        == "data_unavailable"
    )


def test_suspended_stock_is_excluded_only_for_current_observation():
    hits = pd.DataFrame(
        [
            {"code": "sh.688072", "date": "2026-06-26"},
            {"code": "sz.000001", "date": "2026-07-02"},
        ]
    )
    suspended = pd.DataFrame(
        [
            {"code": "sh.688072", "observation_status": "suspended_or_no_trade"},
            {"code": "sz.000001", "observation_status": "traded"},
        ]
    )
    resumed = suspended.copy()
    resumed.loc[resumed["code"] == "sh.688072", "observation_status"] = "traded"

    technical, formal_while_suspended = select_window_unique_hits(hits, suspended)
    _, formal_after_resumption = select_window_unique_hits(hits, resumed)

    assert set(technical["code"]) == {"sh.688072", "sz.000001"}
    assert set(formal_while_suspended["code"]) == {"sz.000001"}
    assert set(formal_after_resumption["code"]) == {"sh.688072", "sz.000001"}


def test_july_second_replay_counts_technical_and_tradable_sets():
    traded_codes = [f"sz.{index:06d}" for index in range(184)]
    suspended_codes = ["sh.688072", "sz.300214"]
    hits = pd.DataFrame(
        [
            {"code": code, "date": "2026-06-26"}
            for code in traded_codes + suspended_codes
        ]
    )
    statuses = pd.DataFrame(
        [
            {"code": code, "observation_status": "traded"}
            for code in traded_codes
        ]
        + [
            {"code": code, "observation_status": "suspended_or_no_trade"}
            for code in suspended_codes
        ]
    )

    technical, formal = select_window_unique_hits(hits, statuses)

    assert len(technical) == 186
    assert len(formal) == 184


def make_fetch_task():
    return (
        "sh.688072",
        "拓荆科技",
        "2026-05-01",
        "2026-07-02",
        {"2026-06-26"},
        None,
        "2022-04-20",
        None,
        300,
        0.0,
        "akshare",
        1,
        0.0,
        False,
        True,
        "pass",
    )


def test_fetch_preserves_historical_hit_when_observation_date_has_no_trade(monkeypatch):
    dates = pd.bdate_range(end="2026-06-26", periods=30)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
        }
    )
    monkeypatch.setattr(formula33, "load_kline_with_cache", lambda *args, **kwargs: frame)
    monkeypatch.setattr(formula33, "calc_kdj_k", lambda df: pd.Series(90.0, index=df.index))
    monkeypatch.setattr(formula33, "calc_wr", lambda df, n: pd.Series(10.0, index=df.index))
    monkeypatch.setattr(formula33, "calc_rsi", lambda series, n=9: pd.Series(80.0, index=series.index))

    rows = formula33.fetch_one_stock(make_fetch_task())

    status = [row for row in rows if row["signal_type"] == "STATUS"]
    xg = [row for row in rows if row["signal_type"] == "XG"]
    assert status == [
        {
            "signal_type": "STATUS",
            "code": "sh.688072",
            "name": "拓荆科技",
            "latest_data_date": "2026-06-26",
            "observation_status": "suspended_or_no_trade",
            "error": "",
        }
    ]
    assert [row["date"] for row in xg] == ["2026-06-26"]


def test_fetch_error_is_data_unavailable_not_suspension(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("source timeout")

    monkeypatch.setattr(formula33, "load_kline_with_cache", fail)
    monkeypatch.setattr(formula33, "load_cached_kline", lambda *args: pd.DataFrame())

    rows = formula33.fetch_one_stock(make_fetch_task())

    assert rows == [
        {
            "signal_type": "STATUS",
            "code": "sh.688072",
            "name": "拓荆科技",
            "latest_data_date": "",
            "observation_status": "data_unavailable",
            "error": "source timeout",
        }
    ]


def test_fetch_error_preserves_cached_technical_hit(monkeypatch):
    dates = pd.bdate_range(end="2026-06-26", periods=30)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
        }
    )

    def fail(*args, **kwargs):
        raise RuntimeError("source timeout")

    monkeypatch.setattr(formula33, "load_kline_with_cache", fail)
    monkeypatch.setattr(formula33, "load_cached_kline", lambda *args: frame)
    monkeypatch.setattr(formula33, "calc_kdj_k", lambda df: pd.Series(90.0, index=df.index))
    monkeypatch.setattr(formula33, "calc_wr", lambda df, n: pd.Series(10.0, index=df.index))
    monkeypatch.setattr(formula33, "calc_rsi", lambda series, n=9: pd.Series(80.0, index=series.index))

    rows = formula33.fetch_one_stock(make_fetch_task())

    status = [row for row in rows if row["signal_type"] == "STATUS"]
    xg = [row for row in rows if row["signal_type"] == "XG"]
    assert status[0]["observation_status"] == "data_unavailable"
    assert status[0]["error"] == "source timeout"
    assert [row["date"] for row in xg] == ["2026-06-26"]
