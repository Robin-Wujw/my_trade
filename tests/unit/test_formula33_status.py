from pathlib import Path

import pandas as pd
import pytest

from stock_research.core.completion_manifest import CompletionManifest
from stock_research.pipelines import formula33
from stock_research.strategies.formula33 import (
    build_count_direction_streaks,
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


def test_window_trend_excludes_no_trade_stock_on_each_observation_date():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    always_traded = "sz.000001"
    intermittently_suspended = "sh.688072"
    hits = pd.DataFrame(
        [
            {"date": date, "code": code}
            for date in dates
            for code in (always_traded, intermittently_suspended)
        ]
    )
    suspended_date = dates[-2]
    coverage = {
        always_traded: dates,
        intermittently_suspended: [
            date for date in dates if date != suspended_date
        ],
    }

    result = build_window_trend(
        hits,
        dates,
        window=21,
        output_days=21,
        trade_coverage=coverage,
    )

    by_date = result.set_index("date")
    assert by_date.loc[suspended_date, "technical_unique_count"] == 2
    assert by_date.loc[suspended_date, "window_unique_count"] == 1
    assert by_date.loc[dates[-1], "window_unique_count"] == 2


def test_current_data_unavailable_is_excluded_even_when_cache_has_date():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [{"date": date, "code": "sz.000001"} for date in dates]
    )
    statuses = pd.DataFrame(
        [
            {
                "code": "sz.000001",
                "observation_status": "data_unavailable",
            }
        ]
    )

    result = build_window_trend(
        hits,
        dates,
        window=21,
        output_days=21,
        trade_coverage={"sz.000001": dates},
        current_statuses=statuses,
    )

    assert result.iloc[-1]["technical_unique_count"] == 1
    assert result.iloc[-1]["window_unique_count"] == 0


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


def test_fetch_skips_too_recently_listed_stock_before_kline(monkeypatch):
    task = list(make_fetch_task())
    task[6] = "2026-06-01"
    task[8] = 300
    monkeypatch.setattr(
        formula33,
        "load_kline_with_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("K-line loader must not run")
        ),
    )

    assert formula33.fetch_one_stock(tuple(task)) == []


def test_rolling_21_day_node_changes_drive_their_own_direction_streaks():
    result = build_count_direction_streaks(
        [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
    )

    assert result["window_change"].tolist() == [
        0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1
    ]
    assert result["window_up_streak"].tolist() == [
        0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0
    ]
    assert result["window_down_streak"].tolist() == [
        0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5
    ]


@pytest.mark.parametrize(
    ("market_cap", "expected_market_cap_hit"),
    [(None, False), (50.0, False), (150.0, True)],
)
def test_market_cap_only_filters_the_separate_formal_pool(
    monkeypatch,
    market_cap,
    expected_market_cap_hit,
):
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
    task = list(make_fetch_task())
    task[5] = market_cap
    task[7] = 100.0
    task[15] = "exclude"
    monkeypatch.setattr(
        formula33,
        "load_kline_with_cache",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        formula33,
        "calc_kdj_k",
        lambda df: pd.Series(90.0, index=df.index),
    )
    monkeypatch.setattr(
        formula33,
        "calc_wr",
        lambda df, n: pd.Series(10.0, index=df.index),
    )
    monkeypatch.setattr(
        formula33,
        "calc_rsi",
        lambda series, n=9: pd.Series(80.0, index=series.index),
    )

    rows = formula33.fetch_one_stock(tuple(task))

    assert [row["date"] for row in rows if row["signal_type"] == "XG"] == [
        "2026-06-26"
    ]
    assert bool(
        [row for row in rows if row["signal_type"] == "MARKET_CAP_XG"]
    ) is expected_market_cap_hit


class FakeDatabase:
    def initialize(self):
        return None


def formula_main_args():
    return [
        "--end-date",
        "2026-07-10",
        "--lookback",
        "21",
        "--workers",
        "1",
        "--metadata-source",
        "akshare",
        "--price-source",
        "akshare",
    ]


def stub_formula_identity(monkeypatch, tmp_path):
    dates = pd.bdate_range(end="2026-07-10", periods=61).strftime("%Y-%m-%d").tolist()
    universe = pd.DataFrame(
        [
            {"code": "sh.600000", "code_name": "PF Bank"},
            {"code": "sz.000001", "code_name": "PA Bank"},
        ]
    )
    manifest_path = tmp_path / "formula33.json"
    universe_cache_path = tmp_path / "stock_universe.csv"
    monkeypatch.setattr(formula33, "Database", FakeDatabase)
    monkeypatch.setattr(formula33, "FORMULA33_MANIFEST_FILE", str(manifest_path))
    monkeypatch.setattr(formula33, "UNIVERSE_CACHE_FILE", str(universe_cache_path))
    monkeypatch.setattr(
        formula33, "FORMULA33_SNAPSHOT_DIR", str(tmp_path / "snapshots")
    )
    monkeypatch.setattr(
        formula33, "get_trade_dates_akshare", lambda *_args, **_kwargs: dates
    )
    monkeypatch.setattr(formula33, "get_universe_akshare", lambda: universe.copy())
    monkeypatch.setattr(
        formula33,
        "load_observation_spot_snapshot",
        lambda *_args, **_kwargs: pd.DataFrame({"snapshot": [1]}),
    )
    monkeypatch.setattr(
        formula33,
        "build_observation_trade_status",
        lambda selected, *_args, **_kwargs: (
            {code: "traded" for code in selected["code"]},
            1.0,
        ),
    )
    return manifest_path, universe


def test_matching_manifest_returns_before_market_cap_and_stock_loaders(
    monkeypatch, tmp_path, capsys
):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    universe.to_csv(formula33.UNIVERSE_CACHE_FILE, index=False)
    xlsx_path = tmp_path / "formula.xlsx"
    csv_path = tmp_path / "formula.csv"
    xlsx_path.write_bytes(b"xlsx")
    csv_path.write_text("date,count\n", encoding="utf-8")
    args = formula33.parse_args(formula_main_args())
    CompletionManifest(manifest_path).finish(
        observation_date="2026-07-10",
        arguments=formula33.build_completion_arguments(args, "2026-07-10"),
        universe_codes=universe["code"].tolist(),
        outputs=[xlsx_path, csv_path],
        summary={"universe": len(universe)},
        code_version=formula33.FORMULA33_CODE_VERSION,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network-backed loader must not be called")

    monkeypatch.setattr(formula33, "load_stock_basic_akshare", fail_if_called)
    monkeypatch.setattr(formula33, "load_market_caps", fail_if_called)
    monkeypatch.setattr(formula33, "fetch_one_stock", fail_if_called)
    monkeypatch.setattr(formula33, "get_universe_akshare", fail_if_called)
    monkeypatch.setattr(formula33, "get_trade_dates_akshare", fail_if_called)

    assert formula33.main(formula_main_args()) == 0
    assert "Formula33 resume: completed manifest hit date=2026-07-10; network_fetch=0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("observation_date", "requested_date", "expected"),
    [
        ("2026-07-10", "2026-07-10", True),
        ("2026-07-10", "2026-07-11", True),
        ("2026-07-10", "2026-07-12", True),
        ("2026-07-09", "2026-07-11", False),
        ("2026-07-08", "2026-07-11", False),
        ("2026-07-10", "2026-07-13", False),
    ],
)
def test_manifest_local_fast_path_only_carries_friday_across_its_weekend(
    monkeypatch,
    tmp_path,
    observation_date,
    requested_date,
    expected,
):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    universe.to_csv(formula33.UNIVERSE_CACHE_FILE, index=False)
    output = tmp_path / "formula.csv"
    output.write_text("date,count\n", encoding="utf-8")
    argv = formula_main_args()
    argv[1] = requested_date
    args = formula33.parse_args(argv)
    CompletionManifest(manifest_path).finish(
        observation_date=observation_date,
        arguments=formula33.build_completion_arguments(args, observation_date),
        universe_codes=universe["code"].tolist(),
        outputs=[output],
        summary={"universe": len(universe)},
        code_version=formula33.FORMULA33_CODE_VERSION,
    )

    assert formula33.reuse_completed_manifest_without_network(args) is expected


def test_cancelled_formula_run_does_not_write_completed_manifest(monkeypatch, tmp_path):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(
        formula33,
        "load_stock_basic_akshare",
        lambda: pd.DataFrame(
            {
                "code": universe["code"],
                "ipoDate": ["2000-01-01", "2000-01-01"],
            }
        ),
    )
    monkeypatch.setattr(
        formula33,
        "load_market_caps",
        lambda *_args, **_kwargs: ({code: 200.0 for code in universe["code"]}, "akshare"),
    )

    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(formula33, "fetch_one_stock", cancel)

    with pytest.raises(KeyboardInterrupt):
        formula33.main(formula_main_args())

    assert not Path(manifest_path).exists()


def test_transient_unavailable_stock_does_not_complete_manifest(monkeypatch, tmp_path):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(
        formula33,
        "load_stock_basic_akshare",
        lambda: pd.DataFrame(
            {
                "code": universe["code"],
                "ipoDate": ["2000-01-01", "2000-01-01"],
            }
        ),
    )
    monkeypatch.setattr(
        formula33,
        "load_market_caps",
        lambda *_args, **_kwargs: (
            {code: 200.0 for code in universe["code"]},
            "akshare",
        ),
    )
    monkeypatch.setattr(
        formula33,
        "fetch_one_stock",
        lambda task: [
            {
                "signal_type": "STATUS",
                "code": task[0],
                "name": task[1],
                "latest_data_date": "2026-07-09",
                "observation_status": "data_unavailable",
                "error": "source timeout",
            }
        ],
    )

    def save_outputs(*_args, **_kwargs):
        xlsx = tmp_path / "formula.xlsx"
        csv = tmp_path / "formula.csv"
        xlsx.write_bytes(b"xlsx")
        csv.write_text("date,count\n", encoding="utf-8")
        return str(xlsx), str(csv)

    monkeypatch.setattr(formula33, "save_workbook", save_outputs)

    with pytest.raises(RuntimeError, match="retryable unavailable"):
        formula33.main(formula_main_args())

    assert not Path(manifest_path).exists()


def test_completion_arguments_normalize_dates_and_ignore_worker_tuning():
    first = formula33.parse_args(
        ["--start-date", "2026/06/01", "--workers", "1", "--retry-delay", "1"]
    )
    second = formula33.parse_args(
        ["--start-date", "2026-06-01", "--workers", "8", "--retry-delay", "9"]
    )

    assert formula33.build_completion_arguments(
        first, "2026-07-10"
    ) == formula33.build_completion_arguments(second, "2026-07-10")


def test_lookup_coverage_rejects_a_silently_missing_market_segment():
    universe = pd.DataFrame(
        {"code": ["sh.600000", "sh.600001", "sz.000001", "sz.000002"]}
    )

    with pytest.raises(RuntimeError, match="上市日期覆盖率"):
        formula33.require_lookup_coverage(
            "上市日期",
            universe,
            {"sh.600000": "2000-01-01", "sh.600001": "2000-01-02"},
            minimum=0.98,
        )


def test_main_rejects_incomplete_listing_date_coverage_without_manifest(
    monkeypatch, tmp_path
):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(
        formula33,
        "load_stock_basic_akshare",
        lambda: pd.DataFrame(
            {"code": [universe.iloc[0]["code"]], "ipoDate": ["2000-01-01"]}
        ),
    )
    monkeypatch.setattr(
        formula33,
        "load_market_caps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("market-cap loading must not start")
        ),
    )
    monkeypatch.setattr(
        formula33,
        "fetch_one_stock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stock scanning must not start")
        ),
    )

    with pytest.raises(RuntimeError):
        formula33.main(formula_main_args())

    assert not Path(manifest_path).exists()


def test_main_rejects_incomplete_market_cap_coverage_without_manifest(
    monkeypatch, tmp_path
):
    manifest_path, universe = stub_formula_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(
        formula33,
        "load_stock_basic_akshare",
        lambda: pd.DataFrame(
            {
                "code": universe["code"],
                "ipoDate": ["2000-01-01", "2000-01-01"],
            }
        ),
    )
    monkeypatch.setattr(
        formula33,
        "load_market_caps",
        lambda *_args, **_kwargs: ({universe.iloc[0]["code"]: 200.0}, "akshare"),
    )
    monkeypatch.setattr(
        formula33,
        "fetch_one_stock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stock scanning must not start")
        ),
    )

    with pytest.raises(RuntimeError):
        formula33.main(formula_main_args())

    assert not Path(manifest_path).exists()


def test_trade_calendar_snapshot_is_reused_only_when_it_covers_request(
    monkeypatch, tmp_path
):
    calendar_path = tmp_path / "trade_calendar.json"
    monkeypatch.setattr(
        formula33, "TRADE_CALENDAR_CACHE_FILE", str(calendar_path)
    )
    calls = []

    def fetch_calendar():
        calls.append(1)
        if len(calls) > 1:
            raise AssertionError("calendar network called again")
        return pd.DataFrame(
            {
                "trade_date": pd.bdate_range(
                    "2026-06-01", "2026-07-10"
                )
            }
        )

    monkeypatch.setattr(
        formula33.ak, "tool_trade_date_hist_sina", fetch_calendar
    )

    first = formula33.get_trade_dates_akshare(
        10, 60, required_through="2026-07-10"
    )
    second = formula33.get_trade_dates_akshare(
        10, 60, required_through="2026-07-10"
    )

    assert first == second
    assert first[-1] == "2026-07-10"
    assert len(calls) == 1
    with pytest.raises(AssertionError, match="called again"):
        formula33.get_trade_dates_akshare(
            10, 60, required_through="2026-07-13"
        )


def test_formula_input_snapshots_reuse_same_observation_and_rotate_next_date(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        formula33, "FORMULA33_SNAPSHOT_DIR", str(tmp_path / "snapshots")
    )
    monkeypatch.setattr(
        formula33, "UNIVERSE_CACHE_FILE", str(tmp_path / "stock_universe.csv")
    )
    universe = pd.DataFrame(
        {
            "code": ["sh.600000", "sz.000001"],
            "code_name": ["PF Bank", "PA Bank"],
        }
    )
    basic = pd.DataFrame(
        {
            "code": universe["code"],
            "ipoDate": ["2000-01-01", "2000-01-01"],
        }
    )
    universe_calls = []
    basic_calls = []
    cap_calls = []
    monkeypatch.setattr(
        formula33,
        "get_universe_akshare",
        lambda: universe_calls.append(1) or universe.copy(),
    )
    monkeypatch.setattr(
        formula33,
        "load_stock_basic_akshare",
        lambda: basic_calls.append(1) or basic.copy(),
    )
    monkeypatch.setattr(
        formula33,
        "load_market_caps",
        lambda *_args, **_kwargs: (
            cap_calls.append(1)
            or ({code: 200.0 for code in universe["code"]}, "akshare-capital")
        ),
    )

    first_universe = formula33.load_universe_snapshot("2026-07-10")
    second_universe = formula33.load_universe_snapshot("2026-07-10")
    first_basic = formula33.load_stock_basic_snapshot(
        "2026-07-10", first_universe
    )
    second_basic = formula33.load_stock_basic_snapshot(
        "2026-07-10", first_universe
    )
    first_caps = formula33.load_market_cap_snapshot(
        "auto", "2026-07-10", first_universe
    )
    second_caps = formula33.load_market_cap_snapshot(
        "auto", "2026-07-10", first_universe
    )

    assert second_universe.equals(first_universe)
    assert second_basic.equals(first_basic)
    assert second_caps == first_caps
    assert universe_calls == [1]
    assert basic_calls == [1]
    assert cap_calls == [1]

    formula33.load_universe_snapshot("2026-07-13")
    formula33.load_stock_basic_snapshot("2026-07-13", universe)
    formula33.load_market_cap_snapshot("auto", "2026-07-13", universe)
    assert universe_calls == [1, 1]
    assert basic_calls == [1, 1]
    assert cap_calls == [1, 1]
