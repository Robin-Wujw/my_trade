from pathlib import Path

import pandas as pd
import pytest

from stock_research.pipelines import formula33
from stock_research.storage import Database, KlineRepository


def _frame_for(dates, code="sz.000001", scale=1.0):
    if isinstance(dates, str):
        dates = [dates]
    return pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 10.0 * scale,
                "high": 12.0 * scale,
                "low": 9.0 * scale,
                "close": 11.0 * scale,
                "volume": 1000.0,
                "tradestatus": "1",
            }
            for date in dates
        ]
    )


def _make_repository(tmp_path, rows):
    database = Database(tmp_path / "kline.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")
    repository.upsert_stock_kline("akshare", "sz.000001", rows)
    return repository


def _spot_frame(codes, suspended=()):
    suspended = set(suspended)
    rows = []
    for code in codes:
        is_suspended = code in suspended
        rows.append(
            {
                "代码": code.replace(".", ""),
                "名称": code,
                "最新价": 10.0,
                "今开": 0.0 if is_suspended else 10.0,
                "最高": 0.0 if is_suspended else 11.0,
                "最低": 0.0 if is_suspended else 9.0,
                "成交量": 0.0 if is_suspended else 1000.0,
                "成交额": 0.0 if is_suspended else 10000.0,
                "时间戳": "15:35:00",
            }
        )
    return pd.DataFrame(rows)


def test_cdr_kline_uses_akshare_tencent_qfq_route(monkeypatch):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "date": ["2026-07-09", "2026-07-10"],
                "open": [37.88, 38.20],
                "close": [38.26, 37.82],
                "high": [38.73, 39.00],
                "low": [37.40, 37.51],
                "amount": [11059424.0, 11886669.0],
            }
        )

    monkeypatch.setattr(formula33.ak, "stock_zh_a_hist_tx", fetch)
    monkeypatch.setattr(
        formula33.ak,
        "stock_zh_a_daily",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("CDR must not use the ordinary A-share route")
        ),
    )

    result = formula33.load_kline_akshare(
        "sh.689009",
        "2026-07-09",
        "2026-07-10",
        retries=1,
        retry_delay=0,
    )

    assert calls == [
        {
            "symbol": "sh689009",
            "start_date": "20260709",
            "end_date": "20260710",
            "adjust": "qfq",
            "timeout": 15,
        }
    ]
    assert result["volume"].tolist() == [11059424.0, 11886669.0]
    assert result["code"].tolist() == ["sh.689009", "sh.689009"]


def test_tushare_kline_calculates_end_date_anchored_qfq(monkeypatch):
    daily = pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "trade_date": ["20260102", "20260105"],
        "open": [10.0, 12.0],
        "high": [11.0, 13.0],
        "low": [9.0, 11.0],
        "close": [10.0, 12.0],
        "vol": [100.0, 120.0],
    })
    factors = pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "trade_date": ["20260102", "20260105"],
        "adj_factor": [1.0, 2.0],
    })
    monkeypatch.setattr(
        formula33.ts_api,
        "query",
        lambda name, **_kwargs: daily if name == "daily" else factors,
    )

    result = formula33.load_kline_tushare(
        "sz.000001", "2026-01-01", "2026-01-05", retries=1
    )

    assert result["date"].tolist() == ["2026-01-02", "2026-01-05"]
    assert result["close"].tolist() == [5.0, 12.0]
    assert result["volume"].tolist() == [100.0, 120.0]


def test_explicit_tushare_source_requires_token(monkeypatch):
    monkeypatch.setattr(formula33, "get_tushare_token", lambda: "")

    with pytest.raises(RuntimeError, match="需要配置"):
        formula33.resolve_price_source(
            "tushare", "2026-01-01", "2026-01-10", 1, 0
        )


def test_explicit_tushare_source_does_not_consume_probe_quota(monkeypatch):
    monkeypatch.setattr(formula33, "get_tushare_token", lambda: "configured")
    monkeypatch.setattr(
        formula33,
        "load_kline_tushare",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source resolution must not consume adj_factor quota")
        ),
    )

    assert formula33.resolve_price_source(
        "tushare", "2026-01-01", "2026-01-10", 1, 0
    ) == "tushare"


def test_duckdb_value_wins_over_stale_csv_for_the_same_trade_date(
    monkeypatch,
    tmp_path,
):
    database = Database(tmp_path / "kline.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")
    repository.upsert_stock_kline(
        "akshare",
        "000001",
        pd.DataFrame(
            [
                {
                    "date": "2026-07-09",
                    "code": "000001",
                    "open": 10.0,
                    "high": 12.0,
                    "low": 9.0,
                    "close": 11.5,
                    "volume": 1000.0,
                    "tradestatus": "1",
                }
            ]
        ),
    )
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    csv_path = Path(formula33.kline_cache_path("akshare", "000001"))
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-07-09",
                "code": "000001",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 9.5,
                "volume": 1000.0,
                "tradestatus": "1",
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network called for a fully cached range")
        ),
    )

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "000001",
        "2026-07-09",
        "2026-07-09",
        repository=repository,
    )

    assert loaded["close"].tolist() == [11.5]
    repaired_csv = formula33.load_cached_kline("akshare", "000001")
    assert repaired_csv["close"].tolist() == [11.5]


def test_fully_cached_range_does_not_call_network(monkeypatch, tmp_path):
    rows = _frame_for(pd.date_range("2026-07-06", "2026-07-10", freq="D").strftime("%Y-%m-%d"))
    repository = _make_repository(tmp_path, rows)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network called for a fully cached range")

    monkeypatch.setattr(formula33, "load_kline_akshare", fail_if_called)

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-06",
        "2026-07-10",
        repository=repository,
    )

    assert loaded["date"].tolist() == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
    ]
    assert formula33.kline_cache_metadata_matches(
        "akshare",
        "sz.000001",
        formula33.load_cached_kline("akshare", "sz.000001"),
    )
    cache_event = {}
    formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-06",
        "2026-07-10",
        repository=repository,
        cache_event=cache_event,
    )
    assert cache_event == {"complete_file_cache": True}

    class RepositoryMustNotOpen:
        @staticmethod
        def load_stock_kline(*_args, **_kwargs):
            raise AssertionError("validated CSV fast path opened DuckDB")

    repeated = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-06",
        "2026-07-10",
        repository=RepositoryMustNotOpen(),
    )
    assert repeated["date"].tolist() == loaded["date"].tolist()


def test_cached_friday_fetches_only_missing_monday(monkeypatch, tmp_path):
    rows = _frame_for(pd.date_range("2026-04-01", "2026-07-10", freq="D").strftime("%Y-%m-%d"))
    repository = _make_repository(tmp_path, rows)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def fetch(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return _frame_for(["2026-07-10", "2026-07-13"])

    monkeypatch.setattr(formula33, "load_kline_akshare", fetch)

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-04-01",
        "2026-07-13",
        repository=repository,
        expected_trade_dates={"2026-07-13"},
    )

    assert calls == [("2026-07-10", "2026-07-13")]
    assert loaded["date"].max() == "2026-07-13"


def test_qfq_change_on_overlap_refreshes_the_full_required_window(
    monkeypatch,
    tmp_path,
):
    old = _frame_for(["2026-07-08", "2026-07-09", "2026-07-10"])
    repository = _make_repository(tmp_path, old)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def fetch(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        if len(calls) == 1:
            return _frame_for(
                ["2026-07-10", "2026-07-13"],
                scale=0.5,
            )
        return _frame_for(
            ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13"],
            scale=0.5,
        )

    monkeypatch.setattr(formula33, "load_kline_akshare", fetch)

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-08",
        "2026-07-13",
        repository=repository,
        expected_trade_dates={"2026-07-13"},
    )

    persisted = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-08",
        end_date="2026-07-13",
    )
    assert calls == [
        ("2026-07-10", "2026-07-13"),
        ("2026-07-08", "2026-07-13"),
    ]
    assert loaded["close"].tolist() == [5.5, 5.5, 5.5, 5.5]
    assert persisted["close"].tolist() == [5.5, 5.5, 5.5, 5.5]


def test_001331_cross_suspension_qfq_rebase_replaces_csv_and_duckdb(
    monkeypatch,
    tmp_path,
):
    code = "sz.001331"
    old = pd.DataFrame(
        [
            {
                "date": "2026-05-26",
                "code": code,
                "open": 68.0,
                "high": 70.0,
                "low": 67.0,
                "close": 68.5,
                "volume": 1000.0,
                "tradestatus": "1",
            },
            {
                "date": "2026-05-27",
                "code": code,
                "open": 69.86,
                "high": 72.56,
                "low": 69.25,
                "close": 69.78,
                "volume": 1200.0,
                "tradestatus": "1",
            },
        ]
    )
    post_resume = pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "code": code,
                "open": 54.89,
                "high": 62.02,
                "low": 54.88,
                "close": 62.02,
                "volume": 1400.0,
                "tradestatus": "1",
            },
            {
                "date": "2026-07-02",
                "code": code,
                "open": 63.05,
                "high": 65.78,
                "low": 60.0,
                "close": 61.6,
                "volume": 1500.0,
                "tradestatus": "1",
            },
        ]
    )
    rebased = pd.concat(
        [
            old.assign(
                open=[46.77, 48.14],
                high=[48.15, 49.99],
                low=[46.08, 47.71],
                close=[47.20, 48.08],
            ),
            post_resume,
        ],
        ignore_index=True,
    )
    database = Database(tmp_path / "001331.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "001331.lock")
    repository.upsert_stock_kline("akshare", code, old)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def fetch(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return post_resume.copy() if len(calls) == 1 else rebased.copy()

    monkeypatch.setattr(formula33, "load_kline_akshare", fetch)

    loaded = formula33.load_kline_with_cache(
        "akshare",
        code,
        "2026-05-26",
        "2026-07-02",
        repository=repository,
        expected_trade_dates={"2026-07-02"},
    )

    persisted = repository.load_stock_kline(
        "akshare",
        code,
        start_date="2026-05-26",
        end_date="2026-07-02",
    )
    cached = formula33.load_cached_kline("akshare", code)
    assert calls == [
        ("2026-05-27", "2026-07-02"),
        ("2026-05-26", "2026-07-02"),
    ]
    assert loaded.loc[loaded["date"] == "2026-05-27", "close"].item() == 48.08
    assert persisted.loc[persisted["date"] == "2026-05-27", "close"].item() == 48.08
    assert cached.loc[cached["date"] == "2026-05-27", "close"].item() == 48.08


def test_partial_duckdb_is_reconciled_from_newer_csv(monkeypatch, tmp_path):
    repository = _make_repository(tmp_path, _frame_for("2026-07-09"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    csv_path = Path(formula33.kline_cache_path("akshare", "sz.000001"))
    csv_path.parent.mkdir(parents=True)
    _frame_for(["2026-07-09", "2026-07-10"]).to_csv(csv_path, index=False)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network called for a range covered by CSV and DuckDB")

    monkeypatch.setattr(formula33, "load_kline_akshare", fail_if_called)

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-09",
        "2026-07-10",
        repository=repository,
    )

    stored = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-10",
        end_date="2026-07-10",
    )
    assert loaded["date"].tolist() == ["2026-07-09", "2026-07-10"]
    assert stored["date"].tolist() == ["2026-07-10"]


def test_fully_cached_range_does_not_apply_network_sleep(monkeypatch, tmp_path):
    repository = _make_repository(
        tmp_path,
        _frame_for(["2026-07-09", "2026-07-10"]),
    )
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(
        formula33.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("sleep called")),
    )

    loaded = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-09",
        "2026-07-10",
        repository=repository,
        request_sleep=5.0,
    )

    assert loaded["date"].tolist() == ["2026-07-09", "2026-07-10"]


def test_fresh_rows_are_saved_to_csv_before_duckdb_write(monkeypatch, tmp_path):
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: _frame_for("2026-07-10"),
    )

    class EmptyRepository:
        @staticmethod
        def load_stock_kline(*_args, **_kwargs):
            return pd.DataFrame()

        @staticmethod
        def upsert_stock_kline(*_args, **_kwargs):
            raise RuntimeError("simulated exhausted DuckDB lock retry")

    with pytest.raises(RuntimeError, match="exhausted DuckDB lock retry"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-10",
            "2026-07-10",
            repository=EmptyRepository(),
        )

    cached = formula33.load_cached_kline("akshare", "sz.000001")
    assert cached["date"].tolist() == ["2026-07-10"]


def test_csv_atomic_write_failure_is_not_swallowed(monkeypatch, tmp_path):
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(
        formula33.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="CSV K-line cache write failed.*disk full"):
        formula33.save_cached_kline(
            "akshare",
            "sz.000001",
            _frame_for("2026-07-10"),
        )

    assert not Path(formula33.kline_cache_path("akshare", "sz.000001")).exists()


def test_duckdb_write_failure_is_not_swallowed():
    class FailingRepository:
        @staticmethod
        def upsert_stock_kline(*_args, **_kwargs):
            raise RuntimeError("write retries exhausted")

    with pytest.raises(RuntimeError, match="DuckDB K-line write failed.*retries exhausted"):
        formula33.save_persisted_kline(
            FailingRepository(),
            "akshare",
            "sz.000001",
            _frame_for("2026-07-10"),
        )


def test_empty_200_missing_expected_trade_date_remains_retryable(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def fetch(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return pd.DataFrame()

    monkeypatch.setattr(formula33, "load_kline_akshare", fetch)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="incomplete K-line response"):
            formula33.load_kline_with_cache(
                "akshare",
                "sz.000001",
                "2026-07-10",
                "2026-07-13",
                repository=repository,
                expected_trade_dates={"2026-07-13"},
            )

    assert calls == [
        ("2026-07-10", "2026-07-13"),
        ("2026-07-10", "2026-07-13"),
    ]


def test_partial_200_missing_expected_trade_date_is_not_persisted(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: _frame_for(["2026-07-10", "2026-07-11"]),
    )

    with pytest.raises(RuntimeError, match="missing expected trade dates: 2026-07-13"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-10",
            "2026-07-13",
            repository=repository,
            expected_trade_dates={"2026-07-13"},
        )

    persisted = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-10",
        end_date="2026-07-13",
    )
    assert persisted["date"].tolist() == ["2026-07-10"]
    assert formula33.load_cached_kline("akshare", "sz.000001").empty


def test_partial_qfq_full_refresh_does_not_replace_cached_window(
    monkeypatch,
    tmp_path,
):
    old = _frame_for(["2026-07-08", "2026-07-09", "2026-07-10"])
    repository = _make_repository(tmp_path, old)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    responses = iter(
        [
            _frame_for(["2026-07-10", "2026-07-13"], scale=0.5),
            _frame_for(["2026-07-10", "2026-07-13"], scale=0.5),
        ]
    )
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="incomplete QFQ full-window refresh"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-08",
            "2026-07-13",
            repository=repository,
            expected_trade_dates={"2026-07-13"},
        )

    persisted = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-08",
        end_date="2026-07-13",
    )
    assert persisted["date"].tolist() == [
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
    ]
    assert persisted["close"].tolist() == [11.0, 11.0, 11.0]


def test_qfq_refresh_invalidates_fast_path_before_cross_store_write(
    monkeypatch,
    tmp_path,
):
    old = _frame_for(["2026-07-08", "2026-07-09", "2026-07-10"])
    repository = _make_repository(tmp_path, old)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    formula33.save_cached_kline("akshare", "sz.000001", old)
    formula33.save_kline_cache_metadata("akshare", "sz.000001", old)
    responses = iter(
        [
            _frame_for(["2026-07-10", "2026-07-13"], scale=0.5),
            _frame_for(
                ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13"],
                scale=0.5,
            ),
        ]
    )
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        formula33,
        "save_cached_kline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated CSV write failure")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated CSV write failure"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-08",
            "2026-07-13",
            repository=repository,
            expected_trade_dates={"2026-07-13"},
        )

    assert not Path(
        formula33.kline_cache_metadata_path("akshare", "sz.000001")
    ).exists()
    persisted = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-08",
        end_date="2026-07-13",
    )
    assert persisted["close"].tolist() == [5.5, 5.5, 5.5, 5.5]


def test_old_stock_shallow_cached_response_is_refetched_and_stays_retryable(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def partial(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return _frame_for("2026-07-10")

    monkeypatch.setattr(formula33, "load_kline_akshare", partial)

    with pytest.raises(RuntimeError, match="history rows below 3"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-08",
            "2026-07-10",
            repository=repository,
            expected_trade_dates={"2026-07-10"},
            minimum_history_rows=3,
        )

    assert calls == [("2026-07-08", "2026-07-10")]
    persisted = repository.load_stock_kline(
        "akshare",
        "sz.000001",
        start_date="2026-07-08",
        end_date="2026-07-10",
    )
    assert persisted["date"].tolist() == ["2026-07-10"]


def test_observation_spot_snapshot_is_cached_by_bound_date(monkeypatch, tmp_path):
    monkeypatch.setattr(
        formula33,
        "OBSERVATION_SPOT_CACHE_DIR",
        str(tmp_path / "spot"),
    )
    calls = []

    def fetch():
        calls.append(1)
        return _spot_frame(["sh.600000", "sz.000001"])

    monkeypatch.setattr(formula33.ak, "stock_zh_a_spot", fetch)
    first = formula33.load_observation_spot_snapshot(
        "2026-07-10",
        allow_network=True,
    )
    cached = formula33.load_observation_spot_snapshot(
        "2026-07-10",
        allow_network=False,
    )
    historical_without_cache = formula33.load_observation_spot_snapshot(
        "2026-07-09",
        allow_network=False,
    )

    assert calls == [1]
    assert set(first["observation_date"]) == {"2026-07-10"}
    assert len(cached) == 2
    assert historical_without_cache.empty


def test_observation_trade_status_requires_coverage_and_classifies_suspension():
    codes = [f"sz.{index:06d}" for index in range(1, 101)]
    universe = pd.DataFrame({"code": codes})
    snapshot = _spot_frame(codes, suspended={codes[-1]})

    statuses, coverage = formula33.build_observation_trade_status(
        universe,
        snapshot,
        minimum=0.98,
    )

    assert coverage == 1.0
    assert statuses[codes[0]] == "traded"
    assert statuses[codes[-1]] == "suspended"

    with pytest.raises(RuntimeError, match="coverage insufficient"):
        formula33.build_observation_trade_status(
            universe,
            snapshot.head(97),
            minimum=0.98,
        )


def test_confirmed_suspension_with_overlap_is_marked_and_reused(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    calls = []

    def fetch(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return _frame_for("2026-07-10")

    monkeypatch.setattr(formula33, "load_kline_akshare", fetch)
    first = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-10",
        "2026-07-13",
        repository=repository,
        expected_trade_dates={"2026-07-13"},
        observation_trade_status="suspended",
    )
    marker = formula33.load_kline_no_trade_marker("akshare", "sz.000001")

    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("confirmed no-trade marker was not reused")
        ),
    )
    second = formula33.load_kline_with_cache(
        "akshare",
        "sz.000001",
        "2026-07-10",
        "2026-07-13",
        repository=repository,
        expected_trade_dates={"2026-07-13"},
        observation_trade_status="suspended",
    )

    assert calls == [("2026-07-10", "2026-07-13")]
    assert marker["observation_date"] == "2026-07-13"
    assert first["date"].tolist() == ["2026-07-10"]
    assert second["date"].tolist() == ["2026-07-10"]


def test_no_trade_marker_is_ignored_without_current_suspension_confirmation(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    formula33.save_kline_no_trade_marker("akshare", "sz.000001", "2026-07-13")
    calls = []

    def empty(_code, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        return pd.DataFrame()

    monkeypatch.setattr(formula33, "load_kline_akshare", empty)

    with pytest.raises(RuntimeError, match="incomplete K-line response"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-10",
            "2026-07-13",
            repository=repository,
            expected_trade_dates={"2026-07-13"},
            observation_trade_status="traded",
        )

    assert calls == [("2026-07-10", "2026-07-13")]


def test_confirmed_suspension_still_rejects_silent_empty_response(
    monkeypatch,
    tmp_path,
):
    repository = _make_repository(tmp_path, _frame_for("2026-07-10"))
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="did not include a cached overlap date"):
        formula33.load_kline_with_cache(
            "akshare",
            "sz.000001",
            "2026-07-10",
            "2026-07-13",
            repository=repository,
            expected_trade_dates={"2026-07-13"},
            observation_trade_status="suspended",
        )

    assert formula33.load_kline_no_trade_marker("akshare", "sz.000001") == {}
