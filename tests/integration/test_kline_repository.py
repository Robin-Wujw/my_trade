import pandas as pd
import pytest
import duckdb

import stock_research.storage.kline_repository as kline_repository_module
from stock_research.storage import Database, KlineRepository


def test_kline_repository_upserts_and_loads_stock_kline(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")

    rows = pd.DataFrame(
        [
            {
                "date": "2026-07-08",
                "code": "000001",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 1000,
                "tradestatus": "1",
            },
            {
                "date": "2026-07-09",
                "code": "000001",
                "open": 10.5,
                "high": 12,
                "low": 10,
                "close": 11.5,
                "volume": 1200,
                "tradestatus": "1",
            },
        ]
    )

    assert repository.upsert_stock_kline("akshare", "000001", rows) == 2

    loaded = repository.load_stock_kline(
        "akshare",
        "000001",
        start_date="2026-07-09",
        end_date="2026-07-09",
    )

    assert loaded[["date", "code", "close"]].to_dict("records") == [
        {"date": "2026-07-09", "code": "000001", "close": 11.5}
    ]


def test_kline_repository_loads_multiple_codes_in_one_batch(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")
    for code, close in [("sh.600000", 10.5), ("sz.000001", 12.5)]:
        repository.upsert_stock_kline(
            "akshare",
            code,
            pd.DataFrame([{
                "date": "2026-07-09", "open": close, "high": close + 1,
                "low": close - 1, "close": close, "volume": 1000,
                "tradestatus": "1",
            }]),
        )

    loaded = repository.load_stock_klines(
        "akshare",
        ["sz.000001", "sh.600000", "missing"],
        start_date="2026-07-09",
        end_date="2026-07-09",
    )

    assert loaded[["code", "close"]].to_dict("records") == [
        {"code": "sh.600000", "close": 10.5},
        {"code": "sz.000001", "close": 12.5},
    ]


def test_kline_repository_load_uses_process_lock(tmp_path, monkeypatch):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
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
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 1000,
                    "tradestatus": "1",
                }
            ]
        ),
    )
    original_process_lock = kline_repository_module._process_lock
    lock_paths = []

    def tracking_process_lock(path):
        lock_paths.append(path)
        return original_process_lock(path)

    monkeypatch.setattr(
        kline_repository_module,
        "_process_lock",
        tracking_process_lock,
    )

    loaded = repository.load_stock_kline(
        "akshare",
        "000001",
        start_date="2026-07-09",
        end_date="2026-07-09",
    )

    assert loaded["close"].tolist() == [10.5]
    assert lock_paths == [repository.lock_path]


def test_process_lock_retries_when_lock_is_temporarily_busy(tmp_path, monkeypatch):
    attempts = []
    sleeps = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fileno, mode, _size):
            attempts.append(mode)
            if mode == FakeMsvcrt.LK_LOCK and attempts.count(mode) < 3:
                raise PermissionError("busy")

    monkeypatch.setattr(kline_repository_module, "msvcrt", FakeMsvcrt)
    monkeypatch.setattr(
        kline_repository_module,
        "time",
        type("FakeTime", (), {"sleep": staticmethod(sleeps.append)}),
        raising=False,
    )

    with kline_repository_module._process_lock(tmp_path / "kline.lock"):
        pass

    assert attempts == [
        FakeMsvcrt.LK_LOCK,
        FakeMsvcrt.LK_LOCK,
        FakeMsvcrt.LK_LOCK,
        FakeMsvcrt.LK_UNLCK,
    ]
    assert sleeps == pytest.approx([0.1, 0.2])


def test_repository_retries_transient_duckdb_lock_error(tmp_path, monkeypatch):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(
        database,
        lock_path=tmp_path / "kline.lock",
        lock_retry_delay=0.01,
    )
    original_connect = database.connect
    attempts = []
    sleeps = []

    def temporarily_locked_connect(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise duckdb.IOException("Could not set lock on file: conflicting lock")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(database, "connect", temporarily_locked_connect)
    monkeypatch.setattr(kline_repository_module.time, "sleep", sleeps.append)

    loaded = repository.load_stock_kline(
        "akshare",
        "000001",
        start_date="2026-07-09",
        end_date="2026-07-09",
    )

    assert loaded.empty
    assert len(attempts) == 3
    assert sleeps == pytest.approx([0.01, 0.02])


def test_repository_does_not_retry_non_lock_duckdb_io_error(tmp_path, monkeypatch):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")
    attempts = []

    def broken_connect(*_args, **_kwargs):
        attempts.append(1)
        raise duckdb.IOException("Cannot open file: invalid path")

    monkeypatch.setattr(database, "connect", broken_connect)

    with pytest.raises(duckdb.IOException, match="invalid path"):
        repository.load_stock_kline(
            "akshare",
            "000001",
            start_date="2026-07-09",
            end_date="2026-07-09",
        )

    assert len(attempts) == 1


def test_repository_raises_after_final_transient_write_retry(tmp_path, monkeypatch):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(
        database,
        lock_path=tmp_path / "kline.lock",
        lock_retry_attempts=3,
        lock_retry_delay=0,
    )
    attempts = []

    def always_locked_connect(*_args, **_kwargs):
        attempts.append(1)
        raise duckdb.IOException("Could not set lock on file: conflicting lock")

    monkeypatch.setattr(database, "connect", always_locked_connect)

    with pytest.raises(duckdb.IOException, match="conflicting lock"):
        repository.upsert_stock_kline(
            "akshare",
            "000001",
            pd.DataFrame(
                [
                    {
                        "date": "2026-07-10",
                        "code": "000001",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                        "tradestatus": "1",
                    }
                ]
            ),
        )

    assert len(attempts) == 3


def test_repository_replaces_adjusted_price_range_atomically(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    repository = KlineRepository(database, lock_path=tmp_path / "kline.lock")
    repository.upsert_stock_kline(
        "akshare",
        "000001",
        pd.DataFrame(
            [
                {
                    "date": date,
                    "code": "000001",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 1000,
                    "tradestatus": "1",
                }
                for date in ["2026-07-08", "2026-07-09"]
            ]
        ),
    )

    replaced = repository.replace_stock_kline_range(
        "akshare",
        "000001",
        pd.DataFrame(
            [
                {
                    "date": "2026-07-08",
                    "code": "000001",
                    "open": 5,
                    "high": 5.5,
                    "low": 4.5,
                    "close": 5.25,
                    "volume": 1000,
                    "tradestatus": "1",
                }
            ]
        ),
        start_date="2026-07-08",
        end_date="2026-07-09",
    )

    loaded = repository.load_stock_kline(
        "akshare",
        "000001",
        start_date="2026-07-08",
        end_date="2026-07-09",
    )
    assert replaced == 1
    assert loaded[["date", "close"]].to_dict("records") == [
        {"date": "2026-07-08", "close": 5.25}
    ]
