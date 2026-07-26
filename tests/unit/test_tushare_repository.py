import pandas as pd

from stock_research.storage import Database, TushareRepository
from stock_research.storage.tushare_repository import normalize_tushare_code


def test_tushare_repository_roundtrips_dataset_rows(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)

    rows = repository.upsert_dataset(
        "daily_basic",
        pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260720",
                "pe": 8.5,
                "pb": 0.75,
            }
        ]),
        params={"trade_date": "20260720"},
    )
    loaded = repository.load_dataset("daily_basic", ts_code="000001.SZ")

    assert rows == 1
    assert loaded.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": "20260720",
            "pe": 8.5,
            "pb": 0.75,
        }
    ]


def test_tushare_daily_kline_loader_normalizes_codes_and_filters_future_rows(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)
    repository.upsert_dataset(
        "daily_kline",
        pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260720",
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "vol": 1000,
                "amount": 2000,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260721",
                "open": 99.0,
                "high": 99.0,
                "low": 99.0,
                "close": 99.0,
                "vol": 1,
                "amount": 1,
            },
        ]),
    )
    repository.upsert_dataset(
        "adj_factor",
        pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260720",
                "adj_factor": 2.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260721",
                "adj_factor": 3.0,
            },
        ]),
    )

    loaded = repository.load_daily_kline_frames(
        ["sz.000001"],
        start_date="2026-07-01",
        end_date="2026-07-20",
    )

    assert normalize_tushare_code("sz.000001") == "000001.SZ"
    assert loaded.to_dict("records") == [
        {
            "date": "2026-07-20",
            "code": "sz.000001",
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "volume": 1000,
            "amount": 2000,
            "tradestatus": "1",
            "raw_to_qfq_factor": 0.5,
        }
    ]
