import json

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


def test_tushare_repository_loads_dataset_for_multiple_codes(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)
    repository.upsert_dataset(
        "daily_basic",
        pd.DataFrame([
            {
                "ts_code": "600000.SH",
                "trade_date": "20240102",
                "turnover_rate": 1.2,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "turnover_rate": 2.3,
            },
        ]),
    )

    frame = repository.load_dataset_for_codes(
        "daily_basic",
        ["sh.600000", "sz.000001"],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )

    assert set(frame["code"]) == {"sh.600000", "sz.000001"}
    assert set(frame["turnover_rate"]) == {1.2, 2.3}


def test_tushare_repository_loads_complete_full_market_daily_version(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)
    repository.upsert_dataset(
        "daily_kline",
        pd.DataFrame([{
            "ts_code": "600000.SH",
            "trade_date": "20240102",
            "open": 10.0,
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "vol": 1000,
            "amount": 1020,
        }]),
    )
    repository.upsert_dataset(
        "adj_factor",
        pd.DataFrame([{
            "ts_code": "600000.SH",
            "trade_date": "20240102",
            "adj_factor": 2.0,
        }]),
    )
    connection = database.connect()
    try:
        for row_key, payload in (
            (
                "daily_basic|short",
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20240102",
                    "turnover_rate": 1.2,
                },
            ),
            (
                "daily_basic|complete",
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20240102",
                    "turnover_rate": 1.3,
                    "turnover_rate_f": 1.5,
                    "total_mv": 100_000,
                    "circ_mv": 80_000,
                    "pe_ttm": 9.0,
                    "pb": 1.1,
                },
            ),
        ):
            connection.execute(
                """
                INSERT INTO raw.tushare_dataset_rows (
                    dataset, row_key, ts_code, trade_date, source, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    "daily_basic",
                    row_key,
                    "600000.SH",
                    "2024-01-02",
                    "test",
                    json.dumps(payload),
                ],
            )
        connection.commit()
    finally:
        connection.close()

    frame = repository.load_market_daily_frame(
        start_date="2024-01-02",
        end_date="2024-01-02",
    )

    row = frame.iloc[0]
    assert row["code"] == "sh.600000"
    assert row["turnover_rate"] == 1.3
    assert row["total_mv"] == 100_000
    assert row["raw_to_qfq_factor"] == 0.5


def test_tushare_dividend_loader_filters_by_ex_date_and_project_code(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)
    repository.upsert_dataset(
        "dividend",
        pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "end_date": "20251231",
                "ex_date": "20260504",
                "stk_bo_rate": 0.3,
                "stk_co_rate": 0.3,
                "stk_div": 0.6,
                "cash_div": 0.1,
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20261231",
                "ex_date": "20270504",
                "stk_bo_rate": 0.1,
            },
        ]),
    )

    actions = repository.load_dividend_actions(
        ["sz.000001"],
        start_date="2026-01-01",
        end_date="2026-12-31",
    )

    assert list(actions) == ["sz.000001"]
    assert len(actions["sz.000001"]) == 1
    assert actions["sz.000001"][0]["ex_date"] == "20260504"
    assert actions["sz.000001"][0]["stk_div"] == 0.6


def test_tushare_event_datasets_keep_distinct_event_rows(tmp_path):
    database = Database(tmp_path / "my_trade.sqlite3", code_version="test")
    database.initialize()
    repository = TushareRepository(database)

    dividend_rows = repository.upsert_dataset(
        "dividend",
        pd.DataFrame([
            {
                "ts_code": "300751.SZ",
                "ann_date": "20260428",
                "end_date": "20251231",
                "div_proc": "预案",
                "cash_div": 0.0,
                "cash_div_tax": 0.5,
            },
            {
                "ts_code": "300751.SZ",
                "ann_date": "20260428",
                "end_date": "20251231",
                "record_date": "20260528",
                "ex_date": "20260529",
                "div_proc": "实施",
                "cash_div": 0.5,
                "cash_div_tax": 0.5,
            },
        ]),
    )
    share_float_rows = repository.upsert_dataset(
        "share_float",
        pd.DataFrame([
            {
                "ts_code": "300751.SZ",
                "ann_date": "20210127",
                "float_date": "20240201",
                "float_share": 2187978.0,
                "float_ratio": 3.8219,
                "holder_name": "王正根",
                "share_type": "定增股份",
            },
            {
                "ts_code": "300751.SZ",
                "ann_date": "20210127",
                "float_date": "20240201",
                "float_share": 2853447.0,
                "float_ratio": 4.9843,
                "holder_name": "周剑",
                "share_type": "定增股份",
            },
        ]),
    )

    assert dividend_rows == 2
    assert share_float_rows == 2
    assert len(repository.load_dataset("dividend", ts_code="300751.SZ")) == 2
    assert len(repository.load_dataset("share_float", ts_code="300751.SZ")) == 2
