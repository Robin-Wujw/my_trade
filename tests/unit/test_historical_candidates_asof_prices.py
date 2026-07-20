import pandas as pd

from stock_research.strategies import historical_candidates
from stock_research.strategies.historical_candidates import _load_prices


def _write_price(path, rows):
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def test_load_prices_reanchors_current_qfq_with_akshare_raw_close(tmp_path):
    qfq_dir = tmp_path / "qfq"
    raw_dir = tmp_path / "raw"
    _write_price(
        qfq_dir / "sz_000001.csv",
        [
            {
                "date": "2024-09-24", "open": 80.0, "high": 90.0,
                "low": 70.0, "close": 83.76, "volume": 1000,
            },
            {
                "date": "2025-04-30", "open": 110.0, "high": 120.0,
                "low": 100.0, "close": 116.31, "volume": 1000,
            },
        ],
    )
    _write_price(
        raw_dir / "sz_000001.csv",
        [
            {
                "date": "2024-09-24", "open": 240.0, "high": 260.0,
                "low": 230.0, "close": 254.29, "volume": 1000,
            },
            {
                "date": "2025-04-30", "open": 340.0, "high": 360.0,
                "low": 330.0, "close": 353.09, "volume": 1000,
            },
        ],
    )

    prices = _load_prices(
        qfq_dir,
        {"000001"},
        "2024-01-01",
        "2025-12-31",
        raw_kline_directory=raw_dir,
    )

    frame = prices["000001"]
    assert frame.loc[pd.Timestamp("2024-09-24"), "_asof_close"] == 254.29
    assert frame.loc[pd.Timestamp("2025-04-30"), "_asof_close"] == 353.09
    assert frame["_asof_price_available"].tolist() == [True, True]


def test_load_prices_marks_missing_raw_as_not_strict_asof(tmp_path):
    qfq_dir = tmp_path / "qfq"
    _write_price(
        qfq_dir / "sz_000001.csv",
        [{
            "date": "2024-09-24", "open": 80.0, "high": 90.0,
            "low": 70.0, "close": 83.76, "volume": 1000,
        }],
    )

    prices = _load_prices(
        qfq_dir,
        {"000001"},
        "2024-01-01",
        "2025-12-31",
        raw_kline_directory=tmp_path / "missing",
    )

    frame = prices["000001"]
    assert "_asof_close" not in frame
    assert frame["_asof_price_available"].tolist() == [False]


def test_load_prices_prefers_bulk_miniqmt_database_and_raw_frames(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2024-09-23", "2024-09-24"])
    qfq = pd.DataFrame({
        "open": [8.0, 8.1], "high": [8.2, 8.3], "low": [7.9, 8.0],
        "close": [8.1, 8.2], "volume": [1000, 1200],
        "amount": [8100, 9840], "tradestatus": [1, 1],
        "_qfq_anchor_date": pd.Timestamp("2026-01-01"),
    }, index=dates)
    raw = pd.DataFrame({
        "open": [10.0, 10.1], "high": [10.2, 10.3], "low": [9.9, 10.0],
        "close": [10.1, 10.2], "volume": [1000, 1200],
        "amount": [10100, 12240], "tradestatus": [1, 1],
    }, index=dates)
    monkeypatch.setattr(
        historical_candidates,
        "_load_miniqmt_qfq_from_database",
        lambda *_args, **_kwargs: {"000001": qfq.copy()},
    )
    monkeypatch.setattr(
        historical_candidates,
        "_load_raw_price_frames_bulk",
        lambda *_args, **_kwargs: {"000001": raw.copy()},
    )

    prices = _load_prices(
        tmp_path / "no-qfq-csv-needed",
        {"000001"},
        "2024-09-23",
        "2024-09-24",
        raw_kline_directory=tmp_path / "no-raw-csv-needed",
        price_source="miniqmt",
    )

    frame = prices["000001"]
    assert frame.loc[pd.Timestamp("2024-09-24"), "_asof_close"] == 10.2
    assert frame["_asof_price_available"].tolist() == [True, True]
