import pandas as pd

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
