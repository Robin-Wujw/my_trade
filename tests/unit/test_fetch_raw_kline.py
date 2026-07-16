import pandas as pd

from scripts import fetch_akshare_raw_kline as raw_kline


def test_normalize_baostock_frame_outputs_raw_kline_schema():
    frame = pd.DataFrame([
        {
            "date": "2024-09-25",
            "code": "sz.002594",
            "open": "260.1",
            "high": "263.0",
            "low": "258.0",
            "close": "261.5",
            "volume": "123456",
            "amount": "321000000.5",
            "turn": "1.23",
            "tradestatus": "1",
        },
        {
            "date": "2024-09-24",
            "code": "sz.002594",
            "open": "254.0",
            "high": "256.0",
            "low": "250.0",
            "close": "254.29",
            "volume": "100000",
            "amount": "254290000",
            "turn": "1.00",
            "tradestatus": "1",
        },
    ])

    result = raw_kline._normalize_baostock_frame(frame, "002594", "sz")

    assert list(result["date"]) == ["2024-09-24", "2024-09-25"]
    assert list(result["code"].unique()) == ["sz.002594"]
    assert result.loc[0, "close"] == 254.29
    assert result.loc[0, "amount"] == 254290000
    assert "turnover" in result.columns


def test_normalize_tushare_payload_converts_amount_to_yuan():
    payload = {
        "code": 0,
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            "items": [["002594.SZ", "20240924", 253.0, 255.5, 246.76, 254.29, 156494.54, 3949080.823]],
        },
    }

    result = raw_kline._normalize_tushare_payload(payload, "002594", "sz")

    assert result.loc[0, "date"] == "2024-09-24"
    assert result.loc[0, "code"] == "sz.002594"
    assert result.loc[0, "close"] == 254.29
    assert result.loc[0, "amount"] == 3949080823.0


def test_auto_provider_falls_back_to_tushare(monkeypatch):
    expected = pd.DataFrame([{
        "date": "2024-09-24",
        "code": "sz.002594",
        "open": 254.0,
        "close": 254.29,
        "high": 256.0,
        "low": 250.0,
        "volume": 100000,
        "amount": 254290000,
    }])

    def fail_eastmoney(*_args, **_kwargs):
        raise RuntimeError("socket hang up")

    def ok_tushare(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(raw_kline, "_fetch_raw_kline", fail_eastmoney)
    monkeypatch.setattr(raw_kline, "_fetch_tushare_raw_kline", ok_tushare)

    frame, provider = raw_kline._fetch_raw_kline_with_provider(
        "002594",
        "sz",
        "2024-01-01",
        "2026-07-14",
        provider="auto",
        node_executable="node",
        allow_insecure=True,
    )

    assert provider == "tushare"
    pd.testing.assert_frame_equal(frame, expected)
