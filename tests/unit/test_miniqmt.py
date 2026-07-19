import sys
import types

import pytest

from stock_research.api.miniqmt import (
    LiveTradingDisabled,
    MiniQmtClient,
    MiniQmtConfig,
    MiniQmtSdkNotFound,
    check_sdk,
    mask_account_fields,
)


def test_check_sdk_reports_missing_xtquant_without_raising(tmp_path):
    config = MiniQmtConfig(qmt_root=tmp_path / "missing-qmt")

    result = check_sdk(config)

    assert result["xtquant_importable"] is False
    assert result["site_packages_exists"] is False
    assert result["error"]


def test_client_uses_stock_account_default_constructor(monkeypatch, tmp_path):
    qmt_root = tmp_path / "qmt"
    (qmt_root / "bin.x64" / "Lib" / "site-packages").mkdir(parents=True)
    (qmt_root / "userdata_mini").mkdir()
    calls = []

    class FakeTrader:
        def __init__(self, userdata_path, session_id):
            self.userdata_path = userdata_path
            self.session_id = session_id

        def start(self):
            return None

        def connect(self):
            return 0

        def stop(self):
            return None

        def subscribe(self, account):
            return 0

        def query_stock_asset(self, account):
            return types.SimpleNamespace(
                account_id=account.account_id,
                cash=100.0,
                total_asset=100.0,
            )

        def query_stock_positions(self, account):
            return [
                types.SimpleNamespace(
                    account_id=account.account_id,
                    stock_code="600000.SH",
                    volume=100,
                )
            ]

    class FakeStockAccount:
        def __init__(self, *args):
            calls.append(args)
            self.account_id = args[0]
            self.account_type = 2

    monkeypatch.setitem(
        sys.modules,
        "xtquant.xttrader",
        types.SimpleNamespace(XtQuantTrader=FakeTrader),
    )
    monkeypatch.setitem(
        sys.modules,
        "xtquant.xttype",
        types.SimpleNamespace(StockAccount=FakeStockAccount),
    )
    monkeypatch.chdir(tmp_path)

    config = MiniQmtConfig(qmt_root=qmt_root, accounts=("000108832878",), session_id=123456)
    with MiniQmtClient(config) as client:
        result = client.query_accounts()

    assert calls == [("000108832878",)]
    assert result["ok"] is True
    assert result["accounts"][0]["account_type"] == 2
    assert result["accounts"][0]["asset"]["account_id"] == "000****2878"
    assert result["accounts"][0]["positions_sample"][0]["account_id"] == "000****2878"
    assert result["accounts"][0]["positions_sample"][0]["stock_code"] == "600000.SH"


def test_client_rejects_live_order_methods(tmp_path):
    client = MiniQmtClient(MiniQmtConfig(qmt_root=tmp_path))

    with pytest.raises(LiveTradingDisabled):
        client.place_order()
    with pytest.raises(LiveTradingDisabled):
        client.cancel_order()


def test_missing_sdk_raises_for_connection(tmp_path):
    client = MiniQmtClient(MiniQmtConfig(qmt_root=tmp_path / "missing"))

    with pytest.raises(MiniQmtSdkNotFound):
        client.connect()


def test_mask_account_fields_recurses_nested_payload():
    payload = {
        "account_id": "000108832878",
        "positions": [{"m_strAccountID": "000500055055", "stock_code": "688041.SH"}],
    }

    assert mask_account_fields(payload) == {
        "account_id": "000****2878",
        "positions": [{"m_strAccountID": "000****5055", "stock_code": "688041.SH"}],
    }
