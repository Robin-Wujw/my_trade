from types import SimpleNamespace

import pandas as pd
import pytest

from apps import tushare_bulk_sync, tushare_financial_history_sync


@pytest.mark.parametrize("module", [tushare_bulk_sync, tushare_financial_history_sync])
def test_financial_sync_includes_delisted_and_paused_from_broad_master(module):
    tables = {
        "stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"], "list_status": ["L"]}),
        "basic": pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", None],
            "list_status": ["L", "D", "P", "G", "D"],
        }),
    }
    repository = SimpleNamespace(load_dataset=lambda name: tables[name])
    assert module._load_stock_codes(repository) == ["000001.SZ", "000002.SZ", "000003.SZ"]


@pytest.mark.parametrize("module", [tushare_bulk_sync, tushare_financial_history_sync])
def test_financial_sync_requires_cached_master(module):
    with pytest.raises(RuntimeError, match="not cached"):
        module._load_stock_codes(SimpleNamespace(load_dataset=lambda name: pd.DataFrame()))
