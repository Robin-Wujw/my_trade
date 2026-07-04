import pandas as pd
import pytest

from stock_research.indicators.sector_metrics import candle_label, pct_change
from stock_research.market.sectors import classify_group, normalize_board_name
from stock_research.strategies.sector_watch import score_direct


def test_sector_helpers_keep_current_classification_and_candle_rules():
    assert classify_group("半导体设备") == "半导体"
    assert normalize_board_name("通信行业板块") == "通信"
    assert pct_change(pd.Series([100.0, 105.0]), 1) == pytest.approx(0.05)
    assert candle_label({"pct_chg": 0.06}) == "长阳"
    assert candle_label({"pct_chg": -0.03}) == "中阴"
    assert score_direct(5, 0, 10) == 50.0
