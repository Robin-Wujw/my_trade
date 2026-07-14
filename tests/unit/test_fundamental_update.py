import pandas as pd

from stock_research.pipelines.fundamental_update import value_market_cap_eligible_count


def test_value_market_cap_count_handles_empty_snapshot():
    assert value_market_cap_eligible_count(pd.DataFrame()) == 0


def test_value_market_cap_count_filters_value_rows_above_threshold():
    snapshot = pd.DataFrame({
        "method": ["VALUE", "VALUE", "GROWTH"],
        "mktcap": [100, 99, 500],
    })

    assert value_market_cap_eligible_count(snapshot) == 1
