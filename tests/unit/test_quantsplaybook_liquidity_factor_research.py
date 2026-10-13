import pandas as pd

from stock_research.research.quantsplaybook_liquidity_factor_research import (
    select_liquidity_factor_top_n,
)


def test_factor_research_filters_liquidity_before_ranking():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B"), (date, "C")],
        names=["date", "code"],
    )
    panel = pd.DataFrame({
        "factor_a": [100.0, 2.0, 1.0],
        "forward_return": [-0.5, 0.2, 0.1],
    }, index=index)
    amount = pd.Series(
        [100_000_000.0, 600_000_000.0, 700_000_000.0],
        index=index,
    )

    selected, daily = select_liquidity_factor_top_n(
        panel,
        amount,
        factors=["factor_a"],
        snapshot_dates=[date],
        minimum_average_amount_yuan=500_000_000.0,
        top_n=1,
    )

    assert selected["code"].tolist() == ["B"]
    assert selected["forward_return"].tolist() == [0.2]
    assert daily.iloc[0]["eligible_count"] == 2
    assert daily.iloc[0]["selected_count"] == 1
