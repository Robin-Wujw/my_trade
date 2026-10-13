import math

import pandas as pd
import pytest

from stock_research.research.quantsplaybook_size_lowvol import (
    SIZE_FACTOR,
    SOURCE_FACTOR,
    add_size_combination,
    calculate_china_size_factor,
    fixed_size_combination_spec,
)


def _index(codes, date="2024-01-02"):
    return pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), code) for code in codes],
        names=["date", "code"],
    )


def _base_panel(codes):
    index = _index(codes)
    return pd.DataFrame({
        "network_cc": range(1, len(codes) + 1),
        "ma_convergence": range(1, len(codes) + 1),
        "disposition_reversal": range(1, len(codes) + 1),
        "high_quality_momentum": range(1, len(codes) + 1),
        "coin_team": range(1, len(codes) + 1),
        "low_market_residual_volatility_20": range(1, len(codes) + 1),
    }, index=index)


def test_china_size_factor_excludes_smallest_thirty_percent():
    codes = [f"C{value:02d}" for value in range(1, 11)]
    market_cap = pd.Series(range(1, 11), index=_index(codes), dtype=float)

    result = calculate_china_size_factor(market_cap)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(-math.log(4.0))
    assert result.iloc[3] > result.iloc[-1]


def test_size_combination_is_fixed_equal_weight_and_fails_closed():
    codes = [f"C{value:02d}" for value in range(1, 11)]
    panel = _base_panel(codes)
    market_cap = pd.Series(range(1, 11), index=panel.index, dtype=float)

    combined, coverage = add_size_combination(panel, market_cap)
    spec = fixed_size_combination_spec()

    assert set(spec["weights"]) == set(spec["selected_factors"])
    assert all(weight == pytest.approx(1 / 7) for weight in spec["weights"].values())
    assert combined.iloc[:3][SIZE_FACTOR].isna().all()
    assert combined.iloc[:3][SOURCE_FACTOR].isna().all()
    assert combined.iloc[3:][SOURCE_FACTOR].notna().all()
    assert coverage["combination_available_rows"] == 7


def test_size_factor_does_not_change_prior_date_when_future_caps_are_added():
    first = pd.Series([10.0, 20.0, 30.0, 40.0], index=_index("ABCD"))
    future = pd.Series(
        [400.0, 300.0, 200.0, 100.0],
        index=_index("ABCD", date="2024-01-03"),
    )

    original = calculate_china_size_factor(first)
    extended = calculate_china_size_factor(pd.concat([first, future]))

    pd.testing.assert_series_equal(original, extended.loc[original.index])
