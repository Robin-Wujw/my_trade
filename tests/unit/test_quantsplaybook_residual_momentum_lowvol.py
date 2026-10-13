import pandas as pd
import pytest

from stock_research.research.quantsplaybook_chunked import (
    ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR,
    PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR,
    PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR,
)
from stock_research.research.quantsplaybook_residual_momentum_lowvol import (
    merge_source_factor_panels,
)


def _index(*codes):
    return pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), code) for code in codes],
        names=["date", "code"],
    )


def test_cached_panel_merge_uses_low_vol_universe_without_filling_residuals():
    residual = pd.DataFrame(
        {PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR: [2.0, 1.0]},
        index=_index("A", "B"),
    )
    low_volatility = pd.DataFrame(
        {
            "network_cc": [1.0, 2.0],
            "ma_convergence": [1.0, 2.0],
            "disposition_reversal": [1.0, 2.0],
            "high_quality_momentum": [9.0, 9.0],
            "coin_team": [1.0, 2.0],
            PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR: [1.0, 2.0],
        },
        index=_index("B", "C"),
    )

    panel, metadata = merge_source_factor_panels(residual, low_volatility)

    factor = ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR
    assert list(panel.index.get_level_values("code")) == ["B", "C"]
    assert panel.loc[(pd.Timestamp("2024-01-02"), "B"), factor] == pytest.approx(
        7.0 / 12.0,
    )
    assert pd.isna(panel.loc[(pd.Timestamp("2024-01-02"), "C"), factor])
    assert metadata["aligned_panel_rows"] == 2
    assert metadata["residual_momentum_available_rows"] == 1


def test_cached_panel_merge_rejects_missing_fixed_factor():
    residual = pd.DataFrame(
        {PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR: [1.0]}, index=_index("A"),
    )
    low_volatility = pd.DataFrame(
        {"network_cc": [1.0]}, index=_index("A"),
    )

    with pytest.raises(ValueError, match="source factors are missing"):
        merge_source_factor_panels(residual, low_volatility)
