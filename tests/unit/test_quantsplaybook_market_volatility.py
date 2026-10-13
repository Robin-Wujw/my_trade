import json

import numpy as np
import pandas as pd
import pytest

from stock_research.research.quantsplaybook_market_volatility import (
    build_market_volatility_candidates,
    calculate_market_volatility_scaling,
    load_equal_weight_market_returns,
)


def test_market_volatility_scaling_uses_strictly_prior_baseline():
    dates = pd.bdate_range("2024-01-02", periods=22)
    returns = pd.Series([0.01] * 20 + [0.10, 0.30], index=dates)

    original = calculate_market_volatility_scaling(returns)
    changed = returns.copy()
    changed.iloc[-1] = -0.50
    replayed = calculate_market_volatility_scaling(changed)

    assert original.iloc[:20]["right_market_volatility_multiplier"].eq(1).all()
    assert 0 < original.iloc[20]["right_market_volatility_multiplier"] < 1
    assert original["right_market_volatility_multiplier"].le(1).all()
    pd.testing.assert_series_equal(original.iloc[:-1, 3], replayed.iloc[:-1, 3])


def test_equal_weight_market_returns_use_saved_panel_membership(tmp_path):
    panel_root = tmp_path / "panels"
    price_root = tmp_path / "prices"
    panel_root.mkdir()
    price_root.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=2)
    panel = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1]],
        "code": ["sh.600000", "sz.000001", "sh.600000"],
        "factor": [1.0, 1.0, 1.0],
    })
    panel.to_pickle(panel_root / "factor_panel_2024.pkl")
    all_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), *dates])
    pd.DataFrame({"date": all_dates, "close": [10.0, 11.0, 12.1]}).to_pickle(
        price_root / "sh_600000.pkl"
    )
    pd.DataFrame({"date": all_dates, "close": [20.0, 18.0, 19.0]}).to_pickle(
        price_root / "sz_000001.pkl"
    )

    result, metadata = load_equal_weight_market_returns(
        panel_root=panel_root,
        price_directory=price_root,
        dates=dates,
    )

    assert result.iloc[0] == pytest.approx(np.mean([0.10, -0.10]))
    assert result.iloc[1] == pytest.approx(0.10)
    assert metadata["eligible_rows"] == 3
    assert metadata["available_return_rows"] == 3


def test_candidate_overlay_marks_research_and_only_sizes_right_rows(tmp_path):
    source = tmp_path / "source"
    panels = tmp_path / "panels"
    prices = tmp_path / "prices"
    output = tmp_path / "output"
    source.mkdir()
    panels.mkdir()
    prices.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=22)
    snapshots = []
    for date in dates:
        date_text = date.strftime("%Y-%m-%d")
        path = source / f"candidates_{date_text}.csv"
        pd.DataFrame([
            {"code": "sh.600000", "allow_right": True, "allow_left": False},
            {"code": "sz.000001", "allow_right": False, "allow_left": True},
        ]).to_csv(path, index=False)
        snapshots.append({"date": date_text, "file": path.name})
    (source / "manifest.json").write_text(
        json.dumps({"snapshots": snapshots, "research_only": False}),
        encoding="utf-8",
    )
    panel = pd.DataFrame({
        "date": dates,
        "code": "sh.600000",
        "factor": 1.0,
    })
    panel.to_pickle(panels / "factor_panel_2024.pkl")
    price_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), *dates])
    closes = 10.0 * np.cumprod([1.0, *([1.01] * 20), 1.10, 1.30])
    pd.DataFrame({"date": price_dates, "close": closes}).to_pickle(
        prices / "sh_600000.pkl"
    )

    build_market_volatility_candidates(
        source_candidate_directory=source,
        panel_root=panels,
        price_directory=prices,
        output_directory=output,
        start_date=dates[0].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"),
    )

    last = pd.read_csv(output / snapshots[-1]["file"])
    assert pd.notna(last.loc[0, "right_market_volatility_multiplier"])
    assert pd.isna(last.loc[1, "right_market_volatility_multiplier"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["future_return_used_for_candidate_selection"] is False
    assert manifest["right_market_volatility_sizing"]["window_sessions"] == 20
