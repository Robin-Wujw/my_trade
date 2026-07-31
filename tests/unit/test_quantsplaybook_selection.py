import numpy as np
import pandas as pd
import pytest

from stock_research.strategies.quantsplaybook_selection import (
    STRATEGY_SPECS,
    _clean_price_frame,
    _finite_turnover_reference_price,
    calculate_daily_factor_panel,
    executable_factor_columns,
    strategy_inventory_frame,
)


def _price_frame(days=170, codes=12):
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for code_index in range(codes):
        exchange = "sh" if code_index % 2 == 0 else "sz"
        code = f"{exchange}.{600001 + code_index:06d}"
        slope = 0.0002 + code_index * 0.00005
        base = 8.0 + code_index
        for index, date in enumerate(dates):
            close = (
                base
                * (1.0 + slope) ** index
                * (1.0 + 0.002 * np.sin(index / 3 + code_index))
            )
            rows.append({
                "date": date,
                "code": code,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 100_000 + index * 100 + code_index,
                "amount": close * (100_000 + index * 100 + code_index) / 10,
                "tradestatus": "1",
                "raw_to_qfq_factor": 0.5,
                "turnover_rate": 1.0 + index / 1000,
                "turnover_rate_f": 1.2 + index / 1000,
                "total_mv": 100_000 + code_index * 10_000 + index,
                "circ_mv": 80_000 + code_index * 8_000 + index,
                "pb": 1.0 + code_index / 10,
            })
    return pd.DataFrame(rows)


def test_quantsplaybook_inventory_accounts_for_all_selection_directories():
    inventory = strategy_inventory_frame()

    assert len(STRATEGY_SPECS) == 27
    assert inventory["key"].is_unique
    assert inventory["source_commit"].nunique() == 1
    assert {
        "missing_intraday_data",
        "missing_historical_holdings",
        "not_a_stock_selector",
        "missing_trained_model",
    }.issubset(set(inventory["status"]))


def test_daily_factor_panel_contains_each_currently_executable_factor():
    panel = calculate_daily_factor_panel(_price_frame())

    expected = set(executable_factor_columns())
    assert expected.issubset(panel.columns)
    assert panel["high_quality_momentum"].notna().any()
    assert panel["shadow_reversal"].notna().any()
    assert panel["ma_convergence"].notna().any()
    assert panel["network_scc"].notna().any()
    assert panel["network_tcc"].notna().any()
    assert panel["network_cc"].notna().any()
    assert panel["playbook_ensemble"].dropna().between(0, 1).all()


def test_daily_factors_do_not_change_when_future_bars_are_appended():
    base = _price_frame()
    cutoff = pd.Timestamp("2024-05-31")
    truncated = base[base["date"] <= cutoff]
    future = base.copy()
    future.loc[future["date"] > cutoff, ["open", "high", "low", "close"]] *= 20

    before = calculate_daily_factor_panel(truncated)
    after = calculate_daily_factor_panel(future)
    shared = before.index.intersection(after.index)

    pd.testing.assert_frame_equal(
        before.loc[shared],
        after.loc[shared, before.columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_qfq_signal_path_multiplies_raw_price_by_adj_factor():
    frame = pd.DataFrame([
        {
            "date": "2024-01-02",
            "code": "sh.600001",
            "open": 10,
            "high": 10,
            "low": 10,
            "close": 10,
            "raw_to_qfq_factor": 0.5,
        },
        {
            "date": "2024-01-03",
            "code": "sh.600001",
            "open": 6,
            "high": 6,
            "low": 6,
            "close": 6,
            "raw_to_qfq_factor": 0.25,
        },
    ])

    cleaned = _clean_price_frame(frame)

    assert cleaned["adjustment_factor"].tolist() == [2.0, 4.0]
    assert cleaned["qfq_close"].tolist() == [20.0, 24.0]


def test_finite_turnover_reference_matches_source_window_formula():
    dates = pd.bdate_range("2024-01-02", periods=5)
    price = pd.DataFrame({"A": [10, 11, 12, 13, 14]}, index=dates)
    turnover_pct = pd.DataFrame({"A": [10, 20, 30, 40, 50]}, index=dates)

    result = _finite_turnover_reference_price(
        price,
        turnover_pct,
        window=3,
    )

    turnover = np.array([0.3, 0.4, 0.5])
    weights = turnover * np.array([
        (1 - 0.4) * (1 - 0.5),
        (1 - 0.5),
        1.0,
    ])
    expected = np.average([12, 13, 14], weights=weights)
    assert result.loc[dates[-1], "A"] == pytest.approx(expected)


def test_adjustment_jump_changes_return_based_factor_history():
    frame = _price_frame()
    first = calculate_daily_factor_panel(frame)["salience_str"]
    scaled = frame.copy()
    target = scaled["code"].eq(scaled["code"].iloc[0])
    scaled.loc[
        target & scaled["date"].lt(pd.Timestamp("2024-04-01")),
        "raw_to_qfq_factor",
    ] = 0.25
    scaled.loc[
        target & scaled["date"].ge(pd.Timestamp("2024-04-01")),
        "raw_to_qfq_factor",
    ] = 0.5
    second = calculate_daily_factor_panel(scaled)["salience_str"]

    assert not first.dropna().equals(second.dropna())
