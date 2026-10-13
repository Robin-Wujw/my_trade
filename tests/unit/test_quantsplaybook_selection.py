import numpy as np
import pandas as pd
import pytest

from stock_research.strategies.quantsplaybook_selection import (
    LOW_MARKET_RESIDUAL_VOLATILITY_FACTOR,
    LOW_MARKET_RESIDUAL_VOLATILITY_LOOKBACK,
    MARKET_RESIDUAL_MOMENTUM_PARAMETERS,
    STRATEGY_SPECS,
    _clean_price_frame,
    _finite_turnover_reference_price,
    _market_residual_momentum_score,
    _market_residual_returns,
    calculate_low_market_residual_volatility_factor,
    calculate_market_residual_momentum_factors,
    calculate_daily_factor_panel,
    executable_factor_columns,
    low_market_residual_volatility_factor_columns,
    low_market_residual_volatility_factor_metadata,
    market_residual_momentum_factor_columns,
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


def _residual_return_fixture(days=360):
    dates = pd.bdate_range("2023-01-02", periods=days)
    market = pd.Series(
        0.002 * np.sin(np.arange(days) / 5.0)
        + 0.001 * np.cos(np.arange(days) / 11.0),
        index=dates,
    )
    stock_a = (
        0.0003 + 1.4 * market
        + 0.0004 * np.sin(np.arange(days) / 3.0)
    )
    stock_b = (
        -0.0002 + 0.8 * market
        + 0.0003 * np.cos(np.arange(days) / 7.0)
    )
    stock_c = 3.0 * market - stock_a - stock_b
    returns = pd.DataFrame(
        {"A": stock_a, "B": stock_b, "C": stock_c},
        index=dates,
    )
    mask = pd.DataFrame(True, index=dates, columns=returns.columns)
    return returns, mask


def test_market_residual_factor_names_register_fixed_parameter_grid():
    assert market_residual_momentum_factor_columns() == [
        "market_residual_momentum_126_skip5",
        "market_residual_momentum_126_skip20",
        "market_residual_momentum_252_skip5",
        "market_residual_momentum_252_skip20",
    ]


def test_low_market_residual_volatility_registers_fixed_price_only_formula():
    assert low_market_residual_volatility_factor_columns() == [
        "low_market_residual_volatility_20",
    ]
    metadata = low_market_residual_volatility_factor_metadata()
    assert metadata["estimation_window"] == 60
    assert metadata["volatility_lookback"] == 20
    assert metadata["industry_data_used"] is False
    assert metadata["financial_data_used"] is False


def test_daily_factor_panel_market_residual_factors_are_explicitly_enabled():
    frame = _price_frame(days=360, codes=4)
    membership = frame[["date", "code"]].copy()

    default_panel = calculate_daily_factor_panel(frame)
    enabled_without_membership = calculate_daily_factor_panel(
        frame,
        include_market_residual_momentum=True,
    )
    with_membership = calculate_daily_factor_panel(
        frame,
        tradable_membership=membership,
        include_market_residual_momentum=True,
    )

    for factor in market_residual_momentum_factor_columns():
        assert factor not in default_panel
        assert enabled_without_membership[factor].isna().all()
        assert with_membership[factor].notna().any()
    assert list(MARKET_RESIDUAL_MOMENTUM_PARAMETERS.values()) == [
        {"lookback": 126, "skip": 5},
        {"lookback": 126, "skip": 20},
        {"lookback": 252, "skip": 5},
        {"lookback": 252, "skip": 20},
    ]


def test_daily_factor_panel_low_residual_volatility_is_explicitly_enabled():
    frame = _price_frame(days=160, codes=4)
    membership = frame[["date", "code"]].copy()

    default_panel = calculate_daily_factor_panel(frame)
    enabled_without_membership = calculate_daily_factor_panel(
        frame,
        include_low_market_residual_volatility=True,
    )
    with_membership = calculate_daily_factor_panel(
        frame,
        tradable_membership=membership,
        include_low_market_residual_volatility=True,
    )

    assert LOW_MARKET_RESIDUAL_VOLATILITY_FACTOR not in default_panel
    assert enabled_without_membership[
        LOW_MARKET_RESIDUAL_VOLATILITY_FACTOR
    ].isna().all()
    assert with_membership[
        LOW_MARKET_RESIDUAL_VOLATILITY_FACTOR
    ].notna().any()


def test_market_return_excludes_missing_and_non_tradable_returns_without_zero_fill():
    dates = pd.bdate_range("2024-01-02", periods=3)
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "B": [0.03, np.nan, 9.0]},
        index=dates,
    )
    mask = pd.DataFrame(True, index=dates, columns=returns.columns)
    mask.loc[dates[2], "B"] = False

    residuals, market = _market_residual_returns(
        returns,
        mask,
        estimation_window=2,
    )

    assert market.loc[dates[0]] == pytest.approx(0.02)
    assert market.loc[dates[1]] == pytest.approx(0.02)
    assert market.loc[dates[2]] == pytest.approx(0.03)
    assert pd.isna(residuals.loc[dates[1], "B"])
    assert pd.isna(residuals.loc[dates[2], "B"])


def test_market_residual_ols_uses_only_sixty_valid_observations_through_d_minus_1():
    days = 65
    dates = pd.bdate_range("2024-01-02", periods=days)
    market = pd.Series(
        0.002 * np.sin(np.arange(days) / 4.0),
        index=dates,
    )
    stock = 0.001 + 1.5 * market
    stock.iloc[60] += 0.02
    peer = (2.0 * market - stock) / 1.0
    returns = pd.DataFrame({"stock": stock, "peer": peer}, index=dates)
    mask = pd.DataFrame(True, index=dates, columns=returns.columns)

    residuals, observed_market = _market_residual_returns(returns, mask)

    pd.testing.assert_series_equal(observed_market, market, check_names=False)
    assert residuals["stock"].iloc[:60].isna().all()
    assert residuals["stock"].iloc[60] == pytest.approx(0.02, abs=1e-12)


def test_market_residual_score_strictly_excludes_skip_window_and_fails_closed():
    dates = pd.bdate_range("2024-01-02", periods=150)
    residuals = pd.DataFrame(
        {"A": np.sin(np.arange(150) / 7.0) + np.arange(150) / 1000.0},
        index=dates,
    )
    baseline = _market_residual_momentum_score(
        residuals,
        lookback=126,
        skip=5,
    )
    excluded_changed = residuals.copy()
    excluded_changed.iloc[-5:, 0] += 1000.0
    after_excluded_change = _market_residual_momentum_score(
        excluded_changed,
        lookback=126,
        skip=5,
    )
    included_changed = residuals.copy()
    included_changed.iloc[-6, 0] += 1000.0
    after_included_change = _market_residual_momentum_score(
        included_changed,
        lookback=126,
        skip=5,
    )
    missing = residuals.copy()
    missing.iloc[-20, 0] = np.nan
    after_missing = _market_residual_momentum_score(
        missing,
        lookback=126,
        skip=5,
    )

    assert after_excluded_change.iloc[-1, 0] == pytest.approx(
        baseline.iloc[-1, 0],
    )
    assert after_included_change.iloc[-1, 0] != pytest.approx(
        baseline.iloc[-1, 0],
    )
    assert pd.isna(after_missing.iloc[-1, 0])


def test_market_residual_factors_do_not_change_when_future_data_is_appended():
    returns, mask = _residual_return_fixture()
    cutoff = returns.index[339]
    before = calculate_market_residual_momentum_factors(
        returns.loc[:cutoff],
        mask.loc[:cutoff],
    )
    changed = returns.copy()
    changed.loc[changed.index > cutoff, "A"] += 5.0
    after = calculate_market_residual_momentum_factors(changed, mask)

    for factor in market_residual_momentum_factor_columns():
        assert before[factor].notna().any().any()
        pd.testing.assert_frame_equal(
            before[factor],
            after[factor].loc[:cutoff],
        )


def test_low_market_residual_volatility_is_pit_and_fails_closed():
    returns, mask = _residual_return_fixture(days=130)
    cutoff = returns.index[119]
    before = calculate_low_market_residual_volatility_factor(
        returns.loc[:cutoff],
        mask.loc[:cutoff],
    )
    changed = returns.copy()
    changed.loc[changed.index > cutoff, "A"] += 5.0
    after = calculate_low_market_residual_volatility_factor(changed, mask)

    assert before.notna().any().any()
    pd.testing.assert_frame_equal(before, after.loc[:cutoff])
    assert before.iloc[
        : 60 + LOW_MARKET_RESIDUAL_VOLATILITY_LOOKBACK - 1
    ].isna().all().all()


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


def test_qfq_signal_path_rejects_missing_adjustment_factor():
    frame = pd.DataFrame([
        {
            "date": "2024-01-02", "code": "sh.600001",
            "open": 10, "high": 10, "low": 10, "close": 10,
            "raw_to_qfq_factor": 0.5,
        },
        {
            "date": "2024-01-03", "code": "sh.600001",
            "open": 10, "high": 10, "low": 10, "close": 10,
            "raw_to_qfq_factor": pd.NA,
        },
    ])

    with pytest.raises(ValueError, match="complete and positive"):
        _clean_price_frame(frame)


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
