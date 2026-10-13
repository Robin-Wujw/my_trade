import pandas as pd

from stock_research.research.quantsplaybook_liquidity_universe import (
    _rolling_average_amount,
    apply_liquidity_universe,
    build_parser,
    frozen_source_score,
    liquidity_factor_name,
)


def test_adv20_requires_twenty_positive_amount_observations():
    dates = pd.bdate_range("2024-01-02", periods=21)
    prices = pd.DataFrame({
        "date": dates,
        "code": ["A"] * len(dates),
        "amount": [pd.NA] + [30_000.0] * 20,
    })

    average = _rolling_average_amount(prices)

    assert pd.isna(average.loc[(dates[-2], "A")])
    assert average.loc[(dates[-1], "A")] == 30_000_000.0


def test_liquidity_universe_filters_before_frozen_factor_rank():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B")], names=["date", "code"],
    )
    panel = pd.DataFrame({"source": [1.0, 2.0]}, index=index)
    score = pd.Series([0.9, 0.8], index=index)
    amount = pd.Series([100_000_000.0, 600_000_000.0], index=index)

    result, diagnostics = apply_liquidity_universe(
        panel,
        score,
        amount,
        [500_000_000.0],
    )
    factor = liquidity_factor_name(500_000_000.0)

    assert pd.isna(result.loc[(date, "A"), factor])
    assert result.loc[(date, "B"), factor] == 0.8
    assert diagnostics[0]["minimum_eligible_symbols"] == 1
    assert factor == "playbook_low_corr_liquidity_500m"


def test_liquidity_universe_supports_a_named_frozen_source_factor():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B")], names=["date", "code"],
    )
    panel = pd.DataFrame(index=index)
    score = pd.Series([0.9, 0.8], index=index)
    amount = pd.Series([600_000_000.0, 700_000_000.0], index=index)

    result, diagnostics = apply_liquidity_universe(
        panel,
        score,
        amount,
        [500_000_000.0],
        source_factor="high_quality_momentum",
    )
    factor = liquidity_factor_name(
        500_000_000.0,
        source_factor="high_quality_momentum",
    )

    assert result[factor].tolist() == [0.9, 0.8]
    assert diagnostics[0]["factor"] == factor
    assert factor == "high_quality_momentum_liquidity_500m"


def test_named_liquid_balanced_combination_uses_frozen_equal_weights():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B")], names=["date", "code"],
    )
    panel = pd.DataFrame({
        "disposition_reversal": [2.0, 1.0],
        "network_scc": [2.0, 1.0],
        "chip_loss_overhang": [2.0, 1.0],
        "high_quality_momentum": [2.0, 1.0],
    }, index=index)

    score = frozen_source_score(
        panel,
        "playbook_liquid_balanced",
        low_corr_weights={},
    )

    assert score.loc[(date, "A")] == 1.0
    assert score.loc[(date, "B")] == 0.5


def test_manifest_declared_combination_uses_its_frozen_weights():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B")], names=["date", "code"],
    )
    panel = pd.DataFrame({
        "factor_a": [2.0, 1.0],
        "factor_b": [1.0, 2.0],
    }, index=index)

    score = frozen_source_score(
        panel,
        "fixed_combination",
        low_corr_weights={},
        combination_specs={
            "fixed_combination": {
                "weights": {"factor_a": 0.75, "factor_b": 0.25},
            },
        },
    )

    assert score.loc[(date, "A")] == 0.875
    assert score.loc[(date, "B")] == 0.625


def test_liquidity_profile_enrichment_skip_is_opt_in():
    assert build_parser().parse_args([]).skip_candidate_profile_enrichment is False
    args = build_parser().parse_args(["--skip-candidate-profile-enrichment"])
    assert args.skip_candidate_profile_enrichment is True
