import json

import pandas as pd
import pytest

from stock_research.research.quantsplaybook_chunked import (
    COMBINATION_FACTORS,
    MARKET_RESIDUAL_REPLACEMENT_FACTORS,
    ONE_NETWORK_BASELINE_FACTORS,
    ONE_NETWORK_CATEGORY_BALANCED_CATEGORIES,
    ONE_NETWORK_CATEGORY_BALANCED_FACTOR,
    ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_CATEGORIES,
    ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_FACTOR,
    ONE_NETWORK_PRICE_ONLY_COMPLETE_LOW_VOLATILITY_FACTOR,
    ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_CATEGORIES,
    ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_FACTOR,
    ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_FACTOR,
    ONE_NETWORK_PRICE_ONLY_MEDIAN_LOW_VOLATILITY_FACTOR,
    ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR,
    ONE_NETWORK_PRICE_ONLY_SCC_LOW_VOLATILITY_FACTOR,
    PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR,
    PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR,
    _capped_normalized_weights,
    _apply_combination,
    _combination_factors_to_generate,
    _combined_data_coverage,
    _declared_combination_specs,
    _market_residual_replacement_specs,
    _merge_candidate_chunks,
    _one_network_category_balanced_spec,
    _one_network_price_only_category_balanced_spec,
    _one_network_price_only_complete_low_volatility_spec,
    _one_network_price_only_low_volatility_category_balanced_spec,
    _one_network_price_only_low_volatility_spec,
    _one_network_price_only_median_low_volatility_spec,
    _one_network_price_only_residual_momentum_low_volatility_spec,
    _one_network_price_only_scc_low_volatility_spec,
    _run_year_source_generation,
    _select_low_correlation_factors,
    build_parser,
    round_robin_factor_combination,
    _year_periods,
)


def test_year_periods_clip_first_and_last_year():
    assert _year_periods("2021-06-01", "2023-03-15") == [
        (2021, "2021-06-01", "2021-12-31"),
        (2022, "2022-01-01", "2022-12-31"),
        (2023, "2023-01-01", "2023-03-15"),
    ]


def test_declared_combinations_use_training_metrics_only():
    training = pd.DataFrame([
        {
            "factor": "high_quality_momentum",
            "mean_rank_ic": 0.10,
            "top20_excess_forward_return": 0.02,
        },
        {
            "factor": "shadow_reversal",
            "mean_rank_ic": 0.05,
            "top20_excess_forward_return": 0.03,
        },
        {
            "factor": "coin_team",
            "mean_rank_ic": -0.02,
            "top20_excess_forward_return": 0.04,
        },
    ]).set_index("factor")

    specs = _declared_combination_specs(
        training,
        positive_ic_top_n=1,
    )

    assert specs["playbook_positive_ic_top5"]["weights"] == {
        "high_quality_momentum": 1.0,
    }
    assert set(specs["playbook_train_ic_weighted"]["weights"]) == {
        "high_quality_momentum",
        "shadow_reversal",
    }
    assert "playbook_category_balanced" in specs
    assert "playbook_capped_ic_weighted" in specs
    assert "playbook_low_corr" in specs
    assert specs["playbook_low_corr"][
        "low_corr_max_network_factors"
    ] is None
    assert specs["playbook_factor_quota"]["combination_type"] == (
        "round_robin_factor_quota"
    )
    assert specs["playbook_strategy_aligned"]["weights"] == {
        "high_quality_momentum": 1.0,
    }


def test_strategy_aligned_combination_uses_three_fixed_equal_weights():
    training = pd.DataFrame([
        {
            "factor": factor,
            "mean_rank_ic": 0.01,
            "top20_excess_forward_return": 0.01,
        }
        for factor in (
            "high_quality_momentum",
            "ma_convergence",
            "buying_pressure",
        )
    ]).set_index("factor")

    specs = _declared_combination_specs(
        training,
        positive_ic_top_n=3,
    )

    assert specs["playbook_strategy_aligned"]["weights"] == {
        "high_quality_momentum": 1.0 / 3.0,
        "ma_convergence": 1.0 / 3.0,
        "buying_pressure": 1.0 / 3.0,
    }


def test_market_residual_replacements_change_only_high_quality_momentum():
    specs = _market_residual_replacement_specs()

    assert tuple(specs) == MARKET_RESIDUAL_REPLACEMENT_FACTORS
    assert len(specs) == 4
    expected_unchanged = set(ONE_NETWORK_BASELINE_FACTORS) - {
        "high_quality_momentum",
    }
    for name, spec in specs.items():
        selected = spec["selected_factors"]
        replacement = spec["replacement_factor"]
        assert name.endswith(replacement)
        assert len(selected) == 6
        assert set(selected) - {replacement} == expected_unchanged
        assert "high_quality_momentum" not in selected
        assert set(spec["weights"]) == set(selected)
        assert set(spec["weights"].values()) == {1.0 / 6.0}
        assert spec["continuous_weight_search_used"] is False
        assert spec["research_only"] is True
        assert spec["market_residual_momentum"]["industry_data_used"] is False


def test_one_network_category_balanced_spec_has_fixed_members_and_weights():
    spec = _one_network_category_balanced_spec()

    assert tuple(spec["selected_factors"]) == ONE_NETWORK_BASELINE_FACTORS
    assert spec["weights"] == {
        "network_cc": 0.25,
        "ma_convergence": 0.25,
        "disposition_reversal": 0.125,
        "high_quality_momentum": 0.125,
        "coin_team": 0.125,
        "low_idiosyncratic_volatility": 0.125,
    }
    assert spec["factor_categories"] == {
        category: list(factors)
        for category, factors in ONE_NETWORK_CATEGORY_BALANCED_CATEGORIES.items()
    }
    assert spec["category_weights"] == {
        "network": 0.25,
        "price_structure": 0.25,
        "behavior": 0.25,
        "risk_momentum": 0.25,
    }
    assert spec["within_category_weights"] == {
        "network": {"network_cc": 1.0},
        "price_structure": {"ma_convergence": 1.0},
        "behavior": {"disposition_reversal": 0.5, "coin_team": 0.5},
        "risk_momentum": {
            "high_quality_momentum": 0.5,
            "low_idiosyncratic_volatility": 0.5,
        },
    }
    assert spec["continuous_weight_search_used"] is False
    assert spec["participates_in_validation_selection"] is False
    assert spec["uses_financial_data"] is True
    assert spec["financial_point_in_time"] is False
    assert spec["strict_financial_point_in_time"] is False
    assert spec["research_only"] is True


def test_one_network_category_balanced_is_default_off_and_opt_in_only():
    parser = build_parser()
    assert parser.parse_args([]).include_one_network_category_balanced is False
    args = parser.parse_args(["--include-one-network-category-balanced"])
    assert args.include_one_network_category_balanced is True

    training = pd.DataFrame([
        {
            "factor": factor,
            "mean_rank_ic": 0.10,
            "top20_excess_forward_return": 0.02,
        }
        for factor in ONE_NETWORK_BASELINE_FACTORS
    ]).set_index("factor")
    default_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
    )
    enabled_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
        include_one_network_category_balanced=True,
    )

    assert ONE_NETWORK_CATEGORY_BALANCED_FACTOR not in default_specs
    assert ONE_NETWORK_CATEGORY_BALANCED_FACTOR in enabled_specs


def test_price_only_category_balanced_replaces_non_pit_factor_only():
    spec = _one_network_price_only_category_balanced_spec()

    assert spec["weights"] == {
        "network_cc": 0.25,
        "ma_convergence": 0.25,
        "disposition_reversal": 0.125,
        "coin_team": 0.125,
        "high_quality_momentum": 0.125,
        PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR: 0.125,
    }
    assert spec["factor_categories"] == {
        category: list(factors)
        for category, factors in (
            ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_CATEGORIES.items()
        )
    }
    assert spec["replaced_factor"] == "low_idiosyncratic_volatility"
    assert spec["replacement_factor"] == PRICE_ONLY_RESIDUAL_MOMENTUM_FACTOR
    assert spec["continuous_weight_search_used"] is False
    assert spec["participates_in_validation_selection"] is False
    assert spec["uses_financial_data"] is False
    assert spec["financial_point_in_time"] is True
    assert spec["strict_financial_point_in_time"] is True
    assert spec["research_only"] is True


def test_price_only_category_balanced_is_default_off_and_opt_in_only():
    parser = build_parser()
    assert (
        parser.parse_args([])
        .include_one_network_price_only_category_balanced
        is False
    )
    args = parser.parse_args([
        "--include-one-network-price-only-category-balanced",
    ])
    assert args.include_one_network_price_only_category_balanced is True

    training = pd.DataFrame([
        {
            "factor": factor,
            "mean_rank_ic": 0.10,
            "top20_excess_forward_return": 0.02,
        }
        for factor in ONE_NETWORK_BASELINE_FACTORS
    ]).set_index("factor")
    default_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
    )
    enabled_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
        include_one_network_price_only_category_balanced=True,
    )

    assert ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_FACTOR not in default_specs
    assert ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_FACTOR in enabled_specs


def test_price_only_low_residual_volatility_replaces_only_non_pit_factor():
    spec = _one_network_price_only_low_volatility_spec()

    assert len(spec["selected_factors"]) == 6
    assert "low_idiosyncratic_volatility" not in spec["selected_factors"]
    assert PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR in spec[
        "selected_factors"
    ]
    assert set(spec["weights"].values()) == {1.0 / 6.0}
    assert spec["replaced_factor"] == "low_idiosyncratic_volatility"
    assert spec["replacement_factor"] == (
        PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR
    )


def test_low_volatility_category_balanced_uses_fixed_frozen_factor_weights():
    spec = _one_network_price_only_low_volatility_category_balanced_spec()

    assert spec["weights"] == {
        "network_cc": 0.25,
        "ma_convergence": 0.25,
        "disposition_reversal": 0.125,
        "coin_team": 0.125,
        "high_quality_momentum": 0.125,
        PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR: 0.125,
    }
    assert spec["factor_categories"] == {
        category: list(factors)
        for category, factors in (
            ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_CATEGORIES.items()
        )
    }
    assert spec["replacement_factor"] == (
        PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR
    )
    assert spec["continuous_weight_search_used"] is False
    assert spec["participates_in_validation_selection"] is False
    assert spec["uses_financial_data"] is False
    assert spec["research_only"] is True


def test_scc_low_volatility_replaces_only_composite_network_factor():
    spec = _one_network_price_only_scc_low_volatility_spec()

    assert len(spec["selected_factors"]) == 6
    assert "network_cc" not in spec["selected_factors"]
    assert "network_scc" in spec["selected_factors"]
    assert PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR in spec[
        "selected_factors"
    ]
    assert set(spec["weights"].values()) == {1.0 / 6.0}
    assert spec["network_replaced_factor"] == "network_cc"
    assert spec["network_replacement_factor"] == "network_scc"
    assert spec["factor_replacements"] == {
        "low_idiosyncratic_volatility": (
            PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR
        ),
        "network_cc": "network_scc",
    }
    assert spec["continuous_weight_search_used"] is False
    assert spec["participates_in_validation_selection"] is False
    assert spec["uses_financial_data"] is False
    assert spec["research_only"] is True


def test_complete_low_volatility_combination_rejects_partial_factor_vectors():
    spec = _one_network_price_only_complete_low_volatility_spec()
    factors = list(spec["weights"])
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "complete"),
            (pd.Timestamp("2024-01-02"), "partial"),
        ],
        names=["date", "code"],
    )
    panel = pd.DataFrame(1.0, index=index, columns=factors)
    panel.loc[(pd.Timestamp("2024-01-02"), "partial"), factors[-1]] = None

    result = _apply_combination(panel, spec)

    assert pd.notna(result.loc[(pd.Timestamp("2024-01-02"), "complete")])
    assert pd.isna(result.loc[(pd.Timestamp("2024-01-02"), "partial")])
    assert spec["require_complete_factors"] is True


def test_median_low_volatility_combination_uses_middle_factor_consensus():
    spec = _one_network_price_only_median_low_volatility_spec()
    factors = list(spec["weights"])
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B"), (date, "C")],
        names=["date", "code"],
    )
    panel = pd.DataFrame({
        factor: [1.0, 2.0, 3.0]
        for factor in factors
    }, index=index)
    panel.loc[(date, "A"), factors[0]] = 100.0

    result = _apply_combination(panel, spec)

    assert result.loc[(date, "A")] == pytest.approx(1.0 / 3.0)
    assert result.loc[(date, "B")] == pytest.approx(2.0 / 3.0)
    assert result.loc[(date, "C")] == pytest.approx(1.0)
    assert spec["combination_type"] == "cross_sectional_rank_median"
    assert spec["continuous_weight_search_used"] is False
    assert spec["participates_in_validation_selection"] is False
    assert spec["uses_financial_data"] is False
    assert spec["financial_point_in_time"] is True
    assert spec["strict_financial_point_in_time"] is True
    assert spec["research_only"] is True


def test_price_only_low_residual_volatility_is_default_off_and_opt_in_only():
    parser = build_parser()
    assert (
        parser.parse_args([])
        .include_one_network_price_only_low_volatility
        is False
    )
    args = parser.parse_args([
        "--include-one-network-price-only-low-volatility",
    ])
    assert args.include_one_network_price_only_low_volatility is True

    training = pd.DataFrame([{
        "factor": factor,
        "mean_rank_ic": 0.10,
        "top20_excess_forward_return": 0.02,
    } for factor in ONE_NETWORK_BASELINE_FACTORS]).set_index("factor")
    default_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
    )
    enabled_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
        include_one_network_price_only_low_volatility=True,
    )

    assert ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_FACTOR not in default_specs
    assert ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_FACTOR in enabled_specs


def test_low_volatility_category_balanced_is_default_off_and_opt_in_only():
    parser = build_parser()
    assert (
        parser.parse_args([])
        .include_one_network_price_only_low_volatility_category_balanced
        is False
    )
    args = parser.parse_args([
        "--include-one-network-price-only-low-volatility-category-balanced",
    ])
    assert (
        args.include_one_network_price_only_low_volatility_category_balanced
        is True
    )

    training = pd.DataFrame([{
        "factor": factor,
        "mean_rank_ic": 0.10,
        "top20_excess_forward_return": 0.02,
    } for factor in ONE_NETWORK_BASELINE_FACTORS]).set_index("factor")
    default_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
    )
    enabled_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
        include_one_network_price_only_low_volatility_category_balanced=True,
    )

    factor = ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_FACTOR
    assert factor not in default_specs
    assert factor in enabled_specs


def test_scc_low_volatility_is_default_off_and_opt_in_only():
    parser = build_parser()
    assert (
        parser.parse_args([])
        .include_one_network_price_only_scc_low_volatility
        is False
    )
    args = parser.parse_args([
        "--include-one-network-price-only-scc-low-volatility",
    ])
    assert args.include_one_network_price_only_scc_low_volatility is True

    training = pd.DataFrame([{
        "factor": factor,
        "mean_rank_ic": 0.10,
        "top20_excess_forward_return": 0.02,
    } for factor in ONE_NETWORK_BASELINE_FACTORS]).set_index("factor")
    default_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
    )
    enabled_specs = _declared_combination_specs(
        training,
        positive_ic_top_n=5,
        include_one_network_price_only_scc_low_volatility=True,
    )

    assert ONE_NETWORK_PRICE_ONLY_SCC_LOW_VOLATILITY_FACTOR not in default_specs
    assert ONE_NETWORK_PRICE_ONLY_SCC_LOW_VOLATILITY_FACTOR in enabled_specs


def test_residual_momentum_low_volatility_is_one_fixed_opt_in_combination():
    parser = build_parser()
    assert (
        parser.parse_args([])
        .include_one_network_price_only_residual_momentum_low_volatility
        is False
    )
    args = parser.parse_args([
        "--include-one-network-price-only-residual-momentum-low-volatility",
    ])
    assert (
        args.include_one_network_price_only_residual_momentum_low_volatility
        is True
    )

    spec = _one_network_price_only_residual_momentum_low_volatility_spec()
    assert spec["momentum_replaced_factor"] == "high_quality_momentum"
    assert spec["momentum_replacement_factor"] == (
        "market_residual_momentum_252_skip20"
    )
    assert set(spec["selected_factors"]) == {
        "network_cc",
        "ma_convergence",
        "disposition_reversal",
        "market_residual_momentum_252_skip20",
        "coin_team",
        PRICE_ONLY_LOW_RESIDUAL_VOLATILITY_FACTOR,
    }
    assert set(spec["weights"].values()) == {1.0 / 6.0}
    assert spec["uses_financial_data"] is False
    assert spec["strict_financial_point_in_time"] is True
    assert spec["market_residual_momentum"]["lookback"] == 252
    assert spec["market_residual_momentum"]["skip"] == 20
    assert spec["require_complete_factors"] is True

    factors = _combination_factors_to_generate(
        include_one_network_price_only_residual_momentum_low_volatility=True,
        include_default_combinations=False,
    )
    assert factors == [
        ONE_NETWORK_PRICE_ONLY_RESIDUAL_MOMENTUM_LOW_VOLATILITY_FACTOR,
    ]


def test_fixed_research_combinations_do_not_change_validation_selection_set():
    original_selection_set = tuple(COMBINATION_FACTORS)

    default_factors = _combination_factors_to_generate()
    enabled_factors = _combination_factors_to_generate(
        include_market_residual_replacements=True,
        include_one_network_category_balanced=True,
        include_one_network_price_only_category_balanced=True,
        include_one_network_price_only_low_volatility=True,
        include_one_network_price_only_low_volatility_category_balanced=True,
        include_one_network_price_only_scc_low_volatility=True,
    )

    assert tuple(default_factors) == original_selection_set
    assert tuple(enabled_factors[:len(original_selection_set)]) == (
        original_selection_set
    )
    assert enabled_factors[len(original_selection_set):] == [
        *MARKET_RESIDUAL_REPLACEMENT_FACTORS,
        ONE_NETWORK_CATEGORY_BALANCED_FACTOR,
        ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_FACTOR,
        ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_FACTOR,
        ONE_NETWORK_PRICE_ONLY_COMPLETE_LOW_VOLATILITY_FACTOR,
        ONE_NETWORK_PRICE_ONLY_MEDIAN_LOW_VOLATILITY_FACTOR,
        ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_FACTOR,
        ONE_NETWORK_PRICE_ONLY_SCC_LOW_VOLATILITY_FACTOR,
    ]
    assert ONE_NETWORK_CATEGORY_BALANCED_FACTOR not in COMBINATION_FACTORS


def test_explicit_research_generation_can_skip_default_combinations():
    factors = _combination_factors_to_generate(
        include_one_network_category_balanced=True,
        include_default_combinations=False,
    )

    assert factors == [ONE_NETWORK_CATEGORY_BALANCED_FACTOR]

    price_only_factors = _combination_factors_to_generate(
        include_one_network_price_only_category_balanced=True,
        include_default_combinations=False,
    )
    assert price_only_factors == [
        ONE_NETWORK_PRICE_ONLY_CATEGORY_BALANCED_FACTOR,
    ]

    low_volatility_factors = _combination_factors_to_generate(
        include_one_network_price_only_low_volatility=True,
        include_default_combinations=False,
    )
    assert low_volatility_factors == [
        ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_FACTOR,
        ONE_NETWORK_PRICE_ONLY_COMPLETE_LOW_VOLATILITY_FACTOR,
        ONE_NETWORK_PRICE_ONLY_MEDIAN_LOW_VOLATILITY_FACTOR,
    ]

    category_balanced_low_volatility_factors = _combination_factors_to_generate(
        include_one_network_price_only_low_volatility_category_balanced=True,
        include_default_combinations=False,
    )
    assert category_balanced_low_volatility_factors == [
        ONE_NETWORK_PRICE_ONLY_LOW_VOLATILITY_CATEGORY_BALANCED_FACTOR,
    ]

    scc_low_volatility_factors = _combination_factors_to_generate(
        include_one_network_price_only_scc_low_volatility=True,
        include_default_combinations=False,
    )
    assert scc_low_volatility_factors == [
        ONE_NETWORK_PRICE_ONLY_SCC_LOW_VOLATILITY_FACTOR,
    ]


def test_chunked_cli_explicitly_enables_market_residual_replacements():
    assert (
        build_parser().parse_args([])
        .include_market_residual_momentum_replacements
        is False
    )
    args = build_parser().parse_args([
        "--include-market-residual-momentum-replacements",
    ])

    assert args.include_market_residual_momentum_replacements is True


def test_chunked_passes_compare_include_flag_only_for_replacements(
    tmp_path,
    monkeypatch,
):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr("stock_research.research.quantsplaybook_chunked.subprocess.run", fake_run)
    common = {
        "year": 2024,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "formula_history": tmp_path / "formula.csv",
        "database_path": tmp_path / "market.db",
        "holding_days": 20,
        "minimum_history": 60,
        "top_n": 50,
    }
    _run_year_source_generation(
        output_root=tmp_path / "default",
        include_market_residual_momentum=False,
        **common,
    )
    _run_year_source_generation(
        output_root=tmp_path / "enabled",
        include_market_residual_momentum=True,
        **common,
    )
    _run_year_source_generation(
        output_root=tmp_path / "low_volatility",
        include_low_market_residual_volatility=True,
        **common,
    )

    assert "--include-market-residual-momentum" not in commands[0][0]
    assert "--include-market-residual-momentum" in commands[1][0]
    assert "--include-low-market-residual-volatility" in commands[2][0]
    assert commands[0][1] is True
    assert commands[1][1] is True
    assert commands[2][1] is True


def test_chunked_can_skip_unused_source_candidate_generation(
    tmp_path,
    monkeypatch,
):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr("stock_research.research.quantsplaybook_chunked.subprocess.run", fake_run)
    _run_year_source_generation(
        year=2024,
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_root=tmp_path,
        formula_history=tmp_path / "formula.csv",
        database_path=tmp_path / "market.db",
        holding_days=20,
        minimum_history=60,
        top_n=100,
        skip_candidate_generation=True,
    )

    assert "--skip-candidate-generation" in commands[0][0]
    assert commands[0][1] is True
    args = build_parser().parse_args(["--skip-source-candidate-generation"])
    assert args.skip_source_candidate_generation is True


def test_capped_weights_sum_to_one_and_respect_cap():
    weights = _capped_normalized_weights(
        {"a": 100.0, "b": 3.0, "c": 2.0, "d": 1.0},
        cap=0.30,
    )

    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert max(weights.values()) <= 0.30 + 1e-12


def test_low_correlation_selection_uses_training_correlation_only():
    training = pd.DataFrame([
        {
            "factor": "network_cc",
            "mean_rank_ic": 0.10,
            "top20_excess_forward_return": 0.02,
        },
        {
            "factor": "network_scc",
            "mean_rank_ic": 0.09,
            "top20_excess_forward_return": 0.02,
        },
        {
            "factor": "shadow_reversal",
            "mean_rank_ic": 0.08,
            "top20_excess_forward_return": 0.02,
        },
    ]).set_index("factor")
    correlation = pd.DataFrame(
        [
            [1.0, 0.99, 0.10],
            [0.99, 1.0, 0.20],
            [0.10, 0.20, 1.0],
        ],
        index=training.index,
        columns=training.index,
    )

    selected = _select_low_correlation_factors(
        training,
        correlation,
        target_count=2,
    )

    assert selected == ["network_cc", "shadow_reversal"]


def test_low_correlation_selection_caps_network_and_fills_from_other_category():
    factors = (
        "network_cc",
        "network_scc",
        "shadow_reversal",
        "salience_str",
        "ma_convergence",
    )
    training = pd.DataFrame([
        {
            "factor": factor,
            "mean_rank_ic": 0.10 - index * 0.01,
            "top20_excess_forward_return": 0.02,
        }
        for index, factor in enumerate(factors)
    ]).set_index("factor")
    correlation = pd.DataFrame(
        0.0,
        index=training.index,
        columns=training.index,
    )
    for factor in factors:
        correlation.loc[factor, factor] = 1.0

    selected = _select_low_correlation_factors(
        training,
        correlation,
        target_count=4,
        max_network_factors=1,
    )

    assert selected == [
        "network_cc",
        "shadow_reversal",
        "salience_str",
        "ma_convergence",
    ]
    assert sum(factor.startswith("network_") for factor in selected) == 1

    specs = _declared_combination_specs(
        training,
        positive_ic_top_n=len(training),
        training_rank_correlation=correlation,
        low_corr_max_network_factors=1,
    )
    assert specs["playbook_low_corr"]["selected_factors"] == selected
    assert specs["playbook_low_corr"][
        "low_corr_max_network_factors"
    ] == 1


def test_low_corr_network_factor_limit_defaults_to_legacy_unlimited_behavior():
    args = build_parser().parse_args([])

    assert args.low_corr_max_network_factors is None


def test_low_corr_network_factor_limit_validates_cli_boundaries():
    parser = build_parser()

    assert parser.parse_args([
        "--low-corr-max-network-factors", "0",
    ]).low_corr_max_network_factors == 0
    assert parser.parse_args([
        "--low-corr-max-network-factors", "1",
    ]).low_corr_max_network_factors == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--low-corr-max-network-factors", "-1"])


def test_factor_quota_rotates_first_places_across_factors():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, code) for code in ("A", "B", "C", "D")],
        names=["date", "code"],
    )
    panel = pd.DataFrame({
        "factor_a": [4.0, 3.0, 2.0, 1.0],
        "factor_b": [1.0, 2.0, 3.0, 4.0],
    }, index=index)

    result = round_robin_factor_combination(
        panel,
        ["factor_a", "factor_b"],
    ).sort_values(ascending=False)

    assert result.index.get_level_values("code").tolist() == [
        "A", "D", "B", "C",
    ]


def test_force_years_argument_accepts_targeted_rebuilds():
    args = build_parser().parse_args(["--force-years", "2021,2024"])

    assert args.force_years == "2021,2024"


def test_candidate_profile_enrichment_can_be_skipped_for_fast_rebuilds():
    args = build_parser().parse_args(["--skip-candidate-profile-enrichment"])

    assert args.skip_candidate_profile_enrichment is True


def test_merge_candidate_chunks_combines_outcome_labels(tmp_path):
    periods = [
        (2021, "2021-01-01", "2021-12-31"),
        (2022, "2022-01-01", "2022-12-31"),
    ]
    for year in (2021, 2022):
        source = tmp_path / "chunks" / str(year) / "candidates" / "factor_a"
        source.mkdir(parents=True)
        date = f"{year}-01-04"
        candidate_path = source / f"candidates_{date}.csv"
        pd.DataFrame([{"date": date, "code": "sh.600000"}]).to_csv(
            candidate_path, index=False,
        )
        outcome_path = source / "candidate_outcomes.csv"
        pd.DataFrame([{
            "date": date,
            "code": "sh.600000",
            "playbook_factor_rank": 1,
            "label_available": True,
            "forward_return": 0.10,
        }]).to_csv(outcome_path, index=False, encoding="utf-8-sig")
        from stock_research.research.quantsplaybook_compare import _sha256

        (source / "manifest.json").write_text(json.dumps({
            "factor": "factor_a",
            "uses_financial_data": True,
            "financial_point_in_time": False,
            "strict_financial_point_in_time": False,
            "research_only": True,
            "snapshots": [{
                "date": date,
                "file": candidate_path.name,
                "sha256": _sha256(candidate_path),
                "candidate_count": 1,
                "financial_point_in_time": False,
                "strict_financial_point_in_time": False,
            }],
            "candidate_outcome_labels": {
                "file": outcome_path.name,
                "sha256": _sha256(outcome_path),
            },
        }), encoding="utf-8")

    _merge_candidate_chunks(
        periods=periods,
        output_root=tmp_path,
        start_date="2021-01-01",
        end_date="2022-12-31",
        combination_manifest={
            "annual_label_boundary": "annual boundaries excluded",
            "combinations": {
                "factor_a": {
                    "low_corr_max_network_factors": 1,
                    "uses_financial_data": True,
                    "financial_point_in_time": False,
                    "strict_financial_point_in_time": False,
                    "research_only": True,
                },
            },
        },
    )

    target = tmp_path / "candidates" / "factor_a"
    outcomes = pd.read_csv(target / "candidate_outcomes.csv")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))

    assert outcomes[["date", "code"]].to_dict("records") == [
        {"date": "2021-01-04", "code": "sh.600000"},
        {"date": "2022-01-04", "code": "sh.600000"},
    ]
    assert manifest["candidate_outcome_labels"]["row_count"] == 2
    assert manifest["candidate_outcome_labels"]["available_count"] == 2
    assert manifest["candidate_outcome_labels"][
        "future_return_used_for_candidate_selection"
    ] is False
    assert manifest["factor_metadata"][
        "low_corr_max_network_factors"
    ] == 1
    assert manifest["uses_financial_data"] is True
    assert manifest["financial_point_in_time"] is False
    assert manifest["strict_financial_point_in_time"] is False
    assert manifest["research_only"] is True
    assert all(
        row["financial_point_in_time"] is False
        and row["strict_financial_point_in_time"] is False
        for row in manifest["snapshots"]
    )


def test_merge_candidate_chunks_rejects_corrupted_candidate_snapshot(tmp_path):
    source = tmp_path / "chunks" / "2021" / "candidates" / "factor_a"
    source.mkdir(parents=True)
    candidate_path = source / "candidates_2021-01-04.csv"
    pd.DataFrame([{
        "date": "2021-01-04",
        "code": "sh.600000",
    }]).to_csv(candidate_path, index=False)
    from stock_research.research.quantsplaybook_compare import _sha256

    original_sha256 = _sha256(candidate_path)
    (source / "manifest.json").write_text(json.dumps({
        "factor": "factor_a",
        "uses_financial_data": False,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "research_only": True,
        "snapshots": [{
            "date": "2021-01-04",
            "file": candidate_path.name,
            "sha256": original_sha256,
            "candidate_count": 1,
            "financial_point_in_time": True,
            "strict_financial_point_in_time": True,
        }],
    }), encoding="utf-8")
    pd.DataFrame([{
        "date": "2021-01-04",
        "code": "sh.600999",
    }]).to_csv(candidate_path, index=False)

    with pytest.raises(RuntimeError, match="candidate snapshot hash mismatch"):
        _merge_candidate_chunks(
            periods=[(2021, "2021-01-01", "2021-12-31")],
            output_root=tmp_path,
            start_date="2021-01-01",
            end_date="2021-12-31",
            combination_manifest={
                "combinations": {
                    "factor_a": {
                        "uses_financial_data": False,
                        "financial_point_in_time": True,
                        "strict_financial_point_in_time": True,
                        "research_only": True,
                    },
                },
            },
        )


def test_combined_data_coverage_sums_disjoint_annual_membership(tmp_path):
    periods = [
        (2021, "2021-01-01", "2021-12-31"),
        (2022, "2022-01-01", "2022-12-31"),
    ]
    for year, eligible_rows, risk_rows in (
        (2021, 100, 3),
        (2022, 120, 4),
    ):
        path = tmp_path / "chunks" / str(year)
        path.mkdir(parents=True)
        (path / "data_coverage.json").write_text(
            json.dumps({
                "database_path": "db.sqlite3",
                "history_start_date": f"{year - 2}-02-01",
                "eligible_rows": eligible_rows,
                "eligible_codes": year - 2000,
                "snapshot_dates": 10,
                "risk_warning_first_date": f"{year}-01-04",
                "risk_warning_last_date": f"{year}-12-30",
                "risk_warning_rows": risk_rows,
            }),
            encoding="utf-8",
        )

    result = _combined_data_coverage(
        periods=periods,
        output_root=tmp_path,
    )

    assert result["eligible_rows"] == 220
    assert result["snapshot_dates"] == 20
    assert result["risk_warning_rows"] == 7
    assert result["risk_warning_first_date"] == "2021-01-04"
