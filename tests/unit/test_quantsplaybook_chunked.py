import json

import pandas as pd

from apps.quantsplaybook_chunked import (
    _capped_normalized_weights,
    _combined_data_coverage,
    _declared_combination_specs,
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
