import json

import pandas as pd

from apps.quantsplaybook_chunked import (
    _combined_data_coverage,
    _declared_combination_specs,
    build_parser,
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
