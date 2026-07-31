import pandas as pd

from apps.quantsplaybook_results import (
    expected_periods,
    strategy_comparison,
)


def test_expected_periods_include_independent_years_and_full_period():
    assert expected_periods(2024, 2026, "2026-07-21") == [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-07-21"),
        ("2024_to_date", "2024-01-01", "2026-07-21"),
    ]


def test_strategy_selection_uses_validation_not_better_oos_return():
    rows = []
    values = {
        "validation_winner": {2024: 10.0, 2025: -2.0, 2026: -1.0},
        "oos_winner": {2024: 5.0, 2025: 20.0, 2026: 10.0},
    }
    for factor, yearly in values.items():
        for year, value in yearly.items():
            rows.append({
                "factor": factor,
                "period": str(year),
                "final_return_pct": value,
                "exclude_top1_approx_final_return_pct": value - 1,
                "exclude_top3_approx_final_return_pct": value - 3,
                "maximum_drawdown_pct": -5.0,
                "profit_loss_ratio": 1.2,
            })
        rows.append({
            "factor": factor,
            "period": "2024_to_date",
            "final_return_pct": sum(yearly.values()),
            "exclude_top1_approx_final_return_pct": sum(yearly.values()) - 1,
            "exclude_top3_approx_final_return_pct": sum(yearly.values()) - 3,
            "maximum_drawdown_pct": -10.0,
            "profit_loss_ratio": 1.1,
            "top1_positive_profit_share_pct": 20.0,
            "top3_positive_profit_share_pct": 40.0,
        })

    result = strategy_comparison(
        pd.DataFrame(rows),
        validation_year=2024,
        oos_start_year=2025,
    )

    assert result.iloc[0]["factor"] == "validation_winner"
    assert result.iloc[0]["selected_by_validation"]
    assert not result.iloc[0]["oos_robust"]
    assert not result.iloc[0]["oos_top3_robust"]
    assert result.iloc[1]["oos_robust"]
    assert result.iloc[1]["oos_top3_robust"]


def test_strategy_selection_uses_raw_validation_return_not_top1_diagnostic():
    rows = []
    for factor, raw, exclude_top1 in (
        ("raw_winner", 20.0, -5.0),
        ("top1_diagnostic_winner", 10.0, 8.0),
    ):
        for year in (2024, 2025, 2026):
            rows.append({
                "factor": factor,
                "period": str(year),
                "final_return_pct": raw if year == 2024 else 1.0,
                "exclude_top1_approx_final_return_pct": (
                    exclude_top1 if year == 2024 else -1.0
                ),
                "exclude_top3_approx_final_return_pct": -2.0,
                "maximum_drawdown_pct": -5.0,
                "profit_loss_ratio": 1.2,
            })
        rows.append({
            "factor": factor,
            "period": "2024_to_date",
            "final_return_pct": raw + 2.0,
            "exclude_top1_approx_final_return_pct": exclude_top1,
            "exclude_top3_approx_final_return_pct": -3.0,
            "maximum_drawdown_pct": -8.0,
            "profit_loss_ratio": 1.2,
            "top1_positive_profit_share_pct": 50.0,
            "top3_positive_profit_share_pct": 80.0,
        })

    result = strategy_comparison(pd.DataFrame(rows))

    assert result.iloc[0]["factor"] == "raw_winner"
    assert result.iloc[0]["oos_robust"]
