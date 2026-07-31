from apps.quantsplaybook_portfolio_matrix import _periods


def test_portfolio_matrix_includes_independent_years_and_full_period():
    assert _periods(2024, 2026, "2026-07-21") == [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-07-21"),
        ("2024_to_date", "2024-01-01", "2026-07-21"),
    ]
