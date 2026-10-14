import pytest

from stock_research.core.financial_period import resolve_report_period


@pytest.mark.parametrize(
    "date, period",
    [
        ("2026-01-01", "2025-09-30"),
        ("2026-04-29", "2025-09-30"),
        ("2026-04-30", "2026-03-31"),
        ("2026-08-30", "2026-03-31"),
        ("2026-08-31", "2026-06-30"),
        ("2026-10-30", "2026-06-30"),
        ("2026-10-31", "2026-09-30"),
    ],
)
def test_report_period_uses_statutory_boundaries(date, period):
    assert resolve_report_period(date) == period


@pytest.mark.parametrize("period", ["2025-12-31", "2026-03-31", "2099-12-31"])
def test_report_period_rejects_undisclosed_override(period):
    with pytest.raises(ValueError, match="not visible"):
        resolve_report_period("2026-04-29", period)


def test_report_period_allows_explicit_older_visible_period():
    assert resolve_report_period("2026-10-14", "2025-12-31") == "2025-12-31"


@pytest.mark.parametrize("period", ["2026-05-01", "NaT", "garbage"])
def test_report_period_rejects_invalid_period(period):
    with pytest.raises(ValueError):
        resolve_report_period("2026-10-14", period)


def test_report_period_rejects_missing_observation_date():
    with pytest.raises(ValueError):
        resolve_report_period("NaT")
