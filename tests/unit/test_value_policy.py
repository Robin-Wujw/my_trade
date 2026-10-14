import pytest

from stock_research.research.value_policy import capped_value_observation


def source():
    return {"observation_date": "2026-10-15", "captured_at": "2026-10-15T17:00:00+08:00",
            "book_value_per_share": 3., "annual_eps_excl_comparable": 1., "earnings_yoy": .5,
            "annual_report_period": "2025-12-31", "annual_announcement_date": "2026-04-01",
            "announcement_date": "2026-08-20", "annual_source_sha256": "a" * 64,
            "eps_adjustment_source_sha256": "b" * 64, "financial_source_sha256": "c" * 64,
            "industry_source_sha256": "d" * 64, "close": 15.}


def test_registered_cap_uses_actual_annual_eps_and_preserves_source_identity():
    row = source()
    result = capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")
    assert result["original_value_line"] == 18 and result["value_line"] == 15
    assert result["industry_whitelist_changed"] is False
    assert result["strict_point_in_time"] is False
    assert row["earnings_yoy"] == .5
    row["earnings_yoy"] = .1
    result = capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")
    assert result["original_value_line"] == result["value_line"]


def test_missing_annual_adjustment_or_future_announcement_is_not_a_valid_comparison():
    row = source()
    row.pop("eps_adjustment_source_sha256")
    assert capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")["value_line"] is None
    row = source()
    row["annual_report_period"] = "2026-06-30"
    with pytest.raises(ValueError, match="annual"):
        capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")
    row = source()
    row["announcement_date"] = "2026-10-16"
    with pytest.raises(ValueError, match="announcements"):
        capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")


def test_historical_backfill_cannot_be_promoted_to_r07_forward_observation():
    row = source()
    row["observation_date"] = "2026-07-21"
    with pytest.raises(ValueError, match="same-day capture"):
        capped_value_observation(row, as_of="2026-10-15T20:00:00+08:00")
