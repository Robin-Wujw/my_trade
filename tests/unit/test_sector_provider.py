import pandas as pd
import pytest

from stock_research.market.sector_provider import (
    coverage_can_still_pass,
    effective_pipeline_retries,
    minimum_fresh_count,
    sector_history_is_fresh,
    validate_coverage,
    validate_sector_histories,
)


def test_coverage_accepts_86_of_90_and_rejects_85():
    accepted = validate_coverage(expected=90, fresh=86, minimum=0.95)
    rejected = validate_coverage(expected=90, fresh=85, minimum=0.95)

    assert accepted.coverage == pytest.approx(86 / 90)
    assert accepted.passed
    assert rejected.coverage == pytest.approx(85 / 90)
    assert not rejected.passed


def test_coverage_early_stop_boundary_for_ninety_boards():
    assert minimum_fresh_count(90, 0.95) == 86
    assert coverage_can_still_pass(
        expected=90,
        completed=4,
        fresh=0,
        minimum=0.95,
    )
    assert not coverage_can_still_pass(
        expected=90,
        completed=5,
        fresh=0,
        minimum=0.95,
    )


def test_pipeline_retry_is_not_multiplied_by_adapter_retry():
    class InternallyRetriedProvider:
        REQUEST_ATTEMPTS = 2

    class PlainProvider:
        pass

    assert effective_pipeline_retries(InternallyRetriedProvider(), 4) == 1
    assert effective_pipeline_retries(PlainProvider(), 4) == 4


def test_validate_sector_histories_excludes_stale_and_missing_frames():
    histories = {
        "fresh-edge": pd.DataFrame([{"date": "2026-07-03", "close": 10.0}]),
        "fresh-current": pd.DataFrame(
            [{"date": pd.Timestamp("2026-07-10"), "close": 11.0}]
        ),
        "stale": pd.DataFrame([{"date": "2026-07-02", "close": 8.0}]),
        "empty": pd.DataFrame(columns=["date", "close"]),
    }

    fresh, coverage = validate_sector_histories(
        ["fresh-edge", "fresh-current", "stale", "empty", "absent"],
        histories,
        observation_date="2026-07-10",
        max_stale_days=7,
        minimum=0.4,
    )

    assert list(fresh) == ["fresh-edge", "fresh-current"]
    assert coverage.expected == 5
    assert coverage.fresh == 2
    assert coverage.stale == 1
    assert coverage.missing == 2
    assert coverage.coverage == pytest.approx(0.4)
    assert coverage.passed
    assert histories["fresh-edge"]["date"].tolist() == ["2026-07-03"]


def test_current_but_truncated_history_fails_depth_coverage_gate():
    truncated = pd.DataFrame(
        [{"date": "2026-07-10", "close": 11.0}]
    )

    fresh, coverage = validate_sector_histories(
        ["truncated"],
        {"truncated": truncated},
        observation_date="2026-07-10",
        max_stale_days=0,
        minimum_rows=21,
        minimum=1.0,
    )

    assert not sector_history_is_fresh(
        truncated,
        observation_date="2026-07-10",
        max_stale_days=0,
        minimum_rows=21,
    )
    assert fresh == {}
    assert coverage.fresh == 0
    assert coverage.missing == 1
    assert not coverage.passed


def test_validate_sector_histories_rejects_invalid_observation_date():
    with pytest.raises(ValueError, match="observation_date"):
        validate_sector_histories([], {}, observation_date="not-a-date")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected": -1, "fresh": 0}, "expected"),
        ({"expected": 1, "fresh": 2}, "fresh"),
        ({"expected": 1, "fresh": 1, "minimum": 1.1}, "minimum"),
    ],
)
def test_validate_coverage_rejects_invalid_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_coverage(**kwargs)
