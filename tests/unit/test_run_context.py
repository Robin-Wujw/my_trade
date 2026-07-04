from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from stock_research.core import RunContext, RunMode


def make_context(**overrides):
    values = {
        "observation_date": date(2026, 6, 30),
        "market_cutoff": date(2026, 6, 30),
        "financial_cutoff": datetime(2026, 6, 30, 15, tzinfo=timezone.utc),
        "report_period": date(2026, 3, 31),
        "mode": RunMode.PRODUCTION,
        "code_version": "abc123",
        "now": datetime(2026, 7, 2, 1, 2, 3, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return RunContext.create(**values)


def test_create_builds_traceable_run_id_and_freezes_context():
    context = make_context()

    assert context.run_id.startswith("20260630-20260702T010203Z-")
    assert len(context.run_id.rsplit("-", 1)[-1]) == 8
    with pytest.raises(FrozenInstanceError):
        context.code_version = "changed"


def test_rejects_market_cutoff_after_observation_date():
    with pytest.raises(ValueError, match="market_cutoff"):
        make_context(market_cutoff=date(2026, 7, 1))


def test_rejects_financial_cutoff_after_observation_day():
    with pytest.raises(ValueError, match="financial_cutoff"):
        make_context(financial_cutoff=datetime(2026, 7, 1, tzinfo=timezone.utc))


def test_financial_cutoff_uses_shanghai_market_day():
    with pytest.raises(ValueError, match="financial_cutoff"):
        make_context(financial_cutoff=datetime(2026, 6, 30, 17, tzinfo=timezone.utc))


def test_rejects_report_period_after_observation_date():
    with pytest.raises(ValueError, match="report_period"):
        make_context(report_period=date(2026, 9, 30))


def test_rejects_naive_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_context(financial_cutoff=datetime(2026, 6, 30, 15))


def test_serializes_values_for_storage():
    payload = make_context(mode=RunMode.OFFLINE).to_record()

    assert payload["mode"] == "offline"
    assert payload["observation_date"] == date(2026, 6, 30)
    assert payload["created_at"] == datetime(2026, 7, 2, 1, 2, 3, tzinfo=timezone.utc)
