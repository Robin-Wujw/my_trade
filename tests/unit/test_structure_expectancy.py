import hashlib

import pandas as pd
import pytest

from stock_research.backtesting.signals import _semantic_right_entry_gate
from stock_research.research.structure_events import EventExpectancy
from stock_research.research.structure_expectancy import StructureExpectancyBook


def estimate(status="forward_sample_ready", mean=4.):
    return EventExpectancy("2027-11-01T17:00:00+08:00", "test", 60, 100, 252,
                           mean, .6, status, "a" * 64)


def test_research_target_consumes_mean_net_r_without_max_price_fallback():
    prices = pd.DataFrame({"date": pd.bdate_range("2027-10-01", periods=2), "close": 10., "high": 999.})
    signal = {"signal_type": "test", "order_type": "close", "trigger": 10., "stop": 9.5, "target_price": 999.}
    ok, result, reason = _semantic_right_entry_gate(prices, 1, {}, signal, research_expectancy=estimate())
    assert ok and reason == "ok" and result["entry_target_price"] == 12.
    assert result["entry_target_basis"] == "visible_forward_structure_net_expectancy"
    assert result["entry_reward_risk"] == 4.
    assert signal["target_price"] == 999.  # caller's original observation remains intact


@pytest.mark.parametrize("value", [estimate("insufficient_visible_campaigns"), estimate(mean=-1.)])
def test_missing_or_nonpositive_forward_expectancy_blocks_instead_of_using_legacy_targets(value):
    prices = pd.DataFrame({"close": [10.]})
    signal = {"signal_type": "test", "order_type": "close", "trigger": 10., "stop": 9.5, "target_price": 999.}
    ok, _, reason = _semantic_right_entry_gate(prices, 0, {}, signal, research_expectancy=value)
    assert not ok and reason.startswith("structure_expectancy_")


def test_empty_or_future_snapshot_cannot_supply_current_expectancy():
    book = StructureExpectancyBook([])
    original = book.estimate("2026-10-14", "test")
    book.snapshots.append((pd.Timestamp("2030-01-01T17:00:00+08:00"), [], [], []))
    later = book.estimate("2026-10-15", "test")
    assert original.mean_net_r is later.mean_net_r is None
    assert original.campaigns == later.campaigns == 0


def test_manifest_receipt_cannot_bypass_source_hash_verification(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        StructureExpectancyBook([{"manifest_path": str(path), "manifest_sha256": hashlib.sha256(b"original").hexdigest()}])


def test_receipt_for_another_file_cannot_bind_the_unhashed_manifest_in_its_directory(tmp_path):
    path = tmp_path / "unrelated.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="exact manifest.json"):
        StructureExpectancyBook([{"manifest_path": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])


@pytest.mark.parametrize("kwargs", [
    {}, {"close_confirmed_execution": "close_proxy", "signals_effective_next_day": True},
    {"close_confirmed_execution": "next_open", "signals_effective_next_day": False},
])
def test_postclose_expectancy_cannot_enter_same_day_execution_paths(kwargs):
    from stock_research.research.portfolio_replay import run_portfolio_research
    with pytest.raises(ValueError, match="explicit next_open"):
        run_portfolio_research(research_policy={}, structure_label_manifests=[], **kwargs)
