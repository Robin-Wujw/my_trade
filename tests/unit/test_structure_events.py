import pandas as pd
import pytest

from stock_research.research import structure_events as events


def sample():
    entries = [{"event_id": "a", "code": "sz.000001", "observation_date": "2026-10-15",
                "captured_at": "2026-10-15T17:00:00+08:00", "signal_type": "test", "source_sha256": "a" * 64},
               {"event_id": "b", "code": "sz.000001", "observation_date": "2026-10-16",
                "captured_at": "2026-10-16T17:00:00+08:00", "signal_type": "test", "source_sha256": "b" * 64}]
    labels = [{"event_id": "a", "horizon": 20, "exit_date": "2026-11-15",
               "available_at": "2026-11-15T17:00:00+08:00", "net_r": 1.5, "executable": True,
               "cost_included": True, "label_source_sha256": "c" * 64},
              {"event_id": "b", "horizon": 20, "exit_date": "2026-11-16",
               "available_at": "2026-11-16T17:00:00+08:00", "net_r": 999., "executable": True,
               "cost_included": True, "label_source_sha256": "d" * 64}]
    captures = [{**{key: row[key] for key in ("observation_date", "captured_at", "source_sha256")},
                 "is_trading_day": True} for row in entries]
    return entries, labels, captures


def test_pending_or_future_outcomes_cannot_change_visible_expectancy():
    entries, labels, captures = sample()
    parameters = dict(as_of="2026-11-15T20:00:00+08:00", signal_type="test", horizon=20)
    estimate = events.visible_expectancy(entries, labels, captures, **parameters)
    assert estimate.campaigns == 1 and estimate.mean_net_r == 1.5
    assert estimate == events.visible_expectancy(entries, labels[:1], captures, **parameters)
    assert events.expected_target(10., 9., estimate)["target"] is None
    pending = events.visible_expectancy(entries, labels, captures, as_of="2026-10-16T20:00:00+08:00", signal_type="test", horizon=20)
    assert pending.mean_net_r is None


def test_overlapping_campaigns_and_historical_backfill_do_not_satisfy_forward_gate():
    entries, labels, captures = sample()
    parameters = dict(as_of="2026-12-01T20:00:00+08:00", signal_type="test", horizon=20)
    estimate = events.visible_expectancy(entries, labels, captures, **parameters)
    assert estimate.campaigns == 1
    entries[0]["observation_date"] = "2021-01-04"
    backfill = events.visible_expectancy(entries, labels, captures, **parameters)
    assert backfill.campaigns == 1 and backfill.mean_net_r == 999.
    labels[1]["executable"] = False
    assert events.visible_expectancy(entries, labels, captures, **parameters).campaigns == 0


def test_unquoted_times_or_missing_cost_evidence_cannot_become_execution_labels():
    entries, labels, captures = sample()
    parameters = dict(as_of="2026-11-15T20:00:00+08:00", signal_type="test", horizon=20)
    labels[0]["cost_included"] = False
    assert events.visible_expectancy(entries, labels, captures, **parameters).campaigns == 0
    labels[0]["available_at"] = "2026-11-15T17:00:00"
    with pytest.raises(ValueError, match="timezone"):
        events.visible_expectancy(entries, labels, captures, **parameters)


def test_full_membership_events_include_non_top50_and_ignore_future_prices(monkeypatch):
    frames = {"sz.000001": pd.DataFrame({"date": pd.bdate_range(end="2026-10-14", periods=61), "close": 10.}),
              "sz.000002": pd.DataFrame({"date": pd.bdate_range(end="2026-10-14", periods=61), "close": 10.})}
    membership = [{"date": "2026-10-14", "code": code, "membership_as_of": "2026-10-14",
                   "is_listed": True, "is_traded": True, "is_st": False, "history_days": 61,
                   "adv20_yuan": 30_000_000} for code in frames]
    def signal(frame, index):
        assert len(frame) == 61 and frame.date.max() == pd.Timestamp("2026-10-14")
        return {"signal_type": "test", "order_type": "close", "stop": 9., "target_price": 13.}
    monkeypatch.setattr(events, "_right_signal", signal)
    frames["sz.000002"] = pd.concat([frames["sz.000002"], pd.DataFrame({"date": [pd.Timestamp("2026-10-15")], "close": [999.]})])
    rows = events.observed_structure_events(frames, membership, "2026-10-14", captured_at="2026-10-14T17:00:00+08:00",
                                             source_sha256="a" * 64, factor_ranks={"sz.000001": 1, "sz.000002": 500})
    assert len(rows) == 2 and rows[1]["factor_rank"] == 500
    assert all(row["planned_r"] == 3 for row in rows)
    membership[1]["is_st"] = True
    assert len(events.observed_structure_events(frames, membership, "2026-10-14", captured_at="2026-10-14T17:00:00+08:00", source_sha256="a" * 64)) == 1
