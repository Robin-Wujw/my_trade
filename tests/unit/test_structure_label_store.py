import json

import pandas as pd
import pytest

from stock_research.research import frozen_forward, structure_events, structure_event_store
from stock_research.research import structure_label_store as store


def sources(tmp_path, monkeypatch, count=24):
    monkeypatch.setattr(frozen_forward, "capture_clock", lambda day: f"{day}T18:00:00+08:00")
    monkeypatch.setattr(structure_events, "_right_signal", lambda *args: {
        "order_type": "close", "signal_type": "test", "stop": 9., "target_price": 13.})
    dates = pd.bdate_range("2026-10-15", periods=count)
    history = pd.bdate_range(end=dates[0], periods=65)
    bars = pd.DataFrame({"date": history, "open": 10., "high": 10.2, "low": 9.8, "close": 10.,
                         "volume": 1_000_000., "tradestatus": 1, "raw_to_qfq_factor": 1.})
    upstream = tmp_path / "sync.json"
    upstream.write_text('{"fixture":true}', encoding="utf-8")
    event_directory = tmp_path / "events"
    structure_event_store.capture_structure_session(event_directory, {"sz.000001": bars}, [{
        "date": str(dates[0].date()), "code": "sz.000001", "membership_as_of": str(dates[0].date()),
        "is_listed": True, "is_traded": True, "is_st": False, "history_days": 65, "adv20_yuan": 30_000_000.}],
        str(dates[0].date()), captured_at=f"{dates[0].date()}T17:00:00+08:00", source_receipts={"sync": upstream})
    outcome = bars.iloc[[-1]].copy()
    outcome = pd.concat([outcome] * len(dates), ignore_index=True)
    outcome["date"] = dates
    outcome = outcome.assign(up_limit=11., down_limit=9.)
    kwargs = dict(event_directories=[event_directory], raw_frames={"sz.000001": outcome}, sessions=list(dates),
                  corporate_actions={}, observation_date=str(dates[-1].date()),
                  captured_at=f"{dates[-1].date()}T17:00:00+08:00", source_receipts={"sync_state": upstream},
                  action_coverage_complete={"sz.000001": True})
    return kwargs


def test_snapshot_records_audited_campaigns_pending_horizons_and_immutable_raw_sources(tmp_path, monkeypatch):
    kwargs = sources(tmp_path, monkeypatch)
    root = tmp_path / "labels"
    receipt = store.capture_structure_labels(root, **kwargs)
    document, labels = store.load_structure_labels(root)
    assert document == receipt and [row["status"] for row in labels] == ["completed", "pending_horizon"]
    assert labels[0]["accounting_audit"]["error_count"] == 0
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics[0]["expectancy"]["campaigns"] == 1
    assert diagnostics[0]["expectancy"]["status"] == "insufficient_visible_campaigns"
    original = (root / "manifest.json").read_bytes()
    assert store.capture_structure_labels(root, **kwargs) == receipt
    assert (root / "manifest.json").read_bytes() == original
    assert not list(tmp_path.glob(".structure-labels-*"))
    (root / "execution_prices.pkl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        store.load_structure_labels(root)


def test_previous_completed_label_keeps_original_availability_and_pending_prefix_cannot_change(tmp_path, monkeypatch):
    kwargs = sources(tmp_path, monkeypatch, count=25)
    day = str(kwargs["sessions"][-2].date())
    previous = {**kwargs, "observation_date": day, "captured_at": f"{day}T17:00:00+08:00",
                "sessions": kwargs["sessions"][:-1]}
    first = tmp_path / "first"
    store.capture_structure_labels(first, **previous)
    _, labels = store.load_structure_labels(first)
    kwargs["previous_labels"] = labels
    kwargs["source_receipts"]["previous_label_manifest"] = first / "manifest.json"
    second = tmp_path / "second"
    receipt = store.capture_structure_labels(second, **kwargs)
    _, latest = store.load_structure_labels(second)
    assert receipt["retained_original_label_count"] == 1
    assert latest[0] == labels[0] and latest[0]["available_at"] != kwargs["captured_at"]
    kwargs["raw_frames"]["sz.000001"].loc[4, "close"] = 10.1
    with pytest.raises(ValueError, match="published paper history changed"):
        store.capture_structure_labels(tmp_path / "revised", **kwargs)
    assert not (tmp_path / "revised/manifest.json").exists()


def test_backfilled_or_unfinished_labels_refuse_publication(tmp_path, monkeypatch):
    kwargs = sources(tmp_path, monkeypatch)
    kwargs["captured_at"] = f"{kwargs['observation_date']}T16:00:00+08:00"
    with pytest.raises(ValueError, match="real same-day"):
        store.capture_structure_labels(tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad/manifest.json").exists()


def test_barrier_report_verifies_archive_and_rejects_wrong_manifest_binding(tmp_path, monkeypatch):
    from stock_research.backtesting.inputs import file_sha256
    from stock_research.reporting.structure_execution_barriers import build_barrier_report

    kwargs = sources(tmp_path, monkeypatch)
    kwargs["raw_frames"]["sz.000001"].loc[1, "low"] = 8.5
    kwargs["raw_frames"]["sz.000001"].loc[1, "down_limit"] = 8.
    root = tmp_path / "labels"
    store.capture_structure_labels(root, **kwargs)
    manifest = root / "manifest.json"
    report = build_barrier_report(manifest, manifest_sha256=file_sha256(manifest))
    assert report["rows"][0]["entry_day_observation"] == "confirmed_entry_day_breach"
    assert [row["horizon"] for row in report["by_horizon"]] == [20, 60]
    wrong = root / "contract.json"
    with pytest.raises(ValueError, match="exact source manifest"):
        build_barrier_report(wrong, manifest_sha256=file_sha256(wrong))
    (root / "execution_prices.pkl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        build_barrier_report(manifest, manifest_sha256=file_sha256(manifest))
