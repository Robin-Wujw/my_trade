import json

import pandas as pd
import pytest

from stock_research.research import frozen_forward, structure_event_store as store, structure_events


def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen_forward, "capture_clock", lambda _: "2026-10-15T10:00:00Z")
    prices = {code: pd.DataFrame({"date": pd.bdate_range(end="2026-10-15", periods=65),
              "open": 10., "high": 10.1, "low": 9.9, "close": 10., "volume": 1000.,
              "raw_to_qfq_factor": 1.}) for code in ("sz.000001", "sz.000002")}
    members = [{"code": code, "date": "2026-10-15", "membership_as_of": "2026-10-15T09:00:00Z",
                "is_listed": True, "is_traded": True, "is_st": False,
                "history_days": 65, "adv20_yuan": 30_000_000} for code in prices]
    upstream = tmp_path / "sync.json"
    upstream.write_text('{"status":"completed"}', encoding="utf-8")
    return dict(price_frames=prices, eligible_members=members, observation_date="2026-10-15",
                captured_at="2026-10-15T17:00:00+08:00", source_receipts={"sync": upstream},
                factor_ranks={"sz.000001": 1, "sz.000002": 500})


def test_immutable_full_capture_clips_future_before_adjustment_and_records_non_top50(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, monkeypatch)
    def signal(frame, index):
        assert frame.date.max() == pd.Timestamp("2026-10-15")
        assert frame.iloc[-1].close == 10. and "ma60" in frame
        return {"signal_type": "test", "order_type": "close", "stop": 9., "target_price": 13.}
    monkeypatch.setattr(structure_events, "_right_signal", signal)
    frame = kwargs["price_frames"]["sz.000002"]
    future = frame.iloc[[-1]].assign(date=pd.Timestamp("2026-10-16"), close=999., raw_to_qfq_factor=999.)
    kwargs["price_frames"]["sz.000002"] = pd.concat([frame, future], ignore_index=True)
    root = tmp_path / "capture"
    receipt = store.capture_structure_session(root, **kwargs)
    loaded, events = store.load_structure_session(root)
    assert loaded == receipt and receipt["member_count"] == receipt["event_count"] == 2
    assert events[1]["factor_rank"] == 500 and events[1]["planned_r"] == 3.
    archive = (root / "manifest.json").read_bytes()
    assert store.capture_structure_session(root, **kwargs) == receipt
    assert (root / "manifest.json").read_bytes() == archive
    assert not list(tmp_path.glob(".structure-capture-*"))
    kwargs["source_receipts"]["sync"].write_text('{"status":"revised"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable"):
        store.capture_structure_session(root, **kwargs)
    assert (root / "manifest.json").read_bytes() == archive


def test_zero_event_day_is_retained_without_fabricating_outcomes_and_tampering_fails(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(structure_events, "_right_signal", lambda *args: None)
    root = tmp_path / "capture"
    receipt = store.capture_structure_session(root, **kwargs)
    assert receipt["event_count"] == 0 and receipt["is_trading_day"] is True
    assert receipt["outcome_status"].startswith("pending")
    assert store.load_structure_session(root)[1] == []
    (root / "events.json").write_text(json.dumps([{"net_r": 999.}]), encoding="utf-8")
    with pytest.raises(ValueError, match="archive changed"):
        store.load_structure_session(root)


@pytest.mark.parametrize("change", ["backfill", "before_close", "no_factor", "duplicate_bar", "missing_member", "missing_old_bar", "unknown_date"])
def test_incomplete_or_backfilled_capture_cannot_enter_forward_store(tmp_path, monkeypatch, change):
    kwargs = inputs(tmp_path, monkeypatch)
    if change == "backfill":
        kwargs["captured_at"] = "2026-10-14T17:00:00+08:00"
    elif change == "before_close":
        kwargs["captured_at"] = "2026-10-15T15:00:00+08:00"
    elif change == "no_factor":
        kwargs["price_frames"]["sz.000001"] = kwargs["price_frames"]["sz.000001"].drop(columns="raw_to_qfq_factor")
    elif change == "duplicate_bar":
        frame = kwargs["price_frames"]["sz.000001"]
        kwargs["price_frames"]["sz.000001"] = pd.concat([frame, frame.iloc[[-1]]])
    elif change == "missing_member":
        kwargs["price_frames"].pop("sz.000002")
    elif change == "missing_old_bar":
        kwargs["price_frames"]["sz.000001"].loc[0, "high"] = float("nan")
    else:
        kwargs["price_frames"]["sz.000001"].loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        store.capture_structure_session(tmp_path / "capture", **kwargs)
    assert not (tmp_path / "capture/manifest.json").exists()
