import json

import pandas as pd
import pytest

from stock_research.research import frozen_forward as forward


def test_capture_clock_refuses_backfills_unfinished_days_and_ambiguous_time():
    assert forward.capture_clock("2026-10-14", "2026-10-14T20:30:00+08:00") == "2026-10-14T12:30:00+00:00"
    for day, now in [("2026-10-13", "2026-10-14T20:30:00+08:00"),
                     ("2026-10-14", "2026-10-14T16:29:00+08:00"),
                     ("2026-10-15", "2026-10-14T20:30:00+08:00"),
                     ("2026-10-14", "2026-10-14T20:30:00")]:
        with pytest.raises(ValueError):
            forward.capture_clock(day, now)


def test_sync_requires_same_day_capture_and_financials(tmp_path):
    state = {"status": "completed", "latest_session": "2026-10-14", "database": str(tmp_path / "db"),
             "financial_capture_requested": True, "completed": {
                 f"{dataset}:2026-10-14": {"captured_at": "2026-10-14T10:00:00Z"}
                 for dataset in (*forward.DAILY_DATASETS, "dividend")}}
    forward.validate_sync_receipt(state, tmp_path / "db", "2026-10-14", "2026-10-14T12:00:00Z")
    state["completed"]["daily_kline:2026-10-14"]["captured_at"] = "2026-10-15T00:00:00Z"
    with pytest.raises(ValueError, match="same-day"):
        forward.validate_sync_receipt(state, tmp_path / "db", "2026-10-14", "2026-10-14T12:00:00Z")
    state["financial_capture_requested"] = False
    with pytest.raises(ValueError, match="financial"):
        forward.validate_sync_receipt(state, tmp_path / "db", "2026-10-14", "2026-10-14T12:00:00Z")


def test_paper_intent_is_conditional_and_never_invents_fill_or_quantity():
    result = {"entry_blocks": [
        {"date": "2026-10-14", "code": "sh.600000", "signal_type": "breakout",
         "reason": "standard_right_next_open_date_unavailable", "signal_trigger_raw_price": 10,
         "signal_stop_raw_price": 9, "entry_target_raw_price": 13},
        {"date": "2026-10-14", "code": "sh.600001", "reason": "liquidity_failed"},
        {"date": "2026-10-13", "code": "sh.600002", "reason": "standard_right_next_open_date_unavailable"},
    ]}
    orders = forward.paper_orders(result, "2026-10-14")
    assert len(orders) == 1
    assert orders[0]["quantity"] is None and orders[0]["filled"] is False
    assert orders[0]["planned_price"] == 10
    assert orders[0]["status"] == "conditional_next_open_recheck"


def test_left_snapshot_hash_and_capture_time_are_required(tmp_path):
    path = tmp_path / "candidates.csv"
    pd.DataFrame([{"code": "sh.600000", "candidate_source": "value_model"}]).to_csv(path, index=False)
    item = {"date": "2026-10-14", "file": path.name, "sha256": forward.sha256(path),
            "captured_at": "2026-10-14T10:00:00Z"}
    (tmp_path / "manifest.json").write_text(json.dumps({"snapshots": [item]}), encoding="utf-8")
    rows, _ = forward.load_left_snapshot(tmp_path, "2026-10-14", "2026-10-14T12:00:00Z")
    assert rows[0]["candidate_source"] == "value_model"
    path.write_text("code,candidate_source\nsh.600001,value_model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        forward.load_left_snapshot(tmp_path, "2026-10-14", "2026-10-14T12:00:00Z")


def test_replay_rejects_future_execution_data_before_engine(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("engine must not run with future inputs")
    monkeypatch.setattr(forward, "run_portfolio_backtest", fail)
    with pytest.raises(ValueError, match="future execution"):
        forward.replay_paper({"A": pd.DataFrame({"date": ["2026-10-15"]})},
                             {"2026-10-14": []}, {}, "2026-10-14",
                             corporate_actions={}, security_end_dates={})


def test_publication_lock_is_not_stolen(tmp_path):
    (tmp_path / ".capture.lock").write_text("123", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already running"):
        forward.run_capture(output_root=tmp_path)
    assert (tmp_path / ".capture.lock").read_text() == "123"


def test_replay_prefix_detects_revisions_but_allows_new_sessions():
    previous = {"equity_curve": [{"date": "2026-10-14", "equity": 100.0}],
                "events": [], "trade_ledger": []}
    current = {"equity_curve": [*previous["equity_curve"], {"date": "2026-10-15", "equity": 101}],
               "events": [], "trade_ledger": []}
    forward.verify_replay_prefix(previous, current, "2026-10-14")
    current["equity_curve"] = [{"date": "2026-10-14", "equity": 99}]
    with pytest.raises(ValueError, match="history changed"):
        forward.verify_replay_prefix(previous, current, "2026-10-14")


def test_pending_exit_projection_preserves_state():
    from copy import deepcopy
    from stock_research.backtesting.paper_actions import pending_exit_intents
    from stock_research.backtesting.state import PositionState
    states = {"A": PositionState(pending_lot_exits={"R1": "T+1; stop"},
                                 pending_profit_ids={"maximum_profit_half"})}
    before = deepcopy(states)
    intents = pending_exit_intents(states, "2026-10-14", left_exits={"B": "value_falsified"})
    assert states == before
    assert len(intents) == 3
    assert all(row["side"] == "sell" and row["quantity"] is None and not row["filled"] for row in intents)


def test_right_ranks_before_liquidity_and_ignores_future_ipo_and_bars(monkeypatch):
    days = pd.bdate_range(end="2026-10-14", periods=85)
    prices = pd.DataFrame([
        {"date": day, "code": code, "open": 10, "high": 11, "low": 9, "close": 10,
         "volume": 10000, "amount": 50000 if code != "A" else 1000,
         "raw_to_qfq_factor": 1, "tradestatus": "1", "turnover_rate": 1}
        for day in [*days, pd.Timestamp("2026-10-15")] for code in ["A", "B", "C", "FUTURE"]])
    master = pd.DataFrame([
        {"code": code, "list_date": "2026-10-15" if code == "FUTURE" else "2020-01-01", "delist_date": None}
        for code in ["A", "B", "C", "FUTURE"]])
    captured = {}

    def fake_factors(frame, **kwargs):
        assert frame["date"].max() == pd.Timestamp("2026-10-14")
        assert "FUTURE" not in set(frame["code"])
        assert kwargs["include_low_market_residual_volatility"] is True
        index = pd.MultiIndex.from_frame(frame[["date", "code"]])
        return pd.DataFrame({"f1": [3 if code == "A" else 2 if code == "B" else 1 for _, code in index]}, index=index)

    def combine(panel, spec):
        captured["rank_universe"] = set(panel.index.get_level_values("code"))
        return panel["f1"].groupby(level="date").rank(pct=True)

    monkeypatch.setattr(forward, "calculate_daily_factor_panel", fake_factors)
    monkeypatch.setattr(forward, "_apply_combination", combine)
    monkeypatch.setattr(forward, "load_baseline", lambda: {"model_id": "frozen", "right_top_n": 2})
    rows = forward.build_right_snapshot(prices, master, pd.DataFrame(), "2026-10-14")
    assert captured["rank_universe"] == {"A", "B", "C"}
    assert [row["code"] for row in rows] == ["B", "C"]
    assert rows[0]["playbook_factor_score"] == pytest.approx(2 / 3)
