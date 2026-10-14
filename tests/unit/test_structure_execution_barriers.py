import copy

import pytest

from stock_research.reporting.structure_execution_barriers import execution_barrier_diagnostics


SESSIONS = ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21"]
SOURCE = "c" * 64


def label(*, phase="open", sell_date="2026-10-21", status="completed", blocks=None):
    events = [{"date": "2026-10-16", "code": "sz.000001", "execution_quantity": 1000.,
               "execution_price": 10., "execution_phase": phase}]
    if sell_date is not None:
        events.append({"date": sell_date, "code": "sz.000001", "execution_quantity": -1000.,
                       "execution_price": 8., "execution_phase": "open"})
    return {"event_id": "one", "horizon": 20, "label_source_sha256": SOURCE, "status": status,
            "initial_risk_amount": 1000., "net_r": -2.1 if status == "completed" else None,
            "execution": {"events": events, "equity_curve": [{"date": day} for day in SESSIONS]},
            "blocked_exits": blocks if blocks is not None else [
                {"date": "2026-10-19", "status": "locked_limit_down"},
                {"date": "2026-10-20", "status": "suspended_or_no_trade"}]}


def bars(low=8., close=9.5, *, source=SOURCE):
    return {(source, "sz.000001", "2026-10-16"): {"low": low, "close": close}}


def test_monday_to_wednesday_delay_uses_sessions_and_keeps_causes():
    report = execution_barrier_diagnostics([label()], SESSIONS, entry_bars=bars())
    row, horizon = report["rows"][0], report["by_horizon"][0]
    assert row["entry_day_observation"] == "confirmed_entry_day_breach"
    assert row["episodes"] == [{"first_block_date": "2026-10-19", "next_sell_date": "2026-10-21",
        "status": "resolved", "blocked_attempts": 2,
        "reasons": {"locked_limit_down": 1, "suspended_or_no_trade": 1},
        "delay_sessions": 2, "observed_age_sessions": None}]
    assert horizon["blocked_attempts"] == 2 and horizon["blocked_label_count"] == 1
    assert horizon["loss_tails"][1] == {"threshold_r": 1.5, "count": 1, "fraction": 1.}


def test_unresolved_barrier_has_unknown_final_delay_and_no_zero_r():
    result = execution_barrier_diagnostics([label(sell_date=None, status="pending_exit_or_settlement")], SESSIONS)
    episode = result["rows"][0]["episodes"][0]
    assert episode["delay_sessions"] is None and episode["observed_age_sessions"] == 2
    assert result["rows"][0]["net_r"] is None
    assert result["by_horizon"][0]["settled_net_r_count"] == 0
    assert result["by_horizon"][0]["loss_tails"][0]["fraction"] is None


def test_intraday_low_before_or_after_buy_is_ambiguous():
    result = execution_barrier_diagnostics([label(phase="intraday")], SESSIONS, entry_bars=bars())
    assert result["rows"][0]["entry_day_observation"] == "ambiguous_intraday_order"
    result = execution_barrier_diagnostics([label(phase="intraday")], SESSIONS, entry_bars=bars(close=8.5))
    assert result["rows"][0]["entry_day_observation"] == "confirmed_entry_day_close_breach"


def test_exact_stop_touch_is_not_reported_as_confirmed_fill_or_block():
    result = execution_barrier_diagnostics([label(blocks=[])], SESSIONS, entry_bars=bars(low=9.))
    assert result["rows"][0]["entry_day_observation"] == "no_entry_day_breach"
    assert result["by_horizon"][0]["blocked_attempts"] == 0


def test_source_revision_cannot_supply_retained_label_entry_observation():
    result = execution_barrier_diagnostics([label()], SESSIONS, entry_bars=bars(source="a" * 64))
    assert result["rows"][0]["entry_day_observation"] == "unavailable_entry_bar"


def test_old_fill_only_or_data_limited_labels_do_not_imply_zero_attempts():
    old = {"event_id": "unknown", "horizon": 20, "status": "data_limited"}
    report = execution_barrier_diagnostics([old], SESSIONS)
    assert report["by_horizon"][0]["blocked_attempts"] is None
    assert report["by_horizon"][0]["unavailable_attempt_labels"] == 1
    assert report["by_horizon"][0]["pending_episode_count"] is None
    assert report["by_horizon"][0]["resolved_episode_count"] is None
    assert report["status"] == "unavailable_attempt_ledger"


def test_two_horizons_remain_separate_and_inputs_are_not_mutated():
    values = [label(), {**label(), "horizon": 60}]
    before = copy.deepcopy(values)
    report = execution_barrier_diagnostics(values, SESSIONS)
    assert values == before
    assert [row["horizon"] for row in report["by_horizon"]] == [20, 60]
    assert all(row["blocked_attempts"] == 2 for row in report["by_horizon"])


def test_different_sales_split_barrier_episodes_instead_of_using_final_exit():
    value = label()
    value["execution"]["events"].insert(1, {"date": "2026-10-19", "code": "sz.000001",
        "execution_quantity": -500., "execution_price": 9., "execution_phase": "intraday"})
    report = execution_barrier_diagnostics([value], SESSIONS)
    assert [row["delay_sessions"] for row in report["rows"][0]["episodes"]] == [0, 1]


@pytest.mark.parametrize("change", ["calendar_gap", "duplicate_attempt", "same_day_sell", "same_day_block", "duplicate_label"])
def test_unknown_session_or_invalid_t_plus_one_cannot_be_counted(change):
    value, sessions, values = label(), list(SESSIONS), None
    if change == "calendar_gap":
        sessions.remove("2026-10-19")
    elif change == "duplicate_attempt":
        value["blocked_exits"].append(dict(value["blocked_exits"][-1]))
    elif change == "same_day_sell":
        value["execution"]["events"][-1]["date"] = "2026-10-16"
    elif change == "same_day_block":
        value["blocked_exits"][0]["date"] = "2026-10-16"
    else:
        values = [value, copy.deepcopy(value)]
    with pytest.raises(ValueError):
        execution_barrier_diagnostics(values or [value], sessions)


def test_empty_real_sample_set_is_explicitly_unavailable():
    report = execution_barrier_diagnostics([], SESSIONS)
    assert report["status"] == "unavailable_no_execution_labels" and report["rows"] == []
