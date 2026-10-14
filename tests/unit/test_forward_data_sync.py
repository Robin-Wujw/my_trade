import json

import pandas as pd
import pytest

from stock_research.data.forward_sync import DAILY_DATASETS, DIVIDEND_FIELDS, completed_session, run_sync, verify_completed_receipts
from stock_research.storage import Database, TushareRepository


def _query_provider(calls, fail_dataset=None):
    def query(name, fields="", **params):
        calls.append((name, params))
        if name == fail_dataset:
            raise RuntimeError("Unauthorized provider response")
        if name == "trade_cal":
            dates = pd.date_range(params["start_date"], params["end_date"])
            return pd.DataFrame({"exchange": "SSE", "cal_date": dates.strftime("%Y%m%d"), "is_open": 1})
        if name == "stock_basic":
            if params["list_status"] == "L":
                return pd.DataFrame({"ts_code": ["000001.SZ"], "list_date": ["19910403"], "list_status": ["L"]})
            return pd.DataFrame(columns=["ts_code", "list_date", "list_status"])
        if name == "stock_st":
            return pd.DataFrame(columns=["ts_code", "trade_date", "type"])
        if name == "suspend_d":
            return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"])
        if name == "index_daily":
            dates = pd.date_range(params["start_date"], params["end_date"])
            return pd.DataFrame({"ts_code": params["ts_code"], "trade_date": dates.strftime("%Y%m%d"), "close": 100.0})
        if name == "dividend":
            return pd.DataFrame(columns=DIVIDEND_FIELDS)
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [params["trade_date"]],
                             "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.0],
                             "vol": [1000.], "amount": [10000.], "turnover_rate": [1.0],
                             "total_mv": [10000.], "circ_mv": [9000.],
                             "up_limit": [11.0], "down_limit": [9.0], "adj_factor": [1.0]})
    return query


def test_sync_captures_all_datasets_and_reuses_successful_daily_receipts(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-21", capture_financials=False)
    result = run_sync(**kwargs, query=_query_provider(calls))
    assert result["status"] == "completed"
    assert len(result["completed"]) == 2 * (len(DAILY_DATASETS) + 1) + 6
    assert result["completed"]["suspend_d:2026-07-21"]["rows"] == 0
    assert result["historical_backfill_is_forward_evidence"] is False
    assert result["research_only"] is True
    repository = TushareRepository(Database(kwargs["database_path"]))
    assert len(repository.load_dataset("daily_kline")) == 2
    calls.clear()
    assert run_sync(**kwargs, query=_query_provider(calls))["status"] == "completed"
    assert not any(name in ("daily", "adj_factor", "stock_st", "suspend_d") for name, _ in calls)


@pytest.mark.parametrize("dataset", ["basic", "stock_basic", "trade_cal"])
def test_master_and_calendar_changes_invalidate_forward_receipts(tmp_path, dataset):
    database = Database(tmp_path / "data.sqlite3")
    result = run_sync(database_path=database.path, state_path=tmp_path / "state.json",
        start_date="2026-07-21", end_date="2026-07-21", capture_financials=False,
        query=_query_provider([]))
    assert f"{dataset}:2026-07-21" in result["completed"]
    with database.connect() as connection:
        connection.execute("UPDATE raw.tushare_dataset_rows SET payload_json=json_set(payload_json, '$.name', 'changed') WHERE dataset=?", [dataset])
    with pytest.raises(ValueError, match="no longer match SQLite"):
        verify_completed_receipts(result, TushareRepository(database))


def test_failed_batch_is_retryable_and_completed_prices_survive(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-21", capture_financials=False)
    failed = run_sync(**kwargs, query=_query_provider(calls, fail_dataset="adj_factor"))
    assert failed["status"] == "failed"
    assert "daily_kline:2026-07-20" in failed["completed"]
    assert "adj_factor:2026-07-20" not in failed["completed"]
    assert "Unauthorized" in failed["errors"][0]["message"]
    calls.clear()
    # Default resumption must revisit an earlier incomplete day, even though
    # daily_kline for that date is already present.
    kwargs.pop("start_date")
    completed = run_sync(**kwargs, query=_query_provider(calls))
    assert completed["status"] == "completed"
    assert completed["requested_start"] == "2026-07-20"
    assert ("adj_factor", {"trade_date": "20260720"}) in calls
    assert not any(name == "daily" and params["trade_date"] == "20260720" for name, params in calls)


def test_empty_core_response_cannot_be_marked_as_no_trade(tmp_path):
    provider = _query_provider([])

    def query(name, **params):
        if name == "daily":
            return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"])
        return provider(name, **params)

    result = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                      start_date="2026-07-20", end_date="2026-07-20", query=query)
    assert result["status"] == "failed"
    assert "daily_kline:2026-07-20" not in result["completed"]
    assert "retryable" in result["errors"][0]["message"]


def test_request_budget_does_not_claim_completion(tmp_path):
    calls = []
    result = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                      start_date="2026-07-20", end_date="2026-07-20", max_requests=2,
                      query=_query_provider(calls))
    assert result["status"] == "failed"
    assert result["requests"] == len(calls) == 2
    assert "budget" in result["errors"][0]["message"]


@pytest.mark.parametrize("bad_dataset", ["stock_st", "suspend_d"])
def test_failed_sparse_request_never_becomes_empty_receipt(tmp_path, bad_dataset):
    result = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                      start_date="2026-07-20", end_date="2026-07-20",
                      query=_query_provider([], fail_dataset=bad_dataset))
    assert result["status"] == "failed"
    assert f"{bad_dataset}:2026-07-20" not in result["completed"]


def test_completed_session_uses_china_clock_and_trading_calendar():
    calendar = pd.DataFrame({"cal_date": ["20261009", "20261010", "20261011", "20261012"],
                             "is_open": [1, 0, 0, 1], "exchange": ["SSE"] * 4})
    assert completed_session(calendar, "2026-10-12T08:00:00Z") == "2026-10-09"
    assert completed_session(calendar, "2026-10-12T09:00:00Z") == "2026-10-12"


def test_financial_capture_uses_later_announcement_and_real_capture_time(tmp_path):
    before_capture = pd.Timestamp.now(tz="UTC").floor("s")
    calls = []
    provider = _query_provider(calls)

    def query(name, **params):
        if name in ("income", "balancesheet", "cashflow", "fina_indicator"):
            return pd.DataFrame({
                "ts_code": ["000001.SZ", "000002.SZ"], "end_date": [params["period"]] * 2,
                "ann_date": ["20260615"] * 2, "f_ann_date": ["20260616", "20260722"],
                "total_assets": [100.0, 100.0], "n_income": [10., 10.],
                "n_cashflow_act": [10., 10.], "roe": [2., 2.],
            })
        return provider(name, **params)

    database_path = tmp_path / "data.sqlite3"
    result = run_sync(database_path=database_path, state_path=tmp_path / "state.json",
                      start_date="2026-07-20", end_date="2026-07-21", query=query)
    assert result["status"] == "completed"
    with Database(database_path).connect(read_only=True) as connection:
        rows = connection.execute("SELECT ts_code, captured_at FROM raw.tushare_dataset_row_versions WHERE dataset='income'").fetchall()
    assert rows
    assert all(code == "000001.SZ" for code, _ in rows)
    assert all(pd.to_datetime(captured_at, utc=True) >= before_capture for _, captured_at in rows)


def test_receipt_checks_detect_deleted_rows_and_default_resume_repairs_earlier_dates(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-21", capture_financials=False)
    state = run_sync(**kwargs, query=_query_provider(calls))
    database = Database(kwargs["database_path"])
    assert verify_completed_receipts(state, TushareRepository(database))["status"] == "verified"
    with database.connect() as connection:
        connection.execute("DELETE FROM raw.tushare_dataset_rows WHERE dataset='daily_kline' AND trade_date='2026-07-20'")
    with pytest.raises(ValueError, match="no longer match"):
        verify_completed_receipts(state, TushareRepository(database))
    calls.clear()
    kwargs.pop("start_date")
    repaired = run_sync(**kwargs, query=_query_provider(calls))
    assert repaired["status"] == "completed"
    assert repaired["requested_start"] == "2026-07-20"
    assert "daily_kline:2026-07-20" in repaired["invalidated_receipts"]
    assert any(name == "daily" and params["trade_date"] == "20260720" for name, params in calls)


def test_database_replacement_epoch_invalidates_even_empty_sparse_receipts(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20", capture_financials=False)
    first = run_sync(**kwargs, query=_query_provider(calls))
    with Database(kwargs["database_path"]).connect() as connection:
        connection.execute("UPDATE ops.forward_sync_identity SET token='replacement' WHERE singleton=1")
    calls.clear()
    second = run_sync(**kwargs, query=_query_provider(calls))
    assert second["status"] == "completed"
    assert second["database_identity"] != first["database_identity"]
    assert any(name == "suspend_d" for name, _ in calls)


@pytest.mark.parametrize("dataset,field,value", [("daily", "close", None), ("adj_factor", "adj_factor", -1),
    ("daily_basic", "turnover_rate", float("inf")), ("stk_limit", "up_limit", 8)])
def test_invalid_required_market_values_never_receive_success(tmp_path, dataset, field, value):
    provider = _query_provider([])

    def query(name, **params):
        frame = provider(name, **params)
        if name == dataset:
            if value is None:
                frame = frame.drop(columns=[field])
            else:
                frame[field] = value
        return frame

    state = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                     start_date="2026-07-20", end_date="2026-07-20", capture_financials=False, query=query)
    assert state["status"] == "failed"
    key = "daily_kline" if dataset == "daily" else dataset
    assert f"{key}:2026-07-20" not in state["completed"]


def _expanded_provider(calls, *, stock_count=20, missing_daily=False, financial_count=None, suspend_timing=None):
    base = _query_provider(calls)

    def query(name, fields="", **params):
        if name in ("income", "balancesheet", "cashflow", "fina_indicator"):
            calls.append((name, params))
            count = stock_count if financial_count is None else financial_count
            return pd.DataFrame({"ts_code": [f"{i + 1:06d}.SZ" for i in range(count)],
                                 "end_date": params["period"], "ann_date": "20260701",
                                 "total_assets": 100., "n_income": 10., "n_cashflow_act": 8., "roe": 2.})
        frame = base(name, fields=fields, **params)
        if name == "suspend_d" and suspend_timing is not None:
            return pd.DataFrame({"ts_code": [f"{stock_count:06d}.SZ"], "trade_date": [params["trade_date"]],
                                 "suspend_type": ["S"], "suspend_timing": [suspend_timing]})
        if (name == "stock_basic" and params["list_status"] == "L") or name in ("daily", "adj_factor", "daily_basic", "stk_limit"):
            count = stock_count - int(name == "daily" and missing_daily)
            frame = pd.concat([frame] * count, ignore_index=True)
            frame["ts_code"] = [f"{i + 1:06d}.SZ" for i in range(count)]
        return frame

    return query


def test_95_percent_daily_coverage_is_data_limited_and_missing_codes_resume(tmp_path):
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20", capture_financials=False)
    state = run_sync(**kwargs, query=_expanded_provider([], missing_daily=True))
    assert state["status"] == "data_limited"
    assert state["incomplete"]["daily_kline:2026-07-20"]["missing_codes"] == ["000020.SZ"]
    assert state["incomplete"]["daily_kline:2026-07-20"]["coverage_ratio"] == .95
    assert "daily_kline:2026-07-20" not in state["completed"]
    kwargs.pop("start_date")
    state = run_sync(**kwargs, query=_expanded_provider([]))
    assert state["status"] == "completed"
    assert state["incomplete"] == {}


@pytest.mark.parametrize("timing,expected_status", [("全天停牌", "completed"), ("09:30-10:30", "data_limited"), (None, "data_limited")])
def test_only_explicit_full_day_suspension_explains_missing_bar(tmp_path, timing, expected_status):
    state = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                     start_date="2026-07-20", end_date="2026-07-20", capture_financials=False,
                     query=_expanded_provider([], missing_daily=True, suspend_timing=timing))
    assert state["status"] == expected_status


def test_explicit_unfinished_today_is_rejected_before_database_creation(tmp_path):
    with pytest.raises(ValueError, match="16:30"):
        run_sync(database_path=tmp_path / "absent.sqlite3", state_path=tmp_path / "state.json",
                 end_date="2026-07-20", now="2026-07-20T08:00:00Z", query=lambda *a, **k: pytest.fail("unexpected query"))
    assert not (tmp_path / "absent.sqlite3").exists()


def test_partial_financial_coverage_is_reported_and_retried(tmp_path):
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20")
    state = run_sync(**kwargs, query=_expanded_provider([], stock_count=2, financial_count=1))
    assert state["status"] == "data_limited"
    financial_gaps = [value for key, value in state["incomplete"].items() if key.startswith("income:")]
    assert financial_gaps and all(item["missing_codes"] == ["000002.SZ"] for item in financial_gaps)
    state = run_sync(**kwargs, query=_expanded_provider([], stock_count=2))
    assert state["status"] == "completed"


def test_financial_receipts_allow_one_request_per_resume_without_starvation(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20")
    assert run_sync(**kwargs, query=_expanded_provider(calls, stock_count=1), capture_financials=False)["status"] == "completed"
    calls.clear()
    for _ in range(8):
        state = run_sync(**kwargs, query=_expanded_provider(calls, stock_count=1), max_requests=5)
    assert state["status"] == "completed"
    financial_calls = [(name, params["period"]) for name, params in calls if name in ("income", "balancesheet", "cashflow", "fina_indicator")]
    assert len(financial_calls) == len(set(financial_calls)) == 8


def test_financial_page_checkpoint_resumes_at_next_offset(tmp_path):
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20")
    base = _expanded_provider(calls, stock_count=1)
    run_sync(**kwargs, query=base, capture_financials=False)

    def query(name, **params):
        frame = base(name, **params)
        if name == "income" and params["period"] == "20251231" and params["offset"] == 0:
            frame = pd.concat([frame] * 5000, ignore_index=True)
            frame["report_type"] = range(5000)
        return frame

    first = run_sync(**kwargs, query=query, max_requests=5)
    assert first["status"] == "failed"
    key = "income:2025-12-31:as_of:2026-07-20"
    assert first["financial_pagination"][key]["offset"] == 5000
    calls.clear()
    second = run_sync(**kwargs, query=query, max_requests=5)
    assert second["status"] == "failed"  # Next table still exceeds this deliberately tiny cap.
    assert key in second["completed"]
    assert any(name == "income" and params["offset"] == 5000 for name, params in calls)
    assert not any(name == "income" and params["offset"] == 0 for name, params in calls)


def test_repeated_financial_page_is_not_accepted_as_complete(tmp_path):
    base = _expanded_provider([], stock_count=1)

    def query(name, **params):
        frame = base(name, **params)
        if name == "income":
            frame = pd.concat([frame] * 5000, ignore_index=True)
            frame["report_type"] = range(5000)
        return frame

    state = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                     start_date="2026-07-20", end_date="2026-07-20", query=query)
    assert state["status"] == "failed"
    assert "repeated a page" in state["errors"][0]["message"]
    assert not any(key.startswith("income:") for key in state["completed"])


def test_identical_valid_receipts_do_not_suppress_next_day_financial_recapture(tmp_path, monkeypatch):
    from stock_research.data import forward_sync
    calls = []
    kwargs = dict(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                  start_date="2026-07-20", end_date="2026-07-20", query=_expanded_provider(calls, stock_count=1))
    monkeypatch.setattr(forward_sync, "utc_now", lambda: "2026-10-14T12:00:00+00:00")
    assert run_sync(**kwargs)["status"] == "completed"
    calls.clear()
    monkeypatch.setattr(forward_sync, "utc_now", lambda: "2026-10-15T12:00:00+00:00")
    assert run_sync(**kwargs)["status"] == "completed"
    assert sum(name in ("income", "balancesheet", "cashflow", "fina_indicator") for name, _ in calls) == 8


@pytest.mark.parametrize("pay_date,status", [("20260721", "completed"), (None, "data_limited")])
def test_dividend_requires_cash_payment_date_and_actual_capture_receipt(tmp_path, pay_date, status):
    base = _query_provider([])

    def query(name, **params):
        if name == "dividend" and params.get("ex_date") == "20260720":
            return pd.DataFrame([{**dict.fromkeys(DIVIDEND_FIELDS), "ts_code": "000001.SZ",
                                  "ann_date": "20260710", "imp_ann_date": "20260710", "end_date": "20251231",
                                  "record_date": "20260717", "ex_date": "20260720", "pay_date": pay_date,
                                  "div_proc": "实施", "stk_div": 0., "stk_bo_rate": 0., "stk_co_rate": 0.,
                                  "cash_div": .1, "cash_div_tax": .1}])
        return base(name, **params)

    state = run_sync(database_path=tmp_path / "data.sqlite3", state_path=tmp_path / "state.json",
                     start_date="2026-07-20", end_date="2026-07-20", capture_financials=False, query=query)
    assert state["status"] == status
    if status == "completed":
        receipt = state["completed"]["dividend:2026-07-20"]
        assert receipt["dividend_payment_dates_verified"] is True
        assert receipt["database_fingerprint"]["row_count"] == 1
    else:
        assert "dividend:2026-07-20" not in state["completed"]
        assert "cash_pay_date_missing_or_invalid" in state["incomplete"]["dividend:2026-07-20"]["issues"][0]["reasons"]
