import json

import pandas as pd
import pytest

from stock_research.backtesting.baseline import BASELINE_PATH, load_baseline
from stock_research.pipelines import daily_forward
from stock_research.reporting import forward as report
from stock_research.research import forward_left
from stock_research.research.frozen_forward import sha256, write_json


DAY = "2026-10-14"
CAPTURED = "2026-10-14T10:00:00+00:00"
NOW = "2026-10-14T20:30:00+08:00"


def artifacts(root):
    day_root = root / DAY
    day_root.mkdir(parents=True)
    candidates = day_root / "candidates.csv"
    pd.DataFrame([{"code": "sh.600000", "date": DAY, "observation_date": DAY,
                   "captured_at": CAPTURED, "candidate_source": "quantsplaybook_factor",
                   "rank": 1, "score": .9, "research_only": True}]).to_csv(candidates, index=False)
    write_json(day_root / "paper_orders.json", {"orders": [{"code": "sh.600000", "side": "buy",
               "planned_price": 10, "stop_price": 9, "target_price": 13, "filled": False,
               "triggered": False, "observation_date": DAY, "research_only": True,
               "quantity": None}], "paper_only": True, "observation_date": DAY, "captured_at": CAPTURED})
    write_json(day_root / "paper_replay.json", {"paper_only": True, "research_only": True})
    entry = {"date": DAY, "captured_at": CAPTURED, "file": f"{DAY}/candidates.csv",
             "sha256": sha256(candidates), "orders_sha256": sha256(day_root / "paper_orders.json"),
             "paper_replay_sha256": sha256(day_root / "paper_replay.json"), "candidate_count": 1}
    path = root / "manifest.json"
    write_json(path, {"version": "frozen-forward-v1", "snapshots": [entry], "research_only": True,
                     "strict_point_in_time": False, "model_id": load_baseline()["model_id"],
                     "configuration_sha256": sha256(BASELINE_PATH)})
    return path


def test_local_report_verifies_all_artifacts_and_shows_conditional_orders(tmp_path):
    path = artifacts(tmp_path)
    result = report.load_forward_report(path, DAY, now=NOW)
    html = report.render_forward_report(result)
    assert "sh.600000" in html and "未触发，未成交" in html and "研究用途" in html
    (tmp_path / DAY / "paper_replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        report.load_forward_report(path, DAY, now=NOW)


def test_local_report_binds_structure_receipt_before_immutable_reuse(monkeypatch, tmp_path):
    from stock_research.research import structure_event_store
    path = artifacts(tmp_path)
    structure = tmp_path / DAY / "structure_events" / "manifest.json"
    structure.parent.mkdir()
    document = {"observation_date": DAY, "captured_at": CAPTURED, "member_count": 100, "event_count": 0}
    write_json(structure, document)
    manifest = json.loads(path.read_text())
    manifest["snapshots"][0].update(structure_event_manifest=structure.relative_to(tmp_path).as_posix(),
        structure_event_manifest_sha256=sha256(structure), structure_event_receipt=document)
    write_json(path, manifest)
    calls = []
    monkeypatch.setattr(structure_event_store, "load_structure_session", lambda directory: (calls.append(directory) or document, []))
    report.load_forward_report(path, DAY, now=NOW)
    assert calls == [structure.parent]
    structure.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="structure manifest path/hash"):
        report.load_forward_report(path, DAY, now=NOW)


@pytest.mark.parametrize("now", ["2026-10-15T20:30:00+08:00", "2026-10-14T16:00:00+08:00"])
def test_forward_report_never_uses_stale_or_future_captures(tmp_path, now):
    path = artifacts(tmp_path)
    with pytest.raises(ValueError, match="stale, future"):
        report.load_forward_report(path, DAY, now=now)


def test_paper_order_cannot_be_reported_as_filled_even_with_rehashed_file(tmp_path):
    path = artifacts(tmp_path)
    orders = tmp_path / DAY / "paper_orders.json"
    payload = json.loads(orders.read_text())
    payload["orders"][0]["filled"] = True
    write_json(orders, payload)
    manifest = json.loads(path.read_text())
    manifest["snapshots"][0]["orders_sha256"] = sha256(orders)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="cannot claim"):
        report.load_forward_report(path, DAY, now=NOW)


def test_financial_visibility_uses_later_final_announcement():
    rows = pd.DataFrame([
        {"ts_code": "A", "end_date": "20260630", "ann_date": "20260801", "f_ann_date": "20260802", "bps": 5},
        {"ts_code": "A", "end_date": "20260630", "ann_date": "20261001", "f_ann_date": "20261015", "bps": 99},
    ])
    visible = forward_left.visible_financial_rows(rows, DAY)
    assert visible.loc["A", "bps"] == 5


def test_industry_requires_real_coverage_and_rejects_conflicting_memberships():
    rows = pd.DataFrame([{"ts_code": "A", "l2_name": "白酒", "in_date": "20200101", "out_date": ""}])
    assert forward_left.industry_mapping(rows, DAY, {"A"}) == {"A": "白酒"}
    with pytest.raises(ValueError, match="coverage incomplete"):
        forward_left.industry_mapping(rows, DAY, {"A", "B"})
    with pytest.raises(ValueError, match="conflicting"):
        forward_left.industry_mapping(pd.concat([rows, rows.assign(l2_name="银行")]), DAY, {"A"})
    assert forward_left.industry_mapping(rows.assign(l3_name="半导体材料"), DAY, {"A"}) == {"A": "半导体材料"}


def left_inputs():
    dates = pd.bdate_range(end=DAY, periods=80)
    prices = pd.DataFrame({"date": dates, "code": "sh.600000", "open": 15., "high": 16.,
                           "low": 14., "close": 15., "volume": 10000., "amount": 15000.,
                           "total_mv": 2000000., "total_share": 10000., "raw_to_qfq_factor": 1., "tradestatus": "1"})
    basic = pd.DataFrame([{"code": "sh.600000", "name": "样例", "list_date": "2000-01-01", "delist_date": None}])
    financial = forward_left.visible_financial_rows(pd.DataFrame([{
        "ts_code": "600000.SH", "ann_date": "20260820", "end_date": "20260630",
        "bps": 3., "dt_eps": .2, "dt_netprofit_yoy": 30., "roe_dt": 18.,
    }]), DAY)
    return prices, basic, financial, financial, {"600000.SH": "半导体"}


def annual_sources():
    annual = forward_left.visible_financial_rows(pd.DataFrame([{
        "ts_code": "600000.SH", "ann_date": "20260420", "end_date": "20251231",
        "dt_eps": 1., "profit_dedt": 100_000_000., "total_share": 100_000_000.,
    }]), DAY)
    latest = forward_left.visible_financial_rows(pd.DataFrame([{
        "ts_code": "600000.SH", "ann_date": "20260820", "end_date": "20260630",
        "total_share": 100_000_000., "total_hldr_eqy_exc_min_int": 300_000_000.,
    }]), DAY)
    factors = forward_left.comparable_share_factors(pd.DataFrame(columns=forward_left.DIVIDEND_FIELDS),
                                                   "2025-12-31", "2026-06-30", DAY)
    return {"annual_financial": annual, "share_adjustments": {"600000.SH": factors},
            "annual_capital": annual, "latest_capital": latest,
            "annual_source_sha256": "a" * 64, "eps_adjustment_source_sha256": "b" * 64}


def test_value_capture_keeps_observation_only_and_actual_pit_provenance():
    inputs = left_inputs()
    rows, assessed = forward_left.build_value_rows(*inputs, DAY, CAPTURED, **annual_sources())
    assert assessed == 1 and len(rows) == 1
    row = rows[0]
    assert row["candidate_source"] == "value_model" and row["allow_left"] and not row["allow_right"]
    assert row["close"] == 15 and row["value_line"] == 16
    assert row["announcement_date"] == "2026-08-20"
    assert row["annual_eps_excl_comparable"] == 1. and row["annual_report_period"] == "2025-12-31"
    assert row["annual_announcement_date"] == "2026-04-20" and row["book_value_per_share"] == 3.
    assert row["strict_point_in_time"] is False and row["research_only"] is True
    # Future prices do not move the current absolute value or signal score.
    future = inputs[0].tail(1).assign(date=pd.Timestamp("2026-10-15"), close=999)
    changed, _ = forward_left.build_value_rows(pd.concat([inputs[0], future]), *inputs[1:], DAY, CAPTURED, **annual_sources())
    assert changed == rows
    excluded, count = forward_left.build_value_rows(*inputs, DAY, CAPTURED, st_codes={"sh.600000"}, **annual_sources())
    assert excluded == [] and count == 1


def test_missing_financial_or_industry_data_cannot_be_empty_left_success():
    prices, basic, financial, income, industries = left_inputs()
    with pytest.raises(ValueError, match="financial coverage incomplete"):
        forward_left.build_value_rows(prices, basic, financial.iloc[:0], income, industries, DAY, CAPTURED)
    with pytest.raises(ValueError, match="industry coverage incomplete"):
        forward_left.build_value_rows(prices, basic, financial, income, {}, DAY, CAPTURED)


def test_annual_eps_and_share_basis_require_real_complete_sources():
    inputs = left_inputs()
    with pytest.raises(ValueError, match="annual financial coverage"):
        forward_left.build_value_rows(*inputs, DAY, CAPTURED)
    sources = annual_sources()
    sources["annual_financial"] = sources["annual_financial"].assign(dt_eps=float("nan"), eps=99.)
    with pytest.raises(ValueError, match="annual deducted EPS"):
        forward_left.build_value_rows(*inputs, DAY, CAPTURED, **sources)
    sources = annual_sources()
    sources["annual_financial"] = sources["annual_financial"].assign(profit_dedt=200_000_000.)
    with pytest.raises(ValueError, match="share basis cannot be verified"):
        forward_left.build_value_rows(*inputs, DAY, CAPTURED, **sources)
    sources = annual_sources()
    sources["latest_capital"] = sources["latest_capital"].assign(total_share=110_000_000., total_hldr_eqy_exc_min_int=330_000_000.)
    with pytest.raises(ValueError, match="not explained"):
        forward_left.build_value_rows(*inputs, DAY, CAPTURED, **sources)


def test_known_model_exclusions_do_not_require_unneeded_annual_sources():
    prices, basic, financial, income, industries = left_inputs()
    for changes in ({"st_codes": {"sh.600000"}}, {"industries": {"600000.SH": "银行"}},
                    {"prices": prices.assign(total_mv=100000.)}):
        args = {"prices": prices, "basic": basic, "financial": financial.iloc[:0], "income": income.iloc[:0],
                "industries": industries, "observation_date": DAY, "captured_at": CAPTURED, **changes}
        assert forward_left.build_value_rows(**args) == ([], 1)


def test_stock_distributions_rebase_annual_eps_and_bps_through_observation_day():
    source = {field: None for field in forward_left.DIVIDEND_FIELDS}
    rows = pd.DataFrame([
        {**source, "ts_code": "600000.SH", "ann_date": "20260420", "imp_ann_date": "20260501",
         "ex_date": "20260515", "div_proc": "实施", "stk_div": .5, "stk_bo_rate": .2, "stk_co_rate": .3},
        {**source, "ts_code": "600000.SH", "ann_date": "20260710", "imp_ann_date": "20260715",
         "ex_date": "20260720", "div_proc": "实施", "stk_div": .2, "stk_bo_rate": .2, "stk_co_rate": 0.},
        {**source, "ts_code": "600000.SH", "ann_date": "20261015", "ex_date": "20261020",
         "div_proc": "实施", "stk_div": 9., "stk_bo_rate": 9., "stk_co_rate": 0.},
    ])
    factors = forward_left.comparable_share_factors(rows, "2025-12-31", "2026-06-30", DAY)
    assert factors["annual_to_observation_factor"] == pytest.approx(1.8)
    assert factors["latest_to_observation_factor"] == pytest.approx(1.2)
    prices, *rest = left_inputs()
    prices = prices.assign(total_share=18000.)
    sources = annual_sources()
    sources["annual_financial"] = sources["annual_financial"].assign(dt_eps=2., profit_dedt=200_000_000.)
    sources["share_adjustments"] = {"600000.SH": factors}
    sources["latest_capital"] = sources["latest_capital"].assign(total_share=150_000_000., total_hldr_eqy_exc_min_int=450_000_000.)
    values, _ = forward_left.build_value_rows(prices, *rest, DAY, CAPTURED, **sources)
    assert len(values) == 1
    assert values[0]["annual_eps_excl_comparable"] == pytest.approx(2 / 1.8)
    assert values[0]["book_value_per_share"] == pytest.approx(3 / 1.2)
    assert values[0]["value_line"] == pytest.approx(3 / 1.2 + 2 / 1.8 * 1.3 * 10)
    with pytest.raises(ValueError, match="conflicting"):
        forward_left.comparable_share_factors(rows.assign(stk_div=.1), "2025-12-31", "2026-06-30", DAY)


def test_adjustment_fetch_failure_is_not_zero_distribution_and_success_is_reusable(monkeypatch, tmp_path):
    monkeypatch.setattr(forward_left, "capture_clock", lambda day: CAPTURED)
    def failed(*args, **kwargs):
        raise RuntimeError("source unavailable")
    with pytest.raises(RuntimeError, match="source unavailable"):
        forward_left.capture_share_adjustments(failed, {"600000.SH"}, tmp_path, "2025-12-31", "2026-06-30", DAY)
    assert not (tmp_path / "eps_adjustments.json").exists()
    def valid(*args, **kwargs):
        return pd.DataFrame(columns=forward_left.DIVIDEND_FIELDS)
    result, source = forward_left.capture_share_adjustments(valid, {"600000.SH"}, tmp_path, "2025-12-31", "2026-06-30", DAY)
    assert result["600000.SH"]["annual_to_observation_factor"] == 1
    assert result["600000.SH"]["receipt"]["pagination_complete"] is True
    assert forward_left.capture_share_adjustments(failed, {"600000.SH"}, tmp_path, "2025-12-31", "2026-06-30", DAY)[0] == result


def test_daily_forward_fails_before_network_when_date_is_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_forward, "run_sync", lambda **kw: pytest.fail("must not fetch stale day"))
    with pytest.raises(ValueError, match="today's completed session"):
        daily_forward.run(observation_date="2020-01-01", formula_path="missing", output_root=tmp_path)
    status = json.loads((tmp_path / "daily_status.json").read_text())
    assert status["status"] == "blocked" and status["paper_only"] is True
    assert not (tmp_path / "manifest.json").exists()


def test_failed_sync_never_creates_left_or_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_forward, "capture_clock", lambda day: CAPTURED)
    monkeypatch.setattr(daily_forward, "run_sync", lambda **kw: {"status": "failed"})
    monkeypatch.setattr(daily_forward, "capture_left", lambda **kw: pytest.fail("failed data cannot enter left capture"))
    with pytest.raises(ValueError, match="synchronization incomplete"):
        daily_forward.run(observation_date=DAY, formula_path="missing", output_root=tmp_path)
    assert json.loads((tmp_path / "daily_status.json").read_text())["status"] == "blocked"


def test_daily_forward_wires_sync_value_formula_and_frozen_capture(monkeypatch, tmp_path):
    source = tmp_path / "formula.csv"
    pd.DataFrame([{"date": DAY, "window_up_streak": 5, "window_down_streak": 0, "window_unique_count": 188}]).to_csv(source, index=False)
    output = tmp_path / "forward"
    calls = []
    monkeypatch.setattr(daily_forward, "capture_clock", lambda day: CAPTURED)
    def sync(**kwargs):
        calls.append("sync")
        assert kwargs["capture_financials"] is True and kwargs["end_date"] == DAY
        return {"status": "completed", "latest_session": DAY}
    def left(**kwargs):
        calls.append("left")
        return tmp_path / "left"
    def capture(**kwargs):
        calls.append("capture")
        formula = pd.read_csv(kwargs["formula_history"])
        assert formula.iloc[0]["phase"] == "active"
        assert formula.iloc[0]["window_up_streak"] == 5
        assert formula.iloc[0]["window_down_streak"] == 0
        assert formula.iloc[0]["window_unique_count"] == 188
        assert kwargs["left_directory"] == tmp_path / "left"
        assert kwargs["observation_date"] == DAY
        return {"candidate_count": 50}
    monkeypatch.setattr(daily_forward, "run_sync", sync)
    monkeypatch.setattr(daily_forward, "capture_left", left)
    monkeypatch.setattr(daily_forward, "run_capture", capture)
    monkeypatch.setattr(daily_forward, "load_forward_report", lambda *args: calls.append("verify"))
    result = daily_forward.run(observation_date=DAY, formula_path=source, output_root=output)
    assert result == output / "manifest.json"
    assert calls == ["sync", "left", "capture", "verify"]
    assert json.loads((output / "daily_status.json").read_text())["candidate_count"] == 50


def test_formula_history_preserves_observed_streaks_not_inferred_phase(monkeypatch, tmp_path):
    from stock_research.reporting import daily_report
    monkeypatch.setattr(daily_report, "load_formula_phase_state", lambda path: {})
    source = tmp_path / "source.csv"
    row = {"date": DAY, "window_up_streak": 0, "window_down_streak": 3, "window_unique_count": 137}
    pd.DataFrame([row]).to_csv(source, index=False)
    previous = {"date": "2026-10-13", "formula33": {"phase": "active"}}
    output = tmp_path / "history.csv"
    daily_forward.formula_history(source, DAY, output, previous)
    current = pd.read_csv(output).iloc[0]
    assert current["phase"] == "active" and current["window_down_streak"] == 3
    assert current["window_up_streak"] == 0 and current["window_unique_count"] == 137
    pd.DataFrame([{**row, "window_down_streak": None}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="nonnegative integer"):
        daily_forward.formula_history(source, DAY, output, previous)


def test_left_rejects_unknown_listing_date_and_cannot_enter_on_delisting_day():
    prices, basic, financial, income, industries = left_inputs()
    with pytest.raises(ValueError, match="listing date missing"):
        forward_left.build_value_rows(prices, basic.assign(list_date=pd.NaT), financial, income, industries, DAY, CAPTURED)
    with pytest.raises(ValueError, match="universe was not assessed"):
        forward_left.build_value_rows(prices, basic.assign(delist_date=DAY), financial, income, industries, DAY, CAPTURED)


def test_daily_lock_is_not_stolen(monkeypatch, tmp_path):
    lock = tmp_path / ".daily.lock"
    lock.write_text("existing-run", encoding="utf-8")
    monkeypatch.setattr(daily_forward, "_run", lambda **kw: pytest.fail("must not overwrite active source files"))
    with pytest.raises(RuntimeError, match="already running"):
        daily_forward.run(output_root=tmp_path)
    assert lock.read_text() == "existing-run"


def test_left_source_hashes_are_verified_before_reuse(tmp_path):
    from stock_research.research.frozen_forward import load_left_snapshot
    root = tmp_path
    pd.DataFrame(columns=["code", "candidate_source"]).to_csv(root / "left.csv", index=False)
    (root / "industry.csv").write_text("actual industry source", encoding="utf-8")
    (root / "financial.json").write_text("{}", encoding="utf-8")
    entry = {"date": DAY, "file": "left.csv", "sha256": sha256(root / "left.csv"),
             "captured_at": CAPTURED, "candidate_count": 0, "assessed_count": 500,
             "industry_file": "industry.csv", "industry_sha256": sha256(root / "industry.csv"),
             "financial_file": "financial.json", "financial_sha256": sha256(root / "financial.json")}
    write_json(root / "manifest.json", {"version": "forward-value-left-v1", "snapshots": [entry]})
    assert load_left_snapshot(root, DAY, NOW)[0] == []
    (root / "industry.csv").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="industry source path/hash"):
        load_left_snapshot(root, DAY, NOW)


def test_completed_same_day_is_reused_only_when_formula_and_artifacts_match(monkeypatch, tmp_path):
    manifest_path = artifacts(tmp_path)
    source_path = tmp_path / DAY / "formula_source.csv"
    source_path.write_text("actual input", encoding="utf-8")
    history_path = source_path.with_name("formula_history.csv")
    pd.DataFrame([{"date": DAY, "phase": "active", "formula_source_sha256": sha256(source_path)}]).to_csv(history_path, index=False)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(implementation={"test": True})
    manifest["snapshots"][0]["formula_sha256"] = sha256(history_path)
    write_json(manifest_path, manifest)
    monkeypatch.setattr(daily_forward, "capture_clock", lambda day: CAPTURED)
    monkeypatch.setattr(daily_forward, "implementation_fingerprint", lambda: {"test": True})
    monkeypatch.setattr(daily_forward, "load_forward_report", lambda path, day: report.load_forward_report(path, day, now=NOW))
    monkeypatch.setattr(daily_forward, "run_sync", lambda **kw: pytest.fail("immutable rerun must not fetch"))
    assert daily_forward.run(observation_date=DAY, formula_path=source_path, output_root=tmp_path) == manifest_path
    assert json.loads((tmp_path / "daily_status.json").read_text())["reused"] is True
    source_path.write_text("changed input", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        daily_forward.run(observation_date=DAY, formula_path=source_path, output_root=tmp_path)


def test_sync_to_left_persistence_uses_actual_database_receipts(monkeypatch, tmp_path):
    from stock_research.data import forward_sync
    from stock_research.research.frozen_forward import load_left_snapshot

    def provider(name, **params):
        if name == "trade_cal":
            return pd.DataFrame({"exchange": ["SSE"], "cal_date": ["20261014"], "is_open": [1]})
        if name == "stock_basic":
            columns = ["ts_code", "list_date", "list_status", "name"]
            return pd.DataFrame([["600000.SH", "20000101", "L", "样例"]] if params["list_status"] == "L" else [], columns=columns)
        if name == "stock_st":
            return pd.DataFrame(columns=["ts_code", "trade_date", "type"])
        if name == "suspend_d":
            return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"])
        if name == "dividend":
            return pd.DataFrame(columns=forward_sync.DIVIDEND_FIELDS)
        if name == "index_daily":
            return pd.DataFrame({"ts_code": [params["ts_code"]], "trade_date": ["20261014"], "close": [100.]})
        if name in forward_sync.FINANCIAL_FIELDS:
            period = params["period"]
            return pd.DataFrame([{"ts_code": "600000.SH", "end_date": period,
                "ann_date": "20260820" if period == "20260630" else "20260420",
                "bps": 3., "dt_eps": .2 if period == "20260630" else 1., "dt_netprofit_yoy": 30., "roe_dt": 18.,
                "profit_dedt": 100_000_000., "total_share": 100_000_000., "total_hldr_eqy_exc_min_int": 300_000_000.,
                "roe": 18., "total_assets": 100., "n_income": 10., "n_cashflow_act": 10.}])
        if name == "index_member_all":
            return pd.DataFrame([{"ts_code": "600000.SH", "l2_name": "半导体", "l3_name": "半导体材料",
                                  "in_date": "20200101", "out_date": ""}])
        return pd.DataFrame([{"ts_code": "600000.SH", "trade_date": "20261014", "open": 15., "high": 16.,
            "low": 14., "close": 15., "vol": 10000., "amount": 15000., "adj_factor": 1.,
            "up_limit": 16.5, "down_limit": 13.5, "turnover_rate": 1., "total_mv": 2000000., "total_share": 10000., "circ_mv": 1000000.}])

    monkeypatch.setattr(forward_sync, "utc_now", lambda: CAPTURED)
    monkeypatch.setattr(forward_left, "capture_clock", lambda day: CAPTURED)
    database = tmp_path / "data.sqlite3"
    sync_state = tmp_path / "sync.json"
    result = forward_sync.run_sync(database_path=database, state_path=sync_state,
        start_date=DAY, end_date=DAY, query=provider, now=NOW)
    assert result["status"] == "completed"
    root = forward_left.capture_left(database=database, sync_state=sync_state,
        output_root=tmp_path / "left", observation_date=DAY, query=provider)
    rows, _ = load_left_snapshot(root, DAY, NOW)
    assert len(rows) == 1 and rows[0]["industry"] == "半导体材料"
    assert rows[0]["annual_eps_excl_comparable"] == 1. and rows[0]["eps_excl"] == 1.
    assert rows[0]["annual_report_period"] == "2025-12-31"
    entry = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["snapshots"][0]
    financial = json.loads((root / entry["financial_file"]).read_text(encoding="utf-8"))
    assert financial["fina_indicator"][0]["ann_date"] == "20260820"
    assert rows[0]["financial_source_sha256"] == entry["financial_sha256"]
    assert rows[0]["annual_source_sha256"] == entry["annual_financial_sha256"]
    assert rows[0]["eps_adjustment_source_sha256"] == entry["eps_adjustment_sha256"]
    sidecar = json.loads((root / entry["r07_diagnostics_file"]).read_text(encoding="utf-8"))
    assert sidecar["production_candidates_changed"] is False
    assert sidecar["observations"][0]["value_line"] == 15.
    assert sidecar["observations"][0]["original_value_line"] == rows[0]["value_line"] == 16.
    assert entry["assessed_count"] == 1
    # A retry reuses the verified immutable left snapshot without querying again.
    assert forward_left.capture_left(database=database, sync_state=sync_state, output_root=root,
        observation_date=DAY, query=lambda *a, **kw: pytest.fail("must reuse capture")) == root
    (root / entry["annual_financial_file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="annual_financial source path/hash"):
        load_left_snapshot(root, DAY, NOW)


def test_r07_source_observations_precede_uncapped_price_to_value_filter():
    prices, *rest = left_inputs()
    sources = annual_sources()
    source_observations = []
    rows, assessed = forward_left.build_value_rows(prices.assign(close=12.5), *rest, DAY, CAPTURED,
        value_observations=source_observations, **sources)
    assert assessed == 1 and rows == []  # 12.5/16 < .80 is outside the production entry band.
    assert len(source_observations) == 1
    assert source_observations[0]["annual_eps_excl_comparable"] == 1.
