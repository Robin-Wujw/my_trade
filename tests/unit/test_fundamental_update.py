import pandas as pd
import pytest

from stock_research.pipelines import fundamental_update
from stock_research.pipelines.fundamental_update import value_market_cap_eligible_count


def test_value_market_cap_count_handles_empty_snapshot():
    assert value_market_cap_eligible_count(pd.DataFrame()) == 0


def test_value_market_cap_count_filters_value_rows_above_threshold():
    snapshot = pd.DataFrame({
        "method": ["VALUE", "VALUE", "GROWTH"],
        "mktcap": [100, 99, 500],
    })

    assert value_market_cap_eligible_count(snapshot) == 1


def test_unlimited_rollover_backfill_reuses_cache_and_retries_only_failures(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(fundamental_update, "VALUE_CACHE_DIR", str(tmp_path / "values"))
    monkeypatch.setattr(fundamental_update, "STATE_DIR", str(tmp_path / "state"))
    codes = [f"sz.{number:06d}" for number in range(1, 108)]
    universe = pd.DataFrame({"code": codes})
    markets = {code: {"close": 10.0} for code in codes}
    args = fundamental_update.parse_args([
        "--report-period", "2026-06-30", "--as-of-date", "2026-10-14",
        "--max-updates", "0",
    ])
    cached_path = fundamental_update.financial_path(codes[0], args.report_period)
    fundamental_update.save_json(cached_path, {"value_line": 20.0})
    calls = []

    def fetch(code, close, report_period, retries):
        calls.append(code)
        if code == codes[-1]:
            return code, False, "SSL request failed"
        fundamental_update.save_json(
            fundamental_update.financial_path(code, report_period), {"value_line": 15.0},
        )
        return code, True, ""

    monkeypatch.setattr(fundamental_update, "fetch_one", fetch)
    assert fundamental_update.update_missing(universe, markets, args) == (105, 1, 106)
    assert codes[0] not in calls
    assert fundamental_update.load_json(cached_path) == {"value_line": 20.0}
    calls.clear()
    assert fundamental_update.update_missing(universe, markets, args) == (0, 1, 1)
    assert calls == [codes[-1]]
    state = fundamental_update.load_json(str(tmp_path / "state/financial_20260630.json"))
    assert state["errors"] == {codes[-1]: "SSL request failed"}
    assert codes[-1] not in state["completed"]


def test_offline_backfill_does_not_report_network_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(fundamental_update, "VALUE_CACHE_DIR", str(tmp_path / "values"))
    monkeypatch.setattr(fundamental_update, "STATE_DIR", str(tmp_path / "state"))
    args = fundamental_update.parse_args(["--report-period", "2026-06-30", "--offline"])
    assert fundamental_update.update_missing(
        pd.DataFrame({"code": ["sz.000001"]}), {"sz.000001": {"close": 10}}, args,
    ) == (0, 0, 0)


def test_fundamental_update_can_require_target_financial_coverage(monkeypatch, tmp_path):
    universe = pd.DataFrame({
        "code": ["sh.600000", "sz.000001"],
        "code_name": ["A", "B"],
    })
    markets = {
        "sh.600000": {
            "market_date": "2026-07-14",
            "close": 10.0,
            "liquidity_score": 80.0,
            "avg_amount20": 1_000_000_000,
        },
        "sz.000001": {
            "market_date": "2026-07-14",
            "close": 20.0,
            "liquidity_score": 80.0,
            "avg_amount20": 1_000_000_000,
        },
    }
    snapshot = pd.DataFrame({
        "code": ["sh.600000"],
        "method": ["VALUE"],
        "mktcap": [100.0],
        "industry_known": [True],
    })

    monkeypatch.setattr(fundamental_update, "refresh_universe", lambda offline=False: (universe, "test"))
    monkeypatch.setattr(fundamental_update, "latest_market", lambda code, as_of_date: markets[code])
    monkeypatch.setattr(fundamental_update, "update_missing", lambda universe, markets, args: (0, 0, 0))
    monkeypatch.setattr(fundamental_update, "load_industry_map", lambda as_of_date, offline=False: ({}, "test"))
    monkeypatch.setattr(fundamental_update, "build_snapshot", lambda universe, markets, report_period, industry_map: snapshot)
    monkeypatch.setattr(fundamental_update, "send_pushplus", lambda *args, **kwargs: None)
    monkeypatch.setattr(fundamental_update, "SNAPSHOT_DIR", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        fundamental_update.main([
            "--report-period", "2026-03-31",
            "--as-of-date", "2026-07-14",
            "--offline",
            "--min-price-coverage", "0.90",
            "--min-financial-coverage", "0.35",
            "--target-financial-coverage", "0.95",
            "--require-target-financial-coverage",
            "--output", str(tmp_path / "snapshot.csv"),
        ])

    assert exc.value.code == 3
