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
