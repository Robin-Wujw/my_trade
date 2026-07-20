import pandas as pd
import pytest

from scripts.independent_qvr_backtest import (
    compute_qvr_scores,
    run_qvr_backtest,
    select_monthly_qvr_snapshots,
)


def _row(code, **overrides):
    row = {
        "date": "2026-01-30",
        "code": code,
        "name": code,
        "quality_score": 80,
        "earnings_yoy": 0.2,
        "mktcap": 150,
        "avg_amount_20": 400_000_000,
        "price_to_value": 1.0,
        "return_20d": 0.1,
        "return_60d": 0.2,
        "drawdown_60": -0.05,
        "volatility_20": 0.03,
        "downside_volatility_60": 0.02,
        "range_21_pct": 0.12,
        "known_volume_ratio": 1.0,
        "right_acceleration": 0.02,
        "momentum_60_ex5": 0.12,
        "momentum_120_ex20": 0.18,
        "data_status": "traded",
        "valid_price_bar": True,
        "is_traded_bar": True,
        "candidate_score": 999,
    }
    row.update(overrides)
    return row


def _bars(code, closes):
    dates = pd.bdate_range("2026-01-29", periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [price * 1.01 for price in closes],
        "low": [price * 0.99 for price in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
        "amount": [100_000_000] * len(closes),
    })


def test_qvr_score_uses_independent_factors_not_candidate_score():
    scored = compute_qvr_scores([
        _row("A", candidate_score=1, quality_score=95, price_to_value=0.7, volatility_20=0.015),
        _row("B", candidate_score=9999, quality_score=60, price_to_value=1.8, volatility_20=0.08),
    ])
    selected = {
        row["code"]: row for row in scored
        if row.get("qvr_selected_universe")
    }
    assert selected["A"]["qvr_rank"] == 1
    assert selected["A"]["qvr_score"] > selected["B"]["qvr_score"]


def test_monthly_snapshots_use_last_available_observation_day():
    result = select_monthly_qvr_snapshots({
        "2026-01-29": [_row("A")],
        "2026-01-30": [_row("B", quality_score=90)],
        "2026-02-27": [_row("C")],
    })
    assert list(result) == ["2026-01-30", "2026-02-27"]
    assert result["2026-01-30"][0]["code"] == "B"


def test_qvr_backtest_buys_after_snapshot_and_sells_stop_next_open():
    snapshots = select_monthly_qvr_snapshots({
        "2026-01-30": [_row("A", volatility_20=0.01)],
    })
    prices = {
        "A": _bars("A", [100, 100, 100, 100, 86, 85, 84]),
    }
    result = run_qvr_backtest(
        prices,
        snapshots,
        start_date="2026-01-29",
        end_date="2026-02-06",
        commission_rate=0,
        minimum_commission=0,
        sell_stamp_duty_rate=0,
        estimated_slippage_rate=0,
    )
    buys = [event for event in result["events"] if event["action"] == "buy"]
    sells = [event for event in result["events"] if event["action"] == "sell"]
    assert buys[0]["date"] == "2026-02-02"
    assert sells[0]["date"] == "2026-02-05"
    assert sells[0]["reason"] == "initial_stop_next_open"
    assert result["summary"]["strategy"] == "independent_qvr_monthly"


def test_qvr_backtest_rejects_empty_price_calendar():
    with pytest.raises(ValueError, match="no price dates"):
        run_qvr_backtest({}, {}, start_date="2026-01-01", end_date="2026-01-02")
