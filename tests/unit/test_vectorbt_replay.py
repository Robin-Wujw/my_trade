import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from stock_research.strategies.vectorbt_replay import run_vectorbt_cross_check


def _price_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "open": [10.0, 11.0],
        "high": [10.0, 11.0],
        "low": [10.0, 11.0],
        "close": [10.0, 11.0],
        "volume": [1_000_000, 1_000_000],
    })


def test_vectorbt_replay_matches_simple_round_trip_and_minimum_commission():
    result = {
        "events": [
            {
                "date": "2026-01-05", "code": "A", "price": 10.0,
                "execution_quantity": 100.0,
            },
            {
                "date": "2026-01-06", "code": "A", "price": 11.0,
                "execution_quantity": -100.0,
            },
        ],
        "equity_curve": [
            {"date": "2026-01-05", "equity": 0.99945},
            {"date": "2026-01-06", "equity": 1.00884},
        ],
        "final_return_pct": 0.884,
        "maximum_drawdown_pct": 0.0,
        "final_positions": [],
    }

    checked = run_vectorbt_cross_check(
        {"A": _price_frame()}, result,
        commission_rate=0.000085,
        minimum_commission=5.0,
        initial_capital=10_000.0,
        sell_stamp_duty_rate=0.0005,
        estimated_slippage_rate=0.0005,
    )

    assert checked["requested_order_count"] == 2
    assert checked["filled_order_count"] == 2
    assert checked["partial_fill_count"] == 0
    assert checked["rejected_order_count"] == 0
    assert checked["total_fees"] == pytest.approx(11.6)
    assert checked["final_return_pct"] == pytest.approx(0.884)
    assert checked["final_return_delta_pct"] == pytest.approx(0.0)
    assert checked["max_abs_daily_equity_delta_pct"] == pytest.approx(0.0)
    assert checked["final_cash"] == pytest.approx(10_088.4)
    assert checked["final_positions"] == []


def test_vectorbt_replay_exposes_fee_cash_shortfall_at_full_exposure():
    result = {
        "events": [{
            "date": "2026-01-05", "code": "A", "price": 10.0,
            "execution_quantity": 1_000.0,
        }],
        "equity_curve": [{"date": "2026-01-05", "equity": 0.999}],
        "final_return_pct": -0.1,
        "maximum_drawdown_pct": 0.0,
    }

    checked = run_vectorbt_cross_check(
        {"A": _price_frame()}, result,
        commission_rate=0.000085,
        minimum_commission=5.0,
        initial_capital=10_000.0,
        sell_stamp_duty_rate=0.0005,
        estimated_slippage_rate=0.0005,
    )

    assert checked["filled_order_count"] == 1
    assert checked["partial_fill_count"] == 1
    assert checked["max_abs_daily_equity_delta_pct"] > 0


def test_vectorbt_replay_compares_final_position_direct_quantity():
    result = {
        "events": [{
            "date": "2026-01-05", "code": "A", "price": 10.0,
            "execution_quantity": 100.0,
        }],
        "equity_curve": [
            {"date": "2026-01-05", "equity": 0.99945},
            {"date": "2026-01-06", "equity": 1.00945},
        ],
        "final_return_pct": 0.945,
        "maximum_drawdown_pct": 0.0,
        "final_positions": [{"code": "A", "quantity": 100.0}],
    }

    checked = run_vectorbt_cross_check(
        {"A": _price_frame()}, result,
        commission_rate=0.000085,
        minimum_commission=5.0,
        initial_capital=10_000.0,
        sell_stamp_duty_rate=0.0005,
        estimated_slippage_rate=0.0005,
    )

    position = checked["final_positions"][0]
    assert position["original_model_quantity"] == pytest.approx(100.0)
    assert position["quantity_delta"] == pytest.approx(0.0)
