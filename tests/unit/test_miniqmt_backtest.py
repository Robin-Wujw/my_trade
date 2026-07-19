import pandas as pd

from apps.miniqmt_backtest import miniqmt_lookback_summary
from stock_research.strategies.miniqmt_backtest import run_miniqmt_backtest


def breakout_bars():
    dates = pd.bdate_range("2026-01-01", periods=80)
    closes = [10.0] * 79 + [10.4]
    return pd.DataFrame({
        "date": dates,
        "open": [10.0] * 80,
        "high": [10.0] * 79 + [10.4],
        "low": [10.0] * 80,
        "close": closes,
        "volume": [1000] * 79 + [3000],
    })


def test_miniqmt_backtest_wraps_existing_portfolio_replay_with_profile_metadata():
    bars = breakout_bars()
    date = bars.iloc[-1]["date"].strftime("%Y-%m-%d")

    result = run_miniqmt_backtest(
        {"sh.600699": bars},
        {date: [{"code": "sh.600699"}]},
        {date: {"phase": "active", "window_up_streak": 5}},
        requested_start=date,
        end_date=date,
        initial_capital=33_000,
    )

    assert result["execution_profile"] == "miniqmt"
    assert result["broker_connector"]["name"] == "miniqmt"
    assert result["broker_connector"]["live_trading_enabled"] is False
    assert result["broker_connector"]["mode"] == "read_only_execution_profile"
    assert result["commission_rate"] == 0.000085
    assert result["minimum_commission"] == 5.0
    assert result["sell_stamp_duty_rate"] == 0.0005
    assert result["estimated_slippage_rate"] == 0.0005
    assert result["signals_effective_next_day"] is True


def test_miniqmt_lookback_summary_distinguishes_stale_cache_from_new_listing():
    stale = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=5)})
    new_listing = pd.DataFrame({"date": pd.bdate_range("2024-11-01", periods=40)})
    just_before_start = pd.DataFrame({"date": pd.bdate_range("2024-09-20", periods=12)})
    snapshots = {
        "2024-09-24": [{"code": "sh.600000"}],
        "2024-12-02": [{"code": "sh.688001"}],
        "2024-10-08": [{"code": "sz.001301"}],
    }

    summary = miniqmt_lookback_summary(
        {"sh.600000": stale, "sh.688001": new_listing, "sz.001301": just_before_start},
        snapshots,
        requested_start="2024-09-24",
        min_prior_bars=60,
    )

    assert summary["insufficient_count"] == 1
    assert summary["insufficient_sample"][0]["code"] == "sh.600000"
    assert summary["new_listing_limited_count"] == 2
    assert {item["code"] for item in summary["new_listing_limited_sample"]} == {"sh.688001", "sz.001301"}
