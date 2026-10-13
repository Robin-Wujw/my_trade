import pandas as pd
import pytest

from stock_research.research.quantsplaybook_signal_event_research import (
    label_future_path,
    prior_snapshot_date,
)
from stock_research.reporting.signal_text import block_reason_text


def test_prior_snapshot_date_is_strictly_before_signal_date():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]

    assert prior_snapshot_date("2024-01-03", dates) == pd.Timestamp("2024-01-02")
    assert prior_snapshot_date(
        "2024-01-03", dates, "2024-01-03",
    ) == pd.Timestamp("2024-01-03")


def test_future_path_uses_raw_prices_after_signal_date():
    prices = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=21, freq="D"),
        "raw_close": [10.0] + [10.0 + index * 0.1 for index in range(1, 21)],
        "raw_high": [10.1] + [10.2 + index * 0.1 for index in range(1, 21)],
        "raw_low": [9.9] + [9.8 + index * 0.1 for index in range(1, 21)],
    })

    result = label_future_path(
        prices,
        "2024-01-02",
        trigger_price=10.0,
        stop_price=9.85,
        target_price=11.5,
    )

    assert result["次日收盘仍站上触发价"] is True
    assert result["未来5日收益"] == pytest.approx(0.05)
    assert result["未来20日收益"] == pytest.approx(0.20)
    assert result["20日内触及初始止损"] is False
    assert result["20日内触及计划目标"] is True
    assert result["信号日低点保护止损"] == pytest.approx(9.8505)
    assert result["低点止损与3R目标20日结果"] == "先达到3R目标"
    assert result["次日收盘确认成交状态"] == "确认"
    assert result["次日确认原始收盘价"] == pytest.approx(10.1)
    assert result["确认后未来5日收益"] == pytest.approx(0.5 / 10.1)


def test_successful_entry_gate_is_displayed_in_chinese():
    assert block_reason_text("ok") == "通过"
