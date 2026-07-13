from stock_research.reporting.backtest_trade_report import (
    build_readable_trade_frame,
    render_trade_report_markdown,
)


def test_readable_trade_report_tracks_holdings_and_reasons():
    result = {
        "requested_start": "2026-01-01",
        "actual_start": "2026-01-05",
        "end_date": "2026-01-06",
        "initial_capital": 1_000_000,
        "final_cash": 999_000,
        "final_return_pct": 1.5,
        "maximum_drawdown_pct": -2.0,
        "trade_summary": {"buy_count": 1, "sell_count": 1},
        "trade_ledger": [
            {
                "date": "2026-01-05", "code": "sh.600000", "name": "浦发银行",
                "trade_side": "买入", "quantity": 100, "execution_price": 10,
                "trade_amount": 1000, "transaction_cost_amount": 5,
                "profit_loss_amount": -5, "cash_change_amount": -1005,
                "reason": "R1; 回调波段50%向上突破; intraday_breakout",
            },
            {
                "date": "2026-01-06", "code": "sh.600000", "name": "浦发银行",
                "trade_side": "卖出", "quantity": 100, "execution_price": 11,
                "trade_amount": 1100, "transaction_cost_amount": 6,
                "profit_loss_amount": 89, "profit_loss_pct": 8.9,
                "cash_change_amount": 1094, "reason": "condition stop; 14:55/close proxy",
            },
        ],
        "final_positions": [],
    }

    frame = build_readable_trade_frame(result["trade_ledger"])
    assert frame["交易后持股(股)"].tolist() == [100, 0]
    assert frame.iloc[0]["操作摘要"].startswith("买入浦发银行(sh.600000) 100股")
    assert frame.iloc[1]["交易结果"] == "盈利89.00元 (+8.90%)"
    assert frame.iloc[1]["持仓状态"] == "清仓"
    assert frame.iloc[0]["买卖理由"] == "回调波段50%向上突破；盘中突破成交"
    markdown = render_trade_report_markdown(result)
    assert "每次买卖流水" in markdown
    assert "买入浦发银行(sh.600000) 100股" in markdown
    assert "期末空仓" in markdown
