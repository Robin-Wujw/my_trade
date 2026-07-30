from stock_research.reporting.backtest_trade_report import render_trade_report_markdown


def test_trade_report_renders_profit_concentration_diagnostics():
    result = {
        "requested_start": "2023-01-01",
        "actual_start": "2023-01-03",
        "end_date": "2026-07-21",
        "initial_capital": 1_000_000,
        "final_cash": 800_000,
        "final_return_pct": 64.082,
        "maximum_drawdown_pct": -13.902,
        "trade_summary": {"buy_count": 2, "sell_count": 2},
        "trade_ledger": [
            {
                "date": "2023-01-03",
                "code": "sz.300308",
                "name": "Alpha",
                "trade_side": "买入",
                "quantity": 100,
                "execution_quantity": 100,
                "execution_price": 100,
                "trade_amount": 10000,
                "transaction_cost_amount": 0,
                "cash_change_amount": -10000,
                "cost_basis": 100,
                "initial_risk_per_share_raw": 5,
                "initial_risk_amount": 500,
                "entry_reward_risk": 5,
            },
            {
                "date": "2023-01-10",
                "code": "sz.300308",
                "name": "Alpha",
                "trade_side": "卖出",
                "quantity": 100,
                "execution_quantity": -100,
                "execution_price": 90,
                "trade_amount": 9000,
                "transaction_cost_amount": 0,
                "cash_change_amount": 9000,
                "cost_basis": 100,
                "profit_loss_amount": -1000,
                "profit_loss_pct": -10,
                "initial_risk_per_share_raw": 5,
                "initial_risk_amount": 500,
                "entry_reward_risk": 5,
                "realized_r_multiple": -2,
            },
        ],
        "final_positions": [],
        "r_multiple_summary": {
            "sell_count": 1,
            "winning_sell_count": 0,
            "losing_sell_count": 1,
            "sell_win_rate_pct": 0,
            "average_entry_reward_risk": 5,
            "median_entry_reward_risk": 5,
            "average_realized_r": -2,
            "median_realized_r": -2,
            "average_loss_r": -2,
            "payoff_r": None,
            "profit_factor_r": 0,
            "loss_beyond_one_r_count": 1,
            "loss_beyond_one_r_pct": 100,
            "average_realized_to_planned_rr_pct": -40,
            "exit_reason_r_top": [
                {
                    "exit_reason": "hard_space_stop",
                    "sell_count": 1,
                    "average_realized_r": -2,
                    "median_realized_r": -2,
                    "net_realized_r": -2,
                    "loss_beyond_one_r_count": 1,
                },
            ],
        },
        "profit_concentration_summary": {
            "symbol_count": 3,
            "positive_symbol_count": 2,
            "negative_symbol_count": 1,
            "top1_positive_profit_share_pct": 70.0,
            "top3_positive_profit_share_pct": 100.0,
            "top1_return_contribution_pct": 33.735,
            "top3_return_contribution_pct": 56.0,
            "exclude_top1_approx_final_return_pct": 30.347,
            "exclude_top3_approx_final_return_pct": 8.082,
            "concentration_warning": True,
            "top1_symbol": {
                "code": "sz.300308",
                "name": "Alpha",
                "total_pnl_amount": 337349.47,
                "positive_profit_share_pct": 70.0,
            },
            "top_symbols": [
                {
                    "code": "sz.300308",
                    "name": "Alpha",
                    "industry_tags": ["AI"],
                    "realized_pnl_amount": 236274.75,
                    "unrealized_pnl_amount": 101074.72,
                    "total_pnl_amount": 337349.47,
                    "positive_profit_share_pct": 70.0,
                },
            ],
            "industry_contribution_top": [
                {
                    "industry_tag": "AI",
                    "total_pnl_amount": 337349.47,
                    "total_return_contribution_pct": 33.735,
                    "net_profit_share_pct": 90.0,
                },
            ],
        },
    }

    markdown = render_trade_report_markdown(result)

    assert "Top1" in markdown
    assert "sz.300308" in markdown
    assert "33.73%" in markdown
    assert "30.35%" in markdown
    assert "R 倍数审计" in markdown
    assert "实现-2.00R" in markdown
    assert "hard_space_stop" in markdown
