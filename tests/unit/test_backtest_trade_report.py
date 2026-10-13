import pandas as pd

from stock_research.reporting.backtest_trade_report import (
    apply_stock_basic_display_names,
    build_readable_trade_frame,
    readable_reason,
    render_trade_report_markdown,
)
from stock_research.reporting.signal_text import (
    candidate_failure_reason_text,
    block_reason_text,
    localize_entry_block,
    localize_event,
    localize_signal_counts,
    profit_condition_text,
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
                "cost_basis": 10,
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
    assert frame.iloc[0]["交易结果"] == "成本价10.000元/股；费用5.00元"
    assert frame.iloc[1]["交易结果"] == "盈利89.00元 (+8.90%)"
    assert frame.iloc[1]["持仓状态"] == "清仓"
    assert frame.iloc[0]["买卖理由"] == "回调波段50%向上突破；盘中突破成交"
    markdown = render_trade_report_markdown(result)
    assert "每次买卖流水" in markdown
    assert "买入浦发银行(sh.600000) 100股" in markdown
    assert "成本价10.000元/股；费用5.00元" in markdown
    assert "期末空仓" in markdown


def test_trade_report_keeps_left_grid_prices_and_hides_internal_block_fields():
    result = {
        "requested_start": "2025-06-01",
        "actual_start": "2025-06-01",
        "end_date": "2025-07-31",
        "initial_capital": 1_000_000,
        "final_cash": 900_000,
        "final_return_pct": 10,
        "maximum_drawdown_pct": -5,
        "trade_summary": {"buy_count": 1, "sell_count": 1},
        "trade_ledger": [
            {
                "date": "2025-06-19", "code": "sz.300308", "name": "中际旭创",
                "action": "左侧网格卖出", "trade_side": "卖出", "quantity": 100,
                "execution_price": 124.052, "trade_amount": 12405.2,
                "transaction_cost_amount": 18.0, "profit_loss_amount": 1000,
                "profit_loss_pct": 8.5, "cash_change_amount": 12387.2,
                "reason": "L6; 上一格卖出; intraday_cross",
                "value_line": 124.052177, "grid_slot": 5, "account_mode": "left",
            },
        ],
        "final_positions": [
            {
                "code": "sz.300308", "name": "中际旭创", "quantity": 300,
                "cost": 123.79, "close": 1184.05, "left_value_line": 124.052177,
                "market_value": 355215, "unrealized_pnl_amount": 318044.43,
                "unrealized_pnl_pct": 855.6351, "position_mode": "left",
                "profit_tail": False, "capacity_counted": False,
            },
        ],
        "entry_blocks": [
            {
                "date": "2025-07-01", "code": "sz.300308", "name": "中际旭创",
                "reason": "order_not_filled", "signal_type": "bull_run_half_pullback",
                "entry_evidence_score": 10, "trade_basis_score": 12,
                "leadership_score": 24.094,
            },
        ],
    }

    markdown = render_trade_report_markdown(result)

    assert "上一格网格卖价；盘中触及预设网格价成交" in markdown
    assert "价值线124.052，网格槽5" in markdown
    assert "中际旭创（sz.300308） | 左侧" in markdown
    assert "order_not_filled" not in markdown
    assert "bull_run_half_pullback" not in markdown
    assert "期末持仓相关未成交信号" not in markdown
    assert "旧证据分" not in markdown


def test_trade_report_translates_profit_and_stop_conditions():
    assert readable_reason(
        "止盈条件=profit_floor,trailing_10,maximum_profit_half,"
        "volume_node_break:2025-05-14; "
        "14:55/close proxy",
    ) == (
        "止盈条件：利润保护线、从阶段高点回撤10%、最大浮盈回撤一半、"
        "跌破量价节点（节点日2025-05-14）；"
        "按当日收盘价近似成交（非真实盘中成交）"
    )
    assert readable_reason(
        "R73; hard space stop 3.125; intraday_stop",
    ) == "空间硬止损 3.125；盘中硬止损成交"
    assert readable_reason(
        "R10; entry time condition; 14:55/close proxy",
    ) == "入场时间条件退出；按当日收盘价近似成交（非真实盘中成交）"

    assert readable_reason(
        "R11; 固定持有20个交易日; 收盘代理",
    ) == (
        "持有满20个交易日，按因子周期规则退出；"
        "按当日收盘价近似成交（非真实盘中成交）"
    )


def test_display_names_and_missing_industry_are_attached_after_backtest_only():
    result = {
        "initial_capital": 1_000_000,
        "trade_ledger": [{"code": "sh.600000", "name": "sh.600000"}],
        "final_positions": [{"code": "sh.600000", "name": "sh.600000"}],
        "profit_concentration_summary": {
            "top1_symbol": {"code": "sh.600000", "name": "sh.600000"},
            "top_symbols": [
                {"code": "sh.600000", "name": "sh.600000", "total_pnl_amount": 1000},
            ],
        },
    }
    basic = pd.DataFrame([
        {"code": "sh.600000", "name": "浦发银行", "industry": "银行"},
    ])

    apply_stock_basic_display_names(result, basic)

    assert result["trade_ledger"][0]["name"] == "浦发银行"
    assert result["trade_ledger"][0]["display_industry_tags"] == ["银行"]
    assert result["final_positions"][0]["name"] == "浦发银行"
    assert result["final_positions"][0]["display_industry_tags"] == ["银行"]
    assert result["profit_concentration_summary"]["top1_symbol"]["name"] == "浦发银行"
    assert result["profit_concentration_summary"]["top1_symbol"]["display_industry_tags"] == ["银行"]
    assert result["profit_concentration_summary"]["top_symbols"][0]["display_industry_tags"] == ["银行"]
    assert result["display_name_source"]["decision_input"] is False
    assert result["display_name_source"]["resolved_traded_industry_count"] == 1
    markdown = render_trade_report_markdown(result)
    assert "| 浦发银行(sh.600000) | 银行 |" in markdown
    assert "未分类" not in markdown


def test_signal_text_localizes_raw_backtest_codes():
    event = localize_event({
        "action": "右侧买入",
        "signal_type": "consolidation_breakout",
        "entry_gate_reason": "risk_pct_too_tight",
        "semantic_entry_setup": "factor_only_high_rr_right_entry",
        "reason": "R1; close_confirmed",
    })
    assert event["signal_type_cn"] == "整理平台突破"
    assert event["entry_gate_reason_cn"] == "止损距离过窄"
    assert event["semantic_entry_setup_cn"] == "纯因子右侧高盈亏比买点"
    assert event["reason_cn"] == "收盘确认成交"

    block = localize_entry_block({
        "signal_type": "uptrend_support_pullback",
        "reason": "support_pullback_without_fast_ma_confluence",
        "candidate_failure_reason": "mktcap_below_100",
    })
    assert block["signal_type_cn"] == "上涨波段支撑拉回"
    assert block["reason_cn"] == "支撑拉回缺少快速均线共振"
    assert block["candidate_failure_reason_cn"] == "总市值低于100亿元"

    assert localize_signal_counts({"pullback_50_breakout": 2}) == {
        "回调50%放量突破": 2,
    }


def test_chinese_display_never_leaks_unknown_signal_codes_or_batch_prefixes():
    assert readable_reason("L2; intraday_cross") == "盘中触及结构位成交"
    assert readable_reason("R35(原L2); condition stop") == "收盘条件止损"
    assert localize_event({
        "signal_type": "future_signal_code",
        "entry_gate_reason": "future_block_code",
        "semantic_entry_setup": "future_setup_code",
        "reason": "future_execution_code",
    }) == {
        "signal_type": "future_signal_code",
        "entry_gate_reason": "future_block_code",
        "semantic_entry_setup": "future_setup_code",
        "reason": "future_execution_code",
        "signal_type_cn": "未识别信号",
        "entry_gate_reason_cn": "未识别阻断原因",
        "semantic_entry_setup_cn": "未识别买点",
        "reason_cn": "未识别策略条件",
    }
    assert block_reason_text("future_block_code:42") == "未识别阻断原因"
    assert candidate_failure_reason_text("future_failure_code:42") == (
        "未识别候选阻断原因"
    )
    assert profit_condition_text("future_profit_code") == "未识别止盈条件"
