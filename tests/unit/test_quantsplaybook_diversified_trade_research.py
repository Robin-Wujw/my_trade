import pandas as pd

from stock_research.research.quantsplaybook_diversified_trade_research import (
    _bucket_columns,
    attribute_right_batch_cashflows,
    build_quality_gate_table,
)


def test_batch_cashflow_attribution_handles_fifo_split_and_open_mark():
    events = pd.DataFrame([
        {
            "date": pd.Timestamp("2023-01-03"),
            "code": "A",
            "name": "甲",
            "trade_side": "买入",
            "action": "右侧买入",
            "reason": "R1; 首仓",
            "execution_quantity": 100,
            "cash_change_amount": -1000,
            "signal_type": "consolidation_breakout",
            "signal_type_cn": "整理平台突破",
            "phase": "active",
            "entry_risk_pct": 3,
            "planned_reward_r_multiple": 6,
            "candidate_snapshot_date": "2023-01-02",
        },
        {
            "date": pd.Timestamp("2023-01-04"),
            "code": "A",
            "trade_side": "状态",
            "action": "除权持仓调整",
            "reason": "送转",
            "execution_quantity": 0,
            "corporate_action_quantity_before": 100,
            "corporate_action_quantity_after": 200,
        },
        {
            "date": pd.Timestamp("2023-01-05"),
            "code": "A",
            "trade_side": "状态",
            "action": "现金分红到账",
            "reason": "分红",
            "execution_quantity": 0,
            "corporate_action_cash_amount": 20,
        },
        {
            "date": pd.Timestamp("2023-01-06"),
            "code": "A",
            "trade_side": "卖出",
            "action": "统一分仓收盘止盈1份",
            "reason": "止盈条件=测试",
            "execution_quantity": -100,
            "cash_change_amount": 600,
        },
    ])
    final_positions = [{
        "code": "A",
        "unrealized_pnl_amount": 100,
        "market_value": 600,
        "batches": [{"batch": "R1", "quantity": 100}],
    }]

    result, diagnostics = attribute_right_batch_cashflows(
        events, final_positions,
    )

    assert result.loc[0, "remaining_quantity"] == 100
    assert result.loc[0, "cash_distribution_amount"] == 20
    assert result.loc[0, "net_pnl_amount"] == 220
    assert result.loc[0, "is_open"]
    assert diagnostics["unattributed_sell_count"] == 0


def test_quality_gate_table_requires_each_training_year_to_be_positive():
    rows = []
    for year, pnl in ((2021, 100), (2022, 50), (2023, 25)):
        rows.append({
            "entry_year": year,
            "code": f"A{year}",
            "net_pnl_amount": pnl,
            "signal_type_cn": "整理平台突破",
            "market_phase": "active",
            "entry_risk_pct": 3,
            "planned_reward_r_multiple": 6,
            "playbook_factor_rank": 5,
        })
        rows.append({
            "entry_year": year,
            "code": f"B{year}",
            "net_pnl_amount": -10,
            "signal_type_cn": "龙头枢轴突破",
            "market_phase": "exited",
            "entry_risk_pct": 6,
            "planned_reward_r_multiple": 20,
            "playbook_factor_rank": 30,
        })
    frame = _bucket_columns(pd.DataFrame(rows))

    result = build_quality_gate_table(frame)
    platform = result[result["质量门"].eq("仅整理平台突破")].iloc[0]

    assert bool(platform["年度全为正"])
    assert platform["最少年度批次数"] == 1
    assert platform["2021年净盈亏"] == 100
