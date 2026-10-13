import pandas as pd

from stock_research.research.quantsplaybook_signal_execution_research import (
    build_gate_summary,
    select_independent_events,
)


def test_independent_events_apply_trading_day_cooldown_and_daily_cap():
    dates = pd.bdate_range("2023-01-02", periods=25)
    rows = []
    for date in (dates[0], dates[1], dates[20]):
        for rank in range(1, 8):
            rows.append({
                "date": date,
                "code": f"A{rank}",
                "playbook_factor_rank": rank,
                "candidate_score": 100 - rank,
            })
    events = pd.DataFrame(rows)

    result = select_independent_events(events, dates)

    assert result.groupby("date").size().max() == 5
    assert result["date"].tolist().count(dates[1]) == 2
    assert result["date"].tolist().count(dates[20]) == 5


def test_gate_summary_requires_positive_result_in_each_year():
    dates = pd.bdate_range("2021-01-01", "2023-02-28")
    rows = []
    for year, value in ((2021, 0.02), (2022, 0.01), (2023, 0.03)):
        rows.append({
            "date": pd.Timestamp(year=year, month=1, day=5),
            "code": f"A{year}",
            "执行变体": "固定8%保护止损",
            "execution_label_available": True,
            "net_return": value,
            "review_reason_cn": "止损距离过窄",
            "signal_cn": "整理平台突破",
            "return_60d": -0.1,
            "volatility_20": 0.03,
            "playbook_factor_rank": 5,
            "candidate_score": 99,
        })
    result = build_gate_summary(pd.DataFrame(rows), dates)
    selected = result[
        result["质量门"].eq("止损距离过窄的整理平台突破")
    ].iloc[0]

    assert bool(selected["逐年平均净收益均为正"])
    assert selected["最少年度独立事件数"] == 1
    assert selected["2022年平均净收益"] == 0.01
