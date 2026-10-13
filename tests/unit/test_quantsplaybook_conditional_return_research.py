import pandas as pd

from stock_research.research.quantsplaybook_conditional_return_research import (
    FROZEN_RULE_CN,
    frozen_rule_mask,
    select_frozen_rule,
    summarize_variants,
)


def _panel():
    rows = []
    for year in (2021, 2022, 2023):
        for day in range(1, 6):
            date = pd.Timestamp(year=year, month=1, day=day)
            rows.extend([
                {
                    "date": date,
                    "code": f"A{year}{day}",
                    "playbook_factor_rank": 3,
                    "structure_proximity_score": 50.0,
                    "forward_return": 0.02,
                    "label_available": True,
                    "研究阶段": "训练期2021-2023",
                },
                {
                    "date": date,
                    "code": f"B{year}{day}",
                    "playbook_factor_rank": 8,
                    "structure_proximity_score": 50.0,
                    "forward_return": 0.01,
                    "label_available": True,
                    "研究阶段": "训练期2021-2023",
                },
                {
                    "date": date,
                    "code": f"C{year}{day}",
                    "playbook_factor_rank": 15,
                    "structure_proximity_score": 20.0,
                    "forward_return": -0.03 if year == 2023 else 0.01,
                    "label_available": True,
                    "研究阶段": "训练期2021-2023",
                },
                {
                    "date": date,
                    "code": f"D{year}{day}",
                    "playbook_factor_rank": 10,
                    "structure_proximity_score": 20.0,
                    "forward_return": -0.10 if year == 2023 else 0.01,
                    "label_available": True,
                    "研究阶段": "训练期2021-2023",
                },
            ])
    return pd.DataFrame(rows)


def test_frozen_rule_uses_exclusive_lower_and_inclusive_upper_bound():
    panel = pd.DataFrame({
        "playbook_factor_rank": [1, 1, 10, 11],
        "structure_proximity_score": [40.0, 40.1, 60.0, 50.0],
    })

    assert frozen_rule_mask(panel).tolist() == [False, True, True, False]


def test_rule_selection_requires_cross_year_training_stability():
    panel = _panel()
    metrics = summarize_variants(panel)

    selected = select_frozen_rule(
        metrics, minimum_labels=20, minimum_dates=10,
    )

    assert selected["规则中文"] == FROZEN_RULE_CN
    assert selected["因子名次上限"] == 10
    assert selected["每日候选选择读取未来收益"] is False


def test_out_of_sample_rows_cannot_change_frozen_rule():
    panel = _panel()
    out_of_sample = panel.iloc[:8].copy()
    out_of_sample["date"] = pd.Timestamp("2024-01-02")
    out_of_sample["研究阶段"] = "验证期2024"
    out_of_sample["forward_return"] = -0.90

    baseline = select_frozen_rule(
        summarize_variants(panel), minimum_labels=20, minimum_dates=10,
    )
    stressed = select_frozen_rule(
        summarize_variants(pd.concat([panel, out_of_sample], ignore_index=True)),
        minimum_labels=20,
        minimum_dates=10,
    )

    assert stressed["规则编号"] == baseline["规则编号"]
    assert stressed["训练期指标"] == baseline["训练期指标"]
