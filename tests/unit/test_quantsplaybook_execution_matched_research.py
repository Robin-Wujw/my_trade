import pandas as pd

from stock_research.research.quantsplaybook_execution_matched_research import (
    attach_point_in_time_industry,
    rule_masks,
    select_independent_events,
)


def _panel():
    dates = pd.bdate_range("2026-01-05", periods=25)
    rows = []
    for date in dates:
        for rank, (code, industry) in enumerate((
            ("sh.600001", "行业甲"),
            ("sh.600002", "行业甲"),
            ("sh.600003", "行业甲"),
            ("sh.600004", "行业乙"),
        ), start=1):
            rows.append({
                "date": date,
                "code": code,
                "industry": industry,
                "playbook_factor_rank": rank,
                "playbook_factor_score": 1.0 - rank / 10,
                "structure_proximity_score": 70.0,
                "candidate_score": 100.0 - rank,
            })
    return pd.DataFrame(rows)


def test_rule_accepts_structure_scores_above_old_upper_bound():
    panel = _panel()

    masks = rule_masks(panel)

    assert masks["前10名且结构分大于40"].all()
    assert not masks["原规则_前10名且结构分大于40不高于60"].any()


def test_event_selection_applies_symbol_and_industry_cooldowns():
    panel = _panel()

    events = select_independent_events(
        panel,
        rule_masks(panel)["前10名且结构分大于40"],
    )

    first_day = events[events["date"].eq(events["date"].min())]
    assert set(first_day["code"]) == {
        "sh.600001", "sh.600002", "sh.600004",
    }
    assert events.groupby("code").size().max() == 2
    industry_a = events[events["industry"].eq("行业甲")]
    assert industry_a.groupby(industry_a["date"].dt.to_period("M")).size().max() <= 2


def test_industry_lookup_uses_membership_interval_at_observation_date():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2025-06-01"]),
        "code": ["sh.600001", "sh.600001"],
    })
    history = pd.DataFrame({
        "code": ["sh.600001", "sh.600001"],
        "industry": ["旧行业", "新行业"],
        "in_date_ts": pd.to_datetime(["2020-01-01", "2025-01-01"]),
        "out_date_ts": pd.to_datetime(["2024-12-31", None]),
    })

    result = attach_point_in_time_industry(panel, history)

    assert result["industry"].tolist() == ["旧行业", "新行业"]
    assert result["industry_point_in_time"].all()
