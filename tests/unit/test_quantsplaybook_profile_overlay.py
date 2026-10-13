import pandas as pd

from stock_research.research.quantsplaybook_profile_overlay import add_profile_overlay_score


def test_profile_overlay_keeps_low_corr_and_rewards_entry_readiness():
    date = pd.Timestamp("2024-01-02")
    panel = pd.DataFrame([
        {
            "date": date,
            "code": "A",
            "playbook_factor_rank": 1,
            "candidate_score": 100.0,
            "trade_basis_score": 0.0,
            "structure_proximity_score": 0.0,
            "right_strength_score": 0.0,
            "avg_amount_20": 10_000_000.0,
        },
        {
            "date": date,
            "code": "B",
            "playbook_factor_rank": 2,
            "candidate_score": 99.0,
            "trade_basis_score": 9.0,
            "structure_proximity_score": 80.0,
            "right_strength_score": 35.0,
            "avg_amount_20": 1_000_000_000.0,
        },
    ])

    result = add_profile_overlay_score(panel).set_index("code")

    assert result.loc["B", "profile_overlay_score"] > result.loc[
        "A", "profile_overlay_score"
    ]
    assert result.loc["B", "profile_overlay_rank"] == 1
    assert result.loc["A", "original_candidate_score"] == 100.0
