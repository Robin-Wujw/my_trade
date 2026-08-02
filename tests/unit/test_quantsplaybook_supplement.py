from apps.quantsplaybook_supplement import (
    BLEND_FACTOR,
    merge_primary_with_supplement,
)


def test_supplement_keeps_all_primary_and_ranks_below_primary_floor():
    primary = [
        {"code": "A", "candidate_score": 99.0},
        {"code": "B", "candidate_score": 95.0},
    ]
    supplement = [
        {
            "code": "A",
            "three_lane_membership": "smooth_52week_high",
            "three_lane_scores": "smooth_52week_high=0.99",
        },
        {
            "code": "C",
            "three_lane_membership": "smooth_52week_high+stage2_vcp",
            "three_lane_scores": "smooth_52week_high=0.98;stage2_vcp=0.90",
        },
        {
            "code": "D",
            "three_lane_membership": "stage2_vcp",
            "three_lane_scores": "stage2_vcp=0.97",
        },
    ]

    rows = merge_primary_with_supplement(primary, supplement, supplement_count=5)

    assert [row["code"] for row in rows] == ["A", "B", "C"]
    assert rows[-1]["candidate_score"] < 95.0
    assert all(row["playbook_factor"] == BLEND_FACTOR for row in rows)
    assert rows[-1]["blend_lane"] == "supplement_smooth_52week_high"
