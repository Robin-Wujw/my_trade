from stock_research.strategies.candidate_interface import (
    MAX_DAILY_CANDIDATES,
    normalize_candidate,
    normalize_candidate_snapshots,
)
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION


def _candidate(index, source, score):
    row = {
        "date": "2026-01-02",
        "code": f"sz.{index:06d}",
        "name": f"候选{index}",
        "candidate_source": source,
        "candidate_score": score,
        "quality_score": 80,
        "earnings_yoy": 0.20,
        "mktcap": 200,
        "price_to_value": 1.0,
        "signal_eligible": True,
        "selected_for_trading": True,
        "data_status": "traded",
        "valid_price_bar": True,
    }
    if source == "factor_quant":
        row["right_quant_score"] = score
    if "value_model" in source.split("+"):
        row.update({
            "industry": "汽车电子电气系统",
            "value_industry_allowed": True,
            "value_industry_allowlist_match": "汽车电子电气系统",
            "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        })
    return row


def test_daily_candidate_pool_uses_factor_rank_without_core_reservation():
    rows = []
    for index in range(12):
        rows.append(_candidate(index, "value_model", 100 - index))
    for index in range(12, 80):
        rows.append(_candidate(index, "factor_quant", 200 - index))

    selected = normalize_candidate_snapshots({"2026-01-02": rows})["2026-01-02"]

    assert len(selected) == MAX_DAILY_CANDIDATES
    old_value_count = sum(
        "value_model" in item["candidate_source"].split("+")
        for item in selected
    )
    assert old_value_count == 0
    assert selected[0]["selection_rank"] == 1
    assert selected[0]["code"] == "sz.000012"


def test_daily_candidate_pool_limit_can_be_explicitly_expanded_for_research():
    rows = [
        _candidate(index, "factor_quant", 200 - index)
        for index in range(80)
    ]

    selected = normalize_candidate_snapshots(
        {"2026-01-02": rows},
        max_daily_candidates=75,
    )["2026-01-02"]

    assert len(selected) == 75
    assert selected[-1]["selection_rank"] == 75


def test_historical_value_candidate_requires_allowlisted_industry_audit():
    rejected = normalize_candidate({
        "code": "sh.600612",
        "candidate_source": "value_model",
        "snapshot_version": "unified-selection-v4",
        "industry": "饰品",
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 150,
        "price_to_value": 1.0,
    })
    allowed = normalize_candidate({
        "code": "sz.300502",
        "candidate_source": "value_model",
        "snapshot_version": "unified-selection-v5-value-industry-allowlist",
        "industry": "通信网络设备及器件",
        "value_industry_allowed": True,
        "value_industry_allowlist_match": "通信网络设备及器件",
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 150,
        "price_to_value": 1.0,
    })

    assert rejected["allow_left"] is False
    assert "value_industry_not_allowlisted" in rejected["candidate_failure_reason"]
    assert allowed["allow_left"] is True


def test_value_candidate_industry_audit_fails_closed_at_every_boundary():
    base = {
        "code": "TEST",
        "candidate_source": "value_model",
        "industry": "半导体",
        "value_industry_allowed": True,
        "value_industry_allowlist_match": "半导体",
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 200,
        "price_to_value": 1.0,
    }

    assert normalize_candidate(base)["allow_left"] is True
    for update, reason in [
        ({"industry": ""}, "value_industry_not_allowlisted"),
        ({"industry": "半导体软件"}, "value_industry_not_allowlisted"),
        ({"industry": "电子"}, "value_industry_not_allowlisted"),
        ({"industry": "饰品"}, "value_industry_not_allowlisted"),
        ({"industry": "医疗服务"}, "value_industry_not_allowlisted"),
        ({"value_industry_allowed": False}, "value_industry_audit_conflict"),
        ({"value_industry_rule_version": ""}, "value_industry_rule_version_missing"),
        ({"value_industry_rule_version": "old"}, "value_industry_rule_version_mismatch"),
    ]:
        candidate = {**base, **update}
        normalized = normalize_candidate(candidate)
        assert normalized["allow_left"] is False
        assert reason in normalized["candidate_failure_reason"]

    missing_audit = dict(base)
    missing_audit.pop("value_industry_allowed")
    normalized = normalize_candidate(missing_audit)
    assert normalized["allow_left"] is False
    assert "value_industry_audit_missing" in normalized["candidate_failure_reason"]


def test_mixed_value_factor_candidate_loses_only_left_permission():
    normalized = normalize_candidate({
        "code": "sh.600612",
        "candidate_source": "value_model+factor_quant",
        "industry": "饰品",
        "value_industry_allowed": True,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 200,
        "price_to_value": 1.0,
    })

    assert normalized["allow_left"] is False
    assert normalized["allow_right"] is True
    assert "value_industry_audit_conflict" in normalized["candidate_failure_reason"]


def test_quantsplaybook_right_candidate_does_not_require_fundamentals():
    normalized = normalize_candidate({
        "code": "sz.300001",
        "candidate_source": "quantsplaybook_factor",
        "selection_profile": "quantsplaybook_factor_only",
        "playbook_factor": "playbook_low_corr",
        "playbook_factor_score": 0.8,
        "candidate_score": 95.0,
        "quality_score": 20.0,
        "earnings_yoy": -0.30,
        "mktcap": 20.0,
        "selected_for_trading": True,
        "signal_eligible": True,
        "valid_price_bar": float("nan"),
        "is_traded_bar": float("nan"),
    })

    assert normalized["allow_left"] is False
    assert normalized["allow_right"] is True
    assert normalized["signal_eligible"] is True
    assert normalized["candidate_score"] == 95.0


def test_invalid_value_lane_does_not_disable_quantsplaybook_right_lane():
    normalized = normalize_candidate({
        "code": "sz.300502",
        "candidate_source": "value_model+quantsplaybook_factor",
        "candidate_score": 97.0,
        "industry": "通信网络设备及器件",
        "value_industry_allowed": True,
        "value_industry_allowlist_match": "通信网络设备及器件",
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 60.0,
        "earnings_yoy": 0.05,
        "mktcap": 80.0,
        "price_to_value": 1.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })

    assert normalized["allow_left"] is False
    assert normalized["allow_right"] is True
    assert normalized["signal_eligible"] is True
    assert "quality_score_below_70" in normalized["candidate_failure_reason"]
    assert "earnings_yoy_below_10pct" in normalized["candidate_failure_reason"]
    assert "mktcap_below_100" in normalized["candidate_failure_reason"]


def test_value_and_quantsplaybook_overlap_keeps_both_lanes_and_right_score():
    normalized = normalize_candidate({
        "code": "sz.300502",
        "candidate_source": "value_model+quantsplaybook_factor",
        "selection_profile": "quantsplaybook_factor_only",
        "playbook_factor": "playbook_low_corr",
        "playbook_factor_score": 0.8,
        "candidate_score": 97.0,
        "industry": "通信网络设备及器件",
        "value_industry_allowed": True,
        "value_industry_allowlist_match": "通信网络设备及器件",
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 200,
        "price_to_value": 1.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })

    assert normalized["allow_left"] is True
    assert normalized["allow_right"] is True
    assert normalized["signal_eligible"] is True
    assert normalized["candidate_score"] == 97.0


def test_pe_pb_left_candidates_are_left_executable_without_industry_allowlist():
    pb = normalize_candidate({
        "code": "sh.600153",
        "candidate_source": "pb_model",
        "quality_score": 65.0,
        "earnings_yoy": 0.05,
        "mktcap": 120.0,
        "price_to_value": 1.0,
        "pe_ratio": None,
        "pb_ratio": 0.7,
        "candidate_score": 75.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })
    pe = normalize_candidate({
        "code": "sh.600028",
        "candidate_source": "pe_model",
        "quality_score": 70.0,
        "earnings_yoy": -0.05,
        "mktcap": 110.0,
        "price_to_value": 1.0,
        "pe_ratio": 0.5,
        "pb_ratio": None,
        "candidate_score": 80.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })

    assert pb["allow_left"] is True
    assert pb["allow_right"] is False
    assert pb["signal_eligible"] is True
    assert pb["candidate_score"] == 75.0
    assert pe["allow_left"] is True
    assert pe["allow_right"] is False
    assert pe["signal_eligible"] is True
    assert pe["candidate_score"] == 80.0


def test_pe_pb_left_candidate_below_quality_floor_is_ineligible():
    normalized = normalize_candidate({
        "code": "sh.600153",
        "candidate_source": "pb_model",
        "quality_score": 55.0,
        "earnings_yoy": 0.05,
        "mktcap": 120.0,
        "price_to_value": 1.0,
        "pb_ratio": 0.7,
        "candidate_score": 75.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })

    assert normalized["allow_left"] is True
    assert normalized["signal_eligible"] is False
    assert "quality_score_below_60" in normalized["candidate_failure_reason"]


def test_pe_pb_quantsplaybook_overlap_keeps_both_lanes():
    normalized = normalize_candidate({
        "code": "sh.600153",
        "candidate_source": "pe_model+quantsplaybook_factor",
        "quality_score": 75.0,
        "earnings_yoy": 0.05,
        "mktcap": 120.0,
        "price_to_value": 1.0,
        "pe_ratio": 0.7,
        "pb_ratio": None,
        "candidate_score": 97.0,
        "selected_for_trading": True,
        "signal_eligible": True,
    })

    assert normalized["allow_left"] is True
    assert normalized["allow_right"] is True
    assert normalized["signal_eligible"] is True
    assert normalized["candidate_score"] == 97.0


def test_nullable_candidate_values_do_not_raise_or_grant_permission():
    import pandas as pd
    row = normalize_candidate({"code": "sz.000001", "name": pd.NA,
                               "candidate_source": pd.NA, "selection_reason": pd.NA,
                               "allow_left": pd.NA, "allow_right": pd.NA,
                               "signal_eligible": pd.NA, "selected_for_trading": pd.NA,
                               "valid_price_bar": pd.NA})
    assert not row["signal_eligible"]
    assert not row["allow_right"]
    assert not row["selected_for_trading"]


def test_mixed_pb_source_does_not_bypass_right_growth_gate():
    candidate = _candidate(1, "pb_model+standard_mainline", 90)
    candidate.update(quality_score=65, earnings_yoy=-.1)
    row = normalize_candidate(candidate)
    assert row["allow_left"]
    assert row["signal_eligible"]
    assert not row["allow_right"]
    assert "right_earnings_yoy" in row["candidate_failure_reason"]
