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
