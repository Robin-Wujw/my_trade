import pytest

from stock_research.pipelines.daily_report import ensure_same_observation_date
from stock_research.strategies.fundamental_selection import (
    growth_risk,
    quality_detail,
    value_method_reason,
)


def test_fundamental_explanations_keep_current_wording():
    detail = quality_detail(1.50, 0.50, 100)
    assert "扣非EPS为1.50元，盈利能力较强" in detail
    assert "扣非利润同比50.0%，增长较强" in detail
    assert "近年扣非盈利稳定性较高" in detail
    reason = value_method_reason("计算机、通信", 120, 1.20, 0.20)
    assert "属于制造业" in reason
    assert "市值120.0亿元" in reason
    assert "超过300%" in growth_risk(3.01)
    assert "同比为负" in growth_risk(-0.01)


def test_report_rejects_mixed_observation_dates():
    with pytest.raises(ValueError, match="observation date mismatch"):
        ensure_same_observation_date(
            {"formula33": "2026-07-02", "selection": "2026-07-03"}
        )
