import pytest

from stock_research.strategies.factor_selection import (
    apply_deduction_to_trend,
    build_risk_flags,
    calc_high_quality_score,
    calc_low_value_score,
    calc_total_score,
    classify_selection_bucket,
)


def test_deduction_adjustment_is_bounded_and_medium_long_led():
    improved, bonus = apply_deduction_to_trend(70, {"ma_deduction_score": 90})
    weakened, penalty = apply_deduction_to_trend(70, {"ma_deduction_score": -90})

    assert improved == 85
    assert bonus == 15
    assert weakened == 55
    assert penalty == -15


def test_deduction_risks_explain_pressure_and_entry_timing():
    row = {
        "method": "RIGHT", "selection_bucket": "高质量趋势",
        "valuation_score": 80, "quality_score": 80, "trend_score": 75,
        "liquidity_score": 60, "price_to_value": None,
        "long_ma_overhead_count": 2, "short_ma_down_drag_count": 2,
        "technical_flags_list": [],
    }

    flags = build_risk_flags(row)

    assert "中长下弯均线" in flags
    assert "等待放量" in flags


def test_factor_bucket_and_scores_keep_current_rules():
    row = {
        "method": "VALUE",
        "price_to_value": 0.80,
        "quality_score": 80,
        "liquidity_score": 60,
        "trend_score": 70,
    }
    assert classify_selection_bucket(row) == (
        "低估且高质量",
        "深度低估且高质量",
    )
    assert calc_low_value_score("VALUE", 80, 70, 60, 50) == pytest.approx(72.0)
    assert calc_high_quality_score("VALUE", 80, 70, 60, 50) == pytest.approx(64.0)
    score, mode = calc_total_score("RIGHT", 0, 80, 70, 60)
    assert score == pytest.approx(69.5)
    assert mode == "右侧趋势"


def test_factor_risk_flags_remain_explainable():
    row = {
        "method": "VALUE",
        "selection_bucket": "观察池",
        "valuation_score": 40,
        "quality_score": 40,
        "trend_score": 40,
        "liquidity_score": 30,
        "price_to_value": 0.20,
        "technical_flags_list": ["量能异常"],
    }
    assert build_risk_flags(row) == (
        "估值优势弱、质量偏弱、趋势未确认、流动性偏弱、"
        "价值线折价异常需复核、量能异常"
    )
