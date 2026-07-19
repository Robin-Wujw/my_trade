import pytest

from stock_research.strategies.factor_selection import (
    CORE_BUCKET,
    WATCH_BUCKET,
    apply_deduction_to_trend,
    build_risk_flags,
    calc_high_quality_score,
    calc_low_value_score,
    calc_multi_factor_bonus,
    calc_multi_factor_score,
    calc_total_score,
    classify_selection_bucket,
    passes_multi_factor_right_gate,
)
from stock_research.pipelines.factor_selection import is_value_watch_candidate


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


def test_value_watch_candidate_requires_full_left_safety_gate():
    zhongxinbo_style = {
        "method": "VALUE",
        "price_to_value": 0.9437,
        "quality_score": 92.5,
        "earnings_yoy": 1.9286,
        "mktcap": 127.13,
    }
    deep_discount = {
        **zhongxinbo_style,
        "price_to_value": 0.86,
    }
    missing_growth = {
        **zhongxinbo_style,
        "earnings_yoy": None,
    }

    assert not is_value_watch_candidate(zhongxinbo_style)
    assert is_value_watch_candidate(deep_discount)
    assert not is_value_watch_candidate(missing_growth)


def test_multi_factor_score_is_bounded_and_explainable():
    result = calc_multi_factor_score({
        "valuation_score": 120,
        "quality_score": 110,
        "trend_score": 95,
        "liquidity_score": 90,
        "ret20": 0.10,
        "ret60": 0.18,
    })

    assert 0 <= result["multi_factor_score"] <= 100
    assert result["multi_factor_label"] == "quality_momentum"
    assert "quality=" in result["multi_factor_components"]


def test_multi_factor_prefers_quality_momentum_and_liquidity():
    strong = calc_multi_factor_score({
        "valuation_score": 75,
        "quality_score": 88,
        "trend_score": 82,
        "liquidity_score": 76,
        "ret20": 0.08,
        "ret60": 0.18,
    })
    weak = calc_multi_factor_score({
        "valuation_score": 75,
        "quality_score": 45,
        "trend_score": 42,
        "liquidity_score": 28,
        "ret20": -0.12,
        "ret60": -0.18,
    })

    assert strong["multi_factor_score"] > weak["multi_factor_score"]
    assert strong["multi_factor_score"] >= 75


def test_multi_factor_penalizes_overheated_returns():
    steady = calc_multi_factor_score({
        "valuation_score": 70,
        "quality_score": 85,
        "trend_score": 82,
        "liquidity_score": 70,
        "ret20": 0.12,
        "ret60": 0.25,
    })
    overheated = calc_multi_factor_score({
        "valuation_score": 70,
        "quality_score": 85,
        "trend_score": 82,
        "liquidity_score": 70,
        "ret20": 0.75,
        "ret60": 1.20,
        "long_ma_overhead_count": 2,
        "short_ma_down_drag_count": 2,
    })

    assert overheated["multi_factor_score"] < steady["multi_factor_score"]
    assert overheated["multi_factor_label"] == "overheated_penalty"


def test_multi_factor_right_gate_keeps_hard_conditions():
    qualified = {
        **calc_multi_factor_score({
            "valuation_score": 80,
            "quality_score": 90,
            "trend_score": 85,
            "liquidity_score": 78,
            "ret20": 0.10,
            "ret60": 0.20,
        }),
        "selection_bucket": CORE_BUCKET,
        "quality_score": 90,
        "earnings_yoy": 0.30,
        "mktcap": 180,
        "trend_score": 85,
        "liquidity_score": 78,
        "ret20": 0.10,
        "ret60": 0.20,
    }
    weak_growth = {**qualified, "earnings_yoy": 0.05}
    watch = {**qualified, "selection_bucket": WATCH_BUCKET}

    assert passes_multi_factor_right_gate(qualified)
    assert calc_multi_factor_bonus(qualified) > 0
    assert not passes_multi_factor_right_gate(weak_growth)
    assert calc_multi_factor_bonus(weak_growth) == 0
    assert not passes_multi_factor_right_gate(watch)
