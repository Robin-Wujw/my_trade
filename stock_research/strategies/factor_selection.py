"""Factor-selection eligibility, bucket, score, and risk rules."""

import pandas as pd


VALUE_UNDERVALUED_RATIO = 0.85
VALUE_LOW_VALUE_RATIO = 1.00
VALUE_QUALITY_MAX_RATIO = 1.80
PEPB_QUALITY_MAX_RATIO = 1.70
PEPB_QUALITY_MAX_PERCENTILE = 0.85
QUALITY_MIN_SCORE = 70
PEPB_QUALITY_MIN_SCORE = 65
QUALITY_MIN_TREND_SCORE = 60
RIGHT_MIN_TREND_SCORE = 70
QUALITY_MIN_LIQUIDITY_SCORE = 40

LOW_VALUE_BUCKET = "低估价值"
HIGH_QUALITY_BUCKET = "高质量趋势"
CORE_BUCKET = "低估且高质量"
WATCH_BUCKET = "观察池"


def is_low_value_candidate(row):
    method = row["method"]
    if method == "RIGHT":
        return False
    if method == "VALUE":
        ratio = row.get("price_to_value")
        return ratio is not None and pd.notna(ratio) and ratio <= VALUE_LOW_VALUE_RATIO
    ratio = row.get("pepb_ratio")
    percentile = row.get("valuation_percentile")
    return (ratio is not None and pd.notna(ratio) and ratio <= 1.0) or (
        percentile is not None and pd.notna(percentile) and percentile <= 0.15
    )


def is_high_quality_candidate(row):
    minimum = PEPB_QUALITY_MIN_SCORE if row["method"] in {"PE", "PB"} else QUALITY_MIN_SCORE
    if row["quality_score"] < minimum:
        return False
    if row["liquidity_score"] < QUALITY_MIN_LIQUIDITY_SCORE:
        return False
    if row["method"] == "RIGHT":
        return row["trend_score"] >= RIGHT_MIN_TREND_SCORE
    if row["method"] == "VALUE":
        ratio = row.get("price_to_value")
        return (
            ratio is not None
            and pd.notna(ratio)
            and ratio <= VALUE_QUALITY_MAX_RATIO
            and row["trend_score"] >= QUALITY_MIN_TREND_SCORE
        )
    ratio = row.get("pepb_ratio")
    percentile = row.get("valuation_percentile")
    valuation_ok = (ratio is not None and pd.notna(ratio) and ratio <= PEPB_QUALITY_MAX_RATIO) or (
        percentile is not None
        and pd.notna(percentile)
        and percentile <= PEPB_QUALITY_MAX_PERCENTILE
    )
    return valuation_ok and row["trend_score"] >= QUALITY_MIN_TREND_SCORE


def classify_selection_bucket(row):
    low_value = is_low_value_candidate(row)
    high_quality = is_high_quality_candidate(row)
    if low_value and high_quality:
        if row["method"] == "VALUE" and row.get("price_to_value") is not None and row["price_to_value"] <= VALUE_UNDERVALUED_RATIO:
            return CORE_BUCKET, "深度低估且高质量"
        return CORE_BUCKET, "低估且高质量"
    if low_value:
        if row["method"] == "VALUE" and row.get("price_to_value") is not None and row["price_to_value"] <= VALUE_UNDERVALUED_RATIO:
            return LOW_VALUE_BUCKET, "深度低估"
        return LOW_VALUE_BUCKET, "价值线内/历史低位"
    if high_quality:
        return (
            (HIGH_QUALITY_BUCKET, "右侧高质量")
            if row["method"] == "RIGHT"
            else (HIGH_QUALITY_BUCKET, "估值有约束的高质量")
        )
    return WATCH_BUCKET, "未达主策略"


def build_risk_flags(row):
    flags = []
    if row.get("method") != "RIGHT" and row["valuation_score"] < 45 and row.get("selection_bucket") != LOW_VALUE_BUCKET:
        flags.append("估值优势弱")
    if row["quality_score"] < 45:
        flags.append("质量偏弱")
    if row["trend_score"] < 45:
        flags.append("趋势未确认")
    if row["liquidity_score"] < 35:
        flags.append("流动性偏弱")
    if row.get("price_to_value") is not None and row["price_to_value"] < 0.30:
        flags.append("价值线折价异常需复核")
    flags.extend(row.get("technical_flags_list", []))
    return "、".join(flags) if flags else "正常"


def calc_low_value_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    if method in {"PE", "PB"}:
        return valuation_score * 0.55 + quality_score * 0.25 + trend_score * 0.10 + liquidity_score * 0.10
    return valuation_score * 0.50 + quality_score * 0.30 + trend_score * 0.10 + liquidity_score * 0.10


def calc_high_quality_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    if method == "RIGHT":
        return quality_score * 0.35 + trend_score * 0.45 + liquidity_score * 0.20
    return valuation_score * 0.05 + quality_score * 0.45 + trend_score * 0.35 + liquidity_score * 0.15


def calc_total_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    if method == "RIGHT":
        return quality_score * 0.20 + trend_score * 0.55 + liquidity_score * 0.25, "右侧趋势"
    value_reversion = valuation_score * 0.40 + quality_score * 0.25 + trend_score * 0.25 + liquidity_score * 0.10
    if method in {"PE", "PB"}:
        return value_reversion, "估值修复"
    scores = {
        "估值修复": value_reversion,
        "高质量折价": valuation_score * 0.30 + quality_score * 0.45 + trend_score * 0.10 + liquidity_score * 0.15,
        "成长动量": valuation_score * 0.10 + quality_score * 0.35 + trend_score * 0.40 + liquidity_score * 0.15,
    }
    mode, score = max(scores.items(), key=lambda item: item[1])
    return score, mode
