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


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(number) else number


def _bounded_score(value, default=50.0):
    number = _number(value, default)
    return max(0.0, min(100.0, number))


def _linear_score(value, worst, best, default=50.0):
    number = _number(value)
    if number is None or best == worst:
        return default
    return max(0.0, min(100.0, (number - worst) / (best - worst) * 100.0))


def _risk_control_score(row):
    ret20 = _number(row.get("ret20"), 0.0)
    ret60 = _number(row.get("ret60"), 0.0)
    liquidity = _bounded_score(row.get("liquidity_score"), 50.0)
    penalty = 0.0
    if ret20 > 0.35:
        penalty += min(25.0, (ret20 - 0.35) * 100.0)
    if ret60 > 0.80:
        penalty += min(20.0, (ret60 - 0.80) * 60.0)
    if ret20 < -0.15:
        penalty += min(15.0, (-0.15 - ret20) * 80.0)
    if ret60 < -0.25:
        penalty += min(10.0, (-0.25 - ret60) * 50.0)
    penalty += min(16.0, int(row.get("long_ma_overhead_count") or 0) * 8.0)
    penalty += min(12.0, int(row.get("short_ma_down_drag_count") or 0) * 4.0)
    if liquidity < 35.0:
        penalty += 10.0
    return max(0.0, min(100.0, 100.0 - penalty))


def calc_multi_factor_score(row):
    """Score visible row factors only; no data is fetched inside this function."""
    value = _bounded_score(row.get("valuation_score"), 50.0)
    quality = _bounded_score(row.get("quality_score"), 50.0)
    trend = _bounded_score(row.get("trend_score"), 50.0)
    liquidity = _bounded_score(row.get("liquidity_score"), 50.0)
    ret20 = _number(row.get("relative_ret20"))
    ret60 = _number(row.get("relative_ret60"))
    ret20 = _number(row.get("ret20"), 0.0) if ret20 is None else ret20
    ret60 = _number(row.get("ret60"), 0.0) if ret60 is None else ret60
    momentum = (
        trend * 0.55
        + _linear_score(ret20, -0.08, 0.12) * 0.25
        + _linear_score(ret60, -0.12, 0.20) * 0.20
    )
    risk = _risk_control_score(row)
    score = (
        value * 0.20
        + quality * 0.25
        + momentum * 0.25
        + liquidity * 0.15
        + risk * 0.15
    )
    if risk < 55.0:
        label = "overheated_penalty"
    elif quality >= 75.0 and momentum >= 70.0 and liquidity >= 55.0:
        label = "quality_momentum"
    elif value >= 75.0 and quality >= 65.0:
        label = "value_repair"
    elif liquidity >= 70.0 and trend >= 65.0:
        label = "liquidity_confirmed"
    else:
        label = "balanced_factor"
    components = {
        "value": round(value, 1),
        "quality": round(quality, 1),
        "momentum": round(momentum, 1),
        "liquidity": round(liquidity, 1),
        "risk": round(risk, 1),
    }
    return {
        "multi_factor_score": round(max(0.0, min(100.0, score)), 1),
        "multi_factor_label": label,
        "multi_factor_components": ";".join(
            f"{key}={value:.1f}" for key, value in components.items()
        ),
    }


def passes_multi_factor_right_gate(row, min_score=78.0):
    """Allow only already-qualified rows to receive a small quant-right boost."""
    score = _number(row.get("multi_factor_score"))
    if score is None or score < min_score:
        return False
    if row.get("selection_bucket") not in {CORE_BUCKET, HIGH_QUALITY_BUCKET}:
        return False
    quality = _number(row.get("quality_score"))
    growth = _number(row.get("earnings_yoy"))
    if growth is None:
        growth = _number(row.get("yoy"))
    market_cap = _number(row.get("mktcap"))
    trend = _number(row.get("trend_score"))
    liquidity = _number(row.get("liquidity_score"))
    if (
        quality is None
        or growth is None
        or market_cap is None
        or trend is None
        or liquidity is None
    ):
        return False
    if quality < 70.0 or growth < 0.10 or market_cap < 100.0:
        return False
    if trend < 70.0 or liquidity < 55.0:
        return False
    if _number(row.get("ret20"), 0.0) < 0.0:
        return False
    if _number(row.get("ret60"), 0.0) < -0.05:
        return False
    if _risk_control_score(row) < 60.0:
        return False
    return True


def calc_multi_factor_bonus(row, max_bonus=3.0):
    if not passes_multi_factor_right_gate(row):
        return 0.0
    score = _number(row.get("multi_factor_score"), 0.0)
    return round(min(max_bonus, max(0.0, (score - 75.0) / 5.0)), 1)


def apply_deduction_to_trend(base_score, deduction):
    """Let medium/long structure lead while short drag only times entry."""
    raw = float((deduction or {}).get("ma_deduction_score") or 0)
    adjustment = max(-15.0, min(15.0, raw * 0.20))
    return max(0.0, min(100.0, float(base_score) + adjustment)), adjustment


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
    if int(row.get("long_ma_overhead_count") or 0) >= 2:
        flags.append("两条以上中长下弯均线构成上方压力")
    short_drag = int(row.get("short_ma_down_drag_count") or 0)
    if short_drag:
        flags.append(f"{short_drag}条短期均线扣高且量不足，等待放量")
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
