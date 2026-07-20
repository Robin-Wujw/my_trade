"""Pure explanations and eligibility helpers for fundamental selection."""

from stock_research.indicators.factors import score_direct


# Value-line eligibility is intentionally allowlist-only.  These labels cover
# accounting-heavy manufacturers where book value and recurring earnings are
# meaningful inputs; an unknown or unmatched industry must not default to VALUE.
VALUE_INDUSTRY_RULE_VERSION = "value-industry-allowlist-v1"
VALUE_INDUSTRY_ALLOWLIST_EXACT = {
    "汽车电子电气系统",
    "通信网络设备及器件",
    "半导体",
    "半导体材料",
    "半导体设备",
    "分立器件",
}


def value_industry_allowlist_match(industry) -> str:
    text = str(industry or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return ""
    if text in VALUE_INDUSTRY_ALLOWLIST_EXACT:
        return text
    return ""


def is_value_industry_allowed(industry) -> bool:
    return bool(value_industry_allowlist_match(industry))


def quality_detail(eps, yoy, quality):
    eps_part = score_direct(eps, 0.10, 1.50) * 0.35
    yoy_part = score_direct(min(max(yoy, -0.5), 1.0), -0.10, 0.50) * 0.35
    history_part = max(0.0, min(30.0, quality - eps_part - yoy_part))
    eps_text = "较强" if eps_part >= 28 else "中等" if eps_part >= 17.5 else "偏弱"
    growth_text = "较强" if yoy_part >= 28 else "中等" if yoy_part >= 17.5 else "偏弱"
    history_text = (
        "稳定性较高"
        if history_part >= 24
        else "稳定性尚可"
        if history_part >= 15
        else "稳定性偏弱"
    )
    return (
        f"扣非EPS为{eps:.2f}元，盈利能力{eps_text}；"
        f"扣非利润同比{yoy:.1%}，增长{growth_text}；"
        f"近年扣非盈利{history_text}。综合质量评估{quality:.1f}"
    )


def value_method_reason(industry, mktcap, eps, yoy):
    industry = str(industry or "行业待核验")
    matched = value_industry_allowlist_match(industry)
    economics = (
        f"命中基本价值线行业白名单（{matched}），净资产和持续扣非盈利可用于初步估值"
        if matched else
        "未命中基本价值线行业白名单，不得使用基本价值线自动入选或建立左仓"
    )
    return (
        f"{industry}；{economics}；总市值{mktcap:.1f}亿元（不低于100亿元）、扣非EPS{eps:.2f}元、"
        f"扣非同比{yoy:.1%}均通过财务初筛。仍需人工确认主营稳定性和公式适用性"
    )


def growth_risk(yoy):
    if yoy >= 3:
        return "；风险：扣非同比超过300%，同比评分已封顶，需核验低基数、扭亏或一次性口径变化"
    if yoy < 0:
        return "；风险：扣非同比为负"
    return ""
