"""Pure explanations and eligibility helpers for fundamental selection."""

from stock_research.indicators.factors import score_direct


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
    manufacturing = [
        "计算机、通信", "电子设备", "专用设备", "通用设备",
        "汽车制造", "电气机械",
    ]
    if any(key in industry for key in manufacturing):
        economics = "属于制造业，净资产和持续扣非盈利能够反映经营价值"
    else:
        economics = "不属于金融、强周期资源或纯轻资产预期行业，可先用净资产和扣非盈利观察价值"
    return (
        f"{industry}；{economics}；市值{mktcap:.1f}亿元、扣非EPS{eps:.2f}元、"
        f"扣非同比{yoy:.1%}均通过基本价值线初筛。仍需人工确认行业龙头地位和产业趋势"
    )


def growth_risk(yoy):
    if yoy >= 3:
        return "；风险：扣非同比超过300%，同比评分已封顶，需核验低基数、扭亏或一次性口径变化"
    if yoy < 0:
        return "；风险：扣非同比为负"
    return ""
