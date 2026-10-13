"""Chinese display text for strategy signal and audit codes."""
from __future__ import annotations

import re


_CODE_LIKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]*$")
_BATCH_PATTERN = re.compile(
    r"^(?P<side>[LRlr])(?P<number>\d+)"
    r"(?:\(原(?P<origin_side>[LRlr])(?P<origin_number>\d+)\))?$"
)


REASON_REPLACEMENTS = {
    "leader pivot close breakout": "龙头枢轴收盘突破",
    "stop 5pct below breakout line": "突破线下方5%止损",
    "entry time condition": "入场时间条件退出",
    "hard space stop": "空间硬止损",
    "close_confirmed": "收盘确认成交",
    "intraday_breakout": "盘中突破成交",
    "gap_breakout": "跳空越过触发价成交",
    "gap_or_open_fill": "开盘已触发成交",
    "intraday_cross": "盘中触及结构位成交",
    "intraday_stop": "盘中硬止损成交",
    "gap_stop": "跳空止损成交",
    "condition stop": "收盘条件止损",
    "14:55/close proxy": "按当日收盘价近似成交（非真实盘中成交）",
    "收盘代理": "按当日收盘价近似成交（非真实盘中成交）",
}

PROFIT_CONDITION_REPLACEMENTS = {
    "profit_floor": "利润保护线",
    "trailing_10": "从阶段高点回撤10%",
    "rising_ma20_break_3pct": "跌破上行MA20 3%",
    "maximum_profit_half": "最大浮盈回撤一半",
}

EXIT_REASON_REPLACEMENTS = {
    "condition_stop": "收盘条件止损",
    "hard_space_stop": "空间硬止损",
    "profit_floor": "利润保护线止盈",
    "entry_time_condition": "入场时间条件退出",
    "other": "其他退出",
    "trailing_10": "从阶段高点回撤10%止盈",
    "volume_node_break": "跌破量价节点止盈",
    "maximum_profit_half": "最大浮盈回撤一半，趋势尾仓退出",
}

BLOCK_REASON_REPLACEMENTS = {
    "ok": "通过",
    "missing_signal": "没有有效技术信号",
    "invalid_or_non_protective_stop": "止损价无效或不能保护本金",
    "quality_hard_gate_failed": "质量硬门槛未通过",
    "growth_hard_gate_failed": "增长硬门槛未通过",
    "market_cap_hard_gate_failed": "市值硬门槛未通过",
    "trade_basis_too_weak": "交易基础过弱",
    "liquidity_too_thin": "流动性不足",
    "weak_20d_relative_strength": "20日相对强度不足",
    "weak_relative_strength": "相对强度不足",
    "large_liquid_trend_watch_only": "大市值高流动性趋势核心仅观察",
    "support_pullback_without_fast_ma_confluence": "支撑拉回缺少快速均线共振",
    "volume_confirmation_missing": "缺少量能确认",
    "late_acceleration_chase": "加速末段，不追高",
    "risk_pct_too_tight": "止损距离过窄",
    "close_confirmed_risk_pct_too_tight": "收盘确认买点的止损距离过窄",
    "risk_pct_too_wide": "止损距离过宽",
    "reward_risk_too_low": "计划盈亏比不足",
    "entry_evidence_below_min": "旧证据分低于历史参数（仅解释字段）",
    "order_not_filled": "价格未触发成交",
    "support_pullback_not_first_entry": "支撑拉回首仓放宽后仍未达量化底线，需等待量能/强度/支撑距离改善",
    "add_current_profit_not_met": "现有仓位浮盈未达到加仓条件",
    "total_symbol_limit": "全部实际持股已达安全上限",
    "industry_or_correlation_limit": "同行业或高相关标的数量已达上限",
    "new_symbol_unlock_not_met": "新标的开仓解锁条件未满足",
    "weak_replacement_scheduled": "已安排下一交易日汰弱换强",
    "symbol_limit_no_weak_replacement": "仓位已满且没有可替换弱仓",
    "symbol_exposure_cap": "单股仓位已达上限",
    "gross_exposure_cap": "组合总仓位已达上限",
    "cash_or_board_lot_limit": "现金不足或不足一手",
    "left_to_right_management_only_without_current_candidate": "非当日候选，仅执行左转右管理状态检查",
    "left_to_right_add_on_cash_or_board_lot_limit": "左转右已完成，但现金不足或附加仓不足一手",
    "weak_replacement_executed": "已执行汰弱换强",
    "next_close_confirmation_failed": "次日收盘未站上触发价",
    "next_close_confirmation_too_extended": "次日确认价高于触发价5%，不追高",
    "breadth_unique_count_missing": "信号日Formula33广度数据缺失",
    "breadth_not_low_enough": "信号日Formula33广度尚未降至低位",
    "reversal_60d_decline_not_deep_enough": "候选日60日跌幅不足10%，不属于本反转通道",
    "factor_only_non_reversal_signal_disabled": "纯因子反转通道已关闭其他趋势买点",
    "reversal_symbol_already_held": "该标的已有持仓，不重复开启反转批次",
    "confirmation_locked_limit_up": "确认日封死涨停，无法按收盘价成交",
    "awaiting_next_close_confirmation": "等待次日收盘确认",
    "confirmation_not_available_by_end": "回测结束前没有下一交易日可供确认",
    "confirmed": "次日收盘确认通过",
    "factor_horizon_rule_mismatch": "候选不再满足已冻结的因子周期匹配规则",
    "factor_horizon_return_60d_filter_mismatch": (
        "候选日60日跌幅未达到因子周期深回撤门槛"
    ),
    "factor_horizon_ma20_filter_mismatch": (
        "候选日MA20已经上扬，不符合因子周期深回撤反转条件"
    ),
    "factor_horizon_watch_phase_excluded": (
        "Formula33处于观察阶段，暂不启动因子周期交易"
    ),
    "factor_horizon_capacity_full": "因子周期匹配仓位已满，等待自然到期释放名额",
    "missing_entry_price": "下一交易日开盘价缺失",
    "factor_horizon_rule_matched": "通过已冻结的因子周期匹配规则",
    "factor_only_non_compressed_structure_signal_disabled": (
        "纯因子紧结构通道已关闭其他趋势买点"
    ),
    "compressed_structure_non_platform_signal": "不是整理平台突破信号",
    "compressed_structure_original_gate_mismatch": (
        "原买入门并非仅因止损距离过窄而拒绝"
    ),
    "compressed_structure_quality_gate_failed": (
        "紧结构技术证据分或候选结构接近分不足"
    ),
    "compressed_structure_liquidity_too_thin": (
        "紧结构候选近20日平均成交额低于专用下限"
    ),
    "compressed_structure_rule_matched": "通过紧结构平台延续规则",
    "compressed_structure_next_trade_date_unavailable": (
        "回测结束前没有下一全市场交易日"
    ),
    "compressed_structure_execution_date_missed": "指定执行日已错过，不顺延成交",
    "compressed_structure_price_frame_missing": "缺少该标的行情序列",
    "compressed_structure_no_trade_on_execution_date": (
        "下一全市场交易日该标的无有效K线，不顺延成交"
    ),
    "compressed_structure_missing_entry_price": "下一交易日开盘价缺失",
    "compressed_structure_symbol_already_held": "该标的已有持仓，不重复开仓",
    "compressed_structure_capacity_full": "紧结构平台延续仓位已满",
    "specialized_channel_capacity_full": "该专用通道同时持仓数量已达上限",
}

SIGNAL_TYPE_REPLACEMENTS = {
    "bull_run_half_pullback": "连阳一半拉回",
    "breadth_reversal_half_pullback": "低广度反转连阳拉回买点",
    "consolidation_breakout": "整理平台突破",
    "gap_long_ma_breakout": "跳空长阳突破长期均线",
    "leader_pivot_breakout": "龙头枢轴突破",
    "leader_trend_add": "龙头趋势加仓",
    "uptrend_support_pullback": "上涨波段支撑拉回",
    "pullback_50_breakout": "回调50%放量突破",
    "uptrend_50_reclaim": "上涨50%收复",
    "volume_price_node": "量价节点突破",
    "uptrend_ratio_pullback": "上涨波段比例拉回",
    "factor_horizon_match": "因子周期匹配买点",
    "compressed_structure_continuation": "紧结构平台延续买点",
    "unknown": "未知信号",
}

CANDIDATE_FAILURE_REPLACEMENTS = {
    "value_industry_not_allowlisted": "价值线行业未命中白名单",
    "value_industry_audit_missing": "价值线行业审计字段缺失",
    "value_industry_audit_conflict": "价值线行业审计冲突",
    "value_industry_match_audit_missing": "价值线行业命中字段缺失",
    "value_industry_match_audit_conflict": "价值线行业命中字段冲突",
    "value_industry_rule_version_missing": "价值线行业规则版本缺失",
    "value_industry_rule_version_mismatch": "价值线行业规则版本不一致",
    "quality_score_missing": "质量分缺失",
    "quality_score_below_70": "质量分低于70",
    "earnings_yoy_missing": "扣非增长缺失",
    "earnings_yoy_below_10pct": "扣非增长低于10%",
    "mktcap_missing": "总市值缺失",
    "mktcap_below_100": "总市值低于100亿元",
    "price_to_value_missing": "价格/价值线缺失",
    "price_to_value_out_of_range": "价格/价值线不在可执行区间",
    "left_high_growth_small_cap_needs_deeper_discount": "高增长小市值左侧候选需要更深折价",
    "no_executable_lane": "没有可执行通道",
    "value_financial_falsification": "价值或财务逻辑被证伪",
    "value_safety_rejected": "价值线安全边际不足",
    "risk_warning_stock": "风险警示股票",
    "suspended_or_no_trade": "观察日停牌或无交易",
    "data_unavailable": "行情或财务数据不可用",
    "invalid_price_bar": "观察日价格K线无效",
    "not_selected_for_trading": "未进入当日可交易观察池",
}

SEMANTIC_SETUP_REPLACEMENTS = {
    "factor_only_high_rr_right_entry": "纯因子右侧高盈亏比买点",
    "high_rr_right_entry": "高盈亏比右侧买点",
    "legacy_minimal_candidate_profile": "旧版最小候选画像放行",
    "breadth_reversal_half_pullback": "低广度反转连阳拉回买点",
    "factor_horizon_match": "因子周期匹配买点",
    "compressed_structure_continuation": "紧结构平台延续买点",
}

SUMMARY_CODE_REPLACEMENTS = {
    "semantic_high_reward_risk": "语义高盈亏比买入门",
    "legacy_explanation_only": "旧证据分仅作解释",
}

NOTE_REPLACEMENTS = {
    "exclude_topN is a contribution subtraction diagnostic, not a cash-rebalanced rerun.": (
        "剔除头部贡献只是从收益贡献中做减法诊断，并非重新按现金再平衡回测。"
    ),
    "R is measured from each fill's raw entry price to its initial raw stop.": (
        "R按每笔成交的原始入场价到初始原始止损价之间的风险距离计算。"
    ),
}


def profit_condition_text(value) -> str:
    raw = str(value or "").strip()
    if raw.startswith("volume_node_break:"):
        date = raw.split(":", 1)[1]
        return f"跌破量价节点（节点日{date}）"
    if raw in PROFIT_CONDITION_REPLACEMENTS:
        return PROFIT_CONDITION_REPLACEMENTS[raw]
    return "未识别止盈条件" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw


def block_reason_text(value) -> str:
    raw = str(value or "").strip()
    if ":" in raw:
        prefix, detail = raw.split(":", 1)
        if prefix in BLOCK_REASON_REPLACEMENTS:
            return f"{BLOCK_REASON_REPLACEMENTS[prefix]}（{detail.strip()}）"
        if _CODE_LIKE_PATTERN.fullmatch(prefix):
            return "未识别阻断原因"
    if raw in BLOCK_REASON_REPLACEMENTS:
        return BLOCK_REASON_REPLACEMENTS[raw]
    return "未识别阻断原因" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw or "未成交"


def signal_type_text(value) -> str:
    raw = str(value or "").strip()
    if raw in SIGNAL_TYPE_REPLACEMENTS:
        return SIGNAL_TYPE_REPLACEMENTS[raw]
    return "未识别信号" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw or "—"


def exit_reason_text(value) -> str:
    raw = str(value or "").strip()
    if raw in EXIT_REASON_REPLACEMENTS:
        return EXIT_REASON_REPLACEMENTS[raw]
    return "其他退出" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw or "其他退出"


def note_text(value) -> str:
    raw = str(value or "")
    return NOTE_REPLACEMENTS.get(raw, raw)


def summary_code_text(value) -> str:
    raw = str(value or "").strip()
    if raw in SUMMARY_CODE_REPLACEMENTS:
        return SUMMARY_CODE_REPLACEMENTS[raw]
    return "未识别策略口径" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw


def semantic_setup_text(value) -> str:
    raw = str(value or "").strip()
    if raw in SEMANTIC_SETUP_REPLACEMENTS:
        return SEMANTIC_SETUP_REPLACEMENTS[raw]
    return "未识别买点" if _CODE_LIKE_PATTERN.fullmatch(raw) else raw


def candidate_failure_reason_text(value) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return ""
    parts = []
    for item in raw.split(";"):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            prefix, detail = part.split(":", 1)
            prefix = prefix.strip()
            if prefix in CANDIDATE_FAILURE_REPLACEMENTS:
                parts.append(f"{CANDIDATE_FAILURE_REPLACEMENTS[prefix]}（{detail.strip()}）")
                continue
            if _CODE_LIKE_PATTERN.fullmatch(prefix):
                parts.append("未识别候选阻断原因")
                continue
        if part in CANDIDATE_FAILURE_REPLACEMENTS:
            parts.append(CANDIDATE_FAILURE_REPLACEMENTS[part])
        elif _CODE_LIKE_PATTERN.fullmatch(part):
            parts.append("未识别候选阻断原因")
        else:
            parts.append(part)
    return "；".join(parts)


def readable_reason(value, event: dict | None = None) -> str:
    parts = []
    action = str((event or {}).get("action") or "")
    left_grid = action.startswith("左侧网格")
    for raw in str(value or "").split(";"):
        part = raw.strip()
        if not part:
            continue
        if _BATCH_PATTERN.fullmatch(part):
            continue
        if part.startswith("止盈条件="):
            conditions = part.split("=", 1)[1].split(",")
            part = "止盈条件：" + "、".join(
                profit_condition_text(condition) for condition in conditions
            )
        part = re.sub(
            r"固定持有(\d+)个交易日",
            r"持有满\1个交易日，按因子周期规则退出",
            part,
        )
        if left_grid:
            part = part.replace("上一格卖出", "上一格网格卖价")
        for source, target in REASON_REPLACEMENTS.items():
            replacement = target
            if left_grid and source == "intraday_cross":
                replacement = "盘中触及预设网格价成交"
            elif left_grid and source == "gap_or_open_fill":
                replacement = "开盘触及预设网格价成交"
            part = part.replace(source, replacement)
        if re.search(r"\b[a-z]+_[a-z_]+\b", part):
            part = "未识别策略条件"
        parts.append(part)
    return "；".join(parts) or "策略条件触发"


def localize_signal_counts(counts: dict | None) -> dict:
    return {
        signal_type_text(key): value
        for key, value in (counts or {}).items()
    }


def localize_event(event: dict) -> dict:
    row = dict(event)
    if "signal_type" in row:
        row["signal_type_cn"] = signal_type_text(row.get("signal_type"))
    if "entry_gate_reason" in row:
        row["entry_gate_reason_cn"] = block_reason_text(row.get("entry_gate_reason"))
    if "semantic_entry_setup" in row:
        row["semantic_entry_setup_cn"] = semantic_setup_text(
            row.get("semantic_entry_setup")
        )
    if "candidate_failure_reason" in row:
        row["candidate_failure_reason_cn"] = candidate_failure_reason_text(
            row.get("candidate_failure_reason")
        )
    if "reason" in row:
        row["reason_cn"] = readable_reason(row.get("reason"), row)
    return row


def localize_entry_block(item: dict) -> dict:
    row = dict(item)
    row["signal_type_cn"] = signal_type_text(row.get("signal_type"))
    row["reason_cn"] = block_reason_text(row.get("reason"))
    row["candidate_failure_reason_cn"] = candidate_failure_reason_text(
        row.get("candidate_failure_reason")
    )
    return row


def localize_breadth_reversal_observation(item: dict) -> dict:
    row = dict(item)
    row["signal_type_cn"] = signal_type_text(
        row.get("entry_channel") or row.get("signal_type")
    )
    row["status_cn"] = block_reason_text(row.get("status"))
    return row
