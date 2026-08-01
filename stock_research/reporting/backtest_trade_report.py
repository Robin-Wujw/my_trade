"""Human-readable Chinese trade ledger for portfolio backtests."""
from __future__ import annotations

import math
import re

import pandas as pd


_REASON_REPLACEMENTS = {
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
    "14:55/close proxy": "14:55收盘代理价成交",
}

_PROFIT_CONDITION_REPLACEMENTS = {
    "profit_floor": "利润保护线",
    "trailing_10": "从阶段高点回撤10%",
    "rising_ma20_break_3pct": "跌破上行MA20 3%",
}

_EXIT_REASON_REPLACEMENTS = {
    "condition_stop": "收盘条件止损",
    "hard_space_stop": "空间硬止损",
    "profit_floor": "利润保护线止盈",
    "entry_time_condition": "入场时间条件退出",
    "other": "其他退出",
    "trailing_10": "从阶段高点回撤10%止盈",
    "volume_node_break": "跌破量价节点止盈",
}

_BLOCK_REASON_REPLACEMENTS = {
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
    "total_symbol_limit": "全部持股已达5只上限",
    "industry_or_correlation_limit": "同行业或高相关标的数量已达上限",
    "new_symbol_unlock_not_met": "新标的开仓解锁条件未满足",
    "weak_replacement_scheduled": "已安排下一交易日汰弱换强",
    "symbol_limit_no_weak_replacement": "仓位已满且没有可替换弱仓",
    "symbol_exposure_cap": "单股仓位已达上限",
    "gross_exposure_cap": "组合总仓位已达上限",
    "cash_or_board_lot_limit": "现金不足或不足一手",
    "left_to_right_management_only_without_current_candidate": "非当日候选，仅执行左转右管理状态检查",
    "left_to_right_add_on_cash_or_board_lot_limit": "左转右已完成，但现金不足或附加仓不足一手",
}

_SIGNAL_TYPE_REPLACEMENTS = {
    "bull_run_half_pullback": "连阳一半拉回",
    "consolidation_breakout": "整理平台突破",
    "gap_long_ma_breakout": "跳空长阳突破长期均线",
    "leader_pivot_breakout": "龙头枢轴突破",
    "leader_trend_add": "龙头趋势加仓",
    "uptrend_support_pullback": "上涨波段支撑拉回",
    "pullback_50_breakout": "回调50%放量突破",
    "uptrend_50_reclaim": "上涨50%收复",
    "volume_price_node": "量价节点突破",
}

_NOTE_REPLACEMENTS = {
    "exclude_topN is a contribution subtraction diagnostic, not a cash-rebalanced rerun.": (
        "剔除头部贡献只是从收益贡献中做减法诊断，并非重新按现金再平衡回测。"
    ),
    "R is measured from each fill's raw entry price to its initial raw stop.": (
        "R按每笔成交的原始入场价到初始原始止损价之间的风险距离计算。"
    ),
}


def _fmt_number(value, digits=3) -> str:
    number = _number(value, digits)
    return "" if number is None else f"{number:,.{digits}f}"


def _profit_condition_text(value) -> str:
    raw = str(value or "").strip()
    if raw.startswith("volume_node_break:"):
        date = raw.split(":", 1)[1]
        return f"跌破量价节点（节点日{date}）"
    return _PROFIT_CONDITION_REPLACEMENTS.get(raw, raw)


def readable_reason(value, event: dict | None = None) -> str:
    parts = []
    action = str((event or {}).get("action") or "")
    left_grid = action.startswith("左侧网格")
    for raw in str(value or "").split(";"):
        part = raw.strip()
        if not part or re.fullmatch(r"[Rr]\d+", part):
            continue
        if part.startswith("止盈条件="):
            conditions = part.split("=", 1)[1].split(",")
            part = "止盈条件：" + "、".join(
                _profit_condition_text(condition) for condition in conditions
            )
        if left_grid:
            part = part.replace("上一格卖出", "上一格网格卖价")
        for source, target in _REASON_REPLACEMENTS.items():
            replacement = target
            if left_grid and source == "intraday_cross":
                replacement = "盘中触及预设网格价成交"
            elif left_grid and source == "gap_or_open_fill":
                replacement = "开盘触及预设网格价成交"
            part = part.replace(source, replacement)
        parts.append(part)
    if left_grid and event:
        details = []
        value_line = _fmt_number(event.get("value_line"))
        if value_line:
            details.append(f"价值线{value_line}")
        grid_slot = _number(event.get("grid_slot"), 0)
        if grid_slot is not None:
            details.append(f"网格槽{int(grid_slot)}")
        if details:
            parts.append("，".join(details))
    return "；".join(parts) or "策略条件触发"


def _number(value, digits=2):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _signed_money(value) -> str:
    number = _number(value)
    if number is None:
        return ""
    return f"{number:+,.2f}元"


def _entry_cost_per_share(event: dict):
    cost = _number(event.get("cost_basis"), 3)
    if cost is not None:
        return cost
    price = _number(event.get("execution_price", event.get("price")), 3)
    if price is not None:
        return price
    amount = _number(event.get("trade_amount"), 6)
    quantity = _number(event.get("quantity"), 6)
    if amount is None or quantity in {None, 0.0}:
        return None
    return _number(amount / quantity, 3)


def _action_summary(event: dict, side: str, quantity: int, holdings_after: int) -> str:
    name = str(event.get("name") or event.get("code") or "")
    code = str(event.get("code") or "")
    price = _number(event.get("execution_price", event.get("price")), 3)
    amount = _number(event.get("trade_amount"))
    amount_text = "" if amount is None else f"{amount:,.2f}元"
    cash = _signed_money(event.get("cash_change_amount"))
    if side == "买入":
        return (
            f"买入{name}({code}) {quantity}股，成交价{price}，"
            f"成交金额{amount_text}，现金{cash}，交易后持股{holdings_after}股"
        )
    pnl = _number(event.get("profit_loss_amount"))
    pct = _number(event.get("profit_loss_pct"), 4)
    result = "盈利" if pnl is not None and pnl >= 0 else "亏损"
    position = "清仓" if holdings_after <= 0 else f"剩余{holdings_after}股"
    pct_text = "" if pct is None else f"，收益率{pct:+.2f}%"
    return (
        f"卖出{name}({code}) {quantity}股，成交价{price}，"
        f"成交金额{amount_text}，现金{cash}，本次{result}{_signed_money(pnl)}"
        f"{pct_text}，{position}"
    )


def _trade_result(side: str, event: dict) -> str:
    if side == "买入":
        cost = _entry_cost_per_share(event)
        fee = _number(event.get("transaction_cost_amount"))
        risk = _number(event.get("initial_risk_per_share_raw"), 3)
        rr = _number(event.get("entry_reward_risk"), 3)
        risk_text = ""
        if risk is not None:
            risk_text = f"；初始风险{risk:,.3f}元/股"
        if rr is not None:
            risk_text += f"；计划{rr:,.2f}R"
        if cost is None and fee is None:
            return "建仓" + risk_text
        if cost is None:
            return f"建仓，费用{fee:,.2f}元{risk_text}"
        if fee is None:
            return f"成本价{cost:,.3f}元/股{risk_text}"
        return f"成本价{cost:,.3f}元/股；费用{fee:,.2f}元{risk_text}"
    pnl = _number(event.get("profit_loss_amount"))
    pct = _number(event.get("profit_loss_pct"), 4)
    if pnl is None:
        return "平仓"
    label = "盈利" if pnl >= 0 else "亏损"
    pct_text = "" if pct is None else f" ({pct:+.2f}%)"
    realized_r = _number(event.get("realized_r_multiple"), 3)
    r_text = "" if realized_r is None else f"；实现{realized_r:+.2f}R"
    return f"{label}{pnl:,.2f}元{pct_text}{r_text}"


def build_readable_trade_frame(trade_ledger) -> pd.DataFrame:
    """Convert the engine ledger to a compact chronological buy/sell statement."""
    rows = []
    holdings: dict[str, int] = {}
    for sequence, event in enumerate(trade_ledger or [], 1):
        code = str(event.get("code") or "")
        side = str(event.get("trade_side") or event.get("action") or "")
        quantity = int(round(float(event.get("quantity") or 0)))
        signed = quantity if side == "买入" else -quantity
        holdings[code] = holdings.get(code, 0) + signed
        holdings_after = holdings[code]
        pnl = _number(event.get("profit_loss_amount"))
        rows.append({
            "序号": sequence,
            "日期": str(event.get("date") or ""),
            "股票": str(event.get("name") or code),
            "代码": code,
            "买卖": side,
            "操作摘要": _action_summary(event, side, quantity, holdings_after),
            "交易结果": _trade_result(side, event),
            "成交价": _number(event.get("execution_price", event.get("price")), 3),
            "数量(股)": quantity,
            "成交金额(元)": _number(event.get("trade_amount")),
            "手续费税费滑点(元)": _number(event.get("transaction_cost_amount")),
            "买卖理由": readable_reason(event.get("reason"), event),
            "本次已实现盈亏(元)": pnl,
            "本次收益率(%)": _number(event.get("profit_loss_pct"), 4) if side == "卖出" else None,
            "实现R": _number(event.get("realized_r_multiple"), 3) if side == "卖出" else None,
            "计划R": _number(event.get("entry_reward_risk"), 3),
            "初始风险(元/股)": _number(event.get("initial_risk_per_share_raw"), 6),
            "初始风险金额(元)": _number(event.get("initial_risk_amount")),
            "交易后持股(股)": holdings_after,
            "持仓状态": "清仓" if holdings_after <= 0 else f"持有{holdings_after}股",
            "现金变化(元)": _number(event.get("cash_change_amount")),
            "选股理由": str(event.get("selection_reason") or "") if side == "买入" else "",
            "买卖依据对照": str(event.get("trade_basis_reason") or event.get("selection_reason") or ""),
            "结构比例": _number(event.get("structure_ratio"), 4),
            "锚点低/高": (
                f"{event.get('anchor_low')} / {event.get('anchor_high')}"
                if event.get("anchor_low") is not None and event.get("anchor_high") is not None
                else ""
            ),
            "锚点日期": (
                f"{event.get('anchor_low_date')} → {event.get('anchor_high_date')}"
                if event.get("anchor_low_date") and event.get("anchor_high_date")
                else ""
            ),
            "账户模式": str(event.get("account_mode") or ""),
            "价值线": _number(event.get("value_line"), 3),
            "网格槽": _number(event.get("grid_slot"), 0),
        })
    return pd.DataFrame(rows)


def _money(value) -> str:
    number = _number(value)
    return "—" if number is None else f"¥{number:,.2f}"


def _pct(value) -> str:
    number = _number(value, 4)
    return "—" if number is None else f"{number:+.2f}%"


def _bool_text(value) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "—"


def _mode_text(value) -> str:
    mapping = {"left": "左侧", "right": "右侧", "mixed": "左右混合"}
    return mapping.get(str(value or ""), str(value or "—"))


def _block_reason_text(value) -> str:
    raw = str(value or "")
    if ":" in raw:
        prefix, detail = raw.split(":", 1)
        if prefix in _BLOCK_REASON_REPLACEMENTS:
            return f"{_BLOCK_REASON_REPLACEMENTS[prefix]}（{detail}）"
    return _BLOCK_REASON_REPLACEMENTS.get(raw, raw or "未成交")


def _signal_type_text(value) -> str:
    raw = str(value or "")
    return _SIGNAL_TYPE_REPLACEMENTS.get(raw, raw or "—")


def _exit_reason_text(value) -> str:
    raw = str(value or "")
    return _EXIT_REASON_REPLACEMENTS.get(raw, raw or "其他退出")


def _note_text(value) -> str:
    raw = str(value or "")
    return _NOTE_REPLACEMENTS.get(raw, raw)


def _table_number(value, digits=2):
    number = _number(value, digits)
    return "—" if number is None else number


def _render_profit_concentration(result: dict) -> list[str]:
    summary = result.get("profit_concentration_summary") or {}
    if not summary:
        return []
    top1 = summary.get("top1_symbol") or {}
    lines = [
        "",
        "## 收益集中度诊断",
        "",
        f"- 贡献股票数：{summary.get('symbol_count', 0)}；盈利股票数：{summary.get('positive_symbol_count', 0)}；亏损股票数：{summary.get('negative_symbol_count', 0)}",
        f"- Top1 正贡献占比：{_pct(summary.get('top1_positive_profit_share_pct'))}；Top3 正贡献占比：{_pct(summary.get('top3_positive_profit_share_pct'))}",
        f"- Top1 收益贡献：{_pct(summary.get('top1_return_contribution_pct'))}；Top3 收益贡献：{_pct(summary.get('top3_return_contribution_pct'))}",
        f"- 剔除 Top1 后近似收益：{_pct(summary.get('exclude_top1_approx_final_return_pct'))}；剔除 Top3 后近似收益：{_pct(summary.get('exclude_top3_approx_final_return_pct'))}",
    ]
    if top1:
        lines.append(
            f"- 第一贡献：{top1.get('name') or top1.get('code')}({top1.get('code')})，"
            f"合计贡献{_money(top1.get('total_pnl_amount'))}，"
            f"正贡献占比{_pct(top1.get('positive_profit_share_pct'))}"
        )
    if summary.get("concentration_warning"):
        lines.append("- 集中度警告：收益明显依赖少数股票，必须做剔除头部贡献后的稳健性复核。")
    lines.extend([
        "",
        "| 股票 | 行业/主题 | 已实现 | 期末浮盈 | 合计贡献 | 正贡献占比 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in (summary.get("top_symbols") or [])[:10]:
        tags = "、".join(row.get("industry_tags") or []) or "未分类"
        lines.append(
            f"| {row.get('name') or row.get('code')}({row.get('code')}) | {tags} | "
            f"{_money(row.get('realized_pnl_amount'))} | "
            f"{_money(row.get('unrealized_pnl_amount'))} | "
            f"{_money(row.get('total_pnl_amount'))} | "
            f"{_pct(row.get('positive_profit_share_pct'))} |"
        )
    industries = summary.get("industry_contribution_top") or []
    if industries:
        lines.extend([
            "",
            "| 行业/主题 | 合计贡献 | 收益贡献 | 净利润占比 |",
            "|---|---:|---:|---:|",
        ])
        for row in industries[:10]:
            lines.append(
                f"| {row.get('industry_tag')} | {_money(row.get('total_pnl_amount'))} | "
                f"{_pct(row.get('total_return_contribution_pct'))} | "
                f"{_pct(row.get('net_profit_share_pct'))} |"
            )
    note = summary.get("note")
    if note:
        lines.extend(["", f"> {_note_text(note)}"])
    return lines


def _render_r_multiple_audit(result: dict) -> list[str]:
    summary = result.get("r_multiple_summary") or {}
    if not summary:
        return []
    lines = [
        "",
        "## R 倍数审计",
        "",
        f"- 计划入场盈亏比：均值{_number(summary.get('average_entry_reward_risk'), 2) or '—'}R；中位数{_number(summary.get('median_entry_reward_risk'), 2) or '—'}R",
        f"- 实际卖出 R：均值{_number(summary.get('average_realized_r'), 2) or '—'}R；中位数{_number(summary.get('median_realized_r'), 2) or '—'}R",
        f"- 胜率：{_pct(summary.get('sell_win_rate_pct'))}；赢家均值{_number(summary.get('average_win_r'), 2) or '—'}R；输家均值{_number(summary.get('average_loss_r'), 2) or '—'}R",
        f"- R 盈亏比：{_number(summary.get('payoff_r'), 2) or '—'}；R 利润因子：{_number(summary.get('profit_factor_r'), 2) or '—'}",
        f"- 亏损超过 1R：{summary.get('loss_beyond_one_r_count', 0)} 次，占全部可审计卖出 {_pct(summary.get('loss_beyond_one_r_pct'))}",
    ]
    realized_to_planned = summary.get("average_realized_to_planned_rr_pct")
    if realized_to_planned is not None:
        lines.append(f"- 实际/计划 R 均值：{_pct(realized_to_planned)}")
    rows = summary.get("exit_reason_r_top") or []
    if rows:
        lines.extend([
            "",
            "| 退出原因 | 次数 | 平均R | 中位R | 净R | 超过1R亏损 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {_exit_reason_text(row.get('exit_reason'))} | {row.get('sell_count', 0)} | "
                f"{_number(row.get('average_realized_r'), 2) or '—'} | "
                f"{_number(row.get('median_realized_r'), 2) or '—'} | "
                f"{_number(row.get('net_realized_r'), 2) or '—'} | "
                f"{row.get('loss_beyond_one_r_count', 0)} |"
            )
    note = summary.get("note")
    if note:
        lines.extend(["", f"> {_note_text(note)}"])
    return lines


def _render_relevant_entry_blocks(result: dict, *, max_rows_per_code: int = 20) -> list[str]:
    positions = result.get("final_positions") or []
    held_codes = {str(item.get("code") or "") for item in positions}
    held_codes.discard("")
    if not held_codes:
        return []
    blocks = [
        item for item in result.get("entry_blocks") or []
        if str(item.get("code") or "") in held_codes
    ]
    if not blocks:
        return []
    grouped: dict[str, list[dict]] = {}
    names: dict[str, str] = {
        str(item.get("code") or ""): str(item.get("name") or item.get("code") or "")
        for item in positions
    }
    for item in blocks:
        code = str(item.get("code") or "")
        grouped.setdefault(code, []).append(item)
        block_name = str(item.get("name") or "")
        if block_name and block_name != code:
            names[code] = block_name
    last_trade_dates: dict[str, str] = {}
    for event in result.get("trade_ledger") or []:
        code = str(event.get("code") or "")
        if code in held_codes and event.get("date"):
            last_trade_dates[code] = max(last_trade_dates.get(code, ""), str(event.get("date")))
    lines = [
        "",
        "## 期末持仓相关未成交信号",
        "",
        "| 股票 | 日期 | 信号 | 旧证据分 | 交易分 | 领导力 | 未成交/未左转右原因 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for code in sorted(grouped):
        all_items = sorted(grouped[code], key=lambda row: str(row.get("date") or ""))
        last_trade_date = last_trade_dates.get(code, "")
        after_last_trade = [
            item for item in all_items
            if str(item.get("date") or "") > last_trade_date
        ]
        head_count = max_rows_per_code // 2
        selected = after_last_trade[:head_count] + all_items[-(max_rows_per_code - head_count):]
        seen = set()
        items = []
        for item in sorted(selected, key=lambda row: str(row.get("date") or "")):
            key = (
                str(item.get("date") or ""),
                str(item.get("signal_type") or ""),
                str(item.get("reason") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
        overflow = max(0, len(all_items) - len(items))
        if overflow:
            lines.append(
                f"| {names.get(code, code)}（{code}） | — | — | — | — | — | "
                f"另有{overflow}条同标的未成交信号，详见summary.entry_blocks |"
            )
        for item in items:
            lines.append(
                f"| {names.get(code, code)}（{code}） | {item.get('date') or ''} | "
                f"{_signal_type_text(item.get('signal_type'))} | "
                f"{_table_number(item.get('entry_evidence_score'), 2)} | "
                f"{_table_number(item.get('trade_basis_score'), 2)} | "
                f"{_table_number(item.get('leadership_score'), 2)} | "
                f"{_block_reason_text(item.get('reason'))} |"
            )
    return lines


def render_trade_report_markdown(result: dict) -> str:
    """Render summary, chronological trades and final holdings as Markdown."""
    ledger = build_readable_trade_frame(result.get("trade_ledger") or [])
    trade_summary = result.get("trade_summary") or {}
    lines = [
        "# 组合回测买卖报告",
        "",
        "## 回测结果",
        "",
        f"- 区间：{result.get('actual_start') or result.get('requested_start')} 至 {result.get('end_date')}",
        f"- 初始资金：{_money(result.get('initial_capital'))}",
        f"- 期末现金：{_money(result.get('final_cash'))}",
        f"- 最终收益：{_pct(result.get('final_return_pct'))}",
        f"- 最大回撤：{_pct(result.get('maximum_drawdown_pct'))}",
        f"- 买入/卖出次数：{trade_summary.get('buy_count', 0)} / {trade_summary.get('sell_count', 0)}",
        f"- 已平仓净盈亏：{_money(trade_summary.get('closed_trade_net_pnl_amount'))}",
        f"- 全部交易成本：{_money(trade_summary.get('transaction_cost_amount'))}",
    ]
    lines.extend(_render_profit_concentration(result))
    lines.extend(_render_r_multiple_audit(result))
    lines.extend([
        "",
        "## 每次买卖流水（按时间顺序）",
        "",
    ])
    if ledger.empty:
        lines.append("本区间没有成交。")
    else:
        lines.extend([
            "| # | 日期 | 买卖动作 | 结果 | 理由 |",
            "|---:|---|---|---|---|",
        ])
        for row in ledger.to_dict("records"):
            reason = str(row["买卖理由"]).replace("|", "/")
            action = str(row["操作摘要"]).replace("|", "/")
            result_text = str(row["交易结果"]).replace("|", "/")
            lines.append(
                f"| {row['序号']} | {row['日期']} | {action} | {result_text} | {reason} |"
            )
    lines.extend(["", "## 最终持仓", ""])
    positions = result.get("final_positions") or []
    if not positions:
        lines.append("期末空仓。")
    else:
        lines.extend([
            "| 股票 | 模式 | 数量 | 成本价 | 收盘价 | 价值线 | 市值 | 浮动盈亏 | 浮盈率 | 高收益尾仓 | 计入容量 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for item in positions:
            lines.append(
                f"| {item.get('name') or item.get('code')}（{item.get('code')}） | "
                f"{_mode_text(item.get('position_mode'))} | "
                f"{int(float(item.get('quantity') or 0))} | {item.get('cost')} | "
                f"{item.get('close')} | {_fmt_number(item.get('left_value_line')) or '—'} | "
                f"{_money(item.get('market_value'))} | {_money(item.get('unrealized_pnl_amount'))} | "
                f"{_pct(item.get('unrealized_pnl_pct'))} | "
                f"{_bool_text(item.get('profit_tail'))} | {_bool_text(item.get('capacity_counted'))} |"
            )
    lines.extend(_render_relevant_entry_blocks(result))
    lines.extend([
        "",
        "> 说明：买入行的结果展示成本价/股和当次费用，“本次已实现盈亏”只包含当次交易成本；卖出行包含按该次卖出数量分摊的买入成本与全部卖出费用。",
        "",
    ])
    return "\n".join(lines)
