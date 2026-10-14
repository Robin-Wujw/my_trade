"""Human-readable Chinese trade ledger for portfolio backtests."""
from __future__ import annotations

import math
import json

import pandas as pd

from stock_research.reporting.signal_text import (
    exit_reason_text,
    note_text,
    readable_reason as _readable_signal_reason,
    signal_type_text,
)


def _fmt_number(value, digits=3) -> str:
    number = _number(value, digits)
    return "" if number is None else f"{number:,.{digits}f}"


def _qualification_items(values) -> str:
    rendered = []
    for value in values or []:
        if isinstance(value, dict):
            rendered.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        else:
            rendered.append(str(value))
    return ", ".join(rendered) or "未列明"


def apply_stock_basic_display_names(
    result: dict,
    stock_basic: pd.DataFrame,
) -> dict:
    """Attach current Chinese names and missing industry labels after decisions."""
    def display_text(value) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "<na>"} else text

    names = {}
    industries = {}
    if stock_basic is not None and not stock_basic.empty:
        for _, row in stock_basic.iterrows():
            code = display_text(row.get("code"))
            name = display_text(row.get("name"))
            if code and name and name != code:
                names[code] = name
            industry = display_text(row.get("industry"))
            if code and industry:
                industries[code] = industry

    def update_item(item):
        if not isinstance(item, dict):
            return
        code = display_text(item.get("code"))
        if code in names:
            item["name"] = names[code]
        replacement_code = display_text(item.get("replacement_code"))
        if replacement_code in names:
            item["replacement_name"] = names[replacement_code]

    def backfill_industry(item):
        if not isinstance(item, dict) or item.get("industry_tags"):
            return
        code = display_text(item.get("code"))
        if code in industries:
            item["display_industry_tags"] = [industries[code]]

    for key in ("events", "entry_blocks"):
        for item in result.get(key) or []:
            update_item(item)
    for key in ("trade_ledger", "final_positions", "concentration_blocks"):
        for item in result.get(key) or []:
            update_item(item)
            backfill_industry(item)

    concentration = result.get("profit_concentration_summary") or {}
    update_item(concentration.get("top1_symbol"))
    backfill_industry(concentration.get("top1_symbol"))
    for item in concentration.get("top_symbols") or []:
        update_item(item)
        backfill_industry(item)

    traded_codes = {
        display_text(item.get("code"))
        for item in result.get("trade_ledger") or []
        if display_text(item.get("code"))
    }
    result["display_name_source"] = {
        "source": "local_tushare_stock_basic_current_snapshot",
        "decision_input": False,
        "scope": "artifact_display_only",
        "resolved_traded_symbol_count": len(traded_codes & set(names)),
        "unresolved_traded_codes": sorted(traded_codes - set(names)),
        "resolved_traded_industry_count": len(traded_codes & set(industries)),
        "unresolved_industry_traded_codes": sorted(traded_codes - set(industries)),
        "note": (
            "Current security names and missing industry labels are attached "
            "only after the backtest; historical selection, industry decisions "
            "and ST filters remain point-in-time inputs."
        ),
    }
    return result


def readable_reason(value, event: dict | None = None) -> str:
    text = _readable_signal_reason(value, event)
    if str((event or {}).get("action") or "").startswith("左侧网格") and event:
        parts = [text] if text else []
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
    return text


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
            "信号类型": signal_type_text(event.get("signal_type")),
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
            "账户模式": (
                _mode_text(event.get("account_mode"))
                if event.get("account_mode")
                else ""
            ),
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


def _exit_reason_text(value) -> str:
    return exit_reason_text(value)


def _note_text(value) -> str:
    return note_text(value)


def _display_industry_tags(row: dict) -> list[str]:
    tags = row.get("display_industry_tags")
    if not tags:
        tags = row.get("industry_tags")
    cleaned = []
    for tag in tags or []:
        text = str(tag).strip()
        if text and text.lower() not in {"nan", "none", "<na>", "未分类"}:
            cleaned.append(text)
    return cleaned


def _display_industry_contribution(rows, initial_capital) -> list[dict]:
    totals: dict[str, float] = {}
    for row in rows or []:
        tags = _display_industry_tags(row) or ["未映射"]
        share = float(row.get("total_pnl_amount") or 0.0) / len(tags)
        for tag in tags:
            totals[tag] = totals.get(tag, 0.0) + share
    net_total = sum(float(row.get("total_pnl_amount") or 0.0) for row in rows or [])
    initial_capital = float(initial_capital or 0.0)
    result_rows = []
    for tag, value in totals.items():
        result_rows.append({
            "industry_tag": tag,
            "total_pnl_amount": round(value, 2),
            "total_return_contribution_pct": (
                round(value / initial_capital * 100.0, 3)
                if initial_capital > 0 else 0.0
            ),
            "net_profit_share_pct": (
                round(value / net_total * 100.0, 3)
                if abs(net_total) > 1e-9 else 0.0
            ),
        })
    result_rows.sort(key=lambda item: item["total_pnl_amount"], reverse=True)
    return result_rows


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
        "| 股票 | 行业/主题 | 已实现 | 期末浮盈 | 应收股利 | 合计贡献 | 正贡献占比 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    top_rows = (summary.get("top_symbols") or [])[:10]
    for row in top_rows:
        tags = "、".join(_display_industry_tags(row)) or "未映射"
        lines.append(
            f"| {row.get('name') or row.get('code')}({row.get('code')}) | {tags} | "
            f"{_money(row.get('realized_pnl_amount'))} | "
            f"{_money(row.get('unrealized_pnl_amount'))} | "
            f"{_money(row.get('dividend_receivable_amount'))} | "
            f"{_money(row.get('total_pnl_amount'))} | "
            f"{_pct(row.get('positive_profit_share_pct'))} |"
        )
    industries = (
        _display_industry_contribution(
            top_rows, result.get("initial_capital"),
        )
        if top_rows
        else (summary.get("industry_contribution_top") or [])
    )
    if industries:
        lines.extend([
            "",
            "以下行业汇总仅覆盖上表前十贡献股票；已实现收益含已到账分红，应收股利单列。",
            "",
            "| 行业/主题 | 合计贡献 | 收益贡献 | 上述股票净利润占比 |",
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
    if summary.get("all_sell_count") is not None:
        lines.append(f"- 原始风险覆盖：{summary.get('sell_count', 0)}/{summary['all_sell_count']} 笔卖出（{_pct(summary.get('r_coverage_pct'))}）；缺少定义的 {summary.get('missing_initial_r_count', 0)} 笔不补造R。")
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


def _render_channel_concentration(result):
    if "events" not in result and not result.get("capital_risk_attribution"):
        return []
    lines = ["", "## 原始左右来源的收益集中度", ""]
    if not isinstance(result.get("events"), list):
        return lines + ["缺少完整事件账本，无法计算左右来源集中度；旧卖出流水没有包含全部股息及权益事件。"]
    from stock_research.reporting.portfolio_channel_concentration import build_channel_contribution

    report = build_channel_contribution(result["events"], result.get("final_positions"),
                                        initial_capital=result["initial_capital"])
    lines += ["左转右仍按最初左侧买入来源统计；已实现净利润含分摊买入费、卖出成本及实际到账股息。",
              "", "| 最初来源 | 已实现净利润 | Top1占比 | Top3占比 | 静态扣Top3后利润 | 未结算股票数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for channel in report["by_channel"]:
        first, third = channel["top_stock_concentration"]
        origin = "左侧（含转右）" if channel["origin_channel"] == "left" else "原生右侧"
        first_share, third_share = first["share_of_net_realized_profit"], third["share_of_net_realized_profit"]
        lines.append(f"| {origin} | {_money(channel['realized_net_profit'])} | "
                     f"{_pct(first_share * 100) if first_share is not None else '不可计算'} | "
                     f"{_pct(third_share * 100) if third_share is not None else '不可计算'} | "
                     f"{_money(third['remaining_static_realized_profit'])} | {channel['unsettled_stock_count']} |")
    lines += ["", "静态扣除股票利润不重跑资金、订单竞争及仓位，不是可交易反事实收益；共享资金归因不能证明通道的独立因果增益。浮盈、待上市股数和未到账股息不计入已实现净利润。",
              f"归因覆盖：{report['attribution_coverage_status']}；排除成交/现金记录 {report['coverage']['excluded_cash_or_trade_rows']} 条，排除终仓记录 {report['coverage']['excluded_final_position_rows']} 条。",
              "历史财务修订、行业快照和事后选择限制继续保留；缺来源不视为零收益。"]
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
    qualification = result.get("qualification") or {}
    if result.get("research_only") or result.get("acceptance_status") not in {None, "passed"}:
        lines.extend([
            "",
            "## 研究资格与数据限制",
            "",
            f"- 验收状态：{result.get('acceptance_status') or 'research_only'}",
            f"- research_only：{bool(result.get('research_only'))}",
            f"- 数据限制：{_qualification_items(result.get('data_limitations') or qualification.get('data_limitations') or ['未列明'])}",
            f"- 原因：{_qualification_items(qualification.get('reasons') or ['研究回放'])}",
        ])
    attribution = result.get("capital_risk_attribution") or {}
    if attribution:
        lines.extend([
            "", "## 资金利用与收益来源", "",
            f"- 年化复合收益：{_pct(attribution.get('cagr_pct'))}",
            f"- 平均 / 中位市值仓位：{_pct(attribution.get('mean_market_exposure_pct'))} / {_pct(attribution.get('median_market_exposure_pct'))}",
            f"- 空仓天数：{attribution.get('flat_days', 0)} / {attribution.get('trading_days', 0)}；市值仓位低于30%：{attribution.get('below_30pct_exposure_days', 0)}天",
            "- 成本敞口以初始本金为分母；市值仓位以当日净值为分母。新买入预算仍按初始本金计算。",
            "", "| 最初来源 | 已平仓净盈亏 | 未平仓净浮盈 | 现金分红 | 收益贡献 |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in attribution.get("by_origin", []):
            origin = {"left": "左侧（含转右）", "right": "原生右侧", "unknown": "来源缺失"}.get(row["origin_account"], row["origin_account"])
            lines.append(f"| {origin} | {_money(row['realized_net_pnl'])} | {_money(row['unrealized_net_pnl'])} | {_money(row['dividend_cash'])} | {_pct(row['return_contribution_pct'])} |")
        lines.extend([
            "", f"> {attribution.get('note', '')}",
            f"> 来源利润对账差额：{_money(attribution.get('origin_pnl_reconciliation_difference'))}；现金对账差额：{_money(attribution.get('cash_reconciliation_difference'))}。分批卖出的R统计不等于独立交易周期胜率。",
        ])
    lines.extend(_render_channel_concentration(result))
    lines.extend(_render_profit_concentration(result))
    lines.extend(_render_r_multiple_audit(result))
    factor_holding_days = result.get("factor_horizon_holding_days")
    if factor_holding_days:
        trend_management_return = result.get(
            "factor_horizon_trend_management_return_pct"
        )
        lines.extend([
            "",
            "## 成交口径说明",
            "",
            (
                (
                    f"- 因子周期仓历史最大浮盈达到{_pct(trend_management_return)}后，"
                    "取消固定到期退出，转入统一分仓止盈和趋势尾仓管理；"
                    f"未达到时仍以{int(factor_holding_days)}个交易日为研究周期上限。"
                )
                if trend_management_return is not None
                else (
                    f"- 因子周期交易持有满{int(factor_holding_days)}个交易日后按规则退出；"
                    "这不是止盈或止损信号。"
                )
            ),
            (
                "- 按当日收盘价近似成交：日K回测没有分钟级委托和排队数据，"
                "因此用当日收盘价近似收盘前确认成交，不代表真实盘中成交。"
            ),
        ])
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
    lines.extend([
        "",
        "> 说明：买入行的结果展示成本价/股和当次费用，“本次已实现盈亏”只包含当次交易成本；卖出行包含按该次卖出数量分摊的买入成本与全部卖出费用。",
        "",
    ])
    return "\n".join(lines)
