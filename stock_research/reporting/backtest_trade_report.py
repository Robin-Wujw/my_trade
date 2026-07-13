"""Human-readable Chinese trade ledger for portfolio backtests."""
from __future__ import annotations

import math
import re

import pandas as pd


_REASON_REPLACEMENTS = {
    "intraday_breakout": "盘中突破成交",
    "gap_breakout": "跳空越过触发价成交",
    "gap_or_open_fill": "开盘已触发成交",
    "intraday_cross": "盘中触及结构位成交",
    "intraday_stop": "盘中硬止损成交",
    "gap_stop": "跳空止损成交",
    "condition stop": "收盘条件止损",
    "14:55/close proxy": "14:55收盘代理价成交",
}


def readable_reason(value) -> str:
    parts = []
    for raw in str(value or "").split(";"):
        part = raw.strip()
        if not part or re.fullmatch(r"[Rr]\d+", part):
            continue
        for source, target in _REASON_REPLACEMENTS.items():
            part = part.replace(source, target)
        parts.append(part)
    return "；".join(parts) or "策略条件触发"


def _number(value, digits=2):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


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
        pnl = _number(event.get("profit_loss_amount"))
        rows.append({
            "序号": sequence,
            "日期": str(event.get("date") or ""),
            "股票": str(event.get("name") or code),
            "代码": code,
            "买卖": side,
            "成交价": _number(event.get("execution_price", event.get("price")), 3),
            "数量(股)": quantity,
            "成交金额(元)": _number(event.get("trade_amount")),
            "手续费税费滑点(元)": _number(event.get("transaction_cost_amount")),
            "买卖理由": readable_reason(event.get("reason")),
            "本次已实现盈亏(元)": pnl,
            "本次收益率(%)": _number(event.get("profit_loss_pct"), 4) if side == "卖出" else None,
            "交易后持股(股)": holdings[code],
            "现金变化(元)": _number(event.get("cash_change_amount")),
            "选股理由": str(event.get("selection_reason") or "") if side == "买入" else "",
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
        })
    return pd.DataFrame(rows)


def _money(value) -> str:
    number = _number(value)
    return "—" if number is None else f"¥{number:,.2f}"


def _pct(value) -> str:
    number = _number(value, 4)
    return "—" if number is None else f"{number:+.2f}%"


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
        "",
        "## 每次买卖流水（按时间顺序）",
        "",
    ]
    if ledger.empty:
        lines.append("本区间没有成交。")
    else:
        lines.extend([
            "| # | 日期 | 股票 | 买卖 | 成交 | 数量 | 成交金额 | 成本 | 本次盈亏 | 交易后持股 | 理由 |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in ledger.to_dict("records"):
            reason = str(row["买卖理由"]).replace("|", "/")
            lines.append(
                f"| {row['序号']} | {row['日期']} | {row['股票']}（{row['代码']}） | "
                f"{row['买卖']} | {row['成交价']} | {row['数量(股)']} | "
                f"{_money(row['成交金额(元)'])} | {_money(row['手续费税费滑点(元)'])} | "
                f"{_money(row['本次已实现盈亏(元)'])} | {row['交易后持股(股)']} | {reason} |"
            )
    lines.extend(["", "## 最终持仓", ""])
    positions = result.get("final_positions") or []
    if not positions:
        lines.append("期末空仓。")
    else:
        lines.extend([
            "| 股票 | 数量 | 成本价 | 收盘价 | 投入成本 | 市值 | 浮动盈亏 | 浮盈率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in positions:
            lines.append(
                f"| {item.get('name') or item.get('code')}（{item.get('code')}） | "
                f"{int(float(item.get('quantity') or 0))} | {item.get('cost')} | "
                f"{item.get('close')} | {_money(item.get('invested_amount'))} | "
                f"{_money(item.get('market_value'))} | {_money(item.get('unrealized_pnl_amount'))} | "
                f"{_pct(item.get('unrealized_pnl_pct'))} |"
            )
    lines.extend([
        "",
        "> 说明：买入行的“本次盈亏”只包含当次交易成本；卖出行包含按该次卖出数量分摊的买入成本与全部卖出费用。",
        "",
    ])
    return "\n".join(lines)
