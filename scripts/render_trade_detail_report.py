"""Render a clean Chinese Markdown trade-detail report from a backtest summary."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


SIGNAL_NAMES = {
    "bull_run_half_pullback": "连阳后一半拉回",
    "pullback_50_breakout": "回调一半位向上突破",
    "w_bottom_neckline": "W底颈线突破",
    "uptrend_support_pullback": "上涨波段比例位支撑拉回",
    "consolidation_breakout": "窄幅平台突破",
    "uptrend_50_reclaim": "上涨波段一半位重新收复",
    "gap_long_ma_breakout": "跳空长阳突破",
    "volume_price_node": "量价节点确认",
}

REASON_REPLACEMENTS = {
    "intraday_cross": "盘中触及结构位成交",
    "intraday_breakout": "盘中突破成交",
    "gap_breakout": "跳空越过触发价成交",
    "gap_or_open_fill": "开盘已触发成交",
    "condition stop": "收盘条件触发退出",
    "14:55/close proxy": "收盘代理价成交",
    "gap_stop": "跳空触发退出",
    "intraday_stop": "盘中触发止损",
    "close_confirmed": "收盘确认",
    "profit_floor": "跌破利润底线",
    "trailing_10": "从高点回撤约10%",
    "divergence_time": "背离后多日未修复",
    "maximum_profit_half": "最大浮盈回撤一半",
}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def money(value) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:,.2f}"


def pct(value) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:+.2f}%"


def num(value, digits=2) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:,.{digits}f}"


def clean_reason(text) -> str:
    parts = []
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part or re.fullmatch(r"[Rr]\d+", part):
            continue
        for source, target in REASON_REPLACEMENTS.items():
            part = part.replace(source, target)
        parts.append(part.replace("|", "/"))
    return "；".join(parts) if parts else "-"


def short_basis(event) -> str:
    if str(event.get("trade_side")) != "买入":
        return "-"
    pieces = []
    signal = SIGNAL_NAMES.get(str(event.get("signal_type") or ""))
    if signal:
        pieces.append(signal)
    if event.get("entry_evidence_score") is not None:
        pieces.append(f"证据分{event.get('entry_evidence_score')}")
    ratio = _finite(event.get("structure_ratio"))
    if ratio is not None:
        pieces.append(f"结构比例{ratio * 100:.1f}%")
    trade_basis = str(event.get("trade_basis_reason") or "").strip()
    if trade_basis:
        pieces.append("；".join(trade_basis.split("；")[:3]))
    return "；".join(pieces).replace("|", "/") if pieces else "-"


def render(summary: dict) -> str:
    ledger = summary.get("trade_ledger", [])
    positions = summary.get("final_positions", [])
    by_code = defaultdict(
        lambda: {
            "buy": 0.0,
            "sell": 0.0,
            "realized": 0.0,
        }
    )
    for event in ledger:
        item = by_code[(event.get("code"), event.get("name"))]
        amount = float(event.get("trade_amount") or 0)
        if event.get("trade_side") == "买入":
            item["buy"] += amount
        elif event.get("trade_side") == "卖出":
            item["sell"] += amount
            item["realized"] += float(event.get("profit_loss_amount") or 0)

    position_by_code = {item["code"]: item for item in positions}
    trade_summary = summary.get("trade_summary") or {}
    lines = [
        "# 2026 年内 60% 回测买卖明细",
        "",
        "## 一、结果总览",
        "",
        f"- 回测区间：{summary.get('actual_start')} 至 {summary.get('end_date')}",
        f"- 初始资金：{money(summary.get('initial_capital'))} 元",
        f"- 最终收益率：{pct(summary.get('final_return_pct'))}",
        f"- 已实现收益率：{pct(summary.get('realized_return_pct'))}",
        f"- 未实现收益率：{pct(summary.get('unrealized_return_pct'))}",
        f"- 最大回撤：{pct(summary.get('maximum_drawdown_pct'))}",
        f"- 交易成本：{money(trade_summary.get('transaction_cost_amount'))} 元，占初始资金 {pct(summary.get('transaction_cost_pct'))}",
        f"- 买入/卖出次数：{trade_summary.get('buy_count')} / {trade_summary.get('sell_count')}",
        f"- 已平仓净盈亏：{money(trade_summary.get('closed_trade_net_pnl_amount'))} 元",
        f"- 卖出胜率：{trade_summary.get('sell_win_rate_pct')}%",
        "",
        "> 这次收益主要来自仍持有的趋势仓位，尤其是华峰测控和华工科技；已平仓部分整体为亏损。",
        "",
        "## 二、期末持仓",
        "",
        "| 股票 | 方向 | 持仓占比 | 数量 | 成本 | 期末价 | 浮盈金额 | 浮盈率 | 批次说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for position in positions:
        batches = []
        for batch in position.get("batches") or []:
            batches.append(
                f"{batch.get('batch')}：{int(float(batch.get('quantity') or 0))}股，"
                f"成本{num(batch.get('cost'), 3)}，最大浮盈{num(batch.get('max_return_pct'), 2)}%"
            )
        for batch in position.get("left_batches") or []:
            batches.append(
                f"{batch.get('batch')}：{int(float(batch.get('quantity') or 0))}股，"
                f"成本{num(batch.get('cost'), 3)}"
            )
        lines.append(
            f"| {position.get('name')}({position.get('code')}) | {position.get('position_mode')} | "
            f"{num(position.get('position_pct'), 2)}% | {int(float(position.get('quantity') or 0))} | "
            f"{num(position.get('cost'), 3)} | {num(position.get('close'), 2)} | "
            f"{money(position.get('unrealized_pnl_amount'))} | {pct(position.get('unrealized_pnl_pct'))} | "
            f"{'；'.join(batches) or '-'} |"
        )
    lines.extend([
        "",
        "## 三、按股票汇总",
        "",
        "| 股票 | 买入金额 | 卖出金额 | 已实现盈亏 | 期末浮盈 | 期末持仓 | 说明 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    ranked_codes = sorted(
        by_code.items(),
        key=lambda item: item[1]["realized"]
        + float(position_by_code.get(item[0][0], {}).get("unrealized_pnl_amount") or 0),
        reverse=True,
    )
    for (code, name), item in ranked_codes:
        position = position_by_code.get(code, {})
        unrealized = float(position.get("unrealized_pnl_amount") or 0)
        note = "期末仍持有" if position else "已清仓"
        lines.append(
            f"| {name}({code}) | {money(item['buy'])} | {money(item['sell'])} | "
            f"{money(item['realized'])} | {money(unrealized)} | "
            f"{num(position.get('position_pct'), 2)}% | {note} |"
        )

    lines.extend([
        "",
        "## 四、逐笔买卖流水",
        "",
        "| # | 日期 | 股票 | 买卖 | 成交价 | 数量 | 仓位变化 | 当笔盈亏 | 原因 | 买入依据 |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for index, event in enumerate(ledger, 1):
        side = event.get("trade_side") or event.get("action") or ""
        pnl_value = float(event.get("profit_loss_amount") or 0)
        pnl_text = f"-{money(abs(pnl_value))}" if side == "买入" and pnl_value < 0 else money(pnl_value)
        lines.append(
            f"| {index} | {event.get('date')} | {event.get('name')}({event.get('code')}) | {side} | "
            f"{num(event.get('execution_price', event.get('price')), 3)} | "
            f"{int(float(event.get('quantity') or 0))} | {pct(event.get('position_change_pct'))} | "
            f"{pnl_text} | {clean_reason(event.get('reason'))} | {short_basis(event)} |"
        )

    lines.extend([
        "",
        "## 五、这套 60% 收益的关键观察",
        "",
        "1. 已平仓交易整体亏损，说明这套结果不是靠高胜率短线卖出赚来的。",
        "2. 收益核心来自保留强趋势尾仓：华峰测控期末浮盈约 58.44 万，华工科技期末浮盈约 9.42 万。",
        "3. 止盈份数改成两份后，强趋势仓没有被过早卖碎，这是 60% 收益的主要来源。",
        "4. 支撑拉回没有被默认当首仓；流水里的支撑拉回买入主要是已有仓位后的浮盈加仓。",
        "5. 这份报告使用冻结本地数据和收盘代理价成交，仍需继续做样本外验证。",
        "",
    ])
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
    parser.add_argument("output_md")
    args = parser.parse_args(argv)
    summary_path = Path(args.summary_json)
    output_path = Path(args.output_md)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.write_text(render(summary), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
