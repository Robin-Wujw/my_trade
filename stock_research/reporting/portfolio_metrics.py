"""Original-risk coverage and complete symbol profit attribution."""
from collections import Counter
import pandas as pd
from stock_research.backtesting.values import _finite_float, _split_industry_tags


def _median(values):
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2.0



def _exit_reason_bucket(reason) -> str:
    text = str(reason or "")
    if "hard space stop" in text:
        return "hard_space_stop"
    if "condition stop" in text:
        return "condition_stop"
    if "entry time condition" in text:
        return "entry_time_condition"
    if "profit_floor" in text:
        return "profit_floor"
    if "trailing_10" in text:
        return "trailing_10"
    if "maximum_profit_half" in text:
        return "maximum_profit_half"
    if "volume_node_break" in text:
        return "volume_node_break"
    if "candidate" in text or "落选" in text:
        return "candidate_exit"
    if "汰弱" in text:
        return "weak_rotation"
    return "other"



def _portfolio_r_multiple_summary(events):
    total_sell_count = sum((_finite_float(event.get("execution_quantity")) or 0.) < 0 for event in events or [])
    sells = []
    for event in events or []:
        quantity = _finite_float(event.get("execution_quantity"))
        realized_r = _finite_float(event.get("realized_r_multiple"))
        if quantity is None or quantity >= 0 or realized_r is None:
            continue
        sells.append(event)
    realized_values = [_finite_float(event.get("realized_r_multiple")) for event in sells]
    realized_values = [value for value in realized_values if value is not None]
    wins = [value for value in realized_values if value > 0]
    losses = [value for value in realized_values if value < 0]
    planned_values = [
        _finite_float(event.get("entry_reward_risk"))
        for event in events or []
        if (_finite_float(event.get("execution_quantity")) or 0.0) > 0
        and _finite_float(event.get("entry_reward_risk")) is not None
    ]
    positive_r = sum(value for value in realized_values if value > 0)
    negative_r = -sum(value for value in realized_values if value < 0)
    exit_rows = []
    by_reason = {}
    for event in sells:
        bucket = _exit_reason_bucket(event.get("reason"))
        by_reason.setdefault(bucket, []).append(_finite_float(event.get("realized_r_multiple")))
    for bucket, values in by_reason.items():
        clean = [value for value in values if value is not None]
        if not clean:
            continue
        exit_rows.append({
            "exit_reason": bucket,
            "sell_count": len(clean),
            "average_realized_r": round(sum(clean) / len(clean), 3),
            "median_realized_r": round(float(_median(clean)), 3),
            "net_realized_r": round(sum(clean), 3),
            "loss_beyond_one_r_count": sum(1 for value in clean if value <= -1.0),
        })
    exit_rows.sort(key=lambda item: item["net_realized_r"])
    planned_to_realized = []
    for event in sells:
        planned = _finite_float(event.get("entry_reward_risk"))
        realized = _finite_float(event.get("realized_r_multiple"))
        if planned is not None and planned > 0 and realized is not None:
            planned_to_realized.append(realized / planned * 100.0)
    return {
        "sell_count": len(realized_values),
        "all_sell_count": total_sell_count,
        "missing_initial_r_count": total_sell_count - len(realized_values),
        "r_coverage_pct": len(realized_values) / total_sell_count * 100 if total_sell_count else None,
        "winning_sell_count": len(wins),
        "losing_sell_count": len(losses),
        "sell_win_rate_pct": round(len(wins) / len(realized_values) * 100.0, 3)
        if realized_values else 0.0,
        "average_realized_r": round(sum(realized_values) / len(realized_values), 3)
        if realized_values else None,
        "median_realized_r": (
            round(float(_median(realized_values)), 3) if realized_values else None
        ),
        "average_win_r": round(sum(wins) / len(wins), 3) if wins else None,
        "average_loss_r": round(sum(losses) / len(losses), 3) if losses else None,
        "payoff_r": round((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)), 3)
        if wins and losses else None,
        "profit_factor_r": round(positive_r / negative_r, 3) if negative_r > 0 else None,
        "loss_beyond_one_r_count": sum(1 for value in losses if value <= -1.0),
        "loss_beyond_one_r_pct": round(
            sum(1 for value in losses if value <= -1.0) / len(realized_values) * 100.0,
            3,
        ) if realized_values else 0.0,
        "average_entry_reward_risk": round(sum(planned_values) / len(planned_values), 3)
        if planned_values else None,
        "median_entry_reward_risk": (
            round(float(_median(planned_values)), 3) if planned_values else None
        ),
        "average_realized_to_planned_rr_pct": round(
            sum(planned_to_realized) / len(planned_to_realized),
            3,
        ) if planned_to_realized else None,
        "exit_reason_r_top": exit_rows[:10],
        "note": "R分母为成交时确定并随送股分摊的初始风险；分子为卖出价格交易净损益（含买卖费用），不含单列的现金分红及应收股利。无原始风险定义的批次不纳入R样本。",
    }



def _portfolio_profit_concentration_summary(
    events,
    final_positions,
    *,
    initial_capital,
    final_return_pct,
):
    initial_capital = float(initial_capital)
    final_return_pct = float(final_return_pct or 0.0)
    realized_by_code = Counter()
    receivable_by_code = Counter()
    sell_count_by_code = Counter()
    win_count_by_code = Counter()
    names = {}
    industry_tags_by_code = {}
    for event in events or []:
        code = str(event.get("code") or "").strip()
        if not code:
            continue
        name = str(event.get("name") or "").strip()
        if name:
            names[code] = name
        tags = _split_industry_tags(event.get("industry_tags"))
        if tags:
            industry_tags_by_code.setdefault(code, set()).update(tags)
        quantity = _finite_float(event.get("execution_quantity")) or 0.
        pnl = _finite_float(event.get("profit_loss_amount_raw", event.get("profit_loss_amount")))
        cash = _finite_float(event.get("corporate_action_cash_amount")) or 0.
        receivable = _finite_float(event.get("dividend_receivable_change")) or 0.
        if cash:
            realized_by_code[code] += cash
        if receivable:
            receivable_by_code[code] += receivable
        if quantity < 0 and pnl is not None:
            realized_by_code[code] += pnl
            sell_count_by_code[code] += 1
            if pnl > 0:
                win_count_by_code[code] += 1

    unrealized_by_code = Counter()
    market_value_by_code = Counter()
    for position in final_positions or []:
        code = str(position.get("code") or "").strip()
        if not code:
            continue
        name = str(position.get("name") or "").strip()
        if name:
            names[code] = name
        tags = _split_industry_tags(position.get("industry_tags"))
        if tags:
            industry_tags_by_code.setdefault(code, set()).update(tags)
        unrealized = pd.to_numeric(position.get("unrealized_pnl_amount"), errors="coerce")
        allocations = [lot.get("origin_allocation") for lot in
                       position.get("left_batches", []) + position.get("batches", [])]
        close = _finite_float(position.get("close"))
        if close is not None and allocations and all(allocations):
            unrealized = sum(close * item["quantity"] - item["cost_amount"] - item["entry_fee_amount"]
                             for item in allocations)
        if pd.notna(unrealized):
            unrealized_by_code[code] += float(unrealized)
        market_value = pd.to_numeric(position.get("market_value"), errors="coerce")
        if pd.notna(market_value):
            market_value_by_code[code] += float(market_value)

    codes = sorted(set(realized_by_code) | set(unrealized_by_code) | set(market_value_by_code) | set(receivable_by_code))
    rows = []
    for code in codes:
        realized_pnl = float(realized_by_code.get(code, 0.0))
        unrealized_pnl = float(unrealized_by_code.get(code, 0.0))
        receivable = float(receivable_by_code.get(code, 0.))
        total_pnl = realized_pnl + unrealized_pnl + receivable
        tags = sorted(industry_tags_by_code.get(code, set()))
        rows.append({
            "code": code,
            "name": names.get(code) or code,
            "industry_tags": tags,
            "realized_pnl_amount": round(realized_pnl, 2),
            "unrealized_pnl_amount": round(unrealized_pnl, 2),
            "dividend_receivable_amount": round(receivable, 2),
            "total_pnl_amount": round(total_pnl, 2),
            "total_return_contribution_pct": round(
                total_pnl / initial_capital * 100.0, 3,
            ) if initial_capital > 0 else 0.0,
            "sell_count": int(sell_count_by_code.get(code, 0)),
            "winning_sell_count": int(win_count_by_code.get(code, 0)),
            "final_market_value": round(float(market_value_by_code.get(code, 0.0)), 2),
        })
    rows.sort(key=lambda item: item["total_pnl_amount"], reverse=True)

    positive_total = sum(max(0.0, float(row["total_pnl_amount"])) for row in rows)
    net_total = sum(realized_by_code.values()) + sum(unrealized_by_code.values()) + sum(receivable_by_code.values())
    for row in rows:
        row["positive_profit_share_pct"] = (
            round(max(0.0, float(row["total_pnl_amount"])) / positive_total * 100.0, 3)
            if positive_total > 0 else 0.0
        )
        row["net_profit_share_pct"] = (
            round(float(row["total_pnl_amount"]) / net_total * 100.0, 3)
            if abs(net_total) > 1e-9 else 0.0
        )

    top1 = rows[:1]
    top3 = rows[:3]
    top5 = rows[:5]
    top1_pnl = sum(float(row["total_pnl_amount"]) for row in top1)
    top3_pnl = sum(float(row["total_pnl_amount"]) for row in top3)
    top5_pnl = sum(float(row["total_pnl_amount"]) for row in top5)

    industry_totals = Counter()
    for row in rows:
        tags = row.get("industry_tags") or ["未分类"]
        share = float(row["total_pnl_amount"]) / max(1, len(tags))
        for tag in tags:
            industry_totals[tag] += share
    industry_rows = [
        {
            "industry_tag": tag,
            "total_pnl_amount": round(float(value), 2),
            "total_return_contribution_pct": round(float(value) / initial_capital * 100.0, 3)
            if initial_capital > 0 else 0.0,
            "net_profit_share_pct": round(float(value) / net_total * 100.0, 3)
            if abs(net_total) > 1e-9 else 0.0,
        }
        for tag, value in industry_totals.items()
    ]
    industry_rows.sort(key=lambda item: item["total_pnl_amount"], reverse=True)

    def exclude_return(pnl):
        if initial_capital <= 0:
            return final_return_pct
        return round(final_return_pct - float(pnl) / initial_capital * 100.0, 3)

    top1_positive_share = (
        max(0.0, top1_pnl) / positive_total * 100.0 if positive_total > 0 else 0.0
    )
    top3_positive_share = (
        sum(max(0.0, float(row["total_pnl_amount"])) for row in top3) / positive_total * 100.0
        if positive_total > 0 else 0.0
    )
    return {
        "symbol_count": len(rows),
        "positive_symbol_count": sum(1 for row in rows if float(row["total_pnl_amount"]) > 0),
        "negative_symbol_count": sum(1 for row in rows if float(row["total_pnl_amount"]) < 0),
        "net_total_pnl_amount": round(net_total, 2),
        "net_total_pnl_amount_raw": net_total,
        "positive_total_pnl_amount": round(positive_total, 2),
        "top1_symbol": top1[0] if top1 else None,
        "top1_positive_profit_share_pct": round(top1_positive_share, 3),
        "top3_positive_profit_share_pct": round(top3_positive_share, 3),
        "top1_return_contribution_pct": round(top1_pnl / initial_capital * 100.0, 3)
        if initial_capital > 0 else 0.0,
        "top3_return_contribution_pct": round(top3_pnl / initial_capital * 100.0, 3)
        if initial_capital > 0 else 0.0,
        "top5_return_contribution_pct": round(top5_pnl / initial_capital * 100.0, 3)
        if initial_capital > 0 else 0.0,
        "exclude_top1_approx_final_return_pct": exclude_return(top1_pnl),
        "exclude_top3_approx_final_return_pct": exclude_return(top3_pnl),
        "concentration_warning": bool(top1_positive_share >= 50.0 or top3_positive_share >= 80.0),
        "concentration_warning_reason": (
            "top1_or_top3_profit_concentration"
            if top1_positive_share >= 50.0 or top3_positive_share >= 80.0 else ""
        ),
        "top_symbols": rows[:10],
        "industry_contribution_top": industry_rows[:10],
        "note": (
            "exclude_topN is a contribution subtraction diagnostic, not a cash-rebalanced rerun."
        ),
    }

