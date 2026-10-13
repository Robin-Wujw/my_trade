"""Cash, market exposure and original-entry attribution from executed lots."""
from collections import defaultdict

import pandas as pd


def build_portfolio_attribution(events, equity_curve, final_positions, initial_capital):
    capital = float(initial_capital)
    curve = pd.DataFrame(equity_curve)
    origins = defaultdict(lambda: {"realized_net_pnl": 0.0, "unrealized_net_pnl": 0.0,
                                   "dividend_cash": 0.0, "dividend_receivable": 0.0, "transaction_cost": 0.0})
    missing_origin_events = 0
    for event in events:
        quantity = float(event.get("execution_quantity") or 0.0)
        dividend = float(event.get("corporate_action_cash_amount") or 0.0)
        receivable = float(event.get("dividend_receivable_change") or 0.0)
        if not quantity and not dividend and not receivable:
            continue
        allocations = event.get("origin_allocations") or []
        if not allocations:
            missing_origin_events += 1
            allocations = [{"origin_account": "unknown", "quantity": abs(quantity) or 1,
                            "cost_amount": float(event.get("cost_amount") or 0),
                            "entry_fee_amount": float(event.get("allocated_entry_fee_amount") or 0)}]
        total_quantity = sum(float(a["quantity"]) for a in allocations)
        for allocation in allocations:
            share = float(allocation["quantity"]) / total_quantity if total_quantity else 0.0
            bucket = origins[allocation["origin_account"]]
            fee = float(event.get("transaction_cost_amount_raw", event.get("transaction_cost_amount")) or 0) * share
            bucket["transaction_cost"] += fee
            if quantity < 0:
                bucket["realized_net_pnl"] += (
                    float(event["execution_price"]) * float(allocation["quantity"])
                    - float(allocation["cost_amount"]) - float(allocation["entry_fee_amount"]) - fee
                )
            bucket["dividend_cash"] += dividend * share
            bucket["dividend_receivable"] += receivable * share
    for position in final_positions:
        for lot in position.get("left_batches", []) + position.get("batches", []):
            allocation = lot.get("origin_allocation")
            if not allocation:
                continue
            origins[allocation["origin_account"]]["unrealized_net_pnl"] += (
                float(position["close"]) * float(allocation["quantity"])
                - float(allocation["cost_amount"]) - float(allocation["entry_fee_amount"])
            )
    by_origin = []
    total_origin_pnl = 0.
    for origin, values in sorted(origins.items()):
        total = values["realized_net_pnl"] + values["unrealized_net_pnl"] + values["dividend_cash"] + values["dividend_receivable"]
        total_origin_pnl += total
        by_origin.append({"origin_account": origin, **{k: round(v, 2) for k, v in values.items()},
                          "total_net_pnl": round(total, 2), "return_contribution_pct": total / capital * 100})
    if curve.empty:
        return {"by_origin": by_origin, "missing_origin_events": missing_origin_events}
    exposure = pd.to_numeric(curve["market_exposure_pct"] if "market_exposure_pct" in curve else 100 - curve["cash_pct"], errors="coerce")
    trading_curve = curve.loc[curve["is_trading_day"].eq(True)] if "is_trading_day" in curve else curve
    trading_exposure = exposure.loc[trading_curve.index]
    years = (pd.Timestamp(curve.iloc[-1]["date"]) - pd.Timestamp(curve.iloc[0]["date"])).days / 365.25
    final_equity = float(curve.iloc[-1]["equity"])
    expected_profit = (final_equity - 1) * capital
    cash_expected = capital + sum(float(e.get("cash_change_amount_raw", e.get("cash_change_amount")) or 0) for e in events)
    def finite_stat(value):
        return None if pd.isna(value) else float(value)

    annual = []
    previous_equity = 1.0
    for year, group in curve.groupby(pd.to_datetime(curve["date"]).dt.year):
        last_equity = float(group.iloc[-1]["equity"])
        trading_group = trading_curve.loc[trading_curve.index.intersection(group.index)]
        annual.append({"year": int(year), "return_pct": (last_equity / previous_equity - 1) * 100 if previous_equity else None,
                       "mean_market_exposure_pct": finite_stat(exposure.loc[trading_group.index].mean()),
                       "mean_held_symbols": finite_stat(trading_group["total_held_symbol_count"].mean()),
                       "flat_days": int(trading_group["total_held_symbol_count"].eq(0).sum()),
                       "trading_days": len(trading_group)})
        previous_equity = last_equity
    return {
        "cagr_pct": (final_equity ** (1 / years) - 1) * 100 if years > 0 and final_equity >= 0 else None,
        "mean_market_exposure_pct": finite_stat(trading_exposure.mean()),
        "median_market_exposure_pct": finite_stat(trading_exposure.median()),
        "below_30pct_exposure_days": int(trading_exposure.lt(30).sum()),
        "flat_days": int(trading_curve["total_held_symbol_count"].eq(0).sum()),
        "valuation_days": len(curve), "trading_days": len(trading_curve), "annual": annual, "by_origin": by_origin,
        "missing_origin_events": missing_origin_events,
        "origin_pnl_reconciliation_difference": round(total_origin_pnl - expected_profit, 4),
        "cash_reconciliation_difference": round(float(curve.iloc[-1]["cash"]) - cash_expected, 4),
        "cost_exposure_basis": "initial_capital_cost_exposure",
        "market_exposure_basis": "current_equity_market_value",
        "note": "来源按最初买入批次；左转右仍属左侧。已实现收益已扣分摊买入费，未平仓买入费计入浮盈；交易成本仅展示，不重复扣减。",
    }
