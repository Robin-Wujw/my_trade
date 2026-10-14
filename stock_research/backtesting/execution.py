"""Cash fills and dated costs; signal selection and lot eligibility live elsewhere."""
from __future__ import annotations

from dataclasses import dataclass

from stock_research.backtesting.accounting import (
    _affordable_buy_notional, _board_lot_quantity, sell_stamp_duty_rate_for_date,
)


@dataclass(frozen=True)
class ExecutionCosts:
    commission_rate: float
    minimum_commission: float
    slippage_rate: float
    sell_stamp_duty_rate: float
    sell_stamp_duty_rate_before_2023_08_28: float | None

    def components(self, turnover, *, sell=False, trade_date=None):
        turnover = abs(float(turnover))
        if turnover <= 0:
            return {"commission_amount": 0.0, "stamp_duty_amount": 0.0,
                    "slippage_amount": 0.0, "transaction_cost_amount": 0.0}
        commission = max(turnover * float(self.commission_rate), float(self.minimum_commission))
        stamp_duty_rate = (sell_stamp_duty_rate_for_date(
            trade_date, self.sell_stamp_duty_rate, self.sell_stamp_duty_rate_before_2023_08_28,
        ) if sell else 0.0)
        stamp_duty = turnover * stamp_duty_rate
        slippage = turnover * float(self.slippage_rate)
        return {"commission_amount": commission, "stamp_duty_amount": stamp_duty,
                "slippage_amount": slippage,
                "transaction_cost_amount": commission + stamp_duty + slippage}

    def total(self, turnover, *, sell=False, trade_date=None):
        return self.components(turnover, sell=sell, trade_date=trade_date)["transaction_cost_amount"]


@dataclass(frozen=True)
class CashBuyFill:
    size: float
    quantity: float
    normalized_fee: float
    fee_cash: float
    cash_limited: bool
    cash_after: float
    phase_budget_after: float

    def engine_tuple(self):
        if self.quantity <= 0:
            return 0.0, 0.0, 0.0, 0.0, self.cash_limited
        return self.size, self.quantity, -self.normalized_fee, self.fee_cash, self.cash_limited


def fill_cash_buy(code, requested_size, price, *, initial_capital, cash_balance,
                  phase_buy_budget, costs: ExecutionCosts):
    """Spend only phase-start cash, shrink to board lots, and never borrow."""
    available_cash = min(cash_balance, phase_buy_budget)
    requested_notional = max(0.0, float(requested_size)) * float(initial_capital)
    price = float(price)
    def empty(limited=False):
        return CashBuyFill(0.0, 0.0, 0.0, 0.0, limited, cash_balance, phase_buy_budget)
    if requested_notional <= 0 or price <= 0 or available_cash <= float(costs.minimum_commission):
        return empty()
    affordable_notional = _affordable_buy_notional(
        available_cash, requested_notional, commission_rate=costs.commission_rate,
        minimum_commission=costs.minimum_commission, slippage_rate=costs.slippage_rate,
    )
    if affordable_notional <= 1e-8:
        return empty(True)
    cash_limited = affordable_notional < requested_notional - 1e-8
    quantity = _board_lot_quantity(code, affordable_notional / price)
    if quantity <= 0:
        return empty(cash_limited)
    notional = quantity * price
    fee_cash = costs.total(notional)
    cash_after = max(0.0, cash_balance - notional - fee_cash)
    budget_after = max(0.0, phase_buy_budget - notional - fee_cash)
    fee = fee_cash / float(initial_capital)
    return CashBuyFill(notional / float(initial_capital), float(quantity), fee,
                       fee_cash, cash_limited, cash_after, budget_after)


@dataclass(frozen=True)
class CashSellFill:
    size: float
    pnl: float
    normalized_fee: float
    cash_after: float


def fill_cash_sell(quantity, price, cost, *, initial_capital, cash_balance,
                   trade_date, costs: ExecutionCosts):
    """Sell the exact eligible share quantity supplied by the position layer."""
    quantity, price, cost = max(0.0, float(quantity)), float(price), float(cost)
    if quantity <= 0:
        return CashSellFill(0.0, 0.0, 0.0, cash_balance)
    turnover = quantity * price
    fee_cash = costs.total(turnover, sell=True, trade_date=trade_date)
    cash_after = cash_balance + (turnover - fee_cash)
    fee = fee_cash / float(initial_capital)
    cost_value = quantity * cost
    return CashSellFill(cost_value / float(initial_capital),
                        (turnover - cost_value - fee_cash) / float(initial_capital),
                        fee, cash_after)
