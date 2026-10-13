"""MiniQMT execution profile for the point-in-time portfolio backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stock_research.backtesting.defaults import (
    DEFAULT_CLOSE_CONFIRMED_EXECUTION,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    DEFAULT_MINIMUM_COMMISSION,
    DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28,
    DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY,
)
from stock_research.backtesting.engine import run_portfolio_backtest


@dataclass(frozen=True)
class MiniQmtBacktestProfile:
    name: str = "miniqmt"
    commission_rate: float = DEFAULT_COMMISSION_RATE
    minimum_commission: float = DEFAULT_MINIMUM_COMMISSION
    sell_stamp_duty_rate: float = DEFAULT_SELL_STAMP_DUTY_RATE
    sell_stamp_duty_rate_before_2023_08_28: float = (
        DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28
    )
    estimated_slippage_rate: float = DEFAULT_ESTIMATED_SLIPPAGE_RATE
    close_confirmed_execution: str = DEFAULT_CLOSE_CONFIRMED_EXECUTION
    signals_effective_next_day: bool = DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY


DEFAULT_MINIQMT_BACKTEST_PROFILE = MiniQmtBacktestProfile()


def run_miniqmt_backtest(
    price_frames,
    candidate_snapshots,
    formula_phases,
    *,
    profile: MiniQmtBacktestProfile | None = None,
    **kwargs: Any,
):
    """Run the existing portfolio replay with MiniQMT-like cost metadata."""
    execution_profile = profile or DEFAULT_MINIQMT_BACKTEST_PROFILE
    profile_kwargs = {
        "commission_rate": execution_profile.commission_rate,
        "minimum_commission": execution_profile.minimum_commission,
        "sell_stamp_duty_rate": execution_profile.sell_stamp_duty_rate,
        "sell_stamp_duty_rate_before_2023_08_28": (
            execution_profile.sell_stamp_duty_rate_before_2023_08_28
        ),
        "estimated_slippage_rate": execution_profile.estimated_slippage_rate,
        "close_confirmed_execution": execution_profile.close_confirmed_execution,
        "signals_effective_next_day": execution_profile.signals_effective_next_day,
    }
    profile_kwargs.update(kwargs)
    result = run_portfolio_backtest(
        price_frames,
        candidate_snapshots,
        formula_phases,
        **profile_kwargs,
    )
    result["execution_profile"] = execution_profile.name
    result["execution_profile_detail"] = asdict(execution_profile)
    result["research_only"] = True
    result["strict_point_in_time"] = False
    result["acceptance_status"] = (
        "data_limited" if result.get("corporate_action_limitations") else "research_only"
    )
    result["broker_connector"] = {
        "name": "miniqmt",
        "mode": "read_only_execution_profile",
        "live_trading_enabled": False,
        "broker_simulation_scope": (
            "portfolio daily-bar replay with MiniQMT-style cost defaults; "
            "not an intraday order book or partial-fill simulator"
        ),
    }
    return result
