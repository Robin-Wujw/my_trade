"""MiniQMT execution profile for the point-in-time portfolio backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


@dataclass(frozen=True)
class MiniQmtBacktestProfile:
    name: str = "miniqmt"
    commission_rate: float = 0.000085
    minimum_commission: float = 5.0
    sell_stamp_duty_rate: float = 0.0005
    estimated_slippage_rate: float = 0.0005
    close_confirmed_execution: str = "close_proxy"
    signals_effective_next_day: bool = True


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
