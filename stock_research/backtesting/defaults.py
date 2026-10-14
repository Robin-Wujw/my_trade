"""Single source of truth for conservative backtest execution defaults.

These values describe the project's default research execution profile.  A
caller may still opt into an alternative profile explicitly, but an omitted
argument must not silently turn off costs or execute a close-confirmed signal
on the same bar.
"""

from __future__ import annotations

from stock_research.backtesting.baseline import baseline_engine_parameters

_PROFILE = baseline_engine_parameters()


DEFAULT_COMMISSION_RATE = _PROFILE["commission_rate"]
DEFAULT_MINIMUM_COMMISSION = _PROFILE["minimum_commission"]
DEFAULT_SELL_STAMP_DUTY_RATE = _PROFILE["sell_stamp_duty_rate"]
DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28 = _PROFILE["sell_stamp_duty_rate_before_2023_08_28"]
DEFAULT_ESTIMATED_SLIPPAGE_RATE = _PROFILE["estimated_slippage_rate"]

# A daily close-confirmed signal is only known after the bar closes.  The
# conservative executable interpretation is the next market open.  The
# ``close_proxy`` mode remains available as an explicit research sensitivity.
DEFAULT_CLOSE_CONFIRMED_EXECUTION = _PROFILE["close_confirmed_execution"]
DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY = _PROFILE["signals_effective_next_day"]

