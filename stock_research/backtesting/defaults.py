"""Single source of truth for conservative backtest execution defaults.

These values describe the project's default research execution profile.  A
caller may still opt into an alternative profile explicitly, but an omitted
argument must not silently turn off costs or execute a close-confirmed signal
on the same bar.
"""

from __future__ import annotations


DEFAULT_COMMISSION_RATE = 0.000085
DEFAULT_MINIMUM_COMMISSION = 5.0
DEFAULT_SELL_STAMP_DUTY_RATE = 0.0005
DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28 = 0.001
DEFAULT_ESTIMATED_SLIPPAGE_RATE = 0.0005

# A daily close-confirmed signal is only known after the bar closes.  The
# conservative executable interpretation is the next market open.  The
# ``close_proxy`` mode remains available as an explicit research sensitivity.
DEFAULT_CLOSE_CONFIRMED_EXECUTION = "next_open"
DEFAULT_SIGNALS_EFFECTIVE_NEXT_DAY = True

