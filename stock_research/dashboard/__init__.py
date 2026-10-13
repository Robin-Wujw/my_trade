"""Read-only research dashboard helpers.

The dashboard is an optional presentation layer.  Its catalog helpers stay
free of Streamlit so the core project and its tests do not need the optional
GUI dependencies installed.
"""

from .catalog import (
    BacktestArtifact,
    discover_backtests,
    drawdown_frame,
    filter_trades,
    load_table,
    load_summary,
    quality_badges,
    summary_metrics,
)

__all__ = [
    "BacktestArtifact",
    "discover_backtests",
    "drawdown_frame",
    "filter_trades",
    "load_table",
    "load_summary",
    "quality_badges",
    "summary_metrics",
]
