"""Compatibility import for stock_research.backtesting.corporate_actions."""
from stock_research.backtesting import corporate_actions as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
