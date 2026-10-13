"""Compatibility import for stock_research.backtesting.engine."""
from stock_research.backtesting import engine as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
