"""Stable command-line entry point for portfolio backtesting.

Historical helper imports remain readable for external callers; repository
code imports the backtesting package directly.
"""
from stock_research.backtesting.cli import main
from stock_research.backtesting import workflow as _workflow


def __getattr__(name):
    return getattr(_workflow, name)


if __name__ == "__main__":
    main()
