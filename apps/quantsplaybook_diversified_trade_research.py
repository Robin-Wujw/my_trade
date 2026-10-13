"""Compatibility CLI for the quantsplaybook_diversified_trade_research research workflow."""
from stock_research.research import quantsplaybook_diversified_trade_research as _implementation
from stock_research.research.quantsplaybook_diversified_trade_research import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    main()
