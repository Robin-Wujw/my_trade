"""Compatibility CLI for the quantsplaybook_market_volatility research workflow."""
from stock_research.research import quantsplaybook_market_volatility as _implementation
from stock_research.research.quantsplaybook_market_volatility import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
