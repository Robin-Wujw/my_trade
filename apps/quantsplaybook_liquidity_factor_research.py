"""Compatibility CLI for the quantsplaybook_liquidity_factor_research research workflow."""
from stock_research.research import quantsplaybook_liquidity_factor_research as _implementation
from stock_research.research.quantsplaybook_liquidity_factor_research import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
