"""Compatibility CLI for the quantsplaybook_liquidity_universe research workflow."""
from stock_research.research import quantsplaybook_liquidity_universe as _implementation
from stock_research.research.quantsplaybook_liquidity_universe import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
