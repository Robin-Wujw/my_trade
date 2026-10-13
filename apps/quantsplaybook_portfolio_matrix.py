"""Compatibility CLI for the quantsplaybook_portfolio_matrix research workflow."""
from stock_research.research import quantsplaybook_portfolio_matrix as _implementation
from stock_research.research.quantsplaybook_portfolio_matrix import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
