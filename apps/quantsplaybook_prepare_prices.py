"""Compatibility CLI for the quantsplaybook_prepare_prices research workflow."""
from stock_research.research import quantsplaybook_prepare_prices as _implementation
from stock_research.research.quantsplaybook_prepare_prices import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
