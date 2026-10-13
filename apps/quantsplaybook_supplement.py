"""Compatibility CLI for the quantsplaybook_supplement research workflow."""
from stock_research.research import quantsplaybook_supplement as _implementation
from stock_research.research.quantsplaybook_supplement import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
