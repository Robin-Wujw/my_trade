"""Compatibility CLI for the quantsplaybook_hybrid research workflow."""
from stock_research.research import quantsplaybook_hybrid as _implementation
from stock_research.research.quantsplaybook_hybrid import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
