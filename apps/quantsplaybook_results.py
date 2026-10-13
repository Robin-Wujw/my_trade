"""Compatibility CLI for the quantsplaybook_results research workflow."""
from stock_research.research import quantsplaybook_results as _implementation
from stock_research.research.quantsplaybook_results import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
