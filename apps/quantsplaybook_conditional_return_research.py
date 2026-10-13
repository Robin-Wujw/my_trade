"""Compatibility CLI for the quantsplaybook_conditional_return_research research workflow."""
from stock_research.research import quantsplaybook_conditional_return_research as _implementation
from stock_research.research.quantsplaybook_conditional_return_research import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
