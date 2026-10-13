"""Compatibility CLI for the quantsplaybook_robustness research workflow."""
from stock_research.research import quantsplaybook_robustness as _implementation
from stock_research.research.quantsplaybook_robustness import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
