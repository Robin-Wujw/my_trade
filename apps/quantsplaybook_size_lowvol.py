"""Compatibility CLI for the quantsplaybook_size_lowvol research workflow."""
from stock_research.research import quantsplaybook_size_lowvol as _implementation
from stock_research.research.quantsplaybook_size_lowvol import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
