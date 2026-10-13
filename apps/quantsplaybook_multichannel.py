"""Compatibility CLI for the quantsplaybook_multichannel research workflow."""
from stock_research.research import quantsplaybook_multichannel as _implementation
from stock_research.research.quantsplaybook_multichannel import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
