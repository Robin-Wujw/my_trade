"""Compatibility CLI for the quantsplaybook_chunked research workflow."""
from stock_research.research import quantsplaybook_chunked as _implementation
from stock_research.research.quantsplaybook_chunked import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
