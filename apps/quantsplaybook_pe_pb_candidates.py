"""Compatibility CLI for the quantsplaybook_pe_pb_candidates research workflow."""
from stock_research.research import quantsplaybook_pe_pb_candidates as _implementation
from stock_research.research.quantsplaybook_pe_pb_candidates import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
