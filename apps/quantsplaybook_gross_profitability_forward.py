"""Compatibility CLI for the quantsplaybook_gross_profitability_forward research workflow."""
from stock_research.research import quantsplaybook_gross_profitability_forward as _implementation
from stock_research.research.quantsplaybook_gross_profitability_forward import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
