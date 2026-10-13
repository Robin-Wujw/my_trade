"""Compatibility CLI for the quantsplaybook_residual_momentum_lowvol research workflow."""
from stock_research.research import quantsplaybook_residual_momentum_lowvol as _implementation
from stock_research.research.quantsplaybook_residual_momentum_lowvol import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
