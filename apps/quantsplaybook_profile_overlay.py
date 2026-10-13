"""Compatibility CLI for the quantsplaybook_profile_overlay research workflow."""
from stock_research.research import quantsplaybook_profile_overlay as _implementation
from stock_research.research.quantsplaybook_profile_overlay import main


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    raise SystemExit(main())
