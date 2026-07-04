"""Factor-selection command-line entry point."""
from __future__ import annotations

from stock_research.pipelines import factor_selection


def main(argv=None) -> int:
    return int(factor_selection.main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
