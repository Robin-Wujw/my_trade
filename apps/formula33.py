"""Formula33 command-line entry point."""
from __future__ import annotations

from stock_research.pipelines import formula33


def main(argv=None) -> int:
    result = formula33.main(argv)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
