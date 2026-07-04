"""Sector statistics and mainline-watch entry point."""
from __future__ import annotations

import sys

from stock_research.pipelines import sector_statistics, sector_watch


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: sector_analysis.py {stats|watch} [options]")
        return 0
    command, *remaining = arguments
    if command == "stats":
        return int(sector_statistics.main(remaining) or 0)
    if command == "watch":
        return int(sector_watch.main(remaining) or 0)
    print(f"unknown sector command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
