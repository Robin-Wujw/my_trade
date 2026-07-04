"""Fundamental-cache update command-line entry point."""

from stock_research.pipelines import fundamental_update


def main(argv=None) -> int:
    return int(fundamental_update.main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
