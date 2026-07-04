"""Daily fundamental-selection command-line entry point."""

from stock_research.pipelines import fundamental_selection


def main(argv=None) -> int:
    return int(fundamental_selection.main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
