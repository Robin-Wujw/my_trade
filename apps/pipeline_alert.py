"""Pipeline-alert command-line entry point."""

from stock_research.reporting import alerts


def main(argv=None) -> int:
    return alerts.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
