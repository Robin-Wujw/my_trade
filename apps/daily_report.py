"""Daily report command-line entry point."""

from stock_research.pipelines import daily_report


def main(argv=None) -> int:
    return daily_report.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
