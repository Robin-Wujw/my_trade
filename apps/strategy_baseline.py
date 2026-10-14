"""Generate or verify the current baseline documentation."""
import argparse

from stock_research.core.paths import PATHS
from stock_research.reporting.baseline_document import render_baseline_document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = PATHS.project_root / "docs" / "current-baseline.md"
    expected = render_baseline_document()
    if args.check:
        matched = path.is_file() and path.read_text(encoding="utf-8") == expected
        print("baseline documentation current" if matched else "baseline documentation out of date")
        return 0 if matched else 1
    path.write_text(expected, encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
