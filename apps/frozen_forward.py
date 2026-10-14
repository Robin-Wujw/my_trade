"""Capture today's frozen candidates and offline paper portfolio."""
import argparse
import json

from stock_research.core.console import configure_utf8_console
from stock_research.core.paths import PATHS
from stock_research.research.frozen_forward import run_capture


def main(argv=None):
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--sync-state", default=str(PATHS.state / "forward_data_sync.json"))
    parser.add_argument("--left-directory", required=True)
    parser.add_argument("--formula-history", required=True)
    parser.add_argument("--observation-date", required=True)
    parser.add_argument("--output-root", default=str(PATHS.runtime_root / "forward" / "frozen_strategy"))
    args = parser.parse_args(argv)
    print(json.dumps(run_capture(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
