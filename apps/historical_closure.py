"""Run the offline baseline closure ending no later than 2026-07-21."""
import argparse
import json

from stock_research.backtesting.baseline import BASELINE_PATH
from stock_research.backtesting.historical_closure import run_closure
from stock_research.core.console import configure_utf8_console
from stock_research.core.paths import PATHS


def main(argv=None):
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--configuration", default=str(BASELINE_PATH))
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--skip-repeat", action="store_true")
    args = parser.parse_args(argv)
    options = vars(args)
    options["repeat"] = not options.pop("skip_repeat")
    print(json.dumps(run_closure(**options), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
