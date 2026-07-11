"""Single production entry point for the daily research pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stock_research.core.console import configure_utf8_console
from stock_research.core.paths import PATHS
from stock_research.core.config import load_pipeline_config
from stock_research.pipelines.daily import (
    STEP_NAMES,
    build_default_steps,
    run_daily_pipeline,
)


def build_parser():
    parser = argparse.ArgumentParser(description="运行每日A股研究与选股流水线")
    parser.add_argument(
        "--config",
        default=str(PATHS.project_root / "config" / "pipeline.toml"),
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        default=os.environ.get("NO_PUSH") == "1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    configure_utf8_console()
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(Path(args.config))
    steps = build_default_steps(
        config,
        no_push=args.no_push,
        report_period=os.environ.get("REPORT_PERIOD", "").strip(),
    )
    if args.dry_run:
        print("\n".join(STEP_NAMES))
        return 0
    result = run_daily_pipeline(
        steps=steps,
        no_push=args.no_push,
    )
    if result.failed_steps:
        print("FAILED STEPS: " + ", ".join(result.failed_steps))
    if result.skipped_steps:
        print("SKIPPED STEPS: " + ", ".join(result.skipped_steps))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
