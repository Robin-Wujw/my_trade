"""Persist reconstructed Formula33 phase history for research backtests."""
from __future__ import annotations

import argparse
from pathlib import Path

from stock_research.core.paths import PATHS
from stock_research.strategies.historical_formula import rebuild_formula_history


def main(argv=None):
    parser = argparse.ArgumentParser(description="回建Formula33市场阶段历史")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-10")
    parser.add_argument(
        "--output",
        default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"),
    )
    parser.add_argument(
        "--kline-directory",
        default=str(PATHS.cache / "formula33_kline" / "akshare"),
    )
    args = parser.parse_args(argv)
    frame = rebuild_formula_history(
        args.kline_directory,
        args.start_date,
        args.end_date,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PATHS.project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output)
    print(f"Formula33阶段历史: {output}，共{len(frame)}个交易日")


if __name__ == "__main__":
    main()
