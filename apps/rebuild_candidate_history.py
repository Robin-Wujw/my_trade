"""Rebuild and persist conservative daily candidate snapshots."""
from __future__ import annotations

import argparse

from stock_research.core.paths import PATHS
from stock_research.storage import Database, ResearchRepository
from stock_research.strategies.historical_candidates import (
    build_historical_candidate_snapshots,
    save_historical_candidate_snapshots,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="回建逐交易日研究候选快照")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-10")
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v3"),
    )
    args = parser.parse_args(argv)
    database = Database(PATHS.database, code_version="candidate-history-v3")
    database.initialize()
    research_repository = ResearchRepository(database)
    snapshots = build_historical_candidate_snapshots(
        args.start_date,
        args.end_date,
        value_cache_directory=PATHS.cache / "q1_value",
        kline_directory=PATHS.cache / "formula33_kline" / "akshare",
        universe_path=PATHS.cache / "stock_universe.csv",
        mainline_directory=PATHS.cache,
        research_repository=research_repository,
    )
    manifest = save_historical_candidate_snapshots(
        args.output_directory,
        snapshots,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    persisted = research_repository.persist_candidate_snapshots(
        snapshots, version=manifest["version"],
    )
    print(
        f"历史候选快照: {args.output_directory}，"
        f"共{manifest['snapshot_count']}个交易日，数据库{persisted}行"
    )


if __name__ == "__main__":
    main()
