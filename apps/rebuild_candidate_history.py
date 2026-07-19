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
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
    )
    parser.add_argument(
        "--raw-kline-directory",
        default=str(PATHS.cache / "formula33_kline" / "akshare_raw"),
        help="AkShare不复权K线缓存目录，用于反推观察日锚定前复权价",
    )
    parser.add_argument("--price-source", choices=("akshare", "miniqmt"), default="akshare")
    parser.add_argument("--kline-directory", default="")
    parser.add_argument(
        "--allow-unsafe-financial",
        action="store_true",
        help="research only: use the conservative report-period cache even when per-company announcement dates are not visible",
    )
    parser.add_argument(
        "--skip-database-persist",
        action="store_true",
        help="write CSV snapshots and manifest only; useful for offline backtest inputs",
    )
    args = parser.parse_args(argv)
    kline_directory = args.kline_directory or (
        PATHS.cache / "miniqmt_kline" / "1d" / "front"
        if args.price_source == "miniqmt"
        else PATHS.cache / "formula33_kline" / "akshare"
    )
    raw_kline_directory = args.raw_kline_directory
    if args.price_source == "miniqmt" and raw_kline_directory == str(PATHS.cache / "formula33_kline" / "akshare_raw"):
        raw_kline_directory = str(PATHS.cache / "miniqmt_kline" / "1d" / "none")
    research_repository = None
    if not args.skip_database_persist:
        database = Database(PATHS.database, code_version="candidate-history-v3")
        database.initialize()
        research_repository = ResearchRepository(database)
    snapshots = build_historical_candidate_snapshots(
        args.start_date,
        args.end_date,
        value_cache_directory=PATHS.cache / "q1_value",
        kline_directory=kline_directory,
        raw_kline_directory=raw_kline_directory,
        universe_path=PATHS.cache / "stock_universe.csv",
        mainline_directory=PATHS.cache,
        research_repository=research_repository,
        price_source=args.price_source,
        strict_financial_point_in_time=not args.allow_unsafe_financial,
    )
    manifest = save_historical_candidate_snapshots(
        args.output_directory,
        snapshots,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    persisted = 0
    if not args.skip_database_persist:
        persisted = research_repository.persist_candidate_snapshots(
            snapshots, version=manifest["version"],
        )
    print(
        f"历史候选快照: {args.output_directory}，"
        f"共{manifest['snapshot_count']}个交易日，数据库{persisted}行"
    )


if __name__ == "__main__":
    main()
