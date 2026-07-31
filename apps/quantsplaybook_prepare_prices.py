"""Build resumable per-code prepared price frames for factor portfolio replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from apps.quantsplaybook_compare import prepared_price_cache_path
from apps import portfolio_backtest
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


DEFAULT_CANDIDATE_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)


def candidate_codes(candidate_root: str | Path) -> set[str]:
    codes = set()
    for manifest_path in sorted(
        Path(candidate_root).glob("candidates/*/manifest.json"),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        directory = manifest_path.parent
        for row in manifest.get("snapshots") or []:
            path = directory / str(row["file"])
            frame = pd.read_csv(path, usecols=["code"], dtype={"code": str})
            codes.update(frame["code"].dropna().astype(str))
    return codes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare execution price frames once for repeated factor backtests",
    )
    parser.add_argument(
        "--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT),
    )
    parser.add_argument("--start-date", default="2019-02-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--output-directory", default="")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    candidate_root = Path(args.candidate_root)
    output = (
        Path(args.output_directory)
        if args.output_directory
        else candidate_root / "prepared_prices"
    )
    output.mkdir(parents=True, exist_ok=True)
    codes = sorted(candidate_codes(candidate_root))
    pending = [
        code for code in codes
        if not prepared_price_cache_path(output, code).is_file()
    ]
    repository = TushareRepository(Database(args.database_path))
    batch_size = max(1, int(args.batch_size))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        print(
            f"[prepared-prices] {offset + 1}-{offset + len(batch)}/"
            f"{len(pending)} total_codes={len(codes)}",
            flush=True,
        )
        prices = repository.load_daily_kline_frames(
            batch,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        limits = repository.load_dataset_for_codes(
            "stk_limit",
            batch,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if not limits.empty:
            limits = limits[
                ["date", "code", "up_limit", "down_limit"]
            ].copy()
            limits["date"] = pd.to_datetime(limits["date"], errors="coerce")
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
            prices = prices.merge(
                limits.drop_duplicates(["date", "code"], keep="last"),
                on=["date", "code"],
                how="left",
            )
        raw_frames = {
            str(code): group.drop(columns=["code"]).reset_index(drop=True)
            for code, group in prices.groupby("code", sort=True)
        }
        missing = sorted(set(batch) - set(raw_frames))
        if missing:
            raise RuntimeError(
                f"price cache batch has no rows: {missing[:10]}",
            )
        prepared = portfolio_backtest.prepare_portfolio_price_frames(
            raw_frames,
        )
        for code, frame in prepared.items():
            target = prepared_price_cache_path(output, code)
            temporary = target.with_suffix(".pkl.tmp")
            frame.to_pickle(temporary)
            temporary.replace(target)
    manifest = {
        "version": 1,
        "database_path": str(Path(args.database_path)),
        "candidate_root": str(candidate_root),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "code_count": len(codes),
        "complete_file_count": sum(
            prepared_price_cache_path(output, code).is_file()
            for code in codes
        ),
        "price_source": "local_tushare_daily_kline_adj_factor_stk_limit",
        "prepared_with": "prepare_portfolio_price_frames",
    }
    if manifest["complete_file_count"] != manifest["code_count"]:
        raise RuntimeError(f"prepared price cache incomplete: {manifest}")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
