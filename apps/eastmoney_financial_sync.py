"""Capture a forward-only Eastmoney annual financial snapshot for GP/A."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from stock_research.api import eastmoney_financials
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


INCOME_DATASET = "eastmoney_income"
BALANCE_DATASET = "eastmoney_balancesheet"
SOURCE = "eastmoney/datacenter-forward-snapshot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_summary(database: Database, report_period: str) -> dict:
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                dataset,
                count(*),
                count(DISTINCT row_key),
                min(captured_at),
                max(captured_at)
            FROM raw.tushare_dataset_row_versions
            WHERE dataset IN (?, ?)
              AND report_period = ?
            GROUP BY dataset
            ORDER BY dataset
            """,
            [INCOME_DATASET, BALANCE_DATASET, report_period],
        ).fetchall()
    finally:
        connection.close()
    return {
        str(dataset): {
            "version_rows": int(version_rows),
            "stable_row_keys": int(row_keys),
            "first_captured_at_utc": str(first_captured),
            "last_captured_at_utc": str(last_captured),
        }
        for dataset, version_rows, row_keys, first_captured, last_captured in rows
    }


def sync_financial_pair(
    *,
    report_period: str,
    database_path: str | Path,
    output_root: str | Path,
    minimum_rows: int = 5000,
    minimum_complete_ratio: float = 0.97,
) -> Path:
    period = pd.to_datetime(report_period, errors="coerce")
    if pd.isna(period) or period.month != 12 or period.day != 31:
        raise ValueError("GP/A forward sync requires a December 31 annual report")
    period_text = period.strftime("%Y-%m-%d")

    raw_income, income_page = eastmoney_financials.fetch_report_snapshot(
        "RPT_DMSK_FN_INCOME", period_text,
    )
    raw_balance, balance_page = eastmoney_financials.fetch_report_snapshot(
        "RPT_DMSK_FN_BALANCE", period_text,
    )
    income = eastmoney_financials.normalize_report_snapshot(
        raw_income,
        report_name="RPT_DMSK_FN_INCOME",
        report_date=period_text,
    )
    balance = eastmoney_financials.normalize_report_snapshot(
        raw_balance,
        report_name="RPT_DMSK_FN_BALANCE",
        report_date=period_text,
    )
    coverage = eastmoney_financials.validate_gross_profitability_pair(
        income,
        balance,
        minimum_rows=minimum_rows,
        minimum_complete_ratio=minimum_complete_ratio,
    )

    database = Database(
        Path(database_path), code_version="eastmoney-financial-forward-v1",
    )
    database.initialize()
    repository = TushareRepository(database)
    before = _version_summary(database, period_text)
    attempted = repository.append_dataset_version_batch(
        {
            INCOME_DATASET: income,
            BALANCE_DATASET: balance,
        },
        source=SOURCE,
    )
    after = _version_summary(database, period_text)

    output = Path(output_root) / period_text
    output.mkdir(parents=True, exist_ok=True)
    income_path = output / "eastmoney_income.csv"
    balance_path = output / "eastmoney_balancesheet.csv"
    income.to_csv(income_path, index=False, encoding="utf-8-sig")
    balance.to_csv(balance_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": 1,
        "source": SOURCE,
        "endpoint": eastmoney_financials.API_URL,
        "report_period": period_text,
        "captured_forward_only": True,
        "historical_backfill_allowed": False,
        "research_only": True,
        "future_announcement_dates_used_early": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": str(Path(database_path).resolve()),
        "datasets": {
            INCOME_DATASET: {
                "report_name": income_page.report_name,
                "provider_rows": income_page.row_count,
                "provider_pages": income_page.page_count,
                "attempted_version_rows": attempted[INCOME_DATASET],
                "snapshot_path": str(income_path.resolve()),
                "snapshot_sha256": _sha256(income_path),
            },
            BALANCE_DATASET: {
                "report_name": balance_page.report_name,
                "provider_rows": balance_page.row_count,
                "provider_pages": balance_page.page_count,
                "attempted_version_rows": attempted[BALANCE_DATASET],
                "snapshot_path": str(balance_path.resolve()),
                "snapshot_sha256": _sha256(balance_path),
            },
        },
        "coverage": coverage,
        "version_summary_before": before,
        "version_summary_after": after,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-period", required=True)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument(
        "--output-root",
        default=str(PATHS.market_exports / "eastmoney_financials"),
    )
    parser.add_argument("--minimum-rows", type=int, default=5000)
    parser.add_argument("--minimum-complete-ratio", type=float, default=0.97)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = sync_financial_pair(
        report_period=args.report_period,
        database_path=args.database,
        output_root=args.output_root,
        minimum_rows=args.minimum_rows,
        minimum_complete_ratio=args.minimum_complete_ratio,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
