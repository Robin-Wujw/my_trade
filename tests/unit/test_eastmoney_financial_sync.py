import json

import pandas as pd

from apps import eastmoney_financial_sync as sync
from stock_research.api.eastmoney_financials import PageMetadata
from stock_research.storage import Database, TushareRepository


def _raw_income(code):
    return {
        "SECUCODE": code,
        "SECURITY_CODE": code.split(".")[0],
        "NOTICE_DATE": "2026-04-20",
        "REPORT_DATE": "2025-12-31",
        "REPORT_TYPE_CODE": "001",
        "DATA_STATE": "2",
        "TOTAL_OPERATE_INCOME": 100.0,
        "OPERATE_COST": 60.0,
    }


def _raw_balance(code):
    return {
        "SECUCODE": code,
        "SECURITY_CODE": code.split(".")[0],
        "NOTICE_DATE": "2026-04-21",
        "REPORT_DATE": "2025-12-31",
        "REPORT_TYPE_CODE": "001",
        "DATA_STATE": "2",
        "TOTAL_ASSETS": 200.0,
    }


def test_sync_financial_pair_writes_atomic_versions_and_audit(monkeypatch, tmp_path):
    codes = ["600000.SH", "920001.BJ"]

    def fetch(report_name, report_date):
        rows = (
            [_raw_income(code) for code in codes]
            if report_name == "RPT_DMSK_FN_INCOME"
            else [_raw_balance(code) for code in codes]
        )
        return pd.DataFrame(rows), PageMetadata(
            report_name=report_name,
            report_date=report_date,
            row_count=2,
            page_count=1,
        )

    monkeypatch.setattr(sync.eastmoney_financials, "fetch_report_snapshot", fetch)
    database_path = tmp_path / "my_trade.sqlite3"
    manifest_path = sync.sync_financial_pair(
        report_period="20251231",
        database_path=database_path,
        output_root=tmp_path / "audit",
        minimum_rows=2,
        minimum_complete_ratio=1.0,
    )

    repository = TushareRepository(Database(database_path))
    income = repository.load_dataset_versions(sync.INCOME_DATASET)
    balance = repository.load_dataset_versions(sync.BALANCE_DATASET)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(income) == len(balance) == 2
    assert income["_captured_at"].unique().tolist() == balance[
        "_captured_at"
    ].unique().tolist()
    assert repository.load_dataset(sync.INCOME_DATASET).empty
    assert repository.load_dataset(sync.BALANCE_DATASET).empty
    assert manifest["historical_backfill_allowed"] is False
    assert manifest["research_only"] is True
    assert manifest["coverage"]["common_codes"] == 2
    assert manifest["coverage"]["beijing_codes"] == 1
    assert manifest["version_summary_after"][sync.INCOME_DATASET][
        "version_rows"
    ] == 2
