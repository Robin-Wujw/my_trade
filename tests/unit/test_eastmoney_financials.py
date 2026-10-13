from types import SimpleNamespace

import pandas as pd
import pytest

from stock_research.api import eastmoney_financials as eastmoney


def _raw_income(code, *, notice="2026-04-20", cost=60.0):
    return {
        "SECUCODE": code,
        "SECURITY_CODE": code.split(".")[0],
        "NOTICE_DATE": notice,
        "REPORT_DATE": "2025-12-31",
        "REPORT_TYPE_CODE": "001",
        "DATA_STATE": "2",
        "TOTAL_OPERATE_INCOME": 100.0,
        "OPERATE_COST": cost,
        "TOTAL_OPERATE_COST": 999.0,
    }


def _raw_balance(code, *, notice="2026-04-21", assets=200.0):
    return {
        "SECUCODE": code,
        "SECURITY_CODE": code.split(".")[0],
        "NOTICE_DATE": notice,
        "REPORT_DATE": "2025-12-31",
        "REPORT_TYPE_CODE": "001",
        "DATA_STATE": "2",
        "TOTAL_ASSETS": assets,
    }


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, _url, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        )


def test_fetch_report_uses_explicit_fields_and_does_not_repeat_first_page(monkeypatch):
    first = [_raw_income(f"{number:06d}.SZ") for number in range(500)]
    second = [_raw_income("600000.SH")]
    session = _Session([
        {"success": True, "result": {"count": 501, "pages": 2, "data": first}},
        {"success": True, "result": {"count": 501, "pages": 2, "data": second}},
    ])
    monkeypatch.setattr(eastmoney._RATE_LIMITER, "wait", lambda: None)

    frame, metadata = eastmoney.fetch_report_snapshot(
        "RPT_DMSK_FN_INCOME",
        "20251231",
        session=session,
        retries=1,
    )

    assert len(frame) == 501
    assert metadata.page_count == 2
    assert [call["params"]["pageNumber"] for call in session.calls] == ["1", "2"]
    requested = set(session.calls[0]["params"]["columns"].split(","))
    assert "OPERATE_COST" in requested
    assert "TOTAL_OPERATE_COST" not in requested


def test_normalize_income_uses_operating_cost_not_total_operating_cost():
    normalized = eastmoney.normalize_report_snapshot(
        pd.DataFrame([_raw_income("600000.SH", cost=61.0)]),
        report_name="RPT_DMSK_FN_INCOME",
        report_date="2025-12-31",
    )

    assert normalized.iloc[0]["oper_cost"] == 61.0
    assert normalized.iloc[0]["total_revenue"] == 100.0
    assert normalized.iloc[0]["report_type"] == "1"


def test_pair_gate_preserves_missing_cost_as_fail_closed_coverage():
    codes = ["600000.SH", "000001.SZ", "920001.BJ"]
    income = eastmoney.normalize_report_snapshot(
        pd.DataFrame([
            _raw_income(code, cost=None if code == "000001.SZ" else 60.0)
            for code in codes
        ]),
        report_name="RPT_DMSK_FN_INCOME",
        report_date="2025-12-31",
    )
    balance = eastmoney.normalize_report_snapshot(
        pd.DataFrame([_raw_balance(code) for code in codes]),
        report_name="RPT_DMSK_FN_BALANCE",
        report_date="2025-12-31",
    )

    coverage = eastmoney.validate_gross_profitability_pair(
        income,
        balance,
        minimum_rows=3,
        minimum_complete_ratio=2 / 3,
    )

    assert coverage["common_codes"] == 3
    assert coverage["beijing_codes"] == 1
    assert coverage["income_missing_oper_cost"] == 1
    assert coverage["complete_gpa_codes"] == 2


def test_pair_gate_rejects_cross_statement_universe_drift():
    income = eastmoney.normalize_report_snapshot(
        pd.DataFrame([_raw_income("600000.SH")]),
        report_name="RPT_DMSK_FN_INCOME",
        report_date="2025-12-31",
    )
    balance = eastmoney.normalize_report_snapshot(
        pd.DataFrame([_raw_balance("000001.SZ")]),
        report_name="RPT_DMSK_FN_BALANCE",
        report_date="2025-12-31",
    )

    with pytest.raises(eastmoney.EastmoneyFinancialError, match="universes differ"):
        eastmoney.validate_gross_profitability_pair(
            income,
            balance,
            minimum_rows=1,
            minimum_complete_ratio=0,
        )


def test_fetch_report_rejects_pagination_drift(monkeypatch):
    first = [_raw_balance(f"{number:06d}.SZ") for number in range(500)]
    session = _Session([
        {"success": True, "result": {"count": 501, "pages": 2, "data": first}},
        {"success": True, "result": {
            "count": 502,
            "pages": 2,
            "data": [_raw_balance("600000.SH")],
        }},
    ])
    monkeypatch.setattr(eastmoney._RATE_LIMITER, "wait", lambda: None)

    with pytest.raises(
        eastmoney.EastmoneyFinancialError,
        match="changed during pagination",
    ):
        eastmoney.fetch_report_snapshot(
            "RPT_DMSK_FN_BALANCE",
            "20251231",
            session=session,
            retries=1,
        )
