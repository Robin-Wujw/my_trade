import hashlib
import json
import re
from pathlib import Path

from openpyxl import Workbook

from stock_research.regression.formula33_screenshot import verify_workbook


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "regression"
    / "formula33_screenshot_20260611_20260710_188.json"
)
EXPECTED_SHA256 = "85d51254e22a99f6bbc0bbab16b8263648b178c62302ecabafb209dbc462c5a7"


def test_formula33_screenshot_188_fixture_is_complete_and_stable():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    codes = payload["codes"]

    assert payload["start_date"] == "2026-06-11"
    assert payload["end_date"] == "2026-07-10"
    assert payload["count"] == 188
    assert len(codes) == payload["count"]
    assert len(set(codes)) == payload["count"]
    assert all(re.fullmatch(r"\d{6}", code) for code in codes)
    assert "001331" in codes

    normalized = "\n".join(sorted(codes)) + "\n"
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == EXPECTED_SHA256


def test_actual_workbook_verifier_checks_all_reviewed_pools(tmp_path):
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    workbook = Workbook()
    summary = workbook.active
    summary.title = "33公式日统计"
    summary.append(["date", "window_unique_count"])
    summary.append([expected["start_date"], 1])
    summary.append([expected["end_date"], expected["formal_count"]])
    for key, sheet_name in {
        "formal_count": "21日技术可交易",
        "technical_count": "21日技术全量",
        "market_cap_count": "市值大于100亿池",
        "suspended_count": "停牌技术命中诊断",
    }.items():
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(["code"])
        codes = expected["codes"] if key == "formal_count" else [
            f"{index:06d}" for index in range(expected[key])
        ]
        for code in codes:
            sheet.append([code])
    path = tmp_path / "formula33.xlsx"
    workbook.save(path)

    assert verify_workbook(path, FIXTURE) == {
        "formal_count": 188,
        "technical_count": 191,
        "market_cap_count": 145,
        "suspended_count": 3,
    }
