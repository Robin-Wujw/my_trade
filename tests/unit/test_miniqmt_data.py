import pandas as pd
import pytest

from stock_research.market import miniqmt_data
from stock_research.market.miniqmt_data import (
    compare_price_frames,
    load_cached_miniqmt_frame,
    load_miniqmt_price_frames,
    miniqmt_code_to_project,
    miniqmt_cache_path,
    normalize_miniqmt_frame,
    normalize_project_code,
    project_code_to_miniqmt,
    save_miniqmt_frame,
)


def test_code_conversion_between_project_and_miniqmt_formats():
    assert normalize_project_code("600000") == "sh.600000"
    assert normalize_project_code("000001") == "sz.000001"
    assert normalize_project_code("688041") == "sh.688041"
    assert project_code_to_miniqmt("sh.600000") == "600000.SH"
    assert project_code_to_miniqmt("sz.000001") == "000001.SZ"
    assert miniqmt_code_to_project("600000.SH") == "sh.600000"
    assert miniqmt_code_to_project("000001.SZ") == "sz.000001"


def test_miniqmt_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCK_RESEARCH_VAR", str(tmp_path))
    frame = pd.DataFrame({
        "date": ["2026-07-14", "2026-07-15"],
        "open": [10, 11],
        "high": [11, 12],
        "low": [9, 10],
        "close": [10.5, 11.5],
        "volume": [1000, 1200],
        "amount": [10_500, 13_800],
    })

    path = save_miniqmt_frame("600000", frame, period="1d", dividend_type="front")
    loaded = load_cached_miniqmt_frame(
        "sh.600000",
        period="1d",
        dividend_type="front",
        start_date="2026-07-15",
        end_date="2026-07-15",
    )

    assert path == tmp_path / "cache" / "miniqmt_kline" / "1d" / "front" / "sh_600000.csv"
    assert path == miniqmt_cache_path("sh.600000", "1d", "front")
    assert len(loaded) == 1
    assert loaded.iloc[0]["code"] == "sh.600000"
    assert loaded.iloc[0]["close"] == pytest.approx(11.5)


def test_normalize_miniqmt_frame_requires_date():
    with pytest.raises(ValueError, match="no date column"):
        normalize_miniqmt_frame(pd.DataFrame({"close": [1.0]}), "600000")


def test_compare_price_frames_reports_differences():
    left = pd.DataFrame({
        "date": ["2026-07-14", "2026-07-15"],
        "open": [10.0, 11.0],
        "high": [10.5, 11.5],
        "low": [9.5, 10.5],
        "close": [10.2, 11.2],
        "volume": [100, 200],
    })
    right = left.copy()
    right.loc[1, "close"] = 11.3

    result = compare_price_frames({"sh.600000": left}, {"sh.600000": right}, tolerance=0.01)

    assert result["compared_codes"] == 1
    assert result["mismatch_cells"] == 1
    assert result["sample"][0]["max_abs_diff"]["close"] == pytest.approx(0.1)


def test_load_miniqmt_price_frames_reports_missing_without_refresh(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCK_RESEARCH_VAR", str(tmp_path))

    frames, summary = load_miniqmt_price_frames(
        ["sh.600000", "sz.000001"],
        start_date="2026-07-14",
        end_date="2026-07-15",
        refresh=False,
        persist=False,
    )

    assert frames == {}
    assert summary["missing_count"] == 2
    assert summary["missing_sample"] == ["sh.600000", "sz.000001"]


def test_load_miniqmt_price_frames_requires_requested_end_coverage(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCK_RESEARCH_VAR", str(tmp_path))
    save_miniqmt_frame(
        "sh.600000",
        pd.DataFrame({
            "date": ["2026-07-15"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [1000],
            "amount": [10_200],
        }),
        period="1d",
        dividend_type="front",
    )

    frames, summary = load_miniqmt_price_frames(
        ["sh.600000"],
        start_date="2026-07-14",
        end_date="2026-07-16",
        refresh=False,
        persist=False,
    )

    assert frames == {}
    assert summary["missing_count"] == 1
    assert summary["missing_sample"] == ["sh.600000"]
