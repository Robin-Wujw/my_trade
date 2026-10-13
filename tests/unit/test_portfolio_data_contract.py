import json

import pandas as pd
import pytest

from stock_research.backtesting.workflow import (
    load_price_frames, validate_daily_price_limit_coverage, validate_price_frame_coverage,
    validate_candidate_manifest_financial_point_in_time,
)


def test_missing_limit_columns_are_reported_without_crashing():
    frames = {"A": pd.DataFrame({"date": ["2025-01-02"], "volume": [100.]})}
    result = validate_daily_price_limit_coverage(frames, "2025-01-02", "2025-01-02")
    assert result["missing_bar_count"] == 1
    assert not result["complete"]


def test_file_values_do_not_claim_verified_tushare_provenance():
    frames = {"A": pd.DataFrame({"date": ["2025-01-02"], "volume": [100.],
                                "up_limit": [11.], "down_limit": [9.]})}
    result = validate_daily_price_limit_coverage(frames, "2025-01-02", "2025-01-02")
    assert result["covered_bar_count"] == 1
    assert result["unverified_covered_bar_count"] == 1
    assert result["verified_tushare_bar_count"] == 0
    assert result["source"] != "local_tushare_stk_limit"


def test_known_qfq_cache_cannot_supply_raw_execution_prices(tmp_path):
    directory = tmp_path / "akshare"
    directory.mkdir()
    pd.DataFrame({"date": ["2025-01-02"], "open": [5.], "high": [5.], "low": [5.],
                  "close": [5.], "volume": [100.]}).to_csv(directory / "sz_000001.csv", index=False)
    with pytest.raises(ValueError, match="Raw execution prices"):
        load_price_frames(["sz.000001"], directory, prefer_database=False, attach_daily_limits=False)


def test_unexplained_internal_gap_is_not_assumed_to_be_suspension():
    frames = {"A": pd.DataFrame({"date": ["2025-01-02", "2025-01-06"]})}
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    with pytest.raises(RuntimeError, match="unexplained_internal_gaps"):
        validate_price_frame_coverage(frames, ["A"], "2025-01-02", "2025-01-06", trade_calendar=calendar)
    diagnostic = validate_price_frame_coverage(frames, ["A"], "2025-01-02", "2025-01-06",
                                              trade_calendar=calendar, require_internal_complete=False)
    assert diagnostic["internal_missing_bar_count"] == 1
    covered = validate_price_frame_coverage(frames, ["A"], "2025-01-02", "2025-01-06",
                                            trade_calendar=calendar,
                                            security_suspension_dates={"A": [pd.Timestamp("2025-01-03")]})
    assert covered["internal_missing_bar_count"] == 0


@pytest.mark.parametrize("value", ["true", "false", 1, 0])
def test_financial_manifest_boolean_cannot_bypass_strict_gate(tmp_path, value):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "uses_financial_data": value, "financial_point_in_time": True,
        "strict_financial_point_in_time": False, "snapshots": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        validate_candidate_manifest_financial_point_in_time(tmp_path)


def test_all_dividend_sync_paths_request_settlement_and_implementation_dates():
    from apps.data_sync import TUSHARE_DEFAULT_FIELDS
    from apps.tushare_bulk_sync import BULK_DATASET_FIELDS
    from apps.tushare_dividend_ex_date_sync import DIVIDEND_FIELDS
    for value in (TUSHARE_DEFAULT_FIELDS["dividend"], BULK_DATASET_FIELDS["dividend"], DIVIDEND_FIELDS):
        assert {"record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date"}.issubset(value.split(","))


def test_raw_qfq_alignment_preserves_a_sliced_frame_index(tmp_path):
    from stock_research.backtesting.workflow import _attach_raw_adjustment_factor
    raw_dir = tmp_path / "akshare_raw"
    raw_dir.mkdir()
    qfq_dir = tmp_path / "akshare"
    qfq_dir.mkdir()
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    pd.DataFrame({"date": dates, "close": [5., 10., 15.]}).to_csv(qfq_dir / "sz_000001.csv", index=False)
    raw = pd.DataFrame({"date": dates, "close": [10., 20., 30.]}).iloc[1:]
    result = _attach_raw_adjustment_factor(raw, raw_dir, "sz.000001")
    assert result.index.tolist() == [1, 2]
    assert result["raw_to_qfq_factor"].tolist() == [2., 2.]
