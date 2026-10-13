import pandas as pd
import pytest
import json
import hashlib

from stock_research.research.quantsplaybook_execution_aligned_research import (
    build_execution_aligned_labels,
    label_execution_path,
    load_candidate_snapshot_panel,
    prepare_execution_frame,
)


def _bars(periods=24):
    dates = pd.bdate_range("2026-01-05", periods=periods)
    return pd.DataFrame({
        "date": dates,
        "code": "sh.600000",
        "open": [10.0] * periods,
        "high": [10.2] * periods,
        "low": [9.8] * periods,
        "close": [10.0] * periods,
        "raw_to_qfq_factor": [1.0] * periods,
    })


def test_execution_label_buys_next_open_and_exits_on_twentieth_holding_bar():
    bars = _bars()
    result = label_execution_path(
        bars,
        bars.iloc[0]["date"],
        expected_entry_date=bars.iloc[1]["date"],
        code="sh.600000",
        commission_rate=0,
        sell_stamp_duty_rate=0,
        slippage_rate=0,
    )

    assert result["execution_label_available"] is True
    assert result["entry_date"] == bars.iloc[1]["date"].strftime("%Y-%m-%d")
    assert result["exit_date"] == bars.iloc[20]["date"].strftime("%Y-%m-%d")
    assert result["holding_bars"] == 20
    assert result["exit_reason_cn"] == "固定持有20个交易日"
    assert result["gross_return"] == 0


def test_execution_label_does_not_stop_on_entry_day_but_stops_after_t_plus_one():
    bars = _bars()
    bars.loc[1, "low"] = 8.0
    bars.loc[2, ["open", "high", "low", "close"]] = [10.0, 10.1, 9.0, 9.5]

    result = label_execution_path(
        bars,
        bars.iloc[0]["date"],
        expected_entry_date=bars.iloc[1]["date"],
        code="sh.600000",
        commission_rate=0,
        sell_stamp_duty_rate=0,
        slippage_rate=0,
    )

    assert result["exit_date"] == bars.iloc[2]["date"].strftime("%Y-%m-%d")
    assert result["exit_reason_cn"] == "固定失效距离保护止损"
    assert result["gross_return"] == pytest.approx(-0.08)
    assert result["realized_r_multiple"] == pytest.approx(-1.0)


def test_execution_label_rejects_locked_limit_up_entry():
    bars = _bars()
    bars.loc[0, ["open", "high", "low", "close"]] = [10.0] * 4
    bars.loc[1, ["open", "high", "low", "close"]] = [11.0] * 4

    result = label_execution_path(
        bars,
        bars.iloc[0]["date"],
        expected_entry_date=bars.iloc[1]["date"],
        code="sh.600000",
    )

    assert result["execution_label_available"] is False
    assert result["execution_status_cn"] == "下一交易日开盘未成交"
    assert result["entry_fill_status"] == "locked_limit_up"


def test_batch_labels_use_market_next_date_and_never_add_future_fields_to_panel():
    bars = _bars()
    panel = pd.DataFrame([{
        "date": bars.iloc[0]["date"],
        "code": "sh.600000",
        "playbook_factor_rank": 1,
        "candidate_score": 100.0,
    }])

    result = build_execution_aligned_labels(
        panel,
        bars,
        bars["date"],
    )

    assert result.iloc[0]["entry_date"] == bars.iloc[1]["date"].strftime("%Y-%m-%d")
    assert "net_return" in result
    assert "net_return" not in panel


def test_execution_label_excludes_exit_crossing_calendar_year_boundary():
    dates = pd.bdate_range("2025-12-01", periods=30)
    bars = _bars(periods=len(dates))
    bars["date"] = dates

    result = label_execution_path(
        bars,
        pd.Timestamp("2025-12-15"),
        expected_entry_date=pd.Timestamp("2025-12-16"),
        code="sh.600000",
    )

    assert result["execution_label_available"] is False
    assert result["execution_status_cn"] == "退出跨越信号年度边界"
    assert result["entry_date"] == "2025-12-16"
    assert "net_return" not in result


def test_snapshot_panel_does_not_require_legacy_outcome_file(tmp_path):
    directory = tmp_path / "candidates"
    directory.mkdir()
    snapshot = directory / "candidates_2026-01-05.csv"
    snapshot.write_text(
        "code,playbook_factor_rank,candidate_score\n"
        "sh.600000,1,99\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({
        "start_date": "2026-01-05",
        "end_date": "2026-01-05",
        "factor": "测试因子",
        "snapshots": [{
            "date": "2026-01-05",
            "file": snapshot.name,
            "sha256": digest,
        }],
    }, ensure_ascii=False), encoding="utf-8")

    panel, sources = load_candidate_snapshot_panel([directory])

    assert panel[["code", "playbook_factor_rank"]].to_dict("records") == [{
        "code": "sh.600000", "playbook_factor_rank": 1,
    }]
    assert sources[0]["factor"] == "测试因子"


def test_prepare_execution_frame_never_backfills_future_adjustment_factor():
    bars = _bars()
    bars.loc[0, "raw_to_qfq_factor"] = pd.NA

    prepared = prepare_execution_frame(bars)

    assert pd.isna(prepared.loc[0, "qfq_to_raw_factor"])
    assert prepared.loc[1, "qfq_to_raw_factor"] == 1.0


def test_execution_label_fails_closed_when_entry_adjustment_factor_is_missing():
    bars = _bars()
    bars.loc[1, "raw_to_qfq_factor"] = pd.NA

    result = label_execution_path(
        bars,
        bars.iloc[0]["date"],
        expected_entry_date=bars.iloc[1]["date"],
        code="sh.600000",
    )

    assert result["execution_label_available"] is False
    assert result["execution_status_cn"] == "开仓日复权因子不完整"
