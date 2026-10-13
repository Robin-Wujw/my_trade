from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_research.dashboard.catalog import (
    discover_backtests,
    drawdown_frame,
    filter_trades,
    load_table,
    quality_badges,
    summary_metrics,
)


def _write_artifact(root, name="run"):
    directory = root / name
    directory.mkdir()
    summary = {
        "end_date": "2026-07-21",
        "final_return_pct": 12.3,
        "maximum_drawdown_pct": -4.5,
        "trade_summary": {"trade_count": 2},
        "research_only": True,
        "acceptance_status": "data_limited",
        "strict_point_in_time": False,
        "close_confirmed_execution": "next_open",
        "coverage_complete": False,
    }
    (directory / "portfolio_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "portfolio_equity.csv").write_text(
        "date,equity\n2026-01-01,1.0\n2026-01-02,1.1\n2026-01-03,1.05\n",
        encoding="utf-8",
    )
    (directory / "portfolio_trades.csv").write_text(
        "date,code,action,reason\n2026-01-02,sh.600000,买入,signal\n2026-01-03,sh.600000,卖出,stop\n",
        encoding="utf-8",
    )
    return directory


def test_discover_backtests_and_load_companion_files(tmp_path):
    directory = _write_artifact(tmp_path)
    artifacts = discover_backtests(tmp_path)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.directory == directory
    assert artifact.equity_path is not None
    assert artifact.trades_path is not None
    assert artifact.summary["final_return_pct"] == 12.3


def test_drawdown_and_trade_filter_are_deterministic(tmp_path):
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"], "equity": [1.0, 1.1, 1.05]})
    result = drawdown_frame(frame)
    assert result["drawdown_pct"].iloc[-1] == pytest.approx(-4.545454545454541)
    trades = pd.DataFrame(
        {"code": ["sh.600000", "sz.000001"], "action": ["买入", "卖出"], "reason": ["signal", "stop"]}
    )
    filtered = filter_trades(trades, code="sh.600000", keyword="signal")
    assert len(filtered) == 1
    assert filtered.iloc[0]["action"] == "买入"


def test_quality_badges_and_metric_order(tmp_path):
    directory = _write_artifact(tmp_path)
    artifact = discover_backtests(tmp_path)[0]
    labels = [label for label, _ in quality_badges(artifact.summary)]
    assert labels == ["研究回测", "数据受限", "非严格时点", "next_open", "候选覆盖不完整"]
    assert [label for label, _ in summary_metrics(artifact.summary)][:3] == ["最终收益", "最大回撤", "交易笔数"]
    table = load_table(artifact.equity_path)
    assert isinstance(table["date"].iloc[0], pd.Timestamp)
