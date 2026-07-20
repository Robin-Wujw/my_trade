import json

import pandas as pd
import pytest

from stock_research.pipelines import fundamental_selection as selection_pipeline
from stock_research.pipelines.daily_report import ensure_same_observation_date
from stock_research.strategies.fundamental_selection import (
    growth_risk,
    is_value_industry_allowed,
    quality_detail,
    value_method_reason,
)
from stock_research.pipelines.factor_selection import classify_method


def test_fundamental_explanations_keep_current_wording():
    detail = quality_detail(1.50, 0.50, 100)
    assert "扣非EPS为1.50元，盈利能力较强" in detail
    assert "扣非利润同比50.0%，增长较强" in detail
    assert "近年扣非盈利稳定性较高" in detail
    reason = value_method_reason("通信网络设备及器件", 120, 1.20, 0.20)
    assert "命中基本价值线行业白名单" in reason
    assert "总市值120.0亿元（不低于100亿元）" in reason
    assert "超过300%" in growth_risk(3.01)
    assert "同比为负" in growth_risk(-0.01)


@pytest.mark.parametrize(
    "industry",
    [
        "汽车电子电气系统", "通信网络设备及器件", "半导体",
        "半导体材料", "半导体设备", "分立器件",
    ],
)
def test_value_industry_allowlist_accepts_requested_manufacturing_groups(industry):
    assert is_value_industry_allowed(industry)
    assert classify_method(industry) == "VALUE"


@pytest.mark.parametrize(
    "industry", ["半导体软件", "电子", "通信设备", "饰品", "医疗服务", "", None],
)
def test_value_industry_allowlist_fails_closed(industry):
    assert not is_value_industry_allowed(industry)
    if industry:
        assert classify_method(industry) != "VALUE"


def test_report_rejects_mixed_observation_dates():
    with pytest.raises(ValueError, match="observation date mismatch"):
        ensure_same_observation_date(
            {"formula33": "2026-07-02", "selection": "2026-07-03"}
        )


def _write_kline(path, dates, volumes=None):
    volumes = volumes or [1000] * len(dates)
    pd.DataFrame(
        {
            "date": dates,
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.0] * len(dates),
            "volume": volumes,
        }
    ).to_csv(path, index=False)


def test_load_kline_truncates_future_rows_at_observation_date(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(selection_pipeline, "KLINE_CACHE_DIR", str(tmp_path))
    _write_kline(
        tmp_path / "sh_600000.csv",
        ["2026-07-09", "2026-07-10", "2026-07-13"],
    )

    result = selection_pipeline.load_kline("sh.600000", "2026-07-10")

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-09",
        "2026-07-10",
    ]


@pytest.mark.parametrize(
    ("dates", "volumes"),
    [
        (["2026-07-09"], [1000]),
        (["2026-07-09", "2026-07-10"], [1000, 0]),
    ],
)
def test_load_kline_rejects_stock_without_observation_day_trade(
    monkeypatch,
    tmp_path,
    dates,
    volumes,
):
    monkeypatch.setattr(selection_pipeline, "KLINE_CACHE_DIR", str(tmp_path))
    _write_kline(tmp_path / "sh_688072.csv", dates, volumes)

    result = selection_pipeline.load_kline("sh.688072", "2026-07-10")

    assert result.empty


def test_normal_selection_excludes_stale_kline_candidate(monkeypatch):
    snapshot = pd.DataFrame(
        [
            {
                "code": "sh.600000",
                "name": "current",
                "industry": "银行",
                "quality_score": 90,
                "liquidity_score": 80,
                "mktcap": 500,
                "earnings_yoy": 0.30,
                "pool_member": True,
            },
            {
                "code": "sh.688072",
                "name": "stale",
                "industry": "银行",
                "quality_score": 90,
                "liquidity_score": 80,
                "mktcap": 500,
                "earnings_yoy": 0.30,
                "pool_member": True,
            },
        ]
    )
    dates = pd.bdate_range(end="2026-07-10", periods=65)
    current = pd.DataFrame(
        {
            "date": dates,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 1000.0,
        }
    )
    monkeypatch.setattr(
        selection_pipeline,
        "load_kline",
        lambda code, observation_date: (
            current.copy() if code == "sh.600000" else pd.DataFrame()
        ),
    )

    result = selection_pipeline.normal_rows(
        "2026-03-31",
        snapshot,
        {},
        {},
        "2026-07-10",
    )

    assert result["code"].tolist() == ["sh.600000"]


def test_value_selection_uses_100e_market_cap_without_leader_proxy(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(selection_pipeline, "VALUE_CACHE_DIR", str(tmp_path))
    for symbol, shares in [("600000", 500_000_000), ("600001", 499_000_000)]:
        (tmp_path / f"{symbol}_20260331.json").write_text(
            json.dumps(
                {
                    "value_line": 25,
                    "eps_excl": 1.2,
                    "yoy": 0.05,
                    "quality_score": 72,
                    "total_share": shares,
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(selection_pipeline, "load_kline", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        selection_pipeline,
        "technical_fields",
        lambda _frame: {"date": "2026-07-10", "close": 20.0},
    )
    snapshot = pd.DataFrame(
        [
            {
                "code": "sh.600000",
                "industry": "汽车电子电气系统",
                "theme": "汽车电子",
                "method": "VALUE",
                "value_applicability_status": "not_proven",
            },
            {
                "code": "sh.600001",
                "industry": "汽车电子电气系统",
                "theme": "汽车电子",
                "method": "VALUE",
                "value_applicability_status": "rule_eligible",
            },
        ]
    )
    method_routes = selection_pipeline.method_routes_from_snapshot(snapshot)

    result = selection_pipeline.value_rows(
        "2026-03-31",
        {"sh.600000": "百亿公司", "sh.600001": "不足百亿公司"},
        snapshot,
        1.08,
        method_routes,
        {},
        "2026-07-10",
    )

    assert result["code"].tolist() == ["sh.600000"]
    assert result.iloc[0]["mktcap"] == pytest.approx(100.0)
    assert "龙头" not in result.iloc[0]["selection_reason"]
