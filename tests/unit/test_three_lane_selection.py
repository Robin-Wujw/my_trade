import numpy as np
import pandas as pd

from stock_research.strategies.three_lane_selection import (
    build_fundamental_events,
    calculate_technical_lanes,
    map_fundamental_lane,
    quota_union_candidates,
)


def _statements():
    fina = pd.DataFrame([
        {"code": "sh.600001", "report_period": "2022-12-31", "ann_date": "2023-03-20", "dt_netprofit_yoy": 10, "roe_dt": 8},
        {"code": "sh.600001", "report_period": "2023-03-31", "ann_date": "2023-04-20", "dt_netprofit_yoy": 20, "roe_dt": 9},
        {"code": "sh.600001", "report_period": "2023-03-31", "ann_date": "2023-05-20", "dt_netprofit_yoy": 40, "roe_dt": 10},
    ])
    income = pd.DataFrame([
        {"code": "sh.600001", "report_period": "2022-03-31", "ann_date": "2022-04-20", "total_revenue": 100, "n_income_attr_p": 10},
        {"code": "sh.600001", "report_period": "2023-03-31", "ann_date": "2023-04-20", "total_revenue": 120, "n_income_attr_p": 12},
    ])
    return {
        "fina_indicator": fina,
        "income": income,
        "balancesheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }


def test_fundamental_revision_is_not_visible_before_announcement():
    events = build_fundamental_events(_statements())
    membership = pd.DataFrame({
        "date": pd.to_datetime(["2023-05-10", "2023-05-22"]),
        "code": ["sh.600001", "sh.600001"],
    })
    mapped = map_fundamental_lane(events, membership)

    assert mapped["profit_yoy"].tolist() == [20.0, 40.0]
    assert mapped["effective_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-04-20", "2023-05-20",
    ]


def _prices(days=300, codes=3):
    dates = pd.bdate_range("2022-01-03", periods=days)
    rows = []
    for number in range(codes):
        for index, date in enumerate(dates):
            close = 10 * (1.001 + number * 0.0001) ** index
            rows.append({
                "date": date,
                "code": f"sh.{600001 + number:06d}",
                "close": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "volume": 100000 + (index % 10) * 100,
                "turnover_rate": 1.0 + number * 0.1,
                "raw_to_qfq_factor": 1.0,
            })
    return pd.DataFrame(rows)


def test_technical_lanes_do_not_change_when_future_bars_change():
    prices = _prices()
    cutoff = pd.Timestamp("2023-01-31")
    before = calculate_technical_lanes(prices[prices["date"].le(cutoff)])
    changed = prices.copy()
    changed.loc[changed["date"].gt(cutoff), ["close", "high", "low"]] *= 10
    after = calculate_technical_lanes(changed)
    shared = before.merge(after, on=["date", "code"], suffixes=("_before", "_after"))

    for lane in ("smooth_52week_high", "stage2_vcp"):
        np.testing.assert_allclose(
            shared[f"{lane}_before"], shared[f"{lane}_after"], equal_nan=True,
        )


def test_quota_union_rotates_lanes_and_deduplicates():
    scores = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E", "F"],
        "fundamental_momentum": [1.0, 0.9, np.nan, np.nan, np.nan, np.nan],
        "smooth_52week_high": [1.0, np.nan, 0.9, np.nan, np.nan, np.nan],
        "stage2_vcp": [np.nan, np.nan, np.nan, 1.0, 0.9, np.nan],
    })

    rows = quota_union_candidates(scores, lane_top_n=2, maximum_candidates=5)

    assert [row["code"] for row in rows] == ["A", "D", "B", "C", "E"]
    assert rows[0]["three_lane_membership"] == (
        "fundamental_momentum+smooth_52week_high"
    )
