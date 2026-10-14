import numpy as np
import pandas as pd
import pytest

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
    assert all(row["research_only"] for row in rows)


def test_future_ipo_cannot_change_earlier_cross_section_scores():
    prices = _prices()
    cutoff = sorted(prices["date"].unique())[-21]
    expected = calculate_technical_lanes(prices[prices["date"].le(cutoff)])
    future_ipo = prices[prices["code"].eq("sh.600001") & prices["date"].gt(cutoff)].copy()
    future_ipo["code"] = "FUTURE_IPO"
    actual = calculate_technical_lanes(pd.concat([prices, future_ipo], ignore_index=True))
    actual = actual[actual["date"].le(cutoff)]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


@pytest.mark.parametrize("invalid_factor", [np.nan, 0.0, -1.0, np.inf, -np.inf])
def test_explicit_invalid_adjustment_factor_fails_closed(invalid_factor):
    prices = _prices()
    prices.loc[prices.index[-1], "raw_to_qfq_factor"] = invalid_factor
    with pytest.raises(ValueError, match="complete, finite and positive"):
        calculate_technical_lanes(prices)


def test_missing_adjustment_column_fails_closed():
    with pytest.raises(ValueError, match="raw_to_qfq_factor is required"):
        calculate_technical_lanes(_prices().drop(columns="raw_to_qfq_factor"))


def test_split_adjustment_preserves_signal_scores():
    prices = _prices()
    expected = calculate_technical_lanes(prices)
    split = prices.copy()
    mask = split["code"].eq("sh.600001") & split["date"].ge("2022-08-01")
    split.loc[mask, ["close", "high", "low"]] /= 2.0
    split.loc[mask, "raw_to_qfq_factor"] = 0.5
    pd.testing.assert_frame_equal(expected, calculate_technical_lanes(split))


def test_ineligible_stocks_cannot_change_rank_but_history_is_retained():
    prices = _prices()
    allowed = prices[prices["code"].ne("sh.600003")]
    membership = allowed[["date", "code"]]
    # Only the last day is requested; the earlier bars still supply warmup.
    membership = membership[membership["date"].eq(membership["date"].max())]
    expected = calculate_technical_lanes(allowed).tail(2).reset_index(drop=True)
    actual = calculate_technical_lanes(prices, eligible_membership=membership)
    pd.testing.assert_frame_equal(expected, actual.reset_index(drop=True))


def test_no_trade_rows_do_not_participate_in_cross_section():
    prices = _prices()
    expected = calculate_technical_lanes(prices[prices["code"].ne("sh.600003")])
    prices["tradestatus"] = 1
    prices.loc[prices["code"].eq("sh.600003"), "tradestatus"] = 0
    actual = calculate_technical_lanes(prices)
    pd.testing.assert_frame_equal(expected, actual)


def test_recovered_cli_generates_explicitly_research_only_snapshots(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from apps import three_lane_candidates as app

    prices = _prices()
    dates = sorted(prices["date"].unique())[-2:]
    panel_root = tmp_path / "source/research_panels"
    panel_root.mkdir(parents=True)
    prices[["date", "code"]].to_pickle(panel_root / "factor_panel_2023.pkl")
    monkeypatch.setattr(app, "load_statement_history", lambda *args, **kwargs: _statements())
    monkeypatch.setattr(app, "_formula_snapshot_dates", lambda *args, **kwargs: dates)
    monkeypatch.setattr(
        app, "TushareRepository",
        lambda database: SimpleNamespace(load_market_daily_frame=lambda **kwargs: prices),
    )
    output = app.generate_candidates(
        database_path=tmp_path / "unused.sqlite3", source_root=tmp_path / "source",
        formula_history=tmp_path / "unused.csv", output_directory=tmp_path / "candidates",
        start_date=pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["strict_financial_point_in_time"] is False
    assert manifest["signal_timing"] == "close_t_signal_earliest_execution_t_plus_1"
    assert manifest["snapshot_count"] == 2
    for snapshot in manifest["snapshots"]:
        path = output / snapshot["file"]
        rows = pd.read_csv(path)
        assert not rows.empty
        assert rows["research_only"].all()
        assert snapshot["research_only"] is True
        assert snapshot["strict_financial_point_in_time"] is False
        assert snapshot["sha256"] == app._sha256(path)
