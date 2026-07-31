import json

import pandas as pd

from apps.quantsplaybook_compare import (
    aggregate_portfolio_summaries,
    audit_portfolio_trades,
    evaluate_factor_panel,
    factor_candidate_snapshots,
    load_prepared_price_cache,
    rank_weighted_factor_combination,
    point_in_time_tradable_universe,
    required_history_start_date,
    save_factor_snapshots,
)


def test_point_in_time_universe_applies_only_validity_rules():
    prices = pd.DataFrame([
        {
            "date": date,
            "code": code,
            "close": 10.0,
            "high": 10.2,
            "low": 9.8,
            "volume": volume,
            "amount": volume * 10,
            "tradestatus": "1" if volume else "0",
        }
        for date in pd.date_range("2024-01-02", periods=3)
        for code, volume in (
            ("sh.600001", 1000),
            ("sz.000002", 1000),
            ("sz.000003", 1000),
            ("sz.000004", 0),
        )
    ])
    basic = pd.DataFrame([
        {
            "code": "sh.600001",
            "list_date": "2020-01-01",
            "delist_date": None,
        },
        {
            "code": "sz.000002",
            "list_date": "2024-01-03",
            "delist_date": None,
        },
        {
            "code": "sz.000003",
            "list_date": "2020-01-01",
            "delist_date": "2024-01-03",
        },
        {
            "code": "sz.000004",
            "list_date": "2020-01-01",
            "delist_date": None,
        },
    ])
    basic[["list_date", "delist_date"]] = basic[
        ["list_date", "delist_date"]
    ].apply(pd.to_datetime)

    membership = point_in_time_tradable_universe(
        prices,
        basic,
        start_date="2024-01-02",
        end_date="2024-01-04",
        minimum_history=1,
    )

    assert set(map(tuple, membership.to_numpy())) == {
        (pd.Timestamp("2024-01-02"), "sh.600001"),
        (pd.Timestamp("2024-01-03"), "sh.600001"),
        (pd.Timestamp("2024-01-04"), "sh.600001"),
        (pd.Timestamp("2024-01-02"), "sz.000003"),
        (pd.Timestamp("2024-01-03"), "sz.000003"),
    }


def test_point_in_time_universe_excludes_dated_st_rows_only():
    prices = pd.DataFrame([
        {
            "date": date,
            "code": "sh.600001",
            "close": 10.0,
            "high": 10.2,
            "low": 9.8,
            "volume": 1000,
            "amount": 10_000,
            "tradestatus": "1",
        }
        for date in pd.date_range("2024-01-02", periods=3)
    ])
    basic = pd.DataFrame([{
        "code": "sh.600001",
        "list_date": pd.Timestamp("2020-01-01"),
        "delist_date": pd.NaT,
    }])
    stock_st = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-03"),
        "code": "sh.600001",
    }])

    membership = point_in_time_tradable_universe(
        prices,
        basic,
        start_date="2024-01-02",
        end_date="2024-01-04",
        minimum_history=1,
        stock_st=stock_st,
    )

    assert membership["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
    ]


def test_point_in_time_universe_excludes_beijing_pre_open_history():
    prices = pd.DataFrame([
        {
            "date": date,
            "code": "bj.832317",
            "close": 10.0,
            "high": 10.2,
            "low": 9.8,
            "volume": 1000,
            "amount": 10_000,
            "tradestatus": "1",
        }
        for date in pd.to_datetime(["2021-11-12", "2021-11-15"])
    ])
    basic = pd.DataFrame([{
        "code": "bj.832317",
        "list_date": pd.Timestamp("2020-07-27"),
        "delist_date": pd.NaT,
    }])

    membership = point_in_time_tradable_universe(
        prices,
        basic,
        start_date="2021-11-12",
        end_date="2021-11-15",
        minimum_history=1,
    )

    assert membership["date"].tolist() == [pd.Timestamp("2021-11-15")]


def test_factor_snapshots_select_top_n_from_full_membership():
    date = pd.Timestamp("2024-06-03")
    index = pd.MultiIndex.from_tuples(
        [(date, code) for code in ("A", "B", "C")],
        names=["date", "code"],
    )
    panel = pd.DataFrame(
        {"high_quality_momentum": [0.1, 0.9, 0.5]},
        index=index,
    )
    membership = index.to_frame(index=False)

    rows = factor_candidate_snapshots(
        panel,
        membership,
        factor_column="high_quality_momentum",
        snapshot_dates=[date],
        top_n=2,
    )["2024-06-03"]

    assert [row["code"] for row in rows] == ["B", "C"]
    assert [row["playbook_factor_rank"] for row in rows] == [1, 2]
    assert all(row["name"] == row["code"] for row in rows)
    assert all(
        row["selection_profile"] == "quantsplaybook_factor_only"
        for row in rows
    )
    forbidden = {
        "quality_score",
        "growth_score",
        "market_cap",
        "industry",
        "formula33_selected",
        "right_quant_score",
    }
    assert not forbidden.intersection(rows[0])


def test_factor_evaluation_uses_future_return_only_as_label():
    dates = pd.bdate_range("2024-01-02", periods=30)
    codes = [f"S{index:03d}" for index in range(60)]
    price_rows = []
    factor_rows = []
    members = []
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(codes):
            close = 10 * (1 + (code_index + 1) / 100_000) ** date_index
            price_rows.append({
                "date": date,
                "code": code,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "raw_to_qfq_factor": 1.0,
            })
            factor_rows.append((date, code, float(code_index)))
            members.append({"date": date, "code": code})
    index = pd.MultiIndex.from_tuples(
        [(date, code) for date, code, _ in factor_rows],
        names=["date", "code"],
    )
    panel = pd.DataFrame(
        {"high_quality_momentum": [value for _, _, value in factor_rows]},
        index=index,
    )

    metrics = evaluate_factor_panel(
        panel,
        pd.DataFrame(price_rows),
        pd.DataFrame(members),
        holding_days=5,
        start_date="2024-01-02",
        end_date="2024-02-29",
    )

    momentum = metrics.set_index("factor").loc["high_quality_momentum"]
    assert momentum["mean_rank_ic"] > 0.99
    assert momentum["top20_excess_forward_return"] > 0
    assert momentum["date_count"] == 25


def test_rank_weighted_combination_renormalizes_missing_factor():
    date = pd.Timestamp("2024-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "A"), (date, "B")],
        names=["date", "code"],
    )
    panel = pd.DataFrame({
        "factor_a": [1.0, 2.0],
        "factor_b": [2.0, None],
    }, index=index)

    result = rank_weighted_factor_combination(
        panel,
        {"factor_a": 0.25, "factor_b": 0.75},
    )

    assert result.loc[(date, "A")] == 0.875
    assert result.loc[(date, "B")] == 1.0


def test_trade_audit_requires_prior_candidate_and_raw_bar_fill():
    frame = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-03"),
        "raw_low": 9.8,
        "raw_high": 10.2,
    }])
    result = {"trade_ledger": [{
        "date": "2024-01-03",
        "code": "sh.600001",
        "trade_side": "买入",
        "execution_price": 10.0,
        "candidate_snapshot_date": "2024-01-02",
        "technical_signal_date": "2024-01-03",
        "trigger": 9.9,
        "trigger_raw_price": 9.9,
    }]}

    audit, summary = audit_portfolio_trades(
        result,
        {"sh.600001": frame},
    )

    assert audit.iloc[0]["violations"] == ""
    assert summary["violation_count"] == 0
    assert summary["all_buy_candidates_strictly_before_execution"] is True


def test_aggregate_portfolio_summaries_reads_json_not_stale_csv(tmp_path):
    output = tmp_path / "portfolio" / "factor_a"
    output.mkdir(parents=True)
    (output / "portfolio_2024-01-01_2024-12-31_summary.json").write_text(
        json.dumps({
            "requested_start": "2024-01-01",
            "actual_start": "2024-01-02",
            "end_date": "2024-12-31",
            "coverage_complete": True,
            "final_return_pct": 12.5,
            "maximum_drawdown_pct": -8.0,
            "trade_summary": {"sell_count": 1, "sell_win_rate_pct": 100.0},
            "trade_ledger": [],
            "profit_concentration_summary": {},
            "trade_audit_summary": {"violation_count": 0},
        }),
        encoding="utf-8",
    )
    (tmp_path / "portfolio_metrics.csv").write_text(
        "factor,final_return_pct\nstale,-99\n",
        encoding="utf-8",
    )

    result = aggregate_portfolio_summaries(tmp_path)

    assert result["factor"].tolist() == ["factor_a"]
    assert result.iloc[0]["final_return_pct"] == 12.5


def test_prepared_price_cache_requires_every_requested_code(tmp_path):
    pd.DataFrame([{
        "date": pd.Timestamp("2024-01-02"),
        "close": 10.0,
    }]).to_pickle(tmp_path / "sh_600001.pkl")

    loaded = load_prepared_price_cache(tmp_path, {"sh.600001"})

    assert list(loaded) == ["sh.600001"]
    assert loaded["sh.600001"].iloc[0]["date"] == pd.Timestamp("2024-01-02")


def test_saved_manifest_declares_close_t_to_t_plus_one_timing(tmp_path):
    output = save_factor_snapshots(
        {"2024-01-02": [{"code": "sh.600001"}]},
        output_directory=tmp_path,
        factor_column="high_quality_momentum",
        top_n=1,
        start_date="2024-01-02",
        end_date="2024-01-02",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["old_candidate_pool_used"] is False
    assert manifest["signal_timing"] == (
        "close_t_signal_earliest_execution_t_plus_1"
    )


def test_factor_replay_requires_same_700_day_execution_warmup():
    assert required_history_start_date("2026-01-01") == pd.Timestamp(
        "2024-02-01",
    )
