import hashlib
import json

import pandas as pd
import pytest

from stock_research.research.quantsplaybook_compare import (
    _drop_ineligible_invalid_adjustment_rows,
    _market_residual_factor_configuration,
    aggregate_portfolio_summaries,
    apply_stock_basic_display_names,
    audit_portfolio_trades,
    build_candidate_trading_profile_enrichment,
    build_parser,
    evaluate_factor_panel,
    factor_candidate_snapshots,
    load_prepared_price_cache,
    rank_weighted_factor_combination,
    point_in_time_tradable_universe,
    prepare_factor_candidates,
    required_history_start_date,
    save_factor_snapshots,
)
from stock_research.strategies.quantsplaybook_selection import (
    market_residual_momentum_factor_metadata,
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


def test_stock_basic_names_are_added_for_display_only():
    trade = {
        "code": "sh.600001",
        "name": "sh.600001",
        "execution_price": 10.5,
        "candidate_snapshot_date": "2024-01-02",
    }
    result = {
        "events": [trade],
        "trade_ledger": [trade],
        "final_positions": [{"code": "sh.600001", "name": "sh.600001"}],
        "profit_concentration_summary": {
            "top1_symbol": {"code": "sh.600001", "name": "sh.600001"},
            "top_symbols": [{"code": "sh.600001", "name": "sh.600001"}],
        },
    }
    basic = pd.DataFrame([{
        "code": "sh.600001",
        "name": "浦发银行",
    }])

    apply_stock_basic_display_names(result, basic)

    assert trade["name"] == "浦发银行"
    assert trade["execution_price"] == 10.5
    assert trade["candidate_snapshot_date"] == "2024-01-02"
    assert result["final_positions"][0]["name"] == "浦发银行"
    assert result["profit_concentration_summary"]["top1_symbol"]["name"] == "浦发银行"
    assert result["display_name_source"]["decision_input"] is False
    assert result["display_name_source"]["unresolved_traded_codes"] == []


def test_quantsplaybook_portfolio_defaults_to_two_left_symbols():
    args = build_parser().parse_args([])

    assert args.max_left_positions == 2
    assert args.include_market_residual_momentum is False
    assert args.include_low_market_residual_volatility is False
    assert args.skip_candidate_generation is False


def test_compare_can_skip_single_factor_candidates_for_chunked_cache_builds():
    args = build_parser().parse_args(["--skip-candidate-generation"])

    assert args.skip_candidate_generation is True


def test_compare_market_residual_configuration_is_fail_closed_by_default():
    factors, metadata = _market_residual_factor_configuration(False)
    enabled_factors, enabled_metadata = _market_residual_factor_configuration(
        True,
    )
    args = build_parser().parse_args([
        "--include-market-residual-momentum",
    ])

    assert factors == []
    assert metadata == {}
    assert args.include_market_residual_momentum is True
    assert len(enabled_factors) == 4
    assert set(enabled_metadata) == set(enabled_factors)

    volatility_factors, volatility_metadata = (
        _market_residual_factor_configuration(False, True)
    )
    volatility_args = build_parser().parse_args([
        "--include-low-market-residual-volatility",
    ])
    assert volatility_args.include_low_market_residual_volatility is True
    assert volatility_factors == ["low_market_residual_volatility_20"]
    assert set(volatility_metadata) == set(volatility_factors)
    assert volatility_metadata[volatility_factors[0]][
        "financial_data_used"
    ] is False


def test_invalid_adjustment_is_dropped_only_outside_pit_membership():
    prices = pd.DataFrame([
        {
            "date": "2021-01-04",
            "code": "bj.920001",
            "raw_to_qfq_factor": None,
        },
        {
            "date": "2021-01-04",
            "code": "sh.600001",
            "raw_to_qfq_factor": 0.5,
        },
    ])
    membership = pd.DataFrame([{
        "date": "2021-01-04",
        "code": "sh.600001",
    }])

    result = _drop_ineligible_invalid_adjustment_rows(prices, membership)

    assert result["code"].tolist() == ["sh.600001"]


def test_invalid_adjustment_fails_closed_for_pit_tradable_row():
    prices = pd.DataFrame([{
        "date": "2021-01-04",
        "code": "sh.600001",
        "raw_to_qfq_factor": None,
    }])
    membership = prices[["date", "code"]].copy()

    with pytest.raises(ValueError, match="PIT-tradable rows"):
        _drop_ineligible_invalid_adjustment_rows(prices, membership)


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


def test_factor_candidates_save_top_100_but_only_top_50_tradeable(tmp_path):
    date = pd.Timestamp("2026-08-12")
    codes = [f"S{number:03d}" for number in range(100)]
    index = pd.MultiIndex.from_tuples(
        [(date, code) for code in codes],
        names=["date", "code"],
    )
    panel = pd.DataFrame(
        {"gross_profitability_forward": range(100)},
        index=index,
    )

    prepare_factor_candidates(
        panel,
        index.to_frame(index=False),
        factors=["gross_profitability_forward"],
        snapshot_dates=[date],
        output_root=tmp_path,
        top_n=100,
        trading_top_n=50,
        start_date="2026-08-12",
        end_date="2026-08-12",
        factor_metadata_by_name={
            "gross_profitability_forward": {"trading_top_n": 50},
        },
    )

    output = tmp_path / "candidates" / "gross_profitability_forward"
    candidates = pd.read_csv(output / "candidates_2026-08-12.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(candidates) == 100
    assert candidates.loc[:49, "selected_for_trading"].all()
    assert candidates.loc[:49, "signal_eligible"].all()
    assert not candidates.loc[50:, "selected_for_trading"].any()
    assert not candidates.loc[50:, "signal_eligible"].any()
    assert not candidates.loc[50:, "allow_right"].any()
    assert manifest["top_n"] == 100
    assert manifest["trading_top_n"] == 50
    assert manifest["snapshots"][0]["signal_eligible_count"] == 50


def test_factor_snapshots_merge_trading_profile_enrichment():
    date = pd.Timestamp("2024-06-03")
    index = pd.MultiIndex.from_tuples(
        [(date, code) for code in ("A", "B")],
        names=["date", "code"],
    )
    panel = pd.DataFrame(
        {"high_quality_momentum": [0.1, 0.9]},
        index=index,
    )
    membership = index.to_frame(index=False)

    rows = factor_candidate_snapshots(
        panel,
        membership,
        factor_column="high_quality_momentum",
        snapshot_dates=[date],
        top_n=2,
        candidate_enrichment_by_date={
            "2024-06-03": {
                "B": {
                    "trade_basis_score": 8.0,
                    "candidate_profile_status": "aligned_observation_bar",
                },
            },
        },
    )["2024-06-03"]

    assert rows[0]["code"] == "B"
    assert rows[0]["trade_basis_score"] == 8.0
    assert rows[0]["candidate_profile_status"] == "aligned_observation_bar"
    assert rows[1]["candidate_profile_status"] == "missing_price_profile"


def test_candidate_trading_profile_uses_snapshot_or_prior_bars_only():
    dates = pd.date_range("2024-01-02", periods=12)
    closes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 500, 600]
    prices = pd.DataFrame([
        {
            "date": date,
            "code": "A",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000 + index,
            "amount": close * (1000 + index),
            "tradestatus": 1,
        }
        for index, (date, close) in enumerate(zip(dates, closes))
    ])

    enrichment = build_candidate_trading_profile_enrichment(
        prices,
        [dates[9]],
        codes={"A"},
    )
    profile = enrichment[dates[9].strftime("%Y-%m-%d")]["A"]

    assert profile["candidate_profile_price_date"] == dates[9].strftime(
        "%Y-%m-%d",
    )
    assert profile["candidate_profile_status"] == "aligned_observation_bar"
    assert profile["return_5d"] == closes[9] / closes[4] - 1


def test_candidate_trading_profile_uses_qfq_prices_and_yuan_amounts():
    dates = pd.date_range("2024-01-02", periods=6)
    raw_closes = [10.0, 11.0, 12.0, 6.5, 7.0, 7.5]
    adjustment = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    prices = pd.DataFrame([
        {
            "date": date,
            "code": "A",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
            "amount": 100.0,
            "tradestatus": 1,
            "raw_to_qfq_factor": factor,
        }
        for date, close, factor in zip(dates, raw_closes, adjustment)
    ])

    enrichment = build_candidate_trading_profile_enrichment(
        prices,
        [dates[-1]],
        codes={"A"},
        amount_multiplier=1000.0,
    )
    profile = enrichment[dates[-1].strftime("%Y-%m-%d")]["A"]

    assert profile["return_5d"] == 0.5
    assert profile["avg_amount_20"] == 100_000.0


def test_prepare_factor_candidates_writes_research_outcome_labels(tmp_path):
    dates = pd.date_range("2024-01-02", periods=6)
    index = pd.MultiIndex.from_product(
        [dates[:3], ["A", "B"]],
        names=["date", "code"],
    )
    panel = pd.DataFrame(
        {"high_quality_momentum": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]},
        index=index,
    )
    membership = index.to_frame(index=False)
    prices = pd.DataFrame([
        {
            "date": date,
            "code": code,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
            "amount": close * 1000,
            "tradestatus": 1,
            "raw_to_qfq_factor": 1.0,
        }
        for date_index, date in enumerate(dates)
        for code, close in (
            ("A", 10.0 + date_index),
            ("B", 20.0 - date_index),
        )
    ])

    prepare_factor_candidates(
        panel,
        membership,
        factors=["high_quality_momentum"],
        snapshot_dates=list(dates[:3]),
        output_root=tmp_path,
        top_n=1,
        start_date="2024-01-02",
        end_date="2024-01-07",
        candidate_enrichment_prices=prices,
        candidate_outcome_holding_days=2,
    )

    candidate_path = (
        tmp_path / "candidates" / "high_quality_momentum"
        / "candidates_2024-01-02.csv"
    )
    outcome_path = (
        tmp_path / "candidates" / "high_quality_momentum"
        / "candidate_outcomes.csv"
    )
    manifest = json.loads(
        (
            tmp_path / "candidates" / "high_quality_momentum"
            / "manifest.json"
        ).read_text(encoding="utf-8"),
    )
    candidates = pd.read_csv(candidate_path)
    outcomes = pd.read_csv(outcome_path)

    assert "forward_return" not in candidates.columns
    assert outcomes["label_role"].unique().tolist() == [
        "research_label_only_not_candidate_input",
    ]
    assert outcomes["label_available"].all()
    assert abs(outcomes.loc[0, "forward_return"] - (12.0 / 10.0 - 1)) < 1e-12
    assert manifest["candidate_outcome_labels"]["available_count"] == 3
    assert (
        manifest["candidate_outcome_labels"][
            "future_return_used_for_candidate_selection"
        ]
        is False
    )


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


def test_trade_audit_accepts_prior_planned_left_grid_without_technical_signal():
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
        "account_mode": "left",
        "technical_signal_timing": (
            "preplanned_value_grid_limit_no_technical_signal"
        ),
    }]}

    audit, summary = audit_portfolio_trades(
        result,
        {"sh.600001": frame},
    )

    assert audit.iloc[0]["violations"] == ""
    assert pd.isna(
        audit.iloc[0]["technical_signal_no_later_than_execution"]
    )
    assert summary["violation_count"] == 0


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
    assert manifest["uses_financial_data"] is False
    assert manifest["strict_financial_point_in_time"] is True


def test_saved_manifest_honors_explicit_forward_financial_and_industry_metadata(
    tmp_path,
):
    industry_source = tmp_path / "industry.csv"
    industry_source.write_text("code,industry\nsh.600001,医药\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(industry_source.read_bytes()).hexdigest()
    output = save_factor_snapshots(
        {"2026-08-12": [{"code": "sh.600001"}]},
        output_directory=tmp_path / "candidates",
        factor_column="gross_profitability_forward",
        top_n=1,
        start_date="2026-08-12",
        end_date="2026-08-12",
        factor_metadata={
            "uses_financial_data": True,
            "financial_point_in_time": True,
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
            "industry_point_in_time_status": "safe",
            "industry_data_source": "test interval snapshot",
            "industry_source_path": str(industry_source),
            "industry_source_sha256": source_sha256,
            "industry_as_of_date": "2026-08-11",
            "industry_coverage_ratio": 1.0,
            "research_only": True,
        },
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["uses_financial_data"] is True
    assert manifest["financial_point_in_time"] is True
    assert manifest["strict_financial_point_in_time"] is True
    assert manifest["industry_point_in_time"] is True
    assert manifest["industry_source_sha256"] == source_sha256
    assert manifest["research_only"] is True
    assert manifest["snapshots"][0]["financial_point_in_time"] is True
    assert manifest["snapshots"][0]["industry_point_in_time"] is True


def test_low_idiosyncratic_volatility_manifest_fails_closed_on_pb_history(
    tmp_path,
):
    output = save_factor_snapshots(
        {"2024-01-02": [{"code": "sh.600001"}]},
        output_directory=tmp_path,
        factor_column="playbook_low_corr",
        top_n=1,
        start_date="2024-01-02",
        end_date="2024-01-02",
        factor_metadata={
            "selected_factors": [
                "network_cc", "low_idiosyncratic_volatility",
            ],
            "weights": {
                "network_cc": 0.5,
                "low_idiosyncratic_volatility": 0.5,
            },
        },
    )

    manifest = json.loads((output / "manifest.json").read_text(
        encoding="utf-8",
    ))
    assert manifest["uses_financial_data"] is True
    assert manifest["financial_point_in_time"] is False
    assert manifest["strict_financial_point_in_time"] is False
    assert manifest["snapshots"][0]["financial_point_in_time"] is False
    assert manifest["research_only"] is True
    assert "historical daily PB" in manifest["financial_point_in_time_reason"]


def test_market_residual_replacement_manifest_records_formula_and_pit_rules(
    tmp_path,
):
    residual = "market_residual_momentum_126_skip20"
    metadata = market_residual_momentum_factor_metadata(residual)
    output = save_factor_snapshots(
        {"2024-01-02": [{"code": "sh.600001"}]},
        output_directory=tmp_path,
        factor_column=f"playbook_low_corr_{residual}",
        top_n=1,
        start_date="2024-01-02",
        end_date="2024-01-02",
        factor_metadata={
            "selected_factors": [
                "network_cc",
                "ma_convergence",
                "disposition_reversal",
                residual,
                "coin_team",
                "low_idiosyncratic_volatility",
            ],
            "market_residual_momentum": metadata,
        },
    )

    manifest = json.loads((output / "manifest.json").read_text(
        encoding="utf-8",
    ))
    recorded = manifest["factor_metadata"]["market_residual_momentum"]
    assert recorded["estimation_window"] == 60
    assert recorded["lookback"] == 126
    assert recorded["skip"] == 20
    assert recorded["market_universe"] == (
        "point_in_time_tradable_A_shares_on_each_date"
    )
    assert "never_fill_missing_with_zero" in recorded["missing_return_rule"]
    assert recorded["industry_data_used"] is False
    assert manifest["research_only"] is True
    assert manifest["strict_financial_point_in_time"] is False


def test_factor_replay_requires_same_700_day_execution_warmup():
    assert required_history_start_date("2026-01-01") == pd.Timestamp(
        "2024-02-01",
    )
    assert required_history_start_date(
        "2021-01-01",
        calendar_days=420,
    ) == pd.Timestamp("2019-11-08")
