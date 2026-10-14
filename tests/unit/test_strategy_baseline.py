import inspect

from apps import portfolio_yearly_backtest
from stock_research.backtesting import cli
from stock_research.backtesting.baseline import baseline_engine_parameters, configuration_record, load_baseline
from stock_research.backtesting.engine import run_portfolio_backtest


def test_profile_parameters_remain_compatible_with_engine():
    parameters = baseline_engine_parameters()
    assert set(parameters) <= set(inspect.signature(run_portfolio_backtest).parameters)
    assert load_baseline()["strict_point_in_time"] is False
    assert load_baseline()["selected_before_oos"] is False
    assert load_baseline()["forward_policy"]["broker_order_api_enabled"] is False


def test_public_defaults_match_audited_strategy_and_leave_input_units_explicit():
    args = cli.build_parser().parse_args([])
    record = configuration_record(args)
    assert args.max_left_positions == 2
    assert args.right_entry_size_multiplier == 3.0
    assert args.estimated_slippage_rate == 0.001
    assert args.factor_only_min_avg_amount == 30_000_000
    assert args.max_entry_signal_amount_participation == 0.01
    assert record["matches_baseline_parameters"] is True
    # Generic input bars use yuan. Archived Tushare raw bars require an explicit
    # 1000 multiplier; importing a profile cannot change units silently.
    assert args.price_amount_multiplier == 1.0
    assert args.price_frames_prepared is False


def test_explicit_research_overrides_are_visible_and_risk_budget_keeps_compatibility():
    parser = cli.build_parser()
    args = parser.parse_args(["--right-entry-risk-budget-pct", "0.008"])
    assert args.right_entry_size_multiplier == 1.0
    record = configuration_record(args)
    assert record["matches_baseline_parameters"] is False
    assert record["parameter_deviations"]["right_entry_risk_budget_pct"]["effective"] == 0.008
    conflicting = parser.parse_args([
        "--right-entry-risk-budget-pct=0.008", "--right-entry-size-multiplier=2.0",
    ])
    assert conflicting.right_entry_size_multiplier == 2.0
    assert parser.parse_args([]).right_entry_size_multiplier == 3.0
    uncapped = configuration_record(parser.parse_args(["--max-positions", "0"]))
    assert uncapped["effective_engine_parameters"]["max_positions"] is None


def test_yearly_wrapper_passes_shared_sizing_and_cost_profile():
    args = portfolio_yearly_backtest.build_parser().parse_args([])
    argv, _ = portfolio_yearly_backtest.build_year_argv(args, 2026, "2026-01-01", "2026-10-13")
    forwarded = cli.build_parser().parse_args(argv)
    assert forwarded.max_left_positions == 2
    assert forwarded.right_entry_size_multiplier == 3.0
    assert forwarded.estimated_slippage_rate == 0.001
