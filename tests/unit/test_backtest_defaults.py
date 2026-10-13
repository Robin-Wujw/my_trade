"""Regression tests for the conservative portfolio replay defaults."""

import inspect

from apps import portfolio_yearly_backtest
from stock_research.backtesting import cli
from stock_research.backtesting.defaults import (
    DEFAULT_CLOSE_CONFIRMED_EXECUTION,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    DEFAULT_MINIMUM_COMMISSION,
    DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28,
)
from stock_research.backtesting.engine import run_portfolio_backtest
from stock_research.backtesting.workflow import qualify_backtest_result
from stock_research.reporting.backtest_trade_report import render_trade_report_markdown
from stock_research.strategies.miniqmt_backtest import DEFAULT_MINIQMT_BACKTEST_PROFILE


def test_engine_defaults_use_conservative_execution_profile():
    parameters = inspect.signature(run_portfolio_backtest).parameters

    assert parameters["close_confirmed_execution"].default == DEFAULT_CLOSE_CONFIRMED_EXECUTION
    assert parameters["commission_rate"].default == DEFAULT_COMMISSION_RATE
    assert parameters["minimum_commission"].default == DEFAULT_MINIMUM_COMMISSION
    assert parameters["sell_stamp_duty_rate"].default == DEFAULT_SELL_STAMP_DUTY_RATE
    assert (
        parameters["sell_stamp_duty_rate_before_2023_08_28"].default
        == DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28
    )
    assert parameters["estimated_slippage_rate"].default == DEFAULT_ESTIMATED_SLIPPAGE_RATE


def test_all_public_backtest_parsers_share_safe_defaults():
    portfolio = cli.build_parser().parse_args([])
    yearly = portfolio_yearly_backtest.build_parser().parse_args([])
    profile = DEFAULT_MINIQMT_BACKTEST_PROFILE

    for args in (portfolio, yearly):
        assert args.close_confirmed_execution == DEFAULT_CLOSE_CONFIRMED_EXECUTION
        assert args.commission_rate == DEFAULT_COMMISSION_RATE
        assert args.minimum_commission == DEFAULT_MINIMUM_COMMISSION
        assert args.sell_stamp_duty_rate == DEFAULT_SELL_STAMP_DUTY_RATE
        assert args.estimated_slippage_rate == DEFAULT_ESTIMATED_SLIPPAGE_RATE

    assert profile.close_confirmed_execution == DEFAULT_CLOSE_CONFIRMED_EXECUTION
    assert profile.commission_rate == DEFAULT_COMMISSION_RATE
    assert profile.minimum_commission == DEFAULT_MINIMUM_COMMISSION
    assert profile.sell_stamp_duty_rate == DEFAULT_SELL_STAMP_DUTY_RATE
    assert (
        profile.sell_stamp_duty_rate_before_2023_08_28
        == DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28
    )
    assert profile.estimated_slippage_rate == DEFAULT_ESTIMATED_SLIPPAGE_RATE


def test_close_proxy_remains_an_explicit_research_choice():
    args = cli.build_parser().parse_args(["--close-confirmed-execution", "close_proxy"])
    assert args.close_confirmed_execution == "close_proxy"


def test_result_qualification_propagates_upstream_and_data_limits():
    result = {
        "corporate_action_accounting_status": "data_limited",
        "corporate_action_limitations": [{"code": "sh.600000"}],
    }
    qualify_backtest_result(
        result,
        candidate_manifest={"research_only": True},
        price_coverage={"internal_missing_bar_count": 2},
        daily_limit_coverage={"complete": False},
    )

    assert result["research_only"] is True
    assert result["acceptance_status"] == "data_limited"
    assert "upstream_research_only" in result["qualification"]["reasons"]
    assert "unexplained_internal_price_gaps" in result["data_limitations"]
    assert "exact_price_limits_incomplete" in result["data_limitations"]


def test_manifest_flags_alone_cannot_certify_strict_pit():
    result = {}
    qualify_backtest_result(
        result,
        candidate_manifest={
            "research_only": False,
            "strict_point_in_time": True,
            "revision_vintage_verified": True,
        },
        args=type("Args", (), {"close_confirmed_execution": "next_open"})(),
        price_coverage={"internal_missing_bar_count": 0},
        daily_limit_coverage={"complete": True},
    )

    assert result["research_only"] is True
    assert result["strict_point_in_time"] is False
    assert result["acceptance_status"] == "research_only"


def test_result_qualification_preserves_upstream_data_limited_status():
    result = {}
    qualify_backtest_result(
        result,
        candidate_manifest={
            "research_only": True,
            "acceptance_status": "data_limited",
            "data_limitations": ["missing_revision_vintage"],
        },
        price_coverage={"internal_missing_bar_count": 0},
        daily_limit_coverage={"complete": True},
    )

    assert result["acceptance_status"] == "data_limited"
    assert "missing_revision_vintage" in result["data_limitations"]


def test_trade_report_serializes_structured_data_limitations():
    report = render_trade_report_markdown({
        "research_only": True,
        "acceptance_status": "data_limited",
        "data_limitations": [{"code": "sh.600000", "reason": "pay_date_missing"}],
        "trade_ledger": [],
    })
    assert "pay_date_missing" in report

