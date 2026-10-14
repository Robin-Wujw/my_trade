import pandas as pd
import pytest

from stock_research.research.portfolio_policy import (
    gap_tail_estimate, portfolio_entry_scale, quoted_cash_income, tail_budget_notional,
)


def test_gap_tail_uses_adjusted_visible_sessions_and_ignores_future_bars():
    days = pd.bdate_range("2023-01-01", periods=260)
    frame = pd.DataFrame({"date": days, "open": 98., "close": 100., "volume": 1000})
    frame.loc[251:, "open"] = 10.
    result = gap_tail_estimate(frame, days[250])
    assert result.status == "ok" and result.observations == 250
    assert result.loss_quantile == pytest.approx(0.02)
    assert result == gap_tail_estimate(frame.iloc[:251], days[250])
    assert gap_tail_estimate(frame, days[249]).loss_quantile is None
    frame.loc[125, "open"] = float("nan")
    assert gap_tail_estimate(frame, days[250]).status == "insufficient_visible_gap_history"


def test_tail_sizing_scales_with_nav_and_reduces_notional_for_known_gap_risk():
    parameters = dict(entry_price=10., stop_price=9.5, risk_budget=0.008, max_position=0.3)
    structural = tail_budget_notional(equity=1_000_000, gap_loss_fraction=0, **parameters)
    tail = tail_budget_notional(equity=2_000_000, gap_loss_fraction=0.05, **parameters)
    assert structural["notional"] == pytest.approx(160_000)
    assert tail["notional"] == pytest.approx(160_000)
    assert tail["risk_per_share"] == 1.0
    with pytest.raises(ValueError, match="complete"):
        tail_budget_notional(equity=1_000_000, gap_loss_fraction=None, **parameters)


def test_portfolio_controls_ignore_future_equity_and_stop_new_risk_at_drawdown():
    curve = [{"date": "2024-01-02", "equity": 1.}, {"date": "2024-01-03", "equity": 1.1},
             {"date": "2024-01-04", "equity": 0.8}, {"date": "2025-01-01", "equity": 100.}]
    assert portfolio_entry_scale(curve, "2024-01-03", maximum_drawdown=0.2)["scale"] == 1
    assert portfolio_entry_scale(curve, "2024-01-04", maximum_drawdown=0.2)["scale"] == 0
    limited = portfolio_entry_scale(curve, "2024-01-04", annual_volatility_target=0.15, volatility_window=2)
    assert 0 < limited["scale"] < 1
    assert limited == portfolio_entry_scale(curve[:3], "2024-01-04", annual_volatility_target=0.15, volatility_window=2)


def test_cash_income_requires_a_known_covering_quote_and_preserves_calendar_days():
    quote = {"instrument": "fixed_research_instrument", "source": "fixture_quote",
             "source_sha256": "a" * 64, "available_at": "2024-01-05T15:00:00+08:00",
             "start_date": "2024-01-05", "maturity_date": "2024-01-08", "annual_rate": 0.0365,
             "day_count": 365}
    assert quoted_cash_income(1_000_000, "2024-01-05", "2024-01-08", quote) == pytest.approx(300.)
    with pytest.raises(ValueError, match="unavailable"):
        quoted_cash_income(1_000_000, "2024-01-05", "2024-01-08", {**quote, "available_at": "2024-01-06"})
    with pytest.raises(ValueError, match="cover"):
        quoted_cash_income(1_000_000, "2024-01-05", "2024-01-09", quote)
    with pytest.raises(ValueError, match="complete"):
        quoted_cash_income(1_000_000, "2024-01-05", "2024-01-08", {})
    with pytest.raises(ValueError, match="basis"):
        quoted_cash_income(1_000_000, "2024-01-05", "2024-01-08", {**quote, "day_count": 365.9})
    with pytest.raises(ValueError, match="unavailable"):
        quoted_cash_income(1_000_000, "2024-01-05", "2024-01-08", {**quote, "maturity_date": None})
