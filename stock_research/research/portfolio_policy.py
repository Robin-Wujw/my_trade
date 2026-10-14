"""Predeclared research sizing controls with explicit observation cutoffs.

These policies do not change the frozen baseline. Execution adapters must pass
the observed account NAV, visible price history, and quoted funding source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GapTailEstimate:
    as_of: str
    observations: int
    loss_quantile: float | None
    status: str


def gap_tail_estimate(frame, as_of, *, window=250, quantile=0.95):
    """Use 250 prior observed sessions; positive gaps contribute zero loss.

    Adjusted open/close ratios avoid treating a split as an overnight loss.
    The caller supplies the signal cutoff, never the execution day's close.
    """
    if type(window) is not int or window < 2 or not 0 < quantile < 1:
        raise ValueError("Invalid gap-tail window or quantile")
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError("Gap-tail estimate requires an explicit valid cutoff")
    cutoff = cutoff.normalize()
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data.date.le(cutoff)].sort_values("date")
    if data.date.duplicated().any():
        raise ValueError("Duplicate visible price bars")
    if "tradestatus" in data:
        data = data.loc[data.tradestatus.astype(str).isin({"1", "1.0", "True"})]
    if "volume" in data:
        data = data.loc[pd.to_numeric(data.volume, errors="coerce").gt(0)]
    observed = data.tail(window + 1)
    opens = pd.to_numeric(observed["open"], errors="coerce")
    previous = pd.to_numeric(observed["close"], errors="coerce").shift(1)
    gaps = (1 - opens / previous).iloc[1:]
    valid = np.isfinite(gaps) & opens.iloc[1:].gt(0) & previous.iloc[1:].gt(0)
    if len(gaps) != window or not valid.all():
        return GapTailEstimate(str(cutoff.date()), int(valid.sum()), None, "insufficient_visible_gap_history")
    return GapTailEstimate(str(cutoff.date()), window,
                           float(gaps.clip(lower=0).quantile(quantile)), "ok")


def tail_budget_notional(*, equity, entry_price, stop_price, gap_loss_fraction,
                         risk_budget=0.008, max_position=0.30):
    """Risk includes the structural stop plus the predeclared gap loss tail."""
    values = (equity, entry_price, stop_price, gap_loss_fraction, risk_budget, max_position)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise ValueError("Tail sizing needs complete finite observations")
    if (equity <= 0 or entry_price <= 0 or stop_price < 0 or stop_price >= entry_price
            or not 0 <= gap_loss_fraction < 1 or not 0 < risk_budget <= 0.03
            or not 0 < max_position <= 1):
        raise ValueError("Invalid tail risk sizing inputs")
    risk_per_share = entry_price - stop_price + entry_price * gap_loss_fraction
    fraction = min(max_position, risk_budget / (risk_per_share / entry_price))
    return {"notional": equity * fraction, "equity_fraction": fraction,
            "risk_per_share": risk_per_share, "equity_risk_budget": equity * risk_budget,
            "basis": "observed_net_asset_structural_stop_plus_gap_tail"}


def portfolio_entry_scale(equity_curve, as_of, *, annual_volatility_target=None,
                          volatility_window=60, maximum_drawdown=None):
    """Scale new risk using only completed equity observations, never lever up."""
    if annual_volatility_target is not None and not 0 < annual_volatility_target <= 1:
        raise ValueError("Invalid annual volatility target")
    if maximum_drawdown is not None and not 0 < maximum_drawdown < 1:
        raise ValueError("Invalid maximum drawdown limit")
    if type(volatility_window) is not int or volatility_window < 2:
        raise ValueError("Invalid volatility window")
    data = pd.DataFrame(equity_curve)
    if data.empty:
        return {"scale": 0.0, "reason": "insufficient_visible_equity_history"}
    data = data.assign(date=pd.to_datetime(data["date"], errors="coerce"))
    data = data.loc[data.date.le(pd.Timestamp(as_of))].sort_values("date")
    if data.empty or data.date.duplicated().any():
        return {"scale": 0.0, "reason": "invalid_visible_equity_history"}
    equity = pd.to_numeric(data["equity"], errors="coerce")
    if not np.isfinite(equity).all() or equity.le(0).any():
        return {"scale": 0.0, "reason": "invalid_visible_equity_history"}
    drawdown = float(1 - equity.iloc[-1] / equity.max())
    if maximum_drawdown is not None and drawdown >= maximum_drawdown:
        return {"scale": 0.0, "reason": "drawdown_new_risk_block", "drawdown": drawdown}
    scale, volatility = 1.0, None
    if annual_volatility_target is not None:
        # Corporate-action-only valuation dates must not be counted as sessions.
        if "is_trading_day" in data:
            equity = equity.loc[data.is_trading_day.eq(True)]
        returns = equity.pct_change(fill_method=None).dropna().tail(volatility_window)
        if len(returns) != volatility_window:
            return {"scale": 0.0, "reason": "insufficient_visible_equity_history"}
        volatility = float(returns.std(ddof=1) * math.sqrt(252))
        scale = min(1.0, annual_volatility_target / volatility) if volatility > 0 else 1.0
    return {"scale": scale, "reason": "ok", "drawdown": drawdown,
            "annualized_volatility": volatility}


def quoted_cash_income(cash, start_date, end_date, quote):
    """Simple interest on an identified fixed-rate instrument, using a known quote.

    This is an explicit research cash overlay. No assumed risk-free rate or
    retrospective best-instrument selection substitutes for a source record.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Cash quote is unavailable or does not cover the accrual period")
    start, end = start.normalize(), end.normalize()
    required = {"instrument", "source", "source_sha256", "available_at", "start_date",
                "maturity_date", "annual_rate", "day_count"}
    if required - set(quote):
        raise ValueError("Cash income requires a complete identified source quote")
    import re
    if (not quote["instrument"] or not quote["source"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(quote["source_sha256"]))):
        raise ValueError("Cash quote source identity is invalid")
    known = pd.Timestamp(quote["available_at"])
    quote_start = pd.Timestamp(quote["start_date"])
    maturity = pd.Timestamp(quote["maturity_date"])
    if pd.isna(quote_start) or pd.isna(maturity):
        raise ValueError("Cash quote is unavailable or does not cover the accrual period")
    quote_start, maturity = quote_start.normalize(), maturity.normalize()
    cutoff = start + pd.Timedelta(hours=16, minutes=30)
    if known.tzinfo is not None:
        cutoff = cutoff.tz_localize("Asia/Shanghai")
        known = known.tz_convert("Asia/Shanghai")
    if (pd.isna(known) or known > cutoff or pd.isna(start) or pd.isna(end) or end <= start
            or pd.isna(quote_start) or pd.isna(maturity) or quote_start > start or maturity < end):
        raise ValueError("Cash quote is unavailable or does not cover the accrual period")
    rate, day_count = float(quote["annual_rate"]), quote["day_count"]
    if not math.isfinite(float(cash)) or cash < 0 or not math.isfinite(rate) or not -0.1 <= rate <= 1:
        raise ValueError("Invalid cash principal or quoted rate")
    if type(day_count) is not int or day_count not in {360, 365}:
        raise ValueError("Cash quote must state a 360- or 365-day basis")
    return float(cash) * rate * (end - start).days / day_count
