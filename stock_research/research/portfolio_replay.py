"""Explicit research-only portfolio policy entry point and frozen configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math


@dataclass(frozen=True)
class ResearchPolicy:
    schema_version: int = 1
    formula33_enabled: bool = True
    sizing_basis: str = "initial_capital"
    gap_tail_enabled: bool = False
    gap_window: int = 250
    gap_quantile: float = 0.95
    risk_budget: float = 0.008
    max_position: float = 0.30
    annual_volatility_target: float | None = None
    volatility_window: int = 60
    maximum_drawdown: float | None = None

    @property
    def active(self):
        return (not self.formula33_enabled or self.sizing_basis == "visible_nav"
                or self.gap_tail_enabled or self.annual_volatility_target is not None
                or self.maximum_drawdown is not None)


def freeze_research_policy(value):
    """Reject ambiguous/unrecognized settings; copy caller data before replay."""
    if not isinstance(value, dict):
        raise ValueError("research_policy must be an explicit JSON object")
    unknown = set(value) - {field.name for field in fields(ResearchPolicy)}
    if unknown:
        raise ValueError(f"Unknown research policy fields: {sorted(unknown)}")
    try:
        policy = ResearchPolicy(**json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise ValueError("research_policy must contain finite JSON values") from exc
    if type(policy.schema_version) is not int or policy.schema_version != 1:
        raise ValueError("Unsupported research policy schema_version")
    if any(type(getattr(policy, key)) is not bool for key in ("formula33_enabled", "gap_tail_enabled")):
        raise ValueError("Research policy switches must be JSON booleans")
    if policy.sizing_basis not in {"initial_capital", "visible_nav"}:
        raise ValueError("sizing_basis must be initial_capital or visible_nav")
    for key, upper in (("risk_budget", 0.03), ("max_position", 1.0), ("gap_quantile", 1.0)):
        number = getattr(policy, key)
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= upper:
            raise ValueError(f"Invalid research policy {key}")
    if policy.gap_quantile >= 1:
        raise ValueError("gap_quantile must be below one")
    for key in ("gap_window", "volatility_window"):
        if type(getattr(policy, key)) is not int or getattr(policy, key) < 2:
            raise ValueError(f"Invalid research policy {key}")
    for key in ("annual_volatility_target", "maximum_drawdown"):
        number = getattr(policy, key)
        if number is not None and (type(number) not in (int, float) or not math.isfinite(number) or not 0 < number < 1):
            raise ValueError(f"Invalid research policy {key}")
    if policy.gap_tail_enabled and policy.sizing_basis != "visible_nav":
        raise ValueError("Gap-tail research requires visible_nav sizing")
    canonical = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return policy, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_portfolio_research(*, research_policy, cash_contract=None, cash_quotes=None,
                           structure_label_manifests=None, **kwargs):
    """Run an independently identified policy; never alter the production API."""
    from stock_research.backtesting.engine import _run_portfolio_backtest

    policy, canonical, digest = freeze_research_policy(research_policy)
    cash_book = None
    if cash_contract is not None:
        from stock_research.research.cash_book import CashBook
        cash_book = CashBook(cash_contract, [] if cash_quotes is None else cash_quotes)
    elif cash_quotes is not None:
        raise ValueError("cash_quotes require an explicit preselected cash_contract")
    expectancy_book = None
    if structure_label_manifests is not None:
        if (kwargs.get("close_confirmed_execution") != "next_open"
                or kwargs.get("signals_effective_next_day") is not True):
            raise ValueError("Structure expectancy requires explicit next_open and signals_effective_next_day=True")
        unsupported = ("enable_breadth_reversal_half_pullback", "factor_only_breadth_reversal_only",
                       "enable_factor_horizon_channel", "enable_compressed_structure_channel",
                       "factor_only_compressed_structure_only", "enable_diversified_factor_sleeve")
        if any(kwargs.get(key) for key in unsupported):
            raise ValueError("Structure expectancy currently supports the existing standard semantic entry channel")
        from stock_research.research.structure_expectancy import StructureExpectancyBook
        expectancy_book = StructureExpectancyBook(structure_label_manifests)
    result = _run_portfolio_backtest(**kwargs, research_policy=policy if policy.active else None,
                                     research_cash_book=cash_book, research_expectancy_book=expectancy_book)
    result = {**result, "research_policy": json.loads(canonical), "research_policy_sha256": digest,
              "research_only": True, "strict_point_in_time": False,
              "cash_income_status": "not_applied_no_identified_source_quote"}
    if cash_book is not None:
        from stock_research.research.cash_book import audit_cash_research
        cash_book.verify_sources()
        result.update(cash_book.report())
        result["cash_accounting_audit"] = audit_cash_research(result)
        if result["cash_accounting_audit"]["status"] != "passed":
            raise ValueError(f"Cash research accounting failed: {result['cash_accounting_audit']['errors']}")
    if expectancy_book is not None:
        expectancy_book.verify_sources()
        result.update(expectancy_book.report())
    return result
