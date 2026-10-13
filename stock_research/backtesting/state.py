"""Portfolio positions and campaign lifecycle state."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PositionState:
    left: list[dict] = field(default_factory=list)
    left_value_line: float | None = None
    left_grid_started: bool = False
    left_candidate_profile: dict | None = None
    right: list[dict] = field(default_factory=list)
    right_parts: int = 5
    right_sold: set[str] = field(default_factory=set)
    right_plan_date: object | None = None
    pending_lot_exits: dict[str, str] = field(default_factory=dict)
    pending_profit_ids: set[str] = field(default_factory=set)
    pending_tail_capacity_free: bool = False
    right_tail_capacity_free: bool = False



def _is_profit_tail(state: PositionState) -> bool:
    """A final profit tranche remains invested but no longer consumes a slot."""
    return (
        bool(state.right)
        and state.right_parts == 1
        and bool(state.right_sold)
        and state.right_tail_capacity_free
    )



def _qualifies_profit_tail(profit, remaining_parts, minimum_return) -> bool:
    return (
        int(remaining_parts) == 1
        and float(profit.get("maximum_return_pct") or 0.0) / 100
        >= float(minimum_return)
    )



def _profit_ids_to_execute(new_ids, remaining_parts) -> list[str]:
    """Reserve the final tranche for maximum-profit-half only."""
    remaining = max(0, int(remaining_parts))
    ids = set(new_ids)
    if remaining == 1:
        return ["maximum_profit_half"] if "maximum_profit_half" in ids else []
    intermediate = sorted(ids - {"maximum_profit_half"})
    return intermediate[:max(0, remaining - 1)]



def _entry_risk_still_controls_lot(lot: dict, state: PositionState) -> bool:
    """Only a qualified high-profit final tail is released from entry risk."""
    return not (bool(lot.get("merged")) and _is_profit_tail(state))

