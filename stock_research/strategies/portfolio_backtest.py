"""Point-in-time portfolio replay over dated selection snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd

from stock_research.strategies.candidate_interface import (
    MAX_DAILY_CANDIDATES,
    normalize_candidate_snapshots,
)
from stock_research.indicators.position_risk import position_exit_snapshot
from stock_research.indicators.price_structure import (
    configured_price_structures,
    infer_price_structures,
)
from stock_research.indicators.technical_entries import (
    _valid_volume_price_nodes,
    apply_entry_confluence,
    infer_technical_entry,
)
from stock_research.strategies.ohlc_execution import fill_buy_stop, fill_limit_order, fill_sell_stop


@dataclass
class PositionState:
    left: list[dict] = field(default_factory=list)
    left_value_line: float | None = None
    left_grid_started: bool = False
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
        and float(profit.get("current_return_pct") or 0.0) / 100
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


def build_formula_phase_history(formula_rows) -> dict[str, str]:
    frame = pd.DataFrame(formula_rows).copy()
    if frame.empty or "date" not in frame:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    phase = "waiting"
    result = {}

    def streak(value):
        number = pd.to_numeric(value, errors="coerce")
        return 0 if pd.isna(number) else int(number)

    for _, row in frame.iterrows():
        up = streak(row.get("window_up_streak"))
        down = streak(row.get("window_down_streak"))
        if down >= 5:
            phase = "exited"
        elif up >= 5:
            phase = "active"
        elif up >= 3 and phase != "active":
            phase = "watch"
        result[row["date"].strftime("%Y-%m-%d")] = phase
    return result


def _prepare_frame(frame):
    data = frame.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.dropna(subset=["date", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for period in (5, 10, 20, 60, 120):
        data[f"ma{period}"] = data["close"].rolling(period).mean()
    return data


def _enabled(value, *, default=True):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _capped_entry_size(current_exposure, planned_size, max_symbol_exposure=0.625):
    """Cap one symbol's combined left/right exposure without changing ratios."""
    available = max(0.0, float(max_symbol_exposure) - float(current_exposure))
    return max(0.0, min(float(planned_size), available))


def _affordable_buy_notional(
    cash, requested_notional, *, commission_rate, minimum_commission, slippage_rate,
):
    """Maximum notional whose principal plus buy costs fits available cash."""
    cash = max(0.0, float(cash))
    requested = max(0.0, float(requested_notional))
    minimum = max(0.0, float(minimum_commission))
    if requested <= 0 or cash <= minimum:
        return 0.0

    def required(notional):
        commission = max(float(notional) * float(commission_rate), minimum)
        return float(notional) + commission + float(notional) * float(slippage_rate)

    low, high = 0.0, requested
    for _ in range(64):
        middle = (low + high) / 2.0
        if required(middle) <= cash:
            low = middle
        else:
            high = middle
    return low


def board_lot_size(code) -> int:
    """Project execution rule: STAR Market 200 shares, all others 100."""
    normalized = str(code).lower()
    return 200 if normalized.startswith("sh.688") or normalized.startswith("688") else 100


def _board_lot_quantity(code, quantity) -> int:
    lot = board_lot_size(code)
    return max(0, int(float(quantity) // lot) * lot)


def _effective_profit_tranches(code, quantity, configured_parts) -> int:
    """Never promise more exit slices than the acquired board lots support."""
    available_board_lots = int(float(quantity) // board_lot_size(code))
    return min(max(1, int(configured_parts)), max(1, available_board_lots))


def _active_profit_trigger_ids(profit, history, remaining_parts) -> set[str]:
    """Map source-backed protection signals without treating examples as a checklist."""
    maximum_return = float(profit.get("maximum_return_pct") or 0.0)
    if maximum_return < 10:
        return set()
    ids = set()
    if profit.get("profit_floor_triggered"):
        ids.add("profit_floor")
    if maximum_return >= 20 and profit.get("trailing_10_triggered"):
        ids.add("trailing_10")
    if profit.get("divergence_time_take_profit"):
        ids.add("divergence_time")

    if (
        maximum_return >= 20
        and len(history) >= 14
        and {"date", "volume", "low"}.issubset(history.columns)
    ):
        prior_history = history.iloc[:-1]
        valid_nodes = _valid_volume_price_nodes(prior_history)
        if valid_nodes:
            prior_close = float(prior_history.iloc[-1]["close"])
            supporting_nodes = [
                node for node in valid_nodes
                if float(node["support"]) <= prior_close
            ]
            if supporting_nodes:
                closest_node = max(
                    supporting_nodes, key=lambda node: float(node["support"]),
                )
                if float(history.iloc[-1]["low"]) < float(closest_node["support"]):
                    ids.add(f"volume_node_break:{closest_node['date']}")

    close = float(profit.get("close") or 0.0)
    latest_ma20 = pd.to_numeric(history.iloc[-1].get("ma20"), errors="coerce")
    prior_ma20 = (
        pd.to_numeric(history.iloc[-6].get("ma20"), errors="coerce")
        if len(history) >= 6 else pd.NA
    )
    if (
        pd.notna(latest_ma20)
        and pd.notna(prior_ma20)
        and float(latest_ma20) > float(prior_ma20)
        and close < float(latest_ma20) * 0.97
    ):
        ids.add("rising_ma20_break_3pct")

    # Reserve the author's most basic maximum-profit-half rule for the last
    # tranche; it clears the campaign without waiting for a return to cost.
    if int(remaining_parts) == 1 and profit.get("half_profit_triggered"):
        ids.add("maximum_profit_half")
    return ids


def _price_structure_signal(
    data, index, plan=None, *, auto_structure=True, allow_pullback=True,
):
    if index < 40:
        return None
    row = data.iloc[index]
    previous = data.iloc[:index]
    prior_close = float(previous.iloc[-1]["close"])
    volume_baseline = max(
        float(previous["volume"].tail(5).mean()),
        float(previous["volume"].tail(10).mean()),
    )
    close_volume_ratio = (
        float(row["volume"]) / volume_baseline if volume_baseline > 0 else 0.0
    )

    def gap_up_rank(base_rank: int) -> tuple[int, bool]:
        prior_high = pd.to_numeric(previous.iloc[-1].get("high"), errors="coerce")
        day_open = pd.to_numeric(row.get("open"), errors="coerce")
        gap_up = bool(
            pd.notna(prior_high)
            and pd.notna(day_open)
            and float(day_open) > float(prior_high)
        )
        return (base_rank + 1 if gap_up else base_rank), gap_up

    structures = configured_price_structures(plan)
    if auto_structure:
        inferred_structures = infer_price_structures(previous)
        structures.extend(inferred_structures)
        for inferred in (
            item for item in inferred_structures
            if item["kind"] == "uptrend_anchor"
        ):
            # Ratios of the preceding uptrend remain pullback candidates. They
            # become executable only when a pre-known moving average overlaps
            # the level; all three ratios stay available at their useful ages.
            maximum_days = {0.75: 8, 0.625: 21, 0.50: 34}
            age = int(inferred.get("bars_since_high") or 0)
            prior = previous.iloc[-1]
            for ratio, level in inferred["uptrend_levels"].items():
                if age > maximum_days[ratio]:
                    continue
                confluence = []
                for name in ("ma5", "ma10", "ma20", "ma60"):
                    value = pd.to_numeric(prior.get(name), errors="coerce")
                    if pd.notna(value) and abs(float(value) / float(level) - 1) <= 0.02:
                        confluence.append(name.upper())
                if confluence:
                    structures.append({
                        "kind": "uptrend_support", "ratio": ratio,
                        "level": float(level), "confluence": confluence,
                        "uptrend_low": inferred["uptrend_low"],
                        "uptrend_high": inferred["uptrend_high"],
                        "uptrend_low_date": inferred["uptrend_low_date"],
                        "uptrend_high_date": inferred["uptrend_high_date"],
                    })

    # A close-confirmed breach of the uptrend U50 followed by a reclaim is a
    # trend-restoration entry.  It is not the H-P pullback-half breakout.
    reclaim_candidates = []
    for structure in structures:
        if structure.get("kind") != "uptrend_anchor":
            continue
        if not structure.get("amplitude_valid"):
            continue
        trigger = float(structure["uptrend_levels"][0.50])
        high_date = pd.Timestamp(structure["uptrend_high_date"])
        after_high = previous[previous["date"] > high_date]
        effective_breach = bool((after_high["close"] < trigger).any())
        if (
            effective_breach
            and prior_close <= trigger < float(row["close"])
            and close_volume_ratio >= 1.0
        ):
            reclaim_candidates.append((trigger, structure))
    if reclaim_candidates:
        trigger, structure = max(reclaim_candidates, key=lambda item: item[0])
        rank, gap_up = gap_up_rank(4)
        reason = "上涨波段50%有效跌破后收盘放量收复"
        if gap_up:
            reason += "; 跳空向上加分"
        return {
            "rank": rank, "stop": trigger, "trigger": trigger,
            "order_type": "close", "reason": reason,
            "known_volume_ratio": close_volume_ratio,
            "requires_next_day_confirmation": True,
            "structure_ratio": 0.50,
            "signal_type": "uptrend_50_reclaim", "gap_up": gap_up,
            "anchor_low": structure["uptrend_low"],
            "anchor_high": structure["uptrend_high"],
            "anchor_low_date": structure["uptrend_low_date"],
            "anchor_high_date": structure["uptrend_high_date"],
        }

    # Pullback-half breakout is a separate three-anchor structure. It is valid
    # only after the same uptrend's 62.5% was breached, the trend-level span is
    # wide enough, and consolidation has lasted at least 13 trading days.
    for structure in structures:
        if structure["kind"] != "pullback_recovery":
            continue
        if not (
            structure.get("deep_pullback_confirmed")
            and structure.get("amplitude_valid")
            and int(structure.get("consolidation_days") or 0) >= 13
        ):
            continue
        trigger = float(structure["recovery_half"])
        if (
            prior_close <= trigger < float(row["close"])
            and close_volume_ratio >= 1.0
        ):
            rank, gap_up = gap_up_rank(4)
            reason = "回调波段50%收盘放量向上突破"
            if gap_up:
                reason += "; 跳空向上加分"
            return {
                "rank": rank, "stop": trigger, "trigger": trigger,
                "order_type": "close", "reason": reason,
                "known_volume_ratio": close_volume_ratio,
                "requires_next_day_confirmation": True,
                "structure_ratio": 0.50,
                "signal_type": "pullback_50_breakout", "gap_up": gap_up,
                "anchor_low": structure.get("uptrend_low"),
                "anchor_high": structure.get("uptrend_high"),
                "anchor_pullback_low": structure.get("pullback_low"),
                "anchor_low_date": structure.get("uptrend_low_date"),
                "anchor_high_date": structure.get("uptrend_high_date"),
                "anchor_pullback_low_date": structure.get("pullback_low_date"),
            }

    # U75/U625 are pullback-only and require declared technical confluence.
    # They are lower priority than the author's two right-side breakout entries.
    if allow_pullback:
        supports = sorted(
            (item for item in structures if item["kind"] == "uptrend_support"),
            key=lambda item: item["level"], reverse=True,
        )
        for structure in supports:
            level = float(structure["level"])
            ratio = float(structure["ratio"])
            if prior_close > level and float(row["low"]) < level:
                confluence = ",".join(map(str, structure["confluence"]))
                return {
                    "rank": 3, "stop": level, "trigger": level,
                    "order_type": "limit",
                    "reason": f"上涨波段{ratio:.1%}拉回支撑; 共振={confluence}",
                    "known_volume_ratio": 1.0, "structure_ratio": ratio,
                    "signal_type": "uptrend_support_pullback",
                    "anchor_low": structure.get("uptrend_low"),
                    "anchor_high": structure.get("uptrend_high"),
                    "anchor_low_date": structure.get("uptrend_low_date"),
                    "anchor_high_date": structure.get("uptrend_high_date"),
                }
    return None


def _right_signal(
    data, index, plan=None, *, auto_price_structure=True,
    allow_structure_pullback=True,
):
    if index < 60:
        return None
    structure_signal = _price_structure_signal(
        data, index, plan, auto_structure=auto_price_structure,
        allow_pullback=allow_structure_pullback,
    )
    if structure_signal:
        structure_signal = apply_entry_confluence(data, index, structure_signal)
    if configured_price_structures(plan) and structure_signal:
        return structure_signal
    technical_signal = infer_technical_entry(data, index)
    return max(
        (signal for signal in (structure_signal, technical_signal) if signal),
        key=lambda signal: signal["rank"],
        default=None,
    )


def run_portfolio_backtest(
    price_frames,
    candidate_snapshots,
    formula_phases,
    *,
    requested_start,
    end_date,
    trade_plans=None,
    max_positions=3,
    max_total_held_symbols=5,
    max_same_industry=2,
    same_theme_correlation=0.60,
    min_entry_evidence_score=0,
    profit_tranches=5,
    profit_tail_min_return=0.50,
    left_grid_unit=0.02,
    left_grid_step=0.05,
    left_grid_max_exposure=0.20,
    max_symbol_exposure=0.625,
    exit_tail_on_candidate_removal=False,
    signals_effective_next_day=False,
    auto_price_structure=True,
    allow_structure_pullback=True,
    close_confirmed_execution="next_open",
    commission_rate=0.0,
    minimum_commission=0.0,
    initial_capital=1_000_000.0,
    sell_stamp_duty_rate=0.0,
    estimated_slippage_rate=0.0,
):
    """Replay a portfolio without filling dates before candidate coverage."""
    if close_confirmed_execution not in {"next_open", "close_proxy"}:
        raise ValueError("close_confirmed_execution must be next_open or close_proxy")
    if float(initial_capital) <= 0:
        raise ValueError("initial_capital must be positive")
    normalized_snapshots = normalize_candidate_snapshots(
        candidate_snapshots, include_diagnostics=True,
    )
    snapshots = {
        pd.Timestamp(date).normalize(): {str(item["code"]): dict(item) for item in rows}
        for date, rows in normalized_snapshots.items()
    }
    if not snapshots:
        raise ValueError("candidate snapshot history is empty")
    coverage_start = min(snapshots)
    requested = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(end_date).normalize()
    start = max(requested, coverage_start)
    # Cash is shared, so same-day processing order can affect which order gets
    # the remaining balance.  Sort symbols to keep reruns independent of file
    # discovery/dictionary insertion order.
    frames = {code: _prepare_frame(price_frames[code]) for code in sorted(price_frames)}
    plans = dict((trade_plans or {}).get("plans") or {})
    states = {code: PositionState() for code in frames}
    events = []
    realized = 0.0
    transaction_costs = 0.0
    cash_balance = float(initial_capital)
    equity_curve = []
    current_candidates = {}
    current_phase = "waiting"
    current_down_streak = 0
    current_up_streak = 0
    next_batch_id = 1
    last_candidate_scores = {}
    previous_candidate_codes = set()
    pending_candidate_exits = set()
    pending_weak_exits = set()
    pending_left_exits = {}
    pending_left_quota_exits = {}

    candidate_names = {
        str(item["code"]): str(item.get("name") or "")
        for rows in candidate_snapshots.values()
        for item in rows
    }

    def is_st(code):
        return "ST" in candidate_names.get(code, "").upper()

    def occupied_codes():
        return {
            code for code, state in states.items()
            if state.left or state.right
        }

    def left_side_codes():
        return {
            code for code, state in states.items()
            if state.left or state.left_grid_started
        }

    def left_position_counts_capacity(state):
        if not state.left:
            return False
        return any(not bool(lot.get("core")) for lot in state.left)

    def capacity_codes():
        return {
            code for code, state in states.items()
            if left_position_counts_capacity(state)
            or (state.right and not _is_profit_tail(state))
        }

    def gross_exposure():
        return sum(
            sum(float(lot["size"]) for lot in state.left + state.right)
            for state in states.values()
        )

    def symbol_exposure(code):
        state = states[code]
        return sum(float(lot["size"]) for lot in state.left + state.right)

    configured_positions = (
        3 if max_positions is None or int(max_positions) <= 0
        else min(5, int(max_positions))
    )
    configured_total_held_symbols = max(1, min(5, int(max_total_held_symbols)))
    configured_same_industry = max(1, int(max_same_industry))
    configured_theme_correlation = float(same_theme_correlation)
    configured_min_entry_evidence_score = max(
        0.0, float(min_entry_evidence_score),
    )
    configured_profit_tranches = max(2, min(5, int(profit_tranches)))
    configured_profit_tail_min_return = max(0.0, float(profit_tail_min_return))
    configured_left_grid_unit = max(0.0, float(left_grid_unit))
    configured_left_grid_step = max(0.05, float(left_grid_step))
    configured_left_grid_max_exposure = min(
        0.30, max(0.0, float(left_grid_max_exposure)),
    )
    for state in states.values():
        state.right_parts = configured_profit_tranches
    concentration_blocks = []

    def symbol_limit_reached():
        configured = configured_positions
        return len(capacity_codes()) >= configured

    def total_symbol_limit_reached(code=None):
        occupied = occupied_codes()
        if code is not None and code in occupied:
            return False
        return len(occupied) >= configured_total_held_symbols

    def right_market_active():
        return current_phase in {"watch", "active"} or current_up_streak >= 3

    def left_symbol_limit_reached(code=None):
        left_codes = left_side_codes()
        if code is not None and code in left_codes:
            return False
        return len(left_codes) >= 1

    def right_codes():
        return {code for code, state in states.items() if state.right}

    def candidate_industry_tags(candidate):
        raw = candidate.get("industry") or candidate.get("mainline_boards") or ""
        if pd.isna(raw):
            return set()
        aliases = {
            "通信网络设备及器件": "通信设备",
            "光通信设备": "通信设备",
            "通信终端及配件": "通信设备",
        }
        tags = {
            aliases.get(tag.strip(), tag.strip())
            for tag in re.split(r"[、,，;；/|+]", str(raw))
            if tag.strip()
        }
        return tags

    def left_value_falsification_reason(candidate):
        if not candidate:
            return ""
        reason = str(candidate.get("value_falsification_reason") or "").strip()
        if reason:
            return reason
        if _enabled(candidate.get("value_falsified"), default=False):
            return str(
                candidate.get("candidate_failure_reason")
                or "value thesis falsified by financial snapshot"
            ).strip()
        return ""

    def left_candidate_can_add(candidate):
        if not candidate:
            return False
        return (
            _enabled(candidate.get("selected_for_trading"), default=True)
            and _enabled(candidate.get("signal_eligible"), default=True)
            and _enabled(candidate.get("allow_left"), default=False)
            and not left_value_falsification_reason(candidate)
        )

    def held_industry_counts():
        counts = {}
        for state in states.values():
            if _is_profit_tail(state) and not state.left:
                continue
            tags = {
                tag
                for lot in state.left + state.right
                for tag in lot.get("industry_tags", [])
            }
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def return_correlation(code, held_code, date):
        left = frames[code]
        right = frames[held_code]
        left = left[left["date"] < date][["date", "close"]].tail(61)
        right = right[right["date"] < date][["date", "close"]].tail(61)
        aligned = left.merge(right, on="date", suffixes=("_left", "_right"))
        if len(aligned) < 41:
            return None
        returns = aligned[["close_left", "close_right"]].pct_change().dropna()
        if len(returns) < 40:
            return None
        if (returns.std(ddof=0) <= 1e-12).any():
            return None
        correlation = returns["close_left"].corr(returns["close_right"])
        return None if pd.isna(correlation) else float(correlation)

    def industry_limit_reached(code, candidate, date):
        if code in capacity_codes():
            return False
        counts = held_industry_counts()
        tags = candidate_industry_tags(candidate)
        tag_blocked = any(
            counts.get(tag, 0) >= configured_same_industry
            for tag in tags
        )
        correlations = {
            held_code: return_correlation(code, held_code, date)
            for held_code in capacity_codes()
        }
        related = {
            held_code: correlation
            for held_code, correlation in correlations.items()
            if correlation is not None
            and correlation >= configured_theme_correlation
        }
        correlation_blocked = len(related) >= configured_same_industry
        if tag_blocked or correlation_blocked:
            concentration_blocks.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "code": code,
                "name": candidate.get("name") or code,
                "industry_tags": "、".join(sorted(tags)),
                "correlated_held_codes": sorted(related),
                "correlations": {
                    held_code: round(correlation, 4)
                    for held_code, correlation in related.items()
                },
                "reason": "industry_tag" if tag_blocked else "return_correlation",
            })
            return True
        return False

    def has_twenty_percent_float(date):
        for code in occupied_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty:
                continue
            close = float(visible.iloc[-1]["close"])
            if any(close / lot["cost"] - 1 >= 0.20 for lot in state.right):
                return True
        return False

    def symbol_has_float_profit(code, date, threshold):
        state = states[code]
        visible = frames[code][frames[code]["date"] < date]
        if visible.empty or not state.right:
            return False
        close = float(visible.iloc[-1]["close"])
        return any(close / lot["cost"] - 1 >= float(threshold) for lot in state.right)

    def add_on_float_threshold():
        return 0.20 if current_down_streak >= 3 else 0.10

    def right_symbol_returns(date):
        returns = []
        for code in capacity_codes():
            state = states[code]
            visible = frames[code][frames[code]["date"] < date]
            if visible.empty or not state.right:
                continue
            close = float(visible.iloc[-1]["close"])
            cost = sum(lot["quantity"] * lot["cost"] for lot in state.right) / sum(
                lot["quantity"] for lot in state.right
            )
            returns.append(close / cost - 1)
        return sorted(returns, reverse=True)

    def can_open_new_right_symbol(date):
        returns = right_symbol_returns(date)
        if not returns:
            return True
        if len(returns) >= configured_positions:
            return False
        if current_up_streak >= 3:
            return all(value >= 0.10 for value in returns)
        if len(returns) == 1:
            return returns[0] >= 0.20
        return returns[0] >= 0.20 and all(value >= 0.10 for value in returns[1:])

    def trade_fee_components(turnover, *, sell=False):
        turnover = abs(float(turnover))
        if turnover <= 0:
            return {
                "commission_amount": 0.0,
                "stamp_duty_amount": 0.0,
                "slippage_amount": 0.0,
                "transaction_cost_amount": 0.0,
            }
        commission = max(
            turnover * float(commission_rate), float(minimum_commission),
        )
        stamp_duty = turnover * float(sell_stamp_duty_rate) if sell else 0.0
        slippage = turnover * float(estimated_slippage_rate)
        return {
            "commission_amount": commission,
            "stamp_duty_amount": stamp_duty,
            "slippage_amount": slippage,
            "transaction_cost_amount": commission + stamp_duty + slippage,
        }

    def add_event(
        date, code, action, price, size, reason, pnl=0.0, *,
        cost_basis=None, execution_quantity=None, entry_fee_cash=0.0, **metadata,
    ):
        nonlocal realized
        realized += pnl
        execution_basis = (
            float(price) if float(size) > 0
            else float(cost_basis) if float(size) < 0 and cost_basis is not None
            else None
        )
        if execution_quantity is None:
            execution_quantity = (
                float(initial_capital) * float(size) / execution_basis
                if execution_basis not in {None, 0.0} else 0.0
            )
        execution_quantity = float(execution_quantity)
        trade_side = "买入" if execution_quantity > 0 else "卖出" if execution_quantity < 0 else "状态"
        trade_amount = abs(execution_quantity) * float(price)
        fee_components = trade_fee_components(trade_amount, sell=execution_quantity < 0)
        allocated_entry_fee = float(entry_fee_cash) if execution_quantity < 0 else 0.0
        cost_amount = (
            abs(execution_quantity) * float(cost_basis)
            if execution_quantity < 0 and cost_basis is not None else None
        )
        gross_pnl_amount = (
            trade_amount - cost_amount
            if cost_amount is not None else None
        )
        round_trip_pnl_amount = (
            gross_pnl_amount
            - fee_components["transaction_cost_amount"]
            - allocated_entry_fee
            if gross_pnl_amount is not None else None
        )
        if execution_quantity > 0:
            profit_loss_amount = -fee_components["transaction_cost_amount"]
            profit_loss_pct = (
                profit_loss_amount / trade_amount * 100 if trade_amount else None
            )
            cash_change_amount = -trade_amount - fee_components["transaction_cost_amount"]
        elif execution_quantity < 0:
            profit_loss_amount = round_trip_pnl_amount
            profit_loss_pct = (
                round_trip_pnl_amount / (cost_amount + allocated_entry_fee) * 100
                if cost_amount is not None and cost_amount + allocated_entry_fee > 0 else None
            )
            cash_change_amount = trade_amount - fee_components["transaction_cost_amount"]
        else:
            profit_loss_amount = None
            profit_loss_pct = None
            cash_change_amount = 0.0
        events.append({
            "date": date.strftime("%Y-%m-%d"), "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "action": action, "price": round(float(price), 3),
            "execution_price": float(price),
            "position_change_pct": round(size * 100, 2),
            "reason": reason, "phase": current_phase,
            "realized_account_pct": round(pnl * 100, 4),
            # Quantity is emitted for independent broker/accounting replays.
            # Exposure is defined against initial capital in this engine, so
            # buys use their fill and exits use the same exact lot quantity.
            # Keeping the unrounded value avoids reconciliation
            # drift from the display-only percentage fields above.
            "execution_quantity": execution_quantity,
            "cost_basis": execution_basis,
            "trade_side": trade_side,
            "quantity": round(abs(execution_quantity), 8),
            "trade_amount": round(trade_amount, 2),
            "commission_amount": round(fee_components["commission_amount"], 2),
            "stamp_duty_amount": round(fee_components["stamp_duty_amount"], 2),
            "slippage_amount": round(fee_components["slippage_amount"], 2),
            "transaction_cost_amount": round(fee_components["transaction_cost_amount"], 2),
            "allocated_entry_fee_amount": round(allocated_entry_fee, 2),
            "cost_amount": None if cost_amount is None else round(cost_amount, 2),
            "gross_pnl_amount": None if gross_pnl_amount is None else round(gross_pnl_amount, 2),
            "profit_loss_amount": None if profit_loss_amount is None else round(profit_loss_amount, 2),
            "profit_loss_pct": None if profit_loss_pct is None else round(profit_loss_pct, 4),
            "cash_change_amount": round(cash_change_amount, 2),
            **metadata,
        })

    def trade_fee(turnover, *, sell=False):
        return trade_fee_components(turnover, sell=sell)["transaction_cost_amount"]

    def execute_buy(code, requested_size, price):
        """Buy against real shared cash, shrinking rather than borrowing."""
        nonlocal cash_balance, transaction_costs
        requested_notional = max(0.0, float(requested_size)) * float(initial_capital)
        price = float(price)
        if requested_notional <= 0 or price <= 0 or cash_balance <= float(minimum_commission):
            return 0.0, 0.0, 0.0, 0.0, False
        affordable_notional = _affordable_buy_notional(
            cash_balance,
            requested_notional,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_rate=estimated_slippage_rate,
        )
        if affordable_notional <= 1e-8:
            return 0.0, 0.0, 0.0, 0.0, True
        cash_limited = affordable_notional < requested_notional - 1e-8
        quantity = _board_lot_quantity(code, affordable_notional / price)
        if quantity <= 0:
            return 0.0, 0.0, 0.0, 0.0, cash_limited
        notional = quantity * price
        fee_cash = trade_fee(notional)
        cash_balance = max(0.0, cash_balance - notional - fee_cash)
        fee = fee_cash / float(initial_capital)
        transaction_costs += fee
        return notional / float(initial_capital), float(quantity), -fee, fee_cash, cash_limited

    def execute_sell(quantity, sell, cost):
        """Sell exact shares and release net proceeds into shared cash."""
        nonlocal cash_balance, transaction_costs
        quantity = max(0.0, float(quantity))
        sell = float(sell)
        cost = float(cost)
        if quantity <= 0:
            return 0.0, 0.0
        turnover = quantity * sell
        fee_cash = trade_fee(turnover, sell=True)
        cash_balance += turnover - fee_cash
        fee = fee_cash / float(initial_capital)
        transaction_costs += fee
        cost_value = quantity * cost
        size = cost_value / float(initial_capital)
        pnl = (turnover - cost_value - fee_cash) / float(initial_capital)
        return size, pnl

    def consume_merged_lots(code, state, lots, requested_quantity):
        """Consume FIFO board lots without leaving fractional batch shares."""
        remaining = _board_lot_quantity(code, requested_quantity)
        sold_quantity = 0.0
        sold_cost_value = 0.0
        sold_entry_fee = 0.0
        for lot in list(lots):
            if remaining <= 0:
                break
            original_quantity = float(lot["quantity"])
            take = min(_board_lot_quantity(code, original_quantity), remaining)
            if take <= 0:
                continue
            ratio = take / original_quantity
            sold_quantity += take
            sold_cost_value += take * float(lot["cost"])
            sold_entry_fee += float(lot.get("entry_fee_cash") or 0.0) * ratio
            lot["quantity"] = original_quantity - take
            lot["size"] *= 1.0 - ratio
            lot["entry_fee_cash"] = float(lot.get("entry_fee_cash") or 0.0) * (1.0 - ratio)
            remaining -= take
            if lot["quantity"] <= 1e-9:
                state.right.remove(lot)
        sold_cost = sold_cost_value / sold_quantity if sold_quantity else 0.0
        return sold_quantity, sold_cost, sold_entry_fee

    def left_grid_plan(value_line):
        """Ten-unit value grid: five initial units and five lower units."""
        anchor = float(value_line)
        plan = []
        for slot in range(10):
            if slot < 5:
                buy_price = anchor
            else:
                buy_price = anchor * (1.0 - configured_left_grid_step) ** (slot - 4)
            if slot < 3:
                sell_price = None
            elif slot < 5:
                sell_price = anchor * (1.0 + configured_left_grid_step * (slot - 2))
            else:
                sell_price = anchor * (1.0 - configured_left_grid_step) ** (slot - 5)
            plan.append({
                "slot": slot,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "core": slot < 3,
            })
        return plan

    calendar = sorted({
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[(frame["date"] >= start) & (frame["date"] <= end), "date"]
    })
    snapshot_dates = sorted(snapshots)
    formula_by_date = {
        pd.Timestamp(date).normalize(): value
        for date, value in formula_phases.items()
    }
    formula_dates = sorted(formula_by_date)
    for date in calendar:
        exited_today = set()
        # Close-confirmed exits become market-at-open orders on the next bar.
        for code, state in states.items():
            if not state.pending_lot_exits and not state.pending_profit_ids:
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                reason = state.pending_lot_exits.get(lot["batch"])
                if not reason:
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                state.pending_lot_exits.pop(lot["batch"], None)
                add_event(
                    date, code, "收盘条件次日退出", sell, -size,
                    f"{lot['batch']}; {reason}; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                )
                exited_today.add(code)

            merged = [lot for lot in state.right if lot.get("merged")]
            if state.pending_profit_ids and merged and state.right_parts > 0:
                executed_ids = _profit_ids_to_execute(
                    state.pending_profit_ids, state.right_parts,
                )
                parts = len(executed_ids)
                if not parts:
                    state.pending_profit_ids.clear()
                    state.pending_tail_capacity_free = False
                    continue
                merged_quantity = sum(float(lot["quantity"]) for lot in merged)
                merged_cost = sum(
                    float(lot["quantity"]) * float(lot["cost"]) for lot in merged
                ) / merged_quantity
                planned_ratio = parts / state.right_parts
                desired_quantity = merged_quantity * planned_ratio
                quantity = float(_board_lot_quantity(code, desired_quantity))
                if quantity <= 0:
                    continue
                quantity, sold_cost, entry_fee_cash = consume_merged_lots(
                    code, state, merged, quantity,
                )
                size, pnl = execute_sell(quantity, sell, sold_cost)
                state.right_sold.update(executed_ids)
                state.pending_profit_ids.difference_update(executed_ids)
                state.right_parts -= parts
                if state.right_parts == 1 and state.pending_tail_capacity_free:
                    state.right_tail_capacity_free = True
                state.pending_tail_capacity_free = False
                action = (
                    "统一尾仓次日退出" if state.right_parts == 0
                    else f"统一分仓次日止盈{parts}份"
                )
                add_event(
                    date, code, action, sell, -size,
                    f"止盈条件={','.join(executed_ids)}; 收盘确认; {fill['status']}", pnl,
                    cost_basis=sold_cost, execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                )
                exited_today.add(code)
            if not state.right:
                state.right_parts = configured_profit_tranches
                state.right_sold.clear()
                state.right_plan_date = None
                state.pending_lot_exits.clear()
                state.pending_profit_ids.clear()
                state.pending_tail_capacity_free = False
                state.right_tail_capacity_free = False

        eligible_formula_dates = [
            item for item in formula_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        formula_state = (
            formula_by_date[eligible_formula_dates[-1]]
            if eligible_formula_dates else current_phase
        )
        if isinstance(formula_state, dict):
            current_phase = str(formula_state.get("phase") or current_phase)
            current_down_streak = int(formula_state.get("window_down_streak") or 0)
            current_up_streak = int(formula_state.get("window_up_streak") or 0)
        else:
            current_phase = str(formula_state)
            current_down_streak = 0
            current_up_streak = 0

        if len(left_side_codes()) > 1:
            def left_keep_rank(code):
                state = states[code]
                score = float(last_candidate_scores.get(code, 0.0) or 0.0)
                return (
                    bool(state.right),
                    score,
                    symbol_exposure(code),
                    code,
                )

            keep_code = max(left_side_codes(), key=left_keep_rank)
            for code in left_side_codes() - {keep_code}:
                if code in pending_left_exits or code in pending_left_quota_exits:
                    continue
                pending_left_quota_exits[code] = (
                    f"全行情左侧标的限额; 保留={keep_code}"
                )

        for code, state in states.items():
            pending_lots = [
                lot for lot in state.right
                if lot.get("reconfirm_on_next_day") and pd.Timestamp(lot["date"]).normalize() < date
            ]
            if not pending_lots:
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            for lot in pending_lots:
                level = float(lot["reconfirm_level"])
                if float(row["open"]) >= level:
                    lot["reconfirm_on_next_day"] = False
                    add_event(date, code, "突破次日确认持有", row["open"], 0.0, f"{lot['batch']}; 开盘仍在{level:.3f}上方")
                    continue
                fill = fill_sell_stop(
                    row, stop_price=float("inf"), previous_close=previous_close,
                    code=code, is_st=is_st(code),
                )
                if not fill["filled"]:
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                sell = float(fill["price"])
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "突破次日未确认退出", sell, -size,
                    f"{lot['batch']}; 开盘未站上{level:.3f}; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                )
            if not state.right:
                state.right_parts = configured_profit_tranches
                state.right_sold.clear()
                state.right_plan_date = None
        for code in list(pending_candidate_exits):
            state = states.get(code)
            right_size = 0.0 if state is None else sum(lot["size"] for lot in state.right)
            if state is None or not state.right or right_size >= 0.10 - 1e-9:
                pending_candidate_exits.discard(code)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "落选尾仓退出", sell, -size,
                    f"{lot['batch']}; 右侧总仓低于10%; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                )
            state.right_parts = configured_profit_tranches
            state.right_sold.clear()
            state.right_plan_date = None
            pending_candidate_exits.discard(code)
            exited_today.add(code)
        for code in list(pending_weak_exits):
            state = states.get(code)
            if state is None or not state.right:
                pending_weak_exits.discard(code)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.right):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.right.remove(lot)
                add_event(
                    date, code, "汰弱退出", sell, -size,
                    f"{lot['batch']}; 低效率仓次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash,
                )
            state.right_parts = configured_profit_tranches
            state.right_sold.clear()
            state.right_plan_date = None
            pending_weak_exits.discard(code)
            exited_today.add(code)
        for code, reason in list(pending_left_exits.items()):
            state = states.get(code)
            if state is None or not state.left:
                pending_left_exits.pop(code, None)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.left):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.left.remove(lot)
                add_event(
                    date, code, "左侧价值证伪清仓", sell, -size,
                    f"{lot['batch']}; {reason}; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    value_line=state.left_value_line, account_mode="left",
                )
            state.left_value_line = None
            state.left_grid_started = False
            pending_left_exits.pop(code, None)
            exited_today.add(code)
        for code, reason in list(pending_left_quota_exits.items()):
            state = states.get(code)
            if state is None or not state.left:
                if state is not None:
                    state.left_value_line = None
                    state.left_grid_started = False
                pending_left_quota_exits.pop(code, None)
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            fill = fill_sell_stop(
                row, stop_price=float("inf"), previous_close=previous_close,
                code=code, is_st=is_st(code),
            )
            if not fill["filled"]:
                continue
            sell = float(fill["price"])
            for lot in list(state.left):
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, sell, lot["cost"])
                state.left.remove(lot)
                add_event(
                    date, code, "左侧全行情限额清仓", sell, -size,
                    f"{lot['batch']}; {reason}; 次日开盘退出", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    value_line=state.left_value_line, account_mode="left",
                )
            state.left_value_line = None
            state.left_grid_started = False
            pending_left_quota_exits.pop(code, None)
            exited_today.add(code)
        eligible_snapshots = [
            item for item in snapshot_dates
            if item < date or (not signals_effective_next_day and item <= date)
        ]
        if eligible_snapshots:
            current_candidates = snapshots[eligible_snapshots[-1]]
            for candidate_code, candidate in current_candidates.items():
                last_candidate_scores[candidate_code] = float(candidate.get("candidate_score") or 0.0)
        current_candidate_codes = set(current_candidates)
        if exit_tail_on_candidate_removal and previous_candidate_codes:
            for code in set(states) - current_candidate_codes:
                state = states.get(code)
                if state and state.right and sum(lot["size"] for lot in state.right) < 0.10 - 1e-9:
                    pending_candidate_exits.add(code)
        previous_candidate_codes = current_candidate_codes
        for code, state in states.items():
            if not state.left or code in pending_left_exits:
                continue
            reason = left_value_falsification_reason(current_candidates.get(code))
            if reason:
                pending_left_exits[code] = reason
        # Existing grids remain active even after dropping out of the daily
        # top ten. New campaigns, however, compete by candidate score so file
        # or ticker ordering can never decide which four symbols get capital.
        active_left_codes = [
            code for code, state in states.items()
            if state.left_value_line is not None and (state.left or state.left_grid_started)
        ]
        new_left_candidates = []
        for code, candidate in current_candidates.items():
            if code not in states or states[code].left_value_line is not None:
                continue
            if not left_candidate_can_add(candidate):
                continue
            value_line = pd.to_numeric(candidate.get("value_line"), errors="coerce")
            if pd.notna(value_line) and float(value_line) > 0:
                new_left_candidates.append((
                    float(candidate.get("candidate_score") or 0.0),
                    code, float(value_line),
                ))
        left_targets = [
            (code, float(states[code].left_value_line))
            for code in active_left_codes
        ] + [
            (code, value_line)
            for _, code, value_line in sorted(new_left_candidates, reverse=True)
        ]

        for code, grid_anchor in left_targets:
            state = states[code]
            if code in pending_left_exits:
                continue
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            previous_close = data.iloc[index - 1]["close"] if index > 0 else None
            candidate = current_candidates.get(code, {})
            can_add_left = left_candidate_can_add(candidate)
            plan_by_slot = {
                item["slot"]: item for item in left_grid_plan(grid_anchor)
            }
            sold_slots = set()

            # Only non-core grid units have resting sell orders.  Selling a
            # right-side campaign elsewhere never reaches this list.
            for lot in list(state.left):
                sell_price = lot.get("sell_price")
                if sell_price is None:
                    continue
                fill = fill_limit_order(
                    row, side="sell", limit_price=float(sell_price),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
                if not fill["filled"]:
                    continue
                quantity = float(lot["quantity"])
                entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                size, pnl = execute_sell(quantity, float(fill["price"]), lot["cost"])
                state.left.remove(lot)
                sold_slots.add(int(lot["slot"]))
                add_event(
                    date, code, "左侧网格卖出", fill["price"], -size,
                    f"{lot['batch']}; 上一格卖出; {fill['status']}", pnl,
                    cost_basis=lot["cost"], execution_quantity=-quantity,
                    entry_fee_cash=entry_fee_cash, grid_slot=int(lot["slot"]),
                    value_line=grid_anchor, account_mode="left",
                )

            held_slots = {int(lot["slot"]) for lot in state.left}
            starting_new_left_plan = (
                not state.left_grid_started and code not in left_side_codes()
            )
            starting_new_symbol = starting_new_left_plan and code not in capacity_codes()
            if starting_new_left_plan:
                if (
                    total_symbol_limit_reached(code)
                    or left_symbol_limit_reached(code)
                ):
                    continue
                if starting_new_symbol:
                    if symbol_limit_reached() or industry_limit_reached(code, candidate, date):
                        continue
            for planned in plan_by_slot.values():
                if not can_add_left:
                    continue
                slot = int(planned["slot"])
                if slot in held_slots or slot in sold_slots:
                    continue
                # The first order is five equal units at the value line.  The
                # remaining units are pre-placed at successively lower grids.
                if not state.left_grid_started and slot >= 5 and not {
                    0, 1, 2, 3, 4,
                }.issubset(held_slots):
                    continue
                current_left = sum(float(lot["size"]) for lot in state.left)
                if (
                    not left_position_counts_capacity(state)
                    and not bool(planned["core"])
                    and (
                        symbol_limit_reached()
                        or total_symbol_limit_reached(code)
                        or left_symbol_limit_reached(code)
                        or industry_limit_reached(code, candidate, date)
                    )
                ):
                    continue
                requested_size = min(
                    configured_left_grid_unit,
                    configured_left_grid_max_exposure - current_left,
                )
                requested_size = _capped_entry_size(
                    symbol_exposure(code), requested_size, max_symbol_exposure,
                )
                if requested_size <= 1e-9 or gross_exposure() + requested_size > 1.0 + 1e-9:
                    continue
                fill = fill_limit_order(
                    row, side="buy", limit_price=float(planned["buy_price"]),
                    previous_close=previous_close, code=code, is_st=is_st(code),
                )
                if not fill["filled"]:
                    continue
                size, quantity, pnl, entry_fee_cash, cash_limited = execute_buy(
                    code, requested_size, float(fill["price"]),
                )
                if size <= 1e-9:
                    continue
                lot = {
                    "cost": float(fill["price"]), "date": row["date"],
                    "buy_price": float(planned["buy_price"]),
                    "sell_price": planned["sell_price"], "size": size,
                    "quantity": quantity, "entry_fee_cash": entry_fee_cash,
                    "slot": slot, "batch": f"L{slot + 1}",
                    "core": bool(planned["core"]),
                    "industry_tags": sorted(
                        candidate_industry_tags(candidate)
                        or {
                            tag for existing in state.left
                            for tag in existing.get("industry_tags", [])
                        }
                    ),
                }
                if state.left_value_line is None:
                    state.left_value_line = grid_anchor
                state.left.append(lot)
                held_slots.add(slot)
                if {0, 1, 2, 3, 4}.issubset(held_slots):
                    state.left_grid_started = True
                add_event(
                    date, code, "左侧网格买入", fill["price"], size,
                    f"{lot['batch']}; 价值线网格第{slot + 1}份; {fill['status']}", pnl,
                    execution_quantity=quantity, entry_fee_cash=entry_fee_cash,
                    requested_position_pct=requested_size * 100,
                    cash_limited=cash_limited,
                    lot_rounded=size < requested_size - 1e-9 and not cash_limited,
                    board_lot_size=board_lot_size(code), grid_slot=slot,
                    value_line=grid_anchor, account_mode="left",
                )

        for code, state in states.items():
            data = frames[code]
            indexes = data.index[data["date"].dt.normalize() == date]
            if indexes.empty:
                continue
            index = int(indexes[0])
            row = data.iloc[index]
            history = data.iloc[: index + 1]
            if state.right:
                for lot in list(state.right):
                    prior_close = float(data.iloc[index - 1]["close"]) if index > 0 else None
                    if prior_close is not None and prior_close / lot["cost"] - 1 >= 0.10:
                        lot["proven"] = True
                    if not _entry_risk_still_controls_lot(lot, state):
                        continue
                    time_limit = 8 if current_down_streak >= 3 else 13
                    risk = position_exit_snapshot(
                        history, lot["cost"], lot["date"], entry_mode="right",
                        condition_stop=lot["stop"], time_limit_days=time_limit,
                    )
                    if risk.get("space_stop_triggered"):
                        quantity = float(lot["quantity"])
                        stop_price = float(risk["space_stop"])
                        previous_close = data.iloc[index - 1]["close"] if index > 0 else None
                        fill = fill_sell_stop(
                            row, stop_price=stop_price,
                            previous_close=previous_close, code=code, is_st=is_st(code),
                        )
                        if not fill["filled"]:
                            continue
                        execution_price = float(fill["price"])
                        execution_reason = f"hard space stop {stop_price:.3f}; {fill['status']}"
                        entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                        size, pnl = execute_sell(quantity, execution_price, lot["cost"])
                        state.right.remove(lot)
                        state.pending_lot_exits.pop(lot["batch"], None)
                        exited_today.add(code)
                        add_event(
                            date, code, risk["position_action"], execution_price,
                            -size, f"{lot['batch']}; {execution_reason}", pnl,
                            cost_basis=lot["cost"], execution_quantity=-quantity,
                            entry_fee_cash=entry_fee_cash,
                        )
                    elif risk.get("condition_stop_triggered") or risk.get("entry_time_stop"):
                        reason = "condition stop" if risk.get("condition_stop_triggered") else "entry time condition"
                        if close_confirmed_execution == "close_proxy":
                            quantity = float(lot["quantity"])
                            sell = float(row["close"])
                            entry_fee_cash = float(lot.get("entry_fee_cash") or 0.0)
                            size, pnl = execute_sell(quantity, sell, lot["cost"])
                            state.right.remove(lot)
                            state.pending_lot_exits.pop(lot["batch"], None)
                            add_event(
                                date, code, "收盘条件退出", sell, -size,
                                f"{lot['batch']}; {reason}; 14:55/close proxy", pnl,
                                cost_basis=lot["cost"], execution_quantity=-quantity,
                                entry_fee_cash=entry_fee_cash,
                            )
                            exited_today.add(code)
                        else:
                            state.pending_lot_exits.setdefault(lot["batch"], reason)
                if state.right:
                    if not any(lot["merged"] for lot in state.right):
                        state.right_parts = configured_profit_tranches
                        state.right_sold.clear()
                    merge_threshold = 0.20 if current_down_streak >= 3 else 0.10
                    merged_new_batch = False
                    for lot in state.right:
                        if not lot["merged"] and prior_close is not None and prior_close / lot["cost"] - 1 >= merge_threshold:
                            lot["merged"] = True
                            merged_new_batch = True
                            add_event(date, code, "加仓批次合并", row["close"], 0.0, f"{lot['batch']}; 浮盈达到{merge_threshold:.0%}")

                    if merged_new_batch:
                        # A new add-on belongs to the existing campaign.  Keep
                        # already executed profit tranches and the original
                        # peak-observation window instead of resetting both.
                        state.right_parts = max(1, state.right_parts)

                    merged = [lot for lot in state.right if lot["merged"]]
                    if merged and state.right_parts > 0:
                        merged_quantity = sum(float(lot["quantity"]) for lot in merged)
                        merged_cost = sum(
                            float(lot["quantity"]) * float(lot["cost"])
                            for lot in merged
                        ) / merged_quantity
                        merged_entry = state.right_plan_date or min(lot["date"] for lot in merged)
                        profit = position_exit_snapshot(
                            history, merged_cost, merged_entry, entry_mode="right",
                            condition_stop=None, time_limit_days=9999,
                            exit_tranches=configured_profit_tranches,
                        )
                        active_profit_ids = _active_profit_trigger_ids(
                            profit, history, state.right_parts,
                        )
                        new_ids = active_profit_ids - state.right_sold
                        if new_ids:
                            if close_confirmed_execution == "close_proxy":
                                executed_ids = _profit_ids_to_execute(
                                    new_ids, state.right_parts,
                                )
                                parts = len(executed_ids)
                                if not parts:
                                    continue
                                sell = float(row["close"])
                                planned_ratio = parts / state.right_parts
                                desired_quantity = merged_quantity * planned_ratio
                                quantity = float(_board_lot_quantity(code, desired_quantity))
                                if quantity <= 0:
                                    continue
                                quantity, sold_cost, entry_fee_cash = consume_merged_lots(
                                    code, state, merged, quantity,
                                )
                                size, pnl = execute_sell(quantity, sell, sold_cost)
                                state.right_sold.update(executed_ids)
                                state.right_parts -= parts
                                if _qualifies_profit_tail(
                                    profit, state.right_parts,
                                    configured_profit_tail_min_return,
                                ):
                                    state.right_tail_capacity_free = True
                                action = (
                                    "统一尾仓收盘退出" if state.right_parts == 0
                                    else f"统一分仓收盘止盈{parts}份"
                                )
                                add_event(
                                    date, code, action, sell, -size,
                                    f"止盈条件={','.join(executed_ids)}; 14:55/close proxy", pnl,
                                    cost_basis=sold_cost, execution_quantity=-quantity,
                                    entry_fee_cash=entry_fee_cash,
                                )
                                if state.right_parts == 0:
                                    for lot in list(state.right):
                                        if lot.get("merged"):
                                            state.right.remove(lot)
                                    state.right_parts = configured_profit_tranches
                                    state.right_sold.clear()
                                    state.right_plan_date = None
                                    state.right_tail_capacity_free = False
                                    exited_today.add(code)
                            else:
                                executable_ids = _profit_ids_to_execute(
                                    new_ids, state.right_parts,
                                )
                                state.pending_profit_ids.update(executable_ids)
                                scheduled_parts = len(executable_ids)
                                state.pending_tail_capacity_free = _qualifies_profit_tail(
                                    profit, state.right_parts - scheduled_parts,
                                    configured_profit_tail_min_return,
                                )
                    if not state.right:
                        state.right_parts = configured_profit_tranches
                        state.right_sold.clear()
                        state.right_plan_date = None

        candidates = []
        entry_allowed = (
            current_down_streak < 3
            or not capacity_codes()
            or has_twenty_percent_float(date)
        )
        if entry_allowed:
            for code in current_candidates:
                if code not in states or code in exited_today:
                    continue
                if not _enabled(current_candidates[code].get("signal_eligible"), default=True):
                    continue
                candidate = current_candidates[code]
                allow_right = _enabled(candidate.get("allow_right"), default=True)
                # A pure value candidate may add a separately managed right
                # batch only after its left position exists (left-to-right).
                if not allow_right and not states[code].left:
                    continue
                data = frames[code]
                indexes = data.index[data["date"].dt.normalize() == date]
                if indexes.empty:
                    continue
                index = int(indexes[0])
                signal = _right_signal(
                    data, index, plans.get(code),
                    auto_price_structure=auto_price_structure,
                    allow_structure_pullback=allow_structure_pullback,
                )
                if signal:
                    if (
                        float(signal.get("entry_evidence_score") or 0.0)
                        < configured_min_entry_evidence_score
                    ):
                        continue
                    if signal.get("signal_type") == "uptrend_support_pullback":
                        if not states[code].right:
                            continue
                        if not symbol_has_float_profit(code, date, add_on_float_threshold()):
                            continue
                    fundamental_score = float(current_candidates[code].get("candidate_score") or 0.0)
                    entry_priority = (
                        fundamental_score
                        + float(signal["rank"]) * 8
                        + min(float(signal["known_volume_ratio"]), 5.0) * 2
                    )
                    candidates.append((
                        entry_priority, fundamental_score, signal["rank"],
                        code, index, signal,
                    ))
            for _, fundamental_score, __, code, index, signal in sorted(candidates, reverse=True):
                reopening_profit_tail = _is_profit_tail(states[code])
                opening_right_symbol = code not in capacity_codes()
                candidate = current_candidates.get(code, {})
                if opening_right_symbol:
                    if total_symbol_limit_reached(code):
                        continue
                    if industry_limit_reached(code, candidate, date):
                        continue
                    if not can_open_new_right_symbol(date):
                        continue
                    if symbol_limit_reached():
                        incoming_stop_pct = max(0.0, 1 - float(signal["stop"]) / float(frames[code].iloc[index]["close"]))
                        incoming_score = fundamental_score + signal["rank"] * 10 + min(float(signal["known_volume_ratio"]), 5.0) * 2 - incoming_stop_pct * 20
                        weak = []
                        for held_code in capacity_codes() - pending_weak_exits:
                            held_state = states[held_code]
                            if not held_state.right:
                                continue
                            visible = frames[held_code][frames[held_code]["date"] <= date]
                            if visible.empty:
                                continue
                            held_row = visible.iloc[-1]
                            held_quantity = sum(lot["quantity"] for lot in held_state.right)
                            held_cost = sum(
                                lot["quantity"] * lot["cost"] for lot in held_state.right
                            ) / held_quantity
                            held_return = float(held_row["close"]) / held_cost - 1
                            holding_days = max(
                                len(visible[visible["date"] >= min(lot["date"] for lot in held_state.right)]), 1,
                            )
                            below_ma20 = pd.notna(held_row["ma20"]) and float(held_row["close"]) < float(held_row["ma20"])
                            removed = held_code not in current_candidates
                            eligible = below_ma20 and holding_days >= 5 and held_return < 0.10
                            if not eligible:
                                continue
                            held_fundamental = float(last_candidate_scores.get(held_code, 0.0))
                            held_score = held_fundamental + held_return * 100 + (20 if any(lot.get("proven") for lot in held_state.right) else 0) + (-10 if below_ma20 else 10)
                            replacement_margin = 5 if removed else 15
                            if incoming_score >= held_score + replacement_margin:
                                weak.append((held_score, held_code))
                        if weak:
                            pending_weak_exits.add(min(weak)[1])
                        continue
                row = frames[code].iloc[index]
                previous_close = frames[code].iloc[index - 1]["close"] if index > 0 else None
                if signal["order_type"] == "stop":
                    fill = fill_buy_stop(
                        row, trigger_price=signal["trigger"],
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                elif signal["order_type"] == "close":
                    fill = {
                        "filled": True,
                        "price": float(row["close"]),
                        "status": "close_confirmed",
                    }
                else:
                    fill = fill_limit_order(
                        row, side="buy", limit_price=signal["trigger"],
                        previous_close=previous_close, code=code, is_st=is_st(code),
                    )
                if not fill["filled"]:
                    continue
                existing_size = sum(lot["size"] for lot in states[code].right)
                starting_new_right_campaign = not states[code].right or reopening_profit_tail
                if starting_new_right_campaign:
                    left_to_right = (
                        bool(states[code].left)
                        and not _enabled(candidate.get("allow_right"), default=True)
                    )
                    preferred_breakout = signal.get("signal_type") in {
                        "uptrend_50_reclaim",
                        "pullback_50_breakout",
                        "w_bottom_neckline",
                        "gap_long_ma_breakout",
                        "strong_trend_breakout",
                        "consolidation_breakout",
                        "volume_price_node",
                        "bull_run_half_pullback",
                    }
                    inferred_entry = signal.get("signal_type") in {
                        "consolidation_breakout",
                        "strong_trend_breakout",
                        "volume_price_node",
                        "bull_run_half_pullback",
                    }
                    direct_breakout = signal["order_type"] == "stop"
                    size = (
                        0.20 if current_down_streak >= 3
                        else 0.30 if preferred_breakout
                        else 0.20
                    )
                    if preferred_breakout:
                        entry_kind = "33未三日下行首仓" if current_down_streak < 3 else "33三日下行试错首仓"
                    elif inferred_entry:
                        entry_kind = "技术合理买点试探仓"
                    elif direct_breakout:
                        entry_kind = "直接突破首仓"
                    else:
                        entry_kind = "33上行首仓" if current_up_streak >= 3 else "33未上行首仓"
                    if reopening_profit_tail:
                        entry_kind = f"尾仓再入; {entry_kind}"
                    elif left_to_right:
                        left_size = sum(float(lot["size"]) for lot in states[code].left)
                        size = min(size, left_size * 0.50)
                        entry_kind = f"左转右加仓(不超过左仓一半); {entry_kind}"
                else:
                    if any(not lot.get("proven", False) for lot in states[code].right):
                        continue
                    pullback_entry = signal.get("signal_type") in {
                        "uptrend_support_pullback",
                        "bull_run_half_pullback",
                    }
                    ratio = 0.50 if pullback_entry else 1 / 3
                    size = existing_size * ratio
                    entry_kind = f"浮盈加仓{ratio:.0%}"
                size = _capped_entry_size(
                    symbol_exposure(code), size, max_symbol_exposure,
                )
                if size <= 1e-9:
                    continue
                if gross_exposure() + size > 1.0 + 1e-9:
                    continue
                requested_size = size
                size, quantity, pnl, entry_fee_cash, cash_limited = execute_buy(
                    code, requested_size, float(fill["price"]),
                )
                if size <= 1e-9:
                    continue
                if reopening_profit_tail:
                    for lot in states[code].right:
                        lot["merged"] = True
                    states[code].right_parts = configured_profit_tranches
                    states[code].right_sold.clear()
                    states[code].pending_profit_ids.clear()
                    states[code].pending_tail_capacity_free = False
                    states[code].right_tail_capacity_free = False
                    states[code].right_plan_date = row["date"]
                elif not states[code].right:
                    states[code].pending_tail_capacity_free = False
                    states[code].right_tail_capacity_free = False
                batch = f"R{next_batch_id}"
                next_batch_id += 1
                states[code].right.append({
                    "cost": float(fill["price"]), "date": row["date"],
                    "stop": float(signal["stop"]), "size": size,
                    "quantity": quantity,
                    "entry_fee_cash": entry_fee_cash,
                    "batch": batch,
                    "merged": reopening_profit_tail or not bool(states[code].right),
                    "proven": False,
                    "industry_tags": sorted(candidate_industry_tags(candidate)),
                    "reconfirm_level": float(signal["trigger"]),
                    "reconfirm_on_next_day": (
                        bool(signal.get("requires_next_day_confirmation"))
                        or (
                            signal["order_type"] == "stop"
                            and float(row["close"]) < float(signal["trigger"])
                        )
                    ),
                })
                if starting_new_right_campaign:
                    states[code].right_parts = _effective_profit_tranches(
                        code,
                        sum(float(lot["quantity"]) for lot in states[code].right),
                        configured_profit_tranches,
                    )
                if not states[code].right_plan_date:
                    states[code].right_plan_date = row["date"]
                structure_metadata = {
                    key: signal.get(key)
                    for key in (
                        "structure_ratio", "anchor_low", "anchor_high",
                        "anchor_pullback_low", "anchor_low_date",
                        "anchor_high_date", "anchor_pullback_low_date",
                        "signal_type", "gap_up",
                        "entry_evidence_score", "entry_evidence",
                        "volume_node_date", "volume_node_confirmed_on",
                        "valid_volume_node_count",
                    )
                    if signal.get(key) is not None
                }
                structure_metadata["industry_tags"] = "、".join(
                    sorted(candidate_industry_tags(candidate))
                )
                add_event(
                    date, code, "右侧买入", fill["price"], size,
                    f"{batch}; {entry_kind}; {signal['reason']}; {fill['status']}",
                    pnl, execution_quantity=quantity,
                    entry_fee_cash=entry_fee_cash,
                    requested_position_pct=requested_size * 100,
                    cash_limited=cash_limited,
                    lot_rounded=size < requested_size - 1e-9 and not cash_limited,
                    board_lot_size=board_lot_size(code),
                    selection_reason=current_candidates[code].get("selection_reason"),
                    trade_basis_score=current_candidates[code].get("trade_basis_score"),
                    trade_basis_reason=current_candidates[code].get("trade_basis_reason"),
                    technical_alignment=current_candidates[code].get("technical_alignment"),
                    ima_web_validation=current_candidates[code].get("ima_web_validation"),
                    **structure_metadata,
                )

        unrealized = 0.0
        market_value = float(cash_balance)
        for code, state in states.items():
            data = frames[code]
            visible = data[data["date"] <= date]
            if visible.empty:
                continue
            close = float(visible.iloc[-1]["close"])
            if state.left or state.right:
                unrealized += sum(
                    float(lot["quantity"]) * (close - float(lot["cost"]))
                    / float(initial_capital)
                    for lot in state.left + state.right
                )
                market_value += sum(
                    float(lot["quantity"]) * close
                    for lot in state.left + state.right
                )
        equity = market_value / float(initial_capital)
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": float(cash_balance),
            "cash_pct": round(float(cash_balance) / market_value * 100, 4),
            "gross_exposure_pct": round(gross_exposure() * 100, 2),
            "capacity_position_count": len(capacity_codes()),
            "total_held_symbol_count": len(occupied_codes()),
            "left_side_symbol_count": len(left_side_codes()),
            "right_market_left_side_symbol_count": (
                len(left_side_codes()) if right_market_active() else 0
            ),
            "profit_tail_count": sum(
                _is_profit_tail(state) for state in states.values()
            ),
            "candidate_count": len(current_candidates),
        })

    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        maximum_drawdown = 0.0
        final_equity = 1.0
    else:
        drawdown = curve["equity"] / curve["equity"].cummax() - 1
        maximum_drawdown = float(drawdown.min())
        final_equity = float(curve.iloc[-1]["equity"])
    precoverage_dates = {
        date.normalize()
        for frame in frames.values()
        for date in frame.loc[
            (frame["date"] >= requested) & (frame["date"] < coverage_start),
            "date",
        ]
    }
    final_position_details = []
    for code in sorted(occupied_codes()):
        state = states[code]
        frame = frames[code]
        visible = frame[frame["date"] <= end]
        close = None if visible.empty else float(visible.iloc[-1]["close"])
        left_size = sum(float(lot["size"]) for lot in state.left)
        left_quantity = sum(float(lot["quantity"]) for lot in state.left)
        left_value = sum(float(lot["quantity"]) * float(lot["cost"]) for lot in state.left)
        left_entry_fees = sum(float(lot.get("entry_fee_cash") or 0.0) for lot in state.left)
        right_size = sum(float(lot["size"]) for lot in state.right)
        right_quantity = sum(float(lot["quantity"]) for lot in state.right)
        right_value = sum(float(lot["quantity"]) * float(lot["cost"]) for lot in state.right)
        right_entry_fees = sum(float(lot.get("entry_fee_cash") or 0.0) for lot in state.right)
        total_quantity = left_quantity + right_quantity
        total_value = left_value + right_value
        invested_amount = total_value + left_entry_fees + right_entry_fees
        market_value = None if close is None else total_quantity * close
        unrealized_pnl_amount = None if market_value is None else market_value - invested_amount
        final_position_details.append({
            "code": code,
            "name": candidate_names.get(code) or str(plans.get(code, {}).get("name") or code),
            "close": close,
            "position_pct": round((left_size + right_size) * 100, 2),
            "left_position_pct": round(left_size * 100, 2),
            "right_position_pct": round(right_size * 100, 2),
            "position_mode": (
                "left+right" if state.left and state.right
                else "left" if state.left else "right"
            ),
            "capacity_counted": left_position_counts_capacity(state)
            or (bool(state.right) and not _is_profit_tail(state)),
            "profit_tail": _is_profit_tail(state),
            "quantity": round(total_quantity, 8),
            "invested_amount": round(invested_amount, 2),
            "market_value": None if market_value is None else round(market_value, 2),
            "unrealized_pnl_amount": (
                None if unrealized_pnl_amount is None else round(unrealized_pnl_amount, 2)
            ),
            "unrealized_pnl_pct": (
                None if unrealized_pnl_amount is None or invested_amount <= 0
                else round(unrealized_pnl_amount / invested_amount * 100, 4)
            ),
            "cost": None if not total_quantity else round(total_value / total_quantity, 3),
            "left_value_line": state.left_value_line,
            "left_batches": [
                {
                    "batch": lot["batch"], "grid_slot": int(lot["slot"]),
                    "position_pct": round(lot["size"] * 100, 2),
                    "quantity": round(lot["quantity"], 8),
                    "cost": round(lot["cost"], 3),
                    "sell_price": (
                        None if lot.get("sell_price") is None
                        else round(float(lot["sell_price"]), 3)
                    ),
                    "core": bool(lot.get("core")),
                }
                for lot in state.left
            ],
            "batches": [
                {"batch": lot["batch"], "position_pct": round(lot["size"] * 100, 2), "quantity": round(lot["quantity"], 8), "cost": round(lot["cost"], 3), "stop": round(lot["stop"], 3), "merged": lot["merged"], "proven": lot.get("proven", False)}
                for lot in state.right
            ],
        })
    trade_ledger = [
        event for event in events
        if abs(float(event.get("execution_quantity") or 0.0)) > 1e-12
    ]
    buy_trades = [event for event in trade_ledger if event["trade_side"] == "买入"]
    sell_trades = [event for event in trade_ledger if event["trade_side"] == "卖出"]
    profitable_sells = [
        event for event in sell_trades if float(event.get("profit_loss_amount") or 0.0) > 0
    ]
    losing_sells = [
        event for event in sell_trades if float(event.get("profit_loss_amount") or 0.0) < 0
    ]
    trade_summary = {
        "buy_count": len(buy_trades),
        "sell_count": len(sell_trades),
        "buy_amount": round(sum(float(event["trade_amount"]) for event in buy_trades), 2),
        "sell_amount": round(sum(float(event["trade_amount"]) for event in sell_trades), 2),
        "transaction_cost_amount": round(
            sum(float(event["transaction_cost_amount"]) for event in trade_ledger), 2,
        ),
        "closed_trade_net_pnl_amount": round(
            sum(float(event.get("profit_loss_amount") or 0.0) for event in sell_trades), 2,
        ),
        "profitable_sell_count": len(profitable_sells),
        "losing_sell_count": len(losing_sells),
        "sell_win_rate_pct": round(
            len(profitable_sells) / len(sell_trades) * 100, 3,
        ) if sell_trades else 0.0,
    }
    right_buys = [event for event in events if event["action"] == "右侧买入"]
    structure_signal_counts = {
        "uptrend_ratio_pullback": sum(
            "上涨波段" in event["reason"] and "拉回支撑" in event["reason"]
            for event in right_buys
        ),
        "uptrend_50_reclaim": sum(
            "上涨波段50%有效跌破后重新突破" in event["reason"]
            for event in right_buys
        ),
        "pullback_50_breakout": sum(
            "回调波段50%向上突破" in event["reason"]
            for event in right_buys
        ),
    }
    return {
        "requested_start": requested.strftime("%Y-%m-%d"),
        "actual_start": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "candidate_coverage_start": coverage_start.strftime("%Y-%m-%d"),
        "coverage_complete": not precoverage_dates,
        "missing_candidate_trade_dates": sorted(date.strftime("%Y-%m-%d") for date in precoverage_dates),
        "events": events,
        "trade_ledger": trade_ledger,
        "trade_summary": trade_summary,
        "equity_curve": equity_curve,
        "event_count": len(events),
        "cash_limited_order_count": sum(
            bool(event.get("cash_limited")) for event in events
        ),
        "lot_rounded_order_count": sum(
            bool(event.get("lot_rounded")) for event in events
        ),
        "candidate_pool_limit": MAX_DAILY_CANDIDATES,
        "max_positions": configured_positions,
        "max_total_held_symbols": configured_total_held_symbols,
        "profit_tranches": configured_profit_tranches,
        "profit_tails_consume_capacity": False,
        "profit_tail_minimum_current_return_pct": round(
            configured_profit_tail_min_return * 100, 2,
        ),
        "left_grid_unit_pct": round(configured_left_grid_unit * 100, 2),
        "left_grid_step_pct": round(configured_left_grid_step * 100, 2),
        "left_grid_max_exposure_pct": round(
            configured_left_grid_max_exposure * 100, 2,
        ),
        "maximum_total_held_symbols": max(
            (row["total_held_symbol_count"] for row in equity_curve), default=0,
        ),
        "maximum_left_side_symbols": max(
            (row["left_side_symbol_count"] for row in equity_curve), default=0,
        ),
        "maximum_right_market_left_side_symbols": max(
            (
                row["right_market_left_side_symbol_count"]
                for row in equity_curve
            ),
            default=0,
        ),
        "maximum_profit_tail_count": max(
            (row["profit_tail_count"] for row in equity_curve), default=0,
        ),
        "max_same_industry": configured_same_industry,
        "same_theme_correlation": configured_theme_correlation,
        "min_entry_evidence_score": configured_min_entry_evidence_score,
        "concentration_block_count": len(concentration_blocks),
        "concentration_blocks": concentration_blocks,
        "board_lot_policy": {"sh.688": 200, "default": 100},
        "structure_signal_counts": structure_signal_counts,
        "realized_return_pct": round(realized * 100, 3),
        "unrealized_return_pct": round((final_equity - 1 - realized) * 100, 3),
        "final_return_pct": round((final_equity - 1) * 100, 3),
        "maximum_drawdown_pct": round(maximum_drawdown * 100, 3),
        "maximum_gross_exposure_pct": round(
            max((row["gross_exposure_pct"] for row in equity_curve), default=0.0), 2,
        ),
        "transaction_cost_pct": round(transaction_costs * 100, 3),
        "final_cash": round(float(cash_balance), 6),
        "final_cash_pct": round(
            float(cash_balance) / (final_equity * float(initial_capital)) * 100, 6,
        ),
        "commission_rate": float(commission_rate),
        "minimum_commission": float(minimum_commission),
        "initial_capital": float(initial_capital),
        "sell_stamp_duty_rate": float(sell_stamp_duty_rate),
        "estimated_slippage_rate": float(estimated_slippage_rate),
        "max_symbol_exposure_pct": round(float(max_symbol_exposure) * 100, 2),
        "exit_tail_on_candidate_removal": bool(exit_tail_on_candidate_removal),
        "signals_effective_next_day": bool(signals_effective_next_day),
        "auto_price_structure": bool(auto_price_structure),
        "allow_structure_pullback": bool(allow_structure_pullback),
        "close_confirmed_execution": close_confirmed_execution,
        "pending_candidate_exit_codes": sorted(pending_candidate_exits),
        "final_positions": final_position_details,
    }
