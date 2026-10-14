"""Capture the frozen six-factor model and replay its paper portfolio.

Only an actual same-day capture can enter this forward stream. Historical
research uses separate tools. No broker interface is imported or called.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

from stock_research.backtesting.baseline import BASELINE_PATH, load_baseline
from stock_research.backtesting.engine import run_portfolio_backtest
from stock_research.data.forward_sync import DAILY_DATASETS, verify_completed_receipts
from stock_research.research.quantsplaybook_chunked import (
    _apply_combination, _one_network_price_only_low_volatility_spec,
)
from stock_research.research.quantsplaybook_compare import (
    _drop_ineligible_invalid_adjustment_rows, _iter_price_frames,
    factor_candidate_snapshots, point_in_time_tradable_universe,
)
from stock_research.research.quantsplaybook_hybrid import merge_hybrid_rows
from stock_research.research.quantsplaybook_liquidity_universe import _rolling_average_amount
from stock_research.storage import Database, TushareRepository
from stock_research.strategies.quantsplaybook_selection import calculate_daily_factor_panel


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_fingerprint():
    root = BASELINE_PATH.parent.parent
    packages = {}
    for package in ("numpy", "pandas", "networkx", "scipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "packages": packages,
            "files": {path.relative_to(root).as_posix(): sha256(path)
                      for path in sorted((root / "stock_research").rglob("*.py"))}}


def verify_replay_prefix(previous, current, through_date):
    """Later data revisions must never silently rewrite published paper history."""
    def equal(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)
        return left == right

    for field in ("events", "trade_ledger", "equity_curve"):
        old = _safe(previous.get(field, []))
        prefix = _safe([item for item in current.get(field, [])
                        if pd.Timestamp(item["date"]) <= pd.Timestamp(through_date)])
        if not equal(old, prefix):
            raise ValueError(f"published paper history changed: {field}; preserve old stream and investigate revisions")


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(document), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def capture_clock(observation_date, now=None):
    current = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("capture clock must have an explicit timezone")
    current = current.tz_convert("Asia/Shanghai")
    date = pd.Timestamp(observation_date).normalize()
    if date.tzinfo is not None:
        raise ValueError("observation_date must be a local calendar date")
    if date != current.tz_localize(None).normalize() or (current.hour, current.minute) < (16, 30):
        raise ValueError("forward capture requires today's completed session after 16:30")
    return current.tz_convert("UTC").isoformat()


def validate_sync_receipt(state, database, observation_date, captured_at):
    day = pd.Timestamp(observation_date).strftime("%Y-%m-%d")
    if state.get("status") != "completed" or state.get("latest_session") != day:
        raise ValueError("complete synchronization for the observation date is required")
    if Path(state.get("database", "")).resolve() != Path(database).resolve():
        raise ValueError("synchronization receipt belongs to another database")
    if not state.get("financial_capture_requested"):
        raise ValueError("financial capture was skipped")
    now = pd.Timestamp(captured_at)
    for dataset in (*DAILY_DATASETS, "dividend"):
        receipt = state.get("completed", {}).get(f"{dataset}:{day}", {})
        timestamp = pd.to_datetime(receipt.get("captured_at"), utc=True, errors="coerce")
        if pd.isna(timestamp) or timestamp > now or timestamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != day:
            raise ValueError(f"missing actual same-day capture: {dataset}")


def build_right_snapshot(prices, stock_basic, stock_st, observation_date):
    """Use the original full tradable cross-section before the ADV filter."""
    day = pd.Timestamp(observation_date).normalize()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.loc[prices["date"].le(day)].copy()
    if prices.empty or prices["date"].max() != day:
        raise ValueError("observation-date prices are missing")
    if "raw_to_qfq_factor" not in prices:
        raise ValueError("raw_to_qfq_factor is required; adjustment cannot default to one")
    basic = stock_basic.copy()
    for column in ("list_date", "delist_date"):
        basic[column] = pd.to_datetime(basic[column], errors="coerce")
    # IPO/delisting-invalid rows must not contribute to cross-sectional factors.
    bounds = basic[["code", "list_date", "delist_date"]].drop_duplicates("code")
    prices = prices.merge(bounds, on="code", how="inner")
    prices = prices.loc[
        prices["list_date"].notna() & prices["date"].ge(prices["list_date"])
        & (prices["delist_date"].isna() | prices["date"].le(prices["delist_date"]))
    ].drop(columns=["list_date", "delist_date"])
    if prices.empty:
        raise ValueError("no listed price history")
    membership = point_in_time_tradable_universe(
        prices, basic, start_date=prices["date"].min().strftime("%Y-%m-%d"),
        end_date=str(day.date()), minimum_history=60, stock_st=stock_st,
    )
    factor_prices = _drop_ineligible_invalid_adjustment_rows(prices, membership)
    panel = calculate_daily_factor_panel(
        factor_prices, tradable_membership=membership,
        include_low_market_residual_volatility=True,
    )
    eligible = pd.MultiIndex.from_frame(membership[["date", "code"]])
    panel = panel.reindex(eligible).sort_index()
    profile = load_baseline()
    model = profile["model_id"]
    score = _apply_combination(panel, _one_network_price_only_low_volatility_spec())
    adv = _rolling_average_amount(prices).reindex(panel.index)
    panel[model] = score.where(adv.ge(30_000_000))
    snapshots = factor_candidate_snapshots(
        panel, membership, factor_column=model, snapshot_dates=[day],
        top_n=profile["right_top_n"], trading_top_n=profile["right_top_n"],
    )
    rows = snapshots[str(day.date())]
    if len(rows) < profile["right_top_n"]:
        raise ValueError(f"frozen Top50 lacks eligible factors/liquidity: {len(rows)}")
    # Match the frozen model's base candidate contract. Profile enrichment can
    # change execution gates and therefore requires a separate research model.
    return rows


def load_left_snapshot(directory, observation_date, captured_at):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("snapshots", []) if item.get("date") == observation_date]
    if len(matches) != 1:
        raise ValueError("left candidates require exactly one same-day manifest snapshot")
    item = matches[0]
    timestamp = pd.to_datetime(item.get("captured_at", manifest.get("captured_at")), utc=True, errors="coerce")
    if pd.isna(timestamp) or timestamp > pd.Timestamp(captured_at) or timestamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != observation_date:
        raise ValueError("left candidates lack an actual same-day capture timestamp")
    candidate_path = (directory / item["file"]).resolve()
    if not candidate_path.is_relative_to(directory) or sha256(candidate_path) != item.get("sha256"):
        raise ValueError("left snapshot path/hash mismatch")
    rows = pd.read_csv(candidate_path, dtype={"code": str}).to_dict("records")
    if any(str(row.get("candidate_source", "")).split("+")[0] != "value_model" for row in rows):
        raise ValueError("left input must contain value_model rows only")
    return rows, {"manifest_sha256": sha256(path), "snapshot_sha256": item["sha256"],
                  "captured_at": timestamp.isoformat()}


def paper_orders(result, observation_date):
    """Extract engine decisions without pretending the next open has occurred."""
    if "paper_order_intents" in result:
        return [dict(item) for item in result["paper_order_intents"]
                if item.get("observation_date") == observation_date]
    orders = []
    for row in result.get("entry_blocks", []):
        if row.get("date") != observation_date:
            continue
        if row.get("reason") != "standard_right_next_open_date_unavailable":
            continue
        orders.append({
            "observation_date": observation_date, "code": row["code"],
            "side": "buy", "signal": row.get("signal_type"),
            "planned_price": row.get("signal_trigger_raw_price"),
            "stop_price": row.get("signal_stop_raw_price"),
            "target_price": row.get("entry_target_raw_price"),
            "quantity": None, "status": "conditional_next_open_recheck",
            "signal_detected": True, "triggered": False, "filled": False, "research_only": True,
            "reason": "next_open_price_cash_capacity_and_limits_not_yet_known",
        })
    return orders


def replay_paper(price_frames, snapshots, phases, observation_date, *, corporate_actions, security_end_dates):
    if any(pd.Timestamp(date) > pd.Timestamp(observation_date) for date in snapshots):
        raise ValueError("future candidate snapshot")
    for frame in price_frames.values():
        if pd.to_datetime(frame["date"]).gt(pd.Timestamp(observation_date)).any():
            raise ValueError("future execution bar")
    parameters = dict(load_baseline()["engine_parameters"])
    parameters.update(requested_start=min(snapshots), end_date=observation_date, price_frames_prepared=False)
    result = run_portfolio_backtest(
        price_frames, snapshots, phases, corporate_actions=corporate_actions,
        security_end_dates=security_end_dates, **parameters,
    )
    result.update(research_only=True, strict_point_in_time=False, paper_only=True)
    return result


def _run_capture(*, database, sync_state, left_directory, formula_history,
                output_root, observation_date):
    """Use current captures and archived daily candidates for deterministic replay."""
    captured_at = capture_clock(observation_date)
    observation_date = pd.Timestamp(observation_date).strftime("%Y-%m-%d")
    state = json.loads(Path(sync_state).read_text(encoding="utf-8"))
    sync_hash = sha256(sync_state)
    validate_sync_receipt(state, database, observation_date, captured_at)
    repository = TushareRepository(Database(database))
    verify_completed_receipts(state, repository)
    config_hash = sha256(BASELINE_PATH)
    implementation = implementation_fingerprint()
    output = Path(output_root).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "version": "frozen-forward-v1", "model_id": load_baseline()["model_id"],
        "configuration_sha256": config_hash, "snapshots": [],
        "implementation": implementation,
        "research_only": True, "strict_point_in_time": False,
        "historical_backfill_is_forward_evidence": False,
    }
    if manifest["configuration_sha256"] != config_hash:
        raise ValueError("configuration changed; preregister a separate forward stream")
    if manifest.get("implementation") != implementation:
        raise ValueError("code/dependencies changed; preregister a separate forward stream")
    snapshots = {}
    phases = {}
    for entry in manifest["snapshots"]:
        if entry["date"] >= observation_date:
            raise ValueError("forward date already captured or out of order; immutable stream")
        path = (output / entry["file"]).resolve()
        if not path.is_relative_to(output) or sha256(path) != entry["sha256"]:
            raise ValueError("archived candidates were modified")
        snapshots[entry["date"]] = pd.read_csv(path, dtype={"code": str}).to_dict("records")
        phases[entry["date"]] = entry["formula33"]
    if snapshots:
        with repository.database.connect(read_only=True) as connection:
            calendar_rows = connection.execute(
                "SELECT trade_date FROM raw.tushare_dataset_rows WHERE dataset='trade_cal' "
                "AND trade_date>? AND trade_date<=? "
                "AND json_extract(payload_json, '$.exchange')='SSE' "
                "AND json_extract(payload_json, '$.is_open')=1",
                [min(snapshots), observation_date],
            ).fetchall()
        expected = {str(row[0]) for row in calendar_rows}
        actual = set(snapshots) - {min(snapshots)} | {observation_date}
        if not expected or expected != actual:
            raise ValueError(f"forward trading sessions missing or calendar incomplete: {sorted(expected ^ actual)}")
    formula_hash = sha256(formula_history)
    formula = pd.read_csv(formula_history)
    if "date" not in formula or "phase" not in formula:
        raise ValueError("Formula33 history needs date and phase")
    current = formula.loc[pd.to_datetime(formula["date"]).eq(pd.Timestamp(observation_date))]
    if len(current) != 1 or current.iloc[0]["phase"] not in {"active", "watch", "exited", "waiting"}:
        raise ValueError("exactly one valid same-day Formula33 state is required")
    phase = _safe(current.iloc[0].to_dict())
    phases[observation_date] = phase
    left, left_receipt = load_left_snapshot(left_directory, observation_date, captured_at)
    history_start = (pd.Timestamp(min([observation_date, *snapshots])) - pd.Timedelta(days=750)).strftime("%Y-%m-%d")
    prices = repository.load_market_daily_frame(start_date=history_start, end_date=observation_date)
    basic = repository.load_stock_basic_universe()
    st = repository.load_dataset_for_codes("stock_st", basic["code"], start_date=history_start, end_date=observation_date)
    right = build_right_snapshot(prices, basic, st, observation_date)
    candidates = merge_hybrid_rows(right, left, right_factor=load_baseline()["model_id"])
    snapshots[observation_date] = candidates
    codes = {row["code"] for rows in snapshots.values() for row in rows}
    frames = dict(_iter_price_frames(prices, codes=codes))
    if codes - set(frames):
        raise ValueError(f"paper portfolio lacks price history: {sorted(codes - set(frames))}")
    limits = repository.load_dataset_for_codes("stk_limit", codes, start_date=history_start, end_date=observation_date)
    if not {"date", "code", "up_limit", "down_limit"}.issubset(limits):
        raise ValueError("exact execution price limits are missing")
    for code, frame in frames.items():
        daily_limits = limits.loc[limits["code"].eq(code), ["date", "up_limit", "down_limit"]].drop_duplicates("date")
        frames[code] = frame.merge(daily_limits, on="date", how="left")
        execution = frames[code].loc[frames[code]["date"].ge(pd.Timestamp(min(snapshots)))]
        if execution[["up_limit", "down_limit"]].isna().any().any():
            raise ValueError(f"paper execution lacks exact limits: {code}")
    actions = repository.load_dividend_actions(codes, start_date=min(snapshots), end_date=observation_date)
    end_dates = {row.code: str(pd.Timestamp(row.delist_date).date()) for row in basic.itertuples()
                 if row.code in codes and pd.notna(row.delist_date)}
    result = replay_paper(frames, snapshots, phases, observation_date,
                          corporate_actions=actions, security_end_dates=end_dates)
    if manifest["snapshots"]:
        previous_entry = manifest["snapshots"][-1]
        previous_path = (output / previous_entry["file"]).parent / "paper_replay.json"
        if sha256(previous_path) != previous_entry.get("paper_replay_sha256"):
            raise ValueError("published paper replay was modified")
        verify_replay_prefix(json.loads(previous_path.read_text(encoding="utf-8")),
                             result, previous_entry["date"])
    orders = paper_orders(result, observation_date)
    orders_by_code = {row["code"]: row for row in orders}
    captured_at = capture_clock(observation_date)
    for row in candidates:
        order = orders_by_code.get(row["code"], {})
        row.update({"observation_date": observation_date, "date": observation_date,
                    "captured_at": captured_at, "research_only": True,
                    "rank": row.get("playbook_factor_rank"), "score": row.get("candidate_score"),
                    "signal": order.get("signal"), "signal_detected": bool(order), "triggered": False,
                    "planned_price": order.get("planned_price"), "stop_price": order.get("stop_price"),
                    "target_price": order.get("target_price")})
    # Abort if upstream files changed while the full-market calculation ran.
    _, final_left_receipt = load_left_snapshot(left_directory, observation_date, captured_at)
    if (sha256(formula_history) != formula_hash or sha256(BASELINE_PATH) != config_hash
            or sha256(sync_state) != sync_hash or final_left_receipt != left_receipt
            or implementation_fingerprint() != implementation):
        raise ValueError("input/configuration changed during capture")
    verify_completed_receipts(state, repository)
    day_root = output / observation_date
    day_root.mkdir(parents=True, exist_ok=True)
    candidate_path = day_root / "candidates.csv"
    pd.DataFrame(candidates).to_csv(candidate_path, index=False, encoding="utf-8-sig")
    write_json(day_root / "paper_orders.json", {"orders": orders, "paper_only": True,
               "observation_date": observation_date, "captured_at": captured_at,
               "order_scope": "right_close_and_preplaced_intents_and_queued_exits; left_grid_conditions_remain_in_engine; fills and blocks in paper_replay.json"})
    write_json(day_root / "paper_replay.json", result)
    entry = {"date": observation_date, "file": candidate_path.relative_to(output).as_posix(),
             "sha256": sha256(candidate_path), "candidate_count": len(candidates),
             "captured_at": captured_at, "formula33": phase,
             "formula_sha256": formula_hash, "left_receipt": left_receipt,
             "sync_state_sha256": sync_hash,
             "orders_sha256": sha256(day_root / "paper_orders.json"),
             "paper_replay_sha256": sha256(day_root / "paper_replay.json")}
    manifest["snapshots"].append(entry)
    write_json(manifest_path, manifest)
    return entry


def run_capture(**kwargs):
    output = Path(kwargs["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".capture.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"capture already running or interrupted; inspect lock: {lock}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return _run_capture(**kwargs)
    finally:
        lock.unlink()
