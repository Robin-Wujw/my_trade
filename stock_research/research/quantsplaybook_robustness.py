"""Run exact stock-exclusion robustness checks for a portfolio backtest."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence


SCHEMA_VERSION = "quantsplaybook-robustness-v1"
PORTFOLIO_MODULE = "apps.portfolio_backtest"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "robustness_manifest.json"
MATRIX_NAME = "robustness_matrix.csv"
FORBIDDEN_TEMPLATE_OPTIONS = (
    "--codes",
    "--exclude-codes",
    "--output-directory",
)
STRATEGY_CONFIGURATION_FIELDS = (
    "allow_pullback_pilot",
    "allow_structure_pullback",
    "auto_price_structure",
    "board_lot_policy",
    "candidate_mode",
    "candidate_pool_limit",
    "close_confirmed_execution",
    "commission_rate",
    "corporate_action_source",
    "entry_gate_model",
    "estimated_slippage_rate",
    "exit_tail_on_candidate_removal",
    "initial_capital",
    "left_grid_core_slots",
    "left_grid_initial_slots",
    "left_grid_max_exposure_pct",
    "left_grid_step_pct",
    "left_grid_unit_pct",
    "max_left_positions",
    "max_positions",
    "max_same_industry",
    "max_symbol_exposure_pct",
    "max_total_held_symbols",
    "min_entry_evidence_score",
    "minimum_commission",
    "max_entry_avg_amount_participation",
    "max_entry_signal_amount_participation",
    "pe_pb_left_max_exposure_pct",
    "pe_pb_left_max_positions",
    "price_source",
    "profit_tail_minimum_current_return_pct",
    "profit_tail_minimum_maximum_return_pct",
    "profit_tails_consume_capacity",
    "profit_tranches",
    "price_amount_multiplier",
    "factor_only_min_avg_amount",
    "right_entry_priority_mode",
    "right_entry_risk_budget_pct",
    "right_inverse_volatility_sizing_enabled",
    "right_entry_size_multiplier",
    "sell_stamp_duty_rate",
    "sell_stamp_duty_rate_before_2023_08_28",
    "sell_stamp_duty_reduction_date",
    "signals_effective_next_day",
)


class RobustnessError(RuntimeError):
    """Raised when an exact robustness run cannot be proven complete."""


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RobustnessError(f"{label} cannot be read as JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RobustnessError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    target = path.with_suffix(path.suffix + ".tmp")
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    target.replace(path)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RobustnessError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RobustnessError(f"{label} must be finite")
    return result


def _parse_date(value: Any, *, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RobustnessError(f"{label} must be an ISO date") from exc


def _date_sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _five_year_anniversary(value: date) -> date:
    try:
        return value.replace(year=value.year + 5)
    except ValueError:
        return value.replace(year=value.year + 5, day=28)


def _recorded_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RobustnessError(f"{label} is missing")
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def identify_positive_top_codes(
    summary: dict[str, Any],
    *,
    count: int = 3,
) -> list[dict[str, Any]]:
    concentration = summary.get("profit_concentration_summary")
    if not isinstance(concentration, dict):
        raise RobustnessError("baseline profit_concentration_summary is missing")
    symbols = concentration.get("top_symbols")
    if not isinstance(symbols, list):
        raise RobustnessError("baseline top_symbols must be a list")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_pnl = math.inf
    for index, row in enumerate(symbols):
        if not isinstance(row, dict):
            raise RobustnessError(f"baseline top_symbols[{index}] is invalid")
        code = str(row.get("code") or "").strip()
        pnl = _finite_float(
            row.get("total_pnl_amount"),
            label=f"baseline top_symbols[{index}].total_pnl_amount",
        )
        if pnl > previous_pnl + 0.01:
            raise RobustnessError("baseline top_symbols are not ordered by profit")
        previous_pnl = pnl
        if pnl <= 0:
            break
        if not code or code in seen:
            raise RobustnessError("baseline positive top symbols contain an invalid code")
        seen.add(code)
        selected.append({
            "code": code,
            "name": str(row.get("name") or code),
            "total_pnl_amount": pnl,
        })
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RobustnessError(
            f"baseline has only {len(selected)} positive unique symbols; {count} required"
        )

    declared_top1 = concentration.get("top1_symbol")
    if isinstance(declared_top1, dict):
        declared_code = str(declared_top1.get("code") or "").strip()
        if declared_code and declared_code != selected[0]["code"]:
            raise RobustnessError("baseline top1_symbol disagrees with top_symbols")
    return selected


def infer_equity_path(summary_path: str | Path) -> Path:
    path = Path(summary_path)
    suffix = "_summary.json"
    if not path.name.endswith(suffix):
        raise RobustnessError(
            f"summary filename must end with {suffix}: {path}"
        )
    return path.with_name(path.name[: -len(suffix)] + "_equity.csv")


def analyze_equity_curve(
    equity_path: str | Path,
    *,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> dict[str, Any]:
    path = Path(equity_path)
    if not path.is_file():
        raise RobustnessError(f"continuous equity CSV is missing: {path}")

    observations: list[tuple[date, float]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"date", "equity"}.issubset(
                reader.fieldnames
            ):
                raise RobustnessError(
                    f"equity CSV must contain date and equity columns: {path}"
                )
            previous_date: date | None = None
            for row_number, row in enumerate(reader, 2):
                current_date = _parse_date(
                    row.get("date"), label=f"equity row {row_number} date"
                )
                equity = _finite_float(
                    row.get("equity"), label=f"equity row {row_number} equity"
                )
                if equity < 0:
                    raise RobustnessError(
                        f"equity row {row_number} must be non-negative"
                    )
                if previous_date is not None and current_date <= previous_date:
                    raise RobustnessError(
                        "equity dates must be strictly increasing without duplicates"
                    )
                observations.append((current_date, equity))
                previous_date = current_date
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RobustnessError(f"continuous equity CSV cannot be read: {path}") from exc

    if not observations:
        raise RobustnessError("continuous equity CSV must contain observations")
    first_date, first_observed_equity = observations[0]
    initial_equity = 1.0
    last_date, final_equity = observations[-1]
    if expected_start and first_date != _parse_date(
        expected_start, label="expected equity start"
    ):
        raise RobustnessError(
            f"equity start differs from summary: {first_date} != {expected_start}"
        )
    if expected_end and last_date != _parse_date(
        expected_end, label="expected equity end"
    ):
        raise RobustnessError(
            f"equity end differs from summary: {last_date} != {expected_end}"
        )

    year_rows: dict[int, list[tuple[date, float]]] = {}
    peak = initial_equity
    maximum_drawdown = 0.0
    for current_date, equity in observations:
        year_rows.setdefault(current_date.year, []).append((current_date, equity))
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    years = sorted(year_rows)
    expected_years = list(range(first_date.year, last_date.year + 1))
    if years != expected_years:
        raise RobustnessError(
            f"continuous equity is missing calendar-year coverage: {years}"
        )

    annual_returns: list[dict[str, Any]] = []
    annual_multiples: list[float | None] = []
    period_base = initial_equity
    for year in years:
        rows = year_rows[year]
        period_end = rows[-1][1]
        annual_multiple = period_end / period_base if period_base else None
        return_pct = (annual_multiple - 1.0) * 100.0 if annual_multiple is not None else None
        annual_multiples.append(annual_multiple)
        annual_returns.append({
            "year": year,
            "start_date": rows[0][0].isoformat(),
            "end_date": rows[-1][0].isoformat(),
            "return_pct": None if return_pct is None else round(return_pct, 6),
        })
        period_base = period_end

    full_return = (final_equity / initial_equity - 1.0) * 100.0
    compounded = math.prod(value for value in annual_multiples if value is not None)
    if not math.isclose(
        compounded,
        final_equity / initial_equity,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RobustnessError("annual returns do not compound to continuous equity")
    best_index = max((index for index, value in enumerate(annual_multiples) if value is not None),
                     key=annual_multiples.__getitem__)
    best = annual_returns[best_index]
    without_best_multiple = None if any(value is None for value in annual_multiples) else math.prod(
        multiple
        for index, multiple in enumerate(annual_multiples)
        if index != best_index
    )
    observation_dates = [current_date for current_date, _ in observations]
    rolling_five_year_windows = []
    for start_index, (window_start, start_equity) in enumerate(observations):
        anniversary = _five_year_anniversary(window_start)
        if anniversary > last_date:
            break
        end_index = bisect_left(observation_dates, anniversary)
        if end_index <= start_index:
            continue
        window_end, end_equity = observations[end_index]
        multiple = end_equity / start_equity if start_equity else None
        rolling_five_year_windows.append({
            "start_date": window_start.isoformat(),
            "end_date": window_end.isoformat(),
            "equity_multiple": None if multiple is None else round(multiple, 9),
            "return_pct": None if multiple is None else round((multiple - 1.0) * 100.0, 6),
            "reached_three_times": None if multiple is None else multiple >= 3.0 - 1e-12,
        })
    rolling_multiples = [
        row["equity_multiple"] for row in rolling_five_year_windows if row["equity_multiple"] is not None
    ]
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "date_sequence_sha256": _date_sequence_sha256([
            current_date.isoformat() for current_date, _ in observations
        ]),
        "row_count": len(observations),
        "start_date": first_date.isoformat(),
        "end_date": last_date.isoformat(),
        "initial_equity": initial_equity,
        "first_observed_equity": first_observed_equity,
        "return_basis": "initial_capital_nav_1",
        "final_equity": final_equity,
        "equity_multiple": round(final_equity / initial_equity, 9),
        "full_period_return_pct": round(full_return, 6),
        "maximum_drawdown_pct": round(maximum_drawdown * 100.0, 6),
        "annual_returns": annual_returns,
        "best_year": int(best["year"]),
        "best_year_return_pct": float(best["return_pct"]),
        "without_best_year_compound_return_pct": (
            None if without_best_multiple is None else round((without_best_multiple - 1.0) * 100.0, 6)),
        "rolling_five_year_windows": rolling_five_year_windows,
        "rolling_five_year_window_count": len(rolling_five_year_windows),
        "rolling_five_year_three_times_count": sum(
            bool(row["reached_three_times"])
            for row in rolling_five_year_windows
        ),
        "rolling_five_year_minimum_multiple": (
            None if not rolling_multiples else min(rolling_multiples)
        ),
        "rolling_five_year_maximum_multiple": (
            None if not rolling_multiples else max(rolling_multiples)
        ),
        "rolling_five_year_latest_multiple": (
            None
            if not rolling_five_year_windows
            else rolling_five_year_windows[-1]["equity_multiple"]
        ),
    }


def _validate_equity_snapshot_dates(
    summary: dict[str, Any],
    equity: dict[str, Any],
    *,
    label: str,
) -> None:
    snapshot_dates = summary.get("candidate_snapshot_dates")
    if not isinstance(snapshot_dates, list) or not snapshot_dates:
        raise RobustnessError(f"{label} candidate_snapshot_dates is missing")
    normalized: list[str] = []
    previous: date | None = None
    for index, value in enumerate(snapshot_dates):
        current = _parse_date(
            value, label=f"{label} candidate_snapshot_dates[{index}]"
        )
        if previous is not None and current <= previous:
            raise RobustnessError(
                f"{label} candidate_snapshot_dates must be strictly increasing"
            )
        normalized.append(current.isoformat())
        previous = current
    if len(normalized) != equity["row_count"] or _date_sequence_sha256(
        normalized
    ) != equity["date_sequence_sha256"]:
        raise RobustnessError(
            f"{label} continuous equity dates differ from candidate snapshots"
        )


def _option_values(arguments: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            if index + 1 >= len(arguments):
                raise RobustnessError(f"portfolio template option has no value: {option}")
            values.append(arguments[index + 1])
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            values.append(token[len(prefix):])
        index += 1
    return values


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return option in arguments or any(
        token.startswith(option + "=") for token in arguments
    )


def _single_option(arguments: Sequence[str], option: str) -> str:
    values = _option_values(arguments, option)
    if len(values) != 1 or not str(values[0]).strip():
        raise RobustnessError(
            f"portfolio template must provide {option} exactly once"
        )
    return str(values[0]).strip()


def validate_portfolio_template(
    arguments: Sequence[str],
    *,
    baseline_summary: dict[str, Any],
    candidate_manifest_path: Path,
) -> list[str]:
    normalized = [str(item) for item in arguments]
    if not normalized:
        raise RobustnessError("portfolio argument template is empty")
    if "--" in normalized:
        raise RobustnessError("portfolio argument template contains an extra -- separator")
    for option in FORBIDDEN_TEMPLATE_OPTIONS:
        if option in normalized or any(
            token.startswith(option + "=") for token in normalized
        ):
            raise RobustnessError(
                f"portfolio argument template must not override {option}"
            )
    if not _has_option(normalized, "--no-refresh-inputs") or not _has_option(
        normalized, "--allow-frozen-inputs"
    ):
        raise RobustnessError(
            "exact counterfactuals require --no-refresh-inputs and --allow-frozen-inputs"
        )

    start_date = _single_option(normalized, "--start-date")
    end_date = _single_option(normalized, "--end-date")
    expected_start = str(baseline_summary.get("requested_start") or "").strip()
    expected_end = str(
        baseline_summary.get("requested_end_date")
        or baseline_summary.get("end_date")
        or ""
    ).strip()
    if start_date != expected_start or end_date != expected_end:
        raise RobustnessError(
            "portfolio template dates differ from the baseline summary: "
            f"template={start_date}..{end_date} baseline={expected_start}..{expected_end}"
        )

    candidate_directory = _recorded_path(
        _single_option(normalized, "--candidate-directory"),
        label="portfolio template candidate directory",
    )
    if candidate_directory != candidate_manifest_path.parent.resolve():
        raise RobustnessError(
            "portfolio template candidate directory differs from baseline fingerprint"
        )

    fingerprints = baseline_summary.get("input_fingerprints") or {}
    formula_path_value = fingerprints.get("formula_history")
    formula_hash = str(fingerprints.get("formula_history_sha256") or "").strip()
    if formula_path_value or formula_hash:
        if not formula_path_value or not formula_hash:
            raise RobustnessError("baseline formula-history fingerprint is incomplete")
        template_formula = _recorded_path(
            _single_option(normalized, "--formula-history"),
            label="portfolio template formula history",
        )
        recorded_formula = _recorded_path(
            formula_path_value, label="baseline formula history"
        )
        if template_formula != recorded_formula:
            raise RobustnessError(
                "portfolio template formula history differs from baseline fingerprint"
            )
        if not recorded_formula.is_file() or file_sha256(recorded_formula) != formula_hash:
            raise RobustnessError("baseline formula-history SHA256 no longer matches")
    return normalized


def _validate_completed_summary(summary: dict[str, Any], *, label: str) -> None:
    if summary.get("coverage_complete") is not True:
        raise RobustnessError(f"{label} does not declare coverage_complete=true")
    missing = summary.get("missing_candidate_trade_dates")
    if missing not in (None, []):
        raise RobustnessError(f"{label} has missing candidate trade dates")
    for field in ("actual_start", "end_date", "final_return_pct"):
        if field not in summary:
            raise RobustnessError(f"{label} is missing {field}")


def validate_strict_candidate_manifest(
    path: str | Path,
    *,
    allow_research_inputs: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = _read_json_object(manifest_path, label="candidate manifest")
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise RobustnessError("candidate manifest has no snapshots")
    if manifest.get("research_only") is True and not allow_research_inputs:
        raise RobustnessError("candidate manifest is research_only")
    if manifest.get("financial_point_in_time") is not True:
        raise RobustnessError("candidate manifest is not financial point-in-time")
    declared_financial_use = manifest.get("uses_financial_data")
    uses_financial_data = (
        manifest.get("strict_financial_point_in_time") is not None
        if declared_financial_use is None
        else declared_financial_use is True
    )
    if uses_financial_data and manifest.get("strict_financial_point_in_time") is not True:
        raise RobustnessError("candidate manifest is not strict financial point-in-time")
    if manifest.get("industry_point_in_time") is not True:
        raise RobustnessError("candidate manifest is not industry point-in-time")
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict):
            raise RobustnessError(f"candidate snapshot {index} is invalid")
        if snapshot.get("financial_point_in_time") is not True:
            raise RobustnessError(
                f"candidate snapshot {index} is not financial point-in-time"
            )
        if snapshot.get("industry_point_in_time") is not True:
            raise RobustnessError(
                f"candidate snapshot {index} is not industry point-in-time"
            )
    for lane_name in ("right_lane", "left_lane", "left_extra_lane"):
        lane = manifest.get(lane_name)
        if not isinstance(lane, dict):
            continue
        directory_value = str(lane.get("candidate_directory") or "").strip()
        expected_sha = str(lane.get("manifest_sha256") or "").strip().lower()
        if not directory_value or len(expected_sha) != 64:
            raise RobustnessError(f"candidate {lane_name} fingerprint is incomplete")
        lane_manifest = _recorded_path(
            Path(directory_value) / "manifest.json",
            label=f"candidate {lane_name} manifest",
        )
        if not lane_manifest.is_file() or file_sha256(lane_manifest) != expected_sha:
            raise RobustnessError(f"candidate {lane_name} manifest SHA256 changed")
    return manifest


def load_baseline_context(
    summary_path: str | Path,
    *,
    allow_research_inputs: bool = False,
) -> dict[str, Any]:
    path = Path(summary_path).resolve()
    if not path.is_file():
        raise RobustnessError(f"baseline summary is missing: {path}")
    summary = _read_json_object(path, label="baseline summary")
    _validate_completed_summary(summary, label="baseline summary")
    if summary.get("excluded_codes"):
        raise RobustnessError("baseline summary must not already exclude stock codes")

    fingerprints = summary.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        raise RobustnessError("baseline input_fingerprints is missing")
    candidate_manifest = _recorded_path(
        fingerprints.get("candidate_manifest"),
        label="baseline candidate manifest",
    )
    expected_candidate_sha = str(
        fingerprints.get("candidate_manifest_sha256") or ""
    ).strip().lower()
    if len(expected_candidate_sha) != 64:
        raise RobustnessError("baseline candidate-manifest SHA256 is missing or invalid")
    if not candidate_manifest.is_file():
        raise RobustnessError(
            f"baseline candidate manifest is missing: {candidate_manifest}"
        )
    actual_candidate_sha = file_sha256(candidate_manifest)
    if actual_candidate_sha != expected_candidate_sha:
        raise RobustnessError(
            "baseline candidate manifest changed after the parent summary was written"
        )
    candidate_manifest_payload = validate_strict_candidate_manifest(
        candidate_manifest,
        allow_research_inputs=allow_research_inputs,
    )

    equity_path = infer_equity_path(path)
    equity = analyze_equity_curve(
        equity_path,
        expected_start=str(summary["actual_start"]),
        expected_end=str(summary["end_date"]),
    )
    _validate_equity_snapshot_dates(summary, equity, label="baseline summary")
    declared_return = _finite_float(
        summary.get("final_return_pct"), label="baseline final_return_pct"
    )
    if abs(equity["full_period_return_pct"] - declared_return) > 0.02:
        raise RobustnessError(
            "baseline summary return does not match its continuous equity CSV"
        )
    if "maximum_drawdown_pct" in summary:
        declared_drawdown = _finite_float(
            summary.get("maximum_drawdown_pct"),
            label="baseline maximum_drawdown_pct",
        )
        if abs(equity["maximum_drawdown_pct"] - declared_drawdown) > 0.02:
            raise RobustnessError(
                "baseline summary drawdown does not match its continuous equity CSV"
            )

    top_symbols = identify_positive_top_codes(summary, count=3)
    return {
        "summary": summary,
        "summary_path": path,
        "summary_sha256": file_sha256(path),
        "equity": equity,
        "candidate_manifest_path": candidate_manifest,
        "candidate_manifest_sha256": actual_candidate_sha,
        "candidate_manifest_research_only": (
            candidate_manifest_payload.get("research_only") is True
        ),
        "top_symbols": top_symbols,
    }


def build_counterfactual_command(
    *,
    python_executable: str,
    portfolio_arguments: Sequence[str],
    excluded_codes: Sequence[str],
    output_directory: str | Path,
) -> list[str]:
    codes = [str(code).strip() for code in excluded_codes if str(code).strip()]
    if not codes or len(codes) != len(set(codes)):
        raise RobustnessError("counterfactual excluded codes must be unique and non-empty")
    return [
        str(python_executable),
        "-m",
        PORTFOLIO_MODULE,
        *portfolio_arguments,
        "--exclude-codes",
        ",".join(codes),
        "--output-directory",
        str(Path(output_directory).resolve()),
    ]


def _find_single_summary(output_directory: Path) -> Path:
    matches = sorted(output_directory.glob("portfolio_*_summary.json"))
    if len(matches) != 1:
        raise RobustnessError(
            "counterfactual output must contain exactly one portfolio summary: "
            f"directory={output_directory} count={len(matches)}"
        )
    return matches[0].resolve()


def _verify_rerun_summary(
    summary: dict[str, Any],
    *,
    baseline_summary: dict[str, Any],
    excluded_codes: Sequence[str],
    candidate_manifest_sha256: str,
) -> None:
    _validate_completed_summary(summary, label="counterfactual summary")
    actual_excluded = [
        str(code).strip() for code in summary.get("excluded_codes") or []
    ]
    if actual_excluded != list(excluded_codes):
        raise RobustnessError(
            "counterfactual summary does not declare the exact requested exclusions"
        )
    for collection_name in ("trade_ledger", "final_positions"):
        rows = summary.get(collection_name)
        if not isinstance(rows, list):
            raise RobustnessError(
                f"counterfactual summary is missing {collection_name}"
            )
        leaked = sorted({
            str(row.get("code") or "").strip()
            for row in rows
            if isinstance(row, dict)
            and str(row.get("code") or "").strip() in excluded_codes
        })
        if leaked:
            raise RobustnessError(
                f"excluded codes still appear in {collection_name}: {leaked}"
            )

    baseline_fingerprints = baseline_summary.get("input_fingerprints") or {}
    rerun_fingerprints = summary.get("input_fingerprints")
    if not isinstance(rerun_fingerprints, dict):
        raise RobustnessError("counterfactual input_fingerprints is missing")
    if (
        str(rerun_fingerprints.get("candidate_manifest_sha256") or "").lower()
        != candidate_manifest_sha256
    ):
        raise RobustnessError("counterfactual candidate-manifest SHA256 changed")
    for key, expected in baseline_fingerprints.items():
        if not key.endswith("_sha256") or expected is None:
            continue
        if str(rerun_fingerprints.get(key) or "").lower() != str(expected).lower():
            raise RobustnessError(
                f"counterfactual input fingerprint differs from baseline: {key}"
            )

    for field in (
        "requested_start",
        "actual_start",
        "end_date",
        "requested_end_date",
        "effective_end_date",
        "candidate_snapshot_dates",
    ):
        if baseline_summary.get(field) != summary.get(field):
            raise RobustnessError(
                f"counterfactual summary differs from baseline on {field}"
            )
    def configuration_value(payload: dict[str, Any], field: str) -> Any:
        value = payload.get(field)
        if field == "price_source" and isinstance(value, dict):
            return {
                key: value.get(key)
                for key in (
                    "kline_directory", "database_enabled", "source",
                    "database_source", "mode", "daily_limit_source",
                )
            }
        if field == "corporate_action_source" and isinstance(value, dict):
            return value.get("source")
        return value

    for field in STRATEGY_CONFIGURATION_FIELDS:
        if (
            field in baseline_summary
            and configuration_value(baseline_summary, field)
            != configuration_value(summary, field)
        ):
            raise RobustnessError(
                f"counterfactual strategy configuration differs on {field}"
            )


def _scenario_metrics(
    *,
    scenario: str,
    summary_path: Path,
    summary: dict[str, Any],
    equity: dict[str, Any],
    excluded_codes: Sequence[str],
    return_code: int | None,
    exact_counterfactual: bool,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "status": "completed",
        "exact_stock_counterfactual": exact_counterfactual,
        "stock_counterfactual_method": (
            "full_portfolio_rerun_with_fixed_candidate_stream_exclusion"
            if exact_counterfactual
            else "parent_baseline"
        ),
        "candidate_selection_recomputed": False,
        "excluded_codes": list(excluded_codes),
        "return_code": return_code,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "equity": equity,
        "declared_final_return_pct": _finite_float(
            summary.get("final_return_pct"),
            label=f"{scenario} final_return_pct",
        ),
        "declared_maximum_drawdown_pct": _finite_float(
            summary.get("maximum_drawdown_pct"),
            label=f"{scenario} maximum_drawdown_pct",
        ),
    }


def _write_matrix(path: Path, scenarios: Sequence[dict[str, Any]]) -> None:
    years = sorted({
        int(row["year"])
        for scenario in scenarios
        for row in scenario["equity"]["annual_returns"]
    })
    fieldnames = [
        "scenario",
        "status",
        "exact_stock_counterfactual",
        "stock_counterfactual_method",
        "candidate_selection_recomputed",
        "excluded_codes",
        "excluded_code_count",
        "return_code",
        "summary_path",
        "summary_sha256",
        "equity_path",
        "equity_sha256",
        "equity_date_sequence_sha256",
        "equity_row_count",
        "full_period_return_pct",
        "equity_multiple",
        "maximum_drawdown_pct",
        "best_year",
        "best_year_return_pct",
        "without_best_year_compound_return_pct",
        "rolling_five_year_window_count",
        "rolling_five_year_three_times_count",
        "rolling_five_year_minimum_multiple",
        "rolling_five_year_maximum_multiple",
        "rolling_five_year_latest_multiple",
        *[f"annual_return_{year}_pct" for year in years],
    ]
    rows = []
    for scenario in scenarios:
        equity = scenario["equity"]
        annual = {
            int(row["year"]): row["return_pct"]
            for row in equity["annual_returns"]
        }
        row = {
            "scenario": scenario["scenario"],
            "status": scenario["status"],
            "exact_stock_counterfactual": scenario["exact_stock_counterfactual"],
            "stock_counterfactual_method": scenario["stock_counterfactual_method"],
            "candidate_selection_recomputed": scenario[
                "candidate_selection_recomputed"
            ],
            "excluded_codes": ",".join(scenario["excluded_codes"]),
            "excluded_code_count": len(scenario["excluded_codes"]),
            "return_code": (
                "" if scenario["return_code"] is None else scenario["return_code"]
            ),
            "summary_path": scenario["summary_path"],
            "summary_sha256": scenario["summary_sha256"],
            "equity_path": equity["path"],
            "equity_sha256": equity["sha256"],
            "equity_date_sequence_sha256": equity["date_sequence_sha256"],
            "equity_row_count": equity["row_count"],
            "full_period_return_pct": equity["full_period_return_pct"],
            "equity_multiple": equity["equity_multiple"],
            "maximum_drawdown_pct": equity["maximum_drawdown_pct"],
            "best_year": equity["best_year"],
            "best_year_return_pct": equity["best_year_return_pct"],
            "without_best_year_compound_return_pct": equity[
                "without_best_year_compound_return_pct"
            ],
            "rolling_five_year_window_count": equity[
                "rolling_five_year_window_count"
            ],
            "rolling_five_year_three_times_count": equity[
                "rolling_five_year_three_times_count"
            ],
            "rolling_five_year_minimum_multiple": equity[
                "rolling_five_year_minimum_multiple"
            ],
            "rolling_five_year_maximum_multiple": equity[
                "rolling_five_year_maximum_multiple"
            ],
            "rolling_five_year_latest_multiple": equity[
                "rolling_five_year_latest_multiple"
            ],
        }
        row.update({
            f"annual_return_{year}_pct": annual.get(year, "") for year in years
        })
        rows.append(row)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prepare_output_root(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise RobustnessError(
            f"robustness output directory must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_robustness(
    *,
    baseline_summary_path: str | Path,
    output_directory: str | Path,
    portfolio_arguments: Sequence[str],
    python_executable: str = sys.executable,
    runner_command: Sequence[str] | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
    allow_research_inputs: bool = False,
) -> dict[str, Any]:
    output_root = _prepare_output_root(output_directory)
    manifest_path = output_root / MANIFEST_NAME
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_command": list(runner_command or []),
        "working_directory": str(PROJECT_ROOT),
        "portfolio_module": PORTFOLIO_MODULE,
        "approximate_profit_subtraction_used": False,
        "research_inputs_allowed": bool(allow_research_inputs),
        "formal_oos_claim": False,
        "realized_outcome_selected_exclusions": True,
        "counterfactual_scope": (
            "full portfolio replay after filtering the frozen candidate stream; "
            "source-universe ranking is not recomputed or refilled"
        ),
        "runs": [],
    }
    _write_json(manifest_path, manifest)

    try:
        baseline = load_baseline_context(
            baseline_summary_path,
            allow_research_inputs=allow_research_inputs,
        )
        summary = baseline["summary"]
        template = validate_portfolio_template(
            portfolio_arguments,
            baseline_summary=summary,
            candidate_manifest_path=baseline["candidate_manifest_path"],
        )
        manifest.update({
            "parent_summary": {
                "path": str(baseline["summary_path"]),
                "sha256": baseline["summary_sha256"],
            },
            "parent_candidate_manifest": {
                "path": str(baseline["candidate_manifest_path"]),
                "sha256": baseline["candidate_manifest_sha256"],
            },
            "parent_equity": {
                "path": baseline["equity"]["path"],
                "sha256": baseline["equity"]["sha256"],
            },
            "portfolio_argument_template": template,
            "top_positive_symbols": baseline["top_symbols"],
            "candidate_manifest_research_only": baseline[
                "candidate_manifest_research_only"
            ],
            "annual_return_method": (
                "continuous_equity_year_end_over_prior_year_end; first year uses "
                "initial capital NAV 1.0"
            ),
            "without_best_year_method": (
                "compound all continuous-equity annual returns except the highest"
            ),
        })
        _write_json(manifest_path, manifest)

        scenarios = [
            _scenario_metrics(
                scenario="baseline",
                summary_path=baseline["summary_path"],
                summary=summary,
                equity=baseline["equity"],
                excluded_codes=[],
                return_code=None,
                exact_counterfactual=False,
            )
        ]
        top_codes = [row["code"] for row in baseline["top_symbols"]]
        for scenario_name, excluded_codes in (
            ("exclude_top1", top_codes[:1]),
            ("exclude_top3", top_codes[:3]),
        ):
            run_directory = output_root / scenario_name
            run_directory.mkdir()
            command = build_counterfactual_command(
                python_executable=python_executable,
                portfolio_arguments=template,
                excluded_codes=excluded_codes,
                output_directory=run_directory,
            )
            stdout_path = run_directory / "portfolio_backtest.stdout.log"
            stderr_path = run_directory / "portfolio_backtest.stderr.log"
            run_record: dict[str, Any] = {
                "scenario": scenario_name,
                "status": "running",
                "exact_stock_counterfactual": True,
                "excluded_codes": list(excluded_codes),
                "command": command,
                "command_line_windows": subprocess.list2cmdline(command),
                "return_code": None,
                "output_directory": str(run_directory),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
            try:
                completed = command_runner(
                    command,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except OSError as exc:
                run_record.update({"status": "failed", "error": str(exc)})
                _write_json(manifest_path, manifest)
                raise RobustnessError(
                    f"counterfactual process could not start: {scenario_name}"
                ) from exc
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            run_record["return_code"] = int(completed.returncode)
            run_record["stdout_sha256"] = file_sha256(stdout_path)
            run_record["stderr_sha256"] = file_sha256(stderr_path)
            if completed.returncode != 0:
                run_record["status"] = "failed"
                _write_json(manifest_path, manifest)
                raise RobustnessError(
                    f"counterfactual backtest failed: {scenario_name} "
                    f"return_code={completed.returncode}"
                )

            rerun_summary_path = _find_single_summary(run_directory)
            rerun_summary = _read_json_object(
                rerun_summary_path, label=f"{scenario_name} summary"
            )
            _verify_rerun_summary(
                rerun_summary,
                baseline_summary=summary,
                excluded_codes=excluded_codes,
                candidate_manifest_sha256=baseline[
                    "candidate_manifest_sha256"
                ],
            )
            rerun_equity_path = infer_equity_path(rerun_summary_path)
            rerun_equity = analyze_equity_curve(
                rerun_equity_path,
                expected_start=str(rerun_summary["actual_start"]),
                expected_end=str(rerun_summary["end_date"]),
            )
            _validate_equity_snapshot_dates(
                rerun_summary,
                rerun_equity,
                label=f"{scenario_name} summary",
            )
            declared_return = _finite_float(
                rerun_summary.get("final_return_pct"),
                label=f"{scenario_name} final_return_pct",
            )
            if abs(rerun_equity["full_period_return_pct"] - declared_return) > 0.02:
                raise RobustnessError(
                    f"{scenario_name} summary return differs from continuous equity"
                )
            declared_drawdown = _finite_float(
                rerun_summary.get("maximum_drawdown_pct"),
                label=f"{scenario_name} maximum_drawdown_pct",
            )
            if abs(rerun_equity["maximum_drawdown_pct"] - declared_drawdown) > 0.02:
                raise RobustnessError(
                    f"{scenario_name} summary drawdown differs from continuous equity"
                )

            metrics = _scenario_metrics(
                scenario=scenario_name,
                summary_path=rerun_summary_path,
                summary=rerun_summary,
                equity=rerun_equity,
                excluded_codes=excluded_codes,
                return_code=int(completed.returncode),
                exact_counterfactual=True,
            )
            scenarios.append(metrics)
            run_record.update({
                "status": "completed",
                "summary_path": metrics["summary_path"],
                "summary_sha256": metrics["summary_sha256"],
                "equity_path": rerun_equity["path"],
                "equity_sha256": rerun_equity["sha256"],
                "annual_returns": rerun_equity["annual_returns"],
                "best_year": rerun_equity["best_year"],
                "best_year_return_pct": rerun_equity["best_year_return_pct"],
                "without_best_year_compound_return_pct": rerun_equity[
                    "without_best_year_compound_return_pct"
                ],
            })
            _write_json(manifest_path, manifest)

        matrix_path = output_root / MATRIX_NAME
        _write_matrix(matrix_path, scenarios)
        manifest.update({
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "matrix": {
                "path": str(matrix_path),
                "sha256": file_sha256(matrix_path),
                "row_count": len(scenarios),
            },
            "scenarios": scenarios,
        })
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if manifest["runs"] and manifest["runs"][-1].get("status") == "running":
            manifest["runs"][-1].update({
                "status": "failed",
                "error": str(exc),
            })
        manifest.update({
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        _write_json(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact Top1/Top3 stock-exclusion reruns for apps.portfolio_backtest. "
            "Place the original portfolio arguments after --."
        )
    )
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--allow-research-inputs",
        action="store_true",
        help=(
            "allow research_only candidates while retaining PIT and fingerprint "
            "gates; output remains formal_oos_claim=false"
        ),
    )
    return parser


def parse_cli(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if "-h" in raw or "--help" in raw:
        parser.parse_args(raw)
    if "--" not in raw:
        parser.error("portfolio argument template must follow a -- separator")
    divider = raw.index("--")
    args = parser.parse_args(raw[:divider])
    portfolio_arguments = raw[divider + 1:]
    if not portfolio_arguments:
        parser.error("portfolio argument template is empty")
    return args, portfolio_arguments


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args, portfolio_arguments = parse_cli(raw)
    runner_command = [
        sys.executable,
        "-m",
        "apps.quantsplaybook_robustness",
        *raw,
    ]
    try:
        manifest = run_robustness(
            baseline_summary_path=args.baseline_summary,
            output_directory=args.output_directory,
            portfolio_arguments=portfolio_arguments,
            python_executable=args.python_executable,
            runner_command=runner_command,
            allow_research_inputs=args.allow_research_inputs,
        )
    except Exception as exc:
        print(f"[quantsplaybook_robustness] failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(Path(args.output_directory).resolve() / MANIFEST_NAME),
        "matrix": manifest["matrix"]["path"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
