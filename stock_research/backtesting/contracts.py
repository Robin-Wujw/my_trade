"""Coverage, provenance and point-in-time input validation."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
from stock_research.backtesting.inputs import file_sha256

DEFAULT_INDUSTRY_TARGET_COVERAGE = 0.95


def validate_daily_price_limit_coverage(
    price_frames,
    start_date,
    end_date,
    *,
    require_complete=False,
):
    """Audit exact daily price-limit coverage for tradable execution bars."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    required_count = 0
    covered_count = 0
    missing = []
    verified_count = 0
    for code, frame in price_frames.items():
        dates = pd.to_datetime(frame.get("date"), errors="coerce")
        in_range = dates.between(start, end)
        volume = pd.to_numeric(frame.get("volume", pd.Series(float("nan"), index=frame.index)), errors="coerce")
        tradable = in_range & volume.gt(0)
        if "tradestatus" in frame:
            trade_status = pd.to_numeric(frame["tradestatus"], errors="coerce")
            tradable &= trade_status.isna() | trade_status.gt(0)
        up_limit = pd.to_numeric(frame.get("up_limit", pd.Series(float("nan"), index=frame.index)), errors="coerce")
        down_limit = pd.to_numeric(frame.get("down_limit", pd.Series(float("nan"), index=frame.index)), errors="coerce")
        covered = (
            tradable
            & up_limit.gt(0)
            & down_limit.gt(0)
            & up_limit.gt(down_limit)
        )
        required_count += int(tradable.sum())
        covered_count += int(covered.sum())
        if "daily_limit_source" in frame:
            verified_count += int((covered & frame["daily_limit_source"].eq("local_tushare_stk_limit")).sum())
        if bool((tradable & ~covered).any()) and len(missing) < 10:
            for value in dates[tradable & ~covered].head(10 - len(missing)):
                missing.append(f"{code}@{pd.Timestamp(value):%Y-%m-%d}")
    missing_count = required_count - covered_count
    result = {
        "source": (
            "local_tushare_stk_limit"
            if required_count > 0 and verified_count == required_count
            else "mixed_or_unverified_limits_with_board_rule_fallback"
        ),
        "required_tradable_bar_count": required_count,
        "verified_tushare_bar_count": verified_count,
        "unverified_covered_bar_count": covered_count - verified_count,
        "covered_bar_count": covered_count,
        "missing_bar_count": missing_count,
        "coverage_ratio": (
            1.0 if required_count == 0 else covered_count / required_count
        ),
        "missing_sample": missing,
        "complete": missing_count == 0,
    }
    if require_complete and missing_count:
        raise RuntimeError(
            "strict daily price-limit coverage is incomplete: "
            f"missing={missing_count} sample={missing}"
        )
    return result



def validate_price_frame_coverage(
    price_frames,
    codes,
    start_date,
    end_date,
    *,
    code_start_dates=None,
    security_end_dates=None,
    security_suspension_dates=None,
    delist_bar_tolerance_days=14,
    trade_calendar=None,
    require_internal_complete=True,
):
    """Require loaded execution/valuation bars through the backtest endpoint."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    missing = []
    empty = []
    late_start = []
    early_end = []
    invalid = []
    first_by_code = {}
    last_by_code = {}
    for code in sorted({str(item) for item in codes if str(item).strip()}):
        frame = price_frames.get(code)
        if frame is None:
            missing.append(code)
            continue
        if frame.empty:
            empty.append(code)
            continue
        dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
        if dates.empty:
            invalid.append(code)
            continue
        in_range = dates[(dates >= start) & (dates <= end)]
        if in_range.empty:
            invalid.append(code)
            continue
        first_by_code[code] = in_range.min().normalize()
        last_by_code[code] = in_range.max().normalize()
    effective_start = min(first_by_code.values()) if first_by_code else start
    code_start_dates = code_start_dates or {}
    security_end_dates = security_end_dates or {}
    security_suspension_dates = security_suspension_dates or {}
    accepted_security_ends = {}
    accepted_suspensions = {}
    internal_missing = []
    internal_missing_count = 0
    market_dates = sorted({
        pd.Timestamp(value).normalize()
        for value in (
            trade_calendar
            if trade_calendar is not None
            else (
                item
                for frame in price_frames.values()
                if frame is not None and not frame.empty
                for item in pd.to_datetime(
                    frame.get("date"), errors="coerce",
                ).dropna()
            )
        )
        if pd.notna(pd.to_datetime(value, errors="coerce"))
        and start <= pd.Timestamp(value).normalize() <= end
    })
    required_end = (
        market_dates[-1]
        if trade_calendar is not None and market_dates
        else end
    )
    for code, first in first_by_code.items():
        last = last_by_code[code]
        required_start = pd.Timestamp(code_start_dates.get(code, effective_start)).normalize()
        observed_dates = set(pd.to_datetime(price_frames[code]["date"]).dt.normalize())
        missing_inside = {value for value in market_dates if max(first, required_start) <= value <= last}
        missing_inside -= observed_dates
        missing_inside -= {pd.Timestamp(value).normalize() for value in security_suspension_dates.get(code, set())}
        internal_missing_count += len(missing_inside)
        internal_missing.extend(f"{code}@{value:%Y-%m-%d}" for value in sorted(missing_inside)[:max(0, 20 - len(internal_missing))])
        if first > required_start:
            late_start.append(f"{code}:{first:%Y-%m-%d}>{required_start:%Y-%m-%d}")
        if last < required_end:
            security_end = pd.to_datetime(
                security_end_dates.get(code), errors="coerce",
            )
            confirmed_suspensions = {
                pd.Timestamp(value).normalize()
                for value in security_suspension_dates.get(code, set())
            }
            suspension_until_delisting = set()
            if pd.notna(security_end):
                suspension_until_delisting = {
                    value for value in market_dates
                    if last < value < security_end.normalize()
                }
            known_delisting = (
                pd.notna(security_end)
                and last <= security_end.normalize() <= end
                and (
                    (
                        security_end.normalize() - last
                    ).days <= int(delist_bar_tolerance_days)
                    or (
                        suspension_until_delisting
                        and suspension_until_delisting.issubset(
                            confirmed_suspensions,
                        )
                    )
                )
            )
            if known_delisting:
                accepted_security_ends[code] = security_end.normalize().strftime(
                    "%Y-%m-%d",
                )
            else:
                required_after_last = {
                    value for value in market_dates if last < value <= end
                }
                if (
                    required_after_last
                    and required_after_last.issubset(confirmed_suspensions)
                ):
                    accepted_suspensions[code] = {
                        "last_trade_date": last.strftime("%Y-%m-%d"),
                        "confirmed_dates": [
                            value.strftime("%Y-%m-%d")
                            for value in sorted(required_after_last)
                        ],
                    }
                else:
                    early_end.append(f"{code}:{last:%Y-%m-%d}")
    problems = {
        "missing": missing,
        "empty": empty,
        "invalid": invalid,
        "late_start": late_start,
        "early_end": early_end,
    }
    if require_internal_complete and internal_missing:
        problems["unexplained_internal_gaps"] = internal_missing
    failed = {key: value for key, value in problems.items() if value}
    if failed:
        preview = {key: value[:10] for key, value in failed.items()}
        raise RuntimeError(
            "price K-line frames do not cover all backtest candidate codes "
            f"through {required_end:%Y-%m-%d}: {preview}"
        )
    return {
        "code_count": len({str(item) for item in codes if str(item).strip()}),
        "start_date": effective_start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "required_end_date": required_end.strftime("%Y-%m-%d"),
        "accepted_security_end_count": len(accepted_security_ends),
        "accepted_security_ends": accepted_security_ends,
        "internal_missing_bar_count": internal_missing_count,
        "internal_missing_sample": internal_missing,
        "independent_trade_calendar": trade_calendar is not None,
        "accepted_suspension_count": len(accepted_suspensions),
        "accepted_suspensions": accepted_suspensions,
    }



def candidate_manifest_empty_dates(candidate_directory):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    rows = manifest.get("snapshots", [])
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("date"))
        for row in rows
        if int(row.get("candidate_count") or 0) <= 0
    ]



def candidate_manifest_financial_status(candidate_directory):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "available": False,
            "financial_point_in_time": None,
            "unsafe_dates": [],
            "path": str(path),
        }
    rows = manifest.get("snapshots", [])
    unsafe_dates = []
    if isinstance(rows, list):
        unsafe_dates = [
            str(row.get("date"))
            for row in rows
            if row.get("financial_point_in_time") is not True
        ]
    strict_financial = manifest.get("strict_financial_point_in_time")
    declared_uses_financial = manifest.get("uses_financial_data")
    if declared_uses_financial is not None and not isinstance(declared_uses_financial, bool):
        raise ValueError("uses_financial_data must be a JSON boolean")
    # Older price-only factor manifests predate uses_financial_data.  A
    # manifest that declares strict financial status is necessarily using
    # financial inputs; otherwise preserve price-only compatibility.
    uses_financial = (
        strict_financial is not None
        if declared_uses_financial is None
        else declared_uses_financial
    )
    return {
        "available": True,
        "financial_point_in_time": manifest.get("financial_point_in_time"),
        "strict_financial_point_in_time": strict_financial,
        "uses_financial_data": uses_financial,
        "unsafe_dates": unsafe_dates,
        "path": str(path),
    }



def validate_candidate_manifest_financial_point_in_time(
    candidate_directory,
    *,
    allow_unsafe_financial=False,
):
    status = candidate_manifest_financial_status(candidate_directory)
    if allow_unsafe_financial:
        return status
    if not status["available"]:
        raise RuntimeError(
            "candidate manifest is missing or unreadable; "
            "use --allow-unsafe-financial only for research backtests. "
            f"manifest={status['path']}"
        )
    if (
        status["financial_point_in_time"] is not True
        or (
            status["uses_financial_data"] is True
            and status["strict_financial_point_in_time"] is not True
        )
        or status["unsafe_dates"]
    ):
        preview = ", ".join(status["unsafe_dates"][:5])
        raise RuntimeError(
            "candidate manifest is not strict financial point-in-time; "
            "use --allow-unsafe-financial only for research backtests. "
            f"manifest={status['path']} unsafe_dates={preview}"
        )
    return status



def candidate_manifest_industry_status(candidate_directory):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "available": False,
            "industry_point_in_time": None,
            "value_industry_rule_version": None,
            "unsafe_dates": [],
            "metadata_errors": ["manifest_unavailable"],
            "path": str(path),
        }
    rows = manifest.get("snapshots", [])
    unsafe_dates = []
    if isinstance(rows, list):
        unsafe_dates = [
            str(row.get("date"))
            for row in rows
            if row.get("industry_point_in_time") is not True
        ]
    metadata_errors = []
    source_path = manifest.get("industry_source_path")
    source_sha256 = str(manifest.get("industry_source_sha256") or "")
    if not source_path or not Path(source_path).is_file():
        metadata_errors.append("industry_source_missing")
    elif not source_sha256 or file_sha256(source_path) != source_sha256:
        metadata_errors.append("industry_source_hash_mismatch")
    as_of_date = pd.to_datetime(manifest.get("industry_as_of_date"), errors="coerce")
    if pd.isna(as_of_date):
        metadata_errors.append("industry_as_of_date_missing")
    elif isinstance(rows, list):
        if any(
            pd.notna(pd.to_datetime(row.get("date"), errors="coerce"))
            and pd.Timestamp(row.get("date")).normalize() < as_of_date.normalize()
            for row in rows
        ):
            metadata_errors.append("industry_source_is_future_for_snapshot")
    if not str(manifest.get("industry_data_source") or "").strip():
        metadata_errors.append("industry_data_source_missing")
    if manifest.get("industry_point_in_time_status") != "safe":
        metadata_errors.append("industry_source_not_marked_safe")
    try:
        coverage = float(manifest.get("industry_coverage_ratio"))
    except (TypeError, ValueError):
        coverage = 0.0
    if coverage < DEFAULT_INDUSTRY_TARGET_COVERAGE:
        metadata_errors.append("industry_coverage_below_target")
    return {
        "available": True,
        "industry_point_in_time": manifest.get("industry_point_in_time"),
        "value_industry_rule_version": manifest.get("value_industry_rule_version"),
        "unsafe_dates": unsafe_dates,
        "metadata_errors": metadata_errors,
        "path": str(path),
    }



def validate_candidate_manifest_industry_point_in_time(
    candidate_directory,
    *,
    allow_unsafe_industry=False,
):
    status = candidate_manifest_industry_status(candidate_directory)
    if not status["available"]:
        raise RuntimeError(
            "candidate manifest is missing or unreadable; "
            "use --allow-unsafe-industry only for research backtests. "
            f"manifest={status['path']}"
        )
    if status["value_industry_rule_version"] != VALUE_INDUSTRY_RULE_VERSION:
        raise RuntimeError(
            "candidate manifest uses a missing or stale value-industry rule version; "
            f"expected={VALUE_INDUSTRY_RULE_VERSION} "
            f"actual={status['value_industry_rule_version']} manifest={status['path']}"
        )
    if allow_unsafe_industry:
        return status
    if (
        status["industry_point_in_time"] is not True
        or status["unsafe_dates"]
        or status["metadata_errors"]
    ):
        preview = ", ".join(status["unsafe_dates"][:5])
        raise RuntimeError(
            "candidate manifest is not strict industry point-in-time; "
            "use --allow-unsafe-industry only for research backtests. "
            f"manifest={status['path']} unsafe_dates={preview} "
            f"metadata_errors={status['metadata_errors']}"
        )
    return status



def validate_backtest_input_coverage(
    snapshots,
    formula,
    requested_start,
    requested_end,
    *,
    candidate_directory=None,
    allow_unsafe_financial=False,
    allow_unsafe_industry=False,
):
    """Fail closed when every trading day lacks a fresh non-empty selection."""
    if candidate_directory is not None:
        validate_candidate_manifest_financial_point_in_time(
            candidate_directory,
            allow_unsafe_financial=allow_unsafe_financial,
        )
        validate_candidate_manifest_industry_point_in_time(
            candidate_directory,
            allow_unsafe_industry=allow_unsafe_industry,
        )
    if not snapshots:
        raise RuntimeError("candidate snapshots are empty after input refresh")
    if formula.empty or "date" not in formula:
        raise RuntimeError("Formula33 phase history is empty after input refresh")
    formula_dates = pd.to_datetime(formula["date"], errors="coerce").dropna()
    if formula_dates.empty:
        raise RuntimeError("Formula33 phase history contains no valid dates")
    requested_start_date = pd.Timestamp(requested_start).normalize()
    requested_end_date = pd.Timestamp(requested_end).normalize()
    formula_trade_dates = {
        date.normalize()
        for date in formula_dates
        if requested_start_date <= date.normalize() <= requested_end_date
    }
    if not formula_trade_dates:
        raise RuntimeError("Formula33 phase history has no dates in requested backtest range")
    snapshot_trade_dates = {
        pd.Timestamp(value).normalize()
        for value in snapshots
        if requested_start_date <= pd.Timestamp(value).normalize() <= requested_end_date
    }
    missing_snapshot_dates = sorted(formula_trade_dates - snapshot_trade_dates)
    extra_snapshot_dates = sorted(snapshot_trade_dates - formula_trade_dates)
    if missing_snapshot_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in missing_snapshot_dates[:10])
        raise RuntimeError(
            "candidate snapshots do not cover every Formula33 trade date; "
            f"missing={preview}"
        )
    if extra_snapshot_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in extra_snapshot_dates[:10])
        raise RuntimeError(
            "candidate snapshots contain dates outside Formula33 trade calendar; "
            f"extra={preview}"
        )
    candidate_end = max(snapshot_trade_dates)
    formula_end = max(formula_trade_dates)
    if formula_end != candidate_end:
        raise RuntimeError(
            "backtest input dates disagree: "
            f"candidate_end={candidate_end:%Y-%m-%d} "
            f"formula_end={formula_end:%Y-%m-%d}"
        )
    formula_start = min(formula_trade_dates)
    if min(snapshot_trade_dates) != formula_start:
        raise RuntimeError(
            "candidate snapshot coverage must start on the first Formula33 trade date: "
            f"candidate_start={min(snapshot_trade_dates):%Y-%m-%d} "
            f"formula_start={formula_start:%Y-%m-%d}"
        )
    if candidate_end > requested_end_date:
        raise RuntimeError("backtest input coverage unexpectedly exceeds requested end")
    return candidate_end.strftime("%Y-%m-%d")

