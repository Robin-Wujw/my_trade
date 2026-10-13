"""Run the point-in-time candidate portfolio backtest."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from stock_research.core.financial_period import visible_report_periods
from stock_research.core.paths import PATHS
from stock_research.reporting.backtest_trade_report import (
    apply_stock_basic_display_names,
    build_readable_trade_frame,
    render_trade_report_markdown,
)
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.storage import Database, ResearchRepository, TushareRepository
from stock_research.market.miniqmt_data import load_miniqmt_price_frames
from stock_research.backtesting.engine import (
    prepare_portfolio_price_frames,
    run_portfolio_backtest,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


from stock_research.backtesting.inputs import (
    file_sha256,
    _attach_raw_adjustment_factor as _attach_raw_adjustment_factor,
    load_candidate_snapshots,
    load_price_frames,
    load_corporate_actions,
    load_security_end_dates,
    load_security_suspension_dates,
    first_candidate_dates,
)
from stock_research.backtesting.contracts import (
    validate_daily_price_limit_coverage,
    validate_price_frame_coverage,
    candidate_manifest_empty_dates,
    candidate_manifest_financial_status as candidate_manifest_financial_status,
    validate_candidate_manifest_financial_point_in_time,
    candidate_manifest_industry_status as candidate_manifest_industry_status,
    validate_candidate_manifest_industry_point_in_time,
    validate_backtest_input_coverage,
    DEFAULT_INDUSTRY_TARGET_COVERAGE as DEFAULT_INDUSTRY_TARGET_COVERAGE,
)

MIN_FINANCIAL_CACHE_FILES = 1_500
DEFAULT_FINANCIAL_TARGET_COVERAGE = 0.95
DEFAULT_FINANCIAL_CHUNK_SIZE = 10
DEFAULT_FINANCIAL_TIMEOUT = 60
DEFAULT_MINIQMT_KLINE_CHUNK_SIZE = 50
FORMULA33_STATE_ANCHOR_DATE = pd.Timestamp("2023-01-03")
POST_EXCLUSION_QUOTA_REASON = (
    "not_selected_for_trading: post_exclusion_recomputed_rank_not_in_observation_pool"
)
QUOTA_DIAGNOSTIC_PREFIXES = (
    "not_selected_for_trading: factor_rank_not_in_observation_pool",
    "not_selected_for_trading: daily_top10_quota_or_core_reservation",
    POST_EXCLUSION_QUOTA_REASON,
)


def qualify_backtest_result(
    result,
    *,
    candidate_manifest=None,
    args=None,
    price_coverage=None,
    daily_limit_coverage=None,
):
    """Propagate research/data qualifications before persistence and reporting.

    A clean execution profile does not make a result production-grade when an
    upstream candidate manifest is research-only or when historical inputs are
    incomplete.  Keep the qualification on the result itself so the database,
    report, and JSON summary receive the same status.
    """
    manifest = candidate_manifest if isinstance(candidate_manifest, dict) else {}
    reasons = []
    data_limitations = list(result.get("data_limitations") or [])
    research_only = bool(result.get("research_only"))
    upstream_failed = False
    sources = [manifest]
    for key in ("factor_metadata", "right_lane", "left_lane", "left_extra_lane"):
        value = manifest.get(key)
        if isinstance(value, dict):
            sources.append(value)
    for source in sources:
        if source.get("research_only") is True:
            research_only = True
            reasons.append("upstream_research_only")
        for key in ("research_only_reason", "point_in_time_reason", "financial_point_in_time_reason"):
            value = source.get(key)
            if value:
                research_only = True
                reasons.append(str(value))
        if source.get("selected_before_oos") is False:
            research_only = True
            reasons.append("selected_before_oos_false")
        if str(source.get("acceptance_status") or "").lower() in {
            "research_only", "data_limited", "failed",
        }:
            research_only = True
            status = str(source["acceptance_status"]).lower()
            reasons.append(f"upstream_acceptance_{status}")
            upstream_failed = upstream_failed or status == "failed"
            if status == "data_limited":
                data_limitations.append("upstream_data_limited")
        if isinstance(source.get("data_limitations"), (list, tuple, set)):
            data_limitations.extend(str(item) for item in source["data_limitations"])

    # A point-in-time candidate flag alone does not prove that financial
    # revision vintages were captured.  Require an explicit revision-vintage
    # evidence field before allowing a non-research qualification.
    strict_revision_evidence = (
        manifest.get("strict_point_in_time") is True
        and manifest.get("revision_vintage_verified") is True
    )
    if not strict_revision_evidence:
        research_only = True
        reasons.append("strict_point_in_time_not_proven")

    if args is not None:
        if getattr(args, "close_confirmed_execution", None) == "close_proxy":
            research_only = True
            reasons.append("close_proxy_execution")
        if getattr(args, "allow_frozen_inputs", False):
            research_only = True
            reasons.append("frozen_inputs")
        if getattr(args, "allow_unsafe_financial", False):
            research_only = True
            reasons.append("unsafe_financial_override")
        if getattr(args, "allow_unsafe_industry", False):
            research_only = True
            reasons.append("unsafe_industry_override")

    coverage = price_coverage or {}
    if coverage.get("internal_missing_bar_count", 0):
        data_limitations.append("unexplained_internal_price_gaps")
    limits = daily_limit_coverage or {}
    if limits and not limits.get("complete", True):
        data_limitations.append("exact_price_limits_incomplete")
    if result.get("corporate_action_accounting_status") not in {None, "complete"}:
        data_limitations.append("corporate_action_accounting_limited")
    if result.get("corporate_action_limitations"):
        data_limitations.append("corporate_action_pay_date_or_revision_limited")
    data_limitations = sorted(set(str(item) for item in data_limitations if item))
    if data_limitations:
        research_only = True
        reasons.extend(data_limitations)

    result["research_only"] = research_only
    # The local financial inputs do not carry complete revision vintage
    # evidence, so ordinary replays must not claim strict PIT by inference.
    result["strict_point_in_time"] = False
    result["data_limitations"] = data_limitations
    result["qualification"] = {
        "research_only": research_only,
        "data_limited": bool(data_limitations),
        "reasons": sorted(set(reasons)),
        "data_limitations": data_limitations,
    }
    result["acceptance_status"] = (
        "failed" if upstream_failed else
        "data_limited" if data_limitations else
        "research_only" if research_only else
        "passed"
    )
    return result


def log_refresh_step(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[portfolio_backtest][refresh][{timestamp}] {message}", flush=True)


def refresh_price_cache_directory(price_source):
    if str(price_source).startswith("miniqmt"):
        return formula33_price_cache_directory("miniqmt")
    return PATHS.cache / "formula33_kline" / "akshare"


def formula33_price_cache_directory(price_source):
    return PATHS.cache / "formula33_kline" / str(price_source)


def refresh_raw_price_cache_directory(price_source):
    if str(price_source).startswith("miniqmt"):
        return PATHS.cache / "miniqmt_kline" / "1d" / "none"
    return PATHS.cache / "formula33_kline" / "akshare_raw"


def candidate_history_price_source(price_source):
    return "miniqmt" if str(price_source).startswith("miniqmt") else "akshare"


def infer_price_frame_source(kline_directory):
    parts = [str(part).lower() for part in Path(kline_directory).parts]
    if "miniqmt_kline" in parts:
        return "miniqmt"
    if len(parts) >= 2 and parts[-2:] == ["formula33_kline", "miniqmt"]:
        return "miniqmt"
    return "akshare"


def _chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _truthy_flag(value):
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _failure_parts(value):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split(";")
        if item and item.strip() and item.strip().lower() not in {"nan", "none"}
    ]


def _is_quota_diagnostic_reason(reason):
    return any(str(reason).startswith(prefix) for prefix in QUOTA_DIAGNOSTIC_PREFIXES)


def _quota_only_failure(candidate):
    parts = _failure_parts(candidate.get("candidate_failure_reason"))
    return bool(parts) and all(_is_quota_diagnostic_reason(part) for part in parts)


def _candidate_can_reenter_after_exclusion(candidate):
    if _truthy_flag(candidate.get("selected_for_trading")) and _truthy_flag(
        candidate.get("signal_eligible")
    ):
        return True
    if not _quota_only_failure(candidate):
        return False
    if not _truthy_flag(candidate.get("allow_right")):
        return False
    for field, floor in (
        ("quality_score", 70.0),
        ("earnings_yoy", 0.10),
        ("mktcap", 100.0),
    ):
        value = pd.to_numeric(candidate.get(field), errors="coerce")
        if pd.isna(value) or float(value) < floor:
            return False
    return True


def _candidate_sort_score(candidate):
    value = pd.to_numeric(candidate.get("candidate_score"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def apply_candidate_exclusions(snapshots, exclude_codes):
    """Remove codes and rerank the remaining daily executable candidate pool."""
    excluded = {str(code).strip() for code in exclude_codes if str(code).strip()}
    if not excluded:
        return snapshots
    reranked = {}
    for date, rows in snapshots.items():
        source_rows = [dict(row) for row in rows]
        target_count = sum(
            1
            for row in source_rows
            if _truthy_flag(row.get("selected_for_trading"))
            and _truthy_flag(row.get("signal_eligible"))
        )
        remaining = [
            row for row in source_rows
            if str(row.get("code") or "").strip() not in excluded
        ]
        executable = [
            row for row in remaining
            if _candidate_can_reenter_after_exclusion(row)
        ]
        executable.sort(
            key=lambda row: (
                -_candidate_sort_score(row),
                str(row.get("code") or ""),
            )
        )
        selected_codes = {
            str(row.get("code") or "").strip()
            for row in executable[:target_count]
        }
        rank_by_code = {
            str(row.get("code") or "").strip(): rank
            for rank, row in enumerate(executable[:target_count], 1)
        }
        normalized_rows = []
        for row in remaining:
            code = str(row.get("code") or "").strip()
            if code in selected_codes:
                row["selected_for_trading"] = True
                row["signal_eligible"] = True
                row["selection_rank"] = rank_by_code[code]
                non_quota_reasons = [
                    part for part in _failure_parts(row.get("candidate_failure_reason"))
                    if not _is_quota_diagnostic_reason(part)
                ]
                row["candidate_failure_reason"] = ";".join(non_quota_reasons)
            elif _candidate_can_reenter_after_exclusion(row):
                row["selected_for_trading"] = False
                row["signal_eligible"] = False
                row["selection_rank"] = None
                non_quota_reasons = [
                    part for part in _failure_parts(row.get("candidate_failure_reason"))
                    if not _is_quota_diagnostic_reason(part)
                ]
                non_quota_reasons.append(POST_EXCLUSION_QUOTA_REASON)
                row["candidate_failure_reason"] = ";".join(non_quota_reasons)
            normalized_rows.append(row)
        reranked[date] = normalized_rows
    return reranked


def run_logged_refresh_step(name, func, argv):
    log_refresh_step(f"START {name}: {' '.join(map(str, argv))}")
    try:
        result = func(argv)
    except BaseException:
        log_refresh_step(f"FAILED {name}")
        raise
    if result not in (None, 0):
        log_refresh_step(f"FAILED {name}: exit={result}")
        raise RuntimeError(f"{name} failed with exit={result}")
    log_refresh_step(f"DONE {name}")
    return result


def _code_to_kline_cache_name(code):
    text = str(code).strip()
    if "." in text:
        market, symbol = text.split(".", 1)
    else:
        symbol = text.zfill(6)
        market = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{market}_{symbol}.csv"


def _latest_stock_basic_snapshot(snapshot_directory, end_date):
    target = pd.Timestamp(end_date).normalize()
    candidates = []
    for path in Path(snapshot_directory).glob("stock_basic_*.csv"):
        date = pd.to_datetime(path.stem.removeprefix("stock_basic_"), errors="coerce")
        if pd.notna(date) and date.normalize() <= target:
            candidates.append((date.normalize(), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _load_ipo_dates(stock_basic_path):
    if not stock_basic_path:
        return {}
    try:
        frame = pd.read_csv(stock_basic_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        return {}
    if not {"code", "ipoDate"}.issubset(frame.columns):
        return {}
    dates = pd.to_datetime(frame["ipoDate"], errors="coerce")
    return {
        str(code): date.normalize()
        for code, date in zip(frame["code"].astype(str), dates)
        if pd.notna(date)
    }


def summarize_kline_cache_coverage(
    kline_directory,
    universe_path,
    start_date,
    end_date,
    *,
    stock_basic_path=None,
    require_start_coverage=True,
):
    """Return a cheap full-universe K-line coverage summary for backtest inputs."""
    try:
        universe = pd.read_csv(universe_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        universe = pd.DataFrame()
    if universe.empty or "code" not in universe:
        return {
            "universe_count": 0,
            "missing_file_count": 0,
            "missing_start_count": 0,
            "missing_end_count": 0,
            "invalid_file_count": 0,
            "sample": [],
            "complete": False,
        }
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    summary = {
        "universe_count": int(universe["code"].dropna().nunique()),
        "missing_file_count": 0,
        "missing_start_count": 0,
        "post_ipo_start_count": 0,
        "missing_end_count": 0,
        "no_trade_end_count": 0,
        "invalid_file_count": 0,
        "sample": [],
        "incomplete_codes": [],
        "complete": True,
    }
    directory = Path(kline_directory)
    ipo_dates = _load_ipo_dates(stock_basic_path)
    observations = []
    for code in universe["code"].dropna().astype(str).drop_duplicates():
        path = directory / _code_to_kline_cache_name(code)
        if not path.is_file():
            observations.append((code, None, None, "missing_file"))
        else:
            try:
                dates = pd.read_csv(path, usecols=["date"], low_memory=False)
                parsed = pd.to_datetime(dates["date"], errors="coerce").dropna()
            except (OSError, ValueError, EmptyDataError):
                parsed = pd.Series(dtype="datetime64[ns]")
            if parsed.empty:
                observations.append((code, None, None, "invalid_file"))
            else:
                in_range = parsed[(parsed >= start) & (parsed <= end)]
                first = None if in_range.empty else in_range.min().normalize()
                last = None if in_range.empty else in_range.max().normalize()
                observations.append((code, first, last, ""))
    effective_start = min(
        first for _, first, _, reason in observations
        if not reason and first is not None
    ) if any(not reason and first is not None for _, first, _, reason in observations) else start
    for code, first, last, reason in observations:
        path = directory / _code_to_kline_cache_name(code)
        if reason == "missing_file":
            summary["missing_file_count"] += 1
        elif reason == "invalid_file" or first is None or last is None:
            summary["invalid_file_count"] += 1
            reason = "invalid_file"
        elif require_start_coverage and first > start and (ipo_date := ipo_dates.get(code)) is not None and ipo_date > start:
            summary["post_ipo_start_count"] += 1
            reason = ""
        elif require_start_coverage and first > effective_start:
            ipo_date = ipo_dates.get(code)
            if ipo_date is not None and ipo_date > effective_start:
                summary["post_ipo_start_count"] += 1
                reason = ""
            else:
                summary["missing_start_count"] += 1
                reason = f"starts_{first:%Y-%m-%d}"
        elif last < end:
            marker = path.with_name(path.name + ".no-trade.json")
            try:
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                marker_payload = {}
            marker_date = pd.to_datetime(
                marker_payload.get("observation_date"), errors="coerce",
            )
            if pd.notna(marker_date) and marker_date.normalize() == end:
                summary["no_trade_end_count"] += 1
                reason = ""
            else:
                summary["missing_end_count"] += 1
                reason = f"ends_{last:%Y-%m-%d}"
        if reason:
            summary["complete"] = False
            summary["incomplete_codes"].append(code)
            if len(summary["sample"]) < 10:
                summary["sample"].append(f"{code}:{reason}")
    summary["start_date"] = effective_start.strftime("%Y-%m-%d")
    return summary


def invalidate_formula33_manifest_if_kline_cache_incomplete(
    *,
    manifest_path,
    kline_directory,
    universe_path,
    start_date,
    end_date,
):
    stock_basic_path = _latest_stock_basic_snapshot(
        Path(universe_path).parent / "formula33_snapshots",
        end_date,
    )
    coverage = summarize_kline_cache_coverage(
        kline_directory,
        universe_path,
        start_date,
        end_date,
        stock_basic_path=stock_basic_path,
    )
    if coverage["complete"]:
        print(
            "[portfolio_backtest][preflight] K-line cache covers requested "
            f"range for {coverage['universe_count']} symbols",
            flush=True,
        )
        return coverage
    path = Path(manifest_path)
    if (
        path.exists()
        and coverage.get("missing_file_count", 0) > 0
        and coverage.get("missing_end_count", 0) == 0
        and coverage.get("invalid_file_count", 0) == 0
    ):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            manifest = {}
        manifest_date = str(manifest.get("observation_date") or "").strip()
        if manifest.get("status") == "completed" and manifest_date == str(end_date):
            print(
                "[portfolio_backtest][preflight] K-line cache has missing files "
                "or late-start historical files but a completed Formula33 "
                "manifest exists; keep manifest for resume/skip. "
                f"missing_file={coverage['missing_file_count']} "
                f"missing_start={coverage['missing_start_count']} "
                f"sample={coverage['sample']}",
                flush=True,
            )
            coverage["completed_manifest_reusable"] = True
            return coverage
    if path.exists():
        path.unlink()
    print(
        "[portfolio_backtest][preflight] K-line cache is incomplete; "
        "Formula33 manifest was invalidated so the refresh will auto-fetch "
        "missing bars. "
        f"missing_file={coverage['missing_file_count']} "
        f"missing_start={coverage['missing_start_count']} "
        f"missing_end={coverage['missing_end_count']} "
        f"invalid={coverage['invalid_file_count']} "
        f"sample={coverage['sample']}",
        flush=True,
    )
    return coverage


def ensure_miniqmt_kline_cache_for_backtest(
    *,
    kline_directory,
    universe_path,
    start_date,
    end_date,
    dividend_type,
    label,
    chunk_size=DEFAULT_MINIQMT_KLINE_CHUNK_SIZE,
):
    """Strictly refresh the MiniQMT execution/candidate K-line cache."""
    stock_basic_path = _latest_stock_basic_snapshot(
        Path(universe_path).parent / "formula33_snapshots",
        end_date,
    )
    coverage = summarize_kline_cache_coverage(
        kline_directory,
        universe_path,
        start_date,
        end_date,
        stock_basic_path=stock_basic_path,
        require_start_coverage=False,
    )
    if coverage["complete"]:
        print(
            "[portfolio_backtest][preflight] "
            f"{label} MiniQMT K-line cache ready "
            f"universe={coverage['universe_count']}",
            flush=True,
        )
        return coverage

    codes = sorted(set(coverage.get("incomplete_codes") or []))
    if not codes:
        raise RuntimeError(
            f"{label} MiniQMT K-line cache is incomplete but no codes were "
            f"reported: {coverage}"
        )
    log_refresh_step(
        f"auto-fetch {label} MiniQMT K-lines dividend={dividend_type} "
        f"codes={len(codes)} from {start_date} through {end_date} "
        f"sample={coverage.get('sample', [])}"
    )
    for index, chunk in enumerate(_chunks(codes, max(1, int(chunk_size))), start=1):
        _, summary = load_miniqmt_price_frames(
            chunk,
            start_date=start_date,
            end_date=end_date,
            period="1d",
            dividend_type=dividend_type,
            refresh=True,
            persist=True,
        )
        log_refresh_step(
            f"{label} MiniQMT K-line batch {index} "
            f"requested={summary.get('requested_count')} "
            f"loaded={summary.get('loaded_count')} "
            f"missing={summary.get('missing_count')} "
            f"errors={summary.get('fetch', {}).get('errors', [])[:2]}"
        )
    copied_markers = copy_formula33_no_trade_markers(
        codes,
        target_directory=kline_directory,
        observation_date=end_date,
    )
    if copied_markers:
        log_refresh_step(
            f"{label} copied Formula33 no-trade markers count={copied_markers}"
        )

    coverage = summarize_kline_cache_coverage(
        kline_directory,
        universe_path,
        start_date,
        end_date,
        stock_basic_path=stock_basic_path,
        require_start_coverage=False,
    )
    if not coverage["complete"]:
        raise RuntimeError(
            f"{label} MiniQMT K-line cache remains incomplete after auto-fetch: "
            f"missing_file={coverage['missing_file_count']} "
            f"missing_start={coverage['missing_start_count']} "
            f"missing_end={coverage['missing_end_count']} "
            f"invalid={coverage['invalid_file_count']} "
            f"sample={coverage['sample']}"
        )
    print(
        "[portfolio_backtest][preflight] "
        f"{label} MiniQMT K-line cache ready after auto-fetch "
        f"universe={coverage['universe_count']}",
        flush=True,
    )
    return coverage


def _miniqmt_period_and_dividend_type(kline_directory):
    path = Path(kline_directory)
    try:
        if path.parent.parent.name != "miniqmt_kline":
            return None
        return path.parent.name, path.name
    except IndexError:
        return None


def ensure_miniqmt_kline_cache_for_candidate_codes(
    codes,
    *,
    kline_directory,
    start_date,
    end_date,
    code_start_dates=None,
    label="candidate",
    chunk_size=DEFAULT_MINIQMT_KLINE_CHUNK_SIZE,
):
    """Refresh MiniQMT bars for the smaller actual candidate universe."""
    parsed = _miniqmt_period_and_dividend_type(kline_directory)
    if parsed is None:
        return {"skipped": True, "reason": "not_standard_miniqmt_directory"}
    period, dividend_type = parsed
    if period != "1d":
        return {"skipped": True, "reason": f"unsupported_period:{period}"}
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    directory = Path(kline_directory)
    code_start_dates = code_start_dates or {}
    incomplete = []
    for code in sorted({str(item) for item in codes if str(item).strip()}):
        required_start = pd.Timestamp(code_start_dates.get(code, start)).normalize()
        path = directory / _code_to_kline_cache_name(code)
        try:
            dates = pd.to_datetime(
                pd.read_csv(path, usecols=["date"], low_memory=False)["date"],
                errors="coerce",
            ).dropna()
        except (OSError, ValueError, EmptyDataError):
            incomplete.append(code)
            continue
        in_range = dates[(dates >= start) & (dates <= end)]
        if in_range.empty:
            incomplete.append(code)
            continue
        if in_range.min().normalize() > required_start or in_range.max().normalize() < end:
            incomplete.append(code)
    if not incomplete:
        print(
            "[portfolio_backtest][preflight] "
            f"{label} MiniQMT K-line cache covers actual candidates "
            f"codes={len(set(codes))}",
            flush=True,
        )
        return {
            "skipped": False,
            "requested_count": len(set(codes)),
            "refreshed_count": 0,
            "complete": True,
        }
    log_refresh_step(
        f"auto-fetch {label} MiniQMT K-lines for actual candidates "
        f"dividend={dividend_type} codes={len(incomplete)} "
        f"from {start:%Y-%m-%d} through {end:%Y-%m-%d} "
        f"sample={incomplete[:10]}"
    )
    for index, chunk in enumerate(_chunks(incomplete, max(1, int(chunk_size))), start=1):
        _, summary = load_miniqmt_price_frames(
            chunk,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            period=period,
            dividend_type=dividend_type,
            refresh=True,
            persist=True,
        )
        log_refresh_step(
            f"{label} MiniQMT candidate K-line batch {index} "
            f"requested={summary.get('requested_count')} "
            f"loaded={summary.get('loaded_count')} "
            f"missing={summary.get('missing_count')} "
            f"errors={summary.get('fetch', {}).get('errors', [])[:2]}"
        )
    remaining = []
    for code in incomplete:
        required_start = pd.Timestamp(code_start_dates.get(code, start)).normalize()
        path = directory / _code_to_kline_cache_name(code)
        try:
            dates = pd.to_datetime(
                pd.read_csv(path, usecols=["date"], low_memory=False)["date"],
                errors="coerce",
            ).dropna()
        except (OSError, ValueError, EmptyDataError):
            remaining.append(code)
            continue
        in_range = dates[(dates >= start) & (dates <= end)]
        if (
            in_range.empty
            or in_range.min().normalize() > required_start
            or in_range.max().normalize() < end
        ):
            remaining.append(code)
    if remaining:
        raise RuntimeError(
            f"{label} MiniQMT candidate K-line cache remains incomplete "
            f"after auto-fetch: count={len(remaining)} sample={remaining[:10]}"
        )
    print(
        "[portfolio_backtest][preflight] "
        f"{label} MiniQMT K-line cache ready for actual candidates "
        f"refreshed={len(incomplete)} total={len(set(codes))}",
        flush=True,
    )
    return {
        "skipped": False,
        "requested_count": len(set(codes)),
        "refreshed_count": len(incomplete),
        "complete": True,
    }


def copy_formula33_no_trade_markers(codes, *, target_directory, observation_date):
    source_directory = formula33_price_cache_directory("miniqmt")
    target_directory = Path(target_directory)
    copied = 0
    for code in sorted({str(item) for item in codes if str(item).strip()}):
        cache_name = _code_to_kline_cache_name(code)
        source_marker = Path(source_directory) / f"{cache_name}.no-trade.json"
        try:
            payload = json.loads(source_marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        marker_date = pd.to_datetime(payload.get("observation_date"), errors="coerce")
        if pd.isna(marker_date) or marker_date.strftime("%Y-%m-%d") != str(observation_date):
            continue
        target_marker = target_directory / f"{cache_name}.no-trade.json"
        target_marker.parent.mkdir(parents=True, exist_ok=True)
        target_marker.write_text(
            json.dumps({
                "code": code,
                "observation_date": str(observation_date),
                "source": "miniqmt",
                "version": 1,
                "copied_from": str(source_marker),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        copied += 1
    return copied


def repair_formula33_kline_metadata(source, universe_path):
    from stock_research.pipelines import formula33

    try:
        universe = pd.read_csv(universe_path, dtype={"code": str})
    except (OSError, ValueError, EmptyDataError):
        return {"checked": 0, "repaired": 0}
    checked = repaired = 0
    for code in universe.get("code", pd.Series(dtype=str)).dropna().astype(str).drop_duplicates():
        frame = formula33.load_cached_kline(source, code)
        if frame.empty:
            continue
        checked += 1
        if not formula33.kline_cache_metadata_matches(source, code, frame):
            formula33.save_kline_cache_metadata(source, code, frame)
            repaired += 1
    print(
        "[portfolio_backtest][preflight] Formula33 K-line metadata checked "
        f"{checked} symbols; repaired={repaired}",
        flush=True,
    )
    return {"checked": checked, "repaired": repaired}


def report_period_visible_date(report_period):
    period = pd.Timestamp(report_period).normalize()
    year = int(period.year)
    if period.month == 3 and period.day == 31:
        return pd.Timestamp(year=year, month=4, day=30)
    if period.month == 6 and period.day == 30:
        return pd.Timestamp(year=year, month=8, day=31)
    if period.month == 9 and period.day == 30:
        return pd.Timestamp(year=year, month=10, day=31)
    if period.month == 12 and period.day == 31:
        return pd.Timestamp(year=year + 1, month=4, day=30)
    return period


def financial_cache_file_count(directory, report_period):
    suffix = pd.Timestamp(report_period).strftime("%Y%m%d")
    return sum(1 for _path in Path(directory).glob(f"*_{suffix}.json"))


def load_financial_universe_codes(universe_path):
    from stock_research.market.miniqmt_financial import load_universe_codes

    return load_universe_codes(universe_path)


def strict_financial_cache_coverage(codes, report_period, as_of_date):
    from stock_research.market.miniqmt_financial import point_in_time_financial_cache_coverage

    return point_in_time_financial_cache_coverage(
        codes,
        report_period=report_period,
        as_of_date=as_of_date,
        output_directory=PATHS.cache / "q1_value",
    )


def financial_period_eligible_codes(codes, report_period, end_date):
    stock_basic_path = _latest_stock_basic_snapshot(
        PATHS.cache / "formula33_snapshots",
        end_date,
    )
    ipo_dates = _load_ipo_dates(stock_basic_path)
    if not ipo_dates:
        print(
            "[portfolio_backtest][preflight] stock basic snapshot missing IPO dates; "
            "financial coverage denominator falls back to full universe",
            flush=True,
        )
        return list(codes)
    report = pd.Timestamp(report_period).normalize()
    eligible = [
        code for code in codes
        if code in ipo_dates and ipo_dates[code] <= report
    ]
    print(
        "[portfolio_backtest][preflight] financial coverage eligible universe "
        f"period={pd.Timestamp(report_period).strftime('%Y-%m-%d')} "
        f"eligible={len(eligible)}/{len(codes)} "
        f"stock_basic={stock_basic_path}",
        flush=True,
    )
    return eligible


def supplement_strict_financial_cache(codes, report_period, as_of_date, args):
    from stock_research.market.miniqmt_financial import build_miniqmt_financial_cache

    return build_miniqmt_financial_cache(
        codes,
        report_period=report_period,
        as_of_date=as_of_date,
        output_directory=PATHS.cache / "q1_value",
        chunk_size=max(1, int(getattr(args, "financial_chunk_size", DEFAULT_FINANCIAL_CHUNK_SIZE))),
        timeout=max(1, int(getattr(args, "financial_timeout", DEFAULT_FINANCIAL_TIMEOUT))),
        missing_point_in_time_only=True,
    )


def require_strict_financial_cache_coverage(args, codes, report_period, as_of_date):
    target = float(getattr(args, "financial_target_coverage", DEFAULT_FINANCIAL_TARGET_COVERAGE))
    coverage = strict_financial_cache_coverage(codes, report_period, as_of_date)
    if coverage["coverage"] < target:
        print(
            "[portfolio_backtest][preflight] strict financial cache below target; "
            f"auto-fetch MiniQMT announce_time period={report_period} "
            f"as_of={as_of_date} coverage={coverage['coverage']:.1%} "
            f"target={target:.1%}",
            flush=True,
        )
        result = supplement_strict_financial_cache(codes, report_period, as_of_date, args)
        coverage = result.get("point_in_time_coverage") or strict_financial_cache_coverage(
            codes,
            report_period,
            as_of_date,
        )
        print(
            "[portfolio_backtest][preflight] MiniQMT financial auto-fetch result "
            f"requested={result.get('requested_count')} "
            f"saved={result.get('saved_count')} "
            f"skipped_existing={result.get('skipped_existing_count')} "
            f"skipped_no_metrics={result.get('skipped_no_metrics_count')} "
            f"failed_chunks={result.get('failed_chunks')} "
            f"errors={result.get('errors', [])[:3]} "
            f"post_coverage={coverage['complete_count']}/{coverage['requested_count']} "
            f"({coverage['coverage']:.1%})",
            flush=True,
        )
    if coverage["coverage"] < target:
        raise RuntimeError(
            "strict financial point-in-time cache remains below target after auto-fetch: "
            f"period={report_period} as_of={as_of_date} "
            f"coverage={coverage['complete_count']}/{coverage['requested_count']} "
            f"({coverage['coverage']:.1%}) target={target:.1%} "
            f"missing_or_unsafe={coverage['missing_or_unsafe_count']}"
        )
    print(
        "[portfolio_backtest][preflight] strict financial cache ready "
        f"period={report_period} as_of={as_of_date} "
        f"coverage={coverage['complete_count']}/{coverage['requested_count']} "
        f"({coverage['coverage']:.1%})",
        flush=True,
    )
    return coverage


def build_required_fundamental_snapshot(args, report_period, as_of_date):
    from stock_research.pipelines import fundamental_update

    target = float(getattr(args, "financial_target_coverage", DEFAULT_FINANCIAL_TARGET_COVERAGE))
    try:
        fundamental_update.main([
            "--report-period", report_period,
            "--as-of-date", as_of_date,
            "--max-updates", "0",
            "--skip-financial-updates",
            "--workers", "4",
            "--retries", "3",
            "--target-financial-coverage", str(target),
            "--require-target-financial-coverage",
        ])
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(
                "fundamental snapshot failed required coverage gate: "
                f"period={report_period} as_of={as_of_date} exit={exc.code}"
            ) from exc


def ensure_financial_cache_for_backtest(args):
    periods = visible_report_periods(args.start_date, args.end_date)
    if not periods:
        raise RuntimeError("no visible financial report periods found for backtest range")
    universe_codes = load_financial_universe_codes(PATHS.cache / "stock_universe.csv")
    if not universe_codes:
        raise RuntimeError("financial universe is empty; cannot prepare backtest fundamentals")
    for period in periods:
        as_of_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
        eligible_codes = financial_period_eligible_codes(universe_codes, period, args.end_date)
        require_strict_financial_cache_coverage(args, eligible_codes, period, as_of_date)
        build_required_fundamental_snapshot(args, period, as_of_date)


def candidate_manifest_reusable(
    candidate_directory,
    start_date,
    end_date,
    *,
    allow_unsafe_financial=False,
    allow_unsafe_industry=False,
):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if manifest.get("version") != SNAPSHOT_VERSION:
        return False
    requested_start = pd.to_datetime(manifest.get("requested_start"), errors="coerce")
    requested_end = pd.to_datetime(manifest.get("requested_end"), errors="coerce")
    if (
        pd.isna(requested_start)
        or requested_start.normalize() > pd.Timestamp(start_date).normalize()
        or pd.isna(requested_end)
        or requested_end.normalize() < pd.Timestamp(end_date).normalize()
    ):
        return False
    if not allow_unsafe_financial:
        status = validate_candidate_manifest_financial_point_in_time(
            candidate_directory,
            allow_unsafe_financial=False,
        )
        if status.get("unsafe_dates"):
            return False
    if not allow_unsafe_industry:
        status = validate_candidate_manifest_industry_point_in_time(
            candidate_directory,
            allow_unsafe_industry=False,
        )
        if status.get("unsafe_dates"):
            return False
    rows = manifest.get("snapshots", [])
    if not isinstance(rows, list) or not rows:
        return False
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dated_rows = [
        row for row in rows
        if pd.notna(pd.to_datetime(row.get("date"), errors="coerce"))
        and start <= pd.Timestamp(row.get("date")).normalize() <= end
    ]
    if not dated_rows:
        return False
    if any(int(row.get("candidate_count") or 0) <= 0 for row in dated_rows):
        return False
    return True


def default_data_end_date(now=None):
    """Return the latest session whose daily bar should already be available."""
    current = pd.Timestamp(now or pd.Timestamp.now())
    target = current.normalize()
    if current.weekday() >= 5:
        target = target - pd.offsets.BDay(1)
    elif current.hour < 16:
        target = target - pd.offsets.BDay(1)
    return target.strftime("%Y-%m-%d")


def formula33_refresh_window_args(start_date, end_date):
    """Size Formula33's calendar and history windows to cover a backtest range."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"invalid date range: {start_date}..{end_date}")
    warmup_start = (start - pd.offsets.BDay(45)).normalize()
    state_start = min(warmup_start, FORMULA33_STATE_ANCHOR_DATE)
    calendar_days = max(0, (end - state_start).days)
    # Formula33 uses lookback for output rows and history-days for both raw
    # calendar depth and per-symbol K-line depth.  Keep both large enough for
    # the requested range while leaving the model's own indicator warmup intact.
    return {
        "start_date": state_start.strftime("%Y-%m-%d"),
        "lookback": max(21, calendar_days + 30),
        "history_days": max(420, calendar_days + 180),
    }


def refresh_backtest_inputs(args):
    """Refresh source data first, then rebuild every derived backtest artifact."""
    from apps import (
        rebuild_candidate_history,
        rebuild_formula_history,
        rebuild_mainline_history,
    )
    from stock_research.pipelines import formula33
    from stock_research.core.completion_manifest import CompletionManifest

    refresh_price_source = getattr(args, "refresh_price_source", "akshare")
    refresh_metadata_source = getattr(args, "refresh_metadata_source", "akshare")
    refresh_market_cap_source = getattr(args, "refresh_market_cap_source", "auto")
    formula_kline_directory = formula33_price_cache_directory(refresh_price_source)
    candidate_price_source = candidate_history_price_source(refresh_price_source)
    candidate_kline_directory = refresh_price_cache_directory(candidate_price_source)
    candidate_raw_kline_directory = refresh_raw_price_cache_directory(candidate_price_source)
    formula_window = formula33_refresh_window_args(args.start_date, args.end_date)
    log_refresh_step(
        "preflight sources "
        f"price={refresh_price_source} "
        f"metadata={refresh_metadata_source} "
        f"market_cap={refresh_market_cap_source} "
        f"formula_start={formula_window['start_date']} "
        f"formula_kline_dir={formula_kline_directory} "
        f"candidate_kline_dir={candidate_kline_directory}"
    )
    kline_coverage = invalidate_formula33_manifest_if_kline_cache_incomplete(
        manifest_path=formula33.FORMULA33_MANIFEST_FILE,
        kline_directory=formula_kline_directory,
        universe_path=PATHS.cache / "stock_universe.csv",
        start_date=formula_window["start_date"],
        end_date=args.end_date,
    )
    if kline_coverage["complete"]:
        repair_formula33_kline_metadata(refresh_price_source, PATHS.cache / "stock_universe.csv")
    if kline_coverage.get("completed_manifest_reusable"):
        log_refresh_step("Formula33 completed manifest reusable; skip refresh")
    else:
        log_refresh_step(
            "first refresh full-market K-lines "
            f"from {formula_window['start_date']} through {args.end_date} "
            f"lookback={formula_window['lookback']} "
            f"history_days={formula_window['history_days']}"
        )
        run_logged_refresh_step("Formula33 refresh", formula33.main, [
            "--start-date", formula_window["start_date"],
            "--end-date", args.end_date,
            "--lookback", str(formula_window["lookback"]),
            "--history-days", str(formula_window["history_days"]),
            "--workers", "8",
            "--maxtasksperchild", "1000",
            "--retries", "3",
            "--retry-delay", "1",
            "--capital-workers", "1",
            "--require-end-trade",
            "--price-source", refresh_price_source,
            "--metadata-source", refresh_metadata_source,
            "--missing-mktcap-policy", "exclude",
            "--market-cap-source", refresh_market_cap_source,
        ])
    completion = CompletionManifest(formula33.FORMULA33_MANIFEST_FILE).read()
    observation_date = str(completion.get("observation_date") or "").strip()
    if completion.get("status") != "completed" or not observation_date:
        raise RuntimeError(
            "Formula33 refresh did not produce a completed observation-date manifest"
        )
    # The requested end can be a weekend/holiday.  All derived inputs must use
    # the market-confirmed observation date, not a guessed calendar date.
    args.end_date = pd.Timestamp(observation_date).strftime("%Y-%m-%d")
    log_refresh_step(f"market-confirmed effective end {args.end_date}")
    if candidate_price_source == "miniqmt":
        ensure_miniqmt_kline_cache_for_backtest(
            kline_directory=candidate_raw_kline_directory,
            universe_path=PATHS.cache / "stock_universe.csv",
            start_date=args.start_date,
            end_date=args.end_date,
            dividend_type="none",
            label="execution/raw",
        )
    ensure_financial_cache_for_backtest(args)

    # Candidate dates come from the refreshed K-line calendar.  Build once to
    # establish that calendar, rebuild matching dated mainline snapshots, then
    # build candidates again so they consume the new mainline data.
    log_refresh_step("rebuild candidate calendar")
    candidate_args = [
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output-directory", args.candidate_directory,
        "--price-source", candidate_price_source,
        "--kline-directory", str(candidate_kline_directory),
        "--raw-kline-directory", str(candidate_raw_kline_directory),
        "--industry-map-path", str(PATHS.cache / "reference" / "industry_map_latest.csv"),
    ]
    if candidate_manifest_reusable(
        args.candidate_directory,
        args.start_date,
        args.end_date,
        allow_unsafe_financial=getattr(args, "allow_unsafe_financial", False),
        allow_unsafe_industry=getattr(args, "allow_unsafe_industry", False),
    ):
        log_refresh_step("candidate history manifest reusable; skip rebuild")
        empty_candidate_dates = []
    else:
        run_logged_refresh_step("candidate history", rebuild_candidate_history.main, candidate_args)
        empty_candidate_dates = candidate_manifest_empty_dates(args.candidate_directory)
    if empty_candidate_dates:
        log_refresh_step(
            "candidate history has empty days; "
            f"rebuild mainline fallback count={len(empty_candidate_dates)} "
            f"sample={empty_candidate_dates[:10]}"
        )
        run_logged_refresh_step("mainline fallback history", rebuild_mainline_history.main, [
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--candidate-directory", args.candidate_directory,
        ])
        log_refresh_step("rebuild candidates with refreshed mainline")
        run_logged_refresh_step("candidate history after mainline", rebuild_candidate_history.main, candidate_args)
    else:
        log_refresh_step(
            "candidate history is non-empty for every trade day; "
            "skip slow mainline fallback rebuild"
        )
    log_refresh_step("rebuild Formula33 phase history")
    run_logged_refresh_step("Formula33 phase history", rebuild_formula_history.main, [
        "--start-date", formula_window["start_date"],
        "--end-date", args.end_date,
        "--output", args.formula_history,
        "--kline-directory", str(formula_kline_directory),
    ])


def run_job(args):
    if not args.price_kline_directory:
        args.price_kline_directory = str(refresh_price_cache_directory(args.refresh_price_source))
    if not args.end_date:
        args.end_date = default_data_end_date()
    requested_end_date = args.end_date
    if args.no_refresh_inputs and not args.allow_frozen_inputs:
        raise RuntimeError(
            "--no-refresh-inputs is research-only; production backtests must refresh "
            "and complete inputs first. Add --allow-frozen-inputs only for explicit "
            "frozen-cache diagnostics."
        )
    if args.no_refresh_inputs and not (
        args.allow_unsafe_financial and args.allow_unsafe_industry
    ):
        validate_candidate_manifest_financial_point_in_time(
            args.candidate_directory,
            allow_unsafe_financial=args.allow_unsafe_financial,
        )
        validate_candidate_manifest_industry_point_in_time(
            args.candidate_directory,
            allow_unsafe_industry=args.allow_unsafe_industry,
        )
    if not args.no_refresh_inputs:
        refresh_backtest_inputs(args)
    else:
        print(
            "[portfolio_backtest][WARNING] input refresh explicitly disabled; "
            "results use frozen local data by explicit --allow-frozen-inputs"
        )
    snapshots = load_candidate_snapshots(
        args.candidate_directory, args.start_date, args.end_date,
    )
    excluded_codes = [
        item.strip()
        for item in str(args.exclude_codes or "").split(",")
        if item.strip()
    ]
    if excluded_codes:
        snapshots = apply_candidate_exclusions(snapshots, excluded_codes)
    coverage_snapshots = dict(snapshots)
    explicit_items = [item.strip() for item in (args.codes or "").split(",") if item.strip()]
    explicit_pairs = [item.split("=", 1) for item in explicit_items]
    explicit_codes = [parts[0].strip() for parts in explicit_pairs]
    explicit_names = {
        parts[0].strip(): parts[1].strip()
        for parts in explicit_pairs if len(parts) == 2 and parts[1].strip()
    }
    if explicit_codes:
        first_date = min(snapshots) if snapshots else args.start_date
        snapshots = {
            first_date: [
                {
                    "code": code,
                    "name": explicit_names.get(code, code),
                    "strategy_part": "explicit candidate",
                }
                for code in explicit_codes
            ]
        }
    if args.candidate_mode == "fixed-first" and snapshots:
        first_date = min(snapshots)
        snapshots = {first_date: snapshots[first_date]}
    codes = {
        str(row["code"])
        for rows in snapshots.values()
        for row in rows
    }
    code_start_dates = first_candidate_dates(snapshots)
    security_end_dates = load_security_end_dates(codes)
    security_suspension_dates = load_security_suspension_dates(
        codes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    trade_plans = load_trade_plans(args.trade_plans)
    price_kline_directory = Path(args.price_kline_directory)
    if (
        not args.no_refresh_inputs
        and infer_price_frame_source(price_kline_directory) == "miniqmt"
    ):
        ensure_miniqmt_kline_cache_for_candidate_codes(
            codes,
            kline_directory=price_kline_directory,
            start_date=args.start_date,
            end_date=args.end_date,
            code_start_dates=code_start_dates,
            label="execution/valuation",
        )
    use_price_database = (
        not args.no_price_database
        and price_kline_directory.name != "akshare_raw"
    )
    price_frames = load_price_frames(
        codes,
        price_kline_directory,
        start_date=(pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
        end_date=args.end_date,
        source=infer_price_frame_source(price_kline_directory),
        prefer_database=use_price_database,
        database_source=args.price_database_source,
    )
    daily_limit_coverage = validate_daily_price_limit_coverage(
        price_frames,
        args.start_date,
        args.end_date,
        require_complete=args.require_daily_price_limits,
    )
    if args.price_frames_prepared:
        required_prepared_columns = {
            "date", "open", "high", "low", "close", "volume",
            "raw_open", "raw_high", "raw_low", "raw_close",
            "qfq_to_raw_factor", "ma5", "ma10", "ma20", "ma60", "ma120",
            "kd_divergence", "rsi_divergence", "wr_divergence",
            "bearish_divergence_count",
        }
        incomplete = {
            code: sorted(required_prepared_columns - set(frame.columns))
            for code, frame in price_frames.items()
            if not required_prepared_columns.issubset(frame.columns)
        }
        if incomplete:
            print(
                "[portfolio_backtest][WARNING] preparing only incomplete price "
                f"frames count={len(incomplete)}"
            )
            price_frames.update(prepare_portfolio_price_frames({
                code: price_frames[code] for code in incomplete
            }))
            still_incomplete = {
                code: sorted(required_prepared_columns - set(frame.columns))
                for code, frame in price_frames.items()
                if not required_prepared_columns.issubset(frame.columns)
            }
            if still_incomplete:
                sample = dict(list(still_incomplete.items())[:5])
                raise RuntimeError(
                    "prepared price frames still miss required columns: "
                    f"{sample}"
                )
    price_coverage = validate_price_frame_coverage(
        price_frames,
        codes,
        args.start_date,
        args.end_date,
        code_start_dates=code_start_dates,
        security_end_dates=security_end_dates,
        security_suspension_dates=security_suspension_dates,
    )
    formula = pd.read_csv(args.formula_history)
    input_coverage_end = validate_backtest_input_coverage(
        coverage_snapshots,
        formula,
        args.start_date,
        args.end_date,
        candidate_directory=args.candidate_directory,
        allow_unsafe_financial=args.allow_unsafe_financial,
        allow_unsafe_industry=args.allow_unsafe_industry,
    )
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
            "window_unique_count": (
                None
                if pd.isna(pd.to_numeric(
                    row.get("window_unique_count"), errors="coerce",
                ))
                else int(pd.to_numeric(
                    row.get("window_unique_count"), errors="coerce",
                ))
            ),
        }
        for _, row in formula.iterrows()
    }
    corporate_actions = load_corporate_actions(
        codes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    result = run_portfolio_backtest(
        price_frames,
        snapshots,
        phases,
        requested_start=args.start_date,
        end_date=args.end_date,
        trade_plans=trade_plans,
        max_daily_candidates=args.max_daily_candidates,
        max_positions=None if args.max_positions == 0 else args.max_positions,
        max_total_held_symbols=args.max_total_held_symbols,
        max_left_positions=args.max_left_positions,
        max_same_industry=args.max_same_industry,
        same_theme_correlation=args.same_theme_correlation,
        min_entry_evidence_score=args.min_entry_evidence_score,
        right_entry_priority_mode=args.right_entry_priority_mode,
        right_entry_size_multiplier=args.right_entry_size_multiplier,
        right_entry_risk_budget_pct=args.right_entry_risk_budget_pct,
        enable_right_inverse_volatility_sizing=(
            args.enable_right_inverse_volatility_sizing
        ),
        enable_right_market_volatility_sizing=(
            args.enable_right_market_volatility_sizing
        ),
        price_amount_multiplier=args.price_amount_multiplier,
        factor_only_min_avg_amount=args.factor_only_min_avg_amount,
        max_entry_signal_amount_participation=(
            args.max_entry_signal_amount_participation
        ),
        max_entry_avg_amount_participation=(
            args.max_entry_avg_amount_participation
        ),
        profit_tranches=args.profit_tranches,
        profit_tail_min_return=args.profit_tail_min_return,
        left_grid_unit=args.left_grid_unit,
        left_grid_step=args.left_grid_step,
        left_grid_max_exposure=args.left_grid_max_exposure,
        pe_pb_left_max_positions=args.pe_pb_left_max_positions,
        pe_pb_left_max_exposure=args.pe_pb_left_max_exposure,
        exit_tail_on_candidate_removal=args.exit_tail_on_candidate_removal,
        signals_effective_next_day=True,
        auto_price_structure=not args.no_auto_price_structure,
        allow_structure_pullback=not args.no_structure_pullback,
        allow_pullback_pilot=not args.disable_pullback_pilot,
        enable_breadth_reversal_half_pullback=(
            args.enable_breadth_reversal_channel
            or args.factor_only_breadth_reversal_only
        ),
        factor_only_breadth_reversal_only=(
            args.factor_only_breadth_reversal_only
        ),
        breadth_reversal_max_unique_count=(
            args.breadth_reversal_max_unique_count
        ),
        breadth_reversal_risk_budget=args.breadth_reversal_risk_budget,
        breadth_reversal_max_position=args.breadth_reversal_max_position,
        breadth_reversal_max_positions=args.breadth_reversal_max_positions,
        breadth_reversal_time_limit_days=(
            args.breadth_reversal_time_limit_days
        ),
        enable_factor_horizon_channel=args.enable_factor_horizon_channel,
        factor_horizon_risk_budget=args.factor_horizon_risk_budget,
        factor_horizon_stop_loss_pct=args.factor_horizon_stop_loss_pct,
        factor_horizon_max_position=args.factor_horizon_max_position,
        factor_horizon_max_positions=args.factor_horizon_max_positions,
        factor_horizon_holding_days=args.factor_horizon_holding_days,
        factor_horizon_trend_management_return=(
            args.factor_horizon_trend_management_return
        ),
        factor_horizon_max_return_60d=args.factor_horizon_max_return_60d,
        factor_horizon_require_ma20_not_rising=(
            args.factor_horizon_require_ma20_not_rising
        ),
        factor_horizon_exclude_watch_phase=(
            args.factor_horizon_exclude_watch_phase
        ),
        factor_horizon_min_avg_amount=args.factor_horizon_min_avg_amount,
        enable_compressed_structure_channel=(
            args.enable_compressed_structure_channel
        ),
        factor_only_compressed_structure_only=(
            args.factor_only_compressed_structure_only
        ),
        compressed_structure_risk_budget=(
            args.compressed_structure_risk_budget
        ),
        compressed_structure_stop_loss_pct=(
            args.compressed_structure_stop_loss_pct
        ),
        compressed_structure_max_position=(
            args.compressed_structure_max_position
        ),
        compressed_structure_max_positions=(
            args.compressed_structure_max_positions
        ),
        compressed_structure_holding_days=(
            args.compressed_structure_holding_days
        ),
        compressed_structure_min_entry_evidence_score=(
            args.compressed_structure_min_entry_evidence_score
        ),
        compressed_structure_min_proximity_score=(
            args.compressed_structure_min_proximity_score
        ),
        compressed_structure_min_avg_amount=(
            args.compressed_structure_min_avg_amount
        ),
        enable_intraday_right_protective_stop=(
            args.enable_intraday_right_protective_stop
        ),
        enable_diversified_factor_sleeve=(
            args.enable_diversified_factor_sleeve
        ),
        diversified_factor_risk_budget=args.diversified_factor_risk_budget,
        diversified_factor_max_position=(
            args.diversified_factor_max_position
        ),
        close_confirmed_execution=args.close_confirmed_execution,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        initial_capital=args.initial_capital,
        sell_stamp_duty_rate=args.sell_stamp_duty_rate,
        sell_stamp_duty_rate_before_2023_08_28=(
            args.sell_stamp_duty_rate_before_2023_08_28
        ),
        estimated_slippage_rate=args.estimated_slippage_rate,
        corporate_actions=corporate_actions,
        security_end_dates=security_end_dates,
        price_frames_prepared=args.price_frames_prepared,
    )
    result["price_source"] = {
        "kline_directory": str(price_kline_directory),
        "database_enabled": bool(use_price_database),
        "source": infer_price_frame_source(price_kline_directory),
        "database_source": args.price_database_source,
        "mode": "不复权CSV" if not use_price_database else "SQLite统一缓存优先",
        "coverage": price_coverage,
        "daily_limit_source": daily_limit_coverage["source"],
        "daily_limit_coverage": daily_limit_coverage,
        "daily_limit_coverage_required": bool(args.require_daily_price_limits),
    }
    result["corporate_action_source"] = {
        "source": "local_tushare_dividend",
        "loaded_symbol_count": len(corporate_actions),
        "loaded_event_count": sum(len(rows) for rows in corporate_actions.values()),
    }
    candidate_manifest_path = Path(args.candidate_directory) / "manifest.json"
    try:
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        candidate_manifest = {}
    qualify_backtest_result(
        result,
        candidate_manifest=candidate_manifest,
        args=args,
        price_coverage=price_coverage,
        daily_limit_coverage=daily_limit_coverage,
    )
    vectorbt_equity = []
    if args.vectorbt_cross_check:
        from stock_research.strategies.vectorbt_replay import run_vectorbt_cross_check

        cross_check = run_vectorbt_cross_check(
            price_frames,
            result,
            commission_rate=args.commission_rate,
            minimum_commission=args.minimum_commission,
            initial_capital=args.initial_capital,
            sell_stamp_duty_rate=args.sell_stamp_duty_rate,
            sell_stamp_duty_rate_before_2023_08_28=(
                args.sell_stamp_duty_rate_before_2023_08_28
            ),
            estimated_slippage_rate=args.estimated_slippage_rate,
        )
        vectorbt_equity = cross_check.pop("equity_curve", [])
        result["vectorbt_cross_check"] = cross_check
    database = Database(PATHS.database, code_version="portfolio-backtest-v3")
    database.initialize()
    apply_stock_basic_display_names(
        result,
        TushareRepository(database).load_stock_basic_universe(),
    )
    research_repository = ResearchRepository(database)
    research_repository.persist_formula_history(formula, version="formula33-phase-v1")
    run_id = research_repository.persist_backtest_result(result)
    result["database_run_id"] = run_id
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{args.start_date}_{args.end_date}"
    pd.DataFrame(result["events"]).to_csv(
        output / f"{stem}_events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["trade_ledger"]).to_csv(
        output / f"{stem}_trades.csv", index=False, encoding="utf-8-sig",
    )
    build_readable_trade_frame(result["trade_ledger"]).to_csv(
        output / f"{stem}_买卖流水.csv", index=False, encoding="utf-8-sig",
    )
    (output / f"{stem}_买卖报告.md").write_text(
        render_trade_report_markdown(result), encoding="utf-8",
    )
    pd.DataFrame(result["equity_curve"]).to_csv(
        output / f"{stem}_equity.csv", index=False, encoding="utf-8-sig",
    )
    if args.vectorbt_cross_check:
        pd.DataFrame(vectorbt_equity).to_csv(
            output / f"{stem}_vectorbt_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {key: value for key, value in result.items() if key not in {"events", "equity_curve"}}
    summary["candidate_mode"] = args.candidate_mode
    if explicit_codes:
        summary["candidate_mode"] = "explicit_codes"
        summary["explicit_codes"] = explicit_codes
    if excluded_codes:
        summary["excluded_codes"] = excluded_codes
    summary["candidate_snapshot_dates"] = sorted(snapshots)
    summary["input_fingerprints"] = {
        "candidate_directory": str(Path(args.candidate_directory)),
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": (
            file_sha256(candidate_manifest_path) if candidate_manifest_path.exists() else None
        ),
        "formula_history": str(Path(args.formula_history)),
        "formula_history_sha256": file_sha256(args.formula_history),
        "price_kline_directory": str(price_kline_directory),
    }
    summary["inputs_refreshed"] = not args.no_refresh_inputs
    summary["frozen_inputs_allowed"] = bool(args.allow_frozen_inputs)
    summary["input_coverage_end"] = input_coverage_end
    summary["requested_end_date"] = requested_end_date
    summary["effective_end_date"] = args.end_date
    summary["execution_profile"] = {
        "close_confirmed_execution": args.close_confirmed_execution,
        "signals_effective_next_day": True,
        "commission_rate": float(args.commission_rate),
        "minimum_commission": float(args.minimum_commission),
        "sell_stamp_duty_rate": float(args.sell_stamp_duty_rate),
        "sell_stamp_duty_rate_before_2023_08_28": (
            None
            if args.sell_stamp_duty_rate_before_2023_08_28 is None
            else float(args.sell_stamp_duty_rate_before_2023_08_28)
        ),
        "estimated_slippage_rate": float(args.estimated_slippage_rate),
        "close_proxy_is_research_only": args.close_confirmed_execution == "close_proxy",
    }
    # ``qualify_backtest_result`` already propagated upstream research and
    # input data limitations before persistence; preserve that same value in
    # the emitted summary instead of recomputing a weaker CLI-only predicate.
    summary["research_only"] = bool(result.get("research_only"))
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

