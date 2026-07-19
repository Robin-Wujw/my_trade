"""Run the point-in-time candidate portfolio backtest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from stock_research.core.financial_period import visible_report_periods
from stock_research.core.paths import PATHS
from stock_research.reporting.backtest_trade_report import (
    build_readable_trade_frame,
    render_trade_report_markdown,
)
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.storage import Database, KlineRepository, ResearchRepository
from stock_research.market.miniqmt_data import load_miniqmt_price_frames
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


MIN_FINANCIAL_CACHE_FILES = 1_500
DEFAULT_FINANCIAL_TARGET_COVERAGE = 0.95
DEFAULT_FINANCIAL_CHUNK_SIZE = 10
DEFAULT_FINANCIAL_TIMEOUT = 60
DEFAULT_MINIQMT_KLINE_CHUNK_SIZE = 50


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
    parts = {str(part).lower() for part in Path(kline_directory).parts}
    return "miniqmt" if "miniqmt_kline" in parts else "akshare"


def _chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


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


def _attach_raw_adjustment_factor(frame, raw_directory, code):
    """Attach raw/qfq factor so raw-price portfolios can handle ex-rights."""
    if Path(raw_directory).name != "akshare_raw":
        return frame
    qfq_path = Path(raw_directory).parent / "akshare" / f"{str(code).replace('.', '_')}.csv"
    try:
        qfq = pd.read_csv(qfq_path, usecols=["date", "close"])
    except (OSError, ValueError):
        return frame
    result = frame.copy()
    qfq["date"] = pd.to_datetime(qfq["date"], errors="coerce")
    qfq["close"] = pd.to_numeric(qfq["close"], errors="coerce")
    raw_dates = pd.to_datetime(result.get("date"), errors="coerce")
    raw_close = pd.to_numeric(result.get("close"), errors="coerce")
    qfq_close = qfq.dropna(subset=["date"]).drop_duplicates("date").set_index("date")["close"]
    aligned_qfq = qfq_close.reindex(raw_dates).reset_index(drop=True)
    factor = raw_close.div(aligned_qfq.where(aligned_qfq > 0))
    result["raw_to_qfq_factor"] = factor.where(factor > 0)
    return result


def load_candidate_snapshots(directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    snapshots = {}
    for path in sorted(Path(directory).glob("candidates_*.csv")):
        date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
        if pd.isna(date) or not start <= date.normalize() <= end:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except EmptyDataError:
            frame = pd.DataFrame()
        snapshots[date.strftime("%Y-%m-%d")] = frame.to_dict("records")
    return snapshots


def load_price_frames(
    codes,
    directory,
    *,
    start_date=None,
    end_date=None,
    source="akshare",
    prefer_database=True,
):
    """Load OHLC frames for execution and valuation.

    DuckDB is the fast authority for the normal qfq cache.  Raw historical
    execution tests must be able to bypass it, otherwise a raw CSV directory is
    silently overwritten by qfq database rows.
    """
    normalized_codes = sorted({str(code) for code in codes})
    frames = {}
    if prefer_database and PATHS.database.is_file() and normalized_codes:
        repository = KlineRepository(Database(PATHS.database))
        loaded = repository.load_stock_klines(
            source,
            normalized_codes,
            start_date=str(start_date or "1900-01-01"),
            end_date=str(end_date or "2999-12-31"),
            max_qfq_anchor_date=None if end_date is None else str(end_date),
        )
        if not loaded.empty:
            frames.update({
                str(code): group.drop(columns=["code"]).reset_index(drop=True)
                for code, group in loaded.groupby("code", sort=False)
            })
    for code in (item for item in normalized_codes if item not in frames):
        path = Path(directory) / f"{str(code).replace('.', '_')}.csv"
        try:
            frame = pd.read_csv(path)
            if start_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(start_date)]
            if end_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") <= pd.Timestamp(end_date)]
            frame = _attach_raw_adjustment_factor(frame, directory, code)
            frames[str(code)] = frame
        except (OSError, ValueError):
            continue
    return frames


def validate_price_frame_coverage(price_frames, codes, start_date, end_date, *, code_start_dates=None):
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
    for code, first in first_by_code.items():
        last = last_by_code[code]
        required_start = pd.Timestamp(code_start_dates.get(code, effective_start)).normalize()
        if first > required_start:
            late_start.append(f"{code}:{first:%Y-%m-%d}>{required_start:%Y-%m-%d}")
        if last < end:
            early_end.append(f"{code}:{last:%Y-%m-%d}")
    problems = {
        "missing": missing,
        "empty": empty,
        "invalid": invalid,
        "late_start": late_start,
        "early_end": early_end,
    }
    failed = {key: value for key, value in problems.items() if value}
    if failed:
        preview = {key: value[:10] for key, value in failed.items()}
        raise RuntimeError(
            "price K-line frames do not cover all backtest candidate codes "
            f"through {end:%Y-%m-%d}: {preview}"
        )
    return {
        "code_count": len({str(item) for item in codes if str(item).strip()}),
        "start_date": effective_start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }


def first_candidate_dates(snapshots):
    result = {}
    for date_text, rows in sorted(snapshots.items()):
        date = pd.Timestamp(date_text).normalize()
        for row in rows:
            code = str(row.get("code") or "").strip()
            if code and code not in result:
                result[code] = date
    return result


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
            if row.get("financial_point_in_time") is False
        ]
    return {
        "available": True,
        "financial_point_in_time": manifest.get("financial_point_in_time"),
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
    if status["financial_point_in_time"] is False or status["unsafe_dates"]:
        preview = ", ".join(status["unsafe_dates"][:5])
        raise RuntimeError(
            "candidate manifest is not strict financial point-in-time; "
            "use --allow-unsafe-financial only for research backtests. "
            f"manifest={status['path']} unsafe_dates={preview}"
        )
    return status


def candidate_manifest_reusable(candidate_directory, start_date, end_date, *, allow_unsafe_financial=False):
    path = Path(candidate_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    requested_end = pd.to_datetime(manifest.get("requested_end"), errors="coerce")
    if pd.isna(requested_end) or requested_end.normalize() < pd.Timestamp(end_date).normalize():
        return False
    if not allow_unsafe_financial:
        status = validate_candidate_manifest_financial_point_in_time(
            candidate_directory,
            allow_unsafe_financial=False,
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
    calendar_days = max(0, (end - start).days)
    warmup_start = (start - pd.offsets.BDay(45)).normalize()
    # Formula33 uses lookback for output rows and history-days for both raw
    # calendar depth and per-symbol K-line depth.  Keep both large enough for
    # the requested range while leaving the model's own indicator warmup intact.
    return {
        "start_date": warmup_start.strftime("%Y-%m-%d"),
        "lookback": max(21, calendar_days + 30),
        "history_days": 420,
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
    ]
    if candidate_manifest_reusable(
        args.candidate_directory,
        args.start_date,
        args.end_date,
        allow_unsafe_financial=args.allow_unsafe_financial,
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
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output", args.formula_history,
        "--kline-directory", str(formula_kline_directory),
    ])


def validate_backtest_input_coverage(
    snapshots,
    formula,
    requested_start,
    requested_end,
    *,
    candidate_directory=None,
    allow_unsafe_financial=False,
):
    """Fail closed when every trading day lacks a fresh non-empty selection."""
    if candidate_directory is not None:
        validate_candidate_manifest_financial_point_in_time(
            candidate_directory,
            allow_unsafe_financial=allow_unsafe_financial,
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
    empty_candidate_dates = sorted(
        date for date in formula_trade_dates
        if not snapshots.get(date.strftime("%Y-%m-%d"))
    )
    if empty_candidate_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in empty_candidate_dates[:10])
        raise RuntimeError(
            "candidate snapshots contain empty selection days; "
            f"every backtest day must have a fresh non-empty selection result. empty={preview}"
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="候选池、Formula33和四持仓组合回测")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-date", default="",
        help="default: latest session whose daily bar should be available",
    )
    parser.add_argument(
        "--max-positions", type=int, default=3,
        help="main capacity symbols; default 3, hard-capped at 5",
    )
    parser.add_argument(
        "--max-total-held-symbols", type=int, default=5,
        help="hard cap for all held symbols, including left cores and profit tails",
    )
    parser.add_argument(
        "--max-same-industry", type=int, default=2,
        help="maximum simultaneously held symbols sharing a dated industry/board tag",
    )
    parser.add_argument(
        "--same-theme-correlation", type=float, default=0.60,
        help="60-session return correlation used to group related exposure when tags are missing",
    )
    parser.add_argument(
        "--min-entry-evidence-score", type=float, default=7.0,
        help="minimum multi-signal technical evidence score for an executable entry",
    )
    parser.add_argument(
        "--profit-tail-min-return", type=float, default=0.50,
        help="minimum current return for the last tranche not to consume a slot",
    )
    parser.add_argument(
        "--profit-tranches", type=int, choices=(2, 3, 4, 5), default=5,
        help="number of profit-taking tranches; the last is reserved for maximum-profit-half",
    )
    parser.add_argument(
        "--left-grid-unit", type=float, default=0.02,
        help="single left-grid unit as account fraction; use 0 to disable new left-grid buys",
    )
    parser.add_argument(
        "--left-grid-step", type=float, default=0.05,
        help="left-grid price spacing; minimum enforced by strategy is 0.05",
    )
    parser.add_argument(
        "--left-grid-max-exposure", type=float, default=0.20,
        help="maximum left-grid exposure per symbol as account fraction",
    )
    parser.add_argument(
        "--codes",
        help="comma-separated explicit candidate universe; overrides snapshot members",
    )
    parser.add_argument(
        "--exit-tail-on-candidate-removal", action="store_true",
        help="exit right-side tails below 10%% at the next open after candidate removal",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("rolling", "fixed-first"),
        default="rolling",
        help="rolling逐日更新候选；fixed-first冻结首个交易日候选",
    )
    parser.add_argument(
        "--candidate-directory",
        default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"),
    )
    parser.add_argument(
        "--formula-history",
        default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"),
    )
    parser.add_argument(
        "--trade-plans",
        default=str(PATHS.project_root / "config" / "trade_plans.json"),
    )
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "portfolio"),
    )
    parser.add_argument(
        "--price-kline-directory",
        default="",
        help="组合回测成交价和持仓估值使用的K线目录；严格历史回测建议使用不复权缓存",
    )
    parser.add_argument(
        "--no-price-database", action="store_true",
        help="只从 price-kline-directory 读取CSV，不使用DuckDB前复权K线",
    )
    parser.add_argument(
        "--no-refresh-inputs", action="store_true",
        help="research only: skip automatic K-line/financial/Formula33/candidate refresh; requires --allow-unsafe-financial",
    )
    parser.add_argument(
        "--refresh-price-source",
        choices=("akshare", "miniqmt", "miniqmt-akshare"),
        default="miniqmt",
        help="K-line source used by the strict pre-backtest refresh chain",
    )
    parser.add_argument(
        "--refresh-metadata-source",
        choices=("akshare", "baostock", "auto"),
        default="auto",
        help="metadata source used by Formula33 trade calendar/universe/list-date refresh",
    )
    parser.add_argument(
        "--refresh-market-cap-source",
        choices=("auto", "tushare", "akshare", "akshare-capital", "none"),
        default="auto",
        help="market-cap source used by Formula33; none is degraded research only",
    )
    parser.add_argument(
        "--no-auto-price-structure", action="store_true",
        help="disable automatic structure anchors for a sensitivity comparison",
    )
    parser.add_argument(
        "--no-structure-pullback", action="store_true",
        help="allow structure breakout orders but disable structure support pullback orders",
    )
    parser.add_argument(
        "--allow-pullback-pilot", action="store_true",
        help="兼容旧参数；支撑拉回小仓首仓现在默认允许",
    )
    parser.add_argument(
        "--disable-pullback-pilot", action="store_true",
        help="关闭领先族群支撑拉回小仓首仓，用于敏感性对照",
    )
    parser.add_argument(
        "--allow-unsafe-financial", action="store_true",
        help="research only: allow candidate manifests marked financial_point_in_time=false",
    )
    parser.add_argument(
        "--financial-target-coverage",
        type=float,
        default=DEFAULT_FINANCIAL_TARGET_COVERAGE,
        help="minimum strict financial point-in-time coverage required before backtest",
    )
    parser.add_argument(
        "--financial-chunk-size",
        type=int,
        default=DEFAULT_FINANCIAL_CHUNK_SIZE,
        help="MiniQMT financial auto-fetch chunk size",
    )
    parser.add_argument(
        "--financial-timeout",
        type=int,
        default=DEFAULT_FINANCIAL_TIMEOUT,
        help="MiniQMT financial auto-fetch timeout per chunk in seconds",
    )
    parser.add_argument(
        "--close-confirmed-execution", choices=("close_proxy", "next_open"),
        default="close_proxy",
        help="14:55/close proxy follows the strategy; next_open is the conservative sensitivity case",
    )
    parser.add_argument("--commission-rate", type=float, default=0.000085)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--estimated-slippage-rate", type=float, default=0.0005)
    parser.add_argument(
        "--vectorbt-cross-check", action="store_true",
        help="replay emitted fills with vectorbt shared-cash accounting",
    )
    args = parser.parse_args(argv)
    if not args.price_kline_directory:
        args.price_kline_directory = str(refresh_price_cache_directory(args.refresh_price_source))
    if not args.end_date:
        args.end_date = default_data_end_date()
    requested_end_date = args.end_date
    if args.no_refresh_inputs and not args.allow_unsafe_financial:
        raise RuntimeError(
            "--no-refresh-inputs is research-only because strict backtests must "
            "auto-refresh K-line, financial, Formula33, and candidate inputs. "
            "Use --allow-unsafe-financial only for degraded offline research."
        )
    if not args.no_refresh_inputs:
        refresh_backtest_inputs(args)
    else:
        print(
            "[portfolio_backtest][WARNING] input refresh explicitly disabled; "
            "results use frozen local data"
        )
    snapshots = load_candidate_snapshots(
        args.candidate_directory, args.start_date, args.end_date,
    )
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
    trade_plans = load_trade_plans(args.trade_plans)
    price_kline_directory = Path(args.price_kline_directory)
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
    )
    price_coverage = validate_price_frame_coverage(
        price_frames,
        codes,
        args.start_date,
        args.end_date,
        code_start_dates=code_start_dates,
    )
    formula = pd.read_csv(args.formula_history)
    input_coverage_end = validate_backtest_input_coverage(
        coverage_snapshots,
        formula,
        args.start_date,
        args.end_date,
        candidate_directory=args.candidate_directory,
        allow_unsafe_financial=args.allow_unsafe_financial,
    )
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    result = run_portfolio_backtest(
        price_frames,
        snapshots,
        phases,
        requested_start=args.start_date,
        end_date=args.end_date,
        trade_plans=trade_plans,
        max_positions=None if args.max_positions == 0 else args.max_positions,
        max_total_held_symbols=args.max_total_held_symbols,
        max_same_industry=args.max_same_industry,
        same_theme_correlation=args.same_theme_correlation,
        min_entry_evidence_score=args.min_entry_evidence_score,
        profit_tranches=args.profit_tranches,
        profit_tail_min_return=args.profit_tail_min_return,
        left_grid_unit=args.left_grid_unit,
        left_grid_step=args.left_grid_step,
        left_grid_max_exposure=args.left_grid_max_exposure,
        exit_tail_on_candidate_removal=args.exit_tail_on_candidate_removal,
        signals_effective_next_day=True,
        auto_price_structure=not args.no_auto_price_structure,
        allow_structure_pullback=not args.no_structure_pullback,
        allow_pullback_pilot=not args.disable_pullback_pilot,
        close_confirmed_execution=args.close_confirmed_execution,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        initial_capital=args.initial_capital,
        sell_stamp_duty_rate=args.sell_stamp_duty_rate,
        estimated_slippage_rate=args.estimated_slippage_rate,
    )
    result["price_source"] = {
        "kline_directory": str(price_kline_directory),
        "database_enabled": bool(use_price_database),
        "source": infer_price_frame_source(price_kline_directory),
        "mode": "不复权CSV" if not use_price_database else "DuckDB前复权优先",
        "coverage": price_coverage,
    }
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
            estimated_slippage_rate=args.estimated_slippage_rate,
        )
        vectorbt_equity = cross_check.pop("equity_curve")
        result["vectorbt_cross_check"] = cross_check
    database = Database(PATHS.database, code_version="portfolio-backtest-v3")
    database.initialize()
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
    summary["candidate_snapshot_dates"] = sorted(snapshots)
    summary["inputs_refreshed"] = not args.no_refresh_inputs
    summary["input_coverage_end"] = input_coverage_end
    summary["requested_end_date"] = requested_end_date
    summary["effective_end_date"] = args.end_date
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
