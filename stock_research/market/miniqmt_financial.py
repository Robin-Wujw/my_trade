"""MiniQMT financial-data cache builder.

MiniQMT exposes historical financial rows through ``xtdata.get_financial_data``.
This module maps the small subset needed by the existing value-line candidate
pipeline into the project's ``var/cache/q1_value`` JSON format.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.api.miniqmt import MiniQmtConfig, query_financial_data_via_qmt_python
from stock_research.core.paths import PATHS
from stock_research.indicators.factors import clamp, score_direct, score_inverse
from stock_research.market.miniqmt_data import (
    miniqmt_code_to_project,
    normalize_project_code,
)


MINIQMT_FINANCIAL_TABLES = ("Income", "Capital", "PershareIndex")
MINIQMT_FINANCIAL_CACHE_VERSION = "miniqmt-financial-v1"
VALUE_CACHE_DIR = PATHS.cache / "q1_value"


@dataclass(frozen=True)
class MiniQmtFinancialCacheResult:
    requested_count: int
    saved_count: int
    skipped_count: int
    failed_chunks: int
    report_period: str
    as_of_date: str
    output_directory: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "saved_count": self.saved_count,
            "skipped_count": self.skipped_count,
            "failed_chunks": self.failed_chunks,
            "report_period": self.report_period,
            "as_of_date": self.as_of_date,
            "output_directory": self.output_directory,
        }


def default_financial_as_of_date(report_period: str) -> str:
    """Return the conservative disclosure-deadline date for a report period."""
    period = pd.Timestamp(report_period).normalize()
    year = int(period.year)
    month_day = (int(period.month), int(period.day))
    if month_day == (3, 31):
        return f"{year}-04-30"
    if month_day == (6, 30):
        return f"{year}-08-31"
    if month_day == (9, 30):
        return f"{year}-10-31"
    if month_day == (12, 31):
        return f"{year + 1}-04-30"
    raise ValueError(f"unsupported regular report period: {report_period}")


def financial_cache_path(code: str, report_period: str, output_directory: str | Path | None = None) -> Path:
    directory = Path(output_directory) if output_directory else VALUE_CACHE_DIR
    symbol = normalize_project_code(code).split(".")[-1]
    suffix = pd.Timestamp(report_period).strftime("%Y%m%d")
    return directory / f"{symbol}_{suffix}.json"


def point_in_time_financial_cache_complete(
    code: str,
    report_period: str,
    *,
    as_of_date: str,
    output_directory: str | Path | None = None,
) -> bool:
    path = financial_cache_path(code, report_period, output_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if payload.get("financial_point_in_time_source") != "announce_time":
        return False
    announcement = _date(payload.get("announcement_date"))
    as_of = pd.Timestamp(as_of_date).normalize()
    if announcement is None or announcement > as_of:
        return False
    for key in ("value_line", "quality_score", "eps_excl", "yoy", "total_share"):
        if _num(payload.get(key)) is None:
            return False
    return True


def point_in_time_financial_cache_coverage(
    codes: list[str] | tuple[str, ...],
    *,
    report_period: str,
    as_of_date: str,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    normalized_codes = sorted({normalize_project_code(code) for code in codes if str(code).strip()})
    complete = [
        code for code in normalized_codes
        if point_in_time_financial_cache_complete(
            code,
            report_period,
            as_of_date=as_of_date,
            output_directory=output_directory,
        )
    ]
    total = len(normalized_codes)
    return {
        "requested_count": total,
        "complete_count": len(complete),
        "missing_or_unsafe_count": total - len(complete),
        "coverage": len(complete) / total if total else 0.0,
        "report_period": pd.Timestamp(report_period).strftime("%Y-%m-%d"),
        "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
    }


def load_universe_codes(path: str | Path | None = None) -> list[str]:
    source = Path(path) if path else PATHS.cache / "stock_universe.csv"
    frame = pd.read_csv(source, dtype={"code": str})
    codes = []
    for value in frame["code"].astype(str):
        code = normalize_project_code(value)
        if code.startswith(("sh.60", "sh.68", "sz.00", "sz.30")):
            codes.append(code)
    return sorted(set(codes))


def build_miniqmt_financial_cache(
    codes: list[str] | tuple[str, ...],
    *,
    report_period: str,
    as_of_date: str | None = None,
    output_directory: str | Path | None = None,
    chunk_size: int = 20,
    config: MiniQmtConfig | None = None,
    overwrite: bool = False,
    missing_point_in_time_only: bool = False,
    timeout: int = 180,
) -> dict[str, Any]:
    """Fetch MiniQMT finance rows and persist q1_value-compatible JSON files."""
    normalized_codes = sorted({normalize_project_code(code) for code in codes if str(code).strip()})
    report_text = pd.Timestamp(report_period).strftime("%Y-%m-%d")
    as_of_text = pd.Timestamp(as_of_date or default_financial_as_of_date(report_text)).strftime("%Y-%m-%d")
    output_dir = Path(output_directory) if output_directory else VALUE_CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    start_year = max(1990, pd.Timestamp(report_text).year - 4)
    start_time = f"{start_year}0101"
    end_time = pd.Timestamp(as_of_text).strftime("%Y%m%d")

    saved = 0
    skipped = 0
    skipped_existing = 0
    skipped_no_metrics = 0
    failed_chunks = 0
    errors: list[dict[str, Any]] = []
    saved_codes: list[str] = []
    skipped_codes: list[str] = []

    for chunk in _chunks(normalized_codes, max(1, int(chunk_size))):
        pending = [
            code for code in chunk
            if overwrite
            or not financial_cache_path(code, report_text, output_dir).is_file()
            or (
                missing_point_in_time_only
                and not point_in_time_financial_cache_complete(
                    code,
                    report_text,
                    as_of_date=as_of_text,
                    output_directory=output_dir,
                )
            )
        ]
        skipped_existing_chunk = len(chunk) - len(pending)
        skipped += skipped_existing_chunk
        skipped_existing += skipped_existing_chunk
        if skipped_existing_chunk:
            skipped_codes.extend(code for code in chunk if code not in pending)
        if not pending:
            continue
        try:
            payload = query_financial_data_via_qmt_python(
                pending,
                MINIQMT_FINANCIAL_TABLES,
                start_time=start_time,
                end_time=end_time,
                report_type="announce_time",
                config=config,
                row_limit=0,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - record per chunk and keep going.
            failed_chunks += 1
            errors.append({"codes": pending, "error": str(exc)})
            continue
        if not payload.get("ok"):
            failed_chunks += 1
            errors.append({"codes": pending, "error": payload.get("error", "MiniQMT financial query failed")})
            continue
        for provider_code, table_map in payload.get("data", {}).items():
            project_code = miniqmt_code_to_project(provider_code)
            metrics = build_value_metrics_from_miniqmt_tables(
                project_code,
                table_map,
                report_period=report_text,
                as_of_date=as_of_text,
            )
            if not metrics:
                skipped += 1
                skipped_no_metrics += 1
                skipped_codes.append(project_code)
                continue
            save_json_atomic(financial_cache_path(project_code, report_text, output_dir), metrics)
            saved += 1
            saved_codes.append(project_code)

    result = MiniQmtFinancialCacheResult(
        requested_count=len(normalized_codes),
        saved_count=saved,
        skipped_count=skipped,
        failed_chunks=failed_chunks,
        report_period=report_text,
        as_of_date=as_of_text,
        output_directory=str(output_dir),
    ).to_dict()
    result["source"] = "miniqmt/xtdata_financial"
    result["tables"] = list(MINIQMT_FINANCIAL_TABLES)
    result["cache_version"] = MINIQMT_FINANCIAL_CACHE_VERSION
    result["missing_point_in_time_only"] = bool(missing_point_in_time_only)
    result["skipped_existing_count"] = skipped_existing
    result["skipped_no_metrics_count"] = skipped_no_metrics
    result["point_in_time_coverage"] = point_in_time_financial_cache_coverage(
        normalized_codes,
        report_period=report_text,
        as_of_date=as_of_text,
        output_directory=output_dir,
    )
    result["errors"] = errors[:20]
    result["saved_sample"] = saved_codes[:20]
    result["skipped_sample"] = skipped_codes[:20]
    return result


def build_value_metrics_from_miniqmt_tables(
    code: str,
    table_map: dict[str, Any],
    *,
    report_period: str,
    as_of_date: str,
) -> dict[str, Any] | None:
    """Map MiniQMT tables for one stock into value-line metrics."""
    report = pd.Timestamp(report_period).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    pershare = _visible_rows(_summary_rows(table_map, "PershareIndex"), as_of)
    income = _visible_rows(_summary_rows(table_map, "Income"), as_of)
    capital = _visible_rows(_summary_rows(table_map, "Capital"), as_of)

    latest = _row_for_period(pershare, report)
    if latest is None:
        return None
    annual = _latest_annual_row(pershare, report)
    if annual is None:
        return None
    capital_row = _latest_row(capital, as_of, max_period=report)
    if capital_row is None:
        return None

    bvps = _num(latest.get("s_fa_bps"))
    raw_eps = _num(annual.get("adjusted_earnings_per_share"))
    yoy_pct = _num(latest.get("adjusted_net_profit_rate"))
    total_share = _num(capital_row.get("total_capital"))
    if bvps is None or raw_eps is None or yoy_pct is None or total_share is None or total_share <= 0:
        return None

    yoy = yoy_pct / 100.0
    modeled_value_line = bvps + raw_eps * (1 + yoy) * 10
    loss_maker = raw_eps <= 0
    value_line = modeled_value_line
    value_line_policy = "modeled_value_line"
    if loss_maker or value_line <= 0:
        # A loss-maker still has usable point-in-time financial disclosure.
        # Keep the cache complete, but give downstream filters a weak positive
        # anchor so price/value remains computable and quality gates reject it.
        value_line = max(bvps, 0.01)
        value_line_policy = "book_value_floor_for_loss_maker"

    annual_rows = _annual_rows(pershare, report)[-3:]
    annual_income_by_period = {
        _period_key(row): row for row in income if _period_key(row)
    }
    annual_excl = []
    for row in annual_rows:
        period_key = _period_key(row)
        income_row = annual_income_by_period.get(period_key)
        value = _num((income_row or {}).get("net_profit_incl_min_int_inc_after"))
        if value is None:
            eps = _num(row.get("adjusted_earnings_per_share"))
            value = eps * total_share if eps is not None else None
        annual_excl.append(value)
    annual_excl_clean = [value for value in annual_excl if value is not None]
    positive_years = sum(1 for value in annual_excl_clean if value > 0)
    growth_steps = sum(
        1
        for index in range(len(annual_excl_clean) - 1)
        if annual_excl_clean[index + 1] > annual_excl_clean[index]
    )
    quality_yoy = min(max(yoy, -0.5), 1.0)
    quality_score = (
        score_direct(raw_eps, 0.10, 1.50) * 0.35
        + score_direct(quality_yoy, -0.10, 0.50) * 0.35
        + score_direct(positive_years, 1, 3) * 0.15
        + score_direct(growth_steps, 0, 2) * 0.15
    )

    metrics = {
        "cache_version": MINIQMT_FINANCIAL_CACHE_VERSION,
        "value_line": float(value_line),
        "modeled_value_line": float(modeled_value_line),
        "value_line_policy": value_line_policy,
        "price_to_value": None,
        "valuation_score": score_inverse(None, best=0.55, worst=1.25),
        "quality_score": clamp(quality_score),
        "mktcap": None,
        "eps_excl": float(raw_eps),
        "yoy": float(yoy),
        "yoy_source": "miniqmt PershareIndex.adjusted_net_profit_rate",
        "latest_excl_eps": _num(latest.get("adjusted_earnings_per_share")),
        "prev_excl_eps": None,
        "latest_report": report.strftime("%Y-%m-%d"),
        "annual_report": pd.Timestamp(str(_period_key(annual))).strftime("%Y-%m-%d"),
        "data_source": "miniqmt/xtdata_financial",
        "total_share": float(total_share),
        "eps_excl_raw": float(raw_eps),
        "eps_adjustment_factor": 1.0,
        "eps_excl_source": "MiniQMT PershareIndex年报扣非EPS",
        "eps_bonus_detail": None,
        "loss_maker": bool(loss_maker),
        "report_period": report.strftime("%Y-%m-%d"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_history_available": False,
        "financial_point_in_time_source": "announce_time",
        "announcement_date": _date_text(latest.get("m_anntime")),
        "annual_announcement_date": _date_text(annual.get("m_anntime")),
        "capital_announcement_date": _date_text(capital_row.get("m_anntime")),
        "miniqmt_provider_code": _project_to_provider(code),
    }
    return metrics


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{random.randint(100000, 999999)}.tmp")
    tmp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _summary_rows(table_map: dict[str, Any], table: str) -> list[dict[str, Any]]:
    value = table_map.get(table) or table_map.get(table.upper()) or {}
    if isinstance(value, dict):
        rows = value.get("sample") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _visible_rows(rows: list[dict[str, Any]], as_of: pd.Timestamp) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        announce = _date(row.get("m_anntime"))
        period = _date(row.get("m_timetag"))
        if announce is None or period is None:
            continue
        if announce <= as_of:
            result.append(row)
    return sorted(result, key=lambda row: (_date(row.get("m_timetag")) or pd.Timestamp.min, _date(row.get("m_anntime")) or pd.Timestamp.min))


def _row_for_period(rows: list[dict[str, Any]], report: pd.Timestamp) -> dict[str, Any] | None:
    matches = [row for row in rows if (_date(row.get("m_timetag")) or pd.Timestamp.min).normalize() == report]
    return matches[-1] if matches else None


def _latest_annual_row(rows: list[dict[str, Any]], report: pd.Timestamp) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if (_date(row.get("m_timetag")) is not None)
        and (_date(row.get("m_timetag")).month, _date(row.get("m_timetag")).day) == (12, 31)
        and _date(row.get("m_timetag")) <= report
    ]
    return candidates[-1] if candidates else None


def _annual_rows(rows: list[dict[str, Any]], report: pd.Timestamp) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if (_date(row.get("m_timetag")) is not None)
        and (_date(row.get("m_timetag")).month, _date(row.get("m_timetag")).day) == (12, 31)
        and _date(row.get("m_timetag")) <= report
    ]


def _latest_row(rows: list[dict[str, Any]], as_of: pd.Timestamp, *, max_period: pd.Timestamp | None = None) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        period = _date(row.get("m_timetag"))
        announce = _date(row.get("m_anntime"))
        if period is None or announce is None:
            continue
        if announce <= as_of and (max_period is None or period <= max_period):
            candidates.append(row)
    return candidates[-1] if candidates else None


def _period_key(row: dict[str, Any]) -> str | None:
    period = _date(row.get("m_timetag"))
    return None if period is None else period.strftime("%Y%m%d")


def _date(value: Any) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        if text.endswith(".0"):
            text = text[:-2]
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    except Exception:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _date_text(value: Any) -> str | None:
    parsed = _date(value)
    return None if parsed is None else parsed.strftime("%Y-%m-%d")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _project_to_provider(code: str) -> str:
    normalized = normalize_project_code(code)
    market, symbol = normalized.split(".", 1)
    return f"{symbol.upper()}.{market.upper()}"


def _chunks(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
