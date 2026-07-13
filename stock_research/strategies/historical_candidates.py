"""Build conservative dated research candidates from locally cached data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from stock_research.core.financial_period import (
    latest_visible_report_period,
    visible_report_periods,
)
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots


SNAPSHOT_VERSION = "unified-selection-v3"
MAX_MAINLINE_AGE_DAYS = 31


def _load_mainline_snapshots(directory):
    snapshots = {}
    if not directory or not Path(directory).exists():
        return snapshots
    for path in Path(directory).glob("sector_mainline_constituents*.csv"):
        match = re.search(r"_(\d{8})$", path.stem)
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except (OSError, ValueError):
            continue
        if frame.empty or "code" not in frame:
            continue
        if match:
            snapshot_date = pd.Timestamp(match.group(1))
        elif "board_date" in frame and frame["board_date"].notna().any():
            snapshot_date = pd.to_datetime(frame["board_date"], errors="coerce").max()
        else:
            continue
        if pd.isna(snapshot_date):
            continue
        members = {}
        for code, group in frame.groupby(frame["code"].astype(str).str.split(".").str[-1].str.zfill(6)):
            members[code] = "、".join(group.get("board", pd.Series(dtype=str)).astype(str).drop_duplicates())
        snapshots[pd.Timestamp(snapshot_date).normalize()] = members
    return snapshots


def report_period_for(date) -> str:
    return latest_visible_report_period(date)


def _load_financial_cache(directory, report_period):
    suffix = pd.Timestamp(report_period).strftime("%Y%m%d")
    rows = {}
    for path in Path(directory).glob(f"*_{suffix}.json"):
        code = path.stem.split("_", 1)[0].zfill(6)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        rows[code] = payload
    return rows


def _load_prices(kline_directory, codes, start_date, end_date):
    result = {}
    for code in codes:
        market = "sh" if code.startswith(("6", "9")) else "sz"
        path = Path(kline_directory) / f"{market}_{code}.csv"
        try:
            frame = pd.read_csv(path, usecols=["date", "close", "volume"])
        except (OSError, ValueError):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame[
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
        ].dropna(subset=["date", "close"])
        if not frame.empty:
            result[code] = frame.set_index("date")[["close", "volume"]]
    return result


def build_historical_candidate_snapshots(
    start_date,
    end_date,
    *,
    value_cache_directory,
    kline_directory,
    universe_path,
    mainline_directory=None,
    max_mainline_age_days=MAX_MAINLINE_AGE_DAYS,
    research_repository=None,
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    universe = pd.read_csv(universe_path, dtype={"code": str})
    names = {
        str(row["code"]).split(".")[-1]: str(row.get("code_name") or row["code"])
        for _, row in universe.iterrows()
    }
    periods = visible_report_periods(start, end)
    financial = {
        period: _load_financial_cache(value_cache_directory, period)
        for period in periods
    }
    if research_repository is not None:
        research_repository.persist_fundamentals(financial)
    mainline_snapshots = _load_mainline_snapshots(mainline_directory)
    codes = set().union(*(rows.keys() for rows in financial.values()))
    prices = _load_prices(kline_directory, codes, start, end)
    calendar = sorted({date.normalize() for frame in prices.values() for date in frame.index})
    snapshots = {}
    for date in calendar:
        period = report_period_for(date)
        eligible_mainline_dates = [item for item in mainline_snapshots if item <= date]
        mainline_date = max(eligible_mainline_dates) if eligible_mainline_dates else None
        mainline_fresh = bool(
            mainline_date is not None
            and (date - mainline_date).days <= int(max_mainline_age_days)
        )
        mainline_members = mainline_snapshots.get(mainline_date, {}) if mainline_fresh else {}
        value_rows = []
        normal_rows = []
        for code, metrics in financial.get(period, {}).items():
            price_frame = prices.get(code)
            if price_frame is None or date not in price_frame.index:
                continue
            market_row = price_frame.loc[date]
            close = float(market_row["close"])
            volume = pd.to_numeric(market_row.get("volume"), errors="coerce")
            if close <= 0 or pd.isna(volume) or volume <= 0:
                continue
            value_line = pd.to_numeric(metrics.get("value_line"), errors="coerce")
            quality = pd.to_numeric(metrics.get("quality_score"), errors="coerce")
            yoy = pd.to_numeric(metrics.get("yoy"), errors="coerce")
            market_cap = pd.to_numeric(metrics.get("mktcap"), errors="coerce")
            if any(pd.isna(item) for item in (quality, yoy, market_cap)):
                continue
            base = {
                "date": date.strftime("%Y-%m-%d"),
                "code": ("sh." if code.startswith(("6", "9")) else "sz.") + code,
                "name": names.get(code, code),
                "close": close,
                "value_line": None if pd.isna(value_line) else float(value_line),
                "quality_score": float(quality),
                "earnings_yoy": float(yoy),
                "mktcap": float(market_cap),
                "report_period": period,
                "snapshot_version": SNAPSHOT_VERSION,
                "financial_point_in_time": False,
                "price_to_value": None if pd.isna(value_line) or value_line <= 0 else close / float(value_line),
                "mainline_snapshot_date": None if mainline_date is None else mainline_date.strftime("%Y-%m-%d"),
                "mainline_snapshot_fresh": mainline_fresh,
                "mainline_boards": mainline_members.get(code, ""),
            }
            if (
                not pd.isna(value_line)
                and value_line > 0
                and 0.80 <= close / value_line <= 1.08
                and quality >= 70
                and yoy >= 0.10
                and market_cap >= 100
            ):
                value_rows.append({
                    **base,
                    "strategy_part": "1.基本价值线或附近",
                    "candidate_score": float(quality) + min(max(float(yoy), 0.0), 1.0) * 20,
                    "historical_adjustment_check": "price_to_value_between_0.80_and_1.08",
                    "candidate_source": "value_model",
                    "signal_eligible": True,
                    "selection_reason": "基本价值线模型入选",
                })
            if code in mainline_members and quality >= 70 and yoy >= 0.10 and market_cap >= 100:
                normal_rows.append({
                    **base,
                    "strategy_part": "2.正常基本面选股",
                    "candidate_score": float(quality) + min(max(float(yoy), 0.0), 1.0) * 20 + 15,
                    "candidate_source": "standard_mainline",
                    "signal_eligible": True,
                    "selection_reason": "主流标准基本面模型入选",
                })
        normal_rows.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        by_code = {item["code"]: item for item in value_rows}
        for item in normal_rows:
            existing = by_code.get(item["code"])
            if existing:
                existing.update({
                    "strategy_part": "1.基本价值线或附近 + 2.主流标准选股",
                    "candidate_source": "value_model+standard_mainline",
                    "signal_eligible": True,
                    "mainline_boards": item["mainline_boards"],
                    "candidate_score": max(existing["candidate_score"], item["candidate_score"]),
                })
            else:
                by_code[item["code"]] = item
        selected = normalize_candidate_snapshots(
            {date.strftime("%Y-%m-%d"): list(by_code.values())}
        )[date.strftime("%Y-%m-%d")]
        snapshots[date.strftime("%Y-%m-%d")] = selected
    return snapshots


def save_historical_candidate_snapshots(output_directory, snapshots, *, start_date, end_date):
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for date, rows in sorted(snapshots.items()):
        frame = pd.DataFrame(rows)
        path = target / f"candidates_{date}.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
        report_period = rows[0]["report_period"] if rows else report_period_for(date)
        eligible_count = sum(bool(row.get("signal_eligible", True)) for row in rows)
        mainline_date = next((row.get("mainline_snapshot_date") for row in rows if row.get("mainline_snapshot_date")), None)
        mainline_fresh = any(bool(row.get("mainline_snapshot_fresh")) for row in rows)
        manifest_rows.append({
            "date": date,
            "report_period": report_period,
            "candidate_count": len(rows),
            "signal_eligible_count": eligible_count,
            "mainline_snapshot_date": mainline_date,
            "mainline_snapshot_fresh": mainline_fresh,
            "financial_point_in_time": False,
            "file": path.name,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "snapshot_count": len(manifest_rows),
        "financial_point_in_time": False,
        "candidate_pool_formula": "every selection model emits the same candidate interface; no manual candidate injection",
        "selection_standard": {
            "value": "0.80 <= price/value_line <= 1.08, quality >= 70, yoy >= 0.10, mktcap >= 100",
            "normal": "quality >= 70, yoy >= 0.10, mktcap >= 100, and member of a fresh dated mainline snapshot",
            "execution": "all signal_eligible model candidates use the same structure, position and exit engine",
            "manual": "watch lists and trade plans cannot inject candidates",
            "mainline_max_age_days": MAX_MAINLINE_AGE_DAYS,
        },
        "point_in_time_note": (
            "行情严格按观察日截断；财务报告期保守选择，但缺少逐公告修订历史，"
            "不得声明为严格财务时点回测。"
        ),
        "snapshots": manifest_rows,
    }
    manifest_path = target / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
