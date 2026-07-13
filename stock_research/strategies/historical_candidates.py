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
CANDIDATE_SNAPSHOT_COLUMNS = [
    "date", "code", "name", "close", "value_line", "quality_score",
    "earnings_yoy", "mktcap", "report_period", "snapshot_version",
    "financial_point_in_time", "price_to_value", "mainline_snapshot_date",
    "mainline_snapshot_fresh", "mainline_boards", "trade_basis_score",
    "trade_basis_reason", "technical_alignment", "ma20_rising",
    "ma60_rising", "above_ma20", "near_ma20", "near_21d_close_high",
    "known_volume_ratio", "volume_deduction_periods", "ima_web_validation",
    "validation_sources", "strategy_part", "candidate_score",
    "historical_adjustment_check", "candidate_source", "signal_eligible",
    "selection_reason", "selection_rank",
]


IMA_WEB_VALIDATION_SOURCES = [
    {
        "source_type": "ima",
        "title": "均线均量扣抵思想",
        "rule": "扣抵方向、均线支撑和量能确认只能作为结构证据，不能单独生成买卖信号",
    },
    {
        "source_type": "web",
        "title": "东方财富作者页：白白胖胖0",
        "url": "https://i.eastmoney.com/2920015446601888",
        "rule": "市场结构、量、价格、技术指标按优先级互相验证",
    },
    {
        "source_type": "web",
        "title": "均线均量扣抵基础公式",
        "url": "https://caifuhao.eastmoney.com/news/20220502203427525523430",
        "rule": "移动平均方向由新值和扣抵值关系决定",
    },
]


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
            frame = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume"])
        except (OSError, ValueError):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
        ].dropna(subset=["date", "close"])
        if not frame.empty:
            result[code] = frame.set_index("date")[["open", "high", "low", "close", "volume"]]
    return result


def _trade_basis_snapshot(price_frame: pd.DataFrame, date) -> dict:
    """Score model candidates with only information visible at observation close."""
    history = price_frame.loc[:date].copy()
    if history.empty:
        return {
            "trade_basis_score": 0.0,
            "trade_basis_reason": "缺少观察日行情，等待补数",
            "technical_alignment": "missing_price",
        }
    close = history["close"].astype(float)
    volume = history["volume"].astype(float)
    latest_close = float(close.iloc[-1])
    latest_volume = float(volume.iloc[-1]) if pd.notna(volume.iloc[-1]) else 0.0

    def ma(period: int):
        return close.rolling(period).mean()

    ma20 = ma(20)
    ma60 = ma(60)
    volume5 = volume.rolling(5).mean()
    volume10 = volume.rolling(10).mean()
    latest_ma20 = ma20.iloc[-1] if len(ma20) else pd.NA
    latest_ma60 = ma60.iloc[-1] if len(ma60) else pd.NA
    ma20_rising = len(ma20) > 5 and pd.notna(latest_ma20) and latest_ma20 > ma20.iloc[-6]
    ma60_rising = len(ma60) > 5 and pd.notna(latest_ma60) and latest_ma60 > ma60.iloc[-6]
    above_ma20 = pd.notna(latest_ma20) and latest_close >= float(latest_ma20)
    near_ma20 = (
        pd.notna(latest_ma20)
        and latest_ma20 > 0
        and abs(latest_close / float(latest_ma20) - 1.0) <= 0.05
    )
    prior_high = close.iloc[:-1].tail(21).max() if len(close) > 21 else pd.NA
    near_breakout = pd.notna(prior_high) and prior_high > 0 and latest_close >= float(prior_high) * 0.97
    volume_base = max(
        float(volume5.iloc[-2]) if len(volume5) > 1 and pd.notna(volume5.iloc[-2]) else 0.0,
        float(volume10.iloc[-2]) if len(volume10) > 1 and pd.notna(volume10.iloc[-2]) else 0.0,
    )
    volume_ratio = latest_volume / volume_base if volume_base > 0 else 0.0
    deduction_periods = [period for period in (5, 10, 20) if len(volume) > period and latest_volume > volume.iloc[-period - 1]]

    score = 0.0
    reasons = []
    if ma20_rising and ma60_rising:
        score += 4.0
        reasons.append("MA20/MA60同步上扬")
    elif ma20_rising:
        score += 2.0
        reasons.append("MA20上扬")
    if above_ma20:
        score += 2.0
        reasons.append("收盘站上MA20")
    elif near_ma20 and ma20_rising:
        score += 2.0
        reasons.append("贴近上扬MA20支撑")
    if near_breakout:
        score += 3.0
        reasons.append("距离21日收盘高点3%以内")
    if volume_ratio >= 1.2:
        score += 2.0
        reasons.append(f"量能高于5/10日基准{volume_ratio:.2f}倍")
    if len(deduction_periods) >= 2:
        score += 1.0
        reasons.append("多周期均量扣低走高")

    alignment = "trade_ready" if score >= 7 else "watch" if score >= 4 else "fundamental_only"
    return {
        "trade_basis_score": round(score, 3),
        "trade_basis_reason": "；".join(reasons) or "基本面入选，等待价格/量能买点",
        "technical_alignment": alignment,
        "ma20_rising": bool(ma20_rising),
        "ma60_rising": bool(ma60_rising),
        "above_ma20": bool(above_ma20),
        "near_ma20": bool(near_ma20),
        "near_21d_close_high": bool(near_breakout),
        "known_volume_ratio": round(volume_ratio, 4),
        "volume_deduction_periods": ",".join(map(str, deduction_periods)),
        "ima_web_validation": "aligned" if score >= 4 else "needs_price_confirmation",
        "validation_sources": IMA_WEB_VALIDATION_SOURCES,
    }


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
    price_start = start - pd.Timedelta(days=420)
    prices = _load_prices(kline_directory, codes, price_start, end)
    calendar = sorted({
        date.normalize()
        for frame in prices.values()
        for date in frame.index
        if start <= date.normalize() <= end
    })
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
            trade_basis = _trade_basis_snapshot(price_frame, date)
            base.update(trade_basis)
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
                    "candidate_score": (
                        float(quality)
                        + min(max(float(yoy), 0.0), 1.0) * 20
                        + float(trade_basis["trade_basis_score"])
                    ),
                    "historical_adjustment_check": "price_to_value_between_0.80_and_1.08",
                    "candidate_source": "value_model",
                    "signal_eligible": True,
                    "selection_reason": (
                        "基本价值线模型入选；"
                        f"{trade_basis['trade_basis_reason']}"
                    ),
                })
            if code in mainline_members and quality >= 70 and yoy >= 0.10 and market_cap >= 100:
                normal_rows.append({
                    **base,
                    "strategy_part": "2.正常基本面选股",
                    "candidate_score": (
                        float(quality)
                        + min(max(float(yoy), 0.0), 1.0) * 20
                        + 15
                        + float(trade_basis["trade_basis_score"])
                    ),
                    "candidate_source": "standard_mainline",
                    "signal_eligible": True,
                    "selection_reason": (
                        "主流标准基本面模型入选；"
                        f"{trade_basis['trade_basis_reason']}"
                    ),
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
        if frame.empty:
            frame = pd.DataFrame(columns=CANDIDATE_SNAPSHOT_COLUMNS)
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
