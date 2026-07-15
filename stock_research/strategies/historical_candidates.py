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
from stock_research.strategies.candidate_interface import (
    normalize_candidate,
    normalize_candidate_snapshots,
)


SNAPSHOT_VERSION = "unified-selection-v4"
MAX_MAINLINE_AGE_DAYS = 31
CANDIDATE_SNAPSHOT_COLUMNS = [
    "date", "code", "name", "close", "value_line", "quality_score",
    "earnings_yoy", "mktcap", "report_period", "snapshot_version",
    "financial_point_in_time", "price_to_value", "mainline_snapshot_date",
    "mainline_snapshot_fresh", "mainline_boards", "trade_basis_score",
    "trade_basis_reason", "technical_alignment", "ma20_rising",
    "ma60_rising", "above_ma20", "near_ma20", "near_21d_close_high",
    "known_volume_ratio", "volume_deduction_periods", "ima_web_validation",
    "return_20d", "return_60d", "return_120d", "distance_120d_high",
    "leadership_score", "leadership_reason", "long_term_structure_favorable",
    "validation_sources", "strategy_part", "candidate_score",
    "historical_adjustment_check", "candidate_source", "signal_eligible",
    "selected_for_trading", "candidate_failure_reason", "value_falsified",
    "value_falsification_reason",
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


def _validate_required_financial_periods(financial_by_period):
    missing = [
        period for period, rows in sorted(financial_by_period.items())
        if not rows
    ]
    if missing:
        raise RuntimeError(
            "missing financial cache for required point-in-time report periods: "
            + ", ".join(missing)
            + ". Run fundamental_update for these periods before rebuilding backtest candidates."
        )


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


def _candidate_feature_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Precompute rolling candidate features once per symbol."""
    frame = price_frame.copy()
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    frame["_ma20_rising"] = ma20.notna() & ma20.shift(5).notna() & (ma20 > ma20.shift(5))
    frame["_ma60_rising"] = ma60.notna() & ma60.shift(5).notna() & (ma60 > ma60.shift(5))
    frame["_above_ma20"] = ma20.notna() & (close >= ma20)
    frame["_near_ma20"] = ma20.notna() & ma20.gt(0) & (close.div(ma20).sub(1).abs() <= 0.05)
    prior_high = close.shift(1).rolling(21).max()
    frame["_near_breakout"] = prior_high.notna() & prior_high.gt(0) & (close >= prior_high * 0.97)
    volume_base = pd.concat([
        volume.rolling(5).mean().shift(1),
        volume.rolling(10).mean().shift(1),
    ], axis=1).max(axis=1).fillna(0.0)
    frame["_volume_ratio"] = volume.div(volume_base.where(volume_base > 0)).fillna(0.0)
    for period in (5, 10, 20):
        frame[f"_deduction_{period}"] = volume.shift(period).notna() & (
            volume > volume.shift(period)
        )
    for period in (20, 60, 120):
        base = close.shift(period)
        frame[f"_return_{period}"] = close.div(base.where(base > 0)).sub(1)
    rolling_high = close.rolling(120, min_periods=1).max()
    frame["_distance_120d_high"] = close.div(rolling_high.where(rolling_high > 0)).sub(1)
    return frame


def _trade_basis_from_feature_row(row) -> dict:
    ma20_rising = bool(row.get("_ma20_rising", False))
    ma60_rising = bool(row.get("_ma60_rising", False))
    above_ma20 = bool(row.get("_above_ma20", False))
    near_ma20 = bool(row.get("_near_ma20", False))
    near_breakout = bool(row.get("_near_breakout", False))
    volume_ratio = float(row.get("_volume_ratio") or 0.0)
    deduction_periods = [
        period for period in (5, 10, 20)
        if bool(row.get(f"_deduction_{period}", False))
    ]

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
        reasons.append("候选观察：接近21日收盘高点（非买点）")
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


def _trade_basis_snapshot(price_frame: pd.DataFrame, date) -> dict:
    """Score model candidates with only information visible at observation close."""
    history = _candidate_feature_frame(price_frame.loc[:date])
    if history.empty:
        return {
            "trade_basis_score": 0.0,
            "trade_basis_reason": "缺少观察日行情，等待补数",
            "technical_alignment": "missing_price",
        }
    return _trade_basis_from_feature_row(history.iloc[-1])


def _leadership_from_feature_row(row) -> dict:
    def number(value):
        return None if pd.isna(value) else float(value)

    return_20d = number(row.get("_return_20"))
    return_60d = number(row.get("_return_60"))
    return_120d = number(row.get("_return_120"))
    distance_high = number(row.get("_distance_120d_high"))

    def scaled(value, worst: float, best: float, points: float) -> float:
        if value is None:
            return 0.0
        ratio = (float(value) - worst) / (best - worst)
        return min(points, max(0.0, ratio * points))

    score_20 = scaled(return_20d, -0.05, 0.30, 10.0)
    score_60 = scaled(return_60d, 0.00, 0.50, 10.0)
    score_120 = scaled(return_120d, 0.00, 0.60, 6.0)
    high_score = 4.0 if distance_high is not None and distance_high >= -0.05 else (
        2.0 if distance_high is not None and distance_high >= -0.12 else 0.0
    )
    score = score_20 + score_60 + score_120 + high_score
    reasons = []
    for label, value in (("20日", return_20d), ("60日", return_60d), ("120日", return_120d)):
        if value is not None:
            reasons.append(f"{label}强度{value:+.1%}")
    if distance_high is not None:
        reasons.append(f"距120日高点{distance_high:+.1%}")
    return {
        "return_20d": None if return_20d is None else round(return_20d, 6),
        "return_60d": None if return_60d is None else round(return_60d, 6),
        "return_120d": None if return_120d is None else round(return_120d, 6),
        "distance_120d_high": None if distance_high is None else round(distance_high, 6),
        "leadership_score": round(score, 3),
        "leadership_reason": "；".join(reasons),
        "long_term_structure_favorable": bool(score >= 15.0),
    }


def _leadership_snapshot(price_frame: pd.DataFrame, date) -> dict:
    """Rank durable price leadership using only bars visible at observation close."""
    history = _candidate_feature_frame(price_frame.loc[:date])
    if history.empty:
        return {
            "return_20d": None,
            "return_60d": None,
            "return_120d": None,
            "distance_120d_high": None,
            "leadership_score": 0.0,
            "leadership_reason": "缺少观察日行情",
            "long_term_structure_favorable": False,
        }
    return _leadership_from_feature_row(history.iloc[-1])


def _value_falsification_reasons(value_line, quality, yoy, market_cap) -> list[str]:
    reasons = []
    if pd.isna(value_line) or float(value_line) <= 0:
        reasons.append("value_line_missing_or_nonpositive")
    if pd.isna(quality) or float(quality) < 70:
        reasons.append("quality_score_below_70")
    if pd.isna(yoy) or float(yoy) < 0.10:
        reasons.append("earnings_yoy_below_10pct")
    if pd.isna(market_cap) or float(market_cap) < 100:
        reasons.append("mktcap_below_100")
    return reasons


def _value_nonselection_reasons(price_to_value, financial_reasons) -> list[str]:
    reasons = list(financial_reasons)
    if not reasons:
        if price_to_value is None or pd.isna(price_to_value):
            reasons.append("price_to_value_unavailable")
        elif float(price_to_value) < 0.80:
            reasons.append("price_below_value_band_0_80")
        elif float(price_to_value) > 1.08:
            reasons.append("price_above_value_band_1_08")
    return reasons


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
    _validate_required_financial_periods(financial)
    if research_repository is not None:
        research_repository.persist_fundamentals(financial)
    mainline_snapshots = _load_mainline_snapshots(mainline_directory)
    codes = set().union(*(rows.keys() for rows in financial.values()))
    price_start = start - pd.Timedelta(days=420)
    prices = _load_prices(kline_directory, codes, price_start, end)
    prices = {
        code: _candidate_feature_frame(frame)
        for code, frame in prices.items()
    }
    calendar = sorted({
        date.normalize()
        for frame in prices.values()
        for date in frame.index
        if start <= date.normalize() <= end
    })
    snapshots = {}
    tracked_value_codes = set()
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
        leadership_rows = []
        diagnostic_rows = []
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
            full_code = ("sh." if code.startswith(("6", "9")) else "sz.") + code
            price_to_value = (
                None if pd.isna(value_line) or value_line <= 0
                else close / float(value_line)
            )
            value_falsification_reasons = _value_falsification_reasons(
                value_line, quality, yoy, market_cap,
            )
            value_nonselection_reasons = _value_nonselection_reasons(
                price_to_value, value_falsification_reasons,
            )
            base = {
                "date": date.strftime("%Y-%m-%d"),
                "code": full_code,
                "name": names.get(code, code),
                "close": close,
                "value_line": None if pd.isna(value_line) else float(value_line),
                "quality_score": float(quality),
                "earnings_yoy": float(yoy),
                "mktcap": float(market_cap),
                "report_period": period,
                "snapshot_version": SNAPSHOT_VERSION,
                "financial_point_in_time": False,
                "price_to_value": price_to_value,
                "mainline_snapshot_date": None if mainline_date is None else mainline_date.strftime("%Y-%m-%d"),
                "mainline_snapshot_fresh": mainline_fresh,
                "mainline_boards": mainline_members.get(code, ""),
                "selected_for_trading": True,
                "candidate_failure_reason": "",
                "value_falsified": False,
                "value_falsification_reason": "",
            }
            trade_basis = _trade_basis_from_feature_row(market_row)
            base.update(trade_basis)
            leadership = _leadership_from_feature_row(market_row)
            base.update(leadership)
            passes_fundamental_gate = quality >= 70 and yoy >= 0.10 and market_cap >= 100
            if full_code in tracked_value_codes and value_falsification_reasons:
                diagnostic_rows.append({
                    **base,
                    "strategy_part": "value_thesis_failed_diagnostic",
                    "candidate_score": 0.0,
                    "historical_adjustment_check": "financial_falsification",
                    "candidate_source": "value_model",
                    "signal_eligible": False,
                    "selected_for_trading": False,
                    "candidate_failure_reason": (
                        "value_financial_falsification: "
                        + ";".join(value_falsification_reasons)
                    ),
                    "value_falsified": True,
                    "value_falsification_reason": ";".join(value_falsification_reasons),
                    "selection_reason": (
                        "diagnostic row only; value thesis failed current "
                        f"report_period={period}"
                    ),
                })
            if (
                not pd.isna(value_line)
                and value_line > 0
                and 0.80 <= close / value_line <= 1.08
                and passes_fundamental_gate
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
            if code in mainline_members and passes_fundamental_gate:
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
            if (
                passes_fundamental_gate
                and leadership["long_term_structure_favorable"]
                and float(trade_basis["trade_basis_score"]) >= 4.0
            ):
                leadership_rows.append({
                    **base,
                    "strategy_part": "3.右侧强势成长观察",
                    "candidate_score": (
                        float(quality)
                        + min(max(float(yoy), 0.0), 1.0) * 20
                        + float(trade_basis["trade_basis_score"])
                        + float(leadership["leadership_score"])
                    ),
                    "candidate_source": "growth_leadership",
                    "signal_eligible": True,
                    "selection_reason": (
                        "基本面硬条件通过且多周期强度居前；"
                        f"{leadership['leadership_reason']}；"
                        f"{trade_basis['trade_basis_reason']}"
                    ),
                })
        normal_rows.sort(key=lambda item: (-item["candidate_score"], item["code"]))
        by_code = {item["code"]: item for item in value_rows}
        for item in normal_rows + leadership_rows:
            existing = by_code.get(item["code"])
            if existing:
                sources = {
                    source for source in (
                        str(existing.get("candidate_source") or "").split("+")
                        + str(item.get("candidate_source") or "").split("+")
                    ) if source
                }
                existing.update({
                    "strategy_part": " + ".join(dict.fromkeys([
                        str(existing.get("strategy_part") or ""),
                        str(item.get("strategy_part") or ""),
                    ])),
                    "candidate_source": "+".join(sorted(sources)),
                    "signal_eligible": True,
                    "mainline_boards": item["mainline_boards"] or existing.get("mainline_boards", ""),
                    "candidate_score": max(existing["candidate_score"], item["candidate_score"]),
                    "selection_reason": (
                        f"{existing.get('selection_reason', '')}；"
                        f"{item.get('selection_reason', '')}"
                    ).strip("；"),
                })
            else:
                by_code[item["code"]] = item
        pool_rows = list(by_code.values())
        selected = normalize_candidate_snapshots(
            {date.strftime("%Y-%m-%d"): pool_rows}
        )[date.strftime("%Y-%m-%d")]
        selected_codes = {item["code"] for item in selected}
        for item in selected:
            if "value_model" in str(item.get("candidate_source") or "").split("+"):
                tracked_value_codes.add(item["code"])
        for item in pool_rows:
            normalized = normalize_candidate(item)
            if normalized["code"] in selected_codes:
                continue
            diagnostic = dict(normalized)
            diagnostic["signal_eligible"] = False
            diagnostic["selected_for_trading"] = False
            diagnostic["selection_rank"] = None
            diagnostic["candidate_failure_reason"] = (
                "not_selected_for_trading: daily_top10_quota_or_core_reservation"
            )
            if "value_model" in str(diagnostic.get("candidate_source") or "").split("+"):
                price_to_value = diagnostic.get("price_to_value")
                reasons = _value_nonselection_reasons(
                    price_to_value,
                    _value_falsification_reasons(
                        diagnostic.get("value_line"),
                        diagnostic.get("quality_score"),
                        diagnostic.get("earnings_yoy"),
                        diagnostic.get("mktcap"),
                    ),
                )
                diagnostic["candidate_failure_reason"] += (
                    "; value_nonselection=" + ",".join(reasons)
                )
            diagnostic_rows.append(diagnostic)
        selected_by_code = {item["code"]: item for item in selected}
        diagnostics_by_code = {}
        for diagnostic in diagnostic_rows:
            code = diagnostic["code"]
            if code in selected_by_code:
                if diagnostic.get("value_falsified"):
                    selected_by_code[code]["value_falsified"] = True
                    selected_by_code[code]["value_falsification_reason"] = diagnostic.get(
                        "value_falsification_reason", "",
                    )
                    selected_by_code[code]["candidate_failure_reason"] = diagnostic.get(
                        "candidate_failure_reason", "",
                    )
                continue
            existing = diagnostics_by_code.get(code)
            if existing is None or diagnostic.get("value_falsified"):
                diagnostics_by_code[code] = diagnostic
            elif diagnostic.get("candidate_failure_reason"):
                existing_reason = str(existing.get("candidate_failure_reason") or "")
                new_reason = str(diagnostic.get("candidate_failure_reason") or "")
                if new_reason and new_reason not in existing_reason:
                    existing["candidate_failure_reason"] = (
                        f"{existing_reason}; {new_reason}" if existing_reason else new_reason
                    )
        diagnostics = normalize_candidate_snapshots(
            {date.strftime("%Y-%m-%d"): list(diagnostics_by_code.values())},
            include_diagnostics=True,
        )[date.strftime("%Y-%m-%d")]
        snapshots[date.strftime("%Y-%m-%d")] = selected + diagnostics
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
            "leadership": (
                "quality >= 70, yoy >= 0.10, mktcap >= 100, "
                "leadership_score >= 15, and trade_basis_score >= 4"
            ),
            "ranking": (
                "top 10 with at least 5 value/mainline core slots ranked without leadership; "
                "remaining slots use the combined score including 20/60/120-day leadership"
            ),
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
