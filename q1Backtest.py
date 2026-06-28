# -*- coding: utf-8 -*-
"""
2025Q1 截面选股粗回测。

按指定财报期和买入日重建当时可用的因子选股结果，再计算持有到结束日的收益。
这是截面持有回测，不包含调仓、交易成本、涨跌停成交约束。
"""
import argparse
import json
import multiprocessing
import os
from datetime import datetime

import baostock as bs
import akshare as ak
import numpy as np
import pandas as pd

import factorStock
import portfolioSelect
import targetBacktest
from factorStock import (
    AK_TIMEOUT_SECONDS,
    BS_TIMEOUT_SECONDS,
    CORE_BUCKET,
    HIGH_QUALITY_BUCKET,
    LOW_VALUE_BUCKET,
    METHOD_NAME,
    VALUE_MIN_MKTCAP,
    WATCH_BUCKET,
    build_valuation_detail,
    calc_core_score,
    calc_high_quality_score,
    calc_low_value_score,
    classify_method,
    classify_selection_bucket,
    get_effective_valuation_score,
    get_history_metrics,
    get_pe_pb_metrics,
    get_project_path,
    get_score_gate,
    infer_theme,
    pass_score_gate,
    score_direct,
    score_inverse,
    time_limit,
)

try:
    import efinance as ef
except ImportError:
    ef = None


OUTPUT_DIR = get_project_path("回测结果")
Q1_VALUE_CACHE_DIR = get_project_path(".cache/q1_value")
EARNINGS_MAINLINE_BUCKET = "财报后主线候选"
VALUE_LEFT_BUCKET = "价值线左侧确认"
THEME_MOMENTUM_BUCKET = "主题右侧动量"
VALUE_LINE_OVERRIDE_THEMES = {"AI算力/CPO", "半导体/电子", "资源金属"}
THEME_MOMENTUM_THEMES = {"AI算力/CPO", "半导体/电子"}


def format_report_label(report_period):
    try:
        dt = pd.to_datetime(report_period)
    except Exception:
        return str(report_period)
    if dt.month == 12 and dt.day == 31:
        return f"{dt.year}年报"
    if dt.month == 6 and dt.day == 30:
        return f"{dt.year}中报"
    if dt.month == 3 and dt.day == 31:
        return f"{dt.year}Q1"
    if dt.month == 9 and dt.day == 30:
        return f"{dt.year}Q3"
    return dt.strftime("%Y-%m-%d")


def init_worker():
    bs.login()


def parse_args():
    parser = argparse.ArgumentParser(description="2025Q1 截面选股粗回测")
    parser.add_argument("--buy-date", default="2025-05-06", help="买入日期，默认 2025Q1 全部披露后首个交易日")
    parser.add_argument("--report-period", default="2025-03-31", help="财报期")
    parser.add_argument("--end-date", default="2026-05-13", help="收益统计结束日")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=0, help="只展示前N只；0展示全部")
    parser.add_argument("--core-min-score", type=float, default=80)
    parser.add_argument("--low-min-score", type=float, default=75)
    parser.add_argument("--quality-min-score", type=float, default=80)
    parser.add_argument("--value-min-mktcap", type=float, default=VALUE_MIN_MKTCAP)
    parser.add_argument("--offset", type=int, default=0, help="分段续跑用，从截面股票列表第N只开始")
    parser.add_argument("--limit", type=int, default=0, help="调试用，只处理前N只")
    parser.add_argument("--codes", default="", help="只回测指定代码，多个用逗号分隔，如 sz.300502,300308")
    parser.add_argument("--code-prefixes", default="", help="只回测指定代码前缀，多个用逗号分隔，如 sz.300,sh.688")
    parser.add_argument("--maxtasksperchild", type=int, default=120, help="多进程模式下每个worker处理多少任务后重启")
    parser.add_argument("--include-earnings-mainline", action="store_true", help="补充财报后主线候选实验层")
    parser.add_argument("--earnings-mainline-min-score", type=float, default=70)
    parser.add_argument("--include-theme-momentum", action="store_true", help="补充半导体/AI等主题右侧动量实验层")
    parser.add_argument("--theme-momentum-min-score", type=float, default=70)
    parser.add_argument("--portfolio-size", type=int, default=30, help="额外保存最终组合数量；0表示不收敛")
    parser.add_argument(
        "--portfolio-profile",
        choices=sorted(portfolioSelect.PROFILE_CONFIGS),
        default=None,
        help="最终组合收敛风格；默认普通回测 focused，开启主题动量时 theme",
    )
    parser.add_argument("--print-candidates", action="store_true", help="逐只打印宽口径候选")
    return parser.parse_args()


def get_universe(buy_date):
    rs = bs.query_all_stock(day=buy_date)
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return pd.DataFrame()
    df.columns = rs.fields
    mask = (
        df["code"].str.startswith("sh.60")
        | df["code"].str.startswith("sh.68")
        | df["code"].str.startswith("sz.00")
        | df["code"].str.startswith("sz.30")
    )
    df = df[mask & ~df["tradeStatus"].eq("0")]
    df = df[~df["code_name"].str.contains(r"ST|\*ST")]
    return df


def normalize_code(code):
    text = str(code).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower.startswith(("sh.", "sz.")):
        return lower
    if lower.startswith(("sh", "sz")) and len(lower) >= 8:
        return f"{lower[:2]}.{lower[2:8]}"
    if text.startswith(("6", "9")):
        return f"sh.{text}"
    return f"sz.{text}"


def get_industry_map():
    rs = bs.query_stock_industry()
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return {}
    df.columns = rs.fields
    return {row["code"]: row["industry"] for _, row in df.iterrows()}


def make_row(code, name, industry, method, history, valuation_score, quality_score, valuation_ref, extra):
    effective_valuation_score = get_effective_valuation_score(method, valuation_score, extra)
    low_value_score = calc_low_value_score(
        method, effective_valuation_score, quality_score, history["trend_score"], history["liquidity_score"]
    )
    high_quality_score = calc_high_quality_score(
        method, effective_valuation_score, quality_score, history["trend_score"], history["liquidity_score"]
    )
    core_score = calc_core_score(low_value_score, high_quality_score)
    row = {
        "code": code,
        "name": name,
        "industry": industry,
        "theme": infer_theme(name, industry),
        "method": method,
        "method_name": METHOD_NAME[method],
        "close": round(history["close"], 2),
        "valuation_score": round(effective_valuation_score, 1),
        "raw_valuation_score": round(valuation_score, 1),
        "quality_score": round(quality_score, 1),
        "trend_score": round(history["trend_score"], 1),
        "liquidity_score": round(history["liquidity_score"], 1),
        "low_value_score": round(low_value_score, 1),
        "high_quality_score": round(high_quality_score, 1),
        "core_score": round(core_score, 1),
        "valuation_ref": valuation_ref,
        "ret20_at_buy": history["ret20"],
        "ret60_at_buy": history["ret60"],
        "mktcap": extra.get("mktcap"),
        "value_line": extra.get("value_line"),
        "price_to_value": extra.get("price_to_value"),
        "eps_excl": extra.get("eps_excl"),
        "eps_excl_raw": extra.get("eps_excl_raw"),
        "eps_adjustment_factor": extra.get("eps_adjustment_factor"),
        "eps_excl_source": extra.get("eps_excl_source"),
        "current_valuation": extra.get("current_valuation"),
        "low_avg": extra.get("low_avg"),
        "pepb_ratio": extra.get("ratio"),
        "valuation_percentile": extra.get("percentile"),
    }
    row.update(history.get("downtrend_recovery") or {})
    row["selection_bucket"], row["valuation_state"] = classify_selection_bucket(row)
    if row["selection_bucket"] == CORE_BUCKET:
        row["total_score"] = row["core_score"]
        row["selection_mode"] = "低估且高质量"
    elif row["selection_bucket"] == LOW_VALUE_BUCKET:
        row["total_score"] = row["low_value_score"]
        row["selection_mode"] = "低估价值"
    elif row["selection_bucket"] == HIGH_QUALITY_BUCKET:
        row["total_score"] = row["high_quality_score"]
        row["selection_mode"] = "高质量趋势"
    else:
        row["total_score"] = round(max(low_value_score, high_quality_score), 1)
        row["selection_mode"] = "观察"
    return row


def build_earnings_mainline_row(code, name, industry, method, history, value, report_period):
    if not value:
        return None
    ptv = value.get("price_to_value")
    yoy = value.get("yoy")
    quality_score = value.get("quality_score", 0)
    mktcap = value.get("mktcap")
    if any(v is None or pd.isna(v) for v in [ptv, yoy, mktcap]):
        return None

    theme = infer_theme(name, industry)
    max_ptv = 1.45 if theme in {"AI算力/CPO", "半导体/电子", "资源金属"} else 1.25
    if theme == "AI算力/CPO" and yoy >= 0.25 and quality_score >= 80:
        max_ptv = 8.0
    if ptv <= 1.00:
        return None
    if ptv > max_ptv:
        return None
    if mktcap < 100 or quality_score < 70 or history["liquidity_score"] < 55:
        return None
    if yoy < 0.12:
        return None
    valuation_part = score_inverse(ptv, best=0.70, worst=max_ptv)
    growth_part = score_direct(yoy, 0.10, 0.70)
    trend_part = max(
        score_direct(history["ret20"], -0.25, 0.05) if history["ret20"] is not None else 0,
        score_direct(history["ret60"], -0.30, 0.05) if history["ret60"] is not None else 0,
    )
    total_score = (
        valuation_part * 0.20
        + growth_part * 0.25
        + quality_score * 0.30
        + history["liquidity_score"] * 0.15
        + trend_part * 0.10
    )
    row = {
        "code": code,
        "name": name,
        "industry": industry,
        "theme": theme,
        "method": method,
        "method_name": METHOD_NAME[method],
        "close": round(history["close"], 2),
        "valuation_score": round(valuation_part, 1),
        "raw_valuation_score": round(valuation_part, 1),
        "quality_score": round(quality_score, 1),
        "trend_score": round(history["trend_score"], 1),
        "liquidity_score": round(history["liquidity_score"], 1),
        "low_value_score": 0,
        "high_quality_score": 0,
        "core_score": 0,
        "total_score": round(total_score, 1),
        "selection_bucket": EARNINGS_MAINLINE_BUCKET,
        "valuation_state": "价值线附近/业绩强",
        "selection_mode": EARNINGS_MAINLINE_BUCKET,
        "valuation_ref": (
            f"{format_report_label(report_period)}价值线={value['value_line']:.2f}, 现价/价值={ptv:.2f}, "
            f"{value.get('yoy_source') or '扣非同比'}={yoy:.1%}, 市值={mktcap:.1f}亿"
        ),
        "ret20_at_buy": history["ret20"],
        "ret60_at_buy": history["ret60"],
        "mktcap": mktcap,
        "value_line": value.get("value_line"),
        "price_to_value": ptv,
        "eps_excl": value.get("eps_excl"),
        "eps_excl_raw": value.get("eps_excl_raw"),
        "eps_adjustment_factor": value.get("eps_adjustment_factor"),
        "eps_excl_source": value.get("eps_excl_source"),
        "current_valuation": None,
        "low_avg": None,
        "pepb_ratio": None,
        "valuation_percentile": None,
        "earnings_yoy": yoy,
    }
    row.update(history.get("downtrend_recovery") or {})
    return row


def build_value_left_row(code, name, industry, history, value, base_row, report_period):
    if not value:
        return None
    ptv = value.get("price_to_value")
    yoy = value.get("yoy")
    quality_score = value.get("quality_score", 0)
    mktcap = value.get("mktcap")
    if any(v is None or pd.isna(v) for v in [ptv, yoy, mktcap]):
        return None
    if ptv > 1.00 or mktcap < 100 or quality_score < 70 or history["liquidity_score"] < 40:
        return None
    if yoy < 0.05:
        return None
    valuation_floor = max(value.get("valuation_score", 0), 75)
    total_score = valuation_floor * 0.55 + quality_score * 0.35 + history["liquidity_score"] * 0.10
    row = dict(base_row)
    row.update({
        "theme": infer_theme(name, industry),
        "method": "VALUE",
        "method_name": METHOD_NAME["VALUE"],
        "valuation_score": round(valuation_floor, 1),
        "raw_valuation_score": round(value.get("valuation_score", 0), 1),
        "quality_score": round(quality_score, 1),
        "low_value_score": round(total_score, 1),
        "total_score": round(total_score, 1),
        "selection_bucket": VALUE_LEFT_BUCKET,
        "valuation_state": "基本价值线左侧",
        "selection_mode": VALUE_LEFT_BUCKET,
        "valuation_ref": (
            f"{format_report_label(report_period)}价值线={value['value_line']:.2f}, 现价/价值={ptv:.2f}, "
            f"{value.get('yoy_source') or '扣非同比'}={yoy:.1%}, 市值={mktcap:.1f}亿"
        ),
        "mktcap": mktcap,
        "value_line": value.get("value_line"),
        "price_to_value": ptv,
        "eps_excl": value.get("eps_excl"),
        "eps_excl_raw": value.get("eps_excl_raw"),
        "eps_adjustment_factor": value.get("eps_adjustment_factor"),
        "eps_excl_source": value.get("eps_excl_source"),
        "current_valuation": None,
        "low_avg": None,
        "pepb_ratio": None,
        "valuation_percentile": None,
        "earnings_yoy": yoy,
    })
    return row


def build_theme_momentum_row(code, name, industry, method, history, base_row):
    theme = infer_theme(name, industry)
    if theme not in THEME_MOMENTUM_THEMES:
        return None
    mainline = history.get("mainline", {})
    mktcap = base_row.get("mktcap")
    if mktcap is None or pd.isna(mktcap):
        close = history.get("close")
        pb = history.get("pb")
        bvps = base_row.get("bvps")
        if close and pb and pb > 0 and bvps and bvps > 0:
            total_share = close / pb / bvps
            mktcap = close * total_share / 1e8
    if mktcap is None or pd.isna(mktcap) or mktcap < 100:
        return None
    if history["liquidity_score"] < 60 or history["trend_score"] < 78:
        return None
    ret20 = history.get("ret20")
    ret60 = history.get("ret60")
    relative_ret20 = mainline.get("relative_ret20")
    relative_ret60 = mainline.get("relative_ret60")
    volume_ratio = mainline.get("volume_ratio_20_120")
    if ret20 is not None and ret20 < -0.05:
        return None
    if ret60 is not None and ret60 < 0.08:
        return None
    if relative_ret60 is not None and relative_ret60 < 0.05:
        return None
    if volume_ratio is not None and volume_ratio < 0.80:
        return None

    momentum_score = (
        history["trend_score"] * 0.35
        + history["liquidity_score"] * 0.20
        + score_direct(ret60, 0.05, 0.45) * 0.18
        + score_direct(relative_ret60, 0.00, 0.30) * 0.12
        + score_direct(ret20, -0.05, 0.20) * 0.08
        + score_direct(volume_ratio, 0.80, 1.80) * 0.07
    )
    momentum_score = min(momentum_score, 95)
    row = dict(base_row)
    row.update({
        "theme": theme,
        "method": method,
        "method_name": METHOD_NAME.get(method, method),
        "valuation_score": round(max(base_row.get("valuation_score", 0), 50), 1),
        "raw_valuation_score": base_row.get("raw_valuation_score", 0),
        "quality_score": base_row.get("quality_score", 0),
        "trend_score": round(history["trend_score"], 1),
        "liquidity_score": round(history["liquidity_score"], 1),
        "low_value_score": 0,
        "high_quality_score": round(momentum_score, 1),
        "core_score": 0,
        "total_score": round(momentum_score, 1),
        "selection_bucket": THEME_MOMENTUM_BUCKET,
        "valuation_state": "主题右侧确认",
        "selection_mode": THEME_MOMENTUM_BUCKET,
        "valuation_ref": (
            f"{theme}右侧动量, 20/60日涨幅={ret20 or 0:.1%}/{ret60 or 0:.1%}, "
            f"相对60日={relative_ret60 or 0:.1%}, 量能20/120={volume_ratio or 0:.2f}, "
            f"市值={mktcap:.1f}亿"
        ),
        "mktcap": mktcap,
        "theme_momentum_score": round(momentum_score, 1),
        "relative_ret20": relative_ret20,
        "relative_ret60": relative_ret60,
        "volume_ratio_20_120": volume_ratio,
    })
    row.update(history.get("downtrend_recovery") or {})
    return row


def get_history_metrics_retry(code, buy_date, retries=2):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            history = get_history_metrics(code, buy_date)
            if history:
                return history
            last_exc = RuntimeError("history metrics empty")
        except Exception as exc:
            last_exc = exc
        if attempt < retries:
            try:
                bs.logout()
                bs.login()
            except Exception:
                pass
    if last_exc:
        raise last_exc
    return None


def score_task(task):
    (
        code,
        name,
        industry,
        method,
        buy_date,
        report_period,
        value_min_mktcap,
        include_earnings_mainline,
        earnings_mainline_min_score,
        include_theme_momentum,
        theme_momentum_min_score,
    ) = task
    try:
        history = get_history_metrics_retry(code, buy_date)
        if not history:
            return {"row": None, "error": None}

        symbol = code.replace("sh.", "").replace("sz.", "")
        valuation_score = 0
        quality_score = 0
        valuation_ref = ""
        extra = {}
        value_for_mainline = None

        if method == "RIGHT":
            with time_limit(BS_TIMEOUT_SECONDS):
                profit = targetBacktest.get_profit_metrics_asof(code, report_period)
            quality_score = profit["score"]
            valuation_ref = "轻资产行业不做左侧估值，仅按右侧趋势观察"
            extra = profit
        elif method == "VALUE":
            with time_limit(AK_TIMEOUT_SECONDS):
                value = get_value_line_asof_cached(symbol, history["close"], report_period)
            if not value:
                return {"row": None, "error": None}
            value_for_mainline = value
            if value.get("mktcap") is None or value["mktcap"] < value_min_mktcap:
                return {"row": None, "error": None}
            valuation_score = value["valuation_score"]
            quality_score = value["quality_score"]
            valuation_ref = (
                f"价值线={value['value_line']:.2f}, 现价/价值={value['price_to_value']:.2f}, "
                f"扣非EPS={value.get('eps_excl', 0):.2f}, "
                f"{value.get('yoy_source') or '扣非同比'}={value.get('yoy', 0):.1%}, "
                f"市值={value['mktcap']:.1f}亿"
            )
            extra = value
        else:
            field = "peTTM" if method == "PE" else "pbMRQ"
            current = history["pe"] if method == "PE" else history["pb"]
            metrics = get_pe_pb_metrics(history["df"], field, current)
            if not metrics:
                return {"row": None, "error": None}
            with time_limit(BS_TIMEOUT_SECONDS):
                profit = targetBacktest.get_profit_metrics_asof(code, report_period)
            valuation_score = metrics["valuation_score"]
            quality_score = profit["score"]
            valuation_ref = (
                f"{method}={current:.2f}, 低估均值={metrics['low_avg']:.2f}, "
                f"比值={metrics['ratio']:.2f}, 分位={metrics['percentile']:.0%}"
            )
            extra = metrics
            extra.update(profit)
            extra["current_valuation"] = current

        row = make_row(code, name, industry, method, history, valuation_score, quality_score, valuation_ref, extra)
        if include_earnings_mainline or include_theme_momentum:
            theme = infer_theme(name, industry)
            allow_value_line_override = method == "VALUE" or theme in VALUE_LINE_OVERRIDE_THEMES
            should_try_value_line = (
                value_for_mainline is not None
                or allow_value_line_override
            )
            if value_for_mainline is None:
                if should_try_value_line:
                    try:
                        with time_limit(AK_TIMEOUT_SECONDS):
                            value_for_mainline = get_value_line_asof_cached(symbol, history["close"], report_period)
                    except Exception:
                        value_for_mainline = None
            value_left_row = (
                build_value_left_row(code, name, industry, history, value_for_mainline, row, report_period)
                if allow_value_line_override else None
            )
            mainline_row = (
                build_earnings_mainline_row(code, name, industry, method, history, value_for_mainline, report_period)
                if allow_value_line_override else None
            )
            theme_momentum_row = (
                build_theme_momentum_row(code, name, industry, method, history, row)
                if include_theme_momentum else None
            )
            candidates = [row]
            if value_left_row and value_left_row["total_score"] >= 75:
                candidates.append(value_left_row)
            if mainline_row and mainline_row["total_score"] >= earnings_mainline_min_score:
                candidates.append(mainline_row)
            if theme_momentum_row and theme_momentum_row["total_score"] >= theme_momentum_min_score:
                candidates.append(theme_momentum_row)
            row = max(candidates, key=lambda r: r["total_score"])
        return {"row": row, "error": None}
    except Exception as exc:
        return {"row": None, "error": f"{code} {name}: {exc}"}


def q1_value_cache_path(symbol, report_period):
    safe_period = str(report_period).replace("-", "")
    return os.path.join(Q1_VALUE_CACHE_DIR, f"{symbol}_{safe_period}.json")


def get_value_line_asof_cached(symbol, close, report_period):
    path = q1_value_cache_path(symbol, report_period)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            value_line = data.get("value_line")
            total_share = data.get("total_share")
            if value_line and total_share:
                data["price_to_value"] = close / value_line
                data["mktcap"] = close * total_share / 1e8
                data["valuation_score"] = score_inverse(data["price_to_value"], best=0.55, worst=1.25)
                return data
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    value = targetBacktest.get_value_line_asof(symbol, close, report_period)
    if value:
        try:
            os.makedirs(Q1_VALUE_CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(value), f, ensure_ascii=False)
        except OSError:
            pass
    return value


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def close_on(code, date_str, adjustflag):
    rs = bs.query_history_k_data_plus(code, "date,close", start_date=date_str, end_date=date_str, frequency="d", adjustflag=adjustflag)
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return None
    df.columns = rs.fields
    value = pd.to_numeric(df.iloc[0]["close"], errors="coerce")
    return None if pd.isna(value) else float(value)


def close_pair_for_period(code, buy_date, end_date, adjustflag):
    rs = bs.query_history_k_data_plus(
        code,
        "date,close",
        start_date=buy_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return None, None, None, None
    df.columns = rs.fields
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if df.empty:
        return None, None, None, None
    buy_dt = pd.to_datetime(buy_date)
    end_dt = pd.to_datetime(end_date)
    buy_rows = df[df["date"] >= buy_dt]
    end_rows = df[df["date"] <= end_dt]
    if buy_rows.empty or end_rows.empty:
        return None, None, None, None
    buy_row = buy_rows.iloc[0]
    end_row = end_rows.iloc[-1]
    return (
        float(buy_row["close"]),
        float(end_row["close"]),
        buy_row["date"].strftime("%Y-%m-%d"),
        end_row["date"].strftime("%Y-%m-%d"),
    )


def append_forward_return(rows, buy_date, end_date):
    latest_quotes = None
    for row in rows:
        raw_buy, raw_end, raw_buy_date, raw_end_date = close_pair_for_period(row["code"], buy_date, end_date, "3")
        qfq_buy, qfq_end, qfq_buy_date, qfq_end_date = close_pair_for_period(row["code"], buy_date, end_date, "2")
        if raw_end is None and latest_quotes is None:
            latest_quotes = get_latest_quotes_for_rows(rows)
        quote = (latest_quotes or {}).get(row["code"])
        if raw_end is None and quote:
            raw_end = quote.get("latest_price")
            row["end_quote_time"] = quote.get("quote_time")
            row["end_quote_trade_date"] = quote.get("trade_date")
        if qfq_end is None and raw_end is not None and raw_buy and qfq_buy:
            # Front-adjusted prices leave the latest close near the raw close;
            # past buy prices carry subsequent dividends/splits.
            qfq_end = raw_end
        row["buy_close_raw"] = raw_buy
        row["end_close_raw"] = raw_end
        row["buy_close_qfq"] = qfq_buy
        row["end_close_qfq"] = qfq_end
        row["buy_trade_date"] = qfq_buy_date or raw_buy_date
        row["end_trade_date"] = qfq_end_date or raw_end_date
        row["raw_return"] = raw_end / raw_buy - 1 if raw_buy and raw_end else None
        row["qfq_return"] = qfq_end / qfq_buy - 1 if qfq_buy and qfq_end else None
    return rows


def get_latest_quotes_for_rows(rows):
    codes = [row["code"].replace("sh.", "").replace("sz.", "") for row in rows]
    if not codes:
        return {}
    if ef is None:
        return {}
    quotes = {}
    for start in range(0, len(codes), 80):
        batch = codes[start:start + 80]
        try:
            df = ef.stock.get_latest_quote(batch)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for _, quote in df.iterrows():
            code = normalize_code(str(quote.get("代码")).zfill(6))
            latest = pd.to_numeric(quote.get("最新价"), errors="coerce")
            if pd.isna(latest) or latest <= 0:
                continue
            quotes[code] = {
                "latest_price": float(latest),
                "quote_time": str(quote.get("更新时间")),
                "trade_date": str(quote.get("最新交易日")),
            }
    needed = {row["code"] for row in rows}
    missing = needed - set(quotes)
    if not missing:
        return quotes
    try:
        df = ak.stock_zh_a_spot()
    except Exception:
        return quotes
    if df is None or df.empty:
        return quotes
    for _, quote in df.iterrows():
        code = normalize_code(str(quote.get("代码")))
        if code not in missing:
            continue
        latest = pd.to_numeric(quote.get("最新价"), errors="coerce")
        if pd.isna(latest) or latest <= 0:
            continue
        quotes[code] = {
            "latest_price": float(latest),
            "quote_time": str(quote.get("时间戳")),
            "trade_date": "",
        }
    return quotes


def save_csv(rows, buy_date, end_date):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"q1_backtest_{buy_date}_{end_date}_{datetime.now().strftime('%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_portfolio_csv(rows, source_path, size, profile):
    if size <= 0 or not rows:
        return None, pd.DataFrame()
    selected = portfolioSelect.select_portfolio(pd.DataFrame(rows), size=size, profile=profile)
    root, ext = os.path.splitext(source_path)
    out = f"{root}_portfolio_{profile}_{size}{ext}"
    selected.to_csv(out, index=False, encoding="utf-8-sig")
    return out, selected


def valuation_detail_for_display(row):
    if row.get("selection_bucket") in {EARNINGS_MAINLINE_BUCKET, VALUE_LEFT_BUCKET}:
        return row.get("valuation_ref", "")
    if row.get("selection_bucket") == THEME_MOMENTUM_BUCKET:
        return row.get("valuation_ref", "")
    return build_valuation_detail(row)


def main():
    args = parse_args()
    if args.portfolio_profile is None:
        args.portfolio_profile = "theme" if args.include_theme_momentum else "right_side"
    lg = bs.login()
    if lg.error_code != "0":
        print("baostock登录失败:", lg.error_msg)
        return
    universe = get_universe(args.buy_date)
    industry_map = get_industry_map()
    factorStock.BENCHMARK_DF = factorStock.get_benchmark_history(args.buy_date)
    bs.logout()

    if args.codes:
        selected_codes = {normalize_code(code) for code in args.codes.split(",")}
        universe = universe[universe["code"].isin(selected_codes)].reset_index(drop=True)
    if args.code_prefixes:
        prefixes = tuple(prefix.strip().lower() for prefix in args.code_prefixes.split(",") if prefix.strip())
        universe = universe[universe["code"].str.lower().str.startswith(prefixes)].reset_index(drop=True)
    if args.offset:
        universe = universe.iloc[args.offset:].reset_index(drop=True)
    if args.limit:
        universe = universe.head(args.limit)

    tasks = []
    for _, stock in universe.iterrows():
        code, name = stock["code"], stock["code_name"]
        industry = industry_map.get(code, "")
        method = classify_method(industry)
        tasks.append((
            code,
            name,
            industry,
            method,
            args.buy_date,
            args.report_period,
            args.value_min_mktcap,
            args.include_earnings_mainline,
            args.earnings_mainline_min_score,
            args.include_theme_momentum,
            args.theme_momentum_min_score,
        ))

    print(
        f"回测截面: {args.buy_date} | 财报期: {args.report_period} | 结束日: {args.end_date} | "
        f"候选: {len(tasks)} | offset={args.offset} | limit={args.limit} | workers={args.workers}"
    )

    rows = []
    errors = 0
    if args.workers <= 1:
        bs.login()
        iterator = map(score_task, tasks)
    else:
        pool = multiprocessing.Pool(
            processes=max(1, args.workers),
            initializer=init_worker,
            maxtasksperchild=args.maxtasksperchild if args.maxtasksperchild > 0 else None,
        )
        iterator = pool.imap_unordered(score_task, tasks)

    try:
        for idx, result in enumerate(iterator, start=1):
            if result.get("error"):
                errors += 1
                if errors <= 20:
                    print("  处理失败:", result["error"])
            row = result.get("row")
            selected = False
            if row:
                selected = pass_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                selected = selected or (
                    row.get("selection_bucket") == EARNINGS_MAINLINE_BUCKET
                    and row["total_score"] >= args.earnings_mainline_min_score
                ) or (
                    row.get("selection_bucket") == THEME_MOMENTUM_BUCKET
                    and row["total_score"] >= args.theme_momentum_min_score
                ) or (
                    row.get("selection_bucket") == VALUE_LEFT_BUCKET
                    and row["total_score"] >= args.low_min_score
                )
            if row and selected:
                rows.append(row)
                gate = (
                    args.earnings_mainline_min_score
                    if row.get("selection_bucket") == EARNINGS_MAINLINE_BUCKET
                    else args.theme_momentum_min_score
                    if row.get("selection_bucket") == THEME_MOMENTUM_BUCKET
                    else args.low_min_score
                    if row.get("selection_bucket") == VALUE_LEFT_BUCKET
                    else get_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                )
                if args.print_candidates:
                    print(f"  候选 {row['code']} {row['name']} | {row['selection_bucket']} | 分={row['total_score']}/{gate} | {row['valuation_ref']}")
            if idx % 25 == 0:
                print(f"  进度: {idx}/{len(tasks)}, 候选 {len(rows)}, 失败 {errors}")
    finally:
        if args.workers <= 1:
            bs.logout()
        else:
            pool.close()
            pool.join()

    rows.sort(key=lambda r: r["total_score"], reverse=True)
    bs.login()
    rows = append_forward_return(rows, args.buy_date, args.end_date)
    bs.logout()

    path = save_csv(rows, args.buy_date, args.end_date)
    portfolio_path, portfolio_df = save_portfolio_csv(rows, path, args.portfolio_size, args.portfolio_profile)
    df = pd.DataFrame(rows)
    print(f"\n候选池已保存: {path}")
    print(f"候选 {len(rows)} 只，失败 {errors} 只")
    if df.empty:
        return
    print(df["selection_bucket"].value_counts().to_string())
    print(
        "收益统计(qfq): "
        f"均值={df['qfq_return'].mean():.1%}, 中位数={df['qfq_return'].median():.1%}, "
        f"胜率={(df['qfq_return'] > 0).mean():.1%}, 最大={df['qfq_return'].max():.1%}, 最小={df['qfq_return'].min():.1%}"
    )
    if portfolio_path:
        print(f"最终组合已保存: {portfolio_path}")
        if not portfolio_df.empty:
            print(
                "最终组合收益(qfq): "
                f"数量={len(portfolio_df)}, 均值={portfolio_df['qfq_return'].mean():.1%}, "
                f"中位数={portfolio_df['qfq_return'].median():.1%}, "
                f"胜率={(portfolio_df['qfq_return'] > 0).mean():.1%}"
            )
            print(portfolio_df["selection_bucket"].value_counts().to_string())
    display_source = portfolio_df if portfolio_path and not portfolio_df.empty else df
    display = display_source.head(args.top if args.top else len(display_source)).copy()
    display["valuation_detail"] = display.apply(valuation_detail_for_display, axis=1)
    cols = [
        col for col in [
            "final_rank", "code", "name", "selection_bucket", "method_name", "selection_mode",
            "total_score", "portfolio_score", "buy_close_raw", "end_close_raw", "qfq_return",
            "valuation_detail",
        ] if col in display.columns
    ]
    print(display[cols].to_string(index=False, formatters={"qfq_return": lambda v: f"{v:.1%}"}))


if __name__ == "__main__":
    main()
