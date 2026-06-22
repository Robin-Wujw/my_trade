# -*- coding: utf-8 -*-
"""
定点验收回测：
1. 2025 一季报披露后，检查新易盛、中际旭创是否会被因子模型选出。
2. 2025 中报披露后，检查工业富联、洛阳钼业、紫金矿业是否会被因子模型选出。

注意：本脚本用于诊断，不接入每日定时任务。
"""
import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd

from factorStock import (
    CORE_BUCKET,
    DEFAULT_CORE_MIN_SCORE,
    DEFAULT_LOW_VALUE_MIN_SCORE,
    DEFAULT_QUALITY_MIN_SCORE,
    HIGH_QUALITY_BUCKET,
    LOW_VALUE_BUCKET,
    METHOD_NAME,
    VALUE_MIN_MKTCAP,
    WATCH_BUCKET,
    calc_core_score,
    calc_high_quality_score,
    calc_low_value_score,
    classify_method,
    classify_selection_bucket,
    comparable_excl_eps,
    get_effective_valuation_score,
    get_excl_eps_yoy,
    get_value_line_metrics_from_akshare_indicator,
    get_value_line_metrics_from_adata,
    get_pe_pb_metrics,
    infer_excl_eps,
    parse_float,
    score_direct,
    score_inverse,
)


TARGETS = [
    {
        "name": "新易盛",
        "code": "sz.300502",
        "symbol": "300502",
        "report_period": "2025-03-31",
        "disclosure_date": "2025-04-23",
        "target": "2025一季报披露后选到",
    },
    {
        "name": "中际旭创",
        "code": "sz.300308",
        "symbol": "300308",
        "report_period": "2025-03-31",
        "disclosure_date": "2025-04-20",
        "target": "2025一季报披露后选到",
    },
    {
        "name": "工业富联",
        "code": "sh.601138",
        "symbol": "601138",
        "report_period": "2025-06-30",
        "disclosure_date": "2025-08-11",
        "target": "2025中报披露后选到",
    },
    {
        "name": "洛阳钼业",
        "code": "sh.603993",
        "symbol": "603993",
        "report_period": "2025-06-30",
        "disclosure_date": "2025-08-23",
        "target": "2025中报披露后选到",
    },
    {
        "name": "紫金矿业",
        "code": "sh.601899",
        "symbol": "601899",
        "report_period": "2025-06-30",
        "disclosure_date": "2025-08-27",
        "target": "2025中报披露后选到",
    },
]

def parse_yi(value):
    try:
        value = str(value).strip()
        if not value or value.lower() == "nan" or value in {"--", "False"}:
            return None
        if "亿" in value:
            return float(value.replace("亿", "")) * 1e8
        if "万" in value:
            return float(value.replace("万", "")) * 1e4
        return float(value)
    except Exception:
        return None


def parse_pct(value):
    try:
        value = str(value).replace("%", "").strip()
        if not value or value.lower() == "nan" or value == "--":
            return None
        return float(value) / 100
    except Exception:
        return None


def clamp(value, low=0, high=100):
    if value is None or pd.isna(value):
        return 0
    return max(low, min(high, value))


def next_trade_day(date_str):
    start = pd.Timestamp(date_str)
    end = start + pd.DateOffset(days=15)
    rs = bs.query_trade_dates(start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return date_str
    df.columns = rs.fields
    df = df[df["is_trading_day"] == "1"]
    if df.empty:
        return date_str
    return df.iloc[0]["calendar_date"]


def get_history_metrics(code, asof_date):
    start_date = (pd.to_datetime(asof_date) - pd.DateOffset(years=10, days=30)).strftime("%Y-%m-%d")
    fields = "date,close,amount,peTTM,pbMRQ"
    # 与每股财务价值线比较时使用不复权价格，避免前复权把历史节点估值压偏。
    rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=asof_date, frequency="d", adjustflag="3")
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return None
    df.columns = rs.fields
    for col in ["close", "amount", "peTTM", "pbMRQ"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if len(df) < 250:
        return None

    close = df.iloc[-1]["close"]
    close_series = df["close"]
    ma20 = close_series.tail(20).mean()
    ma60 = close_series.tail(60).mean()
    ma120 = close_series.tail(120).mean()
    ret20 = close / close_series.iloc[-21] - 1 if len(df) >= 21 and close_series.iloc[-21] > 0 else None
    ret60 = close / close_series.iloc[-61] - 1 if len(df) >= 61 and close_series.iloc[-61] > 0 else None
    avg_amount20 = df["amount"].tail(20).mean()

    trend_score = 0
    trend_score += 25 if close > ma20 else 0
    trend_score += 25 if ma20 > ma60 else 0
    trend_score += 20 if ma60 > ma120 else 0
    trend_score += score_direct(ret60, -0.10, 0.30) * 0.2
    trend_score += score_direct(ret20, -0.08, 0.18) * 0.1
    if ret20 is not None and ret20 > 0.45:
        trend_score -= 15

    liquidity_score = score_direct(np.log10(avg_amount20) if avg_amount20 and avg_amount20 > 0 else None, 7.0, 9.5)
    return {
        "df": df,
        "close": close,
        "pe": df.iloc[-1]["peTTM"],
        "pb": df.iloc[-1]["pbMRQ"],
        "ret20": ret20,
        "ret60": ret60,
        "trend_score": clamp(trend_score),
        "liquidity_score": liquidity_score,
    }


def get_value_line_asof(symbol, close, report_period):
    ak_indicator_value = get_value_line_metrics_from_akshare_indicator(symbol, close, report_period)
    if ak_indicator_value:
        ak_indicator_value["bvps"] = None
        return ak_indicator_value

    adata_value = get_value_line_metrics_from_adata(symbol, close, report_period)
    if adata_value:
        adata_value["bvps"] = None
        return adata_value

    df_q = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
    df_q = df_q[df_q["扣非净利润"] != False].copy()
    if df_q.empty:
        return None

    report_dt = pd.Timestamp(report_period)
    df_q["报告期_dt"] = pd.to_datetime(df_q["报告期"], errors="coerce")
    df_q = df_q.dropna(subset=["报告期_dt"]).sort_values("报告期_dt")
    df_q = df_q[df_q["报告期_dt"] <= report_dt]
    if df_q.empty:
        return None

    latest = df_q.iloc[-1]
    bvps = parse_yi(latest["每股净资产"])
    yoy_metrics = get_excl_eps_yoy(df_q, latest)
    if bvps is None or not yoy_metrics:
        return None
    yoy = yoy_metrics["yoy"]

    df_annual = df_q[df_q["报告期"].astype(str).str.endswith("12-31")]
    if df_annual.empty:
        return None
    annual = df_annual.iloc[-1]
    net_profit = parse_yi(annual["净利润"])
    basic_eps = parse_float(annual["基本每股收益"])
    raw_eps_excl = infer_excl_eps(annual)
    eps_excl, eps_detail = comparable_excl_eps(symbol, annual["报告期"], latest["报告期"], raw_eps_excl)
    if not net_profit or basic_eps is None or basic_eps <= 0 or eps_excl is None:
        return None

    total_share = net_profit / basic_eps
    if eps_excl <= 0:
        return None

    quality_yoy = min(max(yoy, -0.5), 1.0)
    value_line = bvps + eps_excl * (1 + yoy) * 10
    if value_line <= 0:
        return None

    annual_excl = [parse_yi(r["扣非净利润"]) for _, r in df_annual.tail(3).iterrows()]
    annual_excl = [v for v in annual_excl if v is not None]
    positive_years = sum(1 for v in annual_excl if v > 0)
    growth_steps = sum(1 for i in range(len(annual_excl) - 1) if annual_excl[i + 1] > annual_excl[i])

    price_to_value = close / value_line
    mktcap = close * total_share / 1e8
    valuation_score = score_inverse(price_to_value, best=0.55, worst=1.25)
    quality_score = (
        score_direct(eps_excl, 0.10, 1.50) * 0.35
        + score_direct(quality_yoy, -0.10, 0.50) * 0.35
        + score_direct(positive_years, 1, 3) * 0.15
        + score_direct(growth_steps, 0, 2) * 0.15
    )
    return {
        "value_line": value_line,
        "price_to_value": price_to_value,
        "valuation_score": clamp(valuation_score),
        "quality_score": clamp(quality_score),
        "mktcap": mktcap,
        "bvps": bvps,
        "eps_excl": eps_excl,
        "yoy": yoy,
        "yoy_source": yoy_metrics["yoy_source"],
        "latest_excl_eps": yoy_metrics.get("latest_excl_eps"),
        "prev_excl_eps": yoy_metrics.get("prev_excl_eps"),
        "latest_report": str(latest["报告期"]),
        "annual_report": str(annual["报告期"]),
        "eps_excl_raw": (eps_detail or {}).get("eps_excl_raw"),
        "eps_adjustment_factor": (eps_detail or {}).get("eps_adjustment_factor"),
        "eps_excl_source": (eps_detail or {}).get("eps_excl_source"),
        "eps_bonus_detail": (eps_detail or {}).get("eps_bonus_detail"),
    }


def get_report_year_quarter(report_period):
    dt = pd.Timestamp(report_period)
    if dt.month <= 3:
        quarter = 1
    elif dt.month <= 6:
        quarter = 2
    elif dt.month <= 9:
        quarter = 3
    else:
        quarter = 4
    return dt.year, quarter


def get_profit_metrics_asof(code, report_period):
    year, quarter = get_report_year_quarter(report_period)
    latest = bs.query_profit_data(code=code, year=year, quarter=quarter)
    if latest.error_code != "0" or not latest.next():
        return {"roe": None, "eps": None, "eps_yoy": None, "score": 0}
    latest_df = pd.DataFrame([latest.get_row_data()], columns=latest.fields)

    prev = bs.query_profit_data(code=code, year=year - 1, quarter=quarter)
    prev_df = pd.DataFrame()
    if prev.error_code == "0" and prev.next():
        prev_df = pd.DataFrame([prev.get_row_data()], columns=prev.fields)

    roe = pd.to_numeric(latest_df.iloc[0].get("roeAvg"), errors="coerce")
    eps = pd.to_numeric(latest_df.iloc[0].get("epsTTM"), errors="coerce")
    prev_eps = None
    if not prev_df.empty:
        prev_eps = pd.to_numeric(prev_df.iloc[0].get("epsTTM"), errors="coerce")

    eps_yoy = None
    if prev_eps is not None and pd.notna(prev_eps) and prev_eps > 0 and pd.notna(eps):
        eps_yoy = eps / prev_eps - 1

    score = 0
    score += score_direct(roe, 0.04, 0.18) * 0.55
    score += score_direct(eps_yoy, -0.10, 0.35) * 0.30
    score += score_direct(eps, 0.05, 1.00) * 0.15
    if pd.notna(eps) and eps <= 0:
        score = 0
    return {"roe": roe, "eps": eps, "eps_yoy": eps_yoy, "score": clamp(score)}


def get_industry_map():
    rs = bs.query_stock_industry()
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return {}
    df.columns = rs.fields
    return {row["code"]: row["industry"] for _, row in df.iterrows()}


def build_target_row(target, history, industry):
    method = classify_method(industry)
    close = history["close"]
    valuation_score = 0
    quality_score = 0
    valuation_ref = ""
    extra = {}

    if method == "RIGHT":
        profit = get_profit_metrics_asof(target["code"], target["report_period"])
        quality_score = profit["score"]
        valuation_ref = "轻资产行业不做左侧估值，仅按右侧趋势观察"
        extra = profit
    elif method == "VALUE":
        value = get_value_line_asof(target["symbol"], close, target["report_period"])
        if not value:
            return None, "财务数据不足"
        if value.get("mktcap") is None or value["mktcap"] < VALUE_MIN_MKTCAP:
            return None, f"VALUE市值低于{VALUE_MIN_MKTCAP}亿"
        valuation_score = value["valuation_score"]
        quality_score = value["quality_score"]
        valuation_ref = (
            f"价值线={value['value_line']:.2f}, 现价/价值={value['price_to_value']:.2f}, "
            f"扣非EPS={value.get('eps_excl', 0):.2f}, "
            f"{value.get('yoy_source') or '扣非同比'}={value.get('yoy', 0):.1%}, 市值={value['mktcap']:.1f}亿"
        )
        extra = value
    else:
        field = "peTTM" if method == "PE" else "pbMRQ"
        current = history["pe"] if method == "PE" else history["pb"]
        metrics = get_pe_pb_metrics(history["df"], field, current)
        if not metrics:
            return None, "PE/PB历史估值数据不足"
        profit = get_profit_metrics_asof(target["code"], target["report_period"])
        valuation_score = metrics["valuation_score"]
        quality_score = profit["score"]
        valuation_ref = (
            f"{method}={current:.2f}, 低估均值={metrics['low_avg']:.2f}, "
            f"比值={metrics['ratio']:.2f}, 分位={metrics['percentile']:.0%}"
        )
        extra = metrics
        extra.update(profit)
        extra["current_valuation"] = current

    effective_valuation_score = get_effective_valuation_score(method, valuation_score, extra)
    low_value_score = calc_low_value_score(
        method, effective_valuation_score, quality_score, history["trend_score"], history["liquidity_score"]
    )
    high_quality_score = calc_high_quality_score(
        method, effective_valuation_score, quality_score, history["trend_score"], history["liquidity_score"]
    )
    core_score = calc_core_score(low_value_score, high_quality_score)

    row = {
        "code": target["code"],
        "name": target["name"],
        "industry": industry,
        "method": method,
        "method_name": METHOD_NAME[method],
        "close": close,
        "valuation_score": effective_valuation_score,
        "raw_valuation_score": valuation_score,
        "quality_score": quality_score,
        "trend_score": history["trend_score"],
        "liquidity_score": history["liquidity_score"],
        "low_value_score": low_value_score,
        "high_quality_score": high_quality_score,
        "core_score": core_score,
        "valuation_ref": valuation_ref,
        "value_line": extra.get("value_line"),
        "price_to_value": extra.get("price_to_value"),
        "mktcap": extra.get("mktcap"),
        "current_valuation": extra.get("current_valuation"),
        "low_avg": extra.get("low_avg"),
        "pepb_ratio": extra.get("ratio"),
        "valuation_percentile": extra.get("percentile"),
    }
    row["selection_bucket"], row["valuation_state"] = classify_selection_bucket(row)
    if row["selection_bucket"] == CORE_BUCKET:
        row["total_score"] = row["core_score"]
        row["selection_mode"] = "低估且高质量"
        row["new_threshold"] = DEFAULT_CORE_MIN_SCORE
    elif row["selection_bucket"] == LOW_VALUE_BUCKET:
        row["total_score"] = row["low_value_score"]
        row["selection_mode"] = "低估价值"
        row["new_threshold"] = DEFAULT_LOW_VALUE_MIN_SCORE
    elif row["selection_bucket"] == HIGH_QUALITY_BUCKET:
        row["total_score"] = row["high_quality_score"]
        row["selection_mode"] = "高质量趋势"
        row["new_threshold"] = DEFAULT_QUALITY_MIN_SCORE
    else:
        row["total_score"] = max(row["low_value_score"], row["high_quality_score"])
        row["selection_mode"] = "观察"
        row["new_threshold"] = DEFAULT_QUALITY_MIN_SCORE
    row["selected"] = row["selection_bucket"] != WATCH_BUCKET and row["total_score"] >= row["new_threshold"]
    return row, None


def target_stock_check(target, industry_map):
    asof = next_trade_day(target["disclosure_date"])
    history = get_history_metrics(target["code"], asof)
    if not history:
        return {"target": target, "asof": asof, "error": "行情数据不足"}

    industry = industry_map.get(target["code"], "")
    row, error = build_target_row(target, history, industry)
    if error:
        return {"target": target, "asof": asof, "error": error}

    return {
        "target": target,
        "asof": asof,
        "industry": industry,
        "close": row["close"],
        "method_name": row["method_name"],
        "valuation_ref": row["valuation_ref"],
        "valuation_score": row["valuation_score"],
        "raw_valuation_score": row["raw_valuation_score"],
        "quality_score": row["quality_score"],
        "trend_score": row["trend_score"],
        "liquidity_score": row["liquidity_score"],
        "core_score": row["core_score"],
        "low_value_score": row["low_value_score"],
        "high_quality_score": row["high_quality_score"],
        "total_score": row["total_score"],
        "selection_mode": row["selection_mode"],
        "valuation_state": row["valuation_state"],
        "selection_bucket": row["selection_bucket"],
        "new_threshold": row["new_threshold"],
        "selected": row["selected"],
        "ret20": history["ret20"],
        "ret60": history["ret60"],
    }


def main():
    lg = bs.login()
    if lg.error_code != "0":
        print("baostock登录失败:", lg.error_msg)
        return

    industry_map = get_industry_map()

    print("=== 目标股票验收 ===")
    for target in TARGETS:
        result = target_stock_check(target, industry_map)
        if result.get("error"):
            print(f"{target['name']} | {target['target']} | 失败: {result['error']} | asof={result['asof']}")
            continue
        print(
            f"{target['name']}({target['code']}) | {target['target']} | "
            f"披露日={target['disclosure_date']} | 验证日={result['asof']} | "
            f"行业={result['industry']} | 估值体系={result['method_name']} | "
            f"收盘={result['close']:.2f} | {result['valuation_ref']} | "
            f"估值={result['valuation_score']:.1f}(原始{result['raw_valuation_score']:.1f}) 质量={result['quality_score']:.1f} "
            f"趋势={result['trend_score']:.1f} 流动性={result['liquidity_score']:.1f} | "
            f"核心={result['core_score']:.1f} 低估={result['low_value_score']:.1f} 高质量={result['high_quality_score']:.1f} | "
            f"路径={result['selection_mode']} | 分组={result['selection_bucket']}({result['valuation_state']}) | "
            f"综合分={result['total_score']:.1f}/{result['new_threshold']} | 入选={result['selected']}"
        )

    bs.logout()


if __name__ == "__main__":
    main()
