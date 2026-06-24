# -*- coding: utf-8 -*-
"""
因子模型选股主入口：融合基本价值线、PE/PB历史低估、质量、趋势、流动性。

输出拆分为“低估且高质量”“低估价值”“高质量趋势”三组，旧选股脚本不再参与定时任务。
"""
import argparse
import contextlib
import json
import multiprocessing
import os
import signal
import sys
from datetime import datetime

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd

from trade_utils import build_diff_html, get_project_path, load_last_result, save_current_result, send_pushplus

try:
    import adata
except ImportError:
    adata = None


LAST_RESULT_FILE = get_project_path(".factorStock_last.json")
OUTPUT_DIR = get_project_path("选股结果")
VALUE_CACHE_DIR = get_project_path(".cache/factor_value")
AK_TIMEOUT_SECONDS = 12
BS_TIMEOUT_SECONDS = 20
VALUE_CACHE_TTL_SECONDS = 24 * 60 * 60
VALUE_CACHE_VERSION = 7
BENCHMARK_CODE = "sh.000001"
BENCHMARK_NAME = "上证指数"
VALUE_MIN_MKTCAP = 100
DEFAULT_MIN_SCORE = 80
DEFAULT_VALUE_MIN_SCORE = 80
DEFAULT_QUALITY_MIN_SCORE = DEFAULT_MIN_SCORE
DEFAULT_LOW_VALUE_MIN_SCORE = 75
DEFAULT_CORE_MIN_SCORE = 80
DEFAULT_DIAGNOSTIC_TOP = 30
DEFAULT_VALUE_WATCH_TOP = 20
VALUE_UNDERVALUED_RATIO = 0.85
VALUE_LOW_VALUE_RATIO = 1.00
VALUE_REASONABLE_RATIO = 1.10
VALUE_WATCH_RATIO = 1.08
VALUE_WATCH_MIN_QUALITY_SCORE = 55
VALUE_QUALITY_MAX_RATIO = 1.80
PEPB_REASONABLE_RATIO = 1.10
PEPB_REASONABLE_PERCENTILE = 0.25
PEPB_QUALITY_MAX_RATIO = 1.70
PEPB_QUALITY_MAX_PERCENTILE = 0.85
QUALITY_MIN_SCORE = 70
PEPB_QUALITY_MIN_SCORE = 65
QUALITY_MIN_TREND_SCORE = 60
RIGHT_MIN_TREND_SCORE = 70
QUALITY_MIN_LIQUIDITY_SCORE = 40

LOW_VALUE_BUCKET = "低估价值"
HIGH_QUALITY_BUCKET = "高质量趋势"
CORE_BUCKET = "低估且高质量"
WATCH_BUCKET = "观察池"

BENCHMARK_DF = None
BONUS_DF_CACHE = {}

PE_KEYWORDS = [
    "货币金融", "银行", "保险", "酒、饮料和精制茶", "食品制造", "批发", "零售", "医药制造", "纺织服装",
]

PB_KEYWORDS = [
    "钢铁", "煤炭", "有色", "化学原料", "化学纤维", "建材", "石油", "采矿",
    "非金属矿", "黑色金属", "燃料加工", "电信", "广播电视和卫星", "房地产",
    "电力", "热力", "燃气", "水的生产", "资本市场", "其他金融",
    "建筑", "土木工程", "交通运输", "道路运输", "铁路运输", "水上运输", "航空运输",
    "仓储", "邮政", "农业", "林业", "畜牧", "渔业",
]

RIGHT_SIDE_KEYWORDS = [
    "软件", "互联网", "游戏", "信息技术服务",
]

AI_CPO_NAMES = {
    "新易盛", "中际旭创", "天孚通信", "太辰光", "光迅科技", "剑桥科技", "联特科技", "华工科技",
    "源杰科技", "仕佳光子", "工业富联", "沪电股份", "胜宏科技", "生益科技", "深南电路",
    "寒武纪", "海光信息", "中科曙光", "浪潮信息",
}

SEMICONDUCTOR_KEYWORDS = [
    "半导体", "芯", "晶", "硅", "微", "集成", "封测", "电子",
]

RESOURCE_KEYWORDS = [
    "有色", "铜", "铝", "钼", "金", "矿", "稀土", "锂", "钴", "镍",
]

METHOD_NAME = {
    "VALUE": "基本价值线",
    "PE": "PE低估",
    "PB": "PB低估",
    "RIGHT": "右侧趋势",
}


class DataFetchTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle_timeout(signum, frame):
        raise DataFetchTimeout(f"数据请求超过 {seconds} 秒")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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


def parse_float(value):
    try:
        value = str(value).replace(",", "").strip()
        if not value or value.lower() == "nan" or value in {"--", "False"}:
            return None
        return float(value)
    except Exception:
        return None


def first_existing_value(row, names):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None and str(value).strip() not in {"", "--", "False", "nan"}:
                return value
    return None


def pure_stock_code(symbol):
    text = str(symbol)
    if "." not in text:
        return text
    left, right = text.split(".", 1)
    if left.lower() in {"sh", "sz", "bj"}:
        return right
    return left


def parse_report_date(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(dt) else pd.Timestamp(dt)
    except Exception:
        return None


def get_bonus_adjustment_factor(symbol, annual_report_date, latest_report_date):
    """Return the comparable-share adjustment for Q1 value-line EPS.

    Annual reports disclose per-share earnings on the report's weighted-share
    base. When the same annual plan includes stock bonus/conversion, the recap
    examples use a month-weighted comparable EPS for the following Q1 node.
    """
    annual_dt = parse_report_date(annual_report_date)
    latest_dt = parse_report_date(latest_report_date)
    if annual_dt is None or latest_dt is None:
        return 1.0, None
    if latest_dt.year != annual_dt.year + 1 or latest_dt.month != 3:
        return 1.0, None

    rows = get_bonus_rows(symbol, annual_dt)
    if rows.empty:
        return 1.0, None
    row = rows.iloc[-1]

    total_ratio = parse_float(row.get("送转股份-送转总比例"))
    if total_ratio is None:
        bonus_ratio = parse_float(row.get("送转股份-送股比例")) or 0
        transfer_ratio = parse_float(row.get("送转股份-转股比例")) or 0
        total_ratio = bonus_ratio + transfer_ratio
    if total_ratio is None or total_ratio <= 0:
        return 1.0, None

    ex_date = parse_report_date(row.get("除权除息日"))
    if ex_date is None or ex_date.year != latest_dt.year:
        return 1.0, None

    post_months = 12 - ex_date.month + 1
    if post_months <= 0:
        return 1.0, None
    factor = 1 + (total_ratio / 10) * (post_months / 12)
    if factor <= 1:
        return 1.0, None
    return factor, {
        "bonus_ratio_per_10": total_ratio,
        "ex_right_date": ex_date.strftime("%Y-%m-%d"),
        "post_months": post_months,
    }


def get_bonus_rows(symbol, annual_dt):
    code = pure_stock_code(symbol)
    date_key = annual_dt.strftime("%Y%m%d")
    rows = get_bonus_rows_from_bulk(code, date_key)
    if not rows.empty:
        return rows
    return get_bonus_rows_from_detail(code, annual_dt)


def get_bonus_rows_from_bulk(code, date_key):
    cache_key = f"bulk:{date_key}"
    if cache_key not in BONUS_DF_CACHE:
        try:
            with time_limit(AK_TIMEOUT_SECONDS):
                df_bonus = ak.stock_fhps_em(date=date_key)
        except Exception:
            df_bonus = pd.DataFrame()
        if df_bonus is None:
            df_bonus = pd.DataFrame()
        BONUS_DF_CACHE[cache_key] = df_bonus.copy()

    df_bonus = BONUS_DF_CACHE.get(cache_key, pd.DataFrame())
    if df_bonus.empty or "代码" not in df_bonus.columns:
        return pd.DataFrame()
    rows = df_bonus[df_bonus["代码"].astype(str).str.zfill(6) == str(code).zfill(6)]
    return rows.copy()


def get_bonus_rows_from_detail(code, annual_dt):
    cache_key = f"detail:{code}"
    if cache_key not in BONUS_DF_CACHE:
        try:
            with time_limit(AK_TIMEOUT_SECONDS):
                df_bonus = ak.stock_fhps_detail_em(symbol=code)
        except Exception:
            df_bonus = pd.DataFrame()
        if df_bonus is None:
            df_bonus = pd.DataFrame()
        BONUS_DF_CACHE[cache_key] = df_bonus.copy()

    df_bonus = BONUS_DF_CACHE.get(cache_key, pd.DataFrame())
    if df_bonus.empty or "报告期" not in df_bonus.columns:
        return pd.DataFrame()
    df_bonus = df_bonus.copy()
    df_bonus["报告期_dt"] = pd.to_datetime(df_bonus["报告期"], errors="coerce")
    return df_bonus[df_bonus["报告期_dt"] == annual_dt.normalize()].copy()


def comparable_excl_eps(symbol, annual_report_date, latest_report_date, raw_eps):
    raw_eps = parse_float(raw_eps)
    if raw_eps is None or raw_eps <= 0:
        return None, None

    factor, bonus = get_bonus_adjustment_factor(symbol, annual_report_date, latest_report_date)
    eps = raw_eps / factor
    if bonus:
        source = (
            f"年报扣非EPS送转可比口径(原始{raw_eps:.2f}/"
            f"月度加权{factor:.4f})"
        )
    else:
        source = "年报扣非EPS"
    return eps, {
        "eps_excl_raw": raw_eps,
        "eps_adjustment_factor": factor,
        "eps_excl_source": source,
        "eps_bonus_detail": bonus,
    }


def infer_excl_eps(row):
    """Return non-recurring EPS, preferring direct fields and otherwise inferring it."""
    direct_value = first_existing_value(row, [
        "扣非每股收益",
        "扣除非经常性损益后的基本每股收益",
        "扣除非经常性损益后的每股收益",
        "扣非基本每股收益",
    ])
    direct_eps = parse_float(direct_value)
    if direct_eps is not None:
        return direct_eps

    excl_profit = parse_yi(row.get("扣非净利润"))
    net_profit = parse_yi(row.get("净利润"))
    basic_eps = parse_float(row.get("基本每股收益"))
    if not excl_profit or not net_profit or basic_eps is None or basic_eps <= 0:
        return None
    total_share = net_profit / basic_eps
    if total_share <= 0:
        return None
    return excl_profit / total_share


def get_excl_eps_yoy(df_q, latest):
    """Get the latest report-period non-recurring EPS YoY growth.

    The formula's accurate input is EPS growth. Some data sources only expose
    non-recurring profit growth, so that is kept as the last fallback.
    """
    direct_yoy_value = first_existing_value(latest, [
        "扣非每股收益同比增长率",
        "扣除非经常性损益后的基本每股收益同比增长率",
        "扣除非经常性损益后的每股收益同比增长率",
        "扣非基本每股收益同比增长率",
    ])
    direct_yoy = parse_pct(direct_yoy_value)
    if direct_yoy is not None:
        return {
            "yoy": direct_yoy,
            "yoy_source": "扣非每股收益同比",
            "latest_excl_eps": infer_excl_eps(latest),
            "prev_excl_eps": None,
        }

    latest_dt = latest.get("报告期_dt")
    if latest_dt is not None and pd.notna(latest_dt):
        prev_dt = pd.Timestamp(latest_dt) - pd.DateOffset(years=1)
        prev_rows = df_q[df_q["报告期_dt"] == prev_dt]
        if not prev_rows.empty:
            latest_eps = infer_excl_eps(latest)
            prev_eps = infer_excl_eps(prev_rows.iloc[-1])
            if latest_eps is not None and prev_eps is not None and prev_eps > 0:
                return {
                    "yoy": latest_eps / prev_eps - 1,
                    "yoy_source": "反推扣非EPS同比",
                    "latest_excl_eps": latest_eps,
                    "prev_excl_eps": prev_eps,
                }

    fallback_yoy = parse_pct(latest.get("扣非净利润同比增长率"))
    if fallback_yoy is not None:
        return {
            "yoy": fallback_yoy,
            "yoy_source": "扣非净利润同比",
            "latest_excl_eps": infer_excl_eps(latest),
            "prev_excl_eps": None,
        }
    return None


def clamp(value, low=0, high=100):
    if value is None or pd.isna(value):
        return 0
    return max(low, min(high, value))


def score_direct(value, worst, best):
    if value is None or pd.isna(value):
        return 0
    if best == worst:
        return 0
    return clamp((value - worst) / (best - worst) * 100)


def score_inverse(value, best, worst):
    if value is None or pd.isna(value):
        return 0
    if best == worst:
        return 0
    return clamp((worst - value) / (worst - best) * 100)


def remove_outliers(values):
    if len(values) < 3:
        return values
    median = np.median(values)
    if median <= 0:
        return values
    filtered = [v for v in values if 0.5 * median <= v <= 2.0 * median]
    return filtered if len(filtered) >= 3 else values


def infer_theme(name, industry):
    text = f"{name}{industry}"
    industry_text = str(industry)
    if name in AI_CPO_NAMES:
        return "AI算力/CPO"
    if industry_text.startswith("J") or any(keyword in text for keyword in ["银行", "保险", "证券", "金融", "资本市场"]):
        return "金融"
    if any(keyword in text for keyword in RESOURCE_KEYWORDS):
        return "资源金属"
    if any(keyword in text for keyword in SEMICONDUCTOR_KEYWORDS):
        return "半导体/电子"
    if any(keyword in text for keyword in ["医药", "生物", "医疗", "药"]):
        return "医药医疗"
    if any(keyword in text for keyword in ["食品", "饮料", "酒", "消费"]):
        return "消费"
    return str(industry).split(" ", 1)[0] if industry else "未分类"


def value_cache_path(symbol):
    return os.path.join(VALUE_CACHE_DIR, f"{symbol}.json")


def load_value_cache(symbol):
    path = value_cache_path(symbol)
    try:
        if not os.path.exists(path):
            return None
        if os.path.getmtime(path) < datetime.now().timestamp() - VALUE_CACHE_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_value_cache(symbol, data):
    try:
        os.makedirs(VALUE_CACHE_DIR, exist_ok=True)
        with open(value_cache_path(symbol), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


def cn_sma(series, n=3, m=1, initial=50):
    values = []
    prev = initial
    for value in series:
        if pd.isna(value):
            values.append(prev)
            continue
        prev = (m * value + (n - m) * prev) / n
        values.append(prev)
    return pd.Series(values, index=series.index)


def calc_kd_lines(df):
    high9 = df["high"].rolling(9, min_periods=1).max()
    low9 = df["low"].rolling(9, min_periods=1).min()
    denom = (high9 - low9).replace(0, np.nan)

    close_rsv = ((df["close"] - low9) / denom * 100).clip(0, 100).fillna(50)
    close_k = cn_sma(close_rsv, 3, 1)
    close_d = cn_sma(close_k, 3, 1)

    prev_k = close_k.shift(1).fillna(50)
    prev_d = close_d.shift(1).fillna(50)

    high_rsv = ((df["high"] - low9) / denom * 100).clip(0, 100).fillna(50)
    high_k = prev_k * (2 / 3) + high_rsv / 3
    high_d = prev_d * (2 / 3) + high_k / 3

    low_rsv = ((df["low"] - low9) / denom * 100).clip(0, 100).fillna(50)
    low_k = prev_k * (2 / 3) + low_rsv / 3
    low_d = prev_d * (2 / 3) + low_k / 3

    return pd.DataFrame({
        "close_k": close_k,
        "close_d": close_d,
        "high_k": high_k,
        "high_d": high_d,
        "low_k": low_k,
        "low_d": low_d,
    }, index=df.index)


def calc_rsi_999(close):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 999, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 999, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def detect_kd_divergence(df, kd, lookback=120, recent_days=5):
    if len(df) < 20:
        return []

    start_pos = max(0, len(df) - lookback)
    cycle = kd.iloc[start_pos:-1]
    reset_mask = (cycle["close_k"] < 20) & (cycle["close_d"] < 20)
    if reset_mask.any():
        start_pos = cycle[reset_mask].index[-1] + 1

    segment = df.iloc[start_pos:].copy()
    kd_segment = kd.iloc[start_pos:]
    flags = []
    if len(segment) < 10:
        return flags

    recent_index = set(segment.tail(recent_days).index)

    top_idx = segment["high"].idxmax()
    if top_idx in recent_index:
        before = segment.loc[:top_idx].iloc[:-1]
        if len(before) >= 9:
            prev_idx = before["high"].idxmax()
            if (
                segment.loc[top_idx, "high"] > segment.loc[prev_idx, "high"]
                and kd_segment.loc[top_idx, "high_k"] < kd_segment.loc[prev_idx, "high_k"]
            ):
                d_text = "，D同步背离" if kd_segment.loc[top_idx, "high_d"] < kd_segment.loc[prev_idx, "high_d"] else ""
                flags.append(f"KD顶背离警讯(K {kd_segment.loc[top_idx, 'high_k']:.1f}<{kd_segment.loc[prev_idx, 'high_k']:.1f}{d_text})")

    bottom_idx = segment["low"].idxmin()
    if bottom_idx in recent_index:
        before = segment.loc[:bottom_idx].iloc[:-1]
        if len(before) >= 9:
            prev_idx = before["low"].idxmin()
            if (
                segment.loc[bottom_idx, "low"] < segment.loc[prev_idx, "low"]
                and kd_segment.loc[bottom_idx, "low_k"] > kd_segment.loc[prev_idx, "low_k"]
            ):
                d_text = "，D同步背离" if kd_segment.loc[bottom_idx, "low_d"] > kd_segment.loc[prev_idx, "low_d"] else ""
                flags.append(f"KD底背离修复观察(K {kd_segment.loc[bottom_idx, 'low_k']:.1f}>{kd_segment.loc[prev_idx, 'low_k']:.1f}{d_text})")

    return flags


def get_technical_metrics(df):
    flags = []
    if len(df) < 20 or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return {
            "technical_flags": "技术数据不足",
            "technical_flags_list": ["技术数据不足"],
            "technical_ref": "技术数据不足",
            "kd_k": None,
            "kd_d": None,
            "rsi999": None,
        }

    kd = calc_kd_lines(df)
    latest_k = float(kd.iloc[-1]["close_k"])
    latest_d = float(kd.iloc[-1]["close_d"])
    kd_gap = latest_k - latest_d
    if kd_gap >= 20:
        flags.append(f"KD开口{kd_gap:.1f}>=20，短线有收敛压力")
    elif kd_gap <= -20:
        flags.append(f"KD开口{kd_gap:.1f}<=-20，短线有修复机会")

    flags.extend(detect_kd_divergence(df, kd))

    rsi = calc_rsi_999(df["close"])
    latest_rsi = float(rsi.iloc[-1])
    if latest_rsi >= 70:
        flags.append(f"RSI999={latest_rsi:.1f}，长期超买")
    elif latest_rsi <= 30:
        flags.append(f"RSI999={latest_rsi:.1f}，长期超卖")

    close = float(df.iloc[-1]["close"])
    ene_mid = float(df["close"].tail(10).mean())
    ene_upper = ene_mid * 1.11
    ene_lower = ene_mid * 0.91
    if close >= ene_upper:
        flags.append(f"ENE上轨附近/突破({close / ene_upper:.2f})，短线易震荡")
    elif close <= ene_lower:
        flags.append(f"ENE下轨附近/跌破({close / ene_lower:.2f})，短线有修复观察")

    ret20 = close / df["close"].iloc[-21] - 1 if len(df) >= 21 and df["close"].iloc[-21] > 0 else None
    ret60 = close / df["close"].iloc[-61] - 1 if len(df) >= 61 and df["close"].iloc[-61] > 0 else None
    if ret20 is not None:
        if ret20 > 0.45:
            flags.append(f"20日涨幅{ret20:.0%}过高")
        elif ret20 > 0.30:
            flags.append(f"20日涨幅{ret20:.0%}偏高")
    if ret60 is not None and ret60 > 0.80:
        flags.append(f"60日涨幅{ret60:.0%}过高")

    volume = df["volume"]
    latest_volume = float(volume.iloc[-1])
    vol_ma5 = float(volume.tail(5).mean())
    vol_ma10 = float(volume.tail(10).mean())
    deduct_vol5 = float(volume.iloc[-6]) if len(volume) >= 6 else np.nan
    deduct_vol10 = float(volume.iloc[-11]) if len(volume) >= 11 else np.nan
    vol_baseline_ok = (
        latest_volume > vol_ma5
        and latest_volume > vol_ma10
        and (pd.isna(deduct_vol5) or latest_volume > deduct_vol5)
        and (pd.isna(deduct_vol10) or latest_volume > deduct_vol10)
    )
    if not vol_baseline_ok:
        flags.append("5/10日上涨基准量不足")

    price_deduct_ok = True
    if len(df) >= 6 and close <= df["close"].iloc[-6]:
        price_deduct_ok = False
    if len(df) >= 11 and close <= df["close"].iloc[-11]:
        price_deduct_ok = False
    if not price_deduct_ok:
        flags.append("5/10日扣抵价未完全站上")

    ene_pos = "上轨" if close >= ene_upper else ("下轨" if close <= ene_lower else "轨道内")
    baseline_text = "满足" if vol_baseline_ok else "不足"
    technical_ref = (
        f"KD={latest_k:.1f}/{latest_d:.1f}, RSI999={latest_rsi:.1f}, "
        f"ENE={ene_pos}, 5/10基准量={baseline_text}"
    )
    return {
        "technical_flags": "、".join(flags) if flags else "正常",
        "technical_flags_list": flags,
        "technical_ref": technical_ref,
        "kd_k": latest_k,
        "kd_d": latest_d,
        "rsi999": latest_rsi,
    }


def get_latest_quarter(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    if d.month <= 4:
        return d.year - 1, 4
    if d.month <= 8:
        return d.year, 1
    if d.month <= 10:
        return d.year, 2
    return d.year, 3


def classify_method(industry):
    industry = str(industry)
    if any(keyword in industry for keyword in RIGHT_SIDE_KEYWORDS):
        return "RIGHT"
    if any(keyword in industry for keyword in PB_KEYWORDS):
        return "PB"
    if any(keyword in industry for keyword in PE_KEYWORDS):
        return "PE"
    return "VALUE"


def get_trade_day_and_universe():
    rs_dates = bs.query_trade_dates(
        start_date=(pd.Timestamp.now() - pd.DateOffset(days=15)).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    df_dates = rs_dates.get_data()
    if rs_dates.error_code != "0" or df_dates.empty:
        return None, pd.DataFrame()
    df_dates.columns = rs_dates.fields
    df_dates = df_dates[df_dates["is_trading_day"] == "1"]

    today_str, df_stocks = None, pd.DataFrame()
    for day in reversed(df_dates["calendar_date"].tolist()):
        rs = bs.query_all_stock(day=day)
        tmp = rs.get_data()
        if not tmp.empty:
            tmp.columns = rs.fields
            today_str = day
            df_stocks = tmp
            break
    if df_stocks.empty:
        return None, pd.DataFrame()

    mask = (
        df_stocks["code"].str.startswith("sh.60")
        | df_stocks["code"].str.startswith("sh.68")
        | df_stocks["code"].str.startswith("sz.00")
        | df_stocks["code"].str.startswith("sz.30")
    )
    df_stocks = df_stocks[mask & ~df_stocks["tradeStatus"].eq("0")]
    df_stocks = df_stocks[~df_stocks["code_name"].str.contains(r"ST|\*ST")]
    return today_str, df_stocks


def get_industry_map():
    rs = bs.query_stock_industry()
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return {}
    df.columns = rs.fields
    return {row["code"]: row["industry"] for _, row in df.iterrows()}


def init_worker(today_str=None):
    global BENCHMARK_DF
    bs.login()
    BENCHMARK_DF = get_benchmark_history(today_str) if today_str else None


def score_stock_task(task):
    code, name, industry, method, today_str, year, quarter, value_min_mktcap = task
    try:
        row, skip_reason = score_stock(code, name, industry, method, today_str, year, quarter, value_min_mktcap=value_min_mktcap)
        return {"code": code, "name": name, "row": row, "skip_reason": skip_reason, "error": None}
    except Exception as exc:
        return {"code": code, "name": name, "row": None, "skip_reason": "", "error": str(exc)}


def get_benchmark_history(today_str):
    if not today_str:
        return None
    start_date = (pd.to_datetime(today_str) - pd.DateOffset(days=420)).strftime("%Y-%m-%d")
    fields = "date,close"
    with time_limit(BS_TIMEOUT_SECONDS):
        rs = bs.query_history_k_data_plus(BENCHMARK_CODE, fields, start_date=start_date, end_date=today_str, frequency="d")
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return None
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df if len(df) >= 80 else None


def calc_mainline_metrics(df, benchmark_df):
    if benchmark_df is None or benchmark_df.empty or len(df) < 120:
        return {
            "mainline_score": 0,
            "mainline_label": "基准不足",
            "mainline_ref": "基准不足",
            "relative_ret20": None,
            "relative_ret60": None,
            "volume_ratio_20_120": None,
            "down_day_win_rate": None,
            "early_recovery": 0,
        }

    merged = df[["date", "close", "amount"]].merge(
        benchmark_df[["date", "close"]].rename(columns={"close": "bench_close"}),
        on="date",
        how="inner",
    )
    if len(merged) < 80:
        return {
            "mainline_score": 0,
            "mainline_label": "基准不足",
            "mainline_ref": "基准不足",
            "relative_ret20": None,
            "relative_ret60": None,
            "volume_ratio_20_120": None,
            "down_day_win_rate": None,
            "early_recovery": 0,
        }

    close = merged["close"]
    bench = merged["bench_close"]
    ret20 = close.iloc[-1] / close.iloc[-21] - 1 if len(merged) >= 21 and close.iloc[-21] > 0 else None
    bench_ret20 = bench.iloc[-1] / bench.iloc[-21] - 1 if len(merged) >= 21 and bench.iloc[-21] > 0 else None
    ret60 = close.iloc[-1] / close.iloc[-61] - 1 if len(merged) >= 61 and close.iloc[-61] > 0 else None
    bench_ret60 = bench.iloc[-1] / bench.iloc[-61] - 1 if len(merged) >= 61 and bench.iloc[-61] > 0 else None
    relative_ret20 = ret20 - bench_ret20 if ret20 is not None and bench_ret20 is not None else None
    relative_ret60 = ret60 - bench_ret60 if ret60 is not None and bench_ret60 is not None else None

    avg_amount20 = merged["amount"].tail(20).mean()
    avg_amount120 = merged["amount"].tail(120).mean() if len(merged) >= 120 else merged["amount"].mean()
    volume_ratio = avg_amount20 / avg_amount120 if avg_amount120 and avg_amount120 > 0 else None

    recent = merged.tail(60).copy()
    stock_pct = recent["close"].pct_change()
    bench_pct = recent["bench_close"].pct_change()
    down_mask = bench_pct < 0
    if down_mask.sum() >= 5:
        down_day_win_rate = float((stock_pct[down_mask] > bench_pct[down_mask]).mean())
    else:
        down_day_win_rate = None

    bench_low_pos = recent["bench_close"].idxmin()
    stock_gain_from_bench_low = None
    bench_gain_from_low = None
    early_recovery = 0
    if bench_low_pos in recent.index and recent.loc[bench_low_pos, "close"] > 0 and recent.loc[bench_low_pos, "bench_close"] > 0:
        stock_gain_from_bench_low = recent.iloc[-1]["close"] / recent.loc[bench_low_pos, "close"] - 1
        bench_gain_from_low = recent.iloc[-1]["bench_close"] / recent.loc[bench_low_pos, "bench_close"] - 1
        stock_ma20 = close.tail(20).mean()
        bench_ma20 = bench.tail(20).mean()
        if close.iloc[-1] > stock_ma20 and (bench.iloc[-1] <= bench_ma20 or stock_gain_from_bench_low > bench_gain_from_low + 0.05):
            early_recovery = 1

    mainline_score = 0
    mainline_score += score_direct(relative_ret60, -0.05, 0.25) * 0.35
    mainline_score += score_direct(relative_ret20, -0.03, 0.12) * 0.20
    mainline_score += score_direct(volume_ratio, 0.80, 1.80) * 0.20
    mainline_score += score_direct(down_day_win_rate, 0.45, 0.70) * 0.15
    mainline_score += early_recovery * 10
    mainline_score = clamp(mainline_score)
    if mainline_score >= 80:
        label = "主线强势"
    elif mainline_score >= 65:
        label = "主线观察"
    else:
        label = "普通"
    ref = (
        f"{BENCHMARK_NAME}相对20/60日={fmt_pct(relative_ret20, 0)}/{fmt_pct(relative_ret60, 0)}, "
        f"量能20/120={fmt_num(volume_ratio, 2)}, 下跌日胜率={fmt_pct(down_day_win_rate, 0)}, "
        f"早修复={'是' if early_recovery else '否'}"
    )
    return {
        "mainline_score": round(mainline_score, 1),
        "mainline_label": label,
        "mainline_ref": ref,
        "relative_ret20": relative_ret20,
        "relative_ret60": relative_ret60,
        "volume_ratio_20_120": volume_ratio,
        "down_day_win_rate": down_day_win_rate,
        "early_recovery": early_recovery,
    }


def get_history_metrics(code, today_str):
    start_date = (pd.to_datetime(today_str) - pd.DateOffset(years=10, days=30)).strftime("%Y-%m-%d")
    fields = "date,high,low,close,volume,amount,turn,peTTM,pbMRQ"
    # 估值比较使用每股财务值，价格也要用当日不复权价格，避免前复权压低历史节点价格。
    with time_limit(BS_TIMEOUT_SECONDS):
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=today_str, frequency="d", adjustflag="3")
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return None

    numeric_cols = ["high", "low", "close", "volume", "amount", "turn", "peTTM", "pbMRQ"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if len(df) < 250:
        return None

    latest = df.iloc[-1]
    close = latest["close"]
    if pd.isna(close) or close <= 0:
        return None

    close_series = df["close"]
    ret20 = close / close_series.iloc[-21] - 1 if len(df) >= 21 and close_series.iloc[-21] > 0 else None
    ret60 = close / close_series.iloc[-61] - 1 if len(df) >= 61 and close_series.iloc[-61] > 0 else None
    ma20 = close_series.tail(20).mean()
    ma60 = close_series.tail(60).mean()
    ma120 = close_series.tail(120).mean()
    avg_amount20 = df["amount"].tail(20).mean() if "amount" in df else None
    technical = get_technical_metrics(df)
    mainline = calc_mainline_metrics(df, BENCHMARK_DF)

    trend_score = 0
    trend_score += 25 if close > ma20 else 0
    trend_score += 25 if ma20 > ma60 else 0
    trend_score += 20 if ma60 > ma120 else 0
    trend_score += score_direct(ret60, -0.10, 0.30) * 0.2
    trend_score += score_direct(ret20, -0.08, 0.18) * 0.1
    if ret20 is not None and ret20 > 0.45:
        trend_score -= 15
    trend_score = clamp(trend_score)

    liquidity_score = score_direct(np.log10(avg_amount20) if avg_amount20 and avg_amount20 > 0 else None, 7.0, 9.5)

    return {
        "df": df,
        "close": close,
        "pe": latest.get("peTTM"),
        "pb": latest.get("pbMRQ"),
        "ret20": ret20,
        "ret60": ret60,
        "ma20": ma20,
        "ma60": ma60,
        "avg_amount20": avg_amount20,
        "trend_score": trend_score,
        "liquidity_score": liquidity_score,
        "mainline": mainline,
        "technical": technical,
    }


def get_profit_metrics(code, year, quarter):
    with time_limit(BS_TIMEOUT_SECONDS):
        latest = bs.query_profit_data(code=code, year=year, quarter=quarter)
        if latest.error_code != "0" or not latest.next():
            return {"roe": None, "eps": None, "eps_yoy": None, "score": 0}
        latest_df = pd.DataFrame([latest.get_row_data()], columns=latest.fields)

    prev_year = year - 1
    prev_df = pd.DataFrame()
    with time_limit(BS_TIMEOUT_SECONDS):
        prev = bs.query_profit_data(code=code, year=prev_year, quarter=quarter)
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


def get_value_line_metrics_from_adata(symbol, close, report_period=None):
    if adata is None:
        return None
    try:
        df = adata.stock.finance.get_core_index(symbol)
        if df is None or df.empty:
            return None

        df = df.copy()
        df["报告期_dt"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.dropna(subset=["报告期_dt"]).sort_values("报告期_dt")
        if report_period:
            df = df[df["报告期_dt"] <= pd.Timestamp(report_period)]
        if df.empty:
            return None

        latest = df.iloc[-1]
        bvps = parse_float(latest.get("net_asset_ps"))
        yoy_pct = parse_float(latest.get("non_gaap_net_profit_yoy_gr"))
        if bvps is None or yoy_pct is None:
            return None
        yoy = yoy_pct / 100

        df_annual = df[(df["report_type"] == "年报") | (df["report_date"].astype(str).str.endswith("12-31"))]
        if df_annual.empty:
            return None
        annual = df_annual.iloc[-1]

        # adata exposes the Eastmoney non-recurring EPS in diluted_eps. For Q1
        # nodes after an annual stock bonus/conversion plan, convert it to the
        # recap-comparable EPS used by the value-line examples.
        raw_eps_excl = parse_float(annual.get("diluted_eps"))
        if raw_eps_excl is None:
            raw_eps_excl = parse_float(annual.get("non_gaap_eps"))
        eps_excl, eps_detail = comparable_excl_eps(
            symbol,
            annual.get("report_date"),
            latest.get("report_date"),
            raw_eps_excl,
        )
        if eps_excl is None or eps_excl <= 0:
            return None

        annual_excl_profit = parse_float(annual.get("non_gaap_net_profit"))
        total_share = annual_excl_profit / raw_eps_excl if annual_excl_profit and raw_eps_excl else None
        if not total_share or total_share <= 0:
            annual_net_profit = parse_float(annual.get("net_profit_attr_sh"))
            annual_basic_eps = parse_float(annual.get("basic_eps"))
            if not annual_net_profit or annual_basic_eps is None or annual_basic_eps <= 0:
                return None
            total_share = annual_net_profit / annual_basic_eps
        if total_share <= 0:
            return None

        value_line = bvps + eps_excl * (1 + yoy) * 10
        if value_line <= 0:
            return None

        annual_excl = [parse_float(r.get("non_gaap_net_profit")) for _, r in df_annual.tail(3).iterrows()]
        annual_excl = [v for v in annual_excl if v is not None]
        positive_years = sum(1 for v in annual_excl if v > 0)
        growth_steps = sum(1 for i in range(len(annual_excl) - 1) if annual_excl[i + 1] > annual_excl[i])

        quality_yoy = min(max(yoy, -0.5), 1.0)
        price_to_value = close / value_line
        valuation_score = score_inverse(price_to_value, best=0.55, worst=1.25)
        quality_score = (
            score_direct(eps_excl, 0.10, 1.50) * 0.35
            + score_direct(quality_yoy, -0.10, 0.50) * 0.35
            + score_direct(positive_years, 1, 3) * 0.15
            + score_direct(growth_steps, 0, 2) * 0.15
        )
        mktcap = close * total_share / 1e8

        return {
            "value_line": value_line,
            "price_to_value": price_to_value,
            "valuation_score": valuation_score,
            "quality_score": clamp(quality_score),
            "mktcap": mktcap,
            "eps_excl": eps_excl,
            "yoy": yoy,
            "yoy_source": "adata东方财富扣非净利润同比",
            "latest_excl_eps": parse_float(latest.get("diluted_eps")),
            "prev_excl_eps": None,
            "latest_report": str(latest["report_date"]),
            "annual_report": str(annual["report_date"]),
            "data_source": "adata/eastmoney",
            "total_share": total_share,
            **(eps_detail or {}),
        }
    except Exception:
        return None


def to_em_symbol(symbol):
    code = str(symbol).split(".")[-1]
    if code.startswith(("6", "9")):
        suffix = "SH"
    elif code.startswith(("8", "4")):
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{code}.{suffix}"


def build_value_line_result(
    close,
    bvps,
    eps_excl,
    yoy,
    annual_excl,
    total_share,
    yoy_source,
    latest_report,
    annual_report,
    data_source,
    latest_excl_eps=None,
    prev_excl_eps=None,
    eps_excl_raw=None,
    eps_adjustment_factor=None,
    eps_excl_source=None,
    eps_bonus_detail=None,
):
    if bvps is None or eps_excl is None or yoy is None or eps_excl <= 0:
        return None
    if not total_share or total_share <= 0:
        return None

    value_line = bvps + eps_excl * (1 + yoy) * 10
    if value_line <= 0:
        return None

    annual_excl = [v for v in annual_excl if v is not None]
    positive_years = sum(1 for v in annual_excl if v > 0)
    growth_steps = sum(1 for i in range(len(annual_excl) - 1) if annual_excl[i + 1] > annual_excl[i])
    quality_yoy = min(max(yoy, -0.5), 1.0)
    price_to_value = close / value_line
    valuation_score = score_inverse(price_to_value, best=0.55, worst=1.25)
    quality_score = (
        score_direct(eps_excl, 0.10, 1.50) * 0.35
        + score_direct(quality_yoy, -0.10, 0.50) * 0.35
        + score_direct(positive_years, 1, 3) * 0.15
        + score_direct(growth_steps, 0, 2) * 0.15
    )
    mktcap = close * total_share / 1e8

    return {
        "value_line": value_line,
        "price_to_value": price_to_value,
        "valuation_score": valuation_score,
        "quality_score": clamp(quality_score),
        "mktcap": mktcap,
        "eps_excl": eps_excl,
        "yoy": yoy,
        "yoy_source": yoy_source,
        "latest_excl_eps": latest_excl_eps,
        "prev_excl_eps": prev_excl_eps,
        "latest_report": str(latest_report),
        "annual_report": str(annual_report),
        "data_source": data_source,
        "total_share": total_share,
        "eps_excl_raw": eps_excl_raw,
        "eps_adjustment_factor": eps_adjustment_factor,
        "eps_excl_source": eps_excl_source,
        "eps_bonus_detail": eps_bonus_detail,
    }


def get_value_line_metrics_from_akshare_em(symbol, close, report_period=None):
    """Use AkShare Eastmoney financial indicators with direct EPSKCJB field."""
    try:
        with time_limit(AK_TIMEOUT_SECONDS):
            df = ak.stock_financial_analysis_indicator_em(symbol=to_em_symbol(symbol), indicator="按报告期")
        if df is None or df.empty:
            return None

        df = df.copy()
        df["报告期_dt"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        df = df.dropna(subset=["报告期_dt"]).sort_values("报告期_dt")
        if report_period:
            df = df[df["报告期_dt"] <= pd.Timestamp(report_period)]
        if df.empty:
            return None

        latest = df.iloc[-1]
        latest_dt = latest["报告期_dt"]
        bvps = parse_float(latest.get("BPS"))

        yoy = None
        yoy_source = None
        prev_excl_eps = None
        yoy_pct = parse_float(latest.get("KCFJCXSYJLRTZ"))
        if yoy_pct is not None:
            yoy = yoy_pct / 100
            yoy_source = "akshare东方财富扣非净利润同比"
        else:
            latest_excl_profit = parse_float(latest.get("KCFJCXSYJLR"))
            prev_rows = df[df["报告期_dt"] == latest_dt - pd.DateOffset(years=1)]
            if latest_excl_profit is not None and not prev_rows.empty:
                prev_excl_profit = parse_float(prev_rows.iloc[-1].get("KCFJCXSYJLR"))
                if prev_excl_profit and prev_excl_profit > 0:
                    yoy = latest_excl_profit / prev_excl_profit - 1
                    yoy_source = "akshare东方财富扣非净利润同比(反推)"
                    prev_excl_eps = parse_float(prev_rows.iloc[-1].get("EPSKCJB"))
        if yoy is None:
            return None

        df_annual = df[df["报告期_dt"].dt.strftime("%m-%d") == "12-31"]
        if df_annual.empty:
            return None
        annual = df_annual.iloc[-1]
        raw_eps_excl = parse_float(annual.get("EPSKCJB"))
        eps_excl, eps_detail = comparable_excl_eps(
            symbol,
            annual.get("REPORT_DATE"),
            latest.get("REPORT_DATE"),
            raw_eps_excl,
        )
        if eps_excl is None or eps_excl <= 0:
            return None

        annual_excl_profit = parse_float(annual.get("KCFJCXSYJLR"))
        total_share = annual_excl_profit / raw_eps_excl if annual_excl_profit and raw_eps_excl else None
        if not total_share or total_share <= 0:
            annual_net_profit = parse_float(annual.get("PARENTNETPROFIT"))
            annual_basic_eps = parse_float(annual.get("EPSJB"))
            total_share = annual_net_profit / annual_basic_eps if annual_net_profit and annual_basic_eps else None

        annual_excl = [parse_float(r.get("KCFJCXSYJLR")) for _, r in df_annual.tail(3).iterrows()]
        return build_value_line_result(
            close=close,
            bvps=bvps,
            eps_excl=eps_excl,
            yoy=yoy,
            annual_excl=annual_excl,
            total_share=total_share,
            yoy_source=yoy_source,
            latest_report=latest["REPORT_DATE"],
            annual_report=annual["REPORT_DATE"],
            data_source="akshare/eastmoney_indicator",
            latest_excl_eps=parse_float(latest.get("EPSKCJB")),
            prev_excl_eps=prev_excl_eps,
            **(eps_detail or {}),
        )
    except Exception:
        return None


def get_value_line_metrics_from_akshare_sina(symbol, close, report_period=None):
    """Use AkShare Sina financial indicators with direct non-recurring EPS field."""
    try:
        if report_period:
            start_year = str(max(1900, pd.Timestamp(report_period).year - 3))
        else:
            start_year = str(datetime.now().year - 4)
        with time_limit(AK_TIMEOUT_SECONDS):
            df = ak.stock_financial_analysis_indicator(symbol=str(symbol).split(".")[-1], start_year=start_year)
        if df is None or df.empty:
            return None

        df = df.copy()
        df["报告期_dt"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["报告期_dt"]).sort_values("报告期_dt")
        if report_period:
            df = df[df["报告期_dt"] <= pd.Timestamp(report_period)]
        if df.empty:
            return None

        latest = df.iloc[-1]
        latest_dt = latest["报告期_dt"]
        bvps = parse_float(first_existing_value(latest, ["每股净资产_调整后(元)", "每股净资产_调整前(元)"]))
        latest_excl_profit = parse_float(latest.get("扣除非经常性损益后的净利润(元)"))
        prev_rows = df[df["报告期_dt"] == latest_dt - pd.DateOffset(years=1)]
        if latest_excl_profit is None or prev_rows.empty:
            return None
        prev_excl_profit = parse_float(prev_rows.iloc[-1].get("扣除非经常性损益后的净利润(元)"))
        if not prev_excl_profit or prev_excl_profit <= 0:
            return None
        yoy = latest_excl_profit / prev_excl_profit - 1

        df_annual = df[df["报告期_dt"].dt.strftime("%m-%d") == "12-31"]
        if df_annual.empty:
            return None
        annual = df_annual.iloc[-1]
        raw_eps_excl = parse_float(annual.get("扣除非经常性损益后的每股收益(元)"))
        eps_excl, eps_detail = comparable_excl_eps(
            symbol,
            annual.get("日期"),
            latest.get("日期"),
            raw_eps_excl,
        )
        if eps_excl is None or eps_excl <= 0:
            return None

        annual_excl_profit = parse_float(annual.get("扣除非经常性损益后的净利润(元)"))
        total_share = annual_excl_profit / raw_eps_excl if annual_excl_profit and raw_eps_excl else None
        annual_excl = [parse_float(r.get("扣除非经常性损益后的净利润(元)")) for _, r in df_annual.tail(3).iterrows()]
        return build_value_line_result(
            close=close,
            bvps=bvps,
            eps_excl=eps_excl,
            yoy=yoy,
            annual_excl=annual_excl,
            total_share=total_share,
            yoy_source="akshare新浪扣非净利润同比(反推)",
            latest_report=latest["日期"],
            annual_report=annual["日期"],
            data_source="akshare/sina_indicator",
            latest_excl_eps=parse_float(latest.get("扣除非经常性损益后的每股收益(元)")),
            prev_excl_eps=parse_float(prev_rows.iloc[-1].get("扣除非经常性损益后的每股收益(元)")),
            **(eps_detail or {}),
        )
    except Exception:
        return None


def get_value_line_metrics_from_akshare_indicator(symbol, close, report_period=None):
    metrics = get_value_line_metrics_from_akshare_em(symbol, close, report_period)
    if metrics:
        return metrics
    return get_value_line_metrics_from_akshare_sina(symbol, close, report_period)


def get_value_line_metrics(symbol, close):
    try:
        cached = load_value_cache(symbol)
        if cached:
            if cached.get("cache_version") != VALUE_CACHE_VERSION:
                cached = None
        if cached:
            total_share = cached.get("total_share")
            value_line = cached.get("value_line")
            if total_share and value_line:
                price_to_value = close / value_line
                mktcap = close * total_share / 1e8
                return {
                    "value_line": value_line,
                    "price_to_value": price_to_value,
                    "valuation_score": score_inverse(price_to_value, best=0.55, worst=1.25),
                    "quality_score": cached.get("quality_score", 0),
                    "mktcap": mktcap,
                    "eps_excl": cached.get("eps_excl"),
                    "yoy": cached.get("yoy"),
                    "yoy_source": cached.get("yoy_source"),
                    "latest_excl_eps": cached.get("latest_excl_eps"),
                    "prev_excl_eps": cached.get("prev_excl_eps"),
                    "data_source": cached.get("data_source"),
                    "eps_excl_raw": cached.get("eps_excl_raw"),
                    "eps_adjustment_factor": cached.get("eps_adjustment_factor"),
                    "eps_excl_source": cached.get("eps_excl_source"),
                    "eps_bonus_detail": cached.get("eps_bonus_detail"),
                }

        ak_indicator_metrics = get_value_line_metrics_from_akshare_indicator(symbol, close)
        if ak_indicator_metrics:
            save_value_cache(symbol, {
                "cache_version": VALUE_CACHE_VERSION,
                "value_line": ak_indicator_metrics["value_line"],
                "quality_score": ak_indicator_metrics["quality_score"],
                "total_share": ak_indicator_metrics["total_share"],
                "eps_excl": ak_indicator_metrics["eps_excl"],
                "yoy": ak_indicator_metrics["yoy"],
                "yoy_source": ak_indicator_metrics["yoy_source"],
                "latest_excl_eps": ak_indicator_metrics.get("latest_excl_eps"),
                "prev_excl_eps": ak_indicator_metrics.get("prev_excl_eps"),
                "latest_report": ak_indicator_metrics.get("latest_report"),
                "annual_report": ak_indicator_metrics.get("annual_report"),
                "data_source": ak_indicator_metrics.get("data_source"),
                "eps_excl_raw": ak_indicator_metrics.get("eps_excl_raw"),
                "eps_adjustment_factor": ak_indicator_metrics.get("eps_adjustment_factor"),
                "eps_excl_source": ak_indicator_metrics.get("eps_excl_source"),
                "eps_bonus_detail": ak_indicator_metrics.get("eps_bonus_detail"),
            })
            return ak_indicator_metrics

        adata_metrics = get_value_line_metrics_from_adata(symbol, close)
        if adata_metrics:
            save_value_cache(symbol, {
                "cache_version": VALUE_CACHE_VERSION,
                "value_line": adata_metrics["value_line"],
                "quality_score": adata_metrics["quality_score"],
                "total_share": adata_metrics["total_share"],
                "eps_excl": adata_metrics["eps_excl"],
                "yoy": adata_metrics["yoy"],
                "yoy_source": adata_metrics["yoy_source"],
                "latest_excl_eps": adata_metrics.get("latest_excl_eps"),
                "prev_excl_eps": adata_metrics.get("prev_excl_eps"),
                "latest_report": adata_metrics.get("latest_report"),
                "annual_report": adata_metrics.get("annual_report"),
                "data_source": adata_metrics.get("data_source"),
                "eps_excl_raw": adata_metrics.get("eps_excl_raw"),
                "eps_adjustment_factor": adata_metrics.get("eps_adjustment_factor"),
                "eps_excl_source": adata_metrics.get("eps_excl_source"),
                "eps_bonus_detail": adata_metrics.get("eps_bonus_detail"),
            })
            return adata_metrics

        with time_limit(AK_TIMEOUT_SECONDS):
            df_q = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
        df_q = df_q[df_q["扣非净利润"] != False].copy()
        if df_q.empty:
            return None
        df_q["报告期_dt"] = pd.to_datetime(df_q["报告期"], errors="coerce")
        df_q = df_q.dropna(subset=["报告期_dt"]).sort_values("报告期_dt")
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
        growth_steps = 0
        for i in range(len(annual_excl) - 1):
            if annual_excl[i + 1] > annual_excl[i]:
                growth_steps += 1

        price_to_value = close / value_line
        valuation_score = score_inverse(price_to_value, best=0.55, worst=1.25)
        quality_score = (
            score_direct(eps_excl, 0.10, 1.50) * 0.35
            + score_direct(quality_yoy, -0.10, 0.50) * 0.35
            + score_direct(positive_years, 1, 3) * 0.15
            + score_direct(growth_steps, 0, 2) * 0.15
        )
        mktcap = close * total_share / 1e8
        save_value_cache(symbol, {
            "cache_version": VALUE_CACHE_VERSION,
            "value_line": value_line,
            "quality_score": clamp(quality_score),
            "total_share": total_share,
            "eps_excl": eps_excl,
            "yoy": yoy,
            "yoy_source": yoy_metrics["yoy_source"],
            "latest_excl_eps": yoy_metrics.get("latest_excl_eps"),
            "prev_excl_eps": yoy_metrics.get("prev_excl_eps"),
            "latest_report": str(latest["报告期"]),
            "annual_report": str(annual["报告期"]),
            "data_source": "akshare/ths",
            "eps_excl_raw": (eps_detail or {}).get("eps_excl_raw"),
            "eps_adjustment_factor": (eps_detail or {}).get("eps_adjustment_factor"),
            "eps_excl_source": (eps_detail or {}).get("eps_excl_source"),
            "eps_bonus_detail": (eps_detail or {}).get("eps_bonus_detail"),
        })
        return {
            "value_line": value_line,
            "price_to_value": price_to_value,
            "valuation_score": valuation_score,
            "quality_score": clamp(quality_score),
            "mktcap": mktcap,
            "eps_excl": eps_excl,
            "yoy": yoy,
            "yoy_source": yoy_metrics["yoy_source"],
            "latest_excl_eps": yoy_metrics.get("latest_excl_eps"),
            "prev_excl_eps": yoy_metrics.get("prev_excl_eps"),
            "data_source": "akshare/ths",
            "eps_excl_raw": (eps_detail or {}).get("eps_excl_raw"),
            "eps_adjustment_factor": (eps_detail or {}).get("eps_adjustment_factor"),
            "eps_excl_source": (eps_detail or {}).get("eps_excl_source"),
            "eps_bonus_detail": (eps_detail or {}).get("eps_bonus_detail"),
        }
    except Exception:
        return None


def get_pe_pb_metrics(history_df, field, current_value):
    if current_value is None or pd.isna(current_value) or current_value <= 0:
        return None
    df = history_df.dropna(subset=[field]).copy()
    df = df[df[field] > 0]
    if len(df) < 250:
        return None

    df["year"] = df["date"].str[:4]
    yearly_min = df.groupby("year")[field].min()
    current_year = str(pd.to_datetime(df["date"].iloc[-1]).year)
    if current_year in yearly_min.index and len(df[df["year"] == current_year]) < 60:
        yearly_min = yearly_min.drop(current_year)
    if len(yearly_min) < 3:
        return None

    filtered = remove_outliers(yearly_min.values.tolist())
    low_avg = float(np.mean(filtered))
    ratio = current_value / low_avg if low_avg > 0 else None
    percentile = float((df[field] <= current_value).mean())

    ratio_score = score_inverse(ratio, best=0.60, worst=1.35)
    percentile_score = score_inverse(percentile, best=0.05, worst=0.60)
    valuation_score = ratio_score * 0.65 + percentile_score * 0.35
    return {
        "low_avg": low_avg,
        "ratio": ratio,
        "percentile": percentile,
        "valuation_score": clamp(valuation_score),
    }


def is_low_value_candidate(row):
    method = row["method"]
    if method == "RIGHT":
        return False

    if method == "VALUE":
        price_to_value = row.get("price_to_value")
        return price_to_value is not None and pd.notna(price_to_value) and price_to_value <= VALUE_LOW_VALUE_RATIO

    ratio = row.get("pepb_ratio")
    percentile = row.get("valuation_percentile")
    return (
        ratio is not None and pd.notna(ratio) and ratio <= 1.0
    ) or (
        percentile is not None and pd.notna(percentile) and percentile <= 0.15
    )


def is_high_quality_candidate(row):
    min_quality_score = PEPB_QUALITY_MIN_SCORE if row["method"] in {"PE", "PB"} else QUALITY_MIN_SCORE
    if row["quality_score"] < min_quality_score:
        return False
    if row["liquidity_score"] < QUALITY_MIN_LIQUIDITY_SCORE:
        return False

    if row["method"] == "RIGHT":
        return row["trend_score"] >= RIGHT_MIN_TREND_SCORE

    if row["method"] == "VALUE":
        price_to_value = row.get("price_to_value")
        return (
            price_to_value is not None
            and pd.notna(price_to_value)
            and price_to_value <= VALUE_QUALITY_MAX_RATIO
            and row["trend_score"] >= QUALITY_MIN_TREND_SCORE
        )

    ratio = row.get("pepb_ratio")
    percentile = row.get("valuation_percentile")
    valuation_ok = (
        ratio is not None and pd.notna(ratio) and ratio <= PEPB_QUALITY_MAX_RATIO
    ) or (
        percentile is not None and pd.notna(percentile) and percentile <= PEPB_QUALITY_MAX_PERCENTILE
    )
    return valuation_ok and row["trend_score"] >= QUALITY_MIN_TREND_SCORE


def classify_selection_bucket(row):
    is_low_value = is_low_value_candidate(row)
    is_high_quality = is_high_quality_candidate(row)

    if is_low_value and is_high_quality:
        if row["method"] == "VALUE" and row.get("price_to_value") is not None and row["price_to_value"] <= VALUE_UNDERVALUED_RATIO:
            return CORE_BUCKET, "深度低估且高质量"
        return CORE_BUCKET, "低估且高质量"

    if is_low_value:
        if row["method"] == "VALUE" and row.get("price_to_value") is not None and row["price_to_value"] <= VALUE_UNDERVALUED_RATIO:
            return LOW_VALUE_BUCKET, "深度低估"
        return LOW_VALUE_BUCKET, "价值线内/历史低位"

    if is_high_quality:
        if row["method"] == "RIGHT":
            return HIGH_QUALITY_BUCKET, "右侧高质量"
        return HIGH_QUALITY_BUCKET, "估值有约束的高质量"

    return WATCH_BUCKET, "未达主策略"


def get_effective_valuation_score(method, valuation_score, extra):
    """估值分用于综合评分时，区分深度折价和估值通过。

    原始估值分更偏向“越便宜越高”，适合排序深度低估；
    但高质量成长股在现价低于/接近基本价值线时，已经满足左侧估值条件，
    不应因为折价不深就被压到很低。
    """
    score = valuation_score
    if method == "VALUE":
        price_to_value = extra.get("price_to_value")
        if price_to_value is None or pd.isna(price_to_value):
            return score
        if price_to_value <= VALUE_UNDERVALUED_RATIO:
            return max(score, 85)
        if price_to_value <= 1.00:
            return max(score, 75)
        if price_to_value <= VALUE_REASONABLE_RATIO:
            return max(score, 55)
        return score

    if method in {"PE", "PB"}:
        ratio = extra.get("ratio")
        percentile = extra.get("percentile")
        is_undervalued = (
            ratio is not None and pd.notna(ratio) and ratio <= 1.0
        ) or (
            percentile is not None and pd.notna(percentile) and percentile <= 0.15
        )
        is_reasonable = (
            ratio is not None and pd.notna(ratio) and ratio <= PEPB_REASONABLE_RATIO
        ) or (
            percentile is not None and pd.notna(percentile) and percentile <= PEPB_REASONABLE_PERCENTILE
        )
        if is_undervalued:
            return max(score, 75)
        if is_reasonable:
            return max(score, 60)
    return score


def pass_score_gate(row, quality_min_score, low_min_score, core_min_score):
    if row.get("selection_bucket") == WATCH_BUCKET:
        return False
    threshold = get_score_gate(row, quality_min_score, low_min_score, core_min_score)
    return row["total_score"] >= threshold


def is_actionable_watch(row):
    if row.get("selection_bucket") != WATCH_BUCKET:
        return False
    if row.get("method") == "VALUE":
        price_to_value = row.get("price_to_value")
        if price_to_value is None or pd.isna(price_to_value):
            return False
        return price_to_value <= VALUE_REASONABLE_RATIO and row.get("quality_score", 0) >= 55
    if row.get("method") in {"PE", "PB"}:
        ratio = row.get("pepb_ratio")
        percentile = row.get("valuation_percentile")
        return (
            ratio is not None and pd.notna(ratio) and ratio <= PEPB_REASONABLE_RATIO
        ) or (
            percentile is not None and pd.notna(percentile) and percentile <= PEPB_REASONABLE_PERCENTILE
        )
    return row.get("quality_score", 0) >= QUALITY_MIN_SCORE and row.get("trend_score", 0) >= QUALITY_MIN_TREND_SCORE


def pass_diagnostic_gate(row):
    if row is None:
        return False
    if row.get("selection_bucket") != WATCH_BUCKET:
        return True
    return is_actionable_watch(row) or row.get("total_score", 0) >= 70 or row.get("mainline_score", 0) >= 65


def is_value_watch_candidate(row, max_price_to_value=VALUE_WATCH_RATIO):
    if row is None or row.get("method") != "VALUE":
        return False
    price_to_value = row.get("price_to_value")
    if price_to_value is None or pd.isna(price_to_value):
        return False
    if price_to_value > max_price_to_value:
        return False
    mktcap = row.get("mktcap")
    if mktcap is not None and pd.notna(mktcap) and mktcap < VALUE_MIN_MKTCAP:
        return False
    if row.get("quality_score", 0) < VALUE_WATCH_MIN_QUALITY_SCORE:
        return False
    return True


def get_value_watch_rows(rows, max_price_to_value=VALUE_WATCH_RATIO, top=DEFAULT_VALUE_WATCH_TOP):
    candidates = [row for row in rows if is_value_watch_candidate(row, max_price_to_value)]
    candidates.sort(key=lambda row: (
        abs((row.get("price_to_value") or 0) - 1.0),
        row.get("price_to_value") or 99,
        -(row.get("quality_score") or 0),
        -(row.get("liquidity_score") or 0),
    ))
    return candidates[:top] if top and top > 0 else candidates


def get_block_reason(row, quality_min_score, low_min_score, core_min_score):
    if row is None:
        return ""
    if pass_score_gate(row, quality_min_score, low_min_score, core_min_score):
        return "严格入选"
    if row.get("selection_bucket") == WATCH_BUCKET:
        reasons = []
        method = row.get("method")
        if method == "VALUE":
            price_to_value = row.get("price_to_value")
            if price_to_value is None or pd.isna(price_to_value):
                reasons.append("价值线不可用")
            elif price_to_value > VALUE_LOW_VALUE_RATIO:
                reasons.append(f"现价高于价值线({price_to_value:.2f})")
            if row.get("quality_score", 0) < QUALITY_MIN_SCORE:
                reasons.append(f"质量分不足({row.get('quality_score', 0):.1f})")
            if row.get("trend_score", 0) < QUALITY_MIN_TREND_SCORE:
                reasons.append(f"趋势分不足({row.get('trend_score', 0):.1f})")
        elif method in {"PE", "PB"}:
            ratio = row.get("pepb_ratio")
            percentile = row.get("valuation_percentile")
            if (
                (ratio is None or pd.isna(ratio) or ratio > 1.0)
                and (percentile is None or pd.isna(percentile) or percentile > 0.15)
            ):
                reasons.append("未到历史低估")
            if row.get("quality_score", 0) < PEPB_QUALITY_MIN_SCORE:
                reasons.append(f"质量分不足({row.get('quality_score', 0):.1f})")
            if row.get("trend_score", 0) < QUALITY_MIN_TREND_SCORE:
                reasons.append(f"趋势分不足({row.get('trend_score', 0):.1f})")
        else:
            if row.get("quality_score", 0) < QUALITY_MIN_SCORE:
                reasons.append(f"质量分不足({row.get('quality_score', 0):.1f})")
            if row.get("trend_score", 0) < RIGHT_MIN_TREND_SCORE:
                reasons.append(f"右侧趋势不足({row.get('trend_score', 0):.1f})")
        if row.get("liquidity_score", 0) < QUALITY_MIN_LIQUIDITY_SCORE:
            reasons.append(f"流动性不足({row.get('liquidity_score', 0):.1f})")
        return "；".join(reasons) if reasons else "未达主策略"
    threshold = get_score_gate(row, quality_min_score, low_min_score, core_min_score)
    return f"分数不足({row.get('total_score', 0):.1f}/{threshold})"


def get_score_gate(row, quality_min_score, low_min_score, core_min_score):
    if row.get("selection_bucket") == CORE_BUCKET:
        return core_min_score
    return low_min_score if row.get("selection_bucket") == LOW_VALUE_BUCKET else quality_min_score


def build_risk_flags(row):
    flags = []
    if row.get("method") != "RIGHT" and row["valuation_score"] < 45 and row.get("selection_bucket") != LOW_VALUE_BUCKET:
        flags.append("估值优势弱")
    if row["quality_score"] < 45:
        flags.append("质量偏弱")
    if row["trend_score"] < 45:
        flags.append("趋势未确认")
    if row["liquidity_score"] < 35:
        flags.append("流动性偏弱")
    if row.get("price_to_value") is not None and row["price_to_value"] < 0.30:
        flags.append("价值线折价异常需复核")
    flags.extend(row.get("technical_flags_list", []))
    return "、".join(flags) if flags else "正常"


def calc_low_value_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    if method in {"PE", "PB"}:
        return (
            valuation_score * 0.55
            + quality_score * 0.25
            + trend_score * 0.10
            + liquidity_score * 0.10
        )
    return (
        valuation_score * 0.50
        + quality_score * 0.30
        + trend_score * 0.10
        + liquidity_score * 0.10
    )


def calc_high_quality_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    if method == "RIGHT":
        return quality_score * 0.35 + trend_score * 0.45 + liquidity_score * 0.20
    return (
        valuation_score * 0.05
        + quality_score * 0.45
        + trend_score * 0.35
        + liquidity_score * 0.15
    )


def calc_core_score(low_value_score, high_quality_score):
    return low_value_score * 0.50 + high_quality_score * 0.50


def calc_total_score(method, valuation_score, quality_score, trend_score, liquidity_score):
    """旧验收脚本兼容函数；主流程使用低估/高质量两套独立打分。"""
    if method == "RIGHT":
        score = quality_score * 0.20 + trend_score * 0.55 + liquidity_score * 0.25
        return score, "右侧趋势"

    value_reversion = (
        valuation_score * 0.40
        + quality_score * 0.25
        + trend_score * 0.25
        + liquidity_score * 0.10
    )
    if method in {"PE", "PB"}:
        return value_reversion, "估值修复"

    quality_value = (
        valuation_score * 0.30
        + quality_score * 0.45
        + trend_score * 0.10
        + liquidity_score * 0.15
    )
    growth_momentum = (
        valuation_score * 0.10
        + quality_score * 0.35
        + trend_score * 0.40
        + liquidity_score * 0.15
    )
    scores = {
        "估值修复": value_reversion,
        "高质量折价": quality_value,
        "成长动量": growth_momentum,
    }
    mode, score = max(scores.items(), key=lambda item: item[1])
    return score, mode


def score_stock(code, name, industry, method, today_str, year, quarter, value_min_mktcap=VALUE_MIN_MKTCAP):
    history = get_history_metrics(code, today_str)
    if not history:
        return None, "行情不足"

    close = history["close"]
    symbol = code.replace("sh.", "").replace("sz.", "")
    valuation_score = 0
    quality_score = 0
    valuation_ref = ""
    extra = {}

    if method == "RIGHT":
        profit = get_profit_metrics(code, year, quarter)
        valuation_score = 0
        quality_score = profit["score"]
        valuation_ref = "轻资产行业不做左侧估值，仅按右侧趋势观察"
        extra = profit
    elif method == "VALUE":
        value_metrics = get_value_line_metrics(symbol, close)
        if not value_metrics:
            return None, "价值线数据不足"
        if value_metrics.get("mktcap") is None or value_metrics["mktcap"] < value_min_mktcap:
            return None, f"市值低于{value_min_mktcap:.0f}亿"
        valuation_score = value_metrics["valuation_score"]
        quality_score = value_metrics["quality_score"]
        valuation_ref = (
            f"价值线={value_metrics['value_line']:.2f}, 现价/价值={value_metrics['price_to_value']:.2f}, "
            f"{value_metrics.get('yoy_source') or '扣非同比'}={value_metrics.get('yoy', 0):.1%}, "
            f"市值={value_metrics['mktcap']:.1f}亿"
        )
        extra = value_metrics
    else:
        field = "peTTM" if method == "PE" else "pbMRQ"
        current = history["pe"] if method == "PE" else history["pb"]
        metrics = get_pe_pb_metrics(history["df"], field, current)
        if not metrics:
            return None, f"{method}历史估值数据不足"
        valuation_score = metrics["valuation_score"]
        profit = get_profit_metrics(code, year, quarter)
        quality_score = profit["score"]
        valuation_ref = (
            f"{method}={current:.2f}, 低估均值={metrics['low_avg']:.2f}, "
            f"比值={metrics['ratio']:.2f}, 分位={metrics['percentile']:.0%}"
        )
        extra = metrics
        extra.update(profit)
        extra["current_valuation"] = current

    trend_score = history["trend_score"]
    liquidity_score = history["liquidity_score"]
    technical = history.get("technical", {})
    mainline = history.get("mainline", {})
    raw_valuation_score = valuation_score
    effective_valuation_score = get_effective_valuation_score(method, raw_valuation_score, extra)
    low_value_score = calc_low_value_score(method, effective_valuation_score, quality_score, trend_score, liquidity_score)
    high_quality_score = calc_high_quality_score(method, effective_valuation_score, quality_score, trend_score, liquidity_score)
    core_score = calc_core_score(low_value_score, high_quality_score)

    row = {
        "code": code,
        "name": name,
        "industry": industry,
        "method": method,
        "method_name": METHOD_NAME[method],
        "close": round(close, 2),
        "total_score": 0,
        "valuation_score": round(effective_valuation_score, 1),
        "raw_valuation_score": round(raw_valuation_score, 1),
        "quality_score": round(quality_score, 1),
        "trend_score": round(trend_score, 1),
        "liquidity_score": round(liquidity_score, 1),
        "mainline_score": mainline.get("mainline_score", 0),
        "mainline_label": mainline.get("mainline_label", "基准不足"),
        "mainline_ref": mainline.get("mainline_ref", "基准不足"),
        "low_value_score": round(low_value_score, 1),
        "high_quality_score": round(high_quality_score, 1),
        "core_score": round(core_score, 1),
        "selection_mode": "",
        "valuation_ref": valuation_ref,
        "technical_ref": technical.get("technical_ref", ""),
        "technical_flags": technical.get("technical_flags", "正常"),
        "technical_flags_list": technical.get("technical_flags_list", []),
        "kd_k": technical.get("kd_k"),
        "kd_d": technical.get("kd_d"),
        "rsi999": technical.get("rsi999"),
        "ret20": history["ret20"],
        "ret60": history["ret60"],
        "relative_ret20": mainline.get("relative_ret20"),
        "relative_ret60": mainline.get("relative_ret60"),
        "volume_ratio_20_120": mainline.get("volume_ratio_20_120"),
        "down_day_win_rate": mainline.get("down_day_win_rate"),
        "early_recovery": mainline.get("early_recovery"),
        "theme": infer_theme(name, industry),
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
    row["risk_flags"] = build_risk_flags(row)
    return row, ""


def save_csv(today_str, rows):
    if not rows:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"factor_selection_{today_str}_{datetime.now().strftime('%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_diagnostic_csv(today_str, rows, skipped, top):
    if not rows and not skipped:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"factor_diagnostic_{today_str}_{timestamp}.csv")

    scored = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    output_parts = []
    if not scored.empty:
        scored = scored.sort_values(
            ["selected", "diagnostic_candidate", "total_score", "mainline_score"],
            ascending=[False, False, False, False],
        )
        output_parts.append(scored)
    if not skipped_df.empty:
        output_parts.append(skipped_df)
    if not output_parts:
        return None
    pd.concat(output_parts, ignore_index=True, sort=False).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_diagnostic_summary(all_rows, skipped, top):
    print("\n--- 全量诊断 ---")
    print(f"成功打分 {len(all_rows)} 只，跳过/失败 {len(skipped)} 只")
    if skipped:
        skipped_df = pd.DataFrame(skipped)
        if "skip_reason" in skipped_df.columns:
            print("跳过原因TOP:")
            print(skipped_df["skip_reason"].fillna("").replace("", "未知").value_counts().head(8).to_string())
    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    print("打分分组:")
    print(df["selection_bucket"].value_counts().to_string())
    if "selected" in df.columns:
        print(f"严格入选 {int(df['selected'].sum())} 只，诊断候选 {int(df['diagnostic_candidate'].sum())} 只")

    watch = df[(~df["selected"].astype(bool)) & (df["diagnostic_candidate"].astype(bool))].copy()
    if watch.empty:
        print(f"\n--- 观察候选(0只) ---")
        return
    watch = watch.sort_values(["total_score", "mainline_score"], ascending=False).head(top)
    cols = [
        "code", "name", "theme", "method_name", "close", "total_score", "selection_bucket",
        "valuation_state", "valuation_score", "quality_score", "trend_score", "liquidity_score",
        "mainline_score", "block_reason", "valuation_ref", "risk_flags",
    ]
    print(f"\n--- 观察候选({len(watch)}只，显示前{min(top, len(watch))}只) ---")
    print(watch[cols].to_string(index=False))


def fmt_num(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value, digits=0):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}%}"


def build_valuation_detail(row):
    if row["method"] == "VALUE":
        discount = None
        if row.get("price_to_value") is not None and pd.notna(row.get("price_to_value")):
            discount = 1 - row["price_to_value"]
        return (
            f"价值线={fmt_num(row.get('value_line'))}; "
            f"现价/价值={fmt_num(row.get('price_to_value'))}; "
            f"折价={fmt_pct(discount, 1)}; "
            f"扣非EPS={fmt_num(row.get('eps_excl'))}; "
            f"市值={fmt_num(row.get('mktcap'), 1)}亿"
        )
    if row["method"] in {"PE", "PB"}:
        return (
            f"{row['method']}={fmt_num(row.get('current_valuation'))}; "
            f"低估均值={fmt_num(row.get('low_avg'))}; "
            f"比值={fmt_num(row.get('pepb_ratio'))}; "
            f"分位={fmt_pct(row.get('valuation_percentile'), 0)}"
        )
    return "右侧趋势行业不做左侧估值"


def short_text(value, max_len=80):
    text = "" if value is None or pd.isna(value) else str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def build_html_table(rows):
    if not rows:
        return "<p>无</p>"
    table_rows = "".join(
        f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r.get('theme', '-')}</td><td>{r['selection_bucket']}</td>"
        f"<td>{r['method_name']}</td><td>{r['close']}</td><td>{r['total_score']}</td>"
        f"<td>{r.get('mainline_score', 0)}/{r.get('mainline_label', '-')}</td>"
        f"<td>{short_text(build_valuation_detail(r), 90)}</td><td>{short_text(r['risk_flags'], 90)}</td></tr>"
        for r in rows
    )
    return (
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        "<tr><th>代码</th><th>名称</th><th>主题</th><th>分组</th><th>体系</th><th>现价</th><th>综合分</th>"
        "<th>主线</th><th>估值参考</th><th>风险/瑕疵</th></tr>"
        f"{table_rows}</table>"
    )


def build_theme_summary(rows):
    if not rows:
        return "<p>主线观察：无入选样本</p>"
    df = pd.DataFrame(rows)
    if "theme" not in df.columns:
        return "<p>主线观察：无主题数据</p>"
    summary = (
        df.groupby("theme")
        .agg(
            count=("code", "count"),
            avg_mainline=("mainline_score", "mean"),
            avg_ret20=("ret20", "mean"),
            avg_ret60=("ret60", "mean"),
            names=("name", lambda s: "、".join(list(s.head(5)))),
        )
        .reset_index()
        .sort_values(["avg_mainline", "count"], ascending=[False, False])
        .head(8)
    )
    rows_html = "".join(
        f"<tr><td>{r['theme']}</td><td>{int(r['count'])}</td><td>{r['avg_mainline']:.1f}</td>"
        f"<td>{fmt_pct(r['avg_ret20'], 0)}</td><td>{fmt_pct(r['avg_ret60'], 0)}</td><td>{r['names']}</td></tr>"
        for _, r in summary.iterrows()
    )
    return (
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        "<tr><th>主题</th><th>入选数</th><th>平均主线分</th><th>20日均涨幅</th><th>60日均涨幅</th><th>代表股票</th></tr>"
        f"{rows_html}</table>"
    )


def build_daily_risk_notes(core_rows, low_value_rows, high_quality_rows, value_watch_rows):
    notes = [
        "先看板块/主题是否持续有量，再看个股是否进入右侧或回到价值线附近。",
        "价值线附近观察不等于正式买入，趋势未确认时只适合盯盘和等待右侧信号。",
        "右侧主线候选波动通常更大，追高前需要结合量能、扣抵价和波段分位复核。",
    ]
    if not core_rows and not low_value_rows:
        notes.append("今日正式低估组合较弱，说明左侧安全垫样本不足。")
    if not high_quality_rows:
        notes.append("今日右侧主线候选为空，说明趋势确认不足。")
    if value_watch_rows and not low_value_rows:
        notes.append("有价值线附近股票但低估价值正式入选少，优先观察承接而不是直接按低估买入。")
    return notes


def build_push_list(rows, top):
    if not rows:
        return "<p>无</p>"
    items = "".join(
        "<li>"
        f"{r['name']}({r['code']}) | 分={r['total_score']} | {r.get('theme', '-')} | "
        f"主线={r.get('mainline_score', 0)}/{r.get('mainline_label', '-')} | "
        f"{short_text(build_valuation_detail(r), 70)} | "
        f"{short_text(r['risk_flags'], 70)}"
        "</li>"
        for r in rows[:top]
    )
    return f"<ol>{items}</ol>"


def build_push_table(rows, top):
    return build_html_table(rows[:top])


def build_push_content(diff_html, core_rows, low_value_rows, high_quality_rows, value_watch_rows, quality_min_score, low_min_score, core_min_score, top):
    display_top = min(top, 10)
    all_rows = core_rows + low_value_rows + high_quality_rows
    risk_items = "".join(f"<li>{note}</li>" for note in build_daily_risk_notes(core_rows, low_value_rows, high_quality_rows, value_watch_rows))
    return (
        "<h2>每日交易观察</h2>"
        f"<p>展示顺序：主线主题 -> 右侧主线候选 -> 价值线附近观察 -> 左侧低估组合 -> 风险提示。"
        f"正式入选阈值：低估且高质量 >= {core_min_score}；低估价值 >= {low_min_score}；"
        f"高质量趋势 >= {quality_min_score}。每组最多展示前 {display_top} 只，全量结果见服务器 CSV。</p>"
        f"<h3>0. 较上一日变化</h3>{diff_html}"
        f"<h3>1. 主线主题</h3>{build_theme_summary(all_rows)}"
        f"<h3>2. 右侧主线候选({len(high_quality_rows)}只)</h3>"
        f"{build_push_table(high_quality_rows, display_top)}"
        f"<h3>3. 价值线附近观察({len(value_watch_rows)}只)</h3>"
        f"{build_push_table(value_watch_rows, display_top)}"
        f"<h3>4. 左侧低估组合</h3>"
        f"<h4>低估且高质量({len(core_rows)}只)</h4>{build_push_table(core_rows, display_top)}"
        f"<h4>低估价值({len(low_value_rows)}只)</h4>{build_push_table(low_value_rows, display_top)}"
        f"<h3>5. 风险提示</h3><ul>{risk_items}</ul>"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="因子模型选股")
    parser.add_argument("--top", type=int, default=30, help="每组推送和保存上一日对比使用的前N只股票")
    parser.add_argument("--quality-min-score", "--min-score", dest="quality_min_score", type=float, default=DEFAULT_QUALITY_MIN_SCORE, help="高质量趋势最低综合分")
    parser.add_argument("--low-min-score", "--value-min-score", dest="low_min_score", type=float, default=DEFAULT_LOW_VALUE_MIN_SCORE, help="低估价值最低综合分")
    parser.add_argument("--core-min-score", type=float, default=DEFAULT_CORE_MIN_SCORE, help="低估且高质量最低综合分")
    parser.add_argument("--value-min-mktcap", type=float, default=VALUE_MIN_MKTCAP, help="VALUE股票最低市值，单位亿元")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N只股票，调试用；0表示全量")
    parser.add_argument("--workers", type=int, default=4, help="并行处理进程数；1表示串行")
    parser.add_argument("--no-push", action="store_true", help="只输出结果，不推送")
    parser.add_argument("--diagnostic-top", type=int, default=DEFAULT_DIAGNOSTIC_TOP, help="额外打印和保存观察候选数量")
    parser.add_argument("--value-watch-ratio", type=float, default=VALUE_WATCH_RATIO, help="价值线附近观察池最高现价/价值线")
    parser.add_argument("--value-watch-top", type=int, default=DEFAULT_VALUE_WATCH_TOP, help="价值线附近观察池展示数量；0表示全部")
    return parser.parse_args()


def print_result_section(title, rows, top):
    print(f"\n--- {title}({len(rows)}只，显示前{min(top, len(rows))}只) ---")
    if not rows:
        print("无")
        return
    display_rows = []
    for row in rows[:top]:
        display = row.copy()
        display["valuation_detail"] = build_valuation_detail(row)
        display_rows.append(display)
    cols = [
        "code", "name", "theme", "selection_bucket", "valuation_state", "method_name", "selection_mode", "close",
        "total_score", "core_score", "low_value_score", "high_quality_score", "valuation_score", "raw_valuation_score",
        "quality_score", "trend_score", "liquidity_score", "mainline_score", "mainline_label", "valuation_detail",
        "mainline_ref", "technical_ref", "risk_flags",
    ]
    print(pd.DataFrame(display_rows)[cols].to_string(index=False))


def print_theme_summary(rows):
    if not rows:
        print("\n--- 主线观察：无入选样本 ---")
        return
    df = pd.DataFrame(rows)
    if "theme" not in df.columns:
        print("\n--- 主线观察：无主题数据 ---")
        return
    summary = (
        df.groupby("theme")
        .agg(
            count=("code", "count"),
            avg_mainline=("mainline_score", "mean"),
            avg_ret20=("ret20", "mean"),
            avg_ret60=("ret60", "mean"),
            names=("name", lambda s: "、".join(list(s.head(5)))),
        )
        .reset_index()
        .sort_values(["avg_mainline", "count"], ascending=[False, False])
        .head(8)
    )
    print("\n--- 主线观察 ---")
    print(summary.to_string(index=False, formatters={
        "avg_mainline": lambda v: f"{v:.1f}",
        "avg_ret20": lambda v: fmt_pct(v, 0),
        "avg_ret60": lambda v: fmt_pct(v, 0),
    }))


def print_value_watch_summary(rows, top):
    print(f"\n--- 价值线附近观察({len(rows)}只，显示前{min(top, len(rows))}只) ---")
    if not rows:
        print("无")
        return
    display_rows = []
    for row in rows[:top]:
        display = row.copy()
        display["valuation_detail"] = build_valuation_detail(row)
        display_rows.append(display)
    cols = [
        "code", "name", "theme", "close", "price_to_value", "value_line", "total_score",
        "quality_score", "trend_score", "liquidity_score", "selection_bucket", "block_reason",
        "valuation_detail", "risk_flags",
    ]
    print(pd.DataFrame(display_rows)[cols].to_string(index=False, formatters={
        "price_to_value": lambda v: fmt_num(v, 2),
        "value_line": lambda v: fmt_num(v, 2),
    }))


def print_compact_rows(title, rows, top, columns=None):
    print(f"\n{title}({len(rows)}只，显示前{min(top, len(rows))}只)")
    if not rows:
        print("无")
        return
    base_columns = [
        "code", "name", "theme", "selection_bucket", "close", "total_score",
        "price_to_value", "value_line", "quality_score", "trend_score",
        "liquidity_score", "mainline_score", "mainline_label", "risk_flags",
    ]
    cols = columns or base_columns
    frame = pd.DataFrame(rows[:top])
    cols = [col for col in cols if col in frame.columns]
    print(frame[cols].to_string(index=False, formatters={
        "price_to_value": lambda v: fmt_num(v, 2),
        "value_line": lambda v: fmt_num(v, 2),
        "ret20": lambda v: fmt_pct(v, 0),
        "ret60": lambda v: fmt_pct(v, 0),
    }))


def print_daily_report(today_str, core_rows, low_value_rows, high_quality_rows, value_watch_rows, top):
    all_rows = core_rows + low_value_rows + high_quality_rows
    display_top = min(top, 10)
    print("\n================ 每日交易观察 ================")
    print(f"交易日: {today_str}")
    print("阅读顺序: 主线主题 -> 右侧主线候选 -> 价值线附近观察 -> 左侧低估组合 -> 风险提示")
    print(
        f"正式入选: {len(all_rows)}只 | 右侧主线 {len(high_quality_rows)}只 | "
        f"价值线附近观察 {len(value_watch_rows)}只 | 低估且高质量 {len(core_rows)}只 | 低估价值 {len(low_value_rows)}只"
    )

    print_theme_summary(all_rows)
    print_compact_rows("2. 右侧主线候选", high_quality_rows, display_top, [
        "code", "name", "theme", "close", "total_score", "quality_score", "trend_score",
        "liquidity_score", "mainline_score", "mainline_label", "ret20", "ret60", "risk_flags",
    ])
    print_compact_rows("3. 价值线附近观察", value_watch_rows, display_top, [
        "code", "name", "theme", "close", "price_to_value", "value_line", "total_score",
        "quality_score", "trend_score", "liquidity_score", "block_reason", "risk_flags",
    ])
    print_compact_rows("4.1 低估且高质量", core_rows, display_top)
    print_compact_rows("4.2 低估价值", low_value_rows, display_top)

    print("\n5. 风险提示")
    for idx, note in enumerate(build_daily_risk_notes(core_rows, low_value_rows, high_quality_rows, value_watch_rows), start=1):
        print(f"{idx}. {note}")


def main():
    global BENCHMARK_DF
    args = parse_args()
    lg = bs.login()
    if lg.error_code != "0":
        print("登录失败:", lg.error_msg)
        sys.exit(1)

    today_str, df_stocks = get_trade_day_and_universe()
    if not today_str or df_stocks.empty:
        print("无法获取股票列表")
        bs.logout()
        sys.exit(1)
    industry_map = get_industry_map()
    year, quarter = get_latest_quarter(today_str)
    BENCHMARK_DF = get_benchmark_history(today_str)

    if args.limit > 0:
        df_stocks = df_stocks.head(args.limit)

    print(f"最新交易日: {today_str}")
    print(
        f"候选股票: {len(df_stocks)} | 财报终点: {year}Q{quarter} | "
        f"核心最低分: {args.core_min_score} | 高质量最低分: {args.quality_min_score} | "
        f"低估最低分: {args.low_min_score} | VALUE最低市值: {args.value_min_mktcap}亿"
    )

    tasks = []
    for _, stock in df_stocks.iterrows():
        code, name = stock["code"], stock["code_name"]
        industry = industry_map.get(code, "")
        method = classify_method(industry)
        tasks.append((code, name, industry, method, today_str, year, quarter, args.value_min_mktcap))

    rows = []
    all_rows = []
    skipped = []
    total = len(tasks)
    workers = max(1, args.workers)
    if workers > 1:
        bs.logout()
        with multiprocessing.Pool(processes=workers, initializer=init_worker, initargs=(today_str,)) as pool:
            for idx, result in enumerate(pool.imap_unordered(score_stock_task, tasks), start=1):
                if result["error"]:
                    print(f"  {result['code']} {result['name']} 处理失败: {result['error']}")
                    skipped.append({
                        "code": result["code"],
                        "name": result["name"],
                        "skip_reason": f"异常: {result['error']}",
                    })
                row = result["row"]
                if row:
                    selected = pass_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                    diagnostic_candidate = pass_diagnostic_gate(row)
                    row["selected"] = selected
                    row["diagnostic_candidate"] = diagnostic_candidate
                    row["block_reason"] = get_block_reason(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                    all_rows.append(row)
                    if selected:
                        rows.append(row)
                        gate = get_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                        print(
                            f"  {row['code']} {row['name']} 入选 | {row['selection_bucket']} | {row['method_name']} | "
                            f"综合分={row['total_score']}/{gate} | {build_valuation_detail(row)} | 瑕疵={row['risk_flags']}"
                        )
                elif result.get("skip_reason"):
                    skipped.append({
                        "code": result["code"],
                        "name": result["name"],
                        "skip_reason": result.get("skip_reason"),
                    })
                if idx % 100 == 0:
                    print(f"  进度: {idx}/{total}, 已入选 {len(rows)} 只")
    else:
        for idx, task in enumerate(tasks, start=1):
            result = score_stock_task(task)
            if result["error"]:
                print(f"  {result['code']} {result['name']} 处理失败: {result['error']}")
                skipped.append({
                    "code": result["code"],
                    "name": result["name"],
                    "skip_reason": f"异常: {result['error']}",
                })
            row = result["row"]
            if row:
                selected = pass_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                diagnostic_candidate = pass_diagnostic_gate(row)
                row["selected"] = selected
                row["diagnostic_candidate"] = diagnostic_candidate
                row["block_reason"] = get_block_reason(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                all_rows.append(row)
                if selected:
                    rows.append(row)
                    gate = get_score_gate(row, args.quality_min_score, args.low_min_score, args.core_min_score)
                    print(
                        f"  {row['code']} {row['name']} 入选 | {row['selection_bucket']} | {row['method_name']} | "
                        f"综合分={row['total_score']}/{gate} | {build_valuation_detail(row)} | 瑕疵={row['risk_flags']}"
                    )
            elif result.get("skip_reason"):
                skipped.append({
                    "code": result["code"],
                    "name": result["name"],
                    "skip_reason": result.get("skip_reason"),
                })
            if idx % 100 == 0:
                print(f"  进度: {idx}/{total}, 已入选 {len(rows)} 只")
        bs.logout()

    rows.sort(key=lambda x: x["total_score"], reverse=True)
    core_rows = [r for r in rows if r["selection_bucket"] == CORE_BUCKET]
    low_value_rows = [r for r in rows if r["selection_bucket"] == LOW_VALUE_BUCKET]
    high_quality_rows = [r for r in rows if r["selection_bucket"] == HIGH_QUALITY_BUCKET]
    top_rows = core_rows[:args.top] + low_value_rows[:args.top] + high_quality_rows[:args.top]
    csv_path = save_csv(today_str, rows)
    if csv_path:
        print(f"结果已保存: {csv_path}")
    diagnostic_path = save_diagnostic_csv(today_str, all_rows, skipped, args.diagnostic_top)
    if diagnostic_path:
        print(f"诊断结果已保存: {diagnostic_path}")

    print(f"\n共筛选出 {len(rows)} 只：低估且高质量 {len(core_rows)} 只，低估价值 {len(low_value_rows)} 只，高质量趋势 {len(high_quality_rows)} 只")
    value_watch_rows = get_value_watch_rows(all_rows, args.value_watch_ratio, args.value_watch_top)
    print_daily_report(today_str, core_rows, low_value_rows, high_quality_rows, value_watch_rows, args.top)
    print_diagnostic_summary(all_rows, skipped, args.diagnostic_top)

    last_dict = load_last_result(LAST_RESULT_FILE)
    diff_html = build_diff_html(last_dict, top_rows) if last_dict else "<p>首次运行，无历史对比</p>"
    save_current_result(LAST_RESULT_FILE, today_str, top_rows)

    if args.no_push:
        print("已按 --no-push 跳过推送")
        return

    if not top_rows:
        send_pushplus(f"{today_str} 因子选股", "今日无符合因子模型条件股票" + diff_html)
        return

    content = build_push_content(diff_html, core_rows, low_value_rows, high_quality_rows, value_watch_rows, args.quality_min_score, args.low_min_score, args.core_min_score, args.top)
    send_pushplus(f"{today_str} 因子选股({len(top_rows)}/{len(rows)}只)", content)


if __name__ == "__main__":
    main()
