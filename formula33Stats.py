# -*- coding: utf-8 -*-
"""
33 formula market-structure statistics.

Formula:
KD := (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100;
K := SMA(KD, 3, 1);
WR1 := 100 * (HHV(HIGH, 10) - CLOSE) / (HHV(HIGH, 10) - LLV(LOW, 10));
WR2 := 100 * (HHV(HIGH, 20) - CLOSE) / (HHV(HIGH, 20) - LLV(LOW, 20));
KD80 := K > 80;
WR3 := WR1 < 20 AND WR2 < 20;
RSI70 := SMA(MAX(CLOSE - REF(CLOSE, 1), 0), 9, 1)
    / SMA(ABS(CLOSE - REF(CLOSE, 1)), 9, 1) * 100 > 70;
MKT_CAP := FINANCE(40) / 10000 > 100;
LIST_DAYS := FINANCE(42) > 300;
BASE := KD80 AND WR3 AND RSI70 AND MKT_CAP AND LIST_DAYS;
XG: COUNT(BASE, 5) = 5;

The script records the number of Shanghai/Shenzhen A shares matching XG on
the latest N trading days. Rising for 3/5 days means initial/confirmed
structure improvement; falling for 3/5 days means initial/confirmed weakness.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import multiprocessing
import os
import time
from datetime import datetime

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from trade_utils import get_project_path


OUTPUT_DIR = get_project_path("板块观察")
ADJUST_FLAG_QFQ = "2"
SHARE_CACHE_FILE = get_project_path(".cache/formula33_share_capital.json")


def parse_args():
    parser = argparse.ArgumentParser(description="33公式沪深A股市场结构统计")
    parser.add_argument("--lookback", type=int, default=21, help="统计最近N个交易日")
    parser.add_argument("--history-days", type=int, default=90, help="为指标计算额外拉取的自然日长度")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="调试用，只处理前N只")
    parser.add_argument("--price-source", choices=["baostock", "akshare"], default="akshare", help="前复权K线来源")
    parser.add_argument("--min-mktcap", type=float, default=100.0, help="最低总市值，单位亿元")
    parser.add_argument("--min-list-days", type=int, default=300, help="最低上市天数")
    parser.add_argument(
        "--market-cap-source",
        choices=["auto", "tushare", "akshare", "akshare-capital", "none"],
        default="auto",
        help="总市值来源；none 仅用于临时复核技术指标数量，会跳过 FINANCE(40) 过滤",
    )
    parser.add_argument("--sample", action="store_true", help="生成离线样例，不访问网络")
    return parser.parse_args()


def to_bs_code(raw_code):
    code = str(raw_code).strip()
    if "." in code:
        return code
    low = code.lower()
    if low.startswith("sh") and len(code) >= 8:
        return f"sh.{code[-6:]}"
    if low.startswith("sz") and len(code) >= 8:
        return f"sz.{code[-6:]}"
    if low.startswith("bj") and len(code) >= 8:
        return f"bj.{code[-6:]}"
    code = code.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def pure_code(bs_code):
    return str(bs_code).split(".")[-1]


def get_trade_dates(lookback, extra_days):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(days=extra_days)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return []
    df.columns = rs.fields
    df = df[df["is_trading_day"] == "1"]
    return df["calendar_date"].tail(lookback).tolist()


def get_trade_dates_akshare(lookback, extra_days):
    df = ak.tool_trade_date_hist_sina()
    if df is None or df.empty:
        return []
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=extra_days)
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].dropna(subset=["trade_date"])
    return df["trade_date"].dt.strftime("%Y-%m-%d").tail(lookback).tolist()


def get_universe(latest_date):
    rs = bs.query_all_stock(day=latest_date)
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
    df = df[mask].copy()
    if "tradeStatus" in df.columns:
        df = df[df["tradeStatus"] != "0"]
    if "code_name" in df.columns:
        df = df[~df["code_name"].astype(str).str.contains("ST", na=False)]
    return df[["code", "code_name"]].drop_duplicates("code")


def get_universe_with_fallback(trade_dates):
    for date in reversed(trade_dates):
        df = get_universe(date)
        if not df.empty:
            return date, df
    return "", pd.DataFrame()


def get_universe_akshare():
    df = ak.stock_info_a_code_name()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"code": "raw_code", "name": "code_name"})
    df["code"] = df["raw_code"].map(to_bs_code)
    mask = (
        df["code"].str.startswith("sh.60")
        | df["code"].str.startswith("sh.68")
        | df["code"].str.startswith("sz.00")
        | df["code"].str.startswith("sz.30")
    )
    df = df[mask].copy()
    df = df[~df["code_name"].astype(str).str.contains("ST|退", na=False)]
    return df[["code", "code_name"]].drop_duplicates("code")


def load_stock_basic():
    try:
        rs = bs.query_stock_basic()
        df = rs.get_data()
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = rs.fields
    return df


def load_stock_basic_akshare():
    rows = []
    for symbol in ["\u4e3b\u677fA\u80a1", "\u79d1\u521b\u677f"]:
        try:
            sh = ak.stock_info_sh_name_code(symbol=symbol)
            for _, row in sh.iterrows():
                rows.append({
                    "code": f"sh.{str(row.get('证券代码')).zfill(6)}",
                    "ipoDate": row.get("上市日期"),
                })
        except Exception:
            continue
    try:
        sz = ak.stock_info_sz_name_code(symbol="\u0041\u80a1\u5217\u8868")
        for _, row in sz.iterrows():
            rows.append({
                "code": f"sz.{str(row.get('A股代码')).zfill(6)}",
                "ipoDate": row.get("A股上市日期"),
            })
    except Exception:
        pass
    return pd.DataFrame(rows)


def get_tushare_token():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    token_file = os.environ.get("TUSHARE_TOKEN_FILE", get_project_path(".tushare_token"))
    try:
        with open(token_file, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def load_market_caps_from_tushare(trade_date):
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("未安装 tushare，请先 pip install tushare") from exc
    token = get_tushare_token()
    if not token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 或 .tushare_token")
    pro = ts.pro_api(token)
    ts_date = str(trade_date).replace("-", "")
    df = pro.daily_basic(trade_date=ts_date, fields="ts_code,total_mv")
    if df is None or df.empty:
        raise RuntimeError(f"tushare daily_basic 在 {trade_date} 无数据")
    caps = {}
    for _, row in df.iterrows():
        ts_code = str(row.get("ts_code", ""))
        code = ts_code.split(".")[0]
        if ts_code.endswith(".SH"):
            bs_code = f"sh.{code}"
        elif ts_code.endswith(".SZ"):
            bs_code = f"sz.{code}"
        else:
            bs_code = to_bs_code(code)
        total_mv_wan = pd.to_numeric(row.get("total_mv"), errors="coerce")
        if pd.notna(total_mv_wan):
            caps[bs_code] = float(total_mv_wan) / 10000.0
    return caps


def load_market_caps_from_akshare():
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as exc:
        raise RuntimeError(f"无法获取东方财富A股总市值：{exc}")
    if df is None or df.empty:
        return {}
    code_col = "代码" if "代码" in df.columns else None
    cap_col = "总市值" if "总市值" in df.columns else None
    if not code_col or not cap_col:
        raise RuntimeError("东方财富行情表缺少 代码/总市值 字段")
    caps = {}
    for _, row in df.iterrows():
        cap = pd.to_numeric(row.get(cap_col), errors="coerce")
        if pd.isna(cap):
            continue
        caps[to_bs_code(row.get(code_col))] = float(cap) / 100000000.0
    return caps


def load_share_cache():
    try:
        if os.path.exists(SHARE_CACHE_FILE):
            with open(SHARE_CACHE_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def save_share_cache(cache):
    try:
        os.makedirs(os.path.dirname(SHARE_CACHE_FILE), exist_ok=True)
        with open(SHARE_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def parse_share(value):
    text = str(value).replace(",", "").strip()
    val = pd.to_numeric(text, errors="coerce")
    return None if pd.isna(val) else float(val)


def em_symbol_from_bs(code):
    pure = pure_code(code)
    if str(code).startswith("sh."):
        return f"{pure}.SH"
    if str(code).startswith("sz."):
        return f"{pure}.SZ"
    return pure


def fetch_total_share_from_gbjg(code):
    try:
        gb = ak.stock_zh_a_gbjg_em(symbol=em_symbol_from_bs(code))
        if gb is None or gb.empty or "总股本" not in gb.columns:
            return code, None
        gb = gb.copy()
        gb["变更日期_dt"] = pd.to_datetime(gb["变更日期"], errors="coerce")
        gb = gb.sort_values("变更日期_dt", ascending=False)
        return code, parse_share(gb.iloc[0].get("总股本"))
    except Exception:
        return code, None


def load_market_caps_from_akshare_capital(universe):
    spot = ak.stock_zh_a_spot()
    if spot is None or spot.empty:
        raise RuntimeError("akshare stock_zh_a_spot 无数据，无法取得当前价格")
    code_col = "代码"
    price_col = "最新价"
    price_map = {}
    for _, row in spot.iterrows():
        price = pd.to_numeric(row.get(price_col), errors="coerce")
        if pd.notna(price) and price > 0:
            price_map[to_bs_code(row.get(code_col))] = float(price)

    share_map = {}
    try:
        sz = ak.stock_info_sz_name_code(symbol="\u0041\u80a1\u5217\u8868")
        for _, row in sz.iterrows():
            code = f"sz.{str(row.get('A股代码')).zfill(6)}"
            share = parse_share(row.get("A股总股本"))
            if share:
                share_map[code] = share
    except Exception as exc:
        print(f"深市总股本读取失败: {exc}")

    cache = load_share_cache()
    missing_sh = []
    for _, row in universe.iterrows():
        code = row["code"]
        if code in share_map:
            continue
        cached = cache.get(code)
        if cached and cached.get("total_share"):
            share_map[code] = float(cached["total_share"])
            continue
        if not code.startswith("sh."):
            continue
        missing_sh.append(code)
    if missing_sh:
        print(f"沪市总股本需补充 {len(missing_sh)} 只，使用 akshare 股本结构并发读取")
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(fetch_total_share_from_gbjg, code) for code in missing_sh]
            for idx, future in enumerate(as_completed(futures), start=1):
                code, share = future.result()
                if share:
                    share_map[code] = share
                    cache[code] = {"total_share": share, "updated_at": datetime.now().strftime("%Y-%m-%d")}
                if idx % 200 == 0:
                    print(f"  沪市总股本进度 {idx}/{len(missing_sh)}")
    save_share_cache(cache)

    caps = {}
    for code, share in share_map.items():
        price = price_map.get(code)
        if price:
            caps[code] = share * price / 100000000.0
    if not caps:
        raise RuntimeError("akshare 股本结构法未生成有效总市值")
    return caps


def load_market_caps(source, trade_date, universe=None):
    if source == "none":
        print("已按 --market-cap-source none 跳过 FINANCE(40) 市值过滤，仅用于技术条件复核。")
        return {}, "none"

    errors = []
    sources = ["tushare", "akshare", "akshare-capital"] if source == "auto" else [source]
    for item in sources:
        try:
            if item == "tushare":
                caps = load_market_caps_from_tushare(trade_date)
            elif item == "akshare":
                caps = load_market_caps_from_akshare()
            elif item == "akshare-capital":
                if universe is None:
                    raise RuntimeError("akshare-capital 需要股票池 universe")
                caps = load_market_caps_from_akshare_capital(universe)
            else:
                continue
            if caps:
                print(f"总市值来源: {item}，记录数 {len(caps)}")
                return caps, item
        except Exception as exc:
            errors.append(f"{item}: {exc}")
    raise RuntimeError("无法获取总市值数据；" + " | ".join(errors))


def tdx_sma(series, n, m=1):
    """Tongdaxin SMA(X,N,M): Y=(M*X+(N-M)*Y')/N."""
    prev = np.nan
    values = []
    for raw in pd.to_numeric(series, errors="coerce"):
        if pd.isna(raw):
            values.append(np.nan)
            continue
        if pd.isna(prev):
            prev = raw
        else:
            prev = (m * raw + (n - m) * prev) / n
        values.append(prev)
    return pd.Series(values, index=series.index)


def calc_kdj_k(df, n=9):
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    kd = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    return tdx_sma(kd, 3, 1)


def calc_wr(df, n):
    high_n = df["high"].rolling(n, min_periods=n).max()
    low_n = df["low"].rolling(n, min_periods=n).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * 100


def calc_rsi(series, n=9):
    diff = series.diff()
    up = diff.clip(lower=0)
    abs_diff = diff.abs()
    return tdx_sma(up, n, 1) / tdx_sma(abs_diff, n, 1).replace(0, np.nan) * 100


def init_worker():
    bs.login()


def load_kline_baostock(code, start_date, end_date):
    fields = "date,code,open,high,low,close,volume,tradestatus"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=ADJUST_FLAG_QFQ,
    )
    df = rs.get_data()
    if rs.error_code != "0" or df.empty:
        return pd.DataFrame()
    df.columns = rs.fields
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == "1"]
    return df


def load_kline_akshare(code, start_date, end_date):
    symbol = pure_code(code)
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=str(start_date).replace("-", ""),
        end_date=str(end_date).replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename(columns={
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    })


def fetch_one_stock(task):
    code, name, start_date, end_date, date_set, mktcap_yi, ipo_date, min_mktcap, min_list_days, sleep, price_source = task
    if sleep > 0:
        time.sleep(sleep)
    if ipo_date is None:
        return []
    if min_mktcap is not None and mktcap_yi is None:
        return []
    try:
        if price_source == "akshare":
            df = load_kline_akshare(code, start_date, end_date)
        else:
            df = load_kline_baostock(code, start_date, end_date)
    except Exception as exc:
        return [{"code": code, "name": name, "error": str(exc)}]
    if df.empty:
        return []
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        return []

    k = calc_kdj_k(df)
    wr10 = calc_wr(df, 10)
    wr20 = calc_wr(df, 20)
    rsi9 = calc_rsi(df["close"], 9)
    base = (k > 80) & (wr10 < 20) & (wr20 < 20) & (rsi9 > 70)
    xg = base.rolling(5, min_periods=5).sum() == 5
    latest_close = df["close"].iloc[-1]
    if min_mktcap is not None and (not latest_close or pd.isna(latest_close)):
        return []
    ipo_ts = pd.to_datetime(ipo_date, errors="coerce")
    if pd.isna(ipo_ts):
        return []

    hits = []
    for idx, row in df.iterrows():
        row = df.loc[idx]
        if row["date"] not in date_set:
            continue
        row_date = pd.to_datetime(row["date"], errors="coerce")
        list_days = (row_date - ipo_ts).days if pd.notna(row_date) else None
        if min_mktcap is None:
            mktcap_at_date = np.nan
        else:
            mktcap_at_date = float(mktcap_yi) * float(row["close"]) / float(latest_close)
        if list_days is None or list_days <= min_list_days:
            continue
        if min_mktcap is not None and mktcap_at_date <= min_mktcap:
            continue
        record = {
            "date": row["date"],
            "code": code,
            "name": name,
            "close": row["close"],
            "mktcap_yi": round(mktcap_at_date, 2) if pd.notna(mktcap_at_date) else np.nan,
            "list_days": int(list_days),
            "kdj_k": round(float(k.loc[idx]), 2),
            "wr10": round(float(wr10.loc[idx]), 2),
            "wr20": round(float(wr20.loc[idx]), 2),
            "rsi9": round(float(rsi9.loc[idx]), 2),
        }
        if bool(base.loc[idx]):
            base_record = record.copy()
            base_record["signal_type"] = "BASE"
            hits.append(base_record)
        if bool(xg.loc[idx]):
            xg_record = record.copy()
            xg_record["signal_type"] = "XG"
            hits.append(xg_record)
    return hits


def calc_streaks(counts):
    rows = []
    up_streak = 0
    down_streak = 0
    prev = None
    for date, count in counts:
        change = 0 if prev is None else count - prev
        if prev is None or change == 0:
            up_streak = 0
            down_streak = 0
        elif change > 0:
            up_streak += 1
            down_streak = 0
        else:
            down_streak += 1
            up_streak = 0
        if up_streak >= 5:
            signal = "结构转好确认，右侧成功率提升"
        elif up_streak >= 3:
            signal = "结构初步转好"
        elif down_streak >= 5:
            signal = "结构转坏确认，右侧成功率下降"
        elif down_streak >= 3:
            signal = "结构初步转坏"
        else:
            signal = "观察"
        rows.append({
            "date": date,
            "count": count,
            "change": change,
            "up_streak": up_streak,
            "down_streak": down_streak,
            "signal": signal,
        })
        prev = count
    return pd.DataFrame(rows)


def save_workbook(summary, hits, sample=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_sample" if sample else ""
    path = os.path.join(OUTPUT_DIR, f"formula33_stats_{stamp}{suffix}.xlsx")
    csv_path = os.path.join(OUTPUT_DIR, f"formula33_stats_{stamp}{suffix}.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    wb = Workbook()
    ws = wb.active
    ws.title = "33公式日统计"
    headers = ["date", "base_count", "count", "change", "up_streak", "down_streak", "signal"]
    ws.append(headers)
    for row in summary[headers].to_dict("records"):
        ws.append([row[h] for h in headers])

    ws2 = wb.create_sheet("横向统计")
    ws2.append(["指标"] + summary["date"].tolist())
    ws2.append(["BASE数量"] + summary.get("base_count", pd.Series([np.nan] * len(summary))).tolist())
    ws2.append(["XG数量"] + summary["count"].tolist())
    ws2.append(["较前日变化"] + summary["change"].tolist())
    ws2.append(["连续上升"] + summary["up_streak"].tolist())
    ws2.append(["连续下降"] + summary["down_streak"].tolist())
    ws2.append(["结构信号"] + summary["signal"].tolist())

    ws3 = wb.create_sheet("命中股票")
    if hits.empty:
        ws3.append(["signal_type", "date", "code", "name", "close", "mktcap_yi", "list_days", "kdj_k", "wr10", "wr20", "rsi9"])
    else:
        cols = ["signal_type", "date", "code", "name", "close", "mktcap_yi", "list_days", "kdj_k", "wr10", "wr20", "rsi9"]
        ws3.append(cols)
        for row in hits[cols].to_dict("records"):
            ws3.append([row.get(col) for col in cols])

    yellow = PatternFill("solid", fgColor="FFF2CC")
    green = PatternFill("solid", fgColor="E2F0D9")
    red = PatternFill("solid", fgColor="F4CCCC")
    thin = Side(style="thin", color="666666")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                if cell.row == 1:
                    cell.font = Font(bold=True)
                    cell.fill = yellow
                if isinstance(cell.value, str) and "转好" in cell.value:
                    cell.fill = green
                if isinstance(cell.value, str) and "转坏" in cell.value:
                    cell.fill = red
        for col in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(col)].width = 18
    wb.save(path)
    return path, csv_path


def build_sample(lookback):
    dates = pd.bdate_range(end=pd.Timestamp("2026-06-24"), periods=lookback).strftime("%Y-%m-%d").tolist()
    counts = [42, 45, 48, 52, 57, 60, 58, 55, 51, 48, 46, 49, 53, 58, 64, 70, 74, 72, 69, 66, 63][-lookback:]
    summary = calc_streaks(list(zip(dates, counts)))
    summary["base_count"] = [value + 80 for value in counts]
    summary = summary[["date", "base_count", "count", "change", "up_streak", "down_streak", "signal"]]
    hit_rows = []
    for date, count in zip(dates, counts):
        for idx in range(min(count, 12)):
            hit_rows.append({
                "signal_type": "XG",
                "date": date,
                "code": f"sz.30{idx:04d}",
                "name": f"33样本{idx + 1}",
                "close": 10 + idx,
                "mktcap_yi": 120 + idx * 5,
                "list_days": 500 + idx,
                "kdj_k": 82 + idx % 10,
                "wr10": 8 + idx % 8,
                "wr20": 9 + idx % 7,
                "rsi9": 72 + idx % 12,
            })
    return summary, pd.DataFrame(hit_rows)


def main():
    args = parse_args()
    if args.sample:
        summary, hits = build_sample(args.lookback)
        xlsx_path, csv_path = save_workbook(summary, hits, sample=True)
        print(f"Excel已保存: {xlsx_path}")
        print(f"CSV已保存: {csv_path}")
        print(summary.tail(8).to_string(index=False))
        return

    lg = bs.login()
    bs_available = lg.error_code == "0"
    if not bs_available:
        print(f"Baostock不可用，改用 akshare 元数据: {lg.error_msg}")
        if args.price_source == "baostock":
            raise SystemExit("Baostock不可用时不能使用 --price-source baostock，请改用 --price-source akshare")
    try:
        trade_dates = get_trade_dates(args.lookback, args.history_days + 40) if bs_available else []
        if len(trade_dates) < args.lookback:
            trade_dates = get_trade_dates_akshare(args.lookback, args.history_days + 40)
        if len(trade_dates) < args.lookback:
            raise SystemExit("交易日不足，无法统计最近21个交易日")
        latest_date = trade_dates[-1]
        start_date = (pd.to_datetime(trade_dates[0]) - pd.DateOffset(days=args.history_days)).strftime("%Y-%m-%d")
        if bs_available:
            universe_date, universe = get_universe_with_fallback(trade_dates)
        else:
            universe_date, universe = latest_date, get_universe_akshare()
        if universe.empty:
            raise SystemExit("无法获取沪深A股股票池")
        if universe_date != latest_date:
            print(f"股票池使用 {universe_date}，统计交易日仍使用 {latest_date} 之前最近 {len(trade_dates)} 个交易日")
        if args.limit > 0:
            universe = universe.head(args.limit)
        basic = load_stock_basic() if bs_available else pd.DataFrame()
        if basic.empty:
            basic = load_stock_basic_akshare()
        list_date_map = {}
        if not basic.empty and "code" in basic.columns and "ipoDate" in basic.columns:
            list_date_map = dict(zip(basic["code"], basic["ipoDate"]))
        try:
            cap_map, cap_source = load_market_caps(args.market_cap_source, latest_date, universe)
        except Exception as exc:
            raise SystemExit(f"{exc}\n该公式需要 FINANCE(40)>100亿；可配置 Tushare token，或网络可用时使用 akshare。")
        min_mktcap = None if cap_source == "none" else args.min_mktcap

        date_set = set(trade_dates)
        tasks = []
        for _, row in universe.iterrows():
            code = row["code"]
            ipo = list_date_map.get(code)
            tasks.append((
                code,
                row.get("code_name", ""),
                start_date,
                latest_date,
                date_set,
                cap_map.get(code),
                ipo,
                min_mktcap,
                args.min_list_days,
                args.sleep,
                args.price_source,
            ))

        hits = []
        workers = max(1, args.workers)
        if workers > 1:
            if bs_available:
                bs.logout()
            initializer = init_worker if args.price_source == "baostock" else None
            with multiprocessing.Pool(processes=workers, initializer=initializer) as pool:
                for idx, result in enumerate(pool.imap_unordered(fetch_one_stock, tasks), start=1):
                    hits.extend([item for item in result if not item.get("error")])
                    if idx % 200 == 0:
                        print(f"进度 {idx}/{len(tasks)}，命中记录 {len(hits)}")
        else:
            for idx, task in enumerate(tasks, start=1):
                hits.extend([item for item in fetch_one_stock(task) if not item.get("error")])
                if idx % 200 == 0:
                    print(f"进度 {idx}/{len(tasks)}，命中记录 {len(hits)}")
            if bs_available:
                bs.logout()

        hits_df = pd.DataFrame(hits)
        if hits_df.empty:
            counts = [(date, 0) for date in trade_dates]
            base_counts_by_date = {}
        else:
            xg_hits = hits_df[hits_df["signal_type"] == "XG"]
            base_hits = hits_df[hits_df["signal_type"] == "BASE"]
            counts_by_date = xg_hits.groupby("date").size().to_dict()
            base_counts_by_date = base_hits.groupby("date").size().to_dict()
            counts = [(date, int(counts_by_date.get(date, 0))) for date in trade_dates]
        summary = calc_streaks(counts)
        summary["base_count"] = summary["date"].map(lambda d: int(base_counts_by_date.get(d, 0)))
        summary = summary[["date", "base_count", "count", "change", "up_streak", "down_streak", "signal"]]
        xlsx_path, csv_path = save_workbook(summary, hits_df)
        print(f"Excel已保存: {xlsx_path}")
        print(f"CSV已保存: {csv_path}")
        print(summary.to_string(index=False))
    finally:
        if bs_available:
            try:
                bs.logout()
            except Exception:
                pass


if __name__ == "__main__":
    main()
