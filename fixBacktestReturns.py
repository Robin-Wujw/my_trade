# -*- coding: utf-8 -*-
"""
Repair q1Backtest CSV forward returns using akshare historical A-share bars.

The selection step can be kept intact while this script recalculates buy/end
raw and front-adjusted closes. It uses the last available trading day not later
than the requested date.
"""
import argparse
import os
import time

import akshare as ak
import baostock as bs
import pandas as pd

from trade_utils import get_project_path


def parse_args():
    parser = argparse.ArgumentParser(description="补齐回测CSV的前复权收益")
    parser.add_argument("files", nargs="+", help="q1Backtest 输出 CSV")
    parser.add_argument("--buy-date", required=True, help="买入日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--source", choices=["baostock", "akshare"], default="baostock")
    parser.add_argument("--sleep", type=float, default=0.02, help="逐股请求间隔秒数")
    return parser.parse_args()


def pure_code(code):
    text = str(code).strip()
    if "." in text:
        return text.split(".", 1)[1]
    return text[-6:].zfill(6)


def normalize_code(code):
    text = str(code).strip().lower()
    if text.startswith(("sh.", "sz.")):
        return text
    symbol = pure_code(text)
    return f"sh.{symbol}" if symbol.startswith(("6", "9")) else f"sz.{symbol}"


def close_pair(symbol, buy_date, end_date, adjust):
    start = buy_date.replace("-", "")
    end = end_date.replace("-", "")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        return None, None, None, None
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期")
    if df.empty:
        return None, None, None, None

    buy_dt = pd.to_datetime(buy_date)
    end_dt = pd.to_datetime(end_date)
    buy_rows = df[df["日期"] >= buy_dt]
    end_rows = df[df["日期"] <= end_dt]
    if buy_rows.empty or end_rows.empty:
        return None, None, None, None
    buy_row = buy_rows.iloc[0]
    end_row = end_rows.iloc[-1]
    return (
        float(buy_row["收盘"]),
        float(end_row["收盘"]),
        buy_row["日期"].strftime("%Y-%m-%d"),
        end_row["日期"].strftime("%Y-%m-%d"),
    )


def close_pair_baostock(code, buy_date, end_date, adjustflag):
    fields = "date,close"
    rs = bs.query_history_k_data_plus(
        normalize_code(code),
        fields,
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


def repair_file(path, buy_date, end_date, sleep_seconds, source):
    df = pd.read_csv(path)
    rows = []
    errors = []
    cache = {}
    for idx, row in df.iterrows():
        data = row.to_dict()
        symbol = pure_code(data.get("code", ""))
        if not symbol:
            errors.append((idx, data.get("code"), "empty code"))
            rows.append(data)
            continue
        try:
            if symbol not in cache:
                if source == "baostock":
                    raw = close_pair_baostock(data.get("code", symbol), buy_date, end_date, "3")
                    qfq = close_pair_baostock(data.get("code", symbol), buy_date, end_date, "2")
                else:
                    raw = close_pair(symbol, buy_date, end_date, "")
                    qfq = close_pair(symbol, buy_date, end_date, "qfq")
                cache[symbol] = raw, qfq
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            raw, qfq = cache[symbol]
            raw_buy, raw_end, raw_buy_date, raw_end_date = raw
            qfq_buy, qfq_end, qfq_buy_date, qfq_end_date = qfq
            data["buy_close_raw"] = raw_buy
            data["end_close_raw"] = raw_end
            data["buy_close_qfq"] = qfq_buy
            data["end_close_qfq"] = qfq_end
            data["buy_trade_date"] = qfq_buy_date or raw_buy_date
            data["end_trade_date"] = qfq_end_date or raw_end_date
            data["raw_return"] = raw_end / raw_buy - 1 if raw_buy and raw_end else None
            data["qfq_return"] = qfq_end / qfq_buy - 1 if qfq_buy and qfq_end else None
        except Exception as exc:
            errors.append((idx, data.get("code"), str(exc)))
        rows.append(data)
        if (idx + 1) % 50 == 0:
            print(f"{os.path.basename(path)}: {idx + 1}/{len(df)}, errors={len(errors)}")

    repaired = pd.DataFrame(rows)
    root, ext = os.path.splitext(path)
    out = f"{root}_returns_fixed{ext}"
    if not os.path.isabs(out):
        out = get_project_path(out)
    repaired.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out}")
    print(
        f"行数={len(repaired)}, qfq缺失={repaired['qfq_return'].isna().sum()}, "
        f"均值={repaired['qfq_return'].mean():.1%}, "
        f"中位数={repaired['qfq_return'].median():.1%}, "
        f"胜率={(repaired['qfq_return'] > 0).mean():.1%}"
    )
    if errors:
        print("前10个错误:", errors[:10])
    return out


def main():
    args = parse_args()
    logged_in = False
    if args.source == "baostock":
        lg = bs.login()
        if lg.error_code != "0":
            raise SystemExit(f"baostock登录失败: {lg.error_msg}")
        logged_in = True
    try:
        for path in args.files:
            repair_file(path, args.buy_date, args.end_date, args.sleep, args.source)
    finally:
        if logged_in:
            bs.logout()


if __name__ == "__main__":
    main()
