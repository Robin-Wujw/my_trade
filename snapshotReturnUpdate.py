# -*- coding: utf-8 -*-
"""Update a saved historical selection snapshot with AkShare-cache returns."""
import argparse
import os

import pandas as pd

from trade_utils import get_project_path


CACHE_DIR = get_project_path(".cache/formula33_kline/akshare")


def parse_args():
    parser = argparse.ArgumentParser(description="更新历史截面持有至指定日期的前复权收益")
    parser.add_argument("--source", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def cache_path(code):
    return os.path.join(CACHE_DIR, f"{str(code).replace('.', '_')}.csv")


def get_return(code, buy_date, end_date):
    path = cache_path(code)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
        buy_rows = df[df["date"] >= pd.to_datetime(buy_date)]
        end_rows = df[df["date"] <= pd.to_datetime(end_date)]
        if buy_rows.empty or end_rows.empty:
            return None
        buy_row = buy_rows.iloc[0]
        end_row = end_rows.iloc[-1]
        if end_row["date"] < buy_row["date"] or not buy_row["close"]:
            return None
        return {
            "buy_trade_date": buy_row["date"].strftime("%Y-%m-%d"),
            "end_trade_date": end_row["date"].strftime("%Y-%m-%d"),
            "buy_close_qfq": float(buy_row["close"]),
            "end_close_qfq": float(end_row["close"]),
            "qfq_return": float(end_row["close"] / buy_row["close"] - 1),
            "holding_days": int((end_row["date"] - buy_row["date"]).days),
            "return_status": "ok",
        }
    except Exception:
        return None


def main():
    args = parse_args()
    df = pd.read_csv(args.source, dtype={"code": str})
    updates = []
    for code in df["code"].astype(str):
        result = get_return(code, args.buy_date, args.end_date)
        updates.append(result or {"return_status": "missing"})
    update_df = pd.DataFrame(updates)
    for col in update_df.columns:
        df[col] = update_df[col].values
    out = args.out or os.path.splitext(args.source)[0] + f"_end{args.end_date}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    valid = pd.to_numeric(df.get("qfq_return"), errors="coerce").dropna()
    print(f"收益更新完成: {out}")
    print(f"样本={len(df)} 有效收益={len(valid)} 缺失={len(df)-len(valid)}")
    if len(valid):
        print(
            f"均值={valid.mean():.2%} 中位数={valid.median():.2%} "
            f"胜率={(valid > 0).mean():.2%} 最大={valid.max():.2%} 最小={valid.min():.2%}"
        )


if __name__ == "__main__":
    main()
