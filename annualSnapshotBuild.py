# -*- coding: utf-8 -*-
"""Enrich a historical universe with AkShare financials as of an annual report."""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import q1Backtest
from dailyFundamentalSelect import KLINE_CACHE_DIR, classify_method


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--report-period", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def close_asof(code, buy_date):
    path = os.path.join(KLINE_CACHE_DIR, f"{str(code).replace('.', '_')}.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    rows = frame[frame["date"] <= pd.Timestamp(buy_date)].dropna(subset=["close"]).sort_values("date")
    return None if rows.empty else float(rows.iloc[-1]["close"])


def fetch(row, report_period, buy_date):
    code = str(row["code"])
    close = close_asof(code, buy_date)
    if close is None:
        return code, None
    symbol = code.split(".")[-1]
    last_error = None
    for attempt in range(4):
        try:
            value = q1Backtest.get_value_line_asof_cached(symbol, close, report_period)
            if value:
                return code, value
        except Exception as exc:
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    return code, {"error": str(last_error or "financial data missing")}


def main():
    args = parse_args()
    source = pd.read_csv(args.source, dtype={"code": str}, low_memory=False).drop_duplicates("code").copy()
    updates = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {
            executor.submit(fetch, row, args.report_period, args.buy_date): row["code"]
            for _, row in source.iterrows()
        }
        for index, future in enumerate(as_completed(jobs), 1):
            code, value = future.result()
            updates[code] = value
            if index % 25 == 0 or index == len(jobs):
                valid = sum(bool(v and not v.get("error")) for v in updates.values())
                print(f"progress {index}/{len(jobs)} valid={valid}")

    for index, row in source.iterrows():
        value = updates.get(str(row["code"]))
        if not value or value.get("error"):
            continue
        method = classify_method(row.get("industry", ""))
        source.at[index, "method"] = method
        source.at[index, "quality_score"] = value.get("quality_score")
        source.at[index, "earnings_yoy"] = value.get("yoy")
        source.at[index, "eps_excl"] = value.get("eps_excl")
        source.at[index, "mktcap"] = value.get("mktcap")
        if method == "VALUE":
            source.at[index, "value_line"] = value.get("value_line")
            source.at[index, "price_to_value"] = value.get("price_to_value")
        else:
            source.at[index, "value_line"] = None
            source.at[index, "price_to_value"] = None
    source["financial_report_period"] = args.report_period
    source["financial_status"] = source["code"].map(
        lambda code: "ok" if updates.get(str(code)) and not updates[str(code)].get("error") else "missing"
    )
    source.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"output={args.output} rows={len(source)} ok={(source.financial_status == 'ok').sum()}")


if __name__ == "__main__":
    main()
