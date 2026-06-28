# -*- coding: utf-8 -*-
"""Point-in-time backtest for recovery through 50% of a prior downtrend."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from formula33Stats import load_kline_with_cache
from wave_utils import infer_downtrend_recovery


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--ddl", required=True)
    parser.add_argument("--report-period", required=True)
    parser.add_argument("--signal-end", required=True)
    parser.add_argument("--end", default="2026-06-26")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def number(series):
    return pd.to_numeric(series, errors="coerce")


def enrich_fundamentals(source, report_period):
    source = source.copy()
    suffix = report_period.replace("-", "")
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "q1_value")
    fields = ["price_to_value", "quality_score", "mktcap", "yoy"]
    for index, row in source.iterrows():
        symbol = str(row["code"]).split(".")[-1]
        path = os.path.join(cache_dir, f"{symbol}_{suffix}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            for field in fields:
                cached_field = "earnings_yoy" if field == "yoy" else field
                if value.get(field) is not None:
                    source.at[index, cached_field] = value[field]
        except (OSError, ValueError, TypeError):
            continue
    return source


def evaluate(row, ddl, signal_end, end):
    code = row["code"]
    history_start = (pd.Timestamp(ddl) - pd.DateOffset(years=1, days=10)).strftime("%Y-%m-%d")
    df = load_kline_with_cache("akshare", code, history_start, end, retries=5, retry_delay=2.0)
    if df.empty:
        return None
    recovery = infer_downtrend_recovery(df[df["date"] <= ddl])
    if not recovery:
        return None
    post = df[(df["date"] > ddl) & (df["date"] <= signal_end)].copy()
    hit = post[post["close"] >= recovery["recovery_level_50"]].head(1)
    if hit.empty:
        return {**recovery, "code": code, "triggered": False}
    trigger = hit.iloc[0]
    end_row = df[df["date"] <= end].iloc[-1]
    trigger_close = float(trigger["close"])
    result = {
        **recovery,
        "code": code,
        "triggered": True,
        "trigger_date": trigger["date"],
        "trigger_close": trigger_close,
        "end_trade_date": end_row["date"],
        "end_close": float(end_row["close"]),
        "right_side_return": float(end_row["close"]) / trigger_close - 1,
    }
    confirm = post[(post["date"] >= trigger["date"]) & (post["close"] >= recovery["recovery_level_625"])].head(1)
    result["confirm_625_date"] = None if confirm.empty else confirm.iloc[0]["date"]
    return result


def main():
    args = parse_args()
    source = pd.read_csv(args.snapshot, dtype={"code": str})
    source = enrich_fundamentals(source, args.report_period)
    quality = number(source["quality_score"])
    growth = number(source["earnings_yoy"])
    ptv = number(source["price_to_value"])
    mainline_growth = (
        source["theme"].eq("AI算力/CPO")
        & (growth >= 0.25)
        & (quality >= 80)
    )
    mask = (
        (quality >= 70)
        & (number(source["liquidity_score"]) >= 55)
        & (number(source["mktcap"]) >= 100)
        & ((ptv <= 1.50) | mainline_growth)
    )
    pool = source[mask].drop_duplicates("code").copy()
    pool["is_growth_mainline"] = (
        pool["theme"].eq("AI算力/CPO")
        & (number(pool["earnings_yoy"]) >= 0.25)
        & (number(pool["quality_score"]) >= 80)
    )
    earnings = number(pool["earnings_yoy"]) if "earnings_yoy" in pool else pd.Series(0, index=pool.index)
    pool_ptv = number(pool["price_to_value"])
    pool["fundamental_rank_score"] = (
        number(pool["quality_score"]).fillna(0) * 0.40
        + number(pool["liquidity_score"]).fillna(0) * 0.20
        + earnings.fillna(0).clip(0, 2) * 20
        + (1 - pool_ptv.clip(lower=0, upper=6) / 6) * 20
    )
    pool = pool.sort_values(
        ["fundamental_rank_score", "quality_score", "liquidity_score", "code"],
        ascending=[False, False, False, True],
    )
    core = pool[pool["is_growth_mainline"]].copy()
    rest = pool[~pool["is_growth_mainline"]].copy().head(max(0, args.size - len(core)))
    pool = pd.concat([core, rest], ignore_index=True, sort=False).head(args.size)
    pool["candidate_rank"] = range(1, len(pool) + 1)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {
            executor.submit(evaluate, row, args.ddl, args.signal_end, args.end): row["code"]
            for _, row in pool.iterrows()
        }
        for index, future in enumerate(as_completed(jobs), 1):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"{jobs[future]} failed: {exc}")
            if index % 10 == 0 or index == len(jobs):
                print(f"progress {index}/{len(jobs)}")

    signals = pd.DataFrame(results)
    merged = pool.merge(signals, on="code", how="left")
    triggered = merged[merged["triggered"].fillna(False)].copy()
    selected = triggered.sort_values(["candidate_rank", "code"])
    selected["final_rank"] = range(1, len(selected) + 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"pool={len(pool)} triggered={len(triggered)} selected={len(selected)}")
    if not selected.empty:
        print(
            f"mean={selected.right_side_return.mean():.2%} "
            f"median={selected.right_side_return.median():.2%} "
            f"win={(selected.right_side_return > 0).mean():.2%}"
        )
        cols = ["code", "name", "trigger_date", "recovery_level_50", "confirm_625_date", "right_side_return"]
        print(selected[cols].to_string(index=False))


if __name__ == "__main__":
    main()
