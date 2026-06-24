# -*- coding: utf-8 -*-
"""
Scan selection scales on snapshot backtest CSV files.

The input should be a q1Backtest-style CSV with scores, buckets and qfq_return.
This script does not refetch market or financial data; it only tests different
selection scales on an already generated candidate pool.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from trade_utils import get_project_path


DEFAULT_BUCKET_SETS = {
    "strict_all": ["低估且高质量", "低估价值", "高质量趋势"],
    "core_only": ["低估且高质量"],
    "low_value_only": ["低估价值"],
    "quality_only": ["高质量趋势"],
    "core_low": ["低估且高质量", "低估价值"],
    "with_experiment": ["低估且高质量", "低估价值", "高质量趋势", "财报后主线候选", "价值线左侧确认"],
    "value_line_left": ["低估且高质量", "低估价值", "价值线左侧确认"],
    "earnings_mainline": ["财报后主线候选"],
    "theme_momentum": ["主题右侧动量"],
    "with_theme_momentum": ["低估且高质量", "低估价值", "高质量趋势", "财报后主线候选", "价值线左侧确认", "主题右侧动量"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="扫描截面选股尺度")
    parser.add_argument("files", nargs="+", help="q1Backtest 输出 CSV")
    parser.add_argument("--min-count", type=int, default=5, help="结果至少包含多少只股票")
    parser.add_argument("--top", type=int, default=20, help="每个最小样本约束展示前N个尺度")
    parser.add_argument("--out", default="", help="保存扫描明细 CSV；默认自动保存到 回测结果")
    return parser.parse_args()


def load_file(path):
    df = pd.read_csv(path)
    df["source_file"] = os.path.basename(path)
    if "qfq_return" not in df.columns:
        raise ValueError(f"{path} 缺少 qfq_return 列")
    df["qfq_return"] = pd.to_numeric(df["qfq_return"], errors="coerce")
    df["total_score"] = pd.to_numeric(df.get("total_score"), errors="coerce")
    for col in [
        "price_to_value",
        "mktcap",
        "quality_score",
        "trend_score",
        "liquidity_score",
        "theme_momentum_score",
        "ret20_at_buy",
        "ret60_at_buy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["qfq_return", "total_score", "selection_bucket"])
    return df


def summarize(selected, label, bucket_name, score_floor, top_n, ptv_cap, source_name):
    ret = selected["qfq_return"]
    return {
        "source": source_name,
        "scale": label,
        "bucket_set": bucket_name,
        "score_floor": score_floor,
        "top_n": top_n if top_n is not None else "all",
        "ptv_cap": ptv_cap if ptv_cap is not None else "none",
        "count": len(selected),
        "mean_return": ret.mean(),
        "median_return": ret.median(),
        "win_rate": (ret > 0).mean(),
        "min_return": ret.min(),
        "max_return": ret.max(),
        "names": "、".join(selected["name"].astype(str).head(8).tolist()),
        "codes": ",".join(selected["code"].astype(str).head(8).tolist()),
    }


def sort_candidates(cur, bucket_name):
    if bucket_name == "theme_momentum":
        preferred = [
            "total_score",
            "theme_momentum_score",
            "trend_score",
            "ret60_at_buy",
            "liquidity_score",
            "ret20_at_buy",
            "mktcap",
            "code",
        ]
    elif bucket_name == "with_theme_momentum":
        preferred = [
            "total_score",
            "theme_momentum_score",
            "quality_score",
            "trend_score",
            "ret60_at_buy",
            "liquidity_score",
            "price_to_value",
            "mktcap",
            "code",
        ]
    else:
        preferred = [
            "total_score",
            "quality_score",
            "trend_score",
            "liquidity_score",
            "price_to_value",
            "mktcap",
            "code",
        ]
    sort_cols = [col for col in preferred if col in cur.columns]
    ascending = [col in {"price_to_value", "code"} for col in sort_cols]
    return cur.sort_values(sort_cols, ascending=ascending, na_position="last")


def scan_frame(df, source_name):
    score_floors = list(range(50, 91, 5))
    top_ns = [3, 5, 8, 10, 15, 20, 30, 50, None]
    ptv_caps = [None, 0.60, 0.75, 0.85, 1.00, 1.10, 1.25, 1.45]
    rows = []

    for bucket_name, buckets in DEFAULT_BUCKET_SETS.items():
        base = df[df["selection_bucket"].isin(buckets)].copy()
        if base.empty:
            continue
        for score_floor, top_n, ptv_cap in itertools.product(score_floors, top_ns, ptv_caps):
            cur = base[base["total_score"] >= score_floor].copy()
            if ptv_cap is not None:
                if "price_to_value" not in cur.columns:
                    continue
                cur = cur[cur["price_to_value"].notna() & (cur["price_to_value"] <= ptv_cap)]
            if cur.empty:
                continue
            cur = sort_candidates(cur, bucket_name)
            if top_n is not None:
                cur = cur.head(top_n)
            if cur.empty:
                continue
            label = f"{bucket_name}|score>={score_floor}|top={top_n or 'all'}|ptv<={ptv_cap or 'none'}"
            rows.append(summarize(cur, label, bucket_name, score_floor, top_n, ptv_cap, source_name))
    return pd.DataFrame(rows)


def print_best(scan, min_count, top):
    if scan.empty:
        print("无可扫描结果")
        return
    for floor in sorted(set([min_count, 10, 20, 30])):
        subset = scan[scan["count"] >= floor].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["mean_return", "median_return", "count"], ascending=[False, False, False]).head(top)
        print(f"\n--- 平均收益最高尺度：至少 {floor} 只 ---")
        cols = [
            "source", "bucket_set", "score_floor", "top_n", "ptv_cap", "count",
            "mean_return", "median_return", "win_rate", "min_return", "max_return", "names",
        ]
        print(subset[cols].to_string(index=False, formatters={
            "mean_return": lambda v: f"{v:.1%}",
            "median_return": lambda v: f"{v:.1%}",
            "win_rate": lambda v: f"{v:.1%}",
            "min_return": lambda v: f"{v:.1%}",
            "max_return": lambda v: f"{v:.1%}",
        }))


def main():
    args = parse_args()
    scans = []
    frames = []
    for path in args.files:
        df = load_file(path)
        source_name = os.path.basename(path)
        frames.append(df.assign(scan_source=source_name))
        scans.append(scan_frame(df, source_name))

    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        scans.append(scan_frame(combined, "combined"))

    scan = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame()
    out_path = args.out
    if not out_path:
        out_path = get_project_path(os.path.join("回测结果", f"scale_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"))
    if not scan.empty:
        scan.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"扫描明细已保存: {out_path}")
    print_best(scan, args.min_count, args.top)


if __name__ == "__main__":
    main()
