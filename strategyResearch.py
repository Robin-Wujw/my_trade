# -*- coding: utf-8 -*-
"""
Offline strategy research on q1Backtest-style CSV files.

The goal is not to optimize the production score formula. It compares simple,
readable filters on historical snapshots, then applies the same filters to a
validation snapshot. This is useful for checking whether 300/688 growth stocks
prefer value-line pullbacks, right-side momentum, or a mixed setup.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from trade_utils import get_project_path


DEFAULT_TRAIN_FILES = [
    get_project_path("回测结果/q1_backtest_2025-05-06_2026-05-18_wide_merged_returns_fixed.csv"),
    get_project_path("回测结果/q1_backtest_2025-09-01_2026-05-19_162431_returns_fixed.csv"),
]
DEFAULT_VALIDATE_FILES = [
    get_project_path("回测结果/q1_backtest_2026-05-19_2026-05-19_151928_returns_fixed.csv"),
]

BUCKET_SETS = {
    "all": None,
    "value_pullback": ["低估价值", "低估且高质量", "价值线左侧确认"],
    "right_momentum": ["高质量趋势", "主题右侧动量", "财报后主线候选"],
    "theme_only": ["主题右侧动量"],
    "value_line_only": ["价值线左侧确认"],
    "core_only": ["低估且高质量"],
}

PREFERRED_SORTS = {
    "value": ["price_to_value", "ret20_at_buy", "ret60_at_buy", "liquidity_score", "code"],
    "momentum": ["ret60_at_buy", "ret20_at_buy", "liquidity_score", "price_to_value", "code"],
    "hybrid": ["total_score", "ret60_at_buy", "price_to_value", "liquidity_score", "code"],
    "volume": ["volume_ratio_20_120", "relative_ret60", "ret60_at_buy", "code"],
}

KEY_COLUMNS = [
    "bucket_set", "sort", "ptv_min", "ptv_max", "ret20_min", "ret20_max",
    "ret60_min", "ret60_max", "vol_min", "vol_max", "score_min", "liq_min",
]

DISPLAY_COLUMNS = [
    "bucket_set", "sort", "ptv_min", "ptv_max", "ret20_min", "ret20_max",
    "ret60_min", "ret60_max", "vol_min", "score_min", "liq_min",
    "train_ret_count", "train_mean_return", "train_median_return", "train_win_rate",
    "validate_ret_count", "validate_mean_return", "validate_median_return", "validate_win_rate",
    "overfit_gap", "joint_score", "train_names", "validate_names",
]


def parse_args():
    parser = argparse.ArgumentParser(description="离线扫描简单选股方案")
    parser.add_argument("--train-files", nargs="*", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--validate-files", nargs="*", default=DEFAULT_VALIDATE_FILES)
    parser.add_argument("--prefixes", default="sz.300,sh.688", help="代码前缀，多个用逗号分隔")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--top", type=int, default=20, help="每种方案取前N只")
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--out", default="", help="保存扫描明细 CSV")
    parser.add_argument("--report-out", default="", help="保存联合排名 CSV")
    return parser.parse_args()


def normalize_frame(path, label):
    df = pd.read_csv(path)
    df["source_file"] = os.path.basename(path)
    df["sample"] = label
    for col in [
        "qfq_return", "total_score", "valuation_score", "quality_score", "trend_score",
        "liquidity_score", "price_to_value", "ret20_at_buy", "ret60_at_buy",
        "relative_ret60", "volume_ratio_20_120", "mktcap",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "code" not in df.columns:
        df["code"] = ""
    if "selection_bucket" not in df.columns:
        df["selection_bucket"] = ""
    if "theme" not in df.columns:
        df["theme"] = ""
    return df


def load_files(paths, label):
    frames = [normalize_frame(path, label) for path in paths if path and os.path.exists(path)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def apply_prefixes(df, prefixes):
    prefix_tuple = tuple(item.strip().lower() for item in prefixes.split(",") if item.strip())
    if not prefix_tuple:
        return df
    return df[df["code"].astype(str).str.lower().str.startswith(prefix_tuple)].copy()


def apply_filter(df, spec):
    cur = df.copy()
    buckets = BUCKET_SETS[spec["bucket_set"]]
    if buckets is not None:
        cur = cur[cur["selection_bucket"].isin(buckets)]
    for col, low, high in [
        ("price_to_value", spec["ptv_min"], spec["ptv_max"]),
        ("ret20_at_buy", spec["ret20_min"], spec["ret20_max"]),
        ("ret60_at_buy", spec["ret60_min"], spec["ret60_max"]),
        ("volume_ratio_20_120", spec["vol_min"], spec["vol_max"]),
        ("total_score", spec["score_min"], None),
        ("liquidity_score", spec["liq_min"], None),
    ]:
        if col not in cur.columns:
            continue
        if low is not None:
            cur = cur[cur[col].notna() & (cur[col] >= low)]
        if high is not None:
            cur = cur[cur[col].notna() & (cur[col] <= high)]
    return cur


def sort_frame(df, sort_name):
    sort_cols = [col for col in PREFERRED_SORTS[sort_name] if col in df.columns]
    if not sort_cols:
        return df
    ascending = [col in {"price_to_value", "code"} for col in sort_cols]
    return df.sort_values(sort_cols, ascending=ascending, na_position="last")


def summarize(df, spec, label, top):
    cur = apply_filter(df, spec)
    cur = sort_frame(cur, spec["sort"]).head(top)
    ret = pd.to_numeric(cur.get("qfq_return"), errors="coerce").dropna()
    row = dict(spec)
    row.update({
        "sample": label,
        "count": len(cur),
        "ret_count": len(ret),
        "mean_return": ret.mean() if len(ret) else np.nan,
        "median_return": ret.median() if len(ret) else np.nan,
        "win_rate": (ret > 0).mean() if len(ret) else np.nan,
        "min_return": ret.min() if len(ret) else np.nan,
        "max_return": ret.max() if len(ret) else np.nan,
        "names": "、".join(cur.get("name", pd.Series(dtype=str)).astype(str).head(8).tolist()),
        "codes": ",".join(cur.get("code", pd.Series(dtype=str)).astype(str).head(8).tolist()),
    })
    return row


def build_joint_report(scan, min_count):
    train = scan[scan["sample"] == "train"].copy()
    validate = scan[scan["sample"] == "validate"].copy()
    if validate.empty:
        train = train.rename(columns={
            "ret_count": "train_ret_count",
            "mean_return": "train_mean_return",
            "median_return": "train_median_return",
            "win_rate": "train_win_rate",
            "names": "train_names",
        })
        train["validate_ret_count"] = np.nan
        train["validate_mean_return"] = np.nan
        train["validate_median_return"] = np.nan
        train["validate_win_rate"] = np.nan
        train["validate_names"] = ""
        train["joint_score"] = train["train_mean_return"]
        return train

    train = train.rename(columns={
        "ret_count": "train_ret_count",
        "mean_return": "train_mean_return",
        "median_return": "train_median_return",
        "win_rate": "train_win_rate",
        "names": "train_names",
    })
    validate = validate.rename(columns={
        "ret_count": "validate_ret_count",
        "mean_return": "validate_mean_return",
        "median_return": "validate_median_return",
        "win_rate": "validate_win_rate",
        "names": "validate_names",
    })
    merged = train.merge(validate[KEY_COLUMNS + [
        "validate_ret_count", "validate_mean_return", "validate_median_return",
        "validate_win_rate", "validate_names",
    ]], on=KEY_COLUMNS, how="left")
    merged = merged[
        (merged["train_ret_count"] >= min_count)
        & (merged["validate_ret_count"] >= min_count)
    ].copy()
    if merged.empty:
        return merged
    merged["overfit_gap"] = (
        merged["train_mean_return"].fillna(0) - merged["validate_mean_return"].fillna(0)
    ).clip(lower=0)
    merged["joint_score"] = (
        merged["validate_mean_return"].fillna(-9) * 0.35
        + merged["validate_median_return"].fillna(-9) * 0.25
        + merged["validate_win_rate"].fillna(0) * 0.15
        + merged["train_median_return"].fillna(-9) * 0.15
        + merged["train_win_rate"].fillna(0) * 0.10
        - merged["overfit_gap"] * 0.15
    )
    return merged.sort_values(
        ["joint_score", "validate_mean_return", "validate_win_rate", "train_mean_return"],
        ascending=False,
    )


def spec_grid():
    bucket_sets = ["all", "value_pullback", "right_momentum", "theme_only", "value_line_only"]
    sorts = ["value", "momentum", "hybrid"]
    ptv_ranges = [(None, None), (None, 1.0), (0.75, 1.08), (1.0, 1.8)]
    ret20_ranges = [(None, None), (-0.20, 0.20), (0.0, 0.35)]
    ret60_ranges = [(None, None), (0.0, 0.80), (0.20, 1.50)]
    vol_ranges = [(None, None), (0.8, None)]
    score_mins = [None, 70]
    liq_mins = [None, 60]
    for bucket_set, sort, ptv, ret20, ret60, vol, score_min, liq_min in itertools.product(
        bucket_sets, sorts, ptv_ranges, ret20_ranges, ret60_ranges, vol_ranges, score_mins, liq_mins
    ):
        yield {
            "bucket_set": bucket_set,
            "sort": sort,
            "ptv_min": ptv[0],
            "ptv_max": ptv[1],
            "ret20_min": ret20[0],
            "ret20_max": ret20[1],
            "ret60_min": ret60[0],
            "ret60_max": ret60[1],
            "vol_min": vol[0],
            "vol_max": vol[1],
            "score_min": score_min,
            "liq_min": liq_min,
        }


def main():
    args = parse_args()
    train = apply_prefixes(load_files(args.train_files, "train"), args.prefixes)
    validate = apply_prefixes(load_files(args.validate_files, "validate"), args.prefixes)
    if train.empty:
        raise SystemExit("训练样本为空")

    rows = []
    for spec in spec_grid():
        train_row = summarize(train, spec, "train", args.top)
        if train_row["ret_count"] < args.min_count:
            continue
        rows.append(train_row)
        if not validate.empty:
            rows.append(summarize(validate, spec, "validate", args.top))

    scan = pd.DataFrame(rows)
    if scan.empty:
        raise SystemExit("无有效扫描结果")
    out = args.out or get_project_path(f"回测结果/strategy_research_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    scan.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"扫描明细已保存: {out}")
    report = build_joint_report(scan, args.min_count)
    if report.empty:
        raise SystemExit("无训练/验证同时满足样本数的方案")
    report_out = args.report_out or get_project_path(f"回测结果/strategy_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    report.to_csv(report_out, index=False, encoding="utf-8-sig")
    print(f"联合排名已保存: {report_out}")
    print("\n--- 训练/验证联合表现最好方案 ---")
    print(report[DISPLAY_COLUMNS].head(args.show).to_string(index=False, formatters={
        "train_mean_return": lambda v: f"{v:.1%}",
        "train_median_return": lambda v: f"{v:.1%}",
        "train_win_rate": lambda v: f"{v:.1%}",
        "validate_mean_return": lambda v: f"{v:.1%}",
        "validate_median_return": lambda v: f"{v:.1%}",
        "validate_win_rate": lambda v: f"{v:.1%}",
        "joint_score": lambda v: f"{v:.3f}",
    }))
    if not validate.empty:
        validate_best = (
            scan[(scan["sample"] == "validate") & (scan["ret_count"] >= args.min_count)]
            .sort_values(["mean_return", "median_return", "win_rate"], ascending=False)
            .head(args.show)
        )
        validate_cols = [
            "bucket_set", "sort", "ptv_min", "ptv_max", "ret20_min", "ret20_max",
            "ret60_min", "ret60_max", "vol_min", "score_min", "liq_min",
            "ret_count", "mean_return", "median_return", "win_rate", "names",
        ]
        print("\n--- 验证期表现最好方案 ---")
        print(validate_best[validate_cols].to_string(index=False, formatters={
            "mean_return": lambda v: f"{v:.1%}",
            "median_return": lambda v: f"{v:.1%}",
            "win_rate": lambda v: f"{v:.1%}",
        }))


if __name__ == "__main__":
    main()
