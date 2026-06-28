# -*- coding: utf-8 -*-
"""
Build a smaller final portfolio from a broad q1Backtest candidate CSV.

The broad candidate pool is useful for research, but it is too large for a
usable trading list. This module applies second-pass filters, bucket caps and
ranking using only buy-date information.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from trade_utils import get_project_path


CORE_BUCKET = "低估且高质量"
LOW_VALUE_BUCKET = "低估价值"
HIGH_QUALITY_BUCKET = "高质量趋势"
EARNINGS_MAINLINE_BUCKET = "财报后主线候选"
VALUE_LEFT_BUCKET = "价值线左侧确认"
THEME_MOMENTUM_BUCKET = "主题右侧动量"


PROFILE_CONFIGS = {
    "focused": {
        "min_scores": {
            VALUE_LEFT_BUCKET: 75,
            CORE_BUCKET: 72,
            EARNINGS_MAINLINE_BUCKET: 70,
            THEME_MOMENTUM_BUCKET: 70,
            HIGH_QUALITY_BUCKET: 72,
            LOW_VALUE_BUCKET: 65,
        },
        "ptv_caps": {
            VALUE_LEFT_BUCKET: 1.00,
            CORE_BUCKET: 1.00,
            LOW_VALUE_BUCKET: 1.00,
            EARNINGS_MAINLINE_BUCKET: 1.45,
            HIGH_QUALITY_BUCKET: 1.45,
        },
        "bucket_weights": {
            VALUE_LEFT_BUCKET: 0.45,
            THEME_MOMENTUM_BUCKET: 0.45,
            LOW_VALUE_BUCKET: 0.30,
            CORE_BUCKET: 0.25,
            EARNINGS_MAINLINE_BUCKET: 0.25,
            HIGH_QUALITY_BUCKET: 0.25,
        },
        "theme_weight": 0.35,
    },
    "balanced": {
        "min_scores": {
            VALUE_LEFT_BUCKET: 78,
            CORE_BUCKET: 75,
            EARNINGS_MAINLINE_BUCKET: 72,
            THEME_MOMENTUM_BUCKET: 72,
            HIGH_QUALITY_BUCKET: 75,
            LOW_VALUE_BUCKET: 68,
        },
        "ptv_caps": {
            VALUE_LEFT_BUCKET: 1.00,
            CORE_BUCKET: 1.00,
            LOW_VALUE_BUCKET: 0.95,
            EARNINGS_MAINLINE_BUCKET: 1.35,
            HIGH_QUALITY_BUCKET: 1.35,
        },
        "bucket_weights": {
            VALUE_LEFT_BUCKET: 0.35,
            THEME_MOMENTUM_BUCKET: 0.35,
            LOW_VALUE_BUCKET: 0.25,
            CORE_BUCKET: 0.25,
            EARNINGS_MAINLINE_BUCKET: 0.20,
            HIGH_QUALITY_BUCKET: 0.20,
        },
        "theme_weight": 0.25,
    },
    "theme": {
        "min_scores": {
            THEME_MOMENTUM_BUCKET: 65,
            VALUE_LEFT_BUCKET: 75,
            EARNINGS_MAINLINE_BUCKET: 68,
            CORE_BUCKET: 72,
            HIGH_QUALITY_BUCKET: 72,
            LOW_VALUE_BUCKET: 68,
        },
        "ptv_caps": {
            VALUE_LEFT_BUCKET: 1.00,
            CORE_BUCKET: 1.00,
            LOW_VALUE_BUCKET: 0.95,
            EARNINGS_MAINLINE_BUCKET: 1.45,
            HIGH_QUALITY_BUCKET: 1.45,
        },
        "bucket_weights": {
            THEME_MOMENTUM_BUCKET: 1.00,
            VALUE_LEFT_BUCKET: 0.20,
            EARNINGS_MAINLINE_BUCKET: 0.15,
            CORE_BUCKET: 0.15,
            HIGH_QUALITY_BUCKET: 0.15,
            LOW_VALUE_BUCKET: 0.15,
        },
        "theme_weight": 0.90,
    },
}

# Walk-forward adjustment: keep the original candidate model intact and only
# change the final portfolio convergence rule validated on 2025 Q1 -> H1.
PROFILE_CONFIGS["walk_forward"] = {}
PROFILE_CONFIGS["right_side"] = {}


NUMERIC_COLUMNS = [
    "total_score",
    "quality_score",
    "trend_score",
    "liquidity_score",
    "theme_momentum_score",
    "price_to_value",
    "mktcap",
    "ret20_at_buy",
    "ret60_at_buy",
    "relative_ret20",
    "relative_ret60",
    "volume_ratio_20_120",
    "qfq_return",
    "downtrend_drawdown",
    "recovery_level_50",
    "recovery_level_625",
    "recovery_pct",
    "bars_since_recovery_50_cross",
    "bars_since_recovery_625_cross",
]


def parse_args():
    parser = argparse.ArgumentParser(description="从宽口径候选池收敛出最终组合")
    parser.add_argument("files", nargs="+", help="q1Backtest 候选 CSV")
    parser.add_argument("--size", type=int, default=30, help="最终组合数量")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="focused", help="收敛风格")
    parser.add_argument("--out-dir", default="", help="输出目录，默认写回原文件所在目录")
    return parser.parse_args()


def normalize_frame(df):
    df = df.copy()
    if "valuation_ref" in df.columns:
        parsed = df["valuation_ref"].apply(parse_theme_metrics)
        for col in ["relative_ret60", "volume_ratio_20_120"]:
            if col not in df.columns:
                df[col] = parsed.apply(lambda item: item.get(col))
            else:
                df[col] = df[col].where(df[col].notna(), parsed.apply(lambda item: item.get(col)))
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "theme" not in df.columns:
        df["theme"] = ""
    return df


def parse_theme_metrics(text):
    result = {}
    if not isinstance(text, str):
        return result
    rel_match = re.search(r"相对60日=([+-]?\d+(?:\.\d+)?)%", text)
    if rel_match:
        result["relative_ret60"] = float(rel_match.group(1)) / 100
    vol_match = re.search(r"量能20/120=([+-]?\d+(?:\.\d+)?)", text)
    if vol_match:
        result["volume_ratio_20_120"] = float(vol_match.group(1))
    return result


def _score_direct(value, low, high):
    if value is None or pd.isna(value):
        return 0.0
    if high == low:
        return 0.0
    return max(0.0, min(100.0, (float(value) - low) / (high - low) * 100.0))


def _score_inverse(value, best, worst):
    if value is None or pd.isna(value):
        return 0.0
    if worst == best:
        return 0.0
    return max(0.0, min(100.0, (worst - float(value)) / (worst - best) * 100.0))


def _score_hump(value, low, ideal_low, ideal_high, high):
    if value is None or pd.isna(value):
        return 0.0
    value = float(value)
    if value <= low or value >= high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return (value - low) / (ideal_low - low) * 100.0
    return (high - value) / (high - ideal_high) * 100.0


def bucket_cap(size, bucket, config):
    weight = config["bucket_weights"].get(bucket, 0.20)
    return max(1, int(np.ceil(size * weight)))


def theme_cap(size, config):
    return max(2, int(np.ceil(size * config.get("theme_weight", 0.30))))


def is_eligible(row, config):
    bucket = row.get("selection_bucket")
    score = row.get("total_score", 0)
    if pd.isna(score):
        return False
    if score < config["min_scores"].get(bucket, 70):
        return False

    ptv_cap = config["ptv_caps"].get(bucket)
    if ptv_cap is None:
        return True

    ptv = row.get("price_to_value")
    if ptv is None or pd.isna(ptv):
        return True
    return ptv <= ptv_cap


def calc_portfolio_score(row):
    bucket = row.get("selection_bucket")
    base = row.get("total_score", 0)
    if pd.isna(base):
        base = 0
    if bucket == THEME_MOMENTUM_BUCKET:
        ret20 = row.get("ret20_at_buy")
        ret60 = row.get("ret60_at_buy")
        setup_score = (
            float(base) * 0.24
            + _score_hump(ret60, 0.08, 0.22, 0.75, 1.35) * 0.30
            + _score_hump(ret20, 0.02, 0.08, 0.25, 0.55) * 0.18
            + _score_direct(row.get("liquidity_score"), 55, 90) * 0.10
            + _score_direct(row.get("trend_score"), 78, 98) * 0.08
            + _score_direct(row.get("relative_ret60"), 0.04, 0.35) * 0.06
            + _score_hump(row.get("volume_ratio_20_120"), 0.75, 1.05, 2.20, 4.00) * 0.04
        )
        return round(80.0 + setup_score * 0.48, 3)

    bucket_bonus = {
        VALUE_LEFT_BUCKET: 14,
        CORE_BUCKET: 10,
        THEME_MOMENTUM_BUCKET: 9,
        EARNINGS_MAINLINE_BUCKET: 7,
        HIGH_QUALITY_BUCKET: 4,
        LOW_VALUE_BUCKET: 0,
    }.get(bucket, 0)

    ptv = row.get("price_to_value")
    ptv_bonus = _score_inverse(ptv, 0.45, 1.10) * 0.08 if ptv is not None and not pd.isna(ptv) else 0
    trend_bonus = _score_direct(row.get("ret60_at_buy"), -0.05, 0.45) * 0.06
    liquidity_bonus = _score_direct(row.get("liquidity_score"), 50, 90) * 0.03
    theme_bonus = 0
    if bucket == THEME_MOMENTUM_BUCKET:
        theme_bonus = _score_direct(row.get("theme_momentum_score"), 60, 95) * 0.10

    return round(float(base) + bucket_bonus + ptv_bonus + trend_bonus + liquidity_bonus + theme_bonus, 3)


def select_portfolio(df, size=30, profile="focused"):
    if size <= 0:
        out = normalize_frame(df)
        out["final_selected"] = True
        out["final_rank"] = range(1, len(out) + 1)
        out["portfolio_profile"] = "all"
        return out

    work = normalize_frame(df)
    if profile == "right_side":
        ptv = pd.to_numeric(work.get("price_to_value"), errors="coerce")
        liquidity = pd.to_numeric(work.get("liquidity_score"), errors="coerce")
        quality = pd.to_numeric(work.get("quality_score"), errors="coerce")
        growth = pd.to_numeric(work.get("earnings_yoy"), errors="coerce")
        mktcap = pd.to_numeric(work.get("mktcap"), errors="coerce")
        recovery = pd.to_numeric(work.get("recovery_pct"), errors="coerce")
        mainline_growth = (
            work.get("theme", "").eq("AI算力/CPO")
            & (growth >= 0.25)
            & (quality >= 80)
        )
        selected = work[
            ptv.notna()
            & ((ptv <= 1.50) | mainline_growth)
            & (quality >= 70)
            & (liquidity >= 55)
            & (mktcap >= 100)
            & (recovery >= 50)
        ].copy()
        selected["portfolio_score"] = (
            quality.loc[selected.index].fillna(0) * 0.40
            + liquidity.loc[selected.index].fillna(0) * 0.20
            + growth.loc[selected.index].fillna(0).clip(lower=0, upper=2.0) * 20
            + (1 - ptv.loc[selected.index].clip(lower=0, upper=6) / 6) * 20
        )
        selected["walk_forward_layer"] = np.where(
            recovery.loc[selected.index] >= 62.5,
            "下跌波段62.5%右侧确认",
            "下跌波段50%右侧启动",
        )
        selected = selected.sort_values(
            ["portfolio_score", "quality_score", "liquidity_score", "code"],
            ascending=[False, False, False, True],
        )
        core = selected[mainline_growth.loc[selected.index]].copy()
        rest = selected[~mainline_growth.loc[selected.index]].copy().head(max(0, size - len(core)))
        selected = pd.concat([core, rest], ignore_index=True, sort=False).head(size)
        selected["final_selected"] = True
        selected["final_rank"] = range(1, len(selected) + 1)
        selected["portfolio_profile"] = profile
        return selected
    if profile == "walk_forward":
        ptv = pd.to_numeric(work.get("price_to_value"), errors="coerce")
        liquidity = pd.to_numeric(work.get("liquidity_score"), errors="coerce")
        quality = pd.to_numeric(work.get("quality_score"), errors="coerce")
        growth = pd.to_numeric(work.get("earnings_yoy"), errors="coerce")
        mktcap = pd.to_numeric(work.get("mktcap"), errors="coerce")

        # Keep up to five point-in-time financial mainline leaders even when
        # their pre-buy momentum is weak or price is already above value line.
        core = work[
            work.get("theme", "").eq("AI算力/CPO")
            & (growth >= 0.20)
            & (quality >= 80)
            & (liquidity >= 60)
            & (mktcap >= 100)
            & ptv.notna()
            & (ptv <= 1.50)
        ].copy()
        core["portfolio_score"] = (
            growth.loc[core.index].clip(upper=2.0) * 45
            + quality.loc[core.index] * 0.25
            + liquidity.loc[core.index] * 0.15
            + (1.50 - ptv.loc[core.index]).clip(lower=0) / 1.50 * 15
        )
        core = core.sort_values(["portfolio_score", "total_score"], ascending=False).head(min(5, size))
        core["walk_forward_layer"] = "财报主线核心保留"

        rest = work[
            ~work.get("code").isin(core.get("code", pd.Series(dtype=str)))
            & ptv.notna()
            & (ptv <= 1.0)
            & (liquidity >= 60)
        ].copy()
        sort_cols = [
            col for col in ["ret60_at_buy", "ret20_at_buy", "liquidity_score", "price_to_value", "code"]
            if col in rest.columns
        ]
        ascending = [col in {"price_to_value", "code"} for col in sort_cols]
        rest = rest.sort_values(sort_cols, ascending=ascending, na_position="last").head(size - len(core))
        rest["walk_forward_layer"] = "价值动量补齐"
        rest["portfolio_score"] = range(len(rest), 0, -1)

        selected = pd.concat([core, rest], ignore_index=True, sort=False)
        selected["final_selected"] = True
        selected["final_rank"] = range(1, len(selected) + 1)
        selected["portfolio_profile"] = profile
        return selected

    config = PROFILE_CONFIGS[profile]
    work = work[work.apply(lambda r: is_eligible(r, config), axis=1)].copy()
    if work.empty:
        return work

    work["portfolio_score"] = work.apply(calc_portfolio_score, axis=1)
    sort_cols = [
        col for col in [
            "portfolio_score",
            "total_score",
            "theme_momentum_score",
            "ret60_at_buy",
            "ret20_at_buy",
            "quality_score",
            "trend_score",
            "liquidity_score",
            "price_to_value",
            "mktcap",
            "code",
        ] if col in work.columns
    ]
    ascending = [col in {"price_to_value", "code"} for col in sort_cols]
    work = work.sort_values(sort_cols, ascending=ascending, na_position="last")

    selected = []
    bucket_counts = {}
    theme_counts = {}
    per_theme_cap = theme_cap(size, config)

    def can_take(row, enforce_caps=True):
        if not enforce_caps:
            return True
        bucket = row.get("selection_bucket")
        theme = row.get("theme") or ""
        if bucket_counts.get(bucket, 0) >= bucket_cap(size, bucket, config):
            return False
        if theme and theme_counts.get(theme, 0) >= per_theme_cap:
            return False
        return True

    for enforce_caps in [True, False]:
        for _, row in work.iterrows():
            if len(selected) >= size:
                break
            code = row.get("code")
            if any(existing.get("code") == code for existing in selected):
                continue
            if not can_take(row, enforce_caps=enforce_caps):
                continue
            item = row.to_dict()
            selected.append(item)
            bucket = item.get("selection_bucket")
            theme = item.get("theme") or ""
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if theme:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        if len(selected) >= size:
            break

    out = pd.DataFrame(selected)
    if out.empty:
        return out
    out["final_selected"] = True
    out["final_rank"] = range(1, len(out) + 1)
    out["portfolio_profile"] = profile
    return out


def summarize(df):
    if df.empty or "qfq_return" not in df.columns:
        return "无收益列"
    ret = pd.to_numeric(df["qfq_return"], errors="coerce").dropna()
    if ret.empty:
        return "收益全缺失"
    return (
        f"数量={len(df)}, 均值={ret.mean():.1%}, 中位数={ret.median():.1%}, "
        f"胜率={(ret > 0).mean():.1%}, 最大={ret.max():.1%}, 最小={ret.min():.1%}"
    )


def output_path(path, size, profile, out_dir):
    root, ext = os.path.splitext(os.path.basename(path))
    name = f"{root}_portfolio_{profile}_{size}{ext or '.csv'}"
    directory = out_dir or os.path.dirname(path)
    if not directory:
        directory = get_project_path("回测结果")
    return os.path.join(directory, name)


def main():
    args = parse_args()
    for path in args.files:
        df = pd.read_csv(path)
        selected = select_portfolio(df, size=args.size, profile=args.profile)
        out = output_path(path, args.size, args.profile, args.out_dir)
        selected.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"已保存: {out}")
        print(summarize(selected))
        if "selection_bucket" in selected.columns:
            print(selected["selection_bucket"].value_counts().to_string())
        cols = [c for c in ["final_rank", "code", "name", "selection_bucket", "total_score", "portfolio_score", "qfq_return"] if c in selected.columns]
        if cols:
            print(selected[cols].head(min(20, len(selected))).to_string(index=False, formatters={
                "qfq_return": lambda v: f"{v:.1%}" if pd.notna(v) else "",
            }))


if __name__ == "__main__":
    main()
