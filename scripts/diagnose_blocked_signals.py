"""Diagnose blocked right-side entry signals: what happened after they were rejected?

Loads entry_blocks from the hybrid backtest summary, finds signals rejected for
risk_pct_too_tight / close_confirmed_risk_pct_too_tight, and measures forward
returns (5/10/20/40/60 trading days) using raw prices scaled by the
raw_to_qfq_factor so corporate actions do not distort returns.

Usage:
    python -m scripts.diagnose_blocked_signals
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.storage.database import Database
from stock_research.storage.tushare_repository import TushareRepository


DEFAULT_SUMMARY = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_summary.json"
)


def load_blocked_signals(summary_path: Path) -> pd.DataFrame:
    with open(summary_path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for block in data["entry_blocks"]:
        if block.get("reason") not in (
            "risk_pct_too_tight",
            "close_confirmed_risk_pct_too_tight",
        ):
            continue
        rows.append({
            "signal_date": block.get("date"),
            "code": block.get("code"),
            "name": block.get("name"),
            "signal_type": block.get("signal_type"),
            "entry_risk_pct": block.get("entry_risk_pct"),
            "entry_reward_risk": block.get("entry_reward_risk"),
            "trigger_raw": block.get("signal_trigger_raw_price"),
            "stop_raw": block.get("signal_stop_raw_price"),
            "candidate_score": block.get("candidate_score"),
            "reason": block.get("reason"),
        })
    return pd.DataFrame(rows)


def add_forward_returns(signals: pd.DataFrame, repo: TushareRepository) -> pd.DataFrame:
    """For each signal, compute forward close returns at N trading days."""
    codes = sorted(signals["code"].unique())
    # Load slightly wider than needed to guarantee enough forward bars for
    # signals near the sample end.
    start = signals["signal_date"].min()
    end = "2026-12-31"
    price = repo.load_daily_kline_frames(codes, start_date=start, end_date=end)
    if price.empty:
        return signals
    price = price.sort_values(["code", "date"]).reset_index(drop=True)

    # qfq-equivalent close (raw close * raw_to_qfq_factor rescale is already
    # end-anchored; using factor ratio removes dividend jumps for returns).
    price["qfq_close"] = price["close"] * price["raw_to_qfq_factor"]

    by_code = {code: group.reset_index(drop=True) for code, group in price.groupby("code")}
    horizons = (5, 10, 20, 40, 60)
    out_rows = []
    for sig in signals.to_dict("records"):
        code = sig["code"]
        frame = by_code.get(code)
        if frame is None or frame.empty:
            continue
        dates = frame["date"].astype(str).tolist()
        sig_date = str(sig["signal_date"])
        if sig_date not in dates:
            continue
        idx = dates.index(sig_date)
        base_close = float(frame.iloc[idx]["qfq_close"])
        if not np.isfinite(base_close) or base_close <= 0:
            continue
        row = dict(sig)
        for horizon in horizons:
            j = idx + horizon
            row[f"fwd_{horizon}d"] = np.nan
            if j < len(frame):
                fwd_close = float(frame.iloc[j]["qfq_close"])
                if np.isfinite(fwd_close) and fwd_close > 0:
                    row[f"fwd_{horizon}d"] = fwd_close / base_close - 1.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize(df: pd.DataFrame) -> None:
    print(f"\n=== blocked tight-risk signals: {len(df)} rows ===")
    print("by signal_type:")
    for k, v in Counter(df["signal_type"]).most_common():
        print(f"  {k}: {v}")
    print("by year:")
    for k, v in Counter(df["signal_date"].str[:4]).most_common():
        print(f"  {k}: {v}")

    for horizon in (5, 10, 20, 40, 60):
        col = f"fwd_{horizon}d"
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(
            f"\nforward {horizon:>2}d  n={len(vals):>4}  "
            f"mean={vals.mean()*100:>7.2f}%  median={vals.median()*100:>7.2f}%  "
            f"p25={vals.quantile(0.25)*100:>7.2f}%  p75={vals.quantile(0.75)*100:>7.2f}%  "
            f"win%={(vals>0).mean()*100:>5.1f}%"
        )

    # Breakdown by candidate_score tercile at 20d.
    col = "fwd_20d"
    vals = df.dropna(subset=[col])
    if not vals.empty and "candidate_score" in vals:
        q33, q66 = vals["candidate_score"].quantile([1 / 3, 2 / 3])
        print("\n=== fwd_20d by candidate_score tercile ===")
        for label, mask in (
            ("low (<33%)", vals["candidate_score"] < q33),
            ("mid (33-66%)", (vals["candidate_score"] >= q33) & (vals["candidate_score"] < q66)),
            ("high (>=66%)", vals["candidate_score"] >= q66),
        ):
            v = vals.loc[mask, col]
            if not v.empty:
                print(
                    f"  {label:>14} n={len(v):>4} mean={v.mean()*100:>7.2f}% "
                    f"median={v.median()*100:>7.2f}% win%={(v>0).mean()*100:>5.1f}%"
                )

    # How many reached +20% at any point within 40d (max forward).
    max_cols = [c for c in df.columns if c.startswith("fwd_")]
    if max_cols:
        reach = df[max_cols].max(axis=1, skipna=True)
        reach20 = (reach >= 0.20).sum()
        reach10 = (reach >= 0.10).sum()
        print(
            f"\nwithin 60d: reached +10% at some point: {reach10} "
            f"({reach10/len(df)*100:.1f}%), reached +20%: {reach20} ({reach20/len(df)*100:.1f}%)"
        )

    # How many hit -5% or -10% drawdown vs signal close within 40d.
    if len(max_cols) >= 5:
        trough = df[max_cols].min(axis=1, skipna=True)
        dd5 = (trough <= -0.05).sum()
        dd10 = (trough <= -0.10).sum()
        print(
            f"within 60d: dipped <=-5% at some point: {dd5} ({dd5/len(df)*100:.1f}%), "
            f"<=-10%: {dd10} ({dd10/len(df)*100:.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=str(PATHS.project_root / DEFAULT_SUMMARY),
    )
    args = parser.parse_args()

    repo = TushareRepository(Database())
    signals = load_blocked_signals(Path(args.summary))
    print(f"loaded {len(signals)} blocked tight-risk signals")
    if signals.empty:
        return
    df = add_forward_returns(signals, repo)
    summarize(df)

    out = Path(args.summary).parent / "blocked_signal_forward_returns.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
