"""Measure the right-side candidate pool's forward-return baseline.

Loads every daily hybrid candidate snapshot, takes the right-side (factor) rows,
and measures forward returns (5/10/20/40/60d) from the snapshot date using the
qfq-scaled close basis. Also compares the actual executed buys and the blocked
tight-risk signals against the pool baseline, and splits by year and by
candidate_score tercile.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.storage.database import Database
from stock_research.storage.tushare_repository import TushareRepository


CAND_DIR = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "candidates/hybrid_low_corr_value"
)
SUMMARY = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_summary.json"
)
TRADES = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_trades.csv"
)
HORIZONS = (5, 10, 20, 40, 60)


def load_pool_rows(cand_dir: Path):
    rows = []
    for path in sorted(cand_dir.glob("candidates_*.csv")):
        df = pd.read_csv(path, encoding="utf-8")
        date = path.stem.split("_", 1)[1]
        for rec in df.to_dict("records"):
            # right factor lane only
            if str(rec.get("hybrid_lane", "")).find("right") < 0:
                continue
            rows.append({
                "date": date,
                "code": rec.get("code"),
                "name": rec.get("name"),
                "candidate_score": rec.get("candidate_score"),
                "playbook_factor_rank": rec.get("playbook_factor_rank"),
                "signal_eligible": rec.get("signal_eligible"),
                "selected_for_trading": rec.get("selected_for_trading"),
                "group": "pool",
            })
    return pd.DataFrame(rows)


def add_forward(df, price_by_code):
    out_rows = []
    for rec in df.to_dict("records"):
        frame = price_by_code.get(rec["code"])
        row = dict(rec)
        for h in HORIZONS:
            row[f"fwd_{h}d"] = np.nan
        if frame is None or frame.empty:
            out_rows.append(row)
            continue
        dates = frame["date"].tolist()
        date = rec["date"]
        if date not in dates:
            out_rows.append(row)
            continue
        idx = dates.index(date)
        base_close = float(frame["qfq_close"].iloc[idx])
        if not np.isfinite(base_close) or base_close <= 0:
            out_rows.append(row)
            continue
        for h in HORIZONS:
            j = idx + h
            if j < len(frame):
                fwd = float(frame["qfq_close"].iloc[j])
                if np.isfinite(fwd) and fwd > 0:
                    row[f"fwd_{h}d"] = fwd / base_close - 1.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize(df, label, horizons=HORIZONS):
    print(f"\n=== {label}: n={len(df)} ===")
    for h in horizons:
        col = f"fwd_{h}d"
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(
            f"   fwd {h:>2}d  n={len(vals):>6} mean={vals.mean()*100:>7.2f}% "
            f"median={vals.median()*100:>7.2f}% "
            f"p25={vals.quantile(0.25)*100:>7.2f}% p75={vals.quantile(0.75)*100:>7.2f}% "
            f"win%={(vals>0).mean()*100:>5.1f}%"
        )


def main():
    repo = TushareRepository(Database())
    cand_dir = PATHS.project_root / CAND_DIR
    pool = load_pool_rows(cand_dir)
    print(f"loaded {len(pool)} right-pool rows across {pool['date'].nunique()} days")

    codes = sorted(pool["code"].unique())
    price = repo.load_daily_kline_frames(codes, start_date=pool["date"].min(), end_date="2026-12-31")
    price = price.sort_values(["code", "date"]).reset_index(drop=True)
    price["qfq_close"] = price["close"] * price["raw_to_qfq_factor"]
    price["date"] = price["date"].astype(str)
    price_by_code = {code: g.reset_index(drop=True) for code, g in price.groupby("code")}

    pool = add_forward(pool, price_by_code)
    summarize(pool, "right candidate pool (daily Top50)")

    # by year
    print("\nby year (20d / 40d):")
    for y in sorted(pool["date"].str[:4].unique()):
        sub = pool[pool["date"].str[:4] == y]
        for h in (20, 40):
            v = sub[f"fwd_{h}d"].dropna()
            if not v.empty:
                print(
                    f"   {y} fwd{h}d n={len(v):>5} mean={v.mean()*100:>7.2f}% "
                    f"median={v.median()*100:>7.2f}% win%={(v>0).mean()*100:>5.1f}%"
                )

    # by candidate_score tercile (20d)
    col = "fwd_20d"
    sub = pool.dropna(subset=[col])
    if not sub.empty:
        q33, q66 = sub["candidate_score"].quantile([1 / 3, 2 / 3])
        print(f"\nby candidate_score tercile ({col}):")
        for label, mask in (
            ("low (<33%)", sub["candidate_score"] < q33),
            ("mid (33-66%)", (sub["candidate_score"] >= q33) & (sub["candidate_score"] < q66)),
            ("high (>=66%)", sub["candidate_score"] >= q66),
        ):
            v = sub.loc[mask, col]
            print(
                f"   {label:>12} n={len(v):>6} mean={v.mean()*100:>7.2f}% "
                f"median={v.median()*100:>7.2f}% win%={(v>0).mean()*100:>5.1f}%"
            )

    # Pool baseline vs executed vs blocked
    with open(PATHS.project_root / SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    blocked = []
    for b in summary["entry_blocks"]:
        if b.get("reason") in ("risk_pct_too_tight", "close_confirmed_risk_pct_too_tight"):
            blocked.append({"date": b["date"], "code": b["code"], "group": "blocked"})
    blocked = pd.DataFrame(blocked)

    trades = pd.read_csv(PATHS.project_root / TRADES, encoding="utf-8")
    buys = trades[trades["trade_side"] == "买入"]
    executed = pd.DataFrame({
        "date": buys["date"],
        "code": buys["code"],
        "group": "executed_all",
    })

    comp = pd.concat([pool[["date", "code", "group"]], blocked, executed], ignore_index=True)
    comp = add_forward(comp, price_by_code)
    print("\n=== pool vs blocked vs executed (20d) ===")
    for g, mask in (("pool", comp["group"] == "pool"),
                    ("blocked", comp["group"] == "blocked"),
                    ("executed_all", comp["group"] == "executed_all")):
        v = comp.loc[mask, "fwd_20d"].dropna()
        if not v.empty:
            print(
                f"   {g:>12} n={len(v):>6} mean={v.mean()*100:>7.2f}% "
                f"median={v.median()*100:>7.2f}% win%={(v>0).mean()*100:>5.1f}%"
            )

    out = PATHS.project_root / "var" / "tmp" / "candidate_pool_forward.csv"
    pool.to_csv(out, index=False, encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
