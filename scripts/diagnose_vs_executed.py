"""Compare forward returns: blocked tight-risk signals vs actually-executed right buys.

Both are measured from the signal/buy date using the same qfq-scaled close basis,
so the only difference is the engine's gate decision. If executed buys have much
better forward returns, the gate has selection alpha; if similar, selection lies
upstream in the candidate pool.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.storage.database import Database
from stock_research.storage.tushare_repository import TushareRepository


SUMMARY = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_summary.json"
)
TRADES = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_trades.csv"
)


def forward_returns_for(price_by_code, code, date, horizons=(5, 10, 20, 40, 60)):
    frame = price_by_code.get(code)
    if frame is None or frame.empty:
        return {}
    dates = frame["date"].tolist()
    if date not in dates:
        return {}
    idx = dates.index(date)
    base_close = float(frame["qfq_close"].iloc[idx])
    if not np.isfinite(base_close) or base_close <= 0:
        return {}
    res = {}
    for h in horizons:
        j = idx + h
        res[f"fwd_{h}d"] = np.nan
        if j < len(frame):
            fwd = float(frame["qfq_close"].iloc[j])
            if np.isfinite(fwd) and fwd > 0:
                res[f"fwd_{h}d"] = fwd / base_close - 1.0
    return res


def main():
    repo = TushareRepository(Database())

    with open(PATHS.project_root / SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    blocked = []
    for b in summary["entry_blocks"]:
        if b.get("reason") in ("risk_pct_too_tight", "close_confirmed_risk_pct_too_tight"):
            blocked.append({"date": b["date"], "code": b["code"], "signal_type": b.get("signal_type"),
                            "risk_pct": b.get("entry_risk_pct"), "group": "blocked"})
    blocked = pd.DataFrame(blocked)

    trades = pd.read_csv(PATHS.project_root / TRADES, encoding="utf-8")
    buys = trades[trades["trade_side"] == "买入"]
    right_buys = buys[buys["signal_type"].notna()].copy()
    executed = pd.DataFrame({
        "date": right_buys["date"],
        "code": right_buys["code"],
        "signal_type": right_buys["signal_type"],
        "group": "executed",
    })

    both = pd.concat([blocked, executed], ignore_index=True)
    codes = sorted(both["code"].unique())
    price = repo.load_daily_kline_frames(codes, start_date=both["date"].min(), end_date="2026-12-31")
    price = price.sort_values(["code", "date"]).reset_index(drop=True)
    price["qfq_close"] = price["close"] * price["raw_to_qfq_factor"]
    price["date"] = price["date"].astype(str)
    price_by_code = {code: group.reset_index(drop=True) for code, group in price.groupby("code")}

    rows = []
    for rec in both.to_dict("records"):
        fwd = forward_returns_for(price_by_code, rec["code"], str(rec["date"]))
        row = dict(rec)
        row.update(fwd)
        rows.append(row)
    out = pd.DataFrame(rows)

    print("=== blocked (tight-risk rejected) vs executed right buys ===")
    for group, mask in (("blocked", out["group"] == "blocked"), ("executed", out["group"] == "executed")):
        sub = out[mask]
        print(f"\n--- {group}: n={len(sub)} ---")
        print("by signal_type:")
        for k, v in sub["signal_type"].value_counts().items():
            print(f"   {k}: {v}")
        for h in (5, 10, 20, 40, 60):
            col = f"fwd_{h}d"
            vals = sub[col].dropna()
            if vals.empty:
                continue
            print(
                f"   fwd {h:>2}d  n={len(vals):>4} mean={vals.mean()*100:>7.2f}% "
                f"median={vals.median()*100:>7.2f}% win%={(vals>0).mean()*100:>5.1f}%"
            )

    out.to_csv(PATHS.project_root / "var" / "tmp" / "blocked_vs_executed_forward.csv", index=False, encoding="utf-8")
    print("\nwrote var/tmp/blocked_vs_executed_forward.csv")


if __name__ == "__main__":
    main()
