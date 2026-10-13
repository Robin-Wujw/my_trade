"""Upper-bound estimate: trade the frozen right-side candidate pool directly.

Point-in-time rule: each trading day's snapshot is formed from data up to that
day's close. Execution happens at the NEXT trading day's open (T+1). Portfolio
rebalances every `hold` trading days: on execution day e we liquidate everything
at e's open and buy the top-K factor candidates from the PREVIOUS trading day's
snapshot at e's open. Costs match the engine defaults (commission 0.0085% min 5,
sell stamp duty 0.05%, slippage 0.05% on both sides), board lots of 100.

This isolates the pool's selection alpha from the engine's complex gates. It is
an UPPER BOUND: real execution adds spread, partial fills and limit-up/down
friction that a top-K rotation already ignores. It answers: can the pool alone
reach +300% over 5.5 years?

Usage:
    python -m scripts.diagnose_pool_direct --topk 5 --hold 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.storage.database import Database
from stock_research.storage.tushare_repository import TushareRepository


CAND_DIR = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "candidates/hybrid_low_corr_value"
)
COMMISSION = 0.000085
MIN_COMMISSION = 5.0
STAMP_DUTY = 0.0005
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 1_000_000.0


def load_topk_per_day(cand_dir: Path) -> pd.DataFrame:
    """Rows: snapshot_date, code, playbook_factor_rank — right lane only."""
    rows = []
    for path in sorted(cand_dir.glob("candidates_*.csv")):
        df = pd.read_csv(path, encoding="utf-8")
        date = path.stem.split("_", 1)[1]
        lane = df.get("hybrid_lane")
        mask = lane.isna() if lane is None else lane.str.contains("right", na=False)
        sub = df[mask]
        for rec in sub.to_dict("records"):
            rows.append({
                "snapshot_date": date,
                "code": rec.get("code"),
                "rank": rec.get("playbook_factor_rank"),
            })
    out = pd.DataFrame(rows)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    return out


def run_topk_rotation(snaps, price_by_code, calendar, *, topk, hold):
    """Full-rebalance rotation with overlap-free accounting.

    On execution day e (calendar[k*hold] for k>=1): liquidate all at open of e,
    then buy top-K from snapshot formed on the previous trading day.
    Equity valued at close every calendar day.
    """
    # Snapshot date -> ranked codes (topk); key as str to match calendar.
    snap_top = {}
    for date, group in snaps.groupby("snapshot_date"):
        top = group.sort_values("rank", ascending=True).head(topk)["code"].tolist()
        snap_top[date.strftime("%Y-%m-%d")] = top

    # Execution days: every `hold`-th trading day, starting from first calendar day.
    # snapshot used for execution day e is the previous trading day.
    exec_days = calendar[hold::hold]  # every hold-th index: hold, 2*hold, ...

    # maps
    frames = {}
    for code, fr in price_by_code.items():
        frames[code] = {
            "dates": fr["date"].tolist(),
            "qopen": (fr["open"] * fr["raw_to_qfq_factor"]).to_numpy(),
            "qclose": (fr["close"] * fr["raw_to_qfq_factor"]).to_numpy(),
        }

    holdings = {}  # code -> shares
    cash = INITIAL_CAPITAL
    equity_curve = []

    def val_at(day_str):
        total = cash
        for code, shares in holdings.items():
            f = frames.get(code)
            if f is None or day_str not in f["dates"]:
                continue
            total += shares * float(f["qclose"][f["dates"].index(day_str)])
        return total

    def exec_buy(day_str):
        """Rotate at day open: sell what can be sold (suspended positions carry
        forward), then buy top-K from the previous trading day's snapshot with
        the resulting cash. Strict point-in-time: snapshot formed at the
        previous day's close, executed at this day's open."""
        nonlocal cash, holdings
        pos = calendar.index(day_str)
        snap_day = calendar[pos - 1] if pos >= 1 else None
        top = snap_top.get(snap_day) or []
        if not top:
            return

        # 1) Sell current holdings that have a bar today; keep suspended ones.
        kept = {}
        proceeds = cash
        for code, shares in holdings.items():
            f = frames.get(code)
            if f is None or day_str not in f["dates"]:
                kept[code] = shares
                continue
            idx = f["dates"].index(day_str)
            fill = float(f["qopen"][idx]) * (1 - SLIPPAGE)
            proceeds += shares * fill
            fee = max(COMMISSION * shares * fill, MIN_COMMISSION)
            stamp = shares * fill * STAMP_DUTY
            proceeds -= fee + stamp
        cash = proceeds
        holdings = dict(kept)

        # 2) Buy top-K with available cash, skipping codes already held.
        available = cash
        n = len(top)
        budget_per = available / n
        for code in top:
            if code in holdings:
                continue
            f = frames.get(code)
            if f is None or day_str not in f["dates"]:
                continue
            idx = f["dates"].index(day_str)
            fill = float(f["qopen"][idx]) * (1 + SLIPPAGE)
            if fill <= 0:
                continue
            shares = int(budget_per / fill / 100) * 100
            if shares <= 0:
                continue
            cost = shares * fill
            fee = max(COMMISSION * cost, MIN_COMMISSION)
            if cost + fee > cash:
                shares = int((cash - fee) / fill / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * fill
                fee = max(COMMISSION * cost, MIN_COMMISSION)
            cash -= cost + fee
            holdings[code] = shares

    for day_str in calendar:
        if day_str in exec_days:
            exec_buy(day_str)
        equity_curve.append((day_str, val_at(day_str)))

    return pd.DataFrame(equity_curve, columns=["date", "equity"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--hold", type=int, default=20)
    args = parser.parse_args()

    repo = TushareRepository(Database())
    snaps = load_topk_per_day(PATHS.project_root / CAND_DIR)
    print(f"right-lane snapshot rows: {len(snaps)} across {snaps['snapshot_date'].nunique()} days")

    codes = sorted(snaps["code"].unique())
    start = snaps["snapshot_date"].min().strftime("%Y-%m-%d")
    price = repo.load_daily_kline_frames(codes, start_date=start, end_date="2026-12-31")
    price = price.sort_values(["code", "date"]).reset_index(drop=True)
    price["date"] = price["date"].astype(str)
    price_by_code = {code: g.reset_index(drop=True) for code, g in price.groupby("code")}

    # Global trading calendar = union of all bar dates.
    all_dates = set()
    for fr in price_by_code.values():
        all_dates.update(fr["date"].tolist())
    calendar = sorted(all_dates)
    print(f"union trading calendar: {len(calendar)} days  {calendar[0]}..{calendar[-1]}")

    eq = run_topk_rotation(snaps, price_by_code, calendar, topk=args.topk, hold=args.hold)
    final = eq["equity"].iloc[-1]
    total = final / INITIAL_CAPITAL - 1.0
    years = max(1e-9, (pd.to_datetime(eq["date"].iloc[-1]) - pd.to_datetime(eq["date"].iloc[0])).days / 365.25)
    cagr = (final / INITIAL_CAPITAL) ** (1 / years) - 1.0
    eq["peak"] = eq["equity"].cummax()
    dd = ((eq["equity"] - eq["peak"]) / eq["peak"]).min()

    print(f"\n=== top-{args.topk} direct rotation, hold {args.hold}d ===")
    print(f"period: {eq['date'].iloc[0]} .. {eq['date'].iloc[-1]}  ({years:.2f} yr)")
    print(f"final equity: {final:,.0f}  total return: {total:+.1%}  CAGR: {cagr:+.1%}")
    print(f"max drawdown: {dd:.1%}")

    out = PATHS.project_root / "var" / "tmp" / f"pool_direct_top{args.topk}_hold{args.hold}.csv"
    eq.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
