"""Stamp factor_horizon_channel_eligible onto a COPY of the frozen candidate snapshots.

The engine's factor-horizon channel (candidate next-open buy + fixed-horizon
exit) only fires when a candidate carries factor_horizon_channel_eligible=True.
The frozen hybrid snapshots predate that label. This script writes a new snapshot
directory where every right-lane row with playbook_factor_rank <= max_rank gets
the label, leaving the original directory untouched (point-in-time data intact).

The channel itself re-checks rank, structure proximity, liquidity and return
filters inside the engine, so stamping a generous rank bound is safe.

Usage:
    python -m scripts.stamp_factor_horizon_eligible --max-rank 10 --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS


SRC = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "candidates/hybrid_low_corr_value"
)
DEFAULT_OUT = (
    "var/backtests/factor_horizon_2021_to_20260721/"
    "candidates/hybrid_low_corr_value"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--out", default=str(PATHS.project_root / DEFAULT_OUT))
    args = parser.parse_args()

    src = PATHS.project_root / SRC
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_stamped = 0
    for path in sorted(src.glob("candidates_*.csv")):
        df = pd.read_csv(path, encoding="utf-8")
        # right factor lane rows
        lane = df.get("hybrid_lane")
        is_right = lane.fillna("").astype(str).str.contains("right", na=False)
        rank = pd.to_numeric(df.get("playbook_factor_rank"), errors="coerce")
        eligible = is_right & rank.notna() & (rank <= args.max_rank)
        df["factor_horizon_channel_eligible"] = eligible.fillna(False)
        n_stamped += int(eligible.sum())
        # forward-fill per code: a code stays eligible once it qualifies (the
        # channel can re-enter after exit). Keep it simple: stamp the row as-is,
        # engine handles re-entry via capacity/position checks.
        dest = out / path.name
        df.to_csv(dest, index=False, encoding="utf-8")
        n_files += 1

    print(f"copied {n_files} snapshot files to {out}")
    print(f"stamped {n_stamped} right-lane rows (rank <= {args.max_rank})")


if __name__ == "__main__":
    main()
