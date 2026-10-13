"""Stamp allow_right=False onto a COPY of the frozen candidate snapshots.

This disables the right factor lane entirely (the engine gates right entries on
allow_right), isolating the left-value lane's standalone contribution. The
original directory is untouched.

Usage:
    python -m scripts.stamp_left_only --out DIR
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS


SRC = (
    "var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/"
    "candidates/hybrid_low_corr_value"
)
DEFAULT_OUT = "var/backtests/left_only_2021_to_20260721/candidates/hybrid_low_corr_value"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PATHS.project_root / DEFAULT_OUT))
    args = parser.parse_args()

    src = PATHS.project_root / SRC
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n_files = 0
    n_right_stamped = 0
    for path in sorted(src.glob("candidates_*.csv")):
        df = pd.read_csv(path, encoding="utf-8")
        lane = df.get("hybrid_lane")
        is_right = lane.fillna("").astype(str).str.contains("right", na=False)
        if is_right.any():
            # right_candidate_can_evaluate passes on selected_for_trading +
            # signal_eligible without checking allow_right, so zero them out
            # to fully disable the right lane.
            for col in ("selected_for_trading", "signal_eligible", "allow_right"):
                if col in df.columns:
                    df.loc[is_right, col] = False
            n_right_stamped += int(is_right.sum())
        df.to_csv(out / path.name, index=False, encoding="utf-8")
        n_files += 1

    shutil.copy2(src / "manifest.json", out / "manifest.json")
    print(f"copied {n_files} snapshots to {out}")
    print(f"stamped allow_right=False on {n_right_stamped} right-lane rows")


if __name__ == "__main__":
    main()
