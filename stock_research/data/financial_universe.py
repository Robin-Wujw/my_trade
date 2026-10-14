"""Broad security master for financial synchronization, including delistings."""
from __future__ import annotations

import pandas as pd


def financial_stock_codes(repository) -> list[str]:
    frames = [repository.load_dataset(name) for name in ("stock_basic", "basic")]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("stock_basic/basic is not cached; synchronize all listing statuses first")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "ts_code" not in frame:
        raise RuntimeError("cached security master has no ts_code column")
    # Unknown statuses must not silently turn into current listings. A row
    # without a status can still be included: this is a fetch universe, not a
    # claim of historical trading eligibility.
    if "list_status" in frame:
        status = frame["list_status"].fillna("").astype(str).str.strip().str.upper()
        frame = frame[status.isin(("L", "D", "P", ""))]
    codes = frame["ts_code"].dropna().astype(str).str.strip()
    return sorted(set(codes[codes.ne("")]))
