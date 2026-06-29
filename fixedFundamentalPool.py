# -*- coding: utf-8 -*-
"""Create an immutable fundamental candidate pool at a report formation date."""
import argparse
import os

import pandas as pd

from dailyFundamentalSelect import classify_method, latest_fundamental_snapshot
from point_in_time import audit_dates, write_metadata
from trade_utils import get_project_path


POOL_DIR = get_project_path(".cache/fundamental_pools")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-period", required=True)
    parser.add_argument("--formation-date", required=True)
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def build_pool(snapshot, report_period, formation_date):
    work = snapshot.copy()
    # A reviewed/refactored selection file is authoritative: freeze its normal
    # fundamental section instead of mixing in the separate value-left section.
    if "strategy_part" in work.columns:
        reviewed = work[work["strategy_part"].astype(str).str.startswith("2.")]
        if not reviewed.empty:
            work = reviewed
    work = work.drop_duplicates("code").copy()
    work["method"] = work["industry"].map(classify_method)
    for col in ["quality_score", "liquidity_score", "mktcap", "earnings_yoy"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work = work[
        (work["quality_score"] >= 70)
        & (work["liquidity_score"] >= 55)
        & (work["mktcap"] >= 100)
        & (work["earnings_yoy"] >= 0.10)
    ].copy()
    work["report_period"] = report_period
    work["pool_formation_date"] = formation_date
    work["pool_member"] = True
    work["pool_rule"] = "quality>=70; liquidity>=55; mktcap_at_formation>=100; earnings_yoy>=10%"
    work["mktcap_at_formation"] = work["mktcap"]
    return work.sort_values(["quality_score", "earnings_yoy", "code"], ascending=[False, False, True])


def main():
    args = parse_args()
    dates = audit_dates(args.report_period, args.formation_date)
    if dates["date_status"] == "unsafe":
        raise SystemExit("; ".join(dates["date_issues"]))
    if args.snapshot:
        snapshot_path = os.path.abspath(args.snapshot)
        snapshot = pd.read_csv(snapshot_path, dtype={"code": str}, low_memory=False)
    else:
        snapshot, snapshot_path = latest_fundamental_snapshot(args.report_period)
    if snapshot.empty:
        raise SystemExit("No historical fundamental snapshot found")
    output = args.output or os.path.join(
        POOL_DIR,
        f"fundamental_pool_{args.report_period.replace('-', '')}_{args.formation_date.replace('-', '')}.csv",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    pool = build_pool(snapshot, args.report_period, args.formation_date)
    pool.to_csv(output, index=False, encoding="utf-8-sig")
    write_metadata(output, {
        "kind": "fixed_fundamental_pool",
        "point_in_time_status": "warning",
        "point_in_time_note": "financial revision history is unavailable; report was used after statutory deadline",
        "source_snapshot": snapshot_path,
        **dates,
        "row_count": len(pool),
    })
    print(f"fixed fundamental pool={output} rows={len(pool)} source={snapshot_path}")


if __name__ == "__main__":
    main()
