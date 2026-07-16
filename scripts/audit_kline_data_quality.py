"""Audit cached daily kline data used by right-side selection and backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="审计全市场日线缓存的数据质量")
    parser.add_argument(
        "--kline-directory",
        default="var/cache/formula33_kline/akshare",
        help="日线CSV目录",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--output-directory",
        default="var/cache/kline_data_quality_audit",
        help="审计输出目录",
    )
    return parser.parse_args(argv)


def _pct(part: int, total: int) -> float:
    return round(part / total * 100, 6) if total else 0.0


def audit_file(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    symbol = path.stem
    result = {
        "symbol": symbol,
        "rows": 0,
        "duplicate_dates": 0,
        "invalid_ohlc_rows": 0,
        "missing_or_zero_volume_rows": 0,
        "missing_or_zero_amount_rows": 0,
        "tradestatus_no_trade_with_volume_rows": 0,
        "amount_unit_suspicious_rows": 0,
        "extreme_adjusted_price_jump_rows": 0,
        "first_date": None,
        "last_date": None,
        "error": "",
    }
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - audit script
        result["error"] = f"read_failed:{exc}"
        return result
    if "date" not in frame:
        result["error"] = "missing_date_column"
        return result
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if frame.empty:
        result["error"] = "no_rows_in_range"
        return result
    result["rows"] = int(len(frame))
    result["duplicate_dates"] = int(frame["date"].duplicated().sum())
    result["first_date"] = pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d")
    result["last_date"] = pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d")

    for column in ("open", "high", "low", "close", "volume", "amount", "tradestatus"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    open_ = frame.get("open", pd.Series(float("nan"), index=frame.index))
    high = frame.get("high", pd.Series(float("nan"), index=frame.index))
    low = frame.get("low", pd.Series(float("nan"), index=frame.index))
    close = frame.get("close", pd.Series(float("nan"), index=frame.index))
    volume = frame.get("volume", pd.Series(float("nan"), index=frame.index))
    amount = frame.get("amount", pd.Series(float("nan"), index=frame.index))
    tradestatus = frame.get("tradestatus", pd.Series(float("nan"), index=frame.index))

    valid_ohlc = (
        open_.gt(0)
        & high.gt(0)
        & low.gt(0)
        & close.gt(0)
        & high.ge(low)
        & high.ge(open_)
        & high.ge(close)
        & low.le(open_)
        & low.le(close)
    )
    result["invalid_ohlc_rows"] = int((~valid_ohlc).sum())
    result["missing_or_zero_volume_rows"] = int((volume.isna() | volume.le(0)).sum())
    result["missing_or_zero_amount_rows"] = int((amount.isna() | amount.le(0)).sum())
    result["tradestatus_no_trade_with_volume_rows"] = int(
        (tradestatus.notna() & tradestatus.ne(1) & volume.gt(0)).sum()
    )

    proxy_amount = close.mul(volume)
    amount_ratio = amount.div(proxy_amount.where(proxy_amount > 0))
    # Different providers store volume either as shares or hands.  A healthy
    # yuan amount is therefore usually close to close*volume or close*volume*100.
    result["amount_unit_suspicious_rows"] = int(
        (amount.gt(0) & proxy_amount.gt(0) & ((amount_ratio < 0.2) | (amount_ratio > 500))).sum()
    )
    close_return = close.pct_change().abs()
    result["extreme_adjusted_price_jump_rows"] = int((close_return > 0.35).sum())
    return result


def main(argv=None):
    args = parse_args(argv)
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    output_dir = Path(args.output_directory) / f"{start:%Y%m%d}_{end:%Y%m%d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        audit_file(path, start, end)
        for path in sorted(Path(args.kline_directory).glob("*.csv"))
    ]
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "kline_data_quality_detail.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    total_rows = int(frame["rows"].sum()) if not frame.empty else 0
    issue_columns = [
        "duplicate_dates",
        "invalid_ohlc_rows",
        "missing_or_zero_volume_rows",
        "missing_or_zero_amount_rows",
        "tradestatus_no_trade_with_volume_rows",
        "amount_unit_suspicious_rows",
        "extreme_adjusted_price_jump_rows",
    ]
    summary = {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "symbol_files": int(len(frame)),
        "total_rows": total_rows,
        "symbols_with_errors": int(frame["error"].astype(str).ne("").sum()) if not frame.empty else 0,
        "issue_rows": {
            column: int(frame[column].sum()) for column in issue_columns if column in frame
        },
        "issue_rates_pct": {
            column: _pct(int(frame[column].sum()), total_rows)
            for column in issue_columns if column in frame
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细: {csv_path}")


if __name__ == "__main__":
    main()
