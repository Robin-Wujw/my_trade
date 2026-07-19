"""Fetch Tushare unadjusted daily K-lines by trade date.

This is faster than per-symbol fetching under Tushare's 50 calls/minute limit:
one request returns the whole market for one trade date.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按交易日批量抓取 Tushare 不复权日线")
    parser.add_argument("--universe-path", default="var/cache/stock_universe.csv")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--output-directory",
        default="var/cache/formula33_kline/akshare_raw",
    )
    parser.add_argument("--sleep", type=float, default=1.25)
    parser.add_argument(
        "--node-executable",
        default=str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"),
    )
    parser.add_argument("--allow-insecure", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--flush-days",
        type=int,
        default=40,
        help="累计多少个有效交易日后写一次文件，降低内存占用",
    )
    return parser.parse_args(argv)


def _codes_from_universe(path: str | Path) -> set[str]:
    frame = pd.read_csv(path, dtype={"code": str})
    return {
        str(code).split(".")[-1].zfill(6)
        for code in frame.get("code", pd.Series(dtype=str)).dropna()
    }


def _market_from_tushare_code(ts_code: str) -> tuple[str, str]:
    pure, suffix = str(ts_code).split(".", 1)
    market = "sh" if suffix.upper() == "SH" else "sz"
    return pure.zfill(6), market


def _query_daily_by_date(
    trade_date: str,
    *,
    node_executable: str,
    allow_insecure: bool,
) -> pd.DataFrame:
    helper = Path(__file__).resolve().parent / "tushare_query_params.js"
    env = dict(os.environ)
    env["TUSHARE_ALLOW_INSECURE"] = "1" if allow_insecure else "0"
    completed = subprocess.run(
        [
            node_executable,
            str(helper),
            "daily",
            json.dumps({"trade_date": trade_date}, ensure_ascii=True),
            FIELDS,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"tushare helper exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    if int(payload.get("code", -1)) != 0:
        raise RuntimeError(payload.get("msg") or payload.get("detail") or "Tushare daily failed")
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not items:
        return pd.DataFrame()
    frame = pd.DataFrame(items, columns=fields)
    result = pd.DataFrame({
        "ts_code": frame["ts_code"],
        "date": pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d"),
        "open": frame["open"],
        "close": frame["close"],
        "high": frame["high"],
        "low": frame["low"],
        "volume": frame["vol"],
        "amount": frame["amount"],
    })
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["amount"] = result["amount"] * 1000.0
    return result.dropna(subset=["date", "open", "high", "low", "close"])


def _target_path(output: Path, pure: str, market: str) -> Path:
    return output / f"{market}_{pure}.csv"


def _flush_rows(output: Path, rows_by_code: dict[tuple[str, str], list[dict]], *, force: bool) -> int:
    written = 0
    for (pure, market), rows in rows_by_code.items():
        if not rows:
            continue
        target = _target_path(output, pure, market)
        incoming = pd.DataFrame(rows)
        if target.exists() and not force:
            try:
                existing = pd.read_csv(target)
            except (OSError, ValueError):
                existing = pd.DataFrame()
            frame = pd.concat([existing, incoming], ignore_index=True)
        else:
            frame = incoming
        frame = (
            frame.dropna(subset=["date", "open", "high", "low", "close"])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        temporary = target.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        written += 1
    rows_by_code.clear()
    return written


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    universe = _codes_from_universe(args.universe_path)
    rows_by_code: dict[tuple[str, str], list[dict]] = {}
    valid_days = empty_days = failed_days = written_files = 0
    date_range = pd.date_range(args.start_date, args.end_date, freq="B")

    for index, date in enumerate(date_range, start=1):
        trade_date = date.strftime("%Y%m%d")
        try:
            frame = _query_daily_by_date(
                trade_date,
                node_executable=args.node_executable,
                allow_insecure=args.allow_insecure,
            )
            if frame.empty:
                empty_days += 1
                print(f"[{index}/{len(date_range)}] {trade_date} 空结果")
            else:
                valid_days += 1
                kept = 0
                for row in frame.to_dict("records"):
                    pure, market = _market_from_tushare_code(row["ts_code"])
                    if pure not in universe:
                        continue
                    rows_by_code.setdefault((pure, market), []).append({
                        "date": row["date"],
                        "code": f"{market}.{pure}",
                        "open": row["open"],
                        "close": row["close"],
                        "high": row["high"],
                        "low": row["low"],
                        "volume": row["volume"],
                        "amount": row["amount"],
                    })
                    kept += 1
                print(f"[{index}/{len(date_range)}] {trade_date} rows={len(frame)} kept={kept}")
        except Exception as exc:
            failed_days += 1
            print(f"[{index}/{len(date_range)}] {trade_date} failed: {exc}")
        if valid_days and valid_days % max(1, args.flush_days) == 0:
            written_files += _flush_rows(output, rows_by_code, force=args.force)
            args.force = False
        if args.sleep > 0:
            time.sleep(args.sleep)

    written_files += _flush_rows(output, rows_by_code, force=args.force)
    print(
        "Tushare按日期不复权K线完成 "
        f"valid_days={valid_days} empty_days={empty_days} failed_days={failed_days} "
        f"written_files={written_files} output={output}"
    )
    if failed_days:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
