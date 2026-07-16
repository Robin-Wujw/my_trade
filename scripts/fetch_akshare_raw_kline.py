"""Fetch AkShare unadjusted daily K-lines for as-of qfq reconstruction."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="抓取AkShare不复权日线缓存")
    parser.add_argument("--universe-path", default="var/cache/stock_universe.csv")
    parser.add_argument("--codes", nargs="*", default=None, help="可选：只抓指定代码")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--output-directory",
        default="var/cache/formula33_kline/akshare_raw",
    )
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--node-executable",
        default=str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"),
    )
    parser.add_argument("--allow-insecure", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _codes_from_universe(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"code": str})
    return sorted({
        str(code).split(".")[-1].zfill(6)
        for code in frame.get("code", pd.Series(dtype=str)).dropna()
    })


def _normalize_code(code: str) -> tuple[str, str, str]:
    pure = str(code).split(".")[-1].zfill(6)
    market = "sh" if pure.startswith(("6", "9")) else "sz"
    return pure, market, f"{market}{pure}"


def _eastmoney_url(pure: str, start_date: str, end_date: str) -> str:
    market = "1" if pure.startswith(("6", "9")) else "0"
    beg = start_date.replace("-", "")
    end = end_date.replace("-", "")
    return (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={market}.{pure}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        "&klt=101&fqt=0"
        f"&beg={beg}&end={end}&smplmt=10000&lmt=1000000"
    )


def _normalize_eastmoney_payload(payload: dict, code: str, market: str) -> pd.DataFrame:
    klines = ((payload.get("data") or {}).get("klines") or [])
    rows = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 7:
            continue
        rows.append({
            "date": parts[0],
            "code": f"{market}.{code}",
            "open": parts[1],
            "close": parts[2],
            "high": parts[3],
            "low": parts[4],
            "volume": parts[5],
            "amount": parts[6],
            "turnover": parts[10] if len(parts) > 10 else None,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["date", "open", "high", "low", "close"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _fetch_raw_kline(
    pure: str,
    market: str,
    start_date: str,
    end_date: str,
    *,
    node_executable: str,
    allow_insecure: bool,
) -> pd.DataFrame:
    helper = Path(__file__).resolve().parent / "eastmoney_fetch.js"
    env = dict(os.environ)
    env["EASTMONEY_ALLOW_INSECURE"] = "1" if allow_insecure else "0"
    completed = subprocess.run(
        [node_executable, str(helper), _eastmoney_url(pure, start_date, end_date)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"eastmoney helper exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    return _normalize_eastmoney_payload(payload, pure, market)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    codes = (
        sorted({str(code).split(".")[-1].zfill(6) for code in args.codes})
        if args.codes else _codes_from_universe(args.universe_path)
    )
    ok = failed = skipped = 0
    for index, code in enumerate(codes, start=1):
        pure, market, daily_symbol = _normalize_code(code)
        target = output / f"{market}_{pure}.csv"
        if target.exists() and not args.force:
            skipped += 1
            continue
        try:
            frame = _fetch_raw_kline(
                pure,
                market,
                args.start_date,
                args.end_date,
                node_executable=args.node_executable,
                allow_insecure=args.allow_insecure,
            )
            if frame.empty:
                failed += 1
                print(f"[{index}/{len(codes)}] {pure} 空结果")
                continue
            temporary = target.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
            temporary.replace(target)
            ok += 1
            print(f"[{index}/{len(codes)}] {pure} rows={len(frame)}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(codes)}] {pure} failed: {exc}")
        if args.sleep > 0:
            time.sleep(args.sleep)
    print(f"AkShare不复权K线完成 ok={ok} skipped={skipped} failed={failed} output={output}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
