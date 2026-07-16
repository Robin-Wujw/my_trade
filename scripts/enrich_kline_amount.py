"""Backfill real traded amount into Formula33 K-line cache.

The script only enriches data fields. It never changes OHLC prices, trading
rules, candidate thresholds, or portfolio execution logic.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_research.core.paths import PATHS


AMOUNT_AUDIT_VERSION = "kline-real-amount-v1"


def _code_from_cache_path(path: Path) -> str:
    return path.stem.split("_", 1)[-1].zfill(6)


def _market_code(code: str) -> str:
    pure = str(code).strip().split(".")[-1].zfill(6)
    market = "sh" if pure.startswith(("6", "9")) else "sz"
    return f"{market}.{pure}"


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _powershell_single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _normalize_provider_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "amount", "open", "high", "low", "close", "volume"])
    aliases = {
        "\u65e5\u671f": "date",
        "\u80a1\u7968\u4ee3\u7801": "code",
        "\u5f00\u76d8": "open",
        "\u6536\u76d8": "close",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u6362\u624b\u7387": "turnover",
        "date": "date",
        "code": "code",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover",
    }
    data = frame.rename(
        columns={
            column: aliases[str(column).strip()]
            for column in frame.columns
            if str(column).strip() in aliases
        }
    ).copy()
    if "date" not in data or "amount" not in data:
        return pd.DataFrame(columns=["date", "amount", "open", "high", "low", "close", "volume"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["code"] = _market_code(code)
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = [column for column in ("date", "code", "open", "high", "low", "close", "volume", "amount", "turnover") if column in data]
    return (
        data[keep]
        .dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _eastmoney_secid(code: str) -> str:
    pure = str(code).strip().split(".")[-1].zfill(6)
    market = "1" if pure.startswith(("6", "9")) else "0"
    return f"{market}.{pure}"


def fetch_eastmoney_amount_with_node(
    code: str,
    start_date: str,
    end_date: str,
    *,
    node_executable: str,
    allow_insecure: bool,
) -> pd.DataFrame:
    params = {
        "secid": _eastmoney_secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": str(start_date).replace("-", ""),
        "end": str(end_date).replace("-", ""),
        "smplmt": "10000",
        "lmt": "1000000",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params, safe=",")
    helper = PROJECT_ROOT / "scripts" / "eastmoney_fetch.js"
    env = os.environ.copy()
    ps_command = ""
    if allow_insecure:
        ps_command += "$env:EASTMONEY_ALLOW_INSECURE='1'; "
    ps_command += (
        "& "
        + _powershell_single_quote(node_executable)
        + " "
        + _powershell_single_quote(str(helper))
        + " "
        + _powershell_single_quote(url)
    )
    encoded_command = base64.b64encode(ps_command.encode("utf-16le")).decode("ascii")
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-EncodedCommand",
            encoded_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    payload = json.loads(completed.stdout)
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 11:
            continue
        rows.append({
            "date": parts[0],
            "open": parts[1],
            "close": parts[2],
            "high": parts[3],
            "low": parts[4],
            "volume": parts[5],
            "amount": parts[6],
            "turnover": parts[10],
        })
    return _normalize_provider_frame(pd.DataFrame(rows), code)


def fetch_akshare_amount(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    from stock_research.api import akshare as ak

    pure = str(code).strip().split(".")[-1].zfill(6)
    frame = ak.stock_zh_a_hist(
        symbol=pure,
        period="daily",
        start_date=str(start_date).replace("-", ""),
        end_date=str(end_date).replace("-", ""),
        adjust="qfq",
    )
    return _normalize_provider_frame(frame, pure)


def _provider_payload_to_frame(payload: dict, code: str) -> pd.DataFrame:
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 11:
            continue
        rows.append({
            "date": parts[0],
            "open": parts[1],
            "close": parts[2],
            "high": parts[3],
            "low": parts[4],
            "volume": parts[5],
            "amount": parts[6],
            "turnover": parts[10],
        })
    return _normalize_provider_frame(pd.DataFrame(rows), code)


def fetch_eastmoney_amount_from_json(code: str, json_directory: str) -> pd.DataFrame:
    pure = str(code).strip().split(".")[-1].zfill(6)
    path = Path(json_directory) / f"{pure}.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return _provider_payload_to_frame(payload, pure)


def fetch_tushare_amount_from_json(code: str, json_directory: str) -> pd.DataFrame:
    pure = str(code).strip().split(".")[-1].zfill(6)
    path = Path(json_directory) / f"{pure}.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    rows = data.get("items") or []
    if not fields or not rows:
        return pd.DataFrame(columns=["date", "amount", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=fields)
    renamed = frame.rename(
        columns={
            "trade_date": "date",
            "vol": "volume",
        }
    ).copy()
    if "date" in renamed:
        renamed["date"] = pd.to_datetime(renamed["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    if "amount" in renamed:
        renamed["amount"] = pd.to_numeric(renamed["amount"], errors="coerce") * 1000.0
    return _normalize_provider_frame(renamed, pure)


def fetch_real_amount(
    code: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    retry_delay: float,
    provider: str,
    node_executable: str,
    allow_insecure_node_fetch: bool,
    provider_json_directory: str,
) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if provider == "akshare":
                normalized = fetch_akshare_amount(code, start_date, end_date)
            elif provider == "eastmoney-json":
                normalized = fetch_eastmoney_amount_from_json(code, provider_json_directory)
            elif provider == "tushare-json":
                normalized = fetch_tushare_amount_from_json(code, provider_json_directory)
            else:
                normalized = fetch_eastmoney_amount_with_node(
                    code,
                    start_date,
                    end_date,
                    node_executable=node_executable,
                    allow_insecure=allow_insecure_node_fetch,
                )
            if normalized.empty:
                raise RuntimeError("empty provider amount frame")
            return normalized
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(max(0.0, retry_delay) * attempt)
    raise RuntimeError(f"fetch amount failed for {code}: {last_error}") from last_error


def _ohlc_match_status(local: pd.DataFrame, provider: pd.DataFrame) -> str:
    columns = [column for column in ("open", "high", "low", "close") if column in local and column in provider]
    if not columns:
        return "not_checked"
    merged = local[["date", *columns]].merge(
        provider[["date", *columns]],
        on="date",
        how="inner",
        suffixes=("_local", "_provider"),
    )
    if merged.empty:
        return "no_overlap"
    mismatches = 0
    for column in columns:
        left = pd.to_numeric(merged[f"{column}_local"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_provider"], errors="coerce")
        mismatches += int((left.sub(right).abs() > 0.02).sum())
    return "match" if mismatches == 0 else f"mismatch_rows={mismatches}"


def update_csv_amount(
    path: Path,
    *,
    start_date: str,
    end_date: str,
    force: bool,
    retries: int,
    retry_delay: float,
    provider: str,
    node_executable: str,
    allow_insecure_node_fetch: bool,
    provider_json_directory: str = "",
) -> tuple[pd.DataFrame, dict]:
    code = _code_from_cache_path(path)
    before_hash = _file_sha256(path)
    local = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    if "date" not in local:
        raise RuntimeError("local kline cache has no date column")
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mask = local["date"].between(start_date, end_date)
    if "amount" not in local:
        local["amount"] = pd.NA
    amount = pd.to_numeric(local["amount"], errors="coerce")
    missing = mask & (force | amount.isna() | amount.le(0))
    if not missing.any():
        return pd.DataFrame(), {
            "code": code,
            "cache_path": str(path),
            "before_file_hash": before_hash,
            "after_file_hash": before_hash,
            "updated_csv": False,
            "updated_rows": 0,
            "error_message": "",
        }
    fetch_start = local.loc[missing, "date"].min()
    fetch_end = local.loc[missing, "date"].max()
    provider_frame = fetch_real_amount(
        code,
        fetch_start,
        fetch_end,
        retries=retries,
        retry_delay=retry_delay,
        provider=provider,
        node_executable=node_executable,
        allow_insecure_node_fetch=allow_insecure_node_fetch,
        provider_json_directory=provider_json_directory,
    )
    provider_amount = provider_frame.set_index("date")["amount"]
    old_amount = amount.copy()
    local.loc[missing, "amount"] = local.loc[missing, "date"].map(provider_amount)
    new_amount = pd.to_numeric(local["amount"], errors="coerce")
    changed = missing & new_amount.gt(0)
    audit = local.loc[missing, ["date", "code", "close", "volume", "amount"]].copy()
    audit["code"] = _market_code(code)
    audit["old_amount"] = old_amount.loc[missing].to_numpy()
    audit["new_amount"] = new_amount.loc[missing].to_numpy()
    audit["amount_source_provider"] = provider
    audit["endpoint"] = "stock_zh_a_hist" if provider == "akshare" else "eastmoney_push2his_kline"
    audit["amount_unit"] = "yuan"
    audit["fetch_start"] = fetch_start
    audit["fetch_end"] = fetch_end
    audit["amount_quality_flag"] = audit["new_amount"].gt(0).map({True: "real_amount", False: "missing_after_fetch"})
    audit["ohlc_match_status"] = _ohlc_match_status(local.loc[mask].copy(), provider_frame)
    if changed.any():
        _atomic_write_csv(local, path)
    after_hash = _file_sha256(path)
    summary = {
        "code": code,
        "cache_path": str(path),
        "before_file_hash": before_hash,
        "after_file_hash": after_hash,
        "updated_csv": bool(changed.any()),
        "updated_rows": int(changed.sum()),
        "missing_after_fetch": int((missing & ~new_amount.gt(0)).sum()),
        "fetch_start": str(fetch_start),
        "fetch_end": str(fetch_end),
        "ohlc_match_status": _ohlc_match_status(local.loc[mask].copy(), provider_frame),
        "error_message": "",
    }
    return audit, summary


def sync_duckdb_amount(paths: list[Path], summaries: list[dict]) -> dict:
    try:
        from stock_research.storage import Database, KlineRepository
    except Exception as exc:
        return {"available": False, "updated_symbols": 0, "error_message": str(exc)}
    try:
        database = Database(PATHS.database, code_version=AMOUNT_AUDIT_VERSION)
        database.initialize()
        repository = KlineRepository(database)
        updated = 0
        changed_codes = {str(item.get("code")) for item in summaries if item.get("updated_csv")}
        for path in paths:
            code = _code_from_cache_path(path)
            if code not in changed_codes:
                continue
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            repository.upsert_stock_kline("akshare", _market_code(code), frame)
            updated += 1
        return {"available": True, "updated_symbols": updated, "error_message": ""}
    except Exception as exc:
        return {"available": False, "updated_symbols": 0, "error_message": str(exc)}


def _selected_paths(args) -> list[Path]:
    directory = Path(args.kline_directory)
    if args.codes:
        wanted = {str(code).strip().split(".")[-1].zfill(6) for code in args.codes.split(",") if str(code).strip()}
    elif args.provider == "eastmoney-json" and args.provider_json_directory:
        wanted = {
            path.stem.zfill(6)
            for path in Path(args.provider_json_directory).glob("*.json")
            if not path.name.startswith("_")
        }
    else:
        wanted = set()
    if wanted:
        paths = [
            directory / f"{'sh' if code.startswith(('6', '9')) else 'sz'}_{code}.csv"
            for code in sorted(wanted)
        ]
    else:
        paths = sorted(directory.glob("*.csv"))
    paths = [path for path in paths if path.is_file()]
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backfill real amount into Formula33 K-line cache")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--kline-directory", default=str(PATHS.cache / "formula33_kline" / "akshare"))
    parser.add_argument("--codes", default="", help="comma-separated six-digit codes; empty scans all cache files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace amount in range even if it already exists")
    parser.add_argument("--provider", choices=("eastmoney-node", "eastmoney-json", "tushare-json", "akshare"), default="eastmoney-node")
    parser.add_argument("--provider-json-directory", default="")
    parser.add_argument(
        "--node-executable",
        default=str(
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "bin"
            / "node.exe"
        ),
    )
    parser.add_argument(
        "--allow-insecure-node-fetch",
        action="store_true",
        help="allow Node fetch with NODE_TLS_REJECT_UNAUTHORIZED=0 when local CA validation is broken",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.8)
    args = parser.parse_args(argv)
    if not args.end_date:
        args.end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = PATHS.cache / "formula33_kline_amount_audit" / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    paths = _selected_paths(args)
    all_audit = []
    summaries = []
    for index, path in enumerate(paths, start=1):
        try:
            audit, summary = update_csv_amount(
                path,
                start_date=args.start_date,
                end_date=args.end_date,
                force=args.force,
                retries=args.retries,
                retry_delay=args.retry_delay,
                provider=args.provider,
                node_executable=args.node_executable,
                allow_insecure_node_fetch=args.allow_insecure_node_fetch,
                provider_json_directory=args.provider_json_directory,
            )
        except Exception as exc:
            summary = {
                "code": _code_from_cache_path(path),
                "cache_path": str(path),
                "before_file_hash": _file_sha256(path),
                "after_file_hash": _file_sha256(path),
                "updated_csv": False,
                "updated_rows": 0,
                "error_message": str(exc),
            }
            audit = pd.DataFrame()
        summaries.append(summary)
        if not audit.empty:
            all_audit.append(audit)
        if index % 100 == 0 or index == len(paths):
            print(f"amount backfill progress {index}/{len(paths)}")
    duckdb_sync = sync_duckdb_amount(paths, summaries)
    audit_frame = pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    _atomic_write_csv(audit_frame, audit_dir / "row_audit.csv")
    _atomic_write_csv(summary_frame, audit_dir / "symbol_summary.csv")
    payload = {
        "version": AMOUNT_AUDIT_VERSION,
        "run_id": run_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbol_count": len(paths),
        "updated_symbols": int(summary_frame.get("updated_csv", pd.Series(dtype=bool)).fillna(False).sum()) if not summary_frame.empty else 0,
        "updated_rows": int(summary_frame.get("updated_rows", pd.Series(dtype=float)).fillna(0).sum()) if not summary_frame.empty else 0,
        "error_symbols": int(summary_frame.get("error_message", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()) if not summary_frame.empty else 0,
        "duckdb_sync": duckdb_sync,
        "audit_directory": str(audit_dir),
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["error_symbols"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
