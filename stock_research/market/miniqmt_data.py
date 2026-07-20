"""MiniQMT historical market-data cache for backtests."""
from __future__ import annotations

from contextlib import suppress
import json
from pathlib import Path
import random
import subprocess
import textwrap
from typing import Any

import pandas as pd

from stock_research.api.miniqmt import MiniQmtConfig, MiniQmtError, MiniQmtSdkNotFound, load_miniqmt_config
from stock_research.core.paths import PATHS
from stock_research.storage import Database, KlineRepository


MINIQMT_KLINE_CACHE_VERSION = "miniqmt-kline-v1"
MINIQMT_SOURCE = "miniqmt"
SUPPORTED_BAR_PERIODS = {"1d", "1m", "5m", "15m", "30m", "1h"}
SUPPORTED_DIVIDEND_TYPES = {"none", "front", "back", "front_ratio", "back_ratio"}


def normalize_project_code(code: str) -> str:
    text = str(code).strip()
    if not text:
        return text
    lowered = text.lower()
    if "." in lowered:
        market, symbol = lowered.split(".", 1)
        if market in {"sh", "sz"}:
            return f"{market}.{symbol.zfill(6)}"
        if symbol in {"sh", "sz"}:
            return f"{symbol}.{market.zfill(6)}"
    symbol = lowered.zfill(6)
    market = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{market}.{symbol}"


def project_code_to_miniqmt(code: str) -> str:
    normalized = normalize_project_code(code)
    if "." not in normalized:
        return normalized.upper()
    market, symbol = normalized.split(".", 1)
    return f"{symbol.upper()}.{market.upper()}"


def miniqmt_code_to_project(code: str) -> str:
    text = str(code).strip()
    if "." not in text:
        return normalize_project_code(text)
    symbol, market = text.split(".", 1)
    return normalize_project_code(f"{market.lower()}.{symbol}")


def miniqmt_cache_directory(period: str = "1d", dividend_type: str = "front") -> Path:
    return PATHS.cache / "miniqmt_kline" / period / dividend_type


def miniqmt_cache_path(code: str, period: str = "1d", dividend_type: str = "front") -> Path:
    return miniqmt_cache_directory(period, dividend_type) / f"{normalize_project_code(code).replace('.', '_')}.csv"


def normalize_miniqmt_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[
            "date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus",
        ])
    data = frame.copy()
    if "date" not in data.columns:
        raise ValueError("MiniQMT K-line frame has no date column")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["code"] = normalize_project_code(code)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    if "tradestatus" not in data.columns:
        data["tradestatus"] = None
    columns = ["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus"]
    return (
        data[columns]
        .dropna(subset=["date", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def load_cached_miniqmt_frame(
    code: str,
    *,
    period: str = "1d",
    dividend_type: str = "front",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    path = miniqmt_cache_path(code, period, dividend_type)
    try:
        frame = normalize_miniqmt_frame(pd.read_csv(path, dtype={"code": str}), code)
    except (OSError, ValueError):
        return pd.DataFrame()
    if start_date:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") <= pd.Timestamp(end_date)]
    return frame.reset_index(drop=True)


def save_miniqmt_frame(
    code: str,
    frame: pd.DataFrame,
    *,
    period: str = "1d",
    dividend_type: str = "front",
    merge_existing: bool = False,
) -> Path:
    data = normalize_miniqmt_frame(frame, code)
    if merge_existing:
        existing = load_cached_miniqmt_frame(code, period=period, dividend_type=dividend_type)
        if not existing.empty:
            data = normalize_miniqmt_frame(
                pd.concat([existing, data], ignore_index=True),
                code,
            )
    path = miniqmt_cache_path(code, period, dividend_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{random.randint(100000, 999999)}.tmp")
    data.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(path)
    return path


def fetch_miniqmt_bars_via_qmt_python(
    codes,
    start_date: str,
    end_date: str,
    *,
    period: str = "1d",
    dividend_type: str = "front",
    config: MiniQmtConfig | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    """Fetch MiniQMT bars through QMT's bundled Python and update local CSV cache."""
    if period not in SUPPORTED_BAR_PERIODS:
        raise ValueError(f"unsupported MiniQMT period: {period}")
    if dividend_type not in SUPPORTED_DIVIDEND_TYPES:
        raise ValueError(f"unsupported MiniQMT dividend_type: {dividend_type}")
    cfg = config or load_miniqmt_config()
    python_executable = cfg.resolved_python_executable
    if not python_executable.is_file():
        raise MiniQmtSdkNotFound(f"MiniQMT Python executable not found: {python_executable}")

    normalized_codes = sorted({normalize_project_code(code) for code in codes if str(code).strip()})
    if not normalized_codes:
        return {"ok": True, "fetched": {}, "errors": [], "requested_count": 0}
    work_dir = PATHS.tmp / f"miniqmt_bars_{random.randint(100000, 999999)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "fetch_bars.py"
    result_path = work_dir / "result.json"
    script_path.write_text(
        _history_bridge_script(
            qmt_root=str(cfg.resolved_qmt_root),
            output_dir=str(work_dir),
            result_path=str(result_path),
            codes={code: project_code_to_miniqmt(code) for code in normalized_codes},
            start_time=pd.Timestamp(start_date).strftime("%Y%m%d"),
            end_time=pd.Timestamp(end_date).strftime("%Y%m%d"),
            period=period,
            dividend_type=dividend_type,
        ),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [str(python_executable), str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0 and not result_path.is_file():
            raise MiniQmtError(
                "MiniQMT bar bridge failed before writing a result: "
                f"returncode={completed.returncode} stderr={completed.stderr}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        fetched = {}
        for code, item in payload.get("fetched", {}).items():
            csv_path = Path(item.get("csv_path") or "")
            if not csv_path.is_file():
                continue
            frame = pd.read_csv(csv_path, dtype={"code": str})
            cache_path = save_miniqmt_frame(
                code,
                frame,
                period=period,
                dividend_type=dividend_type,
                merge_existing=True,
            )
            fetched[normalize_project_code(code)] = {
                "rows": int(len(frame)),
                "cache_path": str(cache_path),
                "provider_code": item.get("provider_code"),
            }
        payload["fetched"] = fetched
        payload["requested_count"] = len(normalized_codes)
        payload["bridge_python"] = str(python_executable)
        payload["period"] = period
        payload["dividend_type"] = dividend_type
        return payload
    finally:
        for path in work_dir.glob("*"):
            with suppress(OSError):
                path.unlink()
        with suppress(OSError):
            work_dir.rmdir()


def _history_bridge_script(
    *,
    qmt_root: str,
    output_dir: str,
    result_path: str,
    codes: dict[str, str],
    start_time: str,
    end_time: str,
    period: str,
    dividend_type: str,
) -> str:
    payload = {
        "qmt_root": qmt_root,
        "output_dir": output_dir,
        "result_path": result_path,
        "codes": codes,
        "start_time": start_time,
        "end_time": end_time,
        "period": period,
        "dividend_type": dividend_type,
    }
    return (
        "# coding: utf-8\n"
        "CONFIG = "
        + repr(payload)
        + "\n"
        + textwrap.dedent(
            r'''
            import json
            import os
            import sys
            import time
            import traceback

            import pandas as pd


            def date_from_any(value):
                if value is None:
                    return None
                text = str(value)
                try:
                    number = float(value)
                except Exception:
                    number = None
                if number is not None:
                    if number > 100000000000:
                        return pd.to_datetime(int(number), unit="ms", errors="coerce")
                    if 19000101 <= number <= 29991231:
                        return pd.to_datetime(str(int(number)), format="%Y%m%d", errors="coerce")
                return pd.to_datetime(text, errors="coerce")


            def frame_to_rows(project_code, provider_code, frame):
                if frame is None or len(frame) == 0:
                    return pd.DataFrame()
                data = frame.copy().reset_index()
                date_source = None
                for column in ("date", "index", "timetag", "datetime", "time"):
                    if column in data.columns:
                        parsed = data[column].map(date_from_any)
                        if parsed.notna().any():
                            date_source = parsed
                            break
                if date_source is None:
                    return pd.DataFrame()
                data["date"] = pd.to_datetime(date_source, errors="coerce").dt.strftime("%Y-%m-%d")
                data["code"] = project_code
                for column in ("open", "high", "low", "close", "volume", "amount"):
                    if column not in data.columns:
                        data[column] = None
                if "volume" in data.columns:
                    # MiniQMT stock K-line volume is in board lots.  The rest
                    # of this project stores A-share daily volume in shares.
                    data["volume"] = pd.to_numeric(data["volume"], errors="coerce") * 100
                data["tradestatus"] = None
                keep = ["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus"]
                return data[keep].dropna(subset=["date", "high", "low", "close"])


            def main():
                qmt_root = CONFIG["qmt_root"]
                bin_dir = os.path.join(qmt_root, "bin.x64")
                sys.path.insert(0, os.path.join(bin_dir, "Lib", "site-packages"))
                os.chdir(bin_dir)
                from xtquant import xtdata

                result = {"ok": True, "fetched": {}, "errors": []}
                fields = ["time", "open", "high", "low", "close", "volume", "amount"]
                for project_code, provider_code in CONFIG["codes"].items():
                    try:
                        xtdata.download_history_data(
                            provider_code,
                            CONFIG["period"],
                            CONFIG["start_time"],
                            CONFIG["end_time"],
                        )
                        time.sleep(0.1)
                        data = xtdata.get_market_data_ex(
                            field_list=fields,
                            stock_list=[provider_code],
                            period=CONFIG["period"],
                            start_time=CONFIG["start_time"],
                            end_time=CONFIG["end_time"],
                            count=-1,
                            dividend_type=CONFIG["dividend_type"],
                            fill_data=False,
                        )
                        frame = data.get(provider_code) if isinstance(data, dict) else None
                        rows = frame_to_rows(project_code, provider_code, frame)
                        csv_path = os.path.join(CONFIG["output_dir"], project_code.replace(".", "_") + ".csv")
                        rows.to_csv(csv_path, index=False, encoding="utf-8")
                        result["fetched"][project_code] = {
                            "provider_code": provider_code,
                            "rows": int(len(rows)),
                            "csv_path": csv_path,
                        }
                    except Exception:
                        result["ok"] = False
                        result["errors"].append({
                            "code": project_code,
                            "provider_code": provider_code,
                            "traceback": traceback.format_exc(),
                        })
                return result


            try:
                payload = main()
            except Exception:
                payload = {"ok": False, "fetched": {}, "errors": [{"traceback": traceback.format_exc()}]}
            with open(CONFIG["result_path"], "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            '''
        )
    )


def load_miniqmt_price_frames(
    codes,
    *,
    start_date: str,
    end_date: str,
    period: str = "1d",
    dividend_type: str = "front",
    refresh: bool = False,
    persist: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    normalized_codes = sorted({normalize_project_code(code) for code in codes if str(code).strip()})
    if refresh:
        fetch_summary = fetch_miniqmt_bars_via_qmt_python(
            normalized_codes,
            start_date,
            end_date,
            period=period,
            dividend_type=dividend_type,
        )
    else:
        fetch_summary = {"ok": True, "fetched": {}, "errors": [], "requested_count": len(normalized_codes)}

    frames = {}
    missing = []
    requested_end = pd.Timestamp(end_date) if end_date else None
    for code in normalized_codes:
        frame = load_cached_miniqmt_frame(
            code,
            period=period,
            dividend_type=dividend_type,
            start_date=start_date,
            end_date=end_date,
        )
        dates = pd.to_datetime(frame.get("date"), errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
        coverage_incomplete = bool(
            frame.empty
            or (
                requested_end is not None
                and (dates.dropna().empty or dates.max().normalize() < requested_end.normalize())
            )
        )
        if coverage_incomplete:
            missing.append(code)
        else:
            frames[code] = frame

    if persist and period == "1d" and PATHS.database.is_file() and frames:
        repository = KlineRepository(Database(PATHS.database))
        for code, frame in frames.items():
            repository.replace_stock_kline_range(
                MINIQMT_SOURCE,
                code,
                frame,
                start_date=start_date,
                end_date=end_date,
                adjustment="qfq" if dividend_type == "front" else dividend_type,
                qfq_anchor_date=end_date,
                cache_version=MINIQMT_KLINE_CACHE_VERSION,
            )

    summary = {
        "source": MINIQMT_SOURCE,
        "period": period,
        "dividend_type": dividend_type,
        "requested_count": len(normalized_codes),
        "loaded_count": len(frames),
        "missing_count": len(missing),
        "missing_sample": missing[:10],
        "refresh": bool(refresh),
        "fetch": fetch_summary,
    }
    return frames, summary


def compare_price_frames(
    left_frames: dict[str, pd.DataFrame],
    right_frames: dict[str, pd.DataFrame],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    rows = []
    compared_cells = 0
    mismatch_cells = 0
    for code in sorted(set(left_frames) & set(right_frames)):
        left = left_frames[code].copy()
        right = right_frames[code].copy()
        if left.empty or right.empty:
            continue
        left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = left.merge(right, on="date", suffixes=("_left", "_right"))
        item = {
            "code": code,
            "overlap_rows": int(len(merged)),
            "max_abs_diff": {},
            "mismatch_count": 0,
        }
        for column in ("open", "high", "low", "close", "volume", "amount"):
            left_col = f"{column}_left"
            right_col = f"{column}_right"
            if left_col not in merged or right_col not in merged:
                continue
            diff = (
                pd.to_numeric(merged[left_col], errors="coerce")
                - pd.to_numeric(merged[right_col], errors="coerce")
            ).abs()
            valid = diff.dropna()
            if valid.empty:
                continue
            count = int((valid > tolerance).sum())
            item["max_abs_diff"][column] = float(valid.max())
            item["mismatch_count"] += count
            compared_cells += int(valid.count())
            mismatch_cells += count
        rows.append(item)
    rows.sort(key=lambda item: item["mismatch_count"], reverse=True)
    return {
        "compared_codes": len(rows),
        "compared_cells": compared_cells,
        "mismatch_cells": mismatch_cells,
        "mismatch_ratio": mismatch_cells / compared_cells if compared_cells else 0.0,
        "sample": rows[:20],
    }
