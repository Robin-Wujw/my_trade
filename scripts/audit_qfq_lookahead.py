"""Audit whether cached qfq prices embed future adjustment factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_research.api import tushare as ts_api


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="审计前复权价格是否含未来复权因子")
    parser.add_argument("--symbols", nargs="+", required=True, help="示例: sz.002594 sh.688408")
    parser.add_argument("--observation-dates", nargs="+", required=True, help="观察日列表")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--kline-directory",
        default="var/cache/formula33_kline/akshare",
        help="本地前复权CSV目录",
    )
    parser.add_argument(
        "--provider-json-directory",
        default="",
        help="可选：由 fetch_tushare_qfq_audit_json.ps1 抓取的 Tushare JSON 目录",
    )
    parser.add_argument(
        "--output-directory",
        default="var/cache/qfq_lookahead_audit",
        help="审计输出目录",
    )
    return parser.parse_args(argv)


def _to_tushare_code(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        market, code = text.split(".", 1)
        return f"{code.zfill(6)}.{market.upper()}"
    code = text.zfill(6)
    market = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{market}"


def _cache_path(directory: Path, symbol: str) -> Path:
    text = str(symbol).strip()
    if "." in text:
        market, code = text.split(".", 1)
    else:
        code = text.zfill(6)
        market = "sh" if code.startswith(("6", "9")) else "sz"
    return directory / f"{market.lower()}_{code.zfill(6)}.csv"


def _query_tushare(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "ts_code": _to_tushare_code(symbol),
        "start_date": start_date.replace("-", ""),
        "end_date": end_date.replace("-", ""),
    }
    daily = ts_api.query(
        "daily",
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
        **params,
    )
    factors = ts_api.query(
        "adj_factor",
        fields="ts_code,trade_date,adj_factor",
        **params,
    )
    if daily.empty or factors.empty:
        return pd.DataFrame()
    merged = daily.merge(factors, on=["ts_code", "trade_date"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=["date", "close", "adj_factor"]).sort_values("date")


def _payload_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    return pd.DataFrame(items, columns=fields)


def _query_tushare_json(symbol: str, directory: Path) -> pd.DataFrame:
    code = str(symbol).split(".")[-1].zfill(6)
    daily_path = directory / f"{code}_daily.json"
    factor_path = directory / f"{code}_adj_factor.json"
    if not daily_path.exists() or not factor_path.exists():
        return pd.DataFrame()
    daily = _payload_frame(daily_path)
    factors = _payload_frame(factor_path)
    if daily.empty or factors.empty:
        return pd.DataFrame()
    merged = daily.merge(factors, on=["ts_code", "trade_date"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor"):
        if column not in merged:
            merged[column] = pd.NA
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=["date", "close", "adj_factor"]).sort_values("date")


def _load_cached_close(directory: Path, symbol: str) -> pd.DataFrame:
    path = _cache_path(directory, symbol)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["cached_close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    return frame[["date", "cached_close"]].dropna().sort_values("date")


def audit_symbol(
    symbol: str,
    observation_dates: list[pd.Timestamp],
    start_date: str,
    end_date: str,
    kline_dir: Path,
    provider_json_dir: Path | None = None,
) -> list[dict]:
    tushare = (
        _query_tushare_json(symbol, provider_json_dir)
        if provider_json_dir is not None else pd.DataFrame()
    )
    if tushare.empty:
        tushare = _query_tushare(symbol, start_date, end_date)
    cached = _load_cached_close(kline_dir, symbol)
    if tushare.empty:
        return [{"symbol": symbol, "error": "tushare_empty"}]
    if cached.empty:
        return [{"symbol": symbol, "error": "cached_kline_empty"}]
    merged = tushare.merge(cached, on="date", how="left")
    global_anchor = merged.loc[merged["date"] <= pd.Timestamp(end_date), "adj_factor"].dropna()
    if global_anchor.empty:
        return [{"symbol": symbol, "error": "global_anchor_missing"}]
    global_factor = float(global_anchor.iloc[-1])
    merged["tushare_qfq_global_close"] = merged["close"] * merged["adj_factor"] / global_factor
    rows = []
    for observation_date in observation_dates:
        visible = merged[merged["date"] <= observation_date].copy()
        if visible.empty:
            rows.append({"symbol": symbol, "observation_date": observation_date.strftime("%Y-%m-%d"), "error": "no_visible_rows"})
            continue
        asof_factor = float(visible["adj_factor"].iloc[-1])
        visible["tushare_qfq_asof_close"] = visible["close"] * visible["adj_factor"] / asof_factor
        latest = visible.iloc[-1]
        cache_diff = None
        if pd.notna(latest.get("cached_close")) and latest["tushare_qfq_global_close"]:
            cache_diff = latest["cached_close"] / latest["tushare_qfq_global_close"] - 1
        future_factor_ratio = asof_factor / global_factor - 1
        close_rewrite = latest["tushare_qfq_global_close"] / latest["tushare_qfq_asof_close"] - 1
        factor_changed_after_observation = abs(future_factor_ratio) > 1e-8
        rows.append({
            "symbol": symbol,
            "observation_date": observation_date.strftime("%Y-%m-%d"),
            "asof_factor": asof_factor,
            "global_end_factor": global_factor,
            "future_factor_ratio_pct": round(future_factor_ratio * 100, 6),
            "latest_raw_close": round(float(latest["close"]), 6),
            "qfq_close_asof": round(float(latest["tushare_qfq_asof_close"]), 6),
            "qfq_close_global": round(float(latest["tushare_qfq_global_close"]), 6),
            "global_vs_asof_close_diff_pct": round(float(close_rewrite) * 100, 6),
            "cached_close": None if pd.isna(latest.get("cached_close")) else round(float(latest["cached_close"]), 6),
            "cached_vs_tushare_global_diff_pct": None if cache_diff is None else round(float(cache_diff) * 100, 6),
            "factor_changed_after_observation": bool(factor_changed_after_observation),
            "error": "",
        })
    return rows


def main(argv=None):
    args = parse_args(argv)
    observation_dates = [pd.Timestamp(item).normalize() for item in args.observation_dates]
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in args.symbols:
        rows.extend(
            audit_symbol(
                symbol,
                observation_dates,
                args.start_date,
                args.end_date,
                Path(args.kline_directory),
                Path(args.provider_json_directory) if args.provider_json_directory else None,
            )
        )
    detail = pd.DataFrame(rows)
    detail_path = output_dir / "qfq_lookahead_detail.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = {
        "symbols": args.symbols,
        "observation_dates": [item.strftime("%Y-%m-%d") for item in observation_dates],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": int(len(detail)),
        "factor_changed_rows": int(detail.get("factor_changed_after_observation", pd.Series(dtype=bool)).fillna(False).sum()),
        "max_abs_global_vs_asof_close_diff_pct": (
            None if detail.empty or "global_vs_asof_close_diff_pct" not in detail
            else round(float(pd.to_numeric(detail["global_vs_asof_close_diff_pct"], errors="coerce").abs().max()), 6)
        ),
        "detail": str(detail_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细: {detail_path}")


if __name__ == "__main__":
    main()
