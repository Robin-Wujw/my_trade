# -*- coding: utf-8 -*-
"""Incrementally maintain full-market financial cache and build a daily snapshot."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import akshare as ak
import baostock as bs

from dailyFundamentalSelect import KLINE_CACHE_DIR, VALUE_CACHE_DIR
from factorStock import classify_method, get_value_line_metrics_from_akshare_indicator, infer_theme, score_direct
from point_in_time import write_metadata
from trade_utils import get_project_path, send_pushplus


UNIVERSE_PATH = get_project_path(".cache/stock_universe.csv")
SNAPSHOT_DIR = get_project_path(".cache/fundamental_snapshots")
STATE_DIR = get_project_path(".cache/update_state")
INDUSTRY_DIR = get_project_path(".cache/reference")


def parse_args():
    parser = argparse.ArgumentParser(description="全市场财务缓存增量更新与动态基础截面")
    parser.add_argument("--report-period", default="", help="默认使用财务缓存中的最新报告期")
    parser.add_argument("--as-of-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--max-updates", type=int, default=100, help="本次最多补抓多少只；0表示不限制")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="只使用缓存并生成覆盖率/截面")
    parser.add_argument("--min-price-coverage", type=float, default=0.90)
    parser.add_argument("--min-financial-coverage", type=float, default=0.35)
    parser.add_argument("--target-financial-coverage", type=float, default=0.95)
    parser.add_argument("--output", default="")
    parser.add_argument("--alert", action="store_true", help="覆盖率不足时发送PushPlus告警")
    return parser.parse_args()


def latest_visible_report_period(as_of_date):
    date = pd.Timestamp(as_of_date)
    year = date.year
    if date >= pd.Timestamp(year=year, month=10, day=31):
        return f"{year}-09-30"
    if date >= pd.Timestamp(year=year, month=8, day=31):
        return f"{year}-06-30"
    if date >= pd.Timestamp(year=year, month=4, day=30):
        return f"{year}-03-31"
    return f"{year - 1}-09-30"


def financial_path(code, report_period):
    symbol = str(code).split(".")[-1]
    return os.path.join(VALUE_CACHE_DIR, f"{symbol}_{report_period.replace('-', '')}.json")


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return {} if default is None else default


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    os.replace(tmp, path)


def latest_market(code, as_of_date):
    path = os.path.join(KLINE_CACHE_DIR, f"{str(code).replace('.', '_')}.csv")
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in ["close", "volume"]:
            frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
        if frame.empty:
            return None
        tail = frame.tail(20)
        amount = tail["close"] * tail["volume"] if "volume" in tail else pd.Series(dtype=float)
        avg_amount20 = float(amount.dropna().mean()) if not amount.dropna().empty else None
        return {
            "market_date": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "close": float(frame.iloc[-1]["close"]),
            "liquidity_score": score_direct(
                math.log10(avg_amount20) if avg_amount20 and avg_amount20 > 0 else None,
                7.0,
                9.5,
            ),
            "avg_amount20": avg_amount20,
        }
    except Exception:
        return None


def enrich_cache_metadata(path, report_period):
    data = load_json(path)
    if not data:
        return
    changed = False
    defaults = {
        "report_period": report_period,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_history_available": False,
    }
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
            changed = True
    if changed:
        save_json(path, data)


def fetch_one(code, close, report_period, retries):
    symbol = str(code).split(".")[-1]
    error = None
    for attempt in range(retries):
        try:
            value = get_value_line_metrics_from_akshare_indicator(symbol, close, report_period)
            if value:
                path = financial_path(code, report_period)
                save_json(path, value)
                enrich_cache_metadata(path, report_period)
                return code, True, ""
        except Exception as exc:
            error = str(exc)
        if attempt + 1 < retries:
            time.sleep(1.5 * (attempt + 1))
    return code, False, error or "empty financial response"


def update_missing(universe, markets, args):
    state_path = os.path.join(STATE_DIR, f"financial_{args.report_period.replace('-', '')}.json")
    state = load_json(state_path, {"completed": [], "errors": {}})
    completed = set(state.get("completed", []))
    missing = [
        code for code in universe["code"].astype(str)
        if not os.path.exists(financial_path(code, args.report_period)) and code in markets
    ]
    missing.sort(key=lambda code: (code in state.get("errors", {}), code))
    if args.max_updates > 0:
        missing = missing[:args.max_updates]
    if args.offline or not missing:
        return 0, 0, len(missing)
    success = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {
            executor.submit(fetch_one, code, markets[code]["close"], args.report_period, args.retries): code
            for code in missing
        }
        for index, future in enumerate(as_completed(jobs), 1):
            code, ok, error = future.result()
            if ok:
                success += 1
                completed.add(code)
                state.setdefault("errors", {}).pop(code, None)
            else:
                failed += 1
                state.setdefault("errors", {})[code] = error
            state["completed"] = sorted(completed)
            state["updated_at"] = datetime.now().isoformat()
            if index % 10 == 0 or index == len(jobs):
                save_json(state_path, state)
                print(f"financial update {index}/{len(jobs)} success={success} failed={failed}")
    return success, failed, len(missing)


def normalize_code(symbol):
    symbol = str(symbol).zfill(6)
    return ("sh." if symbol.startswith(("6", "9")) else "sz.") + symbol


def filter_supported_universe(frame):
    result = frame.copy()
    result = result[result["code"].astype(str).str.startswith(("sh.60", "sh.68", "sz.00", "sz.30"))]
    return result.drop_duplicates("code").reset_index(drop=True)


def refresh_universe(offline=False):
    if not offline:
        try:
            frame = ak.stock_zh_a_spot_em()
            code_col = "代码" if "代码" in frame else "code"
            name_col = "名称" if "名称" in frame else "name"
            refreshed = pd.DataFrame({
                "code": frame[code_col].astype(str).map(normalize_code),
                "code_name": frame[name_col].astype(str),
            }).drop_duplicates("code")
            refreshed = filter_supported_universe(refreshed)
            if len(refreshed) >= 4000:
                refreshed.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")
                write_metadata(UNIVERSE_PATH, {
                    "kind": "stock_universe",
                    "point_in_time_status": "safe",
                    "data_source": "akshare/stock_zh_a_spot_em",
                })
                return refreshed, "akshare/stock_zh_a_spot_em"
        except Exception as exc:
            print(f"AkShare universe refresh failed, using cache: {exc}")
    cached = pd.read_csv(UNIVERSE_PATH, dtype={"code": str}).drop_duplicates("code")
    cached = filter_supported_universe(cached)
    cached.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")
    return cached, "local_cache"


def fetch_akshare_industry_map(workers=2):
    boards = ak.stock_board_industry_name_em()
    board_col = "板块名称" if "板块名称" in boards else "名称"
    names = boards[board_col].dropna().astype(str).drop_duplicates().tolist()
    results = {}

    def fetch(board):
        members = ak.stock_board_industry_cons_em(symbol=board)
        code_col = "代码" if "代码" in members else "code"
        return board, [normalize_code(code) for code in members[code_col].astype(str)]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        jobs = {executor.submit(fetch, board): board for board in names}
        for index, future in enumerate(as_completed(jobs), 1):
            try:
                board, codes = future.result()
                results[board] = codes
            except Exception as exc:
                print(f"AkShare industry board failed: {jobs[future]} | {exc}")
            if index % 20 == 0:
                mapped = len({code for codes in results.values() for code in codes})
                print(f"AkShare industry progress {index}/{len(jobs)} mapped={mapped}")
    mapping = {}
    for board in names:
        for code in results.get(board, []):
            mapping.setdefault(code, board)
    return mapping


def load_industry_map(as_of_date, offline=False):
    latest_path = os.path.join(INDUSTRY_DIR, "industry_map_latest.csv")
    if os.path.exists(latest_path) and not offline:
        age_days = (time.time() - os.path.getmtime(latest_path)) / 86400
        if age_days < 7:
            frame = pd.read_csv(latest_path, dtype={"code": str})
            return dict(zip(frame["code"], frame["industry"].fillna(""))), "akshare_cache"
    if not offline:
        try:
            mapping = fetch_akshare_industry_map(workers=2)
            if len(mapping) >= 3000:
                frame = pd.DataFrame(mapping.items(), columns=["code", "industry"])
                os.makedirs(INDUSTRY_DIR, exist_ok=True)
                dated = os.path.join(INDUSTRY_DIR, f"industry_map_{as_of_date.replace('-', '')}.csv")
                frame.to_csv(dated, index=False, encoding="utf-8-sig")
                frame.to_csv(latest_path, index=False, encoding="utf-8-sig")
                write_metadata(dated, {
                    "kind": "industry_map",
                    "point_in_time_status": "safe",
                    "as_of_date": as_of_date,
                    "data_source": "akshare/eastmoney_industry_boards",
                })
                return mapping, "akshare/eastmoney_industry_boards"
            print(f"AkShare industry coverage too low ({len(mapping)}), trying Baostock fallback")
        except Exception as exc:
            print(f"AkShare industry refresh failed, trying Baostock fallback: {exc}")
        try:
            login = bs.login()
            if login.error_code == "0":
                try:
                    result = bs.query_stock_industry()
                    frame = result.get_data()
                    if result.error_code == "0" and not frame.empty:
                        frame.columns = result.fields
                        frame = frame[["code", "industry"]].drop_duplicates("code")
                        os.makedirs(INDUSTRY_DIR, exist_ok=True)
                        dated = os.path.join(INDUSTRY_DIR, f"industry_map_{as_of_date.replace('-', '')}.csv")
                        frame.to_csv(dated, index=False, encoding="utf-8-sig")
                        frame.to_csv(latest_path, index=False, encoding="utf-8-sig")
                        write_metadata(dated, {
                            "kind": "industry_map",
                            "point_in_time_status": "safe",
                            "as_of_date": as_of_date,
                            "data_source": "baostock/query_stock_industry",
                        })
                        return dict(zip(frame["code"], frame["industry"].fillna(""))), "baostock/query_stock_industry"
                finally:
                    bs.logout()
        except Exception as exc:
            print(f"industry refresh failed, using cache: {exc}")
    if not os.path.exists(latest_path):
        return {}, "missing"
    frame = pd.read_csv(latest_path, dtype={"code": str})
    return dict(zip(frame["code"], frame["industry"].fillna(""))), "local_cache"


def build_snapshot(universe, markets, report_period, industry_map):
    rows = []
    for _, stock in universe.iterrows():
        code = str(stock["code"])
        market = markets.get(code)
        financial = load_json(financial_path(code, report_period))
        if not market or not financial:
            continue
        shares = pd.to_numeric(financial.get("total_share"), errors="coerce")
        value_line = pd.to_numeric(financial.get("value_line"), errors="coerce")
        mktcap = market["close"] * shares / 1e8 if pd.notna(shares) else np.nan
        industry = industry_map.get(code, "")
        rows.append({
            "code": code,
            "name": stock.get("code_name", code),
            "industry": industry,
            "theme": infer_theme(stock.get("code_name", code), industry) if industry else "",
            "method": classify_method(industry) if industry else "UNKNOWN",
            "report_period": report_period,
            "market_date": market["market_date"],
            "close": market["close"],
            "quality_score": financial.get("quality_score"),
            "earnings_yoy": financial.get("yoy"),
            "eps_excl": financial.get("eps_excl"),
            "total_share": shares,
            "mktcap": mktcap,
            "liquidity_score": market["liquidity_score"],
            "avg_amount20": market["avg_amount20"],
            "value_line": value_line,
            "price_to_value": market["close"] / value_line if pd.notna(value_line) and value_line > 0 else np.nan,
            "financial_data_source": financial.get("data_source", "unknown"),
            "financial_retrieved_at_utc": financial.get("retrieved_at_utc"),
            "revision_history_available": bool(financial.get("revision_history_available", False)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["industry_known"] = result["industry"].fillna("").astype(str).str.strip().ne("")
    result["industry_peer_count"] = result.groupby("industry")["code"].transform("count")
    result["industry_mktcap_percentile"] = result.groupby("industry")["mktcap"].rank(pct=True)
    result["industry_leader_proxy"] = (
        result["industry_known"]
        & (result["industry_peer_count"] >= 5)
        & (result["industry_mktcap_percentile"] >= 0.80)
    )
    result["value_applicability_status"] = np.where(
        (result["method"] == "VALUE") & result["industry_leader_proxy"],
        "rule_eligible",
        "not_proven",
    )
    return result


def main():
    args = parse_args()
    args.report_period = args.report_period or os.environ.get("REPORT_PERIOD") or latest_visible_report_period(args.as_of_date)
    universe, universe_source = refresh_universe(offline=args.offline)
    markets = {}
    for code in universe["code"].astype(str):
        market = latest_market(code, args.as_of_date)
        if market:
            markets[code] = market
    success, failed, attempted = update_missing(universe, markets, args)
    industry_map, industry_source = load_industry_map(args.as_of_date, offline=args.offline)
    snapshot = build_snapshot(universe, markets, args.report_period, industry_map)
    output = args.output or os.path.join(
        SNAPSHOT_DIR,
        f"fundamental_snapshot_{args.report_period.replace('-', '')}_{args.as_of_date.replace('-', '')}.csv",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    snapshot.to_csv(output, index=False, encoding="utf-8-sig")
    total = len(universe)
    price_coverage = len(markets) / total if total else 0
    financial_coverage = len(snapshot) / total if total else 0
    if price_coverage < args.min_price_coverage or financial_coverage < args.min_financial_coverage:
        status = "unsafe"
    elif financial_coverage < args.target_financial_coverage:
        status = "warning"
    else:
        status = "safe"
    metadata = {
        "kind": "full_market_fundamental_snapshot",
        "point_in_time_status": status,
        "report_period": args.report_period,
        "as_of_date": args.as_of_date,
        "universe_count": total,
        "price_count": len(markets),
        "financial_count": len(snapshot),
        "price_coverage": price_coverage,
        "financial_coverage": financial_coverage,
        "financial_updates_attempted": attempted,
        "financial_updates_success": success,
        "financial_updates_failed": failed,
        "financial_revision_history_available": False,
        "source_priority": ["akshare", "baostock", "local_cache"],
        "universe_source": universe_source,
        "industry_source": industry_source,
        "industry_known_count": int(snapshot.get("industry_known", pd.Series(dtype=bool)).sum()),
        "industry_leader_proxy_count": int(snapshot.get("industry_leader_proxy", pd.Series(dtype=bool)).sum()),
    }
    write_metadata(output, metadata)
    coverage_path = os.path.join(SNAPSHOT_DIR, "latest_coverage.json")
    save_json(coverage_path, {**metadata, "snapshot": output})
    print(
        f"snapshot={output} rows={len(snapshot)} universe={total} "
        f"price={price_coverage:.1%} financial={financial_coverage:.1%} status={status}"
    )
    if args.alert and status != "safe":
        send_pushplus(
            f"全市场数据覆盖率{status}",
            f"<p>报告期：{args.report_period}</p>"
            f"<p>全市场：{total}；行情：{len(markets)} ({price_coverage:.1%})；"
            f"财务：{len(snapshot)} ({financial_coverage:.1%})</p>"
            f"<p>本次补抓：成功{success}，失败{failed}。完整率目标：{args.target_financial_coverage:.0%}。</p>",
        )
    if status == "unsafe":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
