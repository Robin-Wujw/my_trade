# -*- coding: utf-8 -*-
"""
Sector volume and limit-up watch.

This pipeline is intentionally separate from factor selection. It first asks
"which sectors are becoming the main line?", then the stock selector can look
inside those sectors. The scoring favors repeated strength, amount expansion,
weak-market resilience, and limit-up participation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import random
import time
from datetime import datetime, timedelta

from stock_research.api import akshare as ak
import numpy as np
import pandas as pd

from stock_research.core.as_of import write_metadata
from stock_research.core.paths import PATHS


OUTPUT_DIR = str(PATHS.market_exports)
CACHE_DIR = str(PATHS.cache / "sector_watch")
BOARD_CONSTITUENT_FILE = str(PATHS.cache / "sector_mainline_constituents.csv")
BENCHMARK_SYMBOL = "000001"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="板块量能与涨停观察")
    parser.add_argument("--days", type=int, default=80, help="板块历史天数")
    parser.add_argument("--top", type=int, default=30, help="展示前N个板块")
    parser.add_argument("--sleep", type=float, default=0.03, help="逐板块请求间隔")
    parser.add_argument("--retries", type=int, default=4, help="东方财富接口失败重试次数")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="接口失败后的退避基准秒数")
    parser.add_argument("--limit-up-days", type=int, default=5, help="统计近N日涨停板块归属")
    parser.add_argument("--fallback-sample", action="store_true", help="真实板块接口失败时自动生成离线样例，避免每日流程中断")
    parser.add_argument("--workers", type=int, default=4, help="板块历史请求并发数")
    parser.add_argument("--as-of-date", default="", help="历史截止日YYYY-MM-DD；默认今天")
    return parser.parse_args(argv)


def call_with_backoff(func, label, retries=4, retry_delay=2.0):
    last_exc = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait = retry_delay * attempt + random.uniform(0, retry_delay)
            print(f"{label} 请求失败: {exc} | 第 {attempt}/{retries} 次，{wait:.1f}s 后重试")
            time.sleep(wait)
    raise last_exc


def cache_path(kind, key):
    safe = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, kind, f"{safe}.csv")


def read_cache(kind, key, date_cols=None):
    path = cache_path(kind, key)
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in date_cols or []:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
    except Exception as exc:
        print(f"读取缓存失败 {kind}/{key}: {exc}")
    return pd.DataFrame()


def write_cache(kind, key, df):
    if df is None or df.empty:
        return
    path = cache_path(kind, key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"写入缓存失败 {kind}/{key}: {exc}")


def score_direct(value, low, high):
    if value is None or pd.isna(value) or high == low:
        return 0.0
    return max(0.0, min(100.0, (float(value) - low) / (high - low) * 100.0))


def pct_change(series, days):
    if len(series) <= days:
        return np.nan
    base = series.iloc[-days - 1]
    return series.iloc[-1] / base - 1 if base else np.nan


def normalize_board_name(name):
    return str(name).replace("行业板块", "").strip()


def load_board_names(retries=4, retry_delay=2.0):
    try:
        df = call_with_backoff(
            ak.stock_board_industry_name_em,
            "行业板块列表",
            retries=retries,
            retry_delay=retry_delay,
        )
    except Exception as exc:
        cached = read_cache("board_names", "industry")
        if not cached.empty:
            print(f"行业板块列表读取失败，使用缓存: {exc}")
            return cached
        raise
    if df is None or df.empty:
        return pd.DataFrame()
    name_col = "板块名称" if "板块名称" in df.columns else "名称"
    df = df.rename(columns={name_col: "board_name"})
    df["board_name"] = df["board_name"].map(normalize_board_name)
    df = df.dropna(subset=["board_name"])
    write_cache("board_names", "industry", df)
    return df


def load_board_history(board_name, days, as_of_date=None, retries=4, retry_delay=2.0):
    end_dt = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
    end_date = end_dt.strftime("%Y%m%d")
    start_date = (end_dt - pd.Timedelta(days=max(days * 3, 180))).strftime("%Y%m%d")
    cached = read_cache("board_history", board_name, date_cols=["日期"])
    if not cached.empty:
        sliced = cached[cached["日期"] <= end_dt].sort_values("日期").tail(days)
        cache_reaches_cutoff = cached["日期"].max() >= end_dt - pd.Timedelta(days=7)
        if len(sliced) >= min(days, 60) and cache_reaches_cutoff:
            return sliced.reset_index(drop=True)
    try:
        df = call_with_backoff(
            lambda: ak.stock_board_industry_hist_em(
                symbol=board_name,
                start_date=start_date,
                end_date=end_date,
                period="日k",
                adjust="",
            ),
            f"{board_name} 板块K线",
            retries=retries,
            retry_delay=retry_delay,
        )
    except Exception as exc:
        cached = read_cache("board_history", board_name, date_cols=["日期"])
        if not cached.empty:
            print(f"{board_name} K线读取失败，使用缓存: {exc}")
            return cached.tail(days).reset_index(drop=True)
        raise
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for col in ["开盘", "收盘", "最高", "最低", "成交额", "成交量", "涨跌幅"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期").reset_index(drop=True)
    write_cache("board_history", board_name, df)
    return df.tail(days).reset_index(drop=True)


def load_benchmark(days, as_of_date=None):
    df = ak.stock_zh_index_daily_em(symbol=BENCHMARK_SYMBOL)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if as_of_date:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]
    return df.tail(days).reset_index(drop=True)


def calc_board_metrics(board_name, hist, benchmark):
    close = hist["收盘"]
    amount = hist["成交额"] if "成交额" in hist.columns else pd.Series(dtype=float)
    ret1 = pct_change(close, 1)
    ret3 = pct_change(close, 3)
    ret5 = pct_change(close, 5)
    ret20 = pct_change(close, 20)
    amount5 = amount.tail(5).mean() if len(amount) >= 5 else np.nan
    amount20 = amount.tail(20).mean() if len(amount) >= 20 else np.nan
    amount60 = amount.tail(60).mean() if len(amount) >= 60 else np.nan
    amount_ratio_5_20 = amount5 / amount20 if amount20 and amount20 > 0 else np.nan
    amount_ratio_20_60 = amount20 / amount60 if amount60 and amount60 > 0 else np.nan

    resilience = np.nan
    attack = np.nan
    if benchmark is not None and not benchmark.empty:
        merged = hist[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "board_close"}).merge(
            benchmark[["date", "close"]].rename(columns={"close": "bench_close"}),
            on="date",
            how="inner",
        )
        if len(merged) >= 10:
            board_pct = merged["board_close"].pct_change()
            bench_pct = merged["bench_close"].pct_change()
            down = bench_pct < 0
            up = bench_pct > 0
            if down.sum() >= 3:
                resilience = float((board_pct[down] > bench_pct[down]).mean())
            if up.sum() >= 3:
                attack = float((board_pct[up] > bench_pct[up]).mean())

    score = (
        score_direct(ret5, -0.03, 0.10) * 0.20
        + score_direct(ret20, -0.05, 0.25) * 0.20
        + score_direct(amount_ratio_5_20, 0.80, 1.80) * 0.20
        + score_direct(amount_ratio_20_60, 0.85, 1.60) * 0.15
        + score_direct(resilience, 0.45, 0.75) * 0.15
        + score_direct(attack, 0.45, 0.75) * 0.10
    )
    return {
        "board": board_name,
        "date": hist.iloc[-1]["日期"].strftime("%Y-%m-%d"),
        "close": close.iloc[-1],
        "ret1": ret1,
        "ret3": ret3,
        "ret5": ret5,
        "ret20": ret20,
        "amount_5_20": amount_ratio_5_20,
        "amount_20_60": amount_ratio_20_60,
        "weak_resilience": resilience,
        "strong_attack": attack,
        "mainline_score": round(score, 1),
    }


def load_limit_up_counts(days, as_of_date=None):
    counts = {}
    try:
        dates = pd.bdate_range(end=pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today(), periods=days * 2)
        for day in reversed(dates):
            date_str = day.strftime("%Y%m%d")
            try:
                zt = ak.stock_zt_pool_em(date=date_str)
            except Exception:
                continue
            if zt is None or zt.empty:
                continue
            for _, row in zt.iterrows():
                board = row.get("所属行业") or row.get("行业")
                if not board or pd.isna(board):
                    continue
                board = normalize_board_name(board)
                counts[board] = counts.get(board, 0) + 1
            days -= 1
            if days <= 0:
                break
    except Exception:
        return {}
    return counts


def to_market_code(value):
    code = str(value).strip().zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    return code


def save_mainline_constituents(board_df, top=10, retries=4, retry_delay=2.0, sleep=0.1, output_path=None):
    rows = []
    for rank, (_, board_row) in enumerate(board_df.head(top).iterrows(), start=1):
        board = str(board_row["board"])
        try:
            members = call_with_backoff(
                lambda b=board: ak.stock_board_industry_cons_em(symbol=b),
                f"{board} 成分股",
                retries=retries,
                retry_delay=retry_delay,
            )
            if members is None or members.empty or "代码" not in members.columns:
                continue
            for _, member in members.iterrows():
                rows.append({
                    "code": to_market_code(member.get("代码")),
                    "name": member.get("名称", ""),
                    "board": board,
                    "board_rank": rank,
                    "board_score": board_row.get("final_score"),
                    "board_date": board_row.get("date"),
                })
        except Exception as exc:
            print(f"跳过 {board} 成分股: {exc}")
        if sleep > 0:
            time.sleep(sleep)
    if rows:
        target = output_path or BOARD_CONSTITUENT_FILE
        os.makedirs(os.path.dirname(target), exist_ok=True)
        pd.DataFrame(rows).drop_duplicates(["code", "board"]).to_csv(
            target, index=False, encoding="utf-8-sig"
        )
        write_metadata(target, {
            "kind": "sector_mainline_constituents",
            "point_in_time_status": "unsafe" if output_path else "safe",
            "point_in_time_note": (
                "board ranks are historical, but constituent membership was retrieved from the current API"
                if output_path else "current-date board ranks and constituents"
            ),
        })
        print(f"主流板块成分映射已保存: {target}，{len(rows)} 条")


def print_sector_report(df, path, top):
    display = df.head(top).copy()
    cols = [
        "board", "final_score", "mainline_score", "limit_up_count", "ret1", "ret3",
        "ret5", "ret20", "amount_5_20", "amount_20_60", "weak_resilience", "strong_attack",
    ]
    print("\n================ 板块主线观察 ================")
    print(f"结果文件: {path}")
    print("阅读顺序: 总分 -> 主线分 -> 涨停扩散 -> 短中期涨幅 -> 量能 -> 弱市韧性/强市进攻")
    print(display[cols].to_string(index=False, formatters={
        "ret1": lambda v: f"{v:.1%}" if pd.notna(v) else "-",
        "ret3": lambda v: f"{v:.1%}" if pd.notna(v) else "-",
        "ret5": lambda v: f"{v:.1%}" if pd.notna(v) else "-",
        "ret20": lambda v: f"{v:.1%}" if pd.notna(v) else "-",
        "amount_5_20": lambda v: f"{v:.2f}" if pd.notna(v) else "-",
        "amount_20_60": lambda v: f"{v:.2f}" if pd.notna(v) else "-",
        "weak_resilience": lambda v: f"{v:.0%}" if pd.notna(v) else "-",
        "strong_attack": lambda v: f"{v:.0%}" if pd.notna(v) else "-",
    }))
    print("\n复盘提示:")
    print("1. final_score 高且 amount_5_20 > 1，说明短期资金正在放大。")
    print("2. weak_resilience 高，说明大盘弱时更抗跌；strong_attack 高，说明大盘强时更有进攻性。")
    print("3. limit_up_count 连续扩散时，优先回看该板块内的右侧候选和首板/连板结构。")


def sample_watch_rows():
    boards = ["半导体", "通信设备", "电池", "能源金属", "消费电子", "软件服务"]
    rows = []
    for idx, board in enumerate(boards):
        rows.append({
            "board": board,
            "final_score": round(86 - idx * 4.5, 1),
            "mainline_score": round(78 - idx * 3.8, 1),
            "limit_up_count": max(0, 5 - idx),
            "ret1": 0.018 - idx * 0.002,
            "ret3": 0.042 - idx * 0.004,
            "ret5": 0.067 - idx * 0.006,
            "ret20": 0.16 - idx * 0.012,
            "amount_5_20": 1.42 - idx * 0.05,
            "amount_20_60": 1.18 - idx * 0.03,
            "weak_resilience": 0.72 - idx * 0.04,
            "strong_attack": 0.68 - idx * 0.03,
            "data_status": "fallback_sample",
        })
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    as_of_date = args.as_of_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        boards = load_board_names(retries=args.retries, retry_delay=args.retry_delay)
    except Exception as exc:
        if args.fallback_sample:
            print(f"真实板块接口失败，改用 --fallback-sample 样例数据: {exc}")
            df = sample_watch_rows()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"sector_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print_sector_report(df, path, args.top)
            return
        raise
    if boards.empty:
        if args.fallback_sample:
            print("真实板块列表为空，改用 --fallback-sample 样例数据")
            df = sample_watch_rows()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"sector_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print_sector_report(df, path, args.top)
            return
        raise SystemExit("无法获取行业板块列表")
    benchmark = load_benchmark(args.days + 5, as_of_date)
    limit_up_counts = load_limit_up_counts(args.limit_up_days, as_of_date)

    board_names = boards["board_name"].tolist()

    def process_board(board_name):
        try:
            hist = load_board_history(
                board_name,
                args.days,
                as_of_date,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            if len(hist) >= 25:
                row = calc_board_metrics(board_name, hist, benchmark)
                row["limit_up_count"] = limit_up_counts.get(board_name, 0)
                row["final_score"] = round(row["mainline_score"] + score_direct(row["limit_up_count"], 0, 8) * 0.15, 1)
                return row
        except Exception as exc:
            print(f"跳过 {board_name}: {exc}")
        return None

    rows = []
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for board_name in board_names:
            if args.sleep > 0:
                time.sleep(args.sleep)
            futures[executor.submit(process_board, board_name)] = board_name
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row:
                rows.append(row)
            if idx % 20 == 0:
                print(f"进度 {idx}/{len(board_names)}, 有效 {len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        if args.fallback_sample:
            print("真实板块历史为空，改用 --fallback-sample 样例数据")
            df = sample_watch_rows()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"sector_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print_sector_report(df, path, args.top)
            return
        raise SystemExit("无有效板块数据")
    df = df.sort_values(["final_score", "mainline_score", "ret5"], ascending=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    as_of_stamp = pd.Timestamp(as_of_date).strftime("%Y%m%d")
    path = os.path.join(OUTPUT_DIR, f"sector_watch_asof_{as_of_stamp}_{datetime.now().strftime('%H%M%S')}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    constituent_path = str(
        PATHS.cache / f"sector_mainline_constituents_{as_of_stamp}.csv"
    ) if args.as_of_date else BOARD_CONSTITUENT_FILE
    save_mainline_constituents(
        df,
        top=min(10, args.top),
        retries=args.retries,
        retry_delay=args.retry_delay,
        sleep=args.sleep,
        output_path=constituent_path,
    )
    print_sector_report(df, path, args.top)


if __name__ == "__main__":
    main()
