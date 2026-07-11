# -*- coding: utf-8 -*-
"""
Sector volume and limit-up watch.

This pipeline is intentionally separate from factor selection. It first asks
"which sectors are becoming the main line?", then the stock selector can look
inside those sectors. The scoring favors repeated strength, amount expansion,
weak-market resilience, and limit-up participation.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import os
import time
from datetime import datetime

from stock_research.api import akshare as ak
from stock_research.api import ths
from stock_research.api.retry import call_with_backoff
import numpy as np
import pandas as pd

from stock_research.core.as_of import write_metadata
from stock_research.core.paths import PATHS
from stock_research.core.part_logger import PartLogger
from stock_research.market.sectors import normalize_board_name as normalize_sector_board_name
from stock_research.market.sector_provider import (
    coverage_can_still_pass,
    effective_pipeline_retries,
    sector_history_is_fresh,
    validate_sector_histories,
)
from stock_research.storage import Database, SectorRepository


OUTPUT_DIR = str(PATHS.market_exports)
CACHE_DIR = str(PATHS.cache / "sector_watch")
BOARD_CONSTITUENT_FILE = str(PATHS.cache / "sector_mainline_constituents.csv")
BENCHMARK_SYMBOL = "sh000001"
BOARD_LIST_MAX_AGE = "24h"
THS_BOARD_SOURCE = "ths/board_list"
THS_HISTORY_SOURCE = "ths/history"
MINIMUM_SECTOR_COVERAGE = 0.95
WATCH_REQUIRED_HISTORY_ROWS = 60


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
    return normalize_sector_board_name(name)


def _board_names_from_repository(frame):
    return (
        frame[["board_name", "board_code"]]
        .drop_duplicates("board_name")
        .reset_index(drop=True)
    )


def load_board_names(
    retries=4,
    retry_delay=2.0,
    repository=None,
    logger=None,
    provider=None,
):
    provider = provider or ths
    pipeline_retries = effective_pipeline_retries(provider, retries)
    stale_database = pd.DataFrame()
    if repository is not None:
        try:
            cached = repository.load_boards(
                max_age=BOARD_LIST_MAX_AGE,
                source=THS_BOARD_SOURCE,
            )
            if not cached.empty and cached["board_code"].notna().all():
                result = _board_names_from_repository(cached)
                if logger is not None:
                    logger.event(
                        "board_names",
                        "duckdb",
                        "hit",
                        message="同花顺行业板块列表命中 DuckDB",
                        rows=len(result),
                    )
                return result
            stale_database = repository.load_boards(source=THS_BOARD_SOURCE)
        except Exception as exc:
            if logger is not None:
                logger.event("board_names", "duckdb", "failed", message=str(exc))

    try:
        raw = call_with_backoff(
            provider.load_board_list,
            "同花顺行业板块列表",
            retries=pipeline_retries,
            retry_delay=retry_delay,
        )
    except Exception:
        if stale_database.empty:
            raise
        result = _board_names_from_repository(stale_database)
        result.attrs["offline_cache"] = True
        return result

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["board_name", "board_code"])
    result = raw.rename(columns={"name": "board_name", "code": "board_code"}).copy()
    result["board_name"] = result["board_name"].map(normalize_board_name)
    result["board_code"] = result["board_code"].astype(str).str.zfill(6)
    result = result.dropna(subset=["board_name"]).drop_duplicates("board_name")
    result = result[["board_name", "board_code"]].reset_index(drop=True)
    if repository is not None:
        rows = repository.replace_boards(result, source=THS_BOARD_SOURCE)
        if logger is not None:
            logger.event(
                "board_names",
                "ths",
                "write",
                message="同花顺行业板块快照写入 DuckDB",
                rows=rows,
            )
    write_cache("ths_board_names", "industry", result)
    return result


def _to_akshare_board_history(frame):
    rename = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "amount": "成交额",
        "volume": "成交量",
        "pct_chg": "涨跌幅",
    }
    converted = frame.rename(columns=rename).copy()
    if "涨跌幅" in converted.columns:
        converted["涨跌幅"] = pd.to_numeric(
            converted["涨跌幅"], errors="coerce"
        ) * 100.0
    return converted[[column for column in rename.values() if column in converted.columns]]


def load_board_history(
    board_name,
    days,
    as_of_date=None,
    retries=4,
    retry_delay=2.0,
    repository=None,
    logger=None,
    offline_cache=False,
    board_code=None,
    provider=None,
    request_sleep=0.0,
):
    del offline_cache
    provider = provider or ths
    pipeline_retries = effective_pipeline_retries(provider, retries)
    end_dt = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
    end_date = end_dt.strftime("%Y%m%d")
    start_date = (end_dt - pd.Timedelta(days=max(days * 3, 180))).strftime("%Y%m%d")
    stored = pd.DataFrame()
    if repository is not None:
        try:
            stored = repository.load_board_history(
                board_name,
                end_date=end_dt,
                days=days,
                date_column="date",
                source=THS_HISTORY_SOURCE,
            )
            if not stored.empty:
                stored = stored.sort_values("date")
                has_required_depth = len(stored) >= min(days, 60)
                cache_reaches_observation = stored["date"].max() >= end_dt.normalize()
                if has_required_depth and cache_reaches_observation:
                    if logger is not None:
                        logger.event(
                            "board_history",
                            "duckdb",
                            "hit",
                            message=f"{board_name} K线命中 DuckDB",
                            rows=len(stored),
                            context={"board": board_name},
                        )
                    return _to_akshare_board_history(stored).reset_index(drop=True)
                if has_required_depth:
                    start_date = (
                        stored["date"].max() + pd.Timedelta(days=1)
                    ).strftime("%Y%m%d")
        except Exception as exc:
            if logger is not None:
                logger.event(
                    "board_history",
                    "duckdb",
                    "failed",
                    message=str(exc),
                    context={"board": board_name},
                )
    if not board_code:
        raise ValueError(f"{board_name} 缺少同花顺板块代码")
    try:
        if request_sleep > 0:
            time.sleep(request_sleep)
        df = call_with_backoff(
            lambda: provider.load_board_history(
                board_code,
                start_date=start_date,
                end_date=end_date,
            ),
            f"{board_name} 同花顺板块K线",
            retries=pipeline_retries,
            retry_delay=retry_delay,
        )
        if logger is not None:
            logger.event(
                "board_history",
                "ths",
                "hit",
                message=f"{board_name} K线使用同花顺",
                rows=len(df),
                context={"board": board_name, "board_code": str(board_code)},
            )
    except Exception:
        if not stored.empty:
            return _to_akshare_board_history(stored).reset_index(drop=True)
        raise
    if df is None or df.empty:
        return pd.DataFrame()
    canonical = df.copy()
    canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
    canonical = canonical.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if repository is not None:
        try:
            rows = repository.upsert_board_history(
                board_name,
                canonical,
                source=THS_HISTORY_SOURCE,
            )
            if logger is not None:
                logger.event(
                    "board_history",
                    "duckdb",
                    "write",
                    message=f"{board_name} K线写入 DuckDB",
                    rows=rows,
                    context={"board": board_name},
                )
        except Exception as exc:
            if logger is not None:
                logger.event(
                    "board_history",
                    "duckdb",
                    "write_failed",
                    message=str(exc),
                    context={"board": board_name},
                )
    canonical_columns = [
        "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"
    ]
    combined = pd.concat(
        [
            stored[[column for column in canonical_columns if column in stored.columns]],
            canonical[[column for column in canonical_columns if column in canonical.columns]],
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")
    converted = _to_akshare_board_history(combined)
    write_cache("ths_board_history", board_code, converted)
    return converted.tail(days).reset_index(drop=True)


def load_benchmark(days, as_of_date=None):
    df = ak.stock_zh_index_daily(symbol=BENCHMARK_SYMBOL)
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


def load_limit_up_counts(
    days,
    as_of_date=None,
    *,
    date_keys=None,
    retries=2,
    retry_delay=0.5,
):
    counts = {}
    if date_keys is None:
        dates = pd.bdate_range(
            end=pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today(),
            periods=days,
        )
        date_keys = dates.strftime("%Y-%m-%d").tolist()
    required_dates = list(date_keys)[-days:]
    if len(required_dates) < days:
        raise RuntimeError(
            f"涨停池所需交易日不足: expected={days} actual={len(required_dates)}"
        )
    for date_key in required_dates:
        cached = read_cache("limit_up_pool", date_key)
        if cached.empty:
            date_str = str(date_key).replace("-", "")
            try:
                pool = call_with_backoff(
                    lambda date=date_str: ak.stock_zt_pool_em(date=date),
                    f"{date_key} 涨停池",
                    retries=retries,
                    retry_delay=retry_delay,
                )
            except Exception as exc:
                raise RuntimeError(f"涨停池 {date_key} 读取失败: {exc}") from exc
            if pool is None or pool.empty:
                raise RuntimeError(f"涨停池 {date_key} 缺失，拒绝按 0 继续统计")
            rows = []
            for _, row in pool.iterrows():
                board = row.get("所属行业") or row.get("行业")
                if not board or pd.isna(board):
                    continue
                rows.append(
                    {
                        "date_key": date_key,
                        "code": str(row.get("代码", "")).replace(".0", "").zfill(6),
                        "name": row.get("名称", ""),
                        "board": normalize_board_name(board),
                    }
                )
            cached = pd.DataFrame(rows)
            if cached.empty:
                raise RuntimeError(f"涨停池 {date_key} 缺少行业字段")
            write_cache("limit_up_pool", date_key, cached)
        for board, count in cached.groupby("board").size().items():
            counts[str(board)] = counts.get(str(board), 0) + int(count)
    return counts


def to_market_code(value):
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    code = code.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    return code


def _invalidate_constituent_output(target):
    for artifact in (target, f"{target}.meta.json"):
        try:
            os.remove(artifact)
        except FileNotFoundError:
            pass


def _valid_constituent_frame(frame):
    return (
        isinstance(frame, pd.DataFrame)
        and not frame.empty
        and {"code", "name"}.issubset(frame.columns)
        and frame["code"].notna().all()
    )


def save_mainline_constituents(
    board_df,
    top=10,
    retries=4,
    retry_delay=2.0,
    sleep=0.1,
    output_path=None,
    provider=None,
):
    provider = provider or ths
    pipeline_retries = effective_pipeline_retries(provider, retries)
    selected = board_df.head(max(0, int(top)))
    expected_boards = len(selected)
    target = output_path or BOARD_CONSTITUENT_FILE
    _invalidate_constituent_output(target)
    if expected_boards == 0:
        raise RuntimeError("主流板块成分覆盖失败: 没有可抓取的主线板块")

    rows = []
    completed_boards = 0
    for rank, (_, board_row) in enumerate(selected.iterrows(), start=1):
        board = str(board_row["board"])
        raw_board_code = board_row.get("board_code")
        board_code = "" if pd.isna(raw_board_code) else str(raw_board_code).strip()
        if not board_code:
            raise RuntimeError(
                f"主流板块成分覆盖失败: {board} 缺少同花顺板块代码 "
                f"({completed_boards}/{expected_boards})"
            )
        try:
            member_cache_path = cache_path("ths_constituents", board_code)
            cache_is_fresh = (
                os.path.isfile(member_cache_path)
                and time.time() - os.path.getmtime(member_cache_path) <= 24 * 60 * 60
            )
            members = (
                read_cache("ths_constituents", board_code)
                if cache_is_fresh
                else pd.DataFrame()
            )
            if not _valid_constituent_frame(members):
                if sleep > 0:
                    time.sleep(sleep)
                members = call_with_backoff(
                    lambda code=board_code: provider.load_board_constituents(code),
                    f"{board} 成分股",
                    retries=pipeline_retries,
                    retry_delay=retry_delay,
                )
                if not _valid_constituent_frame(members):
                    raise RuntimeError("接口返回空表或缺少 code/name 字段")
                write_cache("ths_constituents", board_code, members)
            for _, member in members.iterrows():
                rows.append({
                    "code": to_market_code(member.get("code")),
                    "name": member.get("name", ""),
                    "board": board,
                    "board_code": board_code,
                    "board_rank": rank,
                    "board_score": board_row.get("final_score"),
                    "board_date": board_row.get("date"),
                })
            completed_boards += 1
        except Exception as exc:
            _invalidate_constituent_output(target)
            raise RuntimeError(
                f"主流板块成分覆盖失败: {board}: {exc} "
                f"({completed_boards}/{expected_boards})"
            ) from exc

    if completed_boards != expected_boards or not rows:
        _invalidate_constituent_output(target)
        raise RuntimeError(
            f"主流板块成分覆盖失败: expected={expected_boards} "
            f"completed={completed_boards}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    saved = pd.DataFrame(rows).drop_duplicates(["code", "board"])
    tmp_path = f"{target}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        saved.to_csv(
            tmp_path, index=False, encoding="utf-8-sig"
        )
        os.replace(tmp_path, target)
        historical_output = os.path.abspath(target) != os.path.abspath(
            BOARD_CONSTITUENT_FILE
        )
        write_metadata(target, {
            "kind": "sector_mainline_constituents",
            "point_in_time_status": "unsafe" if historical_output else "safe",
            "point_in_time_note": (
                "board ranks are historical, but constituent membership was retrieved from the current API"
                if historical_output else "current-date board ranks and constituents"
            ),
            "board_coverage_expected": expected_boards,
            "board_coverage_completed": completed_boards,
            "board_coverage": completed_boards / expected_boards,
        })
    except Exception:
        _invalidate_constituent_output(target)
        raise
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
    print(
        f"主流板块成分映射已保存: {target}，{len(saved)} 条，"
        f"板块覆盖 {completed_boards}/{expected_boards}"
    )
    return saved.reset_index(drop=True)


def apply_constituent_limit_up_counts(board_df, constituents, date_keys):
    """Confirm ranked-board limit-up counts by stock-code intersection."""
    if constituents is None or constituents.empty:
        return board_df
    pool_frames = []
    for date_key in list(date_keys):
        pool = read_cache("limit_up_pool", date_key)
        if pool.empty or "code" not in pool.columns:
            raise RuntimeError(f"涨停池缓存 {date_key} 缺失，无法核对板块成分")
        pool = pool.copy()
        pool["code"] = pool["code"].map(to_market_code)
        pool_frames.append(pool[["code"]])
    all_limit_codes = pd.concat(pool_frames, ignore_index=True)["code"]
    result = board_df.copy()
    for board, members in constituents.groupby("board"):
        member_codes = set(members["code"].astype(str))
        exact_count = int(all_limit_codes.isin(member_codes).sum())
        mask = result["board"].eq(board)
        result.loc[mask, "limit_up_count"] = exact_count
        result.loc[mask, "final_score"] = result.loc[mask, "mainline_score"].map(
            lambda score: round(score + score_direct(exact_count, 0, 8) * 0.15, 1)
        )
    return result


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


def main(argv=None, repository=None, logger=None, provider=None):
    args = parse_args(argv)
    provider = provider or ths
    as_of_date = args.as_of_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    if repository is None:
        database = Database()
        database.initialize()
        repository = SectorRepository(database)
    if logger is None:
        logger = PartLogger("sector_watch", repository=repository)
    try:
        with logger.part("board_names"):
            boards = load_board_names(
                retries=args.retries,
                retry_delay=args.retry_delay,
                repository=repository,
                logger=logger,
                provider=provider,
            )
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
    history_days = max(args.days, WATCH_REQUIRED_HISTORY_ROWS)
    benchmark = load_benchmark(history_days + 5, as_of_date)
    if benchmark.empty:
        print("上证指数数据为空，无法确定板块观察交易日")
        raise SystemExit(2)
    observation_date = benchmark["date"].max().normalize()
    benchmark_dates = (
        benchmark["date"].dt.strftime("%Y-%m-%d").tolist()
        if not benchmark.empty
        else None
    )
    limit_date_keys = (
        benchmark_dates[-args.limit_up_days:]
        if benchmark_dates
        else pd.bdate_range(end=pd.Timestamp(as_of_date), periods=args.limit_up_days)
        .strftime("%Y-%m-%d")
        .tolist()
    )
    limit_up_counts = load_limit_up_counts(
        args.limit_up_days,
        as_of_date,
        date_keys=benchmark_dates,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )

    board_names = boards["board_name"].tolist()
    board_code_by_name = dict(zip(boards["board_name"], boards["board_code"]))
    offline_cache = bool(boards.attrs.get("offline_cache"))

    def process_board(board_name):
        try:
            hist = load_board_history(
                board_name,
                history_days,
                observation_date,
                retries=args.retries,
                retry_delay=args.retry_delay,
                repository=repository,
                logger=logger,
                offline_cache=offline_cache,
                board_code=board_code_by_name.get(board_name),
                provider=provider,
                request_sleep=args.sleep,
            )
            if sector_history_is_fresh(
                hist,
                observation_date=observation_date,
                max_stale_days=0,
                minimum_rows=WATCH_REQUIRED_HISTORY_ROWS,
            ):
                row = calc_board_metrics(board_name, hist, benchmark)
                row["board_code"] = board_code_by_name.get(board_name)
                row["limit_up_count"] = limit_up_counts.get(board_name, 0)
                row["final_score"] = round(row["mainline_score"] + score_direct(row["limit_up_count"], 0, 8) * 0.15, 1)
                return row, hist
            return None, hist
        except Exception as exc:
            print(f"跳过 {board_name}: {exc}")
        return None, pd.DataFrame()

    rows = []
    history_map = {}
    workers = max(1, args.workers)
    expected_boards = len(board_names)
    fresh_completed = 0
    completed = 0
    with logger.part("board_history"):
        executor = ThreadPoolExecutor(max_workers=workers)
        pending = {}
        board_iterator = iter(board_names)
        stopped_early = False

        def submit_until_full():
            while len(pending) < workers:
                try:
                    next_board = next(board_iterator)
                except StopIteration:
                    return
                pending[executor.submit(process_board, next_board)] = next_board

        try:
            submit_until_full()
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    board_name = pending.pop(future)
                    row, history = future.result()
                    completed += 1
                    if history is not None and not history.empty:
                        history_map[board_name] = history
                        if sector_history_is_fresh(
                            history,
                            observation_date=observation_date,
                            max_stale_days=0,
                            minimum_rows=WATCH_REQUIRED_HISTORY_ROWS,
                        ):
                            fresh_completed += 1
                    if row:
                        rows.append(row)
                    if not coverage_can_still_pass(
                        expected=expected_boards,
                        completed=completed,
                        fresh=fresh_completed,
                        minimum=MINIMUM_SECTOR_COVERAGE,
                    ):
                        stopped_early = True
                        message = (
                            f"提前停止: completed={completed} fresh={fresh_completed} "
                            f"remaining={expected_boards - completed}，已无法达到 "
                            f"{MINIMUM_SECTOR_COVERAGE:.0%} 覆盖率"
                        )
                        print(message)
                        logger.event(
                            "board_history",
                            "early_stop",
                            "failed",
                            message=message,
                            rows=fresh_completed,
                        )
                        break
                    if completed % 20 == 0:
                        print(
                            f"进度 {completed}/{len(board_names)}, "
                            f"有效 {len(rows)}"
                        )
                        logger.event(
                            "board_history",
                            "progress",
                            "running",
                            message=(
                                f"进度 {completed}/{len(board_names)}, "
                                f"有效 {len(rows)}"
                            ),
                            rows=len(rows),
                        )
                if stopped_early:
                    for future in pending:
                        future.cancel()
                    break
                submit_until_full()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    history_map, coverage = validate_sector_histories(
        board_names,
        history_map,
        observation_date=observation_date,
        max_stale_days=0,
        minimum_rows=WATCH_REQUIRED_HISTORY_ROWS,
        minimum=MINIMUM_SECTOR_COVERAGE,
    )
    coverage_status = "pass" if coverage.passed else "failed"
    coverage_message = (
        f"provider=ths expected={coverage.expected} fresh={coverage.fresh} "
        f"stale={coverage.stale} missing={coverage.missing} "
        f"coverage={coverage.coverage:.1%}"
    )
    print(f"板块覆盖率: {coverage_message}")
    logger.event(
        "coverage",
        "ths",
        coverage_status,
        message=coverage_message,
        rows=coverage.fresh,
        context={
            "expected": coverage.expected,
            "fresh": coverage.fresh,
            "stale": coverage.stale,
            "missing": coverage.missing,
            "coverage": coverage.coverage,
        },
    )
    if not coverage.passed:
        raise SystemExit(2)

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
    observation_stamp = observation_date.strftime("%Y%m%d")
    path = os.path.join(
        OUTPUT_DIR,
        f"sector_watch_asof_{observation_stamp}_{datetime.now().strftime('%H%M%S')}.csv",
    )
    constituent_path = str(
        PATHS.cache / f"sector_mainline_constituents_{observation_stamp}.csv"
    ) if args.as_of_date else BOARD_CONSTITUENT_FILE
    try:
        constituents = save_mainline_constituents(
            df,
            top=min(10, args.top),
            retries=args.retries,
            retry_delay=args.retry_delay,
            sleep=args.sleep,
            output_path=constituent_path,
            provider=provider,
        )
    except Exception as exc:
        print(f"主流板块成分获取失败，停止板块观察输出: {exc}")
        raise SystemExit(2) from exc
    df = apply_constituent_limit_up_counts(df, constituents, limit_date_keys)
    df = df.sort_values(["final_score", "mainline_score", "ret5"], ascending=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print_sector_report(df, path, args.top)
