"""Build a point-in-time PE/PB low-valuation candidate stream for the left lane.

Determination mirrors factor_selection.get_pe_pb_metrics ("历史低估均值"):
- low_avg = annual-low valuations over completed calendar years strictly
  before the observation year, with outliers removed, then averaged.
- ratio = current valuation / low_avg; low-valuation candidate = ratio <= 1.0.
- PE industries use static pe (tushare pe_ttm is only populated from 2022);
  PB industries use pb.  Industry routing matches factor_selection.

Point-in-time: each day's candidates only use data available up to that day —
the annual minima only cover years that ended before the observation day, and
the valuation row itself is that trading day's daily_basic row.  Execution is
T+1 by the engine.

Each day at most `top_n` candidates survive (sorted by ratio ascending), the
"设每日上限择优" policy.  Quality (latest announced financial event),
market cap and point-in-time tradability filters are applied before the cap.

Usage:
    python -m apps.quantsplaybook_pe_pb_candidates --start 2021-01-01 --end 2026-07-21
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from apps.tushare_financial_cache import _quality_score
from stock_research.core.paths import PATHS
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION

DB_PATH = PATHS.project_root / "var" / "data" / "my_trade.sqlite3"
Q1_VALUE_DIR = PATHS.project_root / "var" / "cache" / "q1_value"
DEFAULT_OUTPUT_ROOT = PATHS.project_root / "var" / "backtests" / "pe_pb_2021_to_20260721"
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests" / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_INDUSTRY_MANIFEST = (
    PATHS.runtime_root / "backtests"
    / "tushare_official_candidates_2021_to_20260721_merged"
    / "manifest.json"
)
DEFAULT_CACHE_DIRECTORY = PATHS.project_root / "var" / "cache" / "pe_pb_pit"
FACTOR = "pe_pb_value"
DEFAULT_MIN_MKTCAP = 100.0
DEFAULT_RATIO_CUT = 1.0
DEFAULT_MIN_QUALITY = 60.0
DEFAULT_TOP_N = 5
DEFAULT_MINIMUM_LISTING_DAYS = 300

# 与 stock_research/pipelines/factor_selection.py 的 PE_KEYWORDS / PB_KEYWORDS 一致
PE_KEYWORDS = [
    "货币金融", "银行", "保险", "酒、饮料和精制茶", "食品制造", "批发", "零售",
    "医药制造", "纺织服装", "住宿", "餐饮", "旅游", "饰品", "医疗服务", "医疗设备",
]
PB_KEYWORDS = [
    "钢铁", "煤炭", "有色", "化学原料", "化学纤维", "建材", "石油", "采矿",
    "非金属矿", "黑色金属", "燃料加工", "电信", "广播电视和卫星", "房地产",
    "电力", "热力", "燃气", "水的生产", "资本市场", "其他金融",
    "建筑", "土木工程", "交通运输", "道路运输", "铁路运输", "水上运输", "航空运输",
    "仓储", "邮政", "农业", "林业", "畜牧", "渔业",
]
INDUSTRY_MANIFEST_FIELDS = (
    "industry_source_path",
    "industry_source_sha256",
    "industry_source_modified_at",
    "industry_as_of_date",
    "industry_data_source",
    "industry_point_in_time_status",
    "industry_mapping_count",
    "industry_universe_count",
    "industry_coverage_ratio",
    "industry_point_in_time_note",
)


def _date_text(value) -> str | None:
    date = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(date) else date.strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _database_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _cached_frame(
    *,
    db: Path,
    cache_directory: str | Path,
    name: str,
    builder,
) -> tuple[pd.DataFrame, dict]:
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{name}.pkl"
    manifest_path = directory / f"{name}.manifest.json"
    fingerprint = _database_fingerprint(db)
    if data_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("database_fingerprint") == fingerprint
                and manifest.get("data_sha256") == _sha256(data_path)
            ):
                frame = pd.read_pickle(data_path)
                return frame, {**manifest, "cache_hit": True}
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            pass
    frame = builder(db)
    temporary = data_path.with_suffix(".tmp")
    frame.to_pickle(temporary)
    temporary.replace(data_path)
    manifest = {
        "cache_schema": 1,
        "name": name,
        "database_fingerprint": fingerprint,
        "row_count": int(len(frame)),
        "data_path": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return frame, {**manifest, "cache_hit": False}


def classify_method(industry) -> str:
    text = str(industry or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "UNKNOWN"
    if any(keyword in text for keyword in PE_KEYWORDS):
        return "PE"
    if any(keyword in text for keyword in PB_KEYWORDS):
        return "PB"
    return "OTHER"


def remove_outliers(values):
    values = [float(v) for v in values]
    if len(values) < 3:
        return values
    median = np.median(values)
    if median <= 0:
        return values
    filtered = [v for v in values if 0.5 * median <= v <= 2.0 * median]
    return filtered if len(filtered) >= 3 else values


def load_stock_basic(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db))
    try:
        frame = pd.read_sql_query(
            """
            SELECT ts_code,
                   json_extract(payload_json, '$.name') AS name,
                   json_extract(payload_json, '$.list_date') AS list_date,
                   json_extract(payload_json, '$.delist_date') AS delist_date
            FROM raw_tushare_dataset_rows
            WHERE dataset='basic'
            ORDER BY ts_code, row_key
            """,
            con,
        )
    finally:
        con.close()
    for field in ("list_date", "delist_date"):
        frame[field] = pd.to_datetime(
            frame[field], format="%Y%m%d", errors="coerce",
        )
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["name"] = frame["name"].fillna(frame["ts_code"]).astype(str)
    return (
        frame.dropna(subset=["list_date"])
        .drop_duplicates("ts_code", keep="last")
        .reset_index(drop=True)
    )


def load_industry_history(
    manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(str(manifest.get("industry_source_path") or ""))
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    if not source_path.is_file():
        raise RuntimeError(f"历史行业来源不存在: {source_path}")
    expected = str(manifest.get("industry_source_sha256") or "")
    if not expected or _sha256(source_path) != expected:
        raise RuntimeError(f"历史行业来源哈希不一致: {source_path}")
    if manifest.get("industry_point_in_time_status") != "safe":
        raise RuntimeError("历史行业来源未标记为时点安全")
    history = pd.read_csv(source_path, dtype=str)
    required = {"ts_code", "industry", "in_date", "out_date"}
    missing = required - set(history.columns)
    if missing:
        raise RuntimeError(f"历史行业来源缺字段: {sorted(missing)}")
    history["in_date"] = pd.to_datetime(
        history["in_date"], format="%Y%m%d", errors="coerce",
    )
    history["out_date"] = pd.to_datetime(
        history["out_date"], format="%Y%m%d", errors="coerce",
    )
    history = history.dropna(subset=["ts_code", "in_date"]).copy()
    history["industry"] = history["industry"].fillna("").astype(str).str.strip()
    return history, manifest


class PointInTimeIndustryResolver:
    def __init__(self, history: pd.DataFrame):
        events = []
        for row in history.itertuples(index=False):
            code = str(row.ts_code)
            start = pd.Timestamp(row.in_date).normalize()
            industry = str(row.industry or "").strip()
            events.append((start, 1, code, start, industry))
            if pd.notna(row.out_date):
                events.append((
                    pd.Timestamp(row.out_date).normalize(),
                    0,
                    code,
                    start,
                    industry,
                ))
        self._events = sorted(events, key=lambda item: (item[0], item[1], item[2]))
        self._position = 0
        self._current: dict[str, tuple[pd.Timestamp, str]] = {}
        self._last_date: pd.Timestamp | None = None

    def advance(self, date) -> None:
        target = pd.Timestamp(date).normalize()
        if self._last_date is not None and target < self._last_date:
            raise ValueError("industry resolver dates must be non-decreasing")
        while (
            self._position < len(self._events)
            and self._events[self._position][0] <= target
        ):
            _, kind, code, start, industry = self._events[self._position]
            if kind == 0:
                if self._current.get(code, (None,))[0] == start:
                    self._current.pop(code, None)
            else:
                current = self._current.get(code)
                if current is None or start >= current[0]:
                    self._current[code] = (start, industry)
            self._position += 1
        self._last_date = target

    def industry(self, ts_code: str) -> str:
        current = self._current.get(str(ts_code))
        return "" if current is None else current[1]


def load_financial_history(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db))
    try:
        frame = pd.read_sql_query(
            """
            SELECT ts_code, ann_date, report_period, row_key,
                   json_extract(payload_json, '$.dt_eps') AS dt_eps,
                   json_extract(payload_json, '$.eps') AS eps,
                   json_extract(payload_json, '$.dt_netprofit_yoy') AS dt_netprofit_yoy,
                   json_extract(payload_json, '$.netprofit_yoy') AS netprofit_yoy,
                   json_extract(payload_json, '$.roe_dt') AS roe_dt,
                   json_extract(payload_json, '$.roe') AS roe
            FROM raw_tushare_dataset_rows
            WHERE dataset='fina_indicator' AND ann_date IS NOT NULL
            ORDER BY ts_code, ann_date, report_period, row_key
            """,
            con,
        )
    finally:
        con.close()
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="coerce")
    frame["report_period"] = pd.to_datetime(
        frame["report_period"], errors="coerce",
    )
    for field in (
        "dt_eps", "eps", "dt_netprofit_yoy", "netprofit_yoy", "roe_dt", "roe",
    ):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "ann_date", "report_period"])
    frame = frame.sort_values(
        ["ts_code", "ann_date", "report_period", "row_key"],
        kind="mergesort",
    ).drop_duplicates(["ts_code", "ann_date", "report_period"], keep="last")
    records = []
    for row in frame.itertuples(index=False):
        eps = row.dt_eps if pd.notna(row.dt_eps) else row.eps
        yoy_pct = (
            row.dt_netprofit_yoy
            if pd.notna(row.dt_netprofit_yoy)
            else row.netprofit_yoy
        )
        roe = row.roe_dt if pd.notna(row.roe_dt) else row.roe
        if pd.isna(eps) or pd.isna(yoy_pct):
            continue
        yoy = float(yoy_pct) / 100.0
        records.append({
            "ts_code": str(row.ts_code),
            "announcement_date": pd.Timestamp(row.ann_date).normalize(),
            "report_period": pd.Timestamp(row.report_period).normalize(),
            "quality_score": _quality_score(
                float(eps), yoy, None if pd.isna(roe) else float(roe),
            ),
            "earnings_yoy": yoy,
        })
    return pd.DataFrame(records)


class PointInTimeFinancialResolver:
    def __init__(self, history: pd.DataFrame):
        self._history = {}
        for code, group in history.groupby("ts_code", sort=False):
            ordered = group.sort_values(
                ["announcement_date", "report_period"], kind="mergesort",
            )
            effective_dates = []
            effective_records = []
            latest_by_period = {}
            for announcement_date, events in ordered.groupby(
                "announcement_date", sort=True,
            ):
                for record in events.to_dict("records"):
                    latest_by_period[pd.Timestamp(record["report_period"])] = record
                latest_period = max(latest_by_period)
                effective_dates.append(pd.Timestamp(announcement_date).normalize())
                effective_records.append(dict(latest_by_period[latest_period]))
            self._history[str(code)] = (effective_dates, effective_records)

    def latest(self, ts_code: str, observation_date) -> dict:
        item = self._history.get(str(ts_code))
        if item is None:
            return {}
        dates, records = item
        index = bisect.bisect_right(
            dates, pd.Timestamp(observation_date).normalize(),
        ) - 1
        return {} if index < 0 else records[index]


def load_yearly_min(db: Path) -> pd.DataFrame:
    """每 (ts_code, 日历年) 的年度最低 pe（静态 PE）/ pb。"""
    con = sqlite3.connect(str(db))
    sql = (
        "SELECT ts_code, substr(trade_date, 1, 4) AS yr, "
        "MIN(json_extract(payload_json, '$.pe')) AS min_pe, "
        "MIN(json_extract(payload_json, '$.pb')) AS min_pb "
        "FROM raw_tushare_dataset_rows WHERE dataset='daily_basic' "
        "GROUP BY ts_code, yr"
    )
    frame = pd.read_sql_query(sql, con)
    con.close()
    for col in ("min_pe", "min_pb"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_yearly_low_avg(yearly: pd.DataFrame, min_years: int = 3) -> dict[str, dict]:
    """{ts_code: {obs_year: {pe: low_avg, pb: low_avg}}}.

    低估均值只用观察年份之前已结束的完整年度；至少 min_years 个完整年度。
    一年内不变，所以预计算。
    """
    yearly = yearly.copy()
    yearly["yr_i"] = yearly["yr"].astype(int)
    out: dict[str, dict] = {}
    for ts_code, group in yearly.groupby("ts_code"):
        years = sorted(group["yr_i"].tolist())
        if len(years) < min_years:
            continue
        by_year = {int(r["yr_i"]): (r["min_pe"], r["min_pb"]) for _, r in group.iterrows()}
        by_year = {y: (float(pe), float(pb)) if not (pd.isna(pe) and pd.isna(pb)) else (None, None) for y, (pe, pb) in by_year.items()}
        rec = {}
        for obs_year in range(years[0] + min_years, years[-1] + 2):
            prior = [y for y in years if y < obs_year]
            if len(prior) < min_years:
                continue
            pe_vals = [by_year[y][0] for y in prior if by_year[y][0] is not None]
            pb_vals = [by_year[y][1] for y in prior if by_year[y][1] is not None]
            item = {}
            if len(pe_vals) >= min_years:
                item["pe"] = float(np.mean(remove_outliers(pe_vals)))
            if len(pb_vals) >= min_years:
                item["pb"] = float(np.mean(remove_outliers(pb_vals)))
            if item:
                rec[obs_year] = item
        if rec:
            out[ts_code] = rec
    return out


def load_st_set(db: Path) -> set[tuple[str, str]]:
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    cur.execute(
        "SELECT json_extract(payload_json, '$.ts_code'), "
        "json_extract(payload_json, '$.trade_date') "
        "FROM raw_tushare_dataset_rows WHERE dataset='stock_st'"
    )
    out = set()
    for ts_code, trade_date in cur.fetchall():
        if ts_code and trade_date:
            day = str(trade_date)
            if len(day) == 8:
                day = f"{day[:4]}-{day[4:6]}-{day[6:]}"
            out.add((ts_code, day))
    con.close()
    return out


def point_in_time_tradable_rows(
    frame: pd.DataFrame,
    stock_basic: pd.DataFrame,
    *,
    observation_date: str,
    minimum_listing_days: int = DEFAULT_MINIMUM_LISTING_DAYS,
    st_set: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Filter a daily cross-section using only information valid that day."""
    day = pd.Timestamp(observation_date).normalize()
    result = frame.merge(
        stock_basic[["ts_code", "list_date", "delist_date"]],
        on="ts_code",
        how="inner",
        validate="many_to_one",
    )
    valid_bar = pd.Series(True, index=result.index)
    for field in ("open", "high", "low", "kline_close", "volume", "amount"):
        valid_bar &= pd.to_numeric(result[field], errors="coerce").gt(0)
    listed = (
        result["list_date"].notna()
        & result["list_date"].le(day)
        & (
            result["delist_date"].isna()
            | result["delist_date"].ge(day)
        )
        & (day - result["list_date"]).dt.days.ge(int(minimum_listing_days))
    )
    result = result.loc[valid_bar & listed].copy()
    if st_set:
        risk = pd.Series(
            [
                (str(code), observation_date) in st_set
                for code in result["ts_code"]
            ],
            index=result.index,
        )
        result = result.loc[~risk]
    return result.reset_index(drop=True)


def load_observation_days(
    formula_history: str | Path,
    start: str,
    end: str,
) -> list[str]:
    frame = pd.read_csv(formula_history, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    return [
        date.strftime("%Y-%m-%d")
        for date in dates.drop_duplicates().sort_values()
        if start_date <= date.normalize() <= end_date
    ]


def deduplicate_market_rows(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    keys = ["ts_code", "trade_date"]
    extra_rows = int(len(frame) - frame[keys].drop_duplicates().shape[0])
    conflicting_keys = 0
    if extra_rows:
        duplicate_rows = frame.loc[frame.duplicated(keys, keep=False)]
        conflicts = duplicate_rows.groupby(keys, dropna=False)[
            list(value_columns)
        ].nunique(dropna=False).gt(1).any(axis=1)
        conflicting_keys = int(conflicts.sum())
    ordered = frame.copy()
    ordered["_updated_at"] = pd.to_datetime(
        ordered.get("updated_at"), errors="coerce", utc=True,
    )
    ordered = (
        ordered.sort_values(
            [*keys, "_updated_at", "row_key"],
            kind="mergesort",
            na_position="first",
        )
        .drop_duplicates(keys, keep="last")
        .drop(columns=["_updated_at"])
        .reset_index(drop=True)
    )
    return ordered, {
        "extra_rows": extra_rows,
        "conflicting_keys": conflicting_keys,
        "resolution": "latest_updated_at_then_row_key",
    }


def load_daily_at(db: Path, days: list[str]) -> pd.DataFrame:
    ph = ",".join("?" * len(days))
    con = sqlite3.connect(str(db))
    frame = pd.read_sql_query(
        "SELECT ts_code, trade_date, row_key, updated_at, "
        "json_extract(payload_json, '$.pe') AS pe, "
        "json_extract(payload_json, '$.pb') AS pb, "
        "json_extract(payload_json, '$.total_mv') AS total_mv, "
        "json_extract(payload_json, '$.close') AS close "
        f"FROM raw_tushare_dataset_rows WHERE dataset='daily_basic' "
        f"AND trade_date IN ({ph})",
        con,
        params=days,
    )
    con.close()
    for col in ("pe", "pb", "total_mv", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame, daily_audit = deduplicate_market_rows(
        frame,
        value_columns=("pe", "pb", "total_mv", "close"),
    )
    frame["total_mv_yi"] = frame["total_mv"] / 10000.0
    kline = sqlite3.connect(str(db))
    try:
        traded = pd.read_sql_query(
            "SELECT ts_code, trade_date, row_key, updated_at, "
            "json_extract(payload_json, '$.open') AS open, "
            "json_extract(payload_json, '$.high') AS high, "
            "json_extract(payload_json, '$.low') AS low, "
            "json_extract(payload_json, '$.close') AS kline_close, "
            "json_extract(payload_json, '$.vol') AS volume, "
            "json_extract(payload_json, '$.amount') AS amount "
            "FROM raw_tushare_dataset_rows WHERE dataset='daily_kline' "
            f"AND trade_date IN ({ph})",
            kline,
            params=days,
        )
    finally:
        kline.close()
    for col in ("open", "high", "low", "kline_close", "volume", "amount"):
        traded[col] = pd.to_numeric(traded[col], errors="coerce")
    traded, kline_audit = deduplicate_market_rows(
        traded,
        value_columns=(
            "open", "high", "low", "kline_close", "volume", "amount",
        ),
    )
    result = frame.merge(
        traded,
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    result.attrs["source_duplicate_audit"] = {
        "daily_basic": daily_audit,
        "daily_kline": kline_audit,
    }
    return result


def build_pe_pb_candidates(
    *,
    output_directory: Path,
    start_date: str,
    end_date: str,
    top_n: int = DEFAULT_TOP_N,
    min_mktcap: float = DEFAULT_MIN_MKTCAP,
    ratio_cut: float = DEFAULT_RATIO_CUT,
    min_quality: float = DEFAULT_MIN_QUALITY,
    min_years: int = 3,
    minimum_listing_days: int = DEFAULT_MINIMUM_LISTING_DAYS,
    cache_directory: str | Path = DEFAULT_CACHE_DIRECTORY,
    formula_history: str | Path = DEFAULT_FORMULA_HISTORY,
    industry_manifest_path: str | Path = DEFAULT_INDUSTRY_MANIFEST,
) -> Path:
    db = DB_PATH
    print("加载股票基础信息 ...")
    stock_basic = load_stock_basic(db)
    stock_names = dict(zip(stock_basic["ts_code"], stock_basic["name"]))
    industry_history, industry_manifest = load_industry_history(
        industry_manifest_path,
    )
    industry_resolver = PointInTimeIndustryResolver(industry_history)
    print(f"  {len(stock_names)} 只股票，{len(industry_history)} 条历史行业关系")

    print("加载年度最低估值 + 预计算低估均值 ...")
    yearly, yearly_cache = _cached_frame(
        db=db,
        cache_directory=cache_directory,
        name="yearly_min_daily_basic",
        builder=load_yearly_min,
    )
    low_avg = build_yearly_low_avg(yearly, min_years)
    print(
        f"  {len(low_avg)} 只股票有低估均值 "
        f"(cache_hit={yearly_cache['cache_hit']})"
    )

    print("加载 ST 集合 + 观察日 ...")
    st_set = load_st_set(db)
    obs_days = load_observation_days(formula_history, start_date, end_date)
    if not obs_days:
        raise RuntimeError("观察日为空")
    print(f"  {len(obs_days)} 个交易日")

    print("加载逐公告日财务质量历史 ...")
    financial_history, financial_cache = _cached_frame(
        db=db,
        cache_directory=cache_directory,
        name="announced_financial_quality",
        builder=load_financial_history,
    )
    financial_resolver = PointInTimeFinancialResolver(
        financial_history,
    )

    # 分月批量加载 daily_basic
    output_directory.mkdir(parents=True, exist_ok=True)
    snapshots = []
    total_cand = 0
    duplicate_audit = {
        source: {
            "extra_rows": 0,
            "conflicting_keys": 0,
            "resolution": "latest_updated_at_then_row_key",
        }
        for source in ("daily_basic", "daily_kline")
    }
    # 按月份分组观察日，减少 SQL 往返
    months = sorted({d[:7] for d in obs_days})
    for month in months:
        month_days = [d for d in obs_days if d.startswith(month)]
        daily = load_daily_at(db, month_days)
        for source, audit in daily.attrs.get(
            "source_duplicate_audit", {}
        ).items():
            duplicate_audit[source]["extra_rows"] += int(audit["extra_rows"])
            duplicate_audit[source]["conflicting_keys"] += int(
                audit["conflicting_keys"]
            )
        by_day = {day: group for day, group in daily.groupby("trade_date")}

        for obs_day in month_days:
            sub = by_day.get(obs_day)
            if sub is None or sub.empty:
                raise RuntimeError(
                    "daily_basic缺少完整观察日，不能生成连续PE/PB候选: "
                    f"{obs_day}"
                )
            industry_resolver.advance(obs_day)
            sub = point_in_time_tradable_rows(
                sub,
                stock_basic,
                observation_date=obs_day,
                minimum_listing_days=minimum_listing_days,
                st_set=st_set,
            )
            sub["industry"] = sub["ts_code"].map(
                industry_resolver.industry,
            )
            sub["method"] = sub["industry"].map(classify_method)
            sub["name"] = sub["ts_code"].map(stock_names).fillna(
                sub["ts_code"],
            )
            sub = sub[sub["method"].isin(["PE", "PB"])]
            obs_year = int(obs_day[:4])
            rows = []
            for _, row in sub.iterrows():
                code = row["ts_code"]
                la = low_avg.get(code, {}).get(obs_year, {})
                if row["method"] == "PE":
                    cur = row["pe"]
                    low = la.get("pe")
                else:
                    cur = row["pb"]
                    low = la.get("pb")
                if cur is None or low is None or not np.isfinite(cur) or cur <= 0 or low <= 0:
                    continue
                if pd.isna(row["total_mv_yi"]) or row["total_mv_yi"] < min_mktcap:
                    continue
                ratio = cur / low
                if ratio > ratio_cut:
                    continue
                quality = financial_resolver.latest(code, obs_day)
                q = quality.get("quality_score")
                if q is None or pd.isna(q) or float(q) < min_quality:
                    continue
                rows.append({
                    "date": obs_day,
                    "code": code,
                    "name": row["name"] or code,
                    "industry": row["industry"] or "",
                    "method": row["method"],
                    "close": row["close"],
                    "value": cur,
                    "low_avg": low,
                    "ratio": ratio,
                    "quality_score": float(q),
                    "earnings_yoy": float(quality["earnings_yoy"]) if pd.notna(quality.get("earnings_yoy")) else None,
                    "financial_report_period": _date_text(
                        quality.get("report_period")
                    ),
                    "financial_effective_date": _date_text(
                        quality.get("announcement_date")
                    ),
                    "mktcap": row["total_mv_yi"],
                    # 与 value_model（quality + yoy×20 + trade_basis）可比区间的
                    # 信号强度：质量 + 低估深度奖励（ratio 越低越深）。避免旧口径
                    # 100*(1-ratio)=0~68 在每日池上限中被右侧高分行挤出。
                    "candidate_score": round(
                        float(q) + max(0.0, (1.0 - ratio)) * 20.0, 2,
                    ),
                })
            best = sorted(rows, key=lambda r: (r["ratio"], r["code"]))[:top_n]
            finalized = [_finalize_row(r) for r in best]
            frame = pd.DataFrame(finalized, columns=PE_PB_CANDIDATE_COLUMNS)
            path = output_directory / f"candidates_{obs_day}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            snapshots.append({
                "date": obs_day,
                "file": path.name,
                "sha256": _sha256(path),
                "candidate_count": len(frame),
                "signal_eligible_count": len(frame),
                "left_candidate_count": len(frame),
                "right_candidate_count": 0,
                "financial_point_in_time": True,
                "strict_financial_point_in_time": False,
                "industry_point_in_time": True,
            })
            total_cand += len(frame)

    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": snapshots[0]["date"] if snapshots else None,
        "end_date": snapshots[-1]["date"] if snapshots else None,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": False,
        "uses_financial_data": True,
        "financial_point_in_time_note": (
            "财务行仅在逐股ann_date不晚于观察日时生效；本地数据不能证明"
            "供应商历史修订版本完整，因此只属于公告日PIT研究。"
        ),
        "unsafe_snapshot_count": 0,
        "industry_point_in_time": True,
        **{
            field: industry_manifest.get(field)
            for field in INDUSTRY_MANIFEST_FIELDS
        },
        "selection_engine": "pe_pb_low_value",
        "selection_profile": FACTOR,
        "tradable_universe_rule": (
            "broad_basic_listed_and_not_delisted;valid_daily_kline;"
            f"minimum_listing_days={int(minimum_listing_days)};dated_stock_st_excluded"
        ),
        "risk_warning_source": "local_tushare_stock_st",
        "source_duplicate_audit": duplicate_audit,
        "valuation_history_cache": yearly_cache,
        "financial_history_cache": financial_cache,
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "lane_policy": "left-lane PE/PB low valuation, top-N per day, point-in-time",
        "research_only": True,
        "total_left_candidate_rows": total_cand,
        "snapshots": snapshots,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n输出: {output_directory}")
    print(f"快照 {len(snapshots)} 个，候选行 {total_cand}（每日 Top {top_n}）")
    return output_directory


def _engine_code(ts_code: str) -> str:
    """tushare 代码 (600153.SH) -> 引擎代码 (sh.600153)。"""
    if not ts_code or "." not in ts_code:
        return ts_code
    symbol, suffix = ts_code.split(".", 1)
    return f"{suffix.lower()}.{symbol}"


def _finalize_row(r: dict) -> dict:
    method = r["method"]
    source = "pe_model" if method == "PE" else "pb_model"
    lane = "left_pe" if method == "PE" else "left_pb"
    ratio = float(r["ratio"])
    close = float(r["close"])
    return {
        "date": r["date"],
        "code": _engine_code(r["code"]),
        "name": r["name"],
        "industry": r["industry"],
        "hybrid_lane": lane,
        "candidate_source": source,
        "selection_engine": "pe_pb_low_value",
        "selection_profile": "pe_pb_left",
        "allow_left": True,
        "allow_right": False,
        "selected_for_trading": True,
        "signal_eligible": True,
        "value_industry_allowed": False,
        "value_industry_allowlist_match": "",
        "financial_point_in_time": True,
        "strict_financial_point_in_time": False,
        "financial_report_period": r.get("financial_report_period"),
        "financial_effective_date": r.get("financial_effective_date"),
        "industry_point_in_time": True,
        "close": close,
        # At unchanged per-share fundamentals, valuation multiple is
        # proportional to price.  close / ratio is therefore the price at the
        # historical-low reference multiple; it is an auditable grid anchor.
        "value_line": close / ratio,
        "price_to_value": ratio,
        "pe_ratio": ratio if method == "PE" else None,
        "pb_ratio": ratio if method == "PB" else None,
        "pe_low_avg": r["low_avg"] if method == "PE" else None,
        "pb_low_avg": r["low_avg"] if method == "PB" else None,
        "quality_score": r["quality_score"],
        "earnings_yoy": r["earnings_yoy"],
        "mktcap": r["mktcap"],
        "candidate_score": r["candidate_score"],
        "selection_rank": 1,
        "snapshot_version": SNAPSHOT_VERSION,
        "factor_horizon_channel_eligible": False,
    }


PE_PB_CANDIDATE_COLUMNS = tuple(_finalize_row({
    "method": "PE",
    "date": "2000-01-01",
    "code": "000001.SZ",
    "name": "",
    "industry": "",
    "close": 1.0,
    "ratio": 1.0,
    "low_avg": 1.0,
    "quality_score": 0.0,
    "earnings_yoy": 0.0,
    "mktcap": 0.0,
    "candidate_score": 0.0,
}).keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-mktcap", type=float, default=DEFAULT_MIN_MKTCAP)
    parser.add_argument("--ratio-cut", type=float, default=DEFAULT_RATIO_CUT)
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY)
    parser.add_argument("--min-years", type=int, default=3)
    parser.add_argument(
        "--minimum-listing-days",
        type=int,
        default=DEFAULT_MINIMUM_LISTING_DAYS,
    )
    parser.add_argument(
        "--cache-directory",
        default=str(DEFAULT_CACHE_DIRECTORY),
    )
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument(
        "--industry-manifest", default=str(DEFAULT_INDUSTRY_MANIFEST),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / "candidates" / FACTOR
    build_pe_pb_candidates(
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        min_mktcap=args.min_mktcap,
        ratio_cut=args.ratio_cut,
        min_quality=args.min_quality,
        min_years=args.min_years,
        minimum_listing_days=args.minimum_listing_days,
        cache_directory=args.cache_directory,
        formula_history=args.formula_history,
        industry_manifest_path=args.industry_manifest,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
