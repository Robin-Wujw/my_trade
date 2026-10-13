"""PE/PB 低估候选池 alpha 诊断（零成本、月度观察日采样）。

判定口径沿用 factor_selection.get_pe_pb_metrics 的"历史低估均值"：
- 低估均值 = 观察日前已结束的完整年度中，每只股票年度最低估值的集合，
  remove_outliers（保留 [0.5*median, 2.0*median]）后取均值。
- ratio = 当前估值 / 低估均值；低估候选 = ratio <= 阈值（默认 1.0）。
- PE 行业用 pe_ttm，PB 行业用 pb（行业路由与 factor_selection 一致）。

严格时点：候选只用观察日及以前的数据（年度 min 只取观察日前完整年度）。
收益从观察日收盘锚点起算（T+20/60/120 收盘），零成本、不模拟执行摩擦，
衡量的是候选池的选择 alpha。对照组 = 同观察日同 PE/PB 行业但 ratio>阈值的股票
（等权），回答"低估的股票是否跑赢不低估的"。

用法:
    python -m scripts.diagnose_pe_pb_pool --start 2021-01-01 --end 2026-07-21
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("var/data/my_trade.sqlite3")
DEFAULT_MIN_MKTCAP = 100.0  # 亿元，与 VALUE_MIN_MKTCAP 一致
DEFAULT_RATIO_CUT = 1.0
DEFAULT_MIN_YEARS = 3
HORIZONS = [20, 60, 120]

# 与 stock_research/pipelines/factor_selection.py 的 PE_KEYWORDS / PB_KEYWORDS 保持一致
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


def load_industry_map(db: Path) -> dict[str, dict]:
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    cur.execute(
        "SELECT ts_code, payload_json FROM raw_tushare_dataset_rows "
        "WHERE dataset='stock_basic'"
    )
    out = {}
    for ts_code, payload in cur.fetchall():
        j = json.loads(payload)
        out[ts_code] = {
            "industry": j.get("industry") or "",
            "name": j.get("name") or ts_code,
        }
    con.close()
    return out


def load_observation_days(db: Path, start: str, end: str) -> list[str]:
    """每月最后一个交易日作为观察日（该日收盘形成候选，T+1 可执行）。"""
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    cur.execute(
        "SELECT MAX(trade_date) FROM raw_tushare_dataset_rows "
        "WHERE dataset='daily_basic' AND trade_date >= ? AND trade_date <= ? "
        "GROUP BY substr(trade_date, 1, 7) ORDER BY 1",
        [start, end],
    )
    days = [r[0] for r in cur.fetchall()]
    con.close()
    return days


def load_daily_basic_at(db: Path, days: list[str]) -> pd.DataFrame:
    con = sqlite3.connect(str(db))
    placeholders = ",".join("?" * len(days))
    sql = (
        "SELECT ts_code, trade_date, "
        "json_extract(payload_json, '$.pe') AS pe, "
        "json_extract(payload_json, '$.pb') AS pb, "
        "json_extract(payload_json, '$.total_mv') AS total_mv, "
        "json_extract(payload_json, '$.close') AS close "
        f"FROM raw_tushare_dataset_rows WHERE dataset='daily_basic' "
        f"AND trade_date IN ({placeholders})"
    )
    frame = pd.read_sql_query(sql, con, params=days)
    con.close()
    for col in ("pe", "pb", "total_mv", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    # total_mv 单位是万元，换算为亿元
    frame["total_mv_yi"] = frame["total_mv"] / 10000.0
    return frame


def load_yearly_min(db: Path, end: str) -> pd.DataFrame:
    """每 (ts_code, 日历年) 的年度最低 pe（静态 PE）/ pb。

    注意：tushare 的 pe_ttm 字段 2022 年起才有，历史不完整；静态 pe 覆盖完整，
    故 PE 行业统一用静态 pe 口径。
    """
    con = sqlite3.connect(str(db))
    sql = (
        "SELECT ts_code, substr(trade_date, 1, 4) AS yr, "
        "MIN(json_extract(payload_json, '$.pe')) AS min_pe, "
        "MIN(json_extract(payload_json, '$.pb')) AS min_pb "
        "FROM raw_tushare_dataset_rows WHERE dataset='daily_basic' "
        "AND trade_date < ? GROUP BY ts_code, yr"
    )
    frame = pd.read_sql_query(sql, con, params=[end])
    con.close()
    for col in ("min_pe", "min_pb"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_st_map(db: Path) -> set[tuple[str, str]]:
    """(ts_code, 交易日 'YYYY-MM-DD') -> 该日处于风险警示（ST/*ST）。"""
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


def build_low_avg(yearly: pd.DataFrame, obs_year: int, field: str, min_years: int = DEFAULT_MIN_YEARS):
    """给定观察日所在年份，返回 {ts_code: 低估均值}。

    只使用 obs_year 之前已结束的完整年度；至少 min_years 个完整年度。
    """
    full = yearly[yearly["yr"].astype(int) < obs_year]
    low_avg = {}
    for ts_code, group in full.groupby("ts_code"):
        vals = group[field].dropna().tolist()
        if len(vals) < min_years:
            continue
        filtered = remove_outliers(vals)
        low_avg[ts_code] = float(np.mean(filtered))
    return low_avg


def load_qfq_close(db: Path, codes, start: str, end: str) -> pd.DataFrame:
    """前复权收盘（close * adj_factor）长格式，限指定股票。"""
    if not codes:
        return pd.DataFrame(columns=["ts_code", "trade_date", "qfq"])
    con = sqlite3.connect(str(db))
    ph = ",".join("?" * len(codes))
    kline = pd.read_sql_query(
        "SELECT ts_code, trade_date, "
        "json_extract(payload_json, '$.close') AS close "
        f"FROM raw_tushare_dataset_rows WHERE dataset='daily_kline' "
        f"AND ts_code IN ({ph}) AND trade_date >= ? AND trade_date <= ?",
        con,
        params=list(codes) + [start, end],
    )
    adj = pd.read_sql_query(
        "SELECT ts_code, trade_date, "
        "json_extract(payload_json, '$.adj_factor') AS af "
        f"FROM raw_tushare_dataset_rows WHERE dataset='adj_factor' "
        f"AND ts_code IN ({ph}) AND trade_date >= ? AND trade_date <= ?",
        con,
        params=list(codes) + [start, end],
    )
    con.close()
    for col in ("close",):
        kline[col] = pd.to_numeric(kline[col], errors="coerce")
    adj["af"] = pd.to_numeric(adj["af"], errors="coerce")
    frame = kline.merge(adj, on=["ts_code", "trade_date"], how="left")
    frame["qfq"] = frame["close"] * frame["af"]
    frame = frame.dropna(subset=["qfq"]).sort_values(["ts_code", "trade_date"])
    return frame.reset_index(drop=True)


def compute_forward_returns(qfq: pd.DataFrame, obs_day: str, horizons) -> dict:
    """观察日收盘锚点 → 后续第 k 个交易日收盘收益率（%）。

    返回 {ts_code: {k: ret_pct}}。锚点行即 obs_day 当日的 qfq。
    """
    frame = qfq[qfq["trade_date"] >= obs_day].copy()
    frame = frame.sort_values(["ts_code", "trade_date"])
    out = {}
    for ts_code, group in frame.groupby("ts_code"):
        closes = group["qfq"].to_numpy()
        if len(closes) < 2:
            continue
        anchor = closes[0]
        if anchor <= 0:
            continue
        rec = {}
        for k in horizons:
            if len(closes) > k:
                rec[k] = (closes[k] / anchor - 1.0) * 100.0
        if rec:
            out[ts_code] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-07-21")
    parser.add_argument("--min-mktcap", type=float, default=DEFAULT_MIN_MKTCAP)
    parser.add_argument("--ratio-cut", type=float, default=DEFAULT_RATIO_CUT)
    parser.add_argument("--min-years", type=int, default=DEFAULT_MIN_YEARS)
    args = parser.parse_args()

    print("加载股票基础信息 ...")
    industry_map = load_industry_map(DB_PATH)
    print(f"  {len(industry_map)} 只股票")

    print("加载观察日 ...")
    obs_days = load_observation_days(DB_PATH, args.start, args.end)
    print(f"  {len(obs_days)} 个观察日: {obs_days[0]} .. {obs_days[-1]}")

    print("加载观察日 daily_basic ...")
    daily = load_daily_basic_at(DB_PATH, obs_days)
    daily["method"] = daily["ts_code"].map(
        lambda c: classify_method(industry_map.get(c, {}).get("industry"))
    )
    daily["industry"] = daily["ts_code"].map(
        lambda c: industry_map.get(c, {}).get("industry")
    )
    print(f"  {len(daily)} 行 daily_basic")

    print("加载年度最低估值 ...")
    yearly = load_yearly_min(DB_PATH, args.end)
    print(f"  {len(yearly)} 行年度聚合")

    # 路由后的 PE/PB 行
    routed = daily[daily["method"].isin(["PE", "PB"])].copy()
    print(f"  PE/PB 行业股票行: {len(routed)}")

    # ST 过滤（观察日处于风险警示则排除）
    st_map = load_st_map(DB_PATH)

    # 构建候选（按观察日）
    cand_rows = []
    bench_rows = []  # 对照组：同观察日同行业池 ratio > 阈值
    for obs_day in obs_days:
        obs_year = int(obs_day[:4])
        sub = routed[routed["trade_date"] == obs_day].copy()
        if sub.empty:
            continue
        low_pe = build_low_avg(yearly, obs_year, "min_pe", args.min_years)
        low_pb = build_low_avg(yearly, obs_year, "min_pb", args.min_years)
        for _, row in sub.iterrows():
            if (row["ts_code"], obs_day) in st_map:
                continue  # ST/*ST 剔除
            if row["method"] == "PE":
                cur = row["pe"]
                low = low_pe.get(row["ts_code"])
            else:
                cur = row["pb"]
                low = low_pb.get(row["ts_code"])
            if cur is None or low is None or not np.isfinite(cur) or cur <= 0 or low <= 0:
                continue
            if pd.isna(row["total_mv_yi"]) or row["total_mv_yi"] < args.min_mktcap:
                continue
            ratio = cur / low
            rec = {
                "obs_date": obs_day,
                "code": row["ts_code"],
                "method": row["method"],
                "industry": row["industry"],
                "value": cur,
                "low_avg": low,
                "ratio": ratio,
                "total_mv": row["total_mv_yi"],
            }
            if ratio <= args.ratio_cut:
                cand_rows.append(rec)
            else:
                bench_rows.append(rec)

    candidates = pd.DataFrame(cand_rows)
    benchmark = pd.DataFrame(bench_rows)
    print("\n=== 候选池统计 ===")
    print(f"候选行数: {len(candidates)}  覆盖股票: {candidates['code'].nunique() if not candidates.empty else 0}")
    if not candidates.empty:
        print(f"观察日覆盖: {candidates['obs_date'].nunique()} 天  平均每天 {len(candidates)/candidates['obs_date'].nunique():.1f} 只")
        print(f"PE 候选: {(candidates['method']=='PE').sum()}  PB 候选: {(candidates['method']=='PB').sum()}")
        print(f"ratio 中位: {candidates['ratio'].median():.3f}  value 中位: {candidates['value'].median():.2f}")
        top_codes = candidates['code'].value_counts().head(10)
        print("最常入选股票:")
        for code, n in top_codes.items():
            info = industry_map.get(code, {})
            print(f"  {code} {info.get('name','')[:8]}({info.get('industry','')[:10]}): {n} 次")

    # 收益计算：候选 + 对照组
    all_codes = sorted(set(candidates["code"]) | set(benchmark["code"]))
    if not all_codes:
        print("无候选，退出")
        return
    print(f"\n加载 {len(all_codes)} 只股票的前复权行情 ...")
    qfq = load_qfq_close(DB_PATH, all_codes, args.start, args.end)
    print(f"  {len(qfq)} 行 kline")

    print("计算后续收益 ...")

    for obs_day in obs_days:
        codes_day = candidates[candidates["obs_date"] == obs_day]["code"].tolist()
        bench_day = benchmark[benchmark["obs_date"] == obs_day]["code"].tolist()
        if not codes_day and not bench_day:
            continue
        need = sorted(set(codes_day) | set(bench_day))
        rets = compute_forward_returns(qfq[qfq["ts_code"].isin(need)], obs_day, HORIZONS)
        for code, rec in rets.items():
            for k, ret in rec.items():
                if code in codes_day:
                    candidates.loc[
                        (candidates["obs_date"] == obs_day) & (candidates["code"] == code),
                        f"ret_{k}d",
                    ] = ret
                else:
                    benchmark.loc[
                        (benchmark["obs_date"] == obs_day) & (benchmark["code"] == code),
                        f"ret_{k}d",
                    ] = ret

    # ---- 汇总 ----
    print("\n=== 候选池 vs 对照组 后续收益 (%) ===")
    for k in HORIZONS:
        col = f"ret_{k}d"
        if col not in candidates or candidates[col].isna().all():
            continue
        c = candidates[col].dropna()
        b = benchmark[col].dropna() if col in benchmark else pd.Series(dtype=float)
        print(f"  {k:>3}d  候选: n={len(c):>5} 均值={c.mean():+6.2f} 中位={c.median():+6.2f} 胜率={ (c>0).mean()*100:5.1f}%   "
              f"对照: n={len(b):>5} 均值={b.mean():+6.2f} 中位={b.median():+6.2f}  差值={ (c.mean()-b.mean() if len(b) else 0):+6.2f}")

    # 分年
    print("\n=== 分年 (20d/120d 均值) ===")
    for yr, group in candidates.groupby(candidates["obs_date"].str[:4]):
        row = {"年": yr, "n": len(group)}
        for k in (20, 120):
            col = f"ret_{k}d"
            if col in group and group[col].notna().any():
                row[f"{k}d"] = group[col].mean()
        print("  ", {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()})

    # 分行业 top
    print("\n=== 分行业 (120d 均值, top 10) ===")
    if "ret_120d" in candidates:
        ind = candidates.groupby("industry").agg(
            n=("code", "size"),
            ret_120=("ret_120d", "mean"),
            ret_20=("ret_20d", "mean"),
        ).sort_values("ret_120", ascending=False)
        print(ind.head(10).to_string())

    # PE vs PB
    print("\n=== PE vs PB 对比 ===")
    for m in ["PE", "PB"]:
        sub = candidates[candidates["method"] == m]
        if sub.empty:
            continue
        for k in HORIZONS:
            col = f"ret_{k}d"
            if col in sub and sub[col].notna().any():
                s = sub[col].dropna()
                print(f"  {m:>2} {k:>3}d: n={len(s):>5} 均值={s.mean():+6.2f} 中位={s.median():+6.2f} 胜率={(s>0).mean()*100:5.1f}%")

    out = Path("var/tmp") / "pe_pb_candidates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not candidates.empty:
        candidates.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n候选明细: {out}")


if __name__ == "__main__":
    main()
