# -*- coding: utf-8 -*-
"""Build the two daily fundamental sections from persistent point-in-time data."""
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from stock_research.core.as_of import audit_source, write_metadata
from stock_research.core.paths import PATHS
from stock_research.indicators.waves import infer_downtrend_recovery, level_price
from stock_research.indicators.technical_quant import technical_snapshot
from stock_research.pipelines.factor_selection import classify_method
from stock_research.strategies.fundamental_selection import (
    growth_risk,
    quality_detail,
    value_method_reason,
)


VALUE_CACHE_DIR = str(PATHS.cache / "q1_value")
KLINE_CACHE_DIR = str(PATHS.cache / "formula33_kline" / "akshare")
OUTPUT_DIR = str(PATHS.selection_exports)
UNIVERSE_PATH = str(PATHS.cache / "stock_universe.csv")
VALUE_MIN_MARKET_CAP = 100.0
VALUE_ROUTE_OVERRIDES = {
    "sz.002415": ("VALUE", "海康威视：大型安防公司，产业与财务可外推，按用户验收口径纳入"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成每日基本价值线和正常基本面两部分选股")
    parser.add_argument("--report-period", default="", help="财报期YYYY-MM-DD；默认使用缓存中的最新报告期")
    parser.add_argument(
        "--observation-date",
        required=True,
        help="观察日YYYY-MM-DD；只允许该日有有效成交K线的股票入选",
    )
    parser.add_argument("--value-ratio", type=float, default=1.08, help="价值线附近最高现价/价值线")
    parser.add_argument("--normal-top", type=int, default=30, help="正常基本面部分最多股票数")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def latest_report_period():
    suffixes = []
    for path in glob.glob(os.path.join(VALUE_CACHE_DIR, "*_????????.json")):
        suffix = os.path.splitext(path)[0].rsplit("_", 1)[-1]
        if suffix.isdigit() and len(suffix) == 8:
            suffixes.append(suffix)
    if not suffixes:
        raise SystemExit("没有基本价值线财务缓存")
    return pd.Timestamp(max(suffixes)).strftime("%Y-%m-%d")


def code_from_symbol(symbol):
    return ("sh." if str(symbol).startswith(("6", "9")) else "sz.") + str(symbol)


def load_kline(code, observation_date=None):
    path = os.path.join(KLINE_CACHE_DIR, f"{code.replace('.', '_')}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df = (
            df.dropna(subset=["date", "high", "low", "close"])
            .sort_values("date")
            .drop_duplicates("date")
        )
        df = df[
            (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["high"] >= df["low"])
        ]
        if observation_date is None:
            return df
        cutoff = pd.Timestamp(observation_date).normalize()
        df = df[df["date"].dt.normalize() <= cutoff]
        if df.empty:
            return pd.DataFrame()
        latest = df.iloc[-1]
        latest_date = pd.Timestamp(latest["date"]).normalize()
        latest_volume = pd.to_numeric(latest.get("volume"), errors="coerce")
        if latest_date != cutoff or pd.isna(latest_volume) or latest_volume <= 0:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def technical_fields(df):
    if df.empty:
        return {}
    close = df["close"]
    volume = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    current = float(close.iloc[-1])
    wave = infer_downtrend_recovery(df, lookback=500) or {}
    ma_values = {
        period: float(close.tail(period).mean()) if len(close) >= period else None
        for period in [5, 10, 20, 60, 120, 240]
    }
    fields = {
        "date": df.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "close": current,
        **{f"ma{period}": value for period, value in ma_values.items()},
        "volume_ratio_5_20": float(volume.tail(5).mean() / volume.tail(20).mean()) if len(volume) >= 20 and volume.tail(20).mean() else None,
        "wave_high": wave.get("downtrend_high"),
        "wave_low": wave.get("downtrend_low"),
        "uptrend_wave_low": wave.get("uptrend_low"),
        "uptrend_wave_level_50": wave.get("uptrend_level_50"),
        "close_wave_high": wave.get("close_wave_high"),
        "close_uptrend_wave_low": wave.get("close_uptrend_low"),
        "uptrend_close_level_50": wave.get("uptrend_close_level_50"),
        "close_pullback_low": wave.get("close_pullback_low"),
        "pullback_close_level_50": wave.get("pullback_close_level_50"),
        "trend_stage": wave.get("trend_stage"),
        "stage_level_50": wave.get("stage_level_50"),
        "stage_level_50_passed": wave.get("stage_level_50_passed"),
        "wave_pct": wave.get("recovery_pct"),
        "wave_breakout_pct": wave.get("breakout_above_high_pct"),
        "wave_level_50": wave.get("recovery_level_50"),
        "wave_level_625": wave.get("recovery_level_625"),
        "wave_level_75": level_price(wave["downtrend_low"], wave["downtrend_high"], 75) if wave else None,
        "wave_zone": wave.get("recovery_zone", "波段不足"),
        **technical_snapshot(df),
    }
    price_periods = []
    volume_periods = []
    combined_periods = []
    latest_volume = float(volume.tail(5).mean()) if len(volume) >= 5 else float(volume.iloc[-1])
    for period in [5, 10, 20, 60, 120, 240]:
        if len(df) < period + 1:
            continue
        price_ok = current > float(close.iloc[-period - 1])
        volume_ok = latest_volume > float(volume.iloc[-period - 1])
        if price_ok:
            price_periods.append(period)
        if volume_ok:
            volume_periods.append(period)
        if price_ok and volume_ok:
            combined_periods.append(period)
    long_periods = {60, 120, 240}
    fields["price_deduct_periods"] = "/".join(map(str, price_periods))
    fields["volume_deduct_periods"] = "/".join(map(str, volume_periods))
    fields["deduct_periods"] = "/".join(map(str, combined_periods))
    fields["long_price_deduct_count"] = len(long_periods.intersection(price_periods))
    fields["long_volume_deduct_count"] = len(long_periods.intersection(volume_periods))
    fields["long_deduct_ready"] = fields["long_price_deduct_count"] >= 2 and fields["long_volume_deduct_count"] >= 2
    long_mas = [ma_values[p] for p in [20, 60, 120, 240]]
    fields["full_bearish"] = all(value is not None for value in long_mas) and current < long_mas[0] < long_mas[1] < long_mas[2] < long_mas[3]
    return fields


def load_names():
    try:
        df = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
        return dict(zip(df["code"], df["code_name"]))
    except Exception:
        return {}


def latest_fundamental_snapshot(report_period):
    suffix = report_period.replace("-", "")
    full_paths = glob.glob(
        os.path.join(str(PATHS.cache / "fundamental_snapshots"), f"fundamental_snapshot_{suffix}_*.csv")
    )
    if full_paths:
        path = max(full_paths, key=os.path.getmtime)
        return pd.read_csv(path, dtype={"code": str}, low_memory=False), path
    return pd.DataFrame(), ""


def method_routes_from_snapshot(snapshot):
    routes = {}
    if not snapshot.empty:
        for _, row in snapshot.drop_duplicates("code").iterrows():
            code = str(row["code"])
            industry = row.get("industry", "")
            method = row.get("method")
            if not method or pd.isna(method):
                method = classify_method(industry) if industry else "UNKNOWN"
            routes[code] = {
                "method": method,
                "industry": industry,
                "theme": row.get("theme", ""),
                "reason": "按当日全市场截面行业规则",
            }
    for code, (method, reason) in VALUE_ROUTE_OVERRIDES.items():
        existing = routes.get(code, {})
        routes[code] = {**existing, "method": method, "reason": reason}
    return routes


def load_mainline_boards():
    path = str(PATHS.cache / "sector_mainline_constituents.csv")
    if not os.path.exists(path):
        return {}
    try:
        frame = pd.read_csv(path, dtype={"code": str})
        if frame.empty:
            return {}
        result = {}
        for code, group in frame.groupby("code"):
            result[str(code)] = "、".join(group.sort_values("board_rank")["board"].astype(str).drop_duplicates())
        return result
    except Exception:
        return {}


def value_rows(
    report_period,
    names,
    snapshot,
    max_ratio,
    method_routes,
    mainline_boards,
    observation_date,
):
    suffix = report_period.replace("-", "")
    industry_map = {}
    theme_map = {}
    if not snapshot.empty:
        industry_map = snapshot.drop_duplicates("code").set_index("code").get("industry", pd.Series(dtype=str)).to_dict()
        theme_map = snapshot.drop_duplicates("code").set_index("code").get("theme", pd.Series(dtype=str)).to_dict()
        method_map = snapshot.drop_duplicates("code").set_index("code").get("method", pd.Series(dtype=str)).to_dict()
    else:
        method_map = {}
    rows = []
    for path in glob.glob(os.path.join(VALUE_CACHE_DIR, f"*_{suffix}.json")):
        symbol = os.path.basename(path).split("_", 1)[0]
        code = code_from_symbol(symbol)
        route = method_routes.get(code, {})
        route_method = route.get("method") or method_map.get(code)
        if route_method != "VALUE":
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            continue
        df = load_kline(code, observation_date)
        tech = technical_fields(df)
        close = tech.get("close")
        value_line = pd.to_numeric(value.get("value_line"), errors="coerce")
        eps = pd.to_numeric(value.get("eps_excl"), errors="coerce")
        yoy = pd.to_numeric(value.get("yoy"), errors="coerce")
        quality = pd.to_numeric(value.get("quality_score"), errors="coerce")
        shares = pd.to_numeric(value.get("total_share"), errors="coerce")
        if any(pd.isna(v) for v in [close, value_line, eps, yoy, quality, shares]) or value_line <= 0 or eps <= 0:
            continue
        mktcap = close * shares / 1e8
        ratio = close / value_line
        # Cache membership means the stock passed the historical VALUE-method
        # industry routing. Growth, size and financial quality are rechecked.
        applicable = mktcap >= VALUE_MIN_MARKET_CAP and yoy >= 0 and quality >= 50
        if not applicable or ratio > max_ratio:
            continue
        industry = route.get("industry") or industry_map.get(code, "")
        method_reason = value_method_reason(industry, mktcap, eps, yoy)
        row = {
            "strategy_part": "1.基本价值线或附近",
            "code": code,
            "name": names.get(code, code),
            "industry": industry,
            "theme": route.get("theme") or theme_map.get(code, ""),
            "report_period": report_period,
            "value_applicable": True,
            "value_applicable_reason": method_reason,
            "value_line": value_line,
            "price_to_value": ratio,
            "quality_score": quality,
            "earnings_yoy": yoy,
            "mktcap": mktcap,
            "eps_excl": eps,
            "quality_reason": quality_detail(eps, yoy, quality),
            "mainline_boards": mainline_boards.get(code) or "未命中",
            "selection_reason": (
                f"基本价值线适用理由：{method_reason}；"
                f"现价/价值线={ratio:.3f}，{'价值线内' if ratio <= 1 else '价值线附近'}；"
                f"{quality_detail(eps, yoy, quality)}；"
                f"当前主流板块={mainline_boards.get(code) or '未命中'}{growth_risk(yoy)}"
            ),
            **tech,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def normal_rows(report_period, snapshot, names, mainline_boards, observation_date):
    if snapshot.empty:
        return pd.DataFrame()
    work = snapshot.drop_duplicates("code").copy()
    work["method"] = work["industry"].map(classify_method)
    method_names = {"VALUE": "基本价值线", "PE": "历史PE", "PB": "历史PB", "RIGHT": "右侧趋势"}
    work["method_name"] = work["method"].map(method_names)
    suffix = report_period.replace("-", "")
    for index, row in work.iterrows():
        symbol = str(row["code"]).split(".")[-1]
        path = os.path.join(VALUE_CACHE_DIR, f"{symbol}_{suffix}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            for source_col, target_col in [
                ("value_line", "value_line"),
                ("quality_score", "quality_score"),
                ("yoy", "earnings_yoy"),
                ("eps_excl", "eps_excl"),
            ]:
                if value.get(source_col) is not None:
                    work.at[index, target_col] = value[source_col]
            if value.get("total_share") is not None:
                work.at[index, "total_share"] = value["total_share"]
        except (OSError, ValueError, TypeError):
            continue
    for col in ["quality_score", "liquidity_score", "mktcap", "earnings_yoy", "total_score", "price_to_value"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work = work[
        (work["quality_score"] >= 70)
        & (work["liquidity_score"] >= 55)
        & (work["mktcap"] >= 100)
        & (work["earnings_yoy"] >= 0.10)
    ].copy()
    rows = []
    for _, source in work.iterrows():
        code = str(source["code"])
        fixed_pool_member = bool(source.get("pool_member", False))
        tech = technical_fields(load_kline(code, observation_date))
        if not tech:
            continue
        row = source.to_dict()
        row.update(tech)
        if tech.get("close") is not None and pd.notna(source.get("value_line")) and source.get("value_line", 0) > 0:
            row["price_to_value"] = tech["close"] / source["value_line"]
        if tech.get("close") is not None and pd.notna(source.get("total_share")):
            row["mktcap"] = tech["close"] * source["total_share"] / 1e8
        row.update({
            "strategy_part": "2.正常基本面选股",
            "name": names.get(code, source.get("name", code)),
            "report_period": report_period,
            "value_applicable": source.get("method") == "VALUE",
            "mainline_boards": mainline_boards.get(code) or "未命中",
        })
        if source.get("method") != "VALUE":
            row["value_line"] = np.nan
            row["price_to_value"] = np.nan
        eps = pd.to_numeric(row.get("eps_excl"), errors="coerce")
        yoy = pd.to_numeric(row.get("earnings_yoy"), errors="coerce")
        quality = pd.to_numeric(row.get("quality_score"), errors="coerce")
        is_mainline = (mainline_boards.get(code) or "") != ""
        wave_pct = pd.to_numeric(row.get("wave_pct"), errors="coerce")
        right_confirmed = pd.notna(wave_pct) and wave_pct >= 50
        long_deduct_ready = bool(row.get("long_deduct_ready"))
        full_bearish = bool(row.get("full_bearish"))
        if full_bearish and fixed_pool_member:
            strategy_layer = "固定基本面池·技术面暂不满足"
            structure_score = 0
        elif full_bearish:
            continue
        elif is_mainline and right_confirmed:
            strategy_layer = "主流板块基本面优秀·右侧确认"
            structure_score = 15
        elif is_mainline and long_deduct_ready:
            strategy_layer = "主流板块基本面优秀·右侧酝酿"
            structure_score = 10
        elif right_confirmed:
            strategy_layer = "非主流板块基本面优秀·右侧观察"
            structure_score = 7
        elif long_deduct_ready and quality >= 85 and yoy >= 0.25:
            strategy_layer = "非主流板块基本面优秀·长期结构观察"
            structure_score = 4
        elif fixed_pool_member:
            strategy_layer = "固定基本面池·等待技术确认"
            structure_score = 0
        else:
            continue
        valuation_score = 0.0
        current_ptv = pd.to_numeric(row.get("price_to_value"), errors="coerce")
        if source.get("method") == "VALUE" and pd.notna(current_ptv):
            valuation_score = max(0.0, 5.0 * (1 - min(current_ptv, 3.0) / 3.0))
        performance_score = quality * 0.45 + min(max(yoy, 0), 1) * 25 + pd.to_numeric(row.get("liquidity_score"), errors="coerce") * 0.10
        row["fundamental_score"] = round(
            performance_score + (15 if is_mainline else 0) + structure_score + valuation_score,
            2,
        )
        row["strategy_layer"] = strategy_layer
        row["is_mainline"] = is_mainline
        row["right_confirmed"] = right_confirmed
        row["signal_eligible"] = strategy_layer not in {
            "固定基本面池·等待技术确认",
            "固定基本面池·技术面暂不满足",
        }
        if pd.notna(eps) and pd.notna(yoy) and pd.notna(quality):
            row["quality_reason"] = quality_detail(eps, yoy, quality)
        else:
            row["quality_reason"] = f"质量{quality:.1f}，原始财务分项缓存不足" if pd.notna(quality) else "质量数据不足"
        row["selection_reason"] = (
            f"{strategy_layer}；财报期{report_period}；{row['quality_reason']}；"
            f"流动性{pd.to_numeric(row.get('liquidity_score'), errors='coerce'):.1f}，"
            f"市值{pd.to_numeric(row.get('mktcap'), errors='coerce'):.1f}亿元；"
            f"主流板块：{mainline_boards.get(code) or '未命中'}；"
            f"长期价格扣抵通过{int(row.get('long_price_deduct_count', 0))}/3，"
            f"长期均量扣抵通过{int(row.get('long_volume_deduct_count', 0))}/3，"
            f"未形成完整空头排列；右侧状态：{row.get('wave_zone', '波段不足')}"
            f"{growth_risk(yoy)}"
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    layer_order = {
        "主流板块基本面优秀·右侧确认": 0,
        "主流板块基本面优秀·右侧酝酿": 1,
        "非主流板块基本面优秀·右侧观察": 2,
        "非主流板块基本面优秀·长期结构观察": 3,
        "固定基本面池·等待技术确认": 4,
        "固定基本面池·技术面暂不满足": 5,
    }
    result["layer_order"] = result["strategy_layer"].map(layer_order).fillna(9)
    return result.sort_values(
        ["layer_order", "fundamental_score", "quality_score", "code"],
        ascending=[True, False, False, True],
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        observation_date = pd.Timestamp(args.observation_date).normalize()
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"无效 observation_date: {args.observation_date!r}") from exc
    if pd.isna(observation_date):
        raise SystemExit(f"无效 observation_date: {args.observation_date!r}")
    observation_date_text = observation_date.strftime("%Y-%m-%d")
    report_period = args.report_period or latest_report_period()
    names = load_names()
    mainline_boards = load_mainline_boards()
    snapshot, snapshot_path = latest_fundamental_snapshot(report_period)
    if snapshot.empty:
        raise SystemExit(f"缺少财报期 {report_period} 的动态基本面截面")
    source_status, source_issues, source_metadata = audit_source(snapshot_path)
    if source_status == "unsafe":
        raise SystemExit(f"动态全市场截面覆盖率不足或时点不安全: {source_issues or source_metadata}")
    pool_metadata = {
        "point_in_time_status": source_status,
        "point_in_time_note": "dynamic selection; financial revision history is unavailable",
        "formation_date": None,
    }
    method_routes = method_routes_from_snapshot(snapshot)
    values = value_rows(
        report_period,
        names,
        snapshot,
        args.value_ratio,
        method_routes,
        mainline_boards,
        observation_date_text,
    )
    normal = normal_rows(
        report_period,
        snapshot,
        names,
        mainline_boards,
        observation_date_text,
    )
    normal = normal.head(max(1, args.normal_top))
    combined = pd.concat([values, normal], ignore_index=True, sort=False)
    if combined.empty:
        raise SystemExit(f"观察日 {observation_date_text} 没有生成每日基本面候选")
    result_dates = set(combined["date"].dropna().astype(str))
    if result_dates != {observation_date_text}:
        raise SystemExit(
            "基本面候选包含非观察日行情: "
            f"expected={observation_date_text} actual={sorted(result_dates)}"
        )
    report_date = observation_date_text
    stamp = datetime.now().strftime("%H%M%S")
    output = args.output or os.path.join(OUTPUT_DIR, f"daily_fundamental_selection_{report_date}_{stamp}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    write_metadata(output, {
        "kind": "daily_fundamental_selection",
        "point_in_time_status": pool_metadata.get("point_in_time_status", "warning"),
        "report_period": report_period,
        "selection_mode": "dynamic",
        "source_snapshot": snapshot_path,
        "observation_date": observation_date_text,
        "daily_technical_cutoff": report_date,
        "dynamic_signal_members": int(normal.get("signal_eligible", pd.Series(dtype=bool)).fillna(False).sum()) if not normal.empty else 0,
    })
    print(f"每日基本面文件: {output}")
    print(f"财报期={report_period} 价值线或附近={len(values)} 正常基本面={len(normal)}")
    print(f"截面来源={snapshot_path or '缺失'}")
    print(f"选股模式=dynamic 时点状态={pool_metadata.get('point_in_time_status', 'unknown')}")
    for code in ["sz.002415", "sz.002236"]:
        hit = values[values["code"] == code]
        if not hit.empty:
            row = hit.iloc[0]
            print(f"验收 {row['name']} {code}: 现价/价值线={row['price_to_value']:.3f}, {row['selection_reason']}")
