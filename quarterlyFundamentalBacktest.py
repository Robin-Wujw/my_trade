# -*- coding: utf-8 -*-
"""Rebuild quarterly fundamental selections at the disclosure deadline."""
import argparse
import glob
import json
import os

import pandas as pd

from dailyFundamentalSelect import (
    BACKTEST_DIR,
    KLINE_CACHE_DIR,
    VALUE_CACHE_DIR,
    classify_method,
    code_from_symbol,
    latest_fundamental_snapshot,
    load_method_routes,
    load_names,
    quality_detail,
    technical_fields,
)
from wave_utils import infer_downtrend_recovery, level_price


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-period", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--normal-top", type=int, default=30)
    parser.add_argument("--value-ratio", type=float, default=1.08)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default="", help="指定历史截面候选CSV")
    return parser.parse_args()


def history(code, end_date):
    path = os.path.join(KLINE_CACHE_DIR, f"{code.replace('.', '_')}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ["high", "low", "close"]:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    return frame.dropna(subset=["date", "high", "low", "close"]).sort_values("date").query("date <= @end_date")


def market_result(code, buy_date, end_date):
    frame = history(code, end_date)
    if frame.empty:
        return None
    buy = frame[frame["date"] >= pd.Timestamp(buy_date)].head(1)
    end = frame[frame["date"] <= pd.Timestamp(end_date)].tail(1)
    if buy.empty or end.empty:
        return None
    buy_row, end_row = buy.iloc[0], end.iloc[0]
    pre = frame[frame["date"] <= buy_row["date"]]
    wave = infer_downtrend_recovery(pre, lookback=240) or {}
    technical = technical_fields(pre)
    return {
        "buy_trade_date": buy_row["date"].strftime("%Y-%m-%d"),
        "buy_close": float(buy_row["close"]),
        "end_trade_date": end_row["date"].strftime("%Y-%m-%d"),
        "end_close": float(end_row["close"]),
        "return": float(end_row["close"] / buy_row["close"] - 1),
        "wave_level_50": wave.get("recovery_level_50"),
        "wave_level_625": wave.get("recovery_level_625"),
        "wave_level_75": level_price(wave["downtrend_low"], wave["downtrend_high"], 75) if wave else None,
        "wave_pct_at_buy": wave.get("recovery_pct"),
        "wave_zone_at_buy": wave.get("recovery_zone", "波段不足"),
        "long_price_deduct_count": technical.get("long_price_deduct_count", 0),
        "long_volume_deduct_count": technical.get("long_volume_deduct_count", 0),
        "long_deduct_ready": technical.get("long_deduct_ready", False),
        "full_bearish": technical.get("full_bearish", False),
    }


def historical_mainline_map(buy_date):
    stamp = pd.Timestamp(buy_date).strftime("%Y%m%d")
    path = os.path.join(os.path.dirname(VALUE_CACHE_DIR), f"sector_mainline_constituents_{stamp}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"缺少历史主流板块映射: {path}")
    frame = pd.read_csv(path, dtype={"code": str})
    return {
        str(code): "、".join(group.sort_values("board_rank")["board"].astype(str).drop_duplicates())
        for code, group in frame.groupby("code")
    }


def load_value_cache(report_period):
    suffix = report_period.replace("-", "")
    rows = []
    for path in glob.glob(os.path.join(VALUE_CACHE_DIR, f"*_{suffix}.json")):
        symbol = os.path.basename(path).split("_", 1)[0]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            value["code"] = code_from_symbol(symbol)
            rows.append(value)
        except (OSError, ValueError):
            continue
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    names = load_names()
    routes = load_method_routes()
    mainline_map = historical_mainline_map(args.buy_date)
    values = load_value_cache(args.report_period)
    if args.snapshot:
        snapshot_path = args.snapshot
        snapshot = pd.read_csv(snapshot_path, dtype={"code": str}, low_memory=False)
    else:
        snapshot, snapshot_path = latest_fundamental_snapshot(args.report_period)

    value_rows = []
    for _, row in values.iterrows():
        code = row["code"]
        route = routes.get(code, {})
        ratio = pd.to_numeric(row.get("price_to_value"), errors="coerce")
        quality = pd.to_numeric(row.get("quality_score"), errors="coerce")
        yoy = pd.to_numeric(row.get("yoy"), errors="coerce")
        eps = pd.to_numeric(row.get("eps_excl"), errors="coerce")
        mktcap = pd.to_numeric(row.get("mktcap"), errors="coerce")
        if route.get("method") != "VALUE" or any(pd.isna(v) for v in [ratio, quality, yoy, eps, mktcap]):
            continue
        if ratio > args.value_ratio or quality < 50 or yoy < 0 or eps <= 0 or mktcap < 100:
            continue
        market = market_result(code, args.buy_date, args.end_date)
        if not market:
            continue
        value_rows.append({
            "strategy_part": "1.基本价值线或附近",
            "code": code,
            "name": names.get(code, code),
            "industry": route.get("industry", ""),
            "quality_score": quality,
            "earnings_yoy": yoy,
            "mktcap": mktcap,
            "value_line": row.get("value_line"),
            "price_to_value_at_buy": ratio,
            "selection_reason": f"价值线比值{ratio:.3f}；{quality_detail(eps, yoy, quality)}",
            **market,
        })

    normal = snapshot.drop_duplicates("code").copy()
    normal["method"] = normal["industry"].map(classify_method)
    cache_map = values.drop_duplicates("code").set_index("code").to_dict("index") if not values.empty else {}
    for index, row in normal.iterrows():
        cached = cache_map.get(str(row["code"]), {})
        for source, target in [("quality_score", "quality_score"), ("yoy", "earnings_yoy"), ("price_to_value", "price_to_value")]:
            if cached.get(source) is not None:
                normal.at[index, target] = cached[source]
    for col in ["quality_score", "liquidity_score", "mktcap", "earnings_yoy", "total_score", "price_to_value"]:
        normal[col] = pd.to_numeric(normal.get(col), errors="coerce")
    normal = normal[
        (normal["quality_score"] >= 70)
        & (normal["liquidity_score"] >= 55)
        & (normal["mktcap"] >= 100)
        & (normal["earnings_yoy"] >= 0.10)
    ].copy()
    normal_rows = []
    for _, row in normal.iterrows():
        market = market_result(str(row["code"]), args.buy_date, args.end_date)
        if not market:
            continue
        if bool(market.get("full_bearish")):
            continue
        code = str(row["code"])
        boards = mainline_map.get(code, "")
        is_mainline = bool(boards)
        wave_pct = pd.to_numeric(market.get("wave_pct_at_buy"), errors="coerce")
        right_confirmed = pd.notna(wave_pct) and wave_pct >= 50
        long_ready = bool(market.get("long_deduct_ready"))
        quality = float(row["quality_score"])
        growth = float(row["earnings_yoy"])
        if is_mainline and right_confirmed:
            layer, structure_score = "主流板块基本面优秀·右侧确认", 15
        elif is_mainline and long_ready:
            layer, structure_score = "主流板块基本面优秀·右侧酝酿", 10
        elif right_confirmed:
            layer, structure_score = "非主流板块基本面优秀·右侧观察", 7
        elif long_ready and quality >= 85 and growth >= 0.25:
            layer, structure_score = "非主流板块基本面优秀·长期结构观察", 4
        else:
            continue
        ptv = pd.to_numeric(row.get("price_to_value"), errors="coerce")
        valuation_score = max(0.0, 5.0 * (1 - min(ptv, 3.0) / 3.0)) if row.get("method") == "VALUE" and pd.notna(ptv) else 0.0
        fundamental_score = (
            quality * 0.45
            + min(max(growth, 0), 1) * 25
            + float(row["liquidity_score"]) * 0.10
            + (15 if is_mainline else 0)
            + structure_score
            + valuation_score
        )
        normal_rows.append({
            **row.to_dict(),
            "strategy_part": "2.正常基本面选股",
            "strategy_layer": layer,
            "mainline_boards": boards or "未命中",
            "fundamental_score": round(fundamental_score, 2),
            **market,
        })

    layer_order = {
        "主流板块基本面优秀·右侧确认": 0,
        "主流板块基本面优秀·右侧酝酿": 1,
        "非主流板块基本面优秀·右侧观察": 2,
        "非主流板块基本面优秀·长期结构观察": 3,
    }
    normal_rows.sort(key=lambda row: (layer_order.get(row["strategy_layer"], 9), -row["fundamental_score"], row["code"]))
    normal_rows = normal_rows[: args.normal_top]

    result = pd.concat([pd.DataFrame(value_rows), pd.DataFrame(normal_rows)], ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"source={snapshot_path}")
    for part, group in result.groupby("strategy_part"):
        returns = pd.to_numeric(group["return"], errors="coerce").dropna()
        print(f"{part}: n={len(group)} mean={returns.mean():.2%} median={returns.median():.2%} win={(returns > 0).mean():.2%}")
        print(group.sort_values("return", ascending=False)[["code", "name", "buy_close", "end_close", "return", "wave_zone_at_buy"]].head(30).to_string(index=False))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
