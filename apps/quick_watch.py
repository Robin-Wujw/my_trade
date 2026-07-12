"""CLI for fast watch-list analysis and a standalone PushPlus reminder."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import os

import pandas as pd

from stock_research.api.pushplus import send_pushplus
from stock_research.core.paths import PATHS
from stock_research.reporting.quick_watch import (
    analyze_watch_stock,
    load_or_refresh_watch_kline,
    load_watch_stocks,
)
from stock_research.reporting.trade_reminders import (
    build_trade_reminders,
    load_trade_plans,
)


def _num(value):
    return "-" if value is None or pd.isna(value) else f"{float(value):.2f}"


def _refresh_market_context(loaded):
    dates = [
        pd.to_datetime(status.get("latest_date"), errors="coerce")
        for _, status in loaded.values()
        if status.get("fresh")
    ]
    dates = [date.normalize() for date in dates if pd.notna(date)]
    if not dates:
        raise RuntimeError("没有新鲜个股行情，无法确定主流板块观察日")
    observation = min(dates)
    target = PATHS.cache / f"sector_mainline_constituents_{observation:%Y%m%d}.csv"
    if target.exists():
        return observation.strftime("%Y-%m-%d")
    from stock_research.pipelines import sector_watch

    sector_watch.main([
        "--as-of-date", observation.strftime("%Y-%m-%d"),
        "--days", "80", "--top", "30", "--workers", "8",
        "--allow-missing-limit-up",
    ])
    if not target.exists():
        raise RuntimeError(f"主流板块快照补充失败: {observation:%Y-%m-%d}")
    return observation.strftime("%Y-%m-%d")


def build_quick_watch(watch_file, plan_file, *, refresh_market_context=False):
    stocks = load_watch_stocks(watch_file)
    plans = load_trade_plans(plan_file)
    by_code = {str(item["code"]): item for item in stocks}
    for code, plan in plans.get("plans", {}).items():
        by_code.setdefault(
            str(code),
            {"code": str(code), "name": plan.get("name", code), "note": "显式持仓计划"},
        )
    stocks = list(by_code.values())
    identity_frame = pd.DataFrame(stocks)
    observation = datetime.now().strftime("%Y-%m-%d")
    loaded = {
        item["code"]: load_or_refresh_watch_kline(item["code"])
        for item in stocks
    }
    context_date = _refresh_market_context(loaded) if refresh_market_context else None
    fresh_frames = {
        code: frame for code, (frame, status) in loaded.items() if status.get("fresh")
    }
    reminders = build_trade_reminders(
        identity_frame, observation,
        lambda code, _date: fresh_frames.get(code, pd.DataFrame()),
        plans,
    )
    analyses = [
        analyze_watch_stock(
            item,
            loaded[item["code"]][0],
            reminders,
            loaded[item["code"]][1],
        )
        for item in stocks
    ]
    parts = ["<h1>持仓与观察股快速分析</h1>"]
    if context_date:
        parts.append(f"<p>个股行情与主流板块数据门禁已通过：{context_date}</p>")
    for item in analyses:
        parts.append(
            f"<h2>{html.escape(str(item.get('name') or item.get('code')))} "
            f"{html.escape(str(item.get('code')))}</h2>"
        )
        if not item.get("available"):
            parts.append(f"<p>{html.escape(item['opinion'])}</p>")
            continue
        parts.append(
            f"<p>数据日{item['date']}，现价<b>{_num(item['close'])}</b>；"
            f"5/10/20/60日均线 {_num(item['ma5'])}/{_num(item['ma10'])}/{_num(item['ma20'])}/{_num(item['ma60'])}。<br>"
            f"量能{'达标' if item['volume_ready'] else '未达标'}。<br>"
            f"<b>意见：</b>{html.escape(item['opinion'])}</p>"
        )
    return analyses, "".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="快速分析持仓和观察股票")
    parser.add_argument(
        "--watch-file",
        default=str(PATHS.project_root / "config" / "watch_stocks.json"),
    )
    parser.add_argument(
        "--plan-file",
        default=str(PATHS.project_root / "config" / "trade_plans.json"),
    )
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)
    analyses, content = build_quick_watch(
        args.watch_file, args.plan_file, refresh_market_context=True,
    )
    os.makedirs(PATHS.report_exports, exist_ok=True)
    path = PATHS.report_exports / f"quick_watch_{datetime.now():%Y%m%d_%H%M%S}.html"
    path.write_text(content, encoding="utf-8")
    print(f"观察股快速分析: {path}，共{len(analyses)}只")
    if not args.no_push and not send_pushplus("持仓与观察股快速提醒", content):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
