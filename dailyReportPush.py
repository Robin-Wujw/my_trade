# -*- coding: utf-8 -*-
"""Build the four-section daily report and bounded PushPlus summary."""
import argparse
import html
import os
from datetime import datetime

import pandas as pd

from trade_utils import get_project_path, send_pushplus


SELECTION_DIR = get_project_path("选股结果")
BOARD_DIR = get_project_path("板块观察")


def parse_args():
    parser = argparse.ArgumentParser(description="生成四部分每日综合报告")
    parser.add_argument("--top", type=int, default=10, help="PushPlus每部分详细展示数量")
    parser.add_argument("--selection-top", type=int, default=30, help="兼容旧参数；正常基本面展示上限")
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def latest_file(directory, prefix, suffix):
    if not os.path.isdir(directory):
        return ""
    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    return max(paths, key=os.path.getmtime) if paths else ""


def num(value, digits=2):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def pct(value, digits=1):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def text(value, fallback="未提供"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return fallback
    return value


def esc(value, fallback="未提供"):
    return html.escape(text(value, fallback))


def wave_position(row):
    value = row.get("wave_pct")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "波段位置数据不足"
    if pd.isna(value):
        return "波段位置数据不足"
    if value > 100:
        return "已突破该下跌波段前高"
    if value < 0:
        return "仍低于该下跌波段后低"
    return f"当前修复至{value:.1f}%"


def right_side_conclusion(row):
    up = int(row.get("up_streak", 0) or 0)
    down = int(row.get("down_streak", 0) or 0)
    change = int(row.get("change", 0) or 0)
    signal = str(row.get("signal", "观察"))
    if down >= 5:
        return "暂停右侧交易", "33结构连续收缩5日，右侧成功率明显下降"
    if down >= 3:
        return "谨慎或暂停右侧", "33结构连续收缩3日，等待重新扩张"
    if up >= 5:
        return "可以右侧交易", "33结构连续扩张5日，右侧环境确认"
    if up >= 3:
        return "可以谨慎右侧", "33结构连续扩张3日，环境初步转好"
    if change > 0:
        return "轻仓观察右侧", "当日扩张但连续性不足"
    return "等待右侧确认", f"当日变化{change:+d}，连续性不足，信号为{signal}"


def stock_line(row, include_value=True):
    base = (
        f"<b>{esc(row.get('name', row.get('code')))} ({esc(row.get('code'))})</b> "
        f"现价{num(row.get('close'))}；"
    )
    show_value = include_value is True or (include_value == "auto" and bool(row.get("value_applicable")))
    if show_value:
        base += f"价值线{num(row.get('value_line'))}，现价/价值线{num(row.get('price_to_value'), 3)}；"
    else:
        base += f"估值方法{esc(row.get('method_name') or row.get('method') or '待核验')}；"
    base += (
        f"行业：{esc(row.get('industry'), '待核验')}；"
        f"当前主流板块：{esc(row.get('mainline_boards'), '未命中')}；"
        f"50%/62.5%/75%={num(row.get('wave_level_50'))}/"
        f"{num(row.get('wave_level_625'))}/{num(row.get('wave_level_75'))}；"
        f"{wave_position(row)}，{esc(row.get('wave_zone'), '波段不足')}；"
        f"扣非同比{pct(row.get('earnings_yoy'))}，质量{num(row.get('quality_score'), 1)}"
    )
    return base


def render_stock_section(title, frame, include_value=True, limit=None):
    shown = frame if limit is None else frame.head(limit)
    parts = [f"<h2>{esc(title)}（{len(frame)}只）</h2><ol>"]
    for _, row in shown.iterrows():
        parts.append(f"<li>{stock_line(row, include_value=include_value)}。理由：{esc(row.get('selection_reason', ''))}</li>")
    parts.append("</ol>")
    if limit is not None and len(frame) > limit:
        names = "、".join(frame.iloc[limit:]["name"].astype(str).tolist())
        parts.append(f"<p>其余{len(frame)-limit}只：{esc(names)}</p>")
    return "".join(parts)


def render_normal_section(frame, limit=None):
    shown = frame if limit is None else frame.head(limit)
    parts = [f"<h2>2. 正常基本面选股（{len(frame)}只）</h2>"]
    for layer, group in shown.groupby("strategy_layer", sort=False):
        parts.append(f"<h3>{esc(layer)}（{len(group)}只）</h3><ol>")
        for _, row in group.iterrows():
            parts.append(f"<li>{stock_line(row, include_value='auto')}。理由：{esc(row.get('selection_reason'))}</li>")
        parts.append("</ol>")
    if limit is not None and len(frame) > limit:
        names = "、".join(frame.iloc[limit:]["name"].astype(str).tolist())
        parts.append(f"<p>其余{len(frame)-limit}只：{esc(names)}</p>")
    return "".join(parts)


def build_reports(top, normal_top, max_chars):
    fundamental_path = latest_file(SELECTION_DIR, "daily_fundamental_selection_", ".csv")
    formula_path = latest_file(BOARD_DIR, "formula33_stats_", ".csv")
    sector_path = latest_file(BOARD_DIR, "sector_watch_", ".csv")
    if not fundamental_path:
        raise SystemExit("未找到daily_fundamental_selection每日基本面文件")
    if not formula_path:
        raise SystemExit("未找到formula33_stats市场结构文件")
    if not sector_path:
        raise SystemExit("未找到sector_watch主流板块文件")

    stocks = pd.read_csv(fundamental_path, dtype={"code": str}, low_memory=False)
    formula = pd.read_csv(formula_path)
    sectors = pd.read_csv(sector_path)
    values = stocks[stocks["strategy_part"] == "1.基本价值线或附近"].copy()
    normal = stocks[stocks["strategy_part"] == "2.正常基本面选股"].copy().head(normal_top)
    values = values.sort_values(["price_to_value", "quality_score"], ascending=[True, False])
    if {"layer_order", "fundamental_score"}.issubset(normal.columns):
        normal = normal.sort_values(["layer_order", "fundamental_score"], ascending=[True, False])
    report_date = str(stocks["date"].dropna().max())

    latest_formula = formula.sort_values("date").iloc[-1]
    right_status, right_reason = right_side_conclusion(latest_formula)
    sector_date = pd.to_datetime(sectors["date"], errors="coerce").max()
    report_dt = pd.to_datetime(report_date)
    sector_fresh = pd.notna(sector_date) and 0 <= (report_dt - sector_date).days <= 7
    top_sectors = sectors.sort_values("final_score", ascending=False).head(10) if sector_fresh else pd.DataFrame()

    heading = (
        f"<h1>{esc(report_date)} 每日四项分析</h1>"
        f"<p>财报期：{esc(stocks.get('report_period', pd.Series(['-'])).dropna().iloc[0])}。"
        "四部分分别判断，不用短期动量补齐名单。</p>"
    )
    full_parts = [heading]
    full_parts.append(render_stock_section("1. 基本价值线或附近（适用股票全量）", values, True, None))
    full_parts.append(render_normal_section(normal, None))
    full_parts.append(
        "<h2>3. 三浪三上行数量与右侧开关</h2>"
        f"<p>当日XG {int(latest_formula.get('count', 0))}只，较前一交易日"
        f"{int(latest_formula.get('change', 0)):+d}只；连续上行{int(latest_formula.get('up_streak', 0))}日，"
        f"连续下行{int(latest_formula.get('down_streak', 0))}日。"
        f"<b>结论：{esc(right_status)}</b>。{esc(right_reason)}。</p>"
    )
    full_parts.append("<h2>4. 主流板块判断</h2>")
    if top_sectors.empty:
        full_parts.append("<p>板块数据缺失或超过7天，不参与当日判断。</p>")
    else:
        full_parts.append(f"<p>板块数据日：{sector_date.strftime('%Y-%m-%d')}。</p><ol>")
        for _, row in top_sectors.iterrows():
            full_parts.append(
                f"<li><b>{esc(row.get('board'))}</b>：总分{num(row.get('final_score'),1)}，"
                f"3/5/20日{pct(row.get('ret3'))}/{pct(row.get('ret5'))}/{pct(row.get('ret20'))}，"
                f"5日/20日量能{num(row.get('amount_5_20'))}，涨停扩散{int(row.get('limit_up_count',0) or 0)}。</li>"
            )
        full_parts.append("</ol>")
    full_parts.append(
        "<hr><p>价值线左侧与50%右侧是两个维度。价值线内但低于50%的股票保留在第一栏等待；"
        "价值线上方但完成50%的高增长主线股票可进入第二栏，并单独提示估值风险。</p>"
    )
    full_html = "".join(full_parts)

    push_parts = [heading]
    push_parts.append(render_stock_section("1. 基本价值线或附近", values, True, top))
    push_parts.append(render_normal_section(normal, top))
    push_parts.extend(full_parts[3:])
    push_html = "".join(push_parts)
    if len(push_html) > max_chars:
        suffix = "<p>PushPlus摘要已按长度截断，完整名单见本地HTML和CSV。</p>"
        push_html = push_html[: max_chars - len(suffix)] + suffix
    return report_date, full_html, push_html, stocks, fundamental_path, formula_path, sector_path


def main():
    args = parse_args()
    result = build_reports(args.top, args.selection_top, args.max_chars)
    report_date, full_html, push_html, stocks, fundamental_path, formula_path, sector_path = result
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(BOARD_DIR, f"daily_report_{stamp}.html")
    output_path = os.path.join(SELECTION_DIR, f"daily_consolidated_selection_{report_date}_{stamp[-6:]}.csv")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(full_html)
    stocks.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"完整四项报告: {report_path}")
    print(f"前两项完整选股: {output_path}，共{len(stocks)}行")
    print(f"数据源: {fundamental_path} | {formula_path} | {sector_path}")
    print(f"PushPlus摘要长度: {len(push_html)}")
    if args.no_push:
        return
    ok = send_pushplus(f"{report_date} 每日四项分析", push_html)
    print("PUSH_RESULT", ok)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
