# -*- coding: utf-8 -*-
"""Build the four-section daily report and bounded PushPlus summary."""
import argparse
from dataclasses import dataclass
import html
import json
import os
from datetime import datetime

import pandas as pd

from stock_research.api.pushplus import send_pushplus
from stock_research.core.paths import PATHS
from stock_research.reporting.diff import (
    SelectionDiff,
    compare_snapshots,
    load_history,
    save_snapshot,
)


SELECTION_DIR = str(PATHS.selection_exports)
BOARD_DIR = str(PATHS.market_exports)
REPORT_DIR = str(PATHS.report_exports)
HISTORY_FILE = str(PATHS.state / "daily_selection_history.json")
FORMULA_PHASE_FILE = str(PATHS.state / "formula33_right_side_phase.json")
FORMULA_PHASE_VERSION = 2
RIGHT_SIDE_WAITING = "等待右侧阶段"
RIGHT_SIDE_WATCH = "观察、可右侧积极做多阶段"
RIGHT_SIDE_ACTIVE = "明确进入右侧阶段"
RIGHT_SIDE_EXITED = "明确退出右侧阶段"


@dataclass(frozen=True)
class DailyReportBundle:
    report_date: str
    full_html: str
    push_parts: tuple[str, str]
    stocks: pd.DataFrame
    fundamental_path: str
    formula_path: str
    sector_path: str
    selection_diff: SelectionDiff | None
    formula_phase_state: dict | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成四部分每日综合报告")
    parser.add_argument("--top", type=int, default=10, help="PushPlus每部分详细展示数量")
    parser.add_argument("--selection-top", type=int, default=30, help="兼容旧参数；正常基本面展示上限")
    parser.add_argument("--max-chars", type=int, default=18000)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--fundamental-path", default="")
    parser.add_argument("--formula-path", default="")
    parser.add_argument("--sector-path", default="")
    return parser.parse_args(argv)


def latest_file(directory, prefix, suffix):
    if not os.path.isdir(directory):
        return ""
    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(suffix)
        and "_sample" not in name.lower()
    ]
    return max(paths, key=os.path.getmtime) if paths else ""


def ensure_same_observation_date(inputs):
    normalized = {}
    for name, value in inputs.items():
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name} observation date missing or invalid: {value}")
        normalized[str(name)] = pd.Timestamp(parsed).strftime("%Y-%m-%d")
    dates = set(normalized.values())
    if len(dates) != 1:
        raise ValueError(f"observation date mismatch: {normalized}")
    return next(iter(dates))


def _resolve_input_path(explicit_path, directory, prefix, label):
    path = str(explicit_path or "").strip() or latest_file(
        directory,
        prefix,
        ".csv",
    )
    if not path or not os.path.isfile(path):
        raise SystemExit(f"未找到{label}文件: {path or 'missing'}")
    if "_sample" in os.path.basename(path).lower():
        raise SystemExit(f"{label}不能使用样例文件: {path}")
    return os.path.abspath(path)


def _observation_date(frame, label):
    if "date" not in frame.columns:
        raise ValueError(f"{label}缺少date字段")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.dropna().empty:
        raise ValueError(f"{label}没有有效observation date")
    return dates.max().strftime("%Y-%m-%d")


def _reject_sample_content(frame, path, label):
    if "data_status" not in frame.columns:
        return
    statuses = frame["data_status"].dropna().astype(str).str.lower()
    if statuses.str.contains("sample", regex=False).any():
        raise ValueError(f"{label}包含样例数据，拒绝生成生产日报: {path}")


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


def render_formula_status(latest_formula):
    """Render only the formal rolling 21-day Formula33 result."""
    window_count = latest_formula.get("window_unique_count")
    if window_count is None or pd.isna(window_count):
        return "近21个交易日三浪三正式结果数据不足。"
    formal = int(float(window_count))
    technical = pd.to_numeric(
        latest_formula.get("technical_unique_count"), errors="coerce"
    )
    suspended = pd.to_numeric(latest_formula.get("suspended_count"), errors="coerce")
    unavailable = pd.to_numeric(
        latest_formula.get("unavailable_count"), errors="coerce"
    )
    details = []
    if pd.notna(technical):
        details.append(f"技术命中{int(technical)}只")
    if pd.notna(suspended):
        details.append(f"观察日停牌或无交易{int(suspended)}只已排除")
    if pd.notna(unavailable):
        details.append(f"数据不可用{int(unavailable)}只")
    suffix = f"（{'；'.join(details)}）" if details else ""
    return (
        f"三浪三正式结果：{formal}只{suffix}。含义：近21个交易日内曾连续5日"
        "满足强势技术条件，且观察日仍可交易的去重股票。"
    )


def load_formula_phase_state(path=FORMULA_PHASE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def advance_formula_phase(formula_rows, previous_state=None):
    """Advance the persistent 3/5-day right-side phase without changing XG."""
    previous = dict(previous_state or {})
    if previous.get("version") != FORMULA_PHASE_VERSION:
        previous = {}
    phase = previous.get("phase") or RIGHT_SIDE_WAITING
    transition_date = str(previous.get("transition_date") or "")
    trigger = str(previous.get("trigger") or "")
    last_observation = pd.to_datetime(
        previous.get("observation_date"), errors="coerce"
    )

    frame = pd.DataFrame(formula_rows).copy()
    required = {"date", "window_up_streak", "window_down_streak"}
    if frame.empty or not required.issubset(frame.columns):
        return {
            "version": FORMULA_PHASE_VERSION,
            "phase": phase,
            "transition_date": transition_date,
            "trigger": trigger,
            "observation_date": str(previous.get("observation_date") or ""),
        }
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if frame.empty:
        return {
            "version": FORMULA_PHASE_VERSION,
            "phase": phase,
            "transition_date": transition_date,
            "trigger": trigger,
            "observation_date": str(previous.get("observation_date") or ""),
        }

    latest_date = frame["date"].max().normalize()
    if pd.notna(last_observation) and last_observation.normalize() > latest_date:
        phase = RIGHT_SIDE_WAITING
        transition_date = ""
        trigger = ""
        last_observation = pd.NaT
    if pd.notna(last_observation):
        frame = frame[frame["date"].dt.normalize() > last_observation.normalize()]

    def streak(value):
        number = pd.to_numeric(value, errors="coerce")
        return 0 if pd.isna(number) else int(number)

    for _, row in frame.iterrows():
        row_date = row["date"].strftime("%Y-%m-%d")
        up = streak(row.get("window_up_streak"))
        down = streak(row.get("window_down_streak"))
        if down >= 5:
            if phase != RIGHT_SIDE_EXITED:
                transition_date = row_date
                trigger = "连续5日负趋势"
            phase = RIGHT_SIDE_EXITED
        elif up >= 5:
            if phase != RIGHT_SIDE_ACTIVE:
                transition_date = row_date
                trigger = "连续5日正趋势"
            phase = RIGHT_SIDE_ACTIVE
        elif up >= 3 and phase != RIGHT_SIDE_ACTIVE:
            if phase != RIGHT_SIDE_WATCH:
                transition_date = row_date
                trigger = "连续3日正趋势"
            phase = RIGHT_SIDE_WATCH

    return {
        "version": FORMULA_PHASE_VERSION,
        "phase": phase,
        "transition_date": transition_date,
        "trigger": trigger,
        "observation_date": latest_date.strftime("%Y-%m-%d"),
    }


def save_formula_phase_state(path, state):
    if not state:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def wave_position(row):
    value = row.get("wave_pct")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "波段位置数据不足"
    if pd.isna(value):
        return "波段位置数据不足"
    if value > 100:
        return f"波段分位{value:.1f}%（已突破前高100%）"
    if value < 0:
        return f"波段分位{value:.1f}%（低于本轮低点0%）"
    return f"波段分位{value:.1f}%（低点=0%，下跌前高=100%）"


def wave_position_compact(row):
    value = pd.to_numeric(row.get("wave_pct"), errors="coerce")
    if pd.isna(value):
        return "分位不足 · 位置待确认"
    return f"{float(value):.1f}% · {_action_label(row)}"


def valuation_percentile_text(row):
    value = pd.to_numeric(row.get("valuation_percentile"), errors="coerce")
    if pd.isna(value):
        return ""
    pct_value = float(value) * 100 if abs(float(value)) <= 1 else float(value)
    return f"估值历史分位{pct_value:.1f}%"


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
        valuation_text = valuation_percentile_text(row)
        if valuation_text:
            base += f"{valuation_text}；"
    base += (
        f"行业：{esc(row.get('industry'), '待核验')}；"
        f"当前主流板块：{esc(row.get('mainline_boards'), '未命中')}；"
        f"{wave_position(row)}；"
        f"本轮低点/前高={num(row.get('wave_low'))}/{num(row.get('wave_high'))}；"
        f"50%/62.5%/75%价位={num(row.get('wave_level_50'))}/"
        f"{num(row.get('wave_level_625'))}/{num(row.get('wave_level_75'))}；"
        f"扣非同比{pct(row.get('earnings_yoy'))}，质量{num(row.get('quality_score'), 1)}"
    )
    return base


def render_stock_section(title, frame, include_value=True, limit=None, changes_html=""):
    shown = frame if limit is None else frame.head(limit)
    parts = [f"<h2>{esc(title)}（{len(frame)}只）</h2>"]
    if changes_html:
        parts.append(changes_html)
    parts.append("<ol>")
    for _, row in shown.iterrows():
        parts.append(f"<li>{stock_line(row, include_value=include_value)}。理由：{esc(row.get('selection_reason', ''))}</li>")
    parts.append("</ol>")
    if limit is not None and len(frame) > limit:
        names = "、".join(frame.iloc[limit:]["name"].astype(str).tolist())
        parts.append(f"<p>其余{len(frame)-limit}只：{esc(names)}</p>")
    return "".join(parts)


def render_normal_section(frame, limit=None, changes_html=""):
    shown = frame if limit is None else frame.head(limit)
    parts = [f"<h2>2. 正常基本面选股（{len(frame)}只）</h2>"]
    if changes_html:
        parts.append(changes_html)
    for layer, group in shown.groupby("strategy_layer", sort=False):
        parts.append(f"<h3>{esc(layer)}（{len(group)}只）</h3><ol>")
        for _, row in group.iterrows():
            parts.append(f"<li>{stock_line(row, include_value='auto')}。理由：{esc(row.get('selection_reason'))}</li>")
        parts.append("</ol>")
    if limit is not None and len(frame) > limit:
        names = "、".join(frame.iloc[limit:]["name"].astype(str).tolist())
        parts.append(f"<p>其余{len(frame)-limit}只：{esc(names)}</p>")
    return "".join(parts)


def _part_name(value):
    value = str(value or "")
    return value.split(".", 1)[-1] if "." in value else value


def render_selection_changes(selection_diff, strategy_part=None):
    if selection_diff is None:
        return "<p><b>与上一交易日相比：</b>首次建立可比较基线。</p>"
    added = [
        item for item in selection_diff.added
        if strategy_part is None or item.get("strategy_part") == strategy_part
    ]
    removed = [
        item for item in selection_diff.removed
        if strategy_part is None or item.get("strategy_part") == strategy_part
    ]
    moved = [
        item for item in selection_diff.moved
        if strategy_part is None
        or item.get("to_part") == strategy_part
    ]
    if not added and not removed and not moved:
        return "<p><b>与上一交易日相比：</b>无新进入、无退出、无分区变化。</p>"
    parts = ["<div><b>与上一交易日相比：</b><ul>"]
    if added:
        items = "、".join(f"{esc(item['name'])}({esc(item['code'])})" for item in added)
        parts.append(f"<li><b>新进入 {len(added)}只：</b>{items}</li>")
    if removed:
        items = "、".join(f"{esc(item['name'])}({esc(item['code'])})" for item in removed)
        parts.append(f"<li><b>退出 {len(removed)}只：</b>{items}</li>")
    if moved:
        items = "、".join(
            f"{esc(item['name'])}({esc(item['code'])})："
            f"{esc(_part_name(item['from_part']))} → {esc(_part_name(item['to_part']))}"
            for item in moved
        )
        parts.append(f"<li><b>分区变化 {len(moved)}只：</b>{items}</li>")
    parts.append("</ul></div>")
    return "".join(parts)


def _risk_label(row):
    explicit = text(row.get("risk"), "")
    if explicit:
        return explicit[:60]
    reason = text(row.get("selection_reason"), "")
    if "风险：" in reason:
        return reason.split("风险：", 1)[1][:60]
    if "待核验" in reason:
        return "存在待核验项"
    return "未见额外风险标签"


def _action_label(row):
    percentile = pd.to_numeric(row.get("wave_pct"), errors="coerce")
    if pd.notna(percentile):
        percentile = float(percentile)
        if percentile >= 75:
            return "强修复"
        if percentile >= 62.5:
            return "右侧确认"
        if percentile >= 50:
            return "右侧启动"
        return "左侧观察"
    zone = text(row.get("wave_zone"), "")
    if "62.5%以上" in zone:
        return "右侧确认"
    if "50%-62.5%" in zone:
        return "右侧启动"
    if "50%以下" in zone:
        return "左侧观察"
    return "位置待确认"


def _phase_guidance(phase):
    return {
        RIGHT_SIDE_WAITING: "市场结构尚未形成连续改善，优先观察，不因单日命中增加而追涨。",
        RIGHT_SIDE_WATCH: "市场结构初步改善，可重点跟踪右侧启动和右侧确认股票，仍需控制试错仓位。",
        RIGHT_SIDE_ACTIVE: "市场结构已连续改善，优先核验右侧确认、基本面质量和主流板块是否共振。",
        RIGHT_SIDE_EXITED: "市场结构已转弱，暂停新增右侧暴露，先处理风险和已有持仓。",
    }.get(str(phase), "阶段信息不足，保持观察并先核验数据。")


def _push_style():
    return (
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.55;"
        "color:#1f2937}h1{font-size:22px}h2{font-size:18px;border-left:4px solid "
        "#2563eb;padding-left:8px}h3{font-size:16px}.summary{background:#eff6ff;"
        "border:1px solid #93c5fd;border-radius:8px;padding:10px;margin:8px 0}"
        ".action{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;"
        "padding:10px;margin:8px 0}.warning{background:#fff7ed;border:1px solid "
        "#fdba74;border-radius:8px;padding:10px;margin:8px 0}table{border-collapse:"
        "collapse;width:100%;font-size:13px}th{background:#f3f4f6}th,td{padding:5px;"
        "vertical-align:top}</style>"
    )


def _compact_stock_table(frame, kind, minimal=False):
    if minimal:
        headers = ("股票/代码", "价值比", "质量", "波段分位")
        parts = [
            "<table><thead><tr>",
            "".join(f"<th>{header}</th>" for header in headers),
            "</tr></thead><tbody>",
        ]
        for _, row in frame.iterrows():
            cells = (
                f"{esc(row.get('name', row.get('code')))}<br>{esc(row.get('code'))}",
                num(row.get("price_to_value"), 3),
                num(row.get("quality_score"), 0),
                esc(wave_position_compact(row)),
            )
            parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)
    if kind == "value":
        headers = ("股票/代码", "现价÷价值线", "质量/100", "波段分位", "主要风险")
    else:
        headers = ("股票/代码", "质量/100", "命中主线/估值", "波段分位", "主要风险")
    parts = [
        "<table border='1' cellspacing='0' cellpadding='4'><thead><tr>",
        "".join(f"<th>{header}</th>" for header in headers),
        "</tr></thead><tbody>",
    ]
    for _, row in frame.iterrows():
        stock = f"{esc(row.get('name', row.get('code')))}<br>{esc(row.get('code'))}"
        if kind == "value":
            cells = (
                stock,
                num(row.get("price_to_value"), 3),
                num(row.get("quality_score"), 1),
                esc(wave_position_compact(row)),
                esc(_risk_label(row)),
            )
        else:
            cells = (
                stock,
                num(row.get("quality_score"), 1),
                (
                    esc(row.get("mainline_boards"), "未命中")
                    + (
                        f"<br>{esc(valuation_percentile_text(row))}"
                        if valuation_percentile_text(row)
                        else ""
                    )
                ),
                esc(wave_position_compact(row)),
                esc(_risk_label(row)),
            )
        parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _priority_details(frame, include_value, limit):
    if not limit:
        return ""
    parts = [f"<h3>优先关注（前{min(limit, len(frame))}只）</h3><ol>"]
    for _, row in frame.head(limit).iterrows():
        parts.append(
            f"<li>{stock_line(row, include_value=include_value)}。"
            f"理由：{esc(row.get('selection_reason'))}</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


def _sector_summary(top_sectors):
    if top_sectors is None or top_sectors.empty:
        return "<p>板块数据缺失或超过7天，不参与本次判断。</p>"
    parts = ["<ol>"]
    for _, row in top_sectors.iterrows():
        parts.append(
            f"<li><b>{esc(row.get('board'))}</b>：总分{num(row.get('final_score'), 1)}；"
            f"3/5/20日{pct(row.get('ret3'))}/{pct(row.get('ret5'))}/{pct(row.get('ret20'))}；"
            f"5日/20日量能{num(row.get('amount_5_20'))}；"
            f"涨停扩散{int(row.get('limit_up_count', 0) or 0)}只。</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


def validate_push_report(content, expected_codes, max_chars):
    if len(content) > max_chars:
        raise ValueError(f"PushPlus正文{len(content)}字符，超过{max_chars}字符上限")
    missing = [str(code) for code in expected_codes if str(code) not in content]
    if missing:
        raise ValueError(f"PushPlus正文遗漏股票代码: {', '.join(missing[:5])}")


def build_push_reports(
    report_date,
    values,
    normal,
    latest_formula,
    top_sectors,
    selection_diff,
    max_chars,
    top=5,
    formula_phase=RIGHT_SIDE_WAITING,
):
    formula_status = render_formula_status(latest_formula)
    value_zones = values.get("wave_zone", pd.Series(dtype=str)).value_counts()
    normal_zones = normal.get("wave_zone", pd.Series(dtype=str)).value_counts()
    value_quality = int((pd.to_numeric(values.get("quality_score"), errors="coerce") >= 80).sum())
    normal_quality = int((pd.to_numeric(normal.get("quality_score"), errors="coerce") >= 80).sum())
    industries = normal.get("industry", pd.Series(dtype=str)).fillna("未知").value_counts()
    top_industry = industries.index[0] if len(industries) else "未知"
    top_industry_count = int(industries.iloc[0]) if len(industries) else 0
    technical_unavailable = pd.to_numeric(
        latest_formula.get("unavailable_count"), errors="coerce"
    )
    technical_unavailable = int(technical_unavailable) if pd.notna(technical_unavailable) else 0

    def zone_count(counts, name):
        return int(counts.get(name, 0))

    def make_parts(detail_count, minimal_tables=False):
        part1 = "".join(
            [
                _push_style(),
                f"<h1>[1/2] {esc(report_date)} 今日结论与价值线池</h1>",
                "<div class='summary'><b>30秒结论</b><br>",
                f"市场阶段：<b>{esc(formula_phase)}</b><br>{formula_status}</div>",
                f"<div class='action'><b>今天怎么用：</b>{esc(_phase_guidance(formula_phase))}</div>",
                "<h2>1. 数据是否可用</h2>",
                f"<p>技术数据不可用 <b>{technical_unavailable}</b> 只。"
                "为0表示本次全市场技术统计完整；非0应先看异常，不把缺数据当成弱势。</p>",
                f"<h2>2. 价值线候选（{len(values)}只）</h2>",
                "<p><b>分层：</b>",
                f"右侧确认 {zone_count(value_zones, '62.5%以上确认')}只；"
                f"右侧启动 {zone_count(value_zones, '50%-62.5%右侧启动')}只；"
                f"左侧观察 {zone_count(value_zones, '50%以下未确认')}只；"
                f"质量分≥80有 {value_quality}只。</p>",
                "<p><b>波段分位口径：</b>本轮下跌低点=0%，下跌前高=100%；50%/62.5%/75%"
                "是从低点向前高修复的价格位置，不是全市场排名。</p>",
                "<p><b>阅读方法：</b>现价÷价值线低不等于可以买；优先顺序是强修复/右侧确认 → "
                "右侧启动 → 左侧观察，再核验质量与风险。</p>",
                render_selection_changes(selection_diff, "1.基本价值线或附近"),
                _priority_details(values, True, detail_count),
                "<h3>完整名单（代码不会省略）</h3>",
                _compact_stock_table(values, "value", minimal=minimal_tables),
                "<div class='warning'><b>边界：</b>价值线适用性仍需核验行业方法和财务口径；"
                "左侧观察只表示价格位置较低，不是买入信号。</div>",
            ]
        )
        part2 = "".join(
            [
                _push_style(),
                f"<h1>[2/2] {esc(report_date)} 基本面候选与主线</h1>",
                "<div class='summary'><b>30秒结论</b><br>"
                f"正常基本面候选 <b>{len(normal)}</b>只；质量分≥80有 <b>{normal_quality}</b>只；"
                f"最多集中于 <b>{esc(top_industry)}</b>（{top_industry_count}只）。</div>",
                "<div class='action'><b>核验顺序：</b>业绩硬条件 → 质量与流动性 → "
                "是否命中主线 → 右侧阶段 → 个股风险。高同比不自动等于可持续增长。</div>",
                f"<h2>3. 基本面候选分层（{len(normal)}只）</h2>",
                f"<p>右侧确认 {zone_count(normal_zones, '62.5%以上确认')}只；"
                f"右侧启动 {zone_count(normal_zones, '50%-62.5%右侧启动')}只；"
                f"左侧观察 {zone_count(normal_zones, '50%以下未确认')}只。</p>",
                render_selection_changes(selection_diff, "2.正常基本面选股"),
                _priority_details(normal, "auto", detail_count),
                "<h3>完整名单（代码不会省略）</h3>",
                _compact_stock_table(normal, "normal", minimal=minimal_tables),
                "<h2>4. 主流板块</h2>",
                "<p><b>分位说明：</b>股票表中的波段分位为精确数值；若显示估值历史分位，"
                "它表示当前PE/PB在自身历史中的位置，两种分位不可混用。</p>",
                "<p><b>怎么看：</b>先看3/5/20日是否同向，再看5日成交额相对20日是否放大，"
                "最后看涨停是否扩散；单项高分不代表主线成立。</p>",
                _sector_summary(top_sectors),
                "<div class='warning'><b>边界：</b>异常同比需核验低基数、扭亏和一次性项目；"
                "板块数据超过7天不参与判断；缺失数据不会自动补成通过。</div>",
            ]
        )
        return part1, part2

    rich_parts = make_parts(top)
    try:
        validate_push_report(rich_parts[0], values["code"], max_chars)
        validate_push_report(rich_parts[1], normal["code"], max_chars)
        return rich_parts
    except ValueError:
        compact_parts = make_parts(0)
        try:
            validate_push_report(compact_parts[0], values["code"], max_chars)
            validate_push_report(compact_parts[1], normal["code"], max_chars)
            return compact_parts
        except ValueError:
            minimal_parts = make_parts(0, minimal_tables=True)
            validate_push_report(minimal_parts[0], values["code"], max_chars)
            validate_push_report(minimal_parts[1], normal["code"], max_chars)
            return minimal_parts


def build_reports(
    top,
    normal_top,
    max_chars,
    fundamental_path="",
    formula_path="",
    sector_path="",
):
    fundamental_path = _resolve_input_path(
        fundamental_path,
        SELECTION_DIR,
        "daily_fundamental_selection_",
        "daily_fundamental_selection每日基本面",
    )
    formula_path = _resolve_input_path(
        formula_path,
        BOARD_DIR,
        "formula33_stats_",
        "formula33_stats市场结构",
    )
    sector_path = _resolve_input_path(
        sector_path,
        BOARD_DIR,
        "sector_watch_",
        "sector_watch主流板块",
    )

    stocks = pd.read_csv(fundamental_path, dtype={"code": str}, low_memory=False)
    formula = pd.read_csv(formula_path)
    sectors = pd.read_csv(sector_path)
    _reject_sample_content(stocks, fundamental_path, "基本面输入")
    _reject_sample_content(formula, formula_path, "Formula33输入")
    _reject_sample_content(sectors, sector_path, "板块输入")
    report_date = ensure_same_observation_date(
        {
            "fundamental": _observation_date(stocks, "基本面输入"),
            "formula33": _observation_date(formula, "Formula33输入"),
            "sector_watch": _observation_date(sectors, "板块输入"),
        }
    )
    values = stocks[stocks["strategy_part"] == "1.基本价值线或附近"].copy()
    normal = stocks[stocks["strategy_part"] == "2.正常基本面选股"].copy().head(normal_top)
    values = values.sort_values(["price_to_value", "quality_score"], ascending=[True, False])
    if {"layer_order", "fundamental_score"}.issubset(normal.columns):
        normal = normal.sort_values(["layer_order", "fundamental_score"], ascending=[True, False])
    history = load_history(HISTORY_FILE)
    previous = history.previous_before(report_date)
    selection_diff = (
        compare_snapshots(previous, stocks.to_dict("records"))
        if previous is not None
        else None
    )

    formula = formula.assign(
        date=pd.to_datetime(formula["date"], errors="coerce")
    )
    latest_formula = formula.sort_values("date").iloc[-1]
    formula_phase_state = advance_formula_phase(
        formula,
        load_formula_phase_state(FORMULA_PHASE_FILE),
    )
    formula_phase = formula_phase_state["phase"]
    sector_date = pd.to_datetime(sectors["date"], errors="coerce").max()
    top_sectors = sectors.sort_values("final_score", ascending=False).head(10)

    heading = (
        f"<h1>{esc(report_date)} 每日四项分析</h1>"
        f"<p>财报期：{esc(stocks.get('report_period', pd.Series(['-'])).dropna().iloc[0])}。"
        "四部分分别判断，不用短期动量补齐名单。</p>"
    )
    full_parts = [heading]
    full_parts.append(
        render_stock_section(
            "1. 基本价值线或附近（适用股票全量）",
            values,
            True,
            None,
            render_selection_changes(selection_diff, "1.基本价值线或附近"),
        )
    )
    full_parts.append(
        render_normal_section(
            normal,
            None,
            render_selection_changes(selection_diff, "2.正常基本面选股"),
        )
    )
    full_parts.append(
        "<h2>3. 最近21个交易日强势技术结果</h2>"
        f"<p>{render_formula_status(latest_formula)}"
        f"<b>当前阶段：{esc(formula_phase)}</b>。</p>"
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
    full_html = "".join(full_parts)

    push_parts = build_push_reports(
        report_date,
        values,
        normal,
        latest_formula,
        top_sectors,
        selection_diff,
        max_chars,
        top=top,
        formula_phase=formula_phase,
    )
    return DailyReportBundle(
        report_date=report_date,
        full_html=full_html,
        push_parts=push_parts,
        stocks=stocks,
        fundamental_path=fundamental_path,
        formula_path=formula_path,
        sector_path=sector_path,
        selection_diff=selection_diff,
        formula_phase_state=formula_phase_state,
    )


def main(argv=None):
    args = parse_args(argv)
    bundle = build_reports(
        args.top,
        args.selection_top,
        args.max_chars,
        args.fundamental_path,
        args.formula_path,
        args.sector_path,
    )
    report_date = bundle.report_date
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"daily_report_{stamp}.html")
    push_preview_path = os.path.join(REPORT_DIR, f"pushplus_preview_{stamp}.html")
    output_path = os.path.join(SELECTION_DIR, f"daily_consolidated_selection_{report_date}_{stamp[-6:]}.csv")
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(SELECTION_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(bundle.full_html)
    with open(push_preview_path, "w", encoding="utf-8") as handle:
        handle.write(
            bundle.push_parts[0]
            + "<hr style='margin:32px 0;border:0;border-top:3px solid #111827'>"
            + bundle.push_parts[1]
        )
    bundle.stocks.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_snapshot(HISTORY_FILE, report_date, bundle.stocks.to_dict("records"))
    save_formula_phase_state(FORMULA_PHASE_FILE, bundle.formula_phase_state)
    print(f"完整四项报告: {report_path}")
    print(f"PushPlus发送前预览: {push_preview_path}")
    print(f"前两项完整选股: {output_path}，共{len(bundle.stocks)}行")
    print(
        f"数据源: {bundle.fundamental_path} | {bundle.formula_path} | "
        f"{bundle.sector_path}"
    )
    for index, content in enumerate(bundle.push_parts, start=1):
        print(f"PushPlus第{index}部分长度: {len(content)}")
    if args.no_push:
        return
    titles = (
        f"[1/2] {report_date} 市场状态与价值线池",
        f"[2/2] {report_date} 基本面候选与主线",
    )
    results = []
    for index, (title, content) in enumerate(
        zip(titles, bundle.push_parts), start=1
    ):
        ok = send_pushplus(title, content)
        results.append(ok)
        print(f"PUSH_RESULT_{index}", ok)
    if not all(results):
        raise SystemExit(2)
