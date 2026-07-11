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
from stock_research.api.email import send_html_email
from stock_research.core.paths import PATHS
from stock_research.reporting.diff import (
    SelectionDiff,
    compare_snapshots,
    load_history,
    save_snapshot,
)
from stock_research.reporting.breakout_watch import (
    load_watch_state,
    recent_pool,
    save_watch_state,
    update_breakout_watch,
)


SELECTION_DIR = str(PATHS.selection_exports)
BOARD_DIR = str(PATHS.market_exports)
REPORT_DIR = str(PATHS.report_exports)
HISTORY_FILE = str(PATHS.state / "daily_selection_history.json")
FORMULA_PHASE_FILE = str(PATHS.state / "formula33_right_side_phase.json")
BREAKOUT_WATCH_FILE = str(PATHS.state / "two_month_breakout_watch.json")
KLINE_CACHE_DIR = str(PATHS.cache / "formula33_kline" / "akshare")
FORMULA_PHASE_VERSION = 2
RIGHT_SIDE_WAITING = "等待右侧阶段"
RIGHT_SIDE_WATCH = "观察、可右侧积极做多阶段"
RIGHT_SIDE_ACTIVE = "明确进入右侧阶段"
RIGHT_SIDE_EXITED = "明确退出右侧阶段"


@dataclass(frozen=True)
class DailyReportBundle:
    report_date: str
    full_html: str
    push_parts: tuple[str, ...]
    stocks: pd.DataFrame
    fundamental_path: str
    formula_path: str
    sector_path: str
    selection_diff: SelectionDiff | None
    formula_phase_state: dict | None = None
    breakout_watch_state: dict | None = None
    breakout_alerts: tuple[dict, ...] = ()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成四部分每日综合报告")
    parser.add_argument("--top", type=int, default=10, help="PushPlus每部分详细展示数量")
    parser.add_argument("--selection-top", type=int, default=30, help="兼容旧参数；正常基本面展示上限")
    parser.add_argument("--max-chars", type=int, default=18000)
    parser.add_argument("--no-push", action="store_true")
    default_delivery = os.environ.get("REPORT_DELIVERY", "").strip().lower()
    if not default_delivery:
        default_delivery = "email" if os.environ.get("REPORT_EMAIL_TO", "").strip() else "pushplus"
    parser.add_argument(
        "--delivery",
        choices=("pushplus", "email", "both"),
        default=default_delivery,
        help="外部投递方式；配置REPORT_EMAIL_TO后默认只发HTML邮件",
    )
    parser.add_argument("--email-to", default=os.environ.get("REPORT_EMAIL_TO", ""))
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
    percentile, breakout = _wave_values(row)
    if percentile is None:
        return "波段位置数据不足"
    suffix = f"；突破前高+{breakout:.1f}%" if breakout > 0 else ""
    return (
        f"波段分位{percentile:.1f}%（低点=0%，下跌前高=100%）{suffix}"
    )


def _wave_values(row):
    raw = pd.to_numeric(row.get("wave_pct"), errors="coerce")
    explicit_breakout = pd.to_numeric(
        row.get("wave_breakout_pct"), errors="coerce"
    )
    if pd.isna(raw):
        return None, 0.0
    raw = float(raw)
    close = pd.to_numeric(row.get("close"), errors="coerce")
    wave_high = pd.to_numeric(row.get("wave_high"), errors="coerce")
    derived_breakout = (
        max(0.0, (float(close) / float(wave_high) - 1) * 100)
        if pd.notna(close) and pd.notna(wave_high) and float(wave_high) > 0
        else max(0.0, raw - 100.0)
    )
    breakout = (
        max(0.0, float(explicit_breakout))
        if pd.notna(explicit_breakout)
        else derived_breakout
    )
    return min(100.0, max(0.0, raw)), breakout


def wave_position_compact(row):
    percentile, breakout = _wave_values(row)
    if percentile is None:
        return "分位不足 · 位置待确认"
    if breakout > 0:
        return f"已突破前高 {breakout:.1f}%"
    return f"{percentile:.1f}%，{_action_label(row)}"


def stage_wave_text(row):
    stage = str(row.get("trend_stage") or "")
    passed = bool(row.get("stage_level_50_passed"))
    level = num(row.get("stage_level_50"))
    if stage == "uptrend":
        return f"上涨阶段：上涨波段50%支撑{level}{'有效' if passed else '失效'}"
    if stage == "pullback_recovery":
        return f"回调修复阶段：{'已突破' if passed else '未突破'}回调50%趋势改变点{level}"
    return "波段阶段待确认"


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
        f"上涨起点/前高/回调低点={num(row.get('uptrend_wave_low'))}/"
        f"{num(row.get('wave_high'))}/{num(row.get('wave_low'))}；"
        f"上涨波段50%={num(row.get('uptrend_wave_level_50'))}；"
        f"回调趋势改变点50%={num(row.get('wave_level_50'))}；"
        f"回调62.5%/75%={num(row.get('wave_level_625'))}/{num(row.get('wave_level_75'))}；"
        f"{stage_wave_text(row)}；"
        f"扣非同比{pct(row.get('earnings_yoy'))}，质量{num(row.get('quality_score'), 1)}"
    )
    if bool(row.get("technical_available")):
        base += f"；{technical_analysis(row)}"
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


def _divergence_label(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or int(number) == 0:
        return "无"
    return "底背离" if int(number) > 0 else "顶背离"


def technical_analysis(row):
    if not bool(row.get("technical_available")):
        return "技术指标不可用（历史数据不足），不参与评分。"
    kd_gap = pd.to_numeric(row.get("kd_gap"), errors="coerce")
    gap_text = "未达极限"
    if pd.notna(kd_gap) and abs(kd_gap) >= 20:
        gap_text = f"开口{abs(kd_gap):.1f}≥20，短线按均值收敛风险处理"
    ene_position = pd.to_numeric(row.get("ene_position"), errors="coerce")
    ene_text = "ENE通道内"
    if pd.notna(ene_position):
        ene_text = "ENE上轨外，追高风险" if ene_position >= 100 else "ENE下轨外，关注止跌" if ene_position <= 0 else f"ENE通道{ene_position:.0f}%位置"
    baseline_count = pd.to_numeric(row.get("volume_baseline_count"), errors="coerce")
    baseline_count = int(baseline_count) if pd.notna(baseline_count) else 0
    volume_text = (
        f"基准量{baseline_count}/4：现量/5日均量={num(row.get('volume_ratio_ma5'),2)}，"
        f"/10日均量={num(row.get('volume_ratio_ma10'),2)}，"
        f"/5日扣抵量={num(row.get('volume_ratio_ref5'),2)}，"
        f"/10日扣抵量={num(row.get('volume_ratio_ref10'),2)}"
    )
    return (
        f"量化：行动{num(row.get('technical_action_score'),1)}/100，机会{num(row.get('technical_opportunity_score'),1)}，"
        f"风险{num(row.get('technical_risk_score'),1)}，置信度{num(row.get('technical_confidence'),1)}；"
        f"盘中KD高点背离={_divergence_label(row.get('kd_divergence'))}，RSI999={num(row.get('rsi999'),1)}"
        f"/{_divergence_label(row.get('rsi_divergence'))}，MACD={_divergence_label(row.get('macd_divergence'))}；"
        f"{gap_text}；{ene_text}；WR10/20={num(row.get('wr10'),1)}/{num(row.get('wr20'),1)}；"
        f"BIAS10={num(row.get('bias10'),1)}%；{volume_text}。背离仅作警讯，不单独作为卖讯。"
    )


def _action_label(row):
    percentile, breakout = _wave_values(row)
    if percentile is not None:
        if breakout > 0:
            return "突破前高"
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
        "vertical-align:top}.danger{color:#b91c1c;font-weight:bold}.ok{color:#047857;"
        "font-weight:bold}</style>"
    )


def _kd_compact(row):
    k_value = pd.to_numeric(row.get("kd_k_close"), errors="coerce")
    d_value = pd.to_numeric(row.get("kd_d_close"), errors="coerce")
    gap = pd.to_numeric(row.get("kd_gap"), errors="coerce")
    warnings = []
    if pd.notna(gap) and abs(float(gap)) >= 20:
        direction = "偏热" if gap > 0 else "偏弱/超卖"
        warnings.append(f"KD开口{abs(gap):.1f}≥20，{direction}，留意收敛")
    divergences = [
        name + _divergence_label(row.get(field))
        for name, field in (
            ("KD", "kd_divergence"),
            ("RSI", "rsi_divergence"),
            ("MACD", "macd_divergence"),
        )
        if _divergence_label(row.get(field)) != "无"
    ]
    if divergences:
        warnings.append("、".join(divergences))
    if not warnings:
        return "暂无指标警讯"
    return "<br>".join(f"<span class='danger'>{esc(item)}</span>" for item in warnings)


def _dual_wave_compact(row):
    stage = str(row.get("trend_stage") or "")
    passed = bool(row.get("stage_level_50_passed"))
    if stage == "uptrend":
        state = "上涨阶段，支撑有效" if passed else "上涨阶段，支撑失效"
    elif stage == "pullback_recovery":
        state = "回调已过50%" if passed else "回调未过50%"
    else:
        state = "阶段待确认"
    state_class = "ok" if passed else "danger"
    close = pd.to_numeric(row.get("close"), errors="coerce")
    pullback_level = pd.to_numeric(row.get("wave_level_50"), errors="coerce")
    distance = (
        (float(close) / float(pullback_level) - 1) * 100
        if pd.notna(close) and pd.notna(pullback_level) and pullback_level > 0
        else None
    )
    distance_text = ""
    if distance is not None:
        distance_text = (
            f"<br>已高于突破价{distance:.1f}%"
            if distance >= 0
            else f"<br>距突破还差{abs(distance):.1f}%"
        )
    return (
        f"上涨支撑价 {num(row.get('uptrend_wave_level_50'))}<br>"
        f"回调突破价 {num(row.get('wave_level_50'))}"
        f"{distance_text}<br><span class='{state_class}'>{state}</span>"
    )


def _volume_compact(row):
    count = pd.to_numeric(row.get("volume_baseline_count"), errors="coerce")
    count = int(count) if pd.notna(count) else 0
    if count == 4:
        return "<span class='ok'>上涨量能达标</span>"
    if count == 3:
        return "量能接近达标<br>还差1项"
    if count == 2:
        return "量能一般<br>2项未达"
    return f"<span class='danger'>量能不足</span><br>{4-count}项未达"


def _valuation_compact(row):
    ratio = pd.to_numeric(row.get("price_to_value"), errors="coerce")
    return "未采用价值线" if pd.isna(ratio) else f"价值线的 {float(ratio):.2f} 倍"


def _compact_stock_table(frame, kind):
    if kind == "value":
        headers = ("股票与现价", "估值位置", "技术评分", "指标警讯", "波段关键价", "量能与位置")
    else:
        headers = ("股票与现价", "基本面质量", "技术评分", "指标警讯", "波段关键价", "量能与位置")
    parts = [
        "<table border='1' cellspacing='0' cellpadding='4'><thead><tr>",
        "".join(f"<th>{header}</th>" for header in headers),
        "</tr></thead><tbody>",
    ]
    for _, row in frame.iterrows():
        stock = (
            f"{esc(row.get('name', row.get('code')))}<br>{esc(row.get('code'))}"
            f"<br>现价<b>{num(row.get('close'))}</b>"
        )
        if kind == "value":
            cells = (
                stock,
                _valuation_compact(row),
                f"可操作性 {num(row.get('technical_action_score'),0)}<br>风险 {num(row.get('technical_risk_score'),0)}",
                _kd_compact(row),
                _dual_wave_compact(row),
                f"{_volume_compact(row)}<br>回调修复 {esc(wave_position_compact(row))}",
            )
        else:
            cells = (
                stock,
                f"质量 {num(row.get('quality_score'),0)}/100",
                f"可操作性 {num(row.get('technical_action_score'),0)}<br>风险 {num(row.get('technical_risk_score'),0)}",
                _kd_compact(row),
                _dual_wave_compact(row),
                f"{_volume_compact(row)}<br>回调修复 {esc(wave_position_compact(row))}",
            )
        parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _breakout_watch_pages(report_date, alerts, max_chars):
    if not alerts:
        return []
    intro = (
        _push_style()
        + f"<h1>{esc(report_date)} 两个月突破强提醒</h1>"
        + "<div class='warning'><b>独立追踪：</b>来源为过去两个月曾进入前两项选股的股票；"
        "回调45%-50%为强提醒区；上穿50%后继续跟踪并累计突破次数，超过60%才停止提醒。"
        "只有突破前高完成且脱离当前全部筛选才从追踪池剔除。</div>"
    )
    pages = []
    offset = 0
    while offset < len(alerts):
        prefix = intro if offset == 0 else _push_style() + f"<h1>{esc(report_date)} 两个月突破强提醒（续）</h1>"
        parts = [prefix, "<table border='1'><tr><th>股票与现价</th><th>突破状态</th><th>关键价位</th><th>当前筛选状态</th></tr>"]
        start = offset
        while offset < len(alerts):
            item = alerts[offset]
            row = (
                f"<tr><td><b>{esc(item.get('name'))}</b><br>{esc(item.get('code'))}<br>现价<b>{num(item.get('close'))}</b></td>"
                f"<td><span class='danger'>{esc(item.get('alert_level'))}</span><br>回调进度{num(item.get('recovery_pct'),1)}%"
                f"<br>累计突破{int(item.get('crossing_count') or 0)}次</td>"
                f"<td>上涨50%={num(item.get('uptrend_level_50'))}<br>回调50%={num(item.get('pullback_level_50'))}"
                f"<br>前高={num(item.get('prior_high'))}</td>"
                f"<td>{'是，继续跟踪' if item.get('in_current_selection') else '否，独立跟踪中'}</td></tr>"
            )
            candidate = "".join(parts) + row + "</table>"
            if len(candidate) > max_chars and offset > start:
                break
            parts.append(row)
            offset += 1
        parts.append("</table>")
        pages.append("".join(parts))
    return pages


def _priority_details(frame, include_value, limit):
    if not limit:
        return ""
    parts = [f"<h3>优先关注（前{min(limit, len(frame))}只）</h3><ol>"]
    for _, row in frame.head(limit).iterrows():
        parts.append(
            f"<li>{stock_line(row, include_value=include_value)}。"
            f"理由：{esc(row.get('selection_reason'))}<br>{esc(technical_analysis(row))}</li>"
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


def _paginate_stock_section(
    intro,
    frame,
    kind,
    ending,
    continuation_title,
    max_chars,
):
    """Split a full-detail stock table across messages without dropping columns."""
    if frame is None or frame.empty:
        content = intro + _compact_stock_table(pd.DataFrame(), kind) + ending
        validate_push_report(content, (), max_chars)
        return [content]
    pages = []
    offset = 0
    first = True
    while offset < len(frame):
        prefix = (
            intro
            if first
            else _push_style()
            + f"<h1>{esc(continuation_title)}</h1>"
            + f"<p>接上条，从第{offset + 1}只继续；字段和风险说明不压缩。</p>"
        )
        low, high, best = 1, len(frame) - offset, 0
        while low <= high:
            middle = (low + high) // 2
            chunk = frame.iloc[offset : offset + middle]
            is_last = offset + middle >= len(frame)
            candidate = (
                prefix
                + _compact_stock_table(chunk, kind)
                + (ending if is_last else "<p>名单未完，下一条继续。</p>")
            )
            if len(candidate) <= max_chars:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError(
                f"PushPlus单页固定说明已超过{max_chars}字符，无法容纳一行完整股票数据"
            )
        chunk = frame.iloc[offset : offset + best]
        offset += best
        is_last = offset >= len(frame)
        pages.append(
            prefix
            + _compact_stock_table(chunk, kind)
            + (ending if is_last else "<p>名单未完，下一条继续。</p>")
        )
        first = False
    return pages


def _number_push_pages(pages):
    total = len(pages)
    return tuple(
        page.replace("<h1>", f"<h1>[{index}/{total}] ", 1)
        for index, page in enumerate(pages, start=1)
    )


def validate_push_pages(pages, expected_codes, max_chars):
    for content in pages:
        if len(content) > max_chars:
            raise ValueError(
                f"PushPlus正文{len(content)}字符，超过{max_chars}字符上限"
            )
    combined = "".join(pages)
    missing = [str(code) for code in expected_codes if str(code) not in combined]
    if missing:
        raise ValueError(f"PushPlus分页遗漏股票代码: {', '.join(missing[:5])}")


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
    breakout_alerts=(),
):
    # Page numbering is added after pagination; reserve enough room for it.
    page_content_limit = max_chars - 32
    formula_status = render_formula_status(latest_formula)
    value_zones = values.get("wave_zone", pd.Series(dtype=str)).value_counts()
    normal_zones = normal.get("wave_zone", pd.Series(dtype=str)).value_counts()
    value_quality = int((pd.to_numeric(values.get("quality_score"), errors="coerce") >= 80).sum())
    normal_quality = int((pd.to_numeric(normal.get("quality_score"), errors="coerce") >= 80).sum())
    value_gaps = pd.to_numeric(
        values.get("kd_gap", pd.Series(index=values.index, dtype=float)),
        errors="coerce",
    )
    normal_gaps = pd.to_numeric(
        normal.get("kd_gap", pd.Series(index=normal.index, dtype=float)),
        errors="coerce",
    )
    value_kd_extreme = int((value_gaps.abs() >= 20).sum())
    normal_kd_extreme = int((normal_gaps.abs() >= 20).sum())
    industries = normal.get("industry", pd.Series(dtype=str)).fillna("未知").value_counts()
    top_industry = industries.index[0] if len(industries) else "未知"
    top_industry_count = int(industries.iloc[0]) if len(industries) else 0
    technical_unavailable = pd.to_numeric(
        latest_formula.get("unavailable_count"), errors="coerce"
    )
    technical_unavailable = int(technical_unavailable) if pd.notna(technical_unavailable) else 0

    def zone_count(counts, name):
        return int(counts.get(name, 0))

    value_intro = "".join(
        [
                _push_style(),
                f"<h1>{esc(report_date)} 今日结论与价值线池</h1>",
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
                f"<p><b>KD开口：</b>|K-D|≥20有 {value_kd_extreme}只，逐股标红提示短线收敛风险。</p>",
                "<p><b>双波段口径：</b>回调阶段先看“前高—回调低点”的50%，突破才有右侧意义；"
                "突破前高进入上涨阶段后，改看“上涨起点—上涨高点”的50%支撑。完整名单逐股显示两条价位。</p>",
                "<p><b>量能口径：</b>今日成交量同时高于5日均量、10日均量、5日前扣抵量和10日前扣抵量，"
                "才显示“上涨量能达标”；接近达标或不足会直接写明差几项。</p>",
                "<p><b>阅读方法：</b>现价÷价值线低不等于可以买；优先顺序是强修复/右侧确认 → "
                "右侧启动 → 左侧观察，再核验质量与风险。</p>",
                render_selection_changes(selection_diff, "1.基本价值线或附近"),
                _priority_details(values, True, top),
                "<h3>完整名单（代码不会省略）</h3>",
        ]
    )
    value_ending = (
        "<div class='warning'><b>边界：</b>价值线适用性仍需核验行业方法和财务口径；"
        "左侧观察只表示价格位置较低，不是买入信号。</div>"
    )
    normal_intro = "".join(
        [
                _push_style(),
                f"<h1>{esc(report_date)} 基本面候选与主线</h1>",
                "<div class='summary'><b>30秒结论</b><br>"
                f"正常基本面候选 <b>{len(normal)}</b>只；质量分≥80有 <b>{normal_quality}</b>只；"
                f"最多集中于 <b>{esc(top_industry)}</b>（{top_industry_count}只）。</div>",
                "<div class='action'><b>核验顺序：</b>业绩硬条件 → 质量与流动性 → "
                "是否命中主线 → 右侧阶段 → 个股风险。高同比不自动等于可持续增长。</div>",
                f"<h2>3. 基本面候选分层（{len(normal)}只）</h2>",
                f"<p>右侧确认 {zone_count(normal_zones, '62.5%以上确认')}只；"
                f"右侧启动 {zone_count(normal_zones, '50%-62.5%右侧启动')}只；"
                f"左侧观察 {zone_count(normal_zones, '50%以下未确认')}只。</p>",
                f"<p><b>KD开口：</b>|K-D|≥20有 {normal_kd_extreme}只，逐股标红提示短线收敛风险。</p>",
                "<p><b>表格读法：</b>正常指标只显示“暂无指标警讯”；出现KD极限开口或背离时才展开。"
                "波段栏直接给出上涨支撑价、回调突破价以及距离。</p>",
                render_selection_changes(selection_diff, "2.正常基本面选股"),
                _priority_details(normal, "auto", top),
                "<h3>完整名单（代码不会省略）</h3>",
        ]
    )
    normal_ending = "".join(
        [
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
    pages = _breakout_watch_pages(report_date, breakout_alerts, page_content_limit)
    pages.extend(_paginate_stock_section(
        value_intro,
        values,
        "value",
        value_ending,
        f"{report_date} 价值线完整名单（续）",
        page_content_limit,
    ))
    pages.extend(
        _paginate_stock_section(
            normal_intro,
            normal,
            "normal",
            normal_ending,
            f"{report_date} 基本面完整名单（续）",
            page_content_limit,
        )
    )
    numbered = _number_push_pages(pages)
    validate_push_pages(
        numbered,
        list(values.get("code", pd.Series(dtype=str)))
        + list(normal.get("code", pd.Series(dtype=str))),
        max_chars,
    )
    return numbered


def _load_watch_kline(code, observation_date):
    path = os.path.join(KLINE_CACHE_DIR, f"{str(code).replace('.', '_')}.csv")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    cutoff = pd.Timestamp(observation_date).normalize()
    frame = frame[frame["date"].dt.normalize() <= cutoff]
    for column in ("high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.dropna(subset=["date", "high", "low", "close"]).sort_values("date")


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
    for frame in (values, normal):
        frame["technical_action_score"] = pd.to_numeric(
            frame.get("technical_action_score"), errors="coerce"
        )
        frame["technical_risk_score"] = pd.to_numeric(
            frame.get("technical_risk_score"), errors="coerce"
        )
    value_sort = ["technical_action_score", "technical_risk_score", "price_to_value", "quality_score"]
    values = values.sort_values(
        value_sort,
        ascending=[False, True, True, False],
        na_position="last",
    )
    normal_sort = ["technical_action_score", "technical_risk_score"]
    normal_ascending = [False, True]
    if {"layer_order", "fundamental_score"}.issubset(normal.columns):
        normal_sort.extend(["layer_order", "fundamental_score"])
        normal_ascending.extend([True, False])
    normal = normal.sort_values(
        normal_sort,
        ascending=normal_ascending,
        na_position="last",
    )
    history = load_history(HISTORY_FILE)
    previous = history.previous_before(report_date)
    selection_diff = (
        compare_snapshots(previous, stocks.to_dict("records"))
        if previous is not None
        else None
    )
    pool = recent_pool(history, stocks.to_dict("records"), report_date)
    breakout_watch_state, breakout_alerts = update_breakout_watch(
        pool,
        stocks.get("code", pd.Series(dtype=str)).astype(str),
        report_date,
        _load_watch_kline,
        load_watch_state(BREAKOUT_WATCH_FILE),
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
        breakout_alerts=breakout_alerts,
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
        breakout_watch_state=breakout_watch_state,
        breakout_alerts=tuple(breakout_alerts),
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
            "<hr style='margin:32px 0;border:0;border-top:3px solid #111827'>".join(
                bundle.push_parts
            )
        )
    bundle.stocks.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_snapshot(HISTORY_FILE, report_date, bundle.stocks.to_dict("records"))
    save_formula_phase_state(FORMULA_PHASE_FILE, bundle.formula_phase_state)
    if bundle.breakout_watch_state:
        save_watch_state(BREAKOUT_WATCH_FILE, bundle.breakout_watch_state)
    print(f"完整四项报告: {report_path}")
    print(f"PushPlus发送前预览: {push_preview_path}")
    print(f"前两项完整选股: {output_path}，共{len(bundle.stocks)}行")
    print(
        f"数据源: {bundle.fundamental_path} | {bundle.formula_path} | "
        f"{bundle.sector_path}"
    )
    print(f"两个月突破强提醒: {len(bundle.breakout_alerts)}只")
    for index, content in enumerate(bundle.push_parts, start=1):
        print(f"PushPlus第{index}部分长度: {len(content)}")
    if args.no_push:
        return
    if args.delivery in {"email", "both"}:
        email_html = _push_style() + bundle.full_html
        email_ok = send_html_email(
            f"{report_date} 每日A股研究报告",
            email_html,
            recipients=args.email_to,
        )
        print("EMAIL_RESULT", email_ok)
        if not email_ok:
            raise SystemExit(2)
    if args.delivery == "email":
        return
    results = []
    total_parts = len(bundle.push_parts)
    for index, content in enumerate(bundle.push_parts, start=1):
        title = f"[{index}/{total_parts}] {report_date} 每日选股报告"
        ok = send_pushplus(title, content)
        results.append(ok)
        print(f"PUSH_RESULT_{index}", ok)
    if not all(results):
        raise SystemExit(2)
