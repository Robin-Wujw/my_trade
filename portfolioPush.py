# -*- coding: utf-8 -*-
"""
PushPlus 推送组合选股结果。

默认读取回测结果目录下最新的 *_portfolio_*.csv，并补充最新行情里的现价和当日涨跌幅。
"""
import argparse
import glob
import html
import os
import re
import time
from datetime import datetime

import akshare as ak
import baostock as bs
import efinance as ef
import pandas as pd

from trade_utils import get_project_path, send_pushplus


THEME_MOMENTUM_BUCKET = "主题右侧动量"
VALUE_LEFT_BUCKET = "价值线左侧确认"
EARNINGS_MAINLINE_BUCKET = "财报后主线候选"
CORE_BUCKET = "低估且高质量"
LOW_VALUE_BUCKET = "低估价值"
HIGH_QUALITY_BUCKET = "高质量趋势"


def parse_args():
    parser = argparse.ArgumentParser(description="推送组合选股结果到 PushPlus")
    parser.add_argument("--portfolio-file", default="", help="最终组合 CSV，默认取回测结果目录最新 *_portfolio_*.csv")
    parser.add_argument("--pool-file", default="", help="宽候选池 CSV，默认从组合文件名推断")
    parser.add_argument("--top", type=int, default=30, help="每次最多展示多少只")
    parser.add_argument("--detail-top", type=int, default=10, help="前N只展示详版，其余展示紧凑名单")
    parser.add_argument("--title", default="", help="自定义推送标题")
    parser.add_argument("--quote-start", default="2026-05-19", help="baostock补行情的起始日期")
    parser.add_argument("--quote-end", default=datetime.now().strftime("%Y-%m-%d"), help="baostock补行情的结束日期")
    parser.add_argument("--prefer-realtime", action="store_true", help="优先尝试东方财富实时行情，失败后再用baostock日线")
    parser.add_argument("--no-push", action="store_true", help="只打印 HTML，不实际推送")
    return parser.parse_args()


def latest_portfolio_file():
    files = glob.glob(get_project_path("回测结果/*_portfolio_*.csv"))
    if not files:
        raise FileNotFoundError("回测结果目录下没有 *_portfolio_*.csv")
    return max(files, key=os.path.getmtime)


def infer_pool_file(portfolio_file):
    path = re.sub(r"_portfolio_[^_/]+_\d+(\.csv)$", r"\1", portfolio_file)
    return path if os.path.exists(path) else ""


def safe_num(value):
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else float(value)


def fmt_num(value, digits=2, suffix=""):
    value = safe_num(value)
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def fmt_pct(value, digits=1, value_is_percent=False):
    value = safe_num(value)
    if value is None:
        return "-"
    if not value_is_percent:
        value *= 100
    return f"{value:.{digits}f}%"


def fmt_text(value):
    if value is None or pd.isna(value):
        return "-"
    text = str(value)
    return text if text else "-"


def short_text(value, limit=180):
    text = fmt_text(value)
    return text if len(text) <= limit else text[:limit] + "..."


def pure_code(code):
    return str(code).replace("sh.", "").replace("sz.", "").zfill(6)


def load_quotes_from_efinance(codes):
    quotes = {}
    for start in range(0, len(codes), 5):
        batch = codes[start:start + 5]
        quote_df = None
        for attempt in range(1, 4):
            try:
                quote_df = ef.stock.get_latest_quote(batch)
                break
            except Exception as exc:
                print(f"东方财富行情接口失败: {exc} | 第 {attempt}/3 次")
                if attempt < 3:
                    time.sleep(2 * attempt)
        if quote_df is None or quote_df.empty:
            continue
        for _, row in quote_df.iterrows():
            code = pure_code(row.get("代码"))
            quotes[code] = {
                "latest_price": safe_num(row.get("最新价")),
                "pct_chg": safe_num(row.get("涨跌幅")),
                "turnover": safe_num(row.get("换手率")),
                "amount": safe_num(row.get("成交额")),
                "quote_date": fmt_text(row.get("最新交易日")),
                "quote_time": fmt_text(row.get("更新时间")),
            }
    missing = [code for code in codes if code not in quotes]
    for code in missing:
        try:
            quote_df = ef.stock.get_latest_quote([code])
        except Exception as exc:
            print(f"{code} 单股行情接口失败: {exc}")
            continue
        if quote_df is None or quote_df.empty:
            continue
        row = quote_df.iloc[0]
        quotes[code] = {
            "latest_price": safe_num(row.get("最新价")),
            "pct_chg": safe_num(row.get("涨跌幅")),
            "turnover": safe_num(row.get("换手率")),
            "amount": safe_num(row.get("成交额")),
            "quote_date": fmt_text(row.get("最新交易日")),
            "quote_time": fmt_text(row.get("更新时间")),
        }
    return quotes


def load_quotes_from_akshare(codes, existing):
    missing = set(codes) - set(existing)
    if not missing:
        return existing
    try:
        quote_df = ak.stock_zh_a_spot()
    except Exception as exc:
        print(f"AkShare行情备用接口失败: {exc}")
        return existing
    if quote_df is None or quote_df.empty:
        return existing
    for _, row in quote_df.iterrows():
        code = pure_code(row.get("代码"))
        if code not in missing:
            continue
        existing[code] = {
            "latest_price": safe_num(row.get("最新价")),
            "pct_chg": safe_num(row.get("涨跌幅")),
            "turnover": safe_num(row.get("换手率")),
            "amount": safe_num(row.get("成交额")),
            "quote_date": datetime.now().strftime("%Y-%m-%d"),
            "quote_time": fmt_text(row.get("时间戳")),
        }
    return existing


def load_quotes_from_baostock(rows, start_date, end_date):
    quotes = {}
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock登录失败: {lg.error_msg}")
        return quotes
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    try:
        for _, row in rows.iterrows():
            code = fmt_text(row.get("code"))
            rs = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            df = rs.get_data()
            if rs.error_code != "0" or df.empty:
                continue
            df.columns = rs.fields
            for col in ["close", "pctChg", "turn", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            if df.empty:
                continue
            latest = df.iloc[-1]
            quotes[pure_code(code)] = {
                "latest_price": safe_num(latest.get("close")),
                "pct_chg": safe_num(latest.get("pctChg")),
                "turnover": safe_num(latest.get("turn")),
                "amount": safe_num(latest.get("amount")),
                "quote_date": fmt_text(latest.get("date")),
                "quote_time": fmt_text(latest.get("date")),
            }
    finally:
        bs.logout()
    return quotes


def load_quotes(rows, quote_start, quote_end, prefer_realtime=False):
    codes = [pure_code(code) for code in rows["code"].tolist()]
    quotes = {}
    if prefer_realtime:
        quotes = load_quotes_from_efinance(codes)
        quotes = load_quotes_from_akshare(codes, quotes)
    daily_quotes = load_quotes_from_baostock(rows, quote_start, quote_end)
    daily_quotes.update(quotes)
    return daily_quotes


def score_direct(value, worst, best):
    value = safe_num(value)
    if value is None:
        return 0.0
    if best == worst:
        return 0.0
    return max(0.0, min(100.0, (value - worst) / (best - worst) * 100.0))


def score_inverse(value, best, worst):
    value = safe_num(value)
    if value is None:
        return 0.0
    if best == worst:
        return 0.0
    return max(0.0, min(100.0, (worst - value) / (worst - best) * 100.0))


def score_calc(row):
    bucket = fmt_text(row.get("selection_bucket"))
    method = fmt_text(row.get("method"))
    total = fmt_num(row.get("total_score"), 1)
    portfolio = fmt_num(row.get("portfolio_score"), 1)
    val = fmt_num(row.get("valuation_score"), 1)
    quality = fmt_num(row.get("quality_score"), 1)
    trend = fmt_num(row.get("trend_score"), 1)
    liquidity = fmt_num(row.get("liquidity_score"), 1)

    if bucket == THEME_MOMENTUM_BUCKET:
        ret60_score = score_direct(row.get("ret60_at_buy"), 0.05, 0.45)
        rel60_score = score_direct(row.get("relative_ret60"), 0.00, 0.30)
        ret20_score = score_direct(row.get("ret20_at_buy"), -0.05, 0.20)
        vol_score = score_direct(row.get("volume_ratio_20_120"), 0.80, 1.80)
        return (
            f"入池分={total}=趋势{trend}*35%+流动性{liquidity}*20%+"
            f"60日动量{ret60_score:.1f}*18%+相对60日{rel60_score:.1f}*12%+"
            f"20日动量{ret20_score:.1f}*8%+量能{vol_score:.1f}*7%；"
            f"组合分={portfolio}=入池分+20/60日阶段位置+相对强弱+量能再排序"
        )

    if bucket == VALUE_LEFT_BUCKET:
        return (
            f"入池分={total}=估值{val}*55%+质量{quality}*35%+流动性{liquidity}*10%；"
            f"组合分={portfolio}=入池分+价值线左侧分类加分+现价/价值+60日趋势+流动性微调"
        )

    if bucket == EARNINGS_MAINLINE_BUCKET:
        growth_score = score_direct(row.get("earnings_yoy"), 0.10, 0.70)
        return (
            f"入池分={total}=估值{val}*20%+业绩增速{growth_score:.1f}*25%+"
            f"质量{quality}*30%+流动性{liquidity}*15%+趋势修复*10%；"
            f"组合分={portfolio}=入池分+分类和交易状态微调"
        )

    if bucket == CORE_BUCKET:
        return (
            f"低估分={fmt_num(row.get('low_value_score'), 1)}，质量趋势分={fmt_num(row.get('high_quality_score'), 1)}；"
            f"入池分={total}=低估分*50%+质量趋势分*50%；组合分={portfolio}"
        )

    if bucket == LOW_VALUE_BUCKET:
        if method == "VALUE":
            formula = f"估值{val}*55%+质量{quality}*25%+趋势{trend}*10%+流动性{liquidity}*10%"
        else:
            formula = f"估值{val}*50%+质量{quality}*30%+趋势{trend}*10%+流动性{liquidity}*10%"
        return f"入池分={total}={formula}；组合分={portfolio}"

    if bucket == HIGH_QUALITY_BUCKET:
        if method == "RIGHT":
            formula = f"质量{quality}*35%+趋势{trend}*45%+流动性{liquidity}*20%"
        else:
            formula = f"估值{val}*5%+质量{quality}*45%+趋势{trend}*35%+流动性{liquidity}*15%"
        return f"入池分={total}={formula}；组合分={portfolio}"

    return f"入池分={total}；组合分={portfolio}；估值/质量/趋势/流动性={val}/{quality}/{trend}/{liquidity}"


def reason_text(row):
    bucket = fmt_text(row.get("selection_bucket"))
    if bucket == THEME_MOMENTUM_BUCKET:
        return (
            f"{fmt_text(row.get('theme'))}右侧动量；20/60日涨幅={fmt_pct(row.get('ret20_at_buy'))}/"
            f"{fmt_pct(row.get('ret60_at_buy'))}；相对60日={fmt_pct(row.get('relative_ret60'))}；"
            f"量能20/120={fmt_num(row.get('volume_ratio_20_120'), 2)}；"
            f"趋势/流动性={fmt_num(row.get('trend_score'), 1)}/{fmt_num(row.get('liquidity_score'), 1)}"
        )
    if bucket == VALUE_LEFT_BUCKET:
        return (
            f"价格仍在基本价值线左侧；现价/价值={fmt_num(row.get('price_to_value'), 2)}；"
            f"扣非同比={fmt_pct(row.get('earnings_yoy'))}；市值={fmt_num(row.get('mktcap'), 1, '亿')}"
        )
    if bucket == EARNINGS_MAINLINE_BUCKET:
        return (
            f"财报后业绩强且价格接近价值线；现价/价值={fmt_num(row.get('price_to_value'), 2)}；"
            f"扣非同比={fmt_pct(row.get('earnings_yoy'))}"
        )
    if bucket == CORE_BUCKET:
        return "低估价值和高质量趋势同时达标。"
    if bucket == LOW_VALUE_BUCKET:
        return f"估值折价或历史估值分位达标；现价/价值={fmt_num(row.get('price_to_value'), 2)}。"
    if bucket == HIGH_QUALITY_BUCKET:
        return f"质量、趋势、流动性组合占优；趋势/流动性={fmt_num(row.get('trend_score'), 1)}/{fmt_num(row.get('liquidity_score'), 1)}。"
    return short_text(row.get("valuation_ref"), 160)


def risk_flags(row):
    flags = []
    bucket = fmt_text(row.get("selection_bucket"))
    ptv = safe_num(row.get("price_to_value"))
    quality = safe_num(row.get("quality_score"))
    trend = safe_num(row.get("trend_score"))
    liquidity = safe_num(row.get("liquidity_score"))
    ret20 = safe_num(row.get("ret20_at_buy"))
    ret60 = safe_num(row.get("ret60_at_buy"))
    volume_ratio = safe_num(row.get("volume_ratio_20_120"))

    if bucket == THEME_MOMENTUM_BUCKET and ptv is not None and ptv > 1.10:
        flags.append(f"高于基本价值线({ptv:.2f}倍)，属于右侧交易")
    if ptv is not None and ptv < 0.30:
        flags.append("价值线折价异常，需复核利润增速可持续性")
    if quality is not None and quality < 60:
        flags.append(f"质量分一般({quality:.1f})")
    if trend is not None and trend < 60:
        flags.append(f"趋势分不足({trend:.1f})")
    if liquidity is not None and liquidity < 50:
        flags.append(f"流动性一般({liquidity:.1f})")
    if ret20 is not None and ret20 > 0.30:
        flags.append(f"20日涨幅偏高({ret20:.1%})")
    if ret60 is not None and ret60 > 0.75:
        flags.append(f"60日涨幅偏高({ret60:.1%})")
    if volume_ratio is not None and volume_ratio > 3.0:
        flags.append(f"量能放大过猛({volume_ratio:.2f})")

    return "；".join(flags) if flags else "正常"


def valuation_line_text(row):
    ptv = fmt_num(row.get("price_to_value"), 2)
    eps = fmt_num(row.get("eps_excl"), 2)
    yoy = fmt_pct(row.get("earnings_yoy"))
    source = fmt_text(row.get("eps_excl_source"))
    return (
        f"基本价值线={fmt_num(row.get('value_line'), 2)}；现价/价值={ptv}；"
        f"扣非EPS={eps}({source})；扣非同比={yoy}"
    )


def esc(value):
    return html.escape(fmt_text(value))


def pct_span(value, value_is_percent=False):
    num = safe_num(value)
    if num is None:
        return "-"
    color = "#d92d20" if num > 0 else "#16803c" if num < 0 else "#667085"
    return f"<span style='color:{color};font-weight:700'>{fmt_pct(num, value_is_percent=value_is_percent)}</span>"


def badge(text, tone="neutral"):
    return f"【{html.escape(str(text))}】"


def amount_text(value):
    value = safe_num(value)
    return "-" if value is None else f"{value / 1e8:.1f}亿"


def score_brief(row):
    bucket = fmt_text(row.get("selection_bucket"))
    base = (
        f"入池{fmt_num(row.get('total_score'), 1)} / 组合{fmt_num(row.get('portfolio_score'), 1)}；"
        f"估/质/趋/流={fmt_num(row.get('valuation_score'), 1)}/"
        f"{fmt_num(row.get('quality_score'), 1)}/"
        f"{fmt_num(row.get('trend_score'), 1)}/"
        f"{fmt_num(row.get('liquidity_score'), 1)}"
    )
    if bucket == THEME_MOMENTUM_BUCKET:
        return (
            f"{base}；动量{fmt_num(row.get('theme_momentum_score'), 1)}；"
            f"20/60日={fmt_pct(row.get('ret20_at_buy'))}/{fmt_pct(row.get('ret60_at_buy'))}；"
            f"相对60日={fmt_pct(row.get('relative_ret60'))}；量能={fmt_num(row.get('volume_ratio_20_120'), 2)}"
        )
    if bucket == CORE_BUCKET:
        return (
            f"{base}；低估分={fmt_num(row.get('low_value_score'), 1)}；"
            f"质量趋势分={fmt_num(row.get('high_quality_score'), 1)}"
        )
    return base


def score_rules_html(portfolio):
    buckets = set(portfolio["selection_bucket"].dropna().astype(str).tolist())
    rules = []
    if THEME_MOMENTUM_BUCKET in buckets:
        rules.append("主题右侧动量：入池分=趋势35%+流动性20%+60日动量18%+相对60日12%+20日动量8%+量能7%；组合分再按20/60日阶段、相对强弱和量能排序。")
    if VALUE_LEFT_BUCKET in buckets:
        rules.append("价值线左侧确认：入池分=估值55%+质量35%+流动性10%；组合分叠加价值线左侧加分、现价/价值、60日趋势和流动性。")
    if CORE_BUCKET in buckets:
        rules.append("低估且高质量：入池分=低估分50%+质量趋势分50%；组合分叠加分类、折价、趋势和流动性。")
    if LOW_VALUE_BUCKET in buckets:
        rules.append("低估价值：VALUE体系约为估值55%+质量25%+趋势10%+流动性10%；PE/PB体系约为估值50%+质量30%+趋势10%+流动性10%。")
    if HIGH_QUALITY_BUCKET in buckets:
        rules.append("高质量趋势：重点看质量、趋势和流动性，估值只作约束或小权重。")
    items = "".join(f"<li>{html.escape(rule)}</li>" for rule in rules)
    return (
        "<h3 style='margin:14px 0 6px'>评分规则</h3>"
        "<ul style='margin:6px 0 10px 18px;padding:0;color:#475467;font-size:13px'>"
        f"{items}</ul>"
    )


def item_html(row, quote, compact=False):
    code = fmt_text(row.get("code"))
    rank = int(safe_num(row.get("final_rank")) or 0)
    name = fmt_text(row.get("name"))
    bucket = fmt_text(row.get("selection_bucket"))
    theme = fmt_text(row.get("theme"))
    latest = quote.get("latest_price") if quote else None
    pct_chg = quote.get("pct_chg") if quote else None
    quote_date = quote.get("quote_date") if quote else "-"
    amount = quote.get("amount") if quote else None
    turnover = quote.get("turnover") if quote else None
    turnover_text = "-" if turnover is None else f"{turnover:.1f}%"
    risk = risk_flags(row)
    risk_color = "#16803c" if risk == "正常" else "#b54708"

    if compact:
        return (
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{html.escape(name)}</b><br>{html.escape(code)}</td>"
            f"<td>{html.escape(bucket)}<br>{html.escape(theme)}</td>"
            f"<td>{fmt_num(latest, 2)}<br>{pct_span(pct_chg, value_is_percent=True)}</td>"
            f"<td>{fmt_num(row.get('portfolio_score'), 1)}<br>入池{fmt_num(row.get('total_score'), 1)}</td>"
            f"<td>{html.escape(short_text(risk, 34))}</td>"
            "</tr>"
        )

    return (
        "<p>"
        f"<b>#{rank} {html.escape(name)}</b> {html.escape(code)}<br>"
        f"{badge(bucket)}{badge(theme)}<br>"
        f"<div>行情：现价 <b>{fmt_num(latest, 2)}</b>，涨跌 {pct_span(pct_chg, value_is_percent=True)}，"
        f"换手 {turnover_text}，成交 {amount_text(amount)}，日 {html.escape(quote_date)}</div>"
        f"<div>价值：价值线 <b>{fmt_num(row.get('value_line'), 2)}</b>，现价/价值 <b>{fmt_num(row.get('price_to_value'), 2)}</b>，"
        f"扣非EPS {fmt_num(row.get('eps_excl'), 2)}，扣非同比 {fmt_pct(row.get('earnings_yoy'))}</div>"
        f"<div>分数：{html.escape(score_brief(row))}</div>"
        f"<div>原因：{html.escape(reason_text(row))}</div>"
        f"<div style='color:{risk_color}'>瑕疵：{html.escape(risk)}</div>"
        "</p><hr>"
    )


def count_chips(series):
    return "　".join(f"{html.escape(str(key))}:{int(value)}" for key, value in series.items())


def summary_table(items):
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td><b>{value}</b></td>"
        "</tr>"
        for label, value in items
    )
    return (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        f"{rows}</table>"
    )


def compact_table(rows, quotes):
    table_rows = "".join(
        item_html(row, quotes.get(pure_code(row.get("code")), {}), compact=True)
        for _, row in rows.iterrows()
    )
    return (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:13px'>"
        "<tr><th>#</th><th>股票</th><th>分类</th><th>行情</th><th>分数</th><th>瑕疵</th></tr>"
        f"{table_rows}</table>"
    )


def build_content(portfolio, pool, quotes, portfolio_file, top, detail_top=15):
    total_count = len(portfolio)
    display_portfolio = portfolio.sort_values("final_rank").head(top).copy()
    detail_portfolio = display_portfolio.head(detail_top).copy()
    compact_portfolio = display_portfolio.iloc[detail_top:].copy()
    quote_dates = sorted({q.get("quote_date") for q in quotes.values() if q.get("quote_date") and q.get("quote_date") != "-"})
    quote_date_text = "、".join(quote_dates) if quote_dates else "-"

    pool_count = len(pool) if pool is not None else "-"
    fail_text = ""
    if pool is not None:
        fail_text = (
            "<p><b>宽候选分布</b><br>"
            f"{count_chips(pool['selection_bucket'].value_counts())}</p>"
        )

    sections = []
    for bucket, rows in detail_portfolio.groupby("selection_bucket", sort=False):
        items = "".join(
            item_html(row, quotes.get(pure_code(row.get("code")), {}))
            for _, row in rows.iterrows()
        )
        sections.append(
            f"<h3>{html.escape(str(bucket))}({len(rows)}只详版)</h3>"
            f"{items}"
        )
    if not compact_portfolio.empty:
        sections.append(
            f"<h3>其余入选({len(compact_portfolio)}只紧凑版)</h3>"
            f"{compact_table(compact_portfolio, quotes)}"
        )

    return (
        "<div>"
        "<h2>当前组合选股结果</h2>"
        f"<p>组合基准日：2026-05-19<br>报告期：2026Q1<br>行情日：{html.escape(quote_date_text)}</p>"
        f"{summary_table([('宽候选池', f'{pool_count}只'), ('最终组合', f'{total_count}只'), ('本次展示', f'{len(display_portfolio)}只')])}"
        "<p>最终组合不是宽候选池全买，而是按组合分、主题集中度和分类约束再次排序。</p>"
        f"{fail_text}"
        "<p><b>最终组合分布</b><br>"
        f"{count_chips(portfolio['selection_bucket'].value_counts())}</p>"
        f"{score_rules_html(portfolio)}"
        f"{''.join(sections)}"
        f"<p>组合CSV：{html.escape(portfolio_file)}</p>"
        "</div>"
    )


def main():
    args = parse_args()
    portfolio_file = args.portfolio_file or latest_portfolio_file()
    pool_file = args.pool_file or infer_pool_file(portfolio_file)
    portfolio = pd.read_csv(portfolio_file)
    pool = pd.read_csv(pool_file) if pool_file and os.path.exists(pool_file) else None
    quotes = load_quotes(portfolio, args.quote_start, args.quote_end, prefer_realtime=args.prefer_realtime)

    content = build_content(portfolio, pool, quotes, portfolio_file, args.top, detail_top=args.detail_top)
    title = args.title or f"{datetime.now().strftime('%Y-%m-%d')} 组合选股复盘({len(portfolio)}只)"

    if args.no_push:
        print(content)
        return

    ok = send_pushplus(title, content)
    print("PUSH_RESULT", ok)


if __name__ == "__main__":
    main()
