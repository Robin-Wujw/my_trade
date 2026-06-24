# -*- coding: utf-8 -*-
import akshare as ak
import baostock as bs
import math
import pandas as pd

from factorStock import comparable_excl_eps, get_excl_eps_yoy, infer_excl_eps, parse_float
from trade_utils import build_diff_html, get_project_path, load_last_result, save_current_result, send_pushplus

LAST_RESULT_FILE = get_project_path('.valueStock_last.json')

# 排除行业关键词：利润周期性强、不可用未来现金流折现外推的行业
EXCLUDE_KEYWORDS = [
    # 周期资源
    '钢铁', '煤炭', '有色', '化工', '建材', '航运', '石油', '采矿', '水泥',
    '黑色金属', '非金属矿', '化学原料', '化学纤维', '燃料加工',
    # 金融
    '银行', '保险', '货币金融', '资本市场', '金融服务', '其他金融', '信托', '证券',
    # 房地产
    '房地产',
    # 农林牧渔（生物周期）
    '农业', '林业', '畜牧', '渔业', '养殖', '农副食品',
    # 交运（强周期）
    '航空运输', '水上运输', '铁路运输', '道路运输',
    # 公用事业（政策定价，增速低）
    '电力', '热力', '燃气', '水的生产',
    # 电信运营商（公用事业属性，增速低且不稳定）
    '电信', '广播电视和卫星',
    # 建筑工程（项目制，利润波动大）
    '房屋建筑', '土木工程', '建筑安装', '建筑装饰',
    # 其他不适用
    '综合', '废弃资源', '开采专业', '公共设施管理',
]

def clamp(value, low=0, high=100):
    try:
        return max(low, min(high, value))
    except Exception:
        return 0

def score_direct(value, worst, best):
    if value is None or pd.isna(value) or best == worst:
        return 0
    return clamp((value - worst) / (best - worst) * 100)

def calc_value_score(discount, mktcap, eps_excl):
    """用于排序的轻量因子分：折价为主，盈利质量和规模为辅。"""
    discount_score = score_direct(discount, 0, 50)
    eps_score = score_direct(eps_excl, 0.3, 2.0)
    size_score = score_direct(math.log10(mktcap) if mktcap and mktcap > 0 else None, 2.3, 4.0)
    return round(discount_score * 0.60 + eps_score * 0.25 + size_score * 0.15, 1)

def parse_yi(s):
    """解析'14.96亿'格式为float"""
    try:
        s = str(s).strip()
        if '亿' in s: return float(s.replace('亿', '')) * 1e8
        if '万' in s: return float(s.replace('万', '')) * 1e4
        return float(s)
    except:
        return None

def parse_pct(s):
    try: return float(str(s).replace('%', '')) / 100
    except: return None

def get_value_line_and_mktcap(symbol, close):
    """
    基本价值线 = 最新报告期BVPS + 最新年报扣非EPS × (1 + 最新报告期扣非EPS同比) × 10
    扣非EPS 优先使用直接披露字段；Q1 节点遇到年报送转时按月度加权股本转成复盘可比口径。
    同时利用反推的总股本计算市值，避免依赖 baostock 的 totalShare 字段

    质量过滤（基于"产业趋势向上、规模够大"的适用条件）：
    1. 扣非EPS <= 0 跳过（亏损，负负得正失真）
    2. 扣非EPS < 0.3 跳过（微利公司，增速放大后失真）
    3. 最新报告期扣非同比 < 0 跳过（趋势拐头，不适用增长外推）
    4. 近3年年报扣非要求连续增长（每年都比前一年高），确保产业趋势向上
    """
    df_q = ak.stock_financial_abstract_ths(symbol=symbol, indicator='按报告期')
    df_q = df_q[df_q['扣非净利润'] != False].copy()
    if df_q.empty: return None, None, None, None
    df_q['报告期_dt'] = pd.to_datetime(df_q['报告期'], errors='coerce')
    df_q = df_q.dropna(subset=['报告期_dt']).sort_values('报告期_dt')
    if df_q.empty: return None, None, None, None

    latest = df_q.iloc[-1]
    bvps = parse_yi(latest['每股净资产'])
    yoy_metrics = get_excl_eps_yoy(df_q, latest)
    if bvps is None or not yoy_metrics: return None, None, None, None
    yoy = yoy_metrics['yoy']

    df_annual = df_q[df_q['报告期'].astype(str).str.endswith('12-31')]
    if df_annual.empty: return None, None, None, None
    annual = df_annual.iloc[-1]
    net_profit = parse_yi(annual['净利润'])
    basic_eps = parse_float(annual['基本每股收益'])
    raw_eps_excl = infer_excl_eps(annual)
    eps_excl, _ = comparable_excl_eps(symbol, annual['报告期'], latest['报告期'], raw_eps_excl)
    if not net_profit or basic_eps is None or basic_eps <= 0 or eps_excl is None: return None, None, None, None

    total_share = net_profit / basic_eps

    # 过滤1: 亏损跳过
    if eps_excl <= 0: return None, None, None, None
    # 过滤2: 微利跳过（扣非EPS太低，增速放大后价值线失真）
    if eps_excl < 0.3: return None, None, None, None

    # 过滤3: 最新报告期扣非同比为负则跳过（趋势已拐头，不适用未来增长外推）
    if yoy < 0: return None, None, None, None

    # 过滤4: 近3年年报扣非要求连续增长（每年都比前一年高），确保产业趋势向上
    if len(df_annual) >= 3:
        recent_annuals = df_annual.tail(3)
        excl_list = [parse_yi(r['扣非净利润']) for _, r in recent_annuals.iterrows()]
        if all(e is not None for e in excl_list):
            # 任何一年扣非为负（亏损），利润不稳定
            if any(e <= 0 for e in excl_list):
                return None, None, None, None
            # 要求严格递增（每年都比前一年高）
            for i in range(len(excl_list) - 1):
                if excl_list[i] >= excl_list[i + 1]:
                    return None, None, None, None
    elif len(df_annual) >= 2:
        recent_annuals = df_annual.tail(2)
        excl_list = [parse_yi(r['扣非净利润']) for _, r in recent_annuals.iterrows()]
        if all(e is not None for e in excl_list):
            if any(e <= 0 for e in excl_list):
                return None, None, None, None
            if excl_list[0] >= excl_list[1]:
                return None, None, None, None

    value_line = bvps + eps_excl * (1 + yoy) * 10
    if value_line <= 0: return None, None, None, None
    mktcap = close * total_share / 1e8
    return value_line, bvps, eps_excl, mktcap

def main():
    lg = bs.login()
    if lg.error_code != '0': print('登录失败:', lg.error_msg); return

    rs_dates = bs.query_trade_dates(
        start_date=(pd.Timestamp.now() - pd.DateOffset(days=15)).strftime('%Y-%m-%d'),
        end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
    )
    df_dates = rs_dates.get_data(); df_dates.columns = rs_dates.fields
    df_dates = df_dates[df_dates['is_trading_day'] == '1']
    today_str, df_stocks = None, pd.DataFrame()
    for d in reversed(df_dates['calendar_date'].tolist()):
        rs = bs.query_all_stock(day=d); tmp = rs.get_data()
        if not tmp.empty: tmp.columns = rs.fields; today_str = d; df_stocks = tmp; break
    if df_stocks.empty: print("无法获取股票列表"); bs.logout(); return
    print(f"最新交易日: {today_str}")

    mask = (df_stocks['code'].str.startswith('sh.60') | df_stocks['code'].str.startswith('sz.00') |
            df_stocks['code'].str.startswith('sz.30') | df_stocks['code'].str.startswith('sh.68'))
    df_stocks = df_stocks[mask & ~df_stocks['tradeStatus'].eq('0')]
    df_stocks = df_stocks[~df_stocks['code_name'].str.contains(r'ST|\*ST')]
    print(f"过滤后A股: {len(df_stocks)}")

    rs_ind = bs.query_stock_industry(); df_ind = rs_ind.get_data(); df_ind.columns = rs_ind.fields
    is_excluded = df_ind['industry'].str.contains('|'.join(EXCLUDE_KEYWORDS), na=False)
    valid_codes = set(df_ind[~is_excluded]['code'].tolist())
    df_stocks = df_stocks[df_stocks['code'].isin(valid_codes)]
    print(f"排除周期/金融/地产等后: {len(df_stocks)}")

    results = []
    total = len(df_stocks)
    for idx, (_, row) in enumerate(df_stocks.iterrows()):
        code, name = row['code'], row['code_name']
        symbol = code.replace('sh.', '').replace('sz.', '')
        try:
            rs_k = bs.query_history_k_data_plus(code, "close", start_date=today_str, end_date=today_str, frequency="d")
            df_k = rs_k.get_data()
            if rs_k.error_code != '0' or df_k.empty: continue
            close = pd.to_numeric(df_k.iloc[0]['close'], errors='coerce')
            if pd.isna(close) or close <= 0: continue

            value_line, bvps, eps_excl, mktcap = get_value_line_and_mktcap(symbol, close)
            if value_line is None or value_line <= 0: continue
            if mktcap is None or mktcap < 200: continue
            if close > value_line: continue

            discount = round((value_line - close) / value_line * 100, 1)
            score = calc_value_score(discount, mktcap, eps_excl)
            risk = '折价异常需复核' if discount > 70 else '正常'
            results.append({'code': code, 'name': name, 'close': close,
                            'value_line': round(value_line, 2), 'discount': discount, 'mktcap': round(mktcap, 1),
                            'eps_excl': round(eps_excl, 2), 'score': score, 'risk': risk})
            print(f"  {code} {name} | 现价={close} 价值线={value_line:.2f} 折价={discount}% 市值={mktcap:.1f}亿 分数={score}")
        except Exception as e:
            continue
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx+1}/{total}, 已入选 {len(results)} 只")

    bs.logout()
    print(f"\n共筛选出 {len(results)} 只")

    # 对比上次结果
    last_dict = load_last_result(LAST_RESULT_FILE)
    diff_html = build_diff_html(last_dict, results) if last_dict else "<p>首次运行，无历史对比</p>"
    # 保存本次结果
    save_current_result(LAST_RESULT_FILE, today_str, results)

    if not results:
        content = "今日无符合条件股票（市值>200亿 + 近3年扣非连续增长 + 股价≤基本价值线）" + diff_html
        send_pushplus(f"{today_str} 基本价值线筛选", content)
        return

    df_r = pd.DataFrame(results).sort_values('score', ascending=False)
    rows = "".join(
        f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r['close']}</td>"
        f"<td>{r['value_line']}</td><td>{r['discount']}%</td><td>{r['eps_excl']}</td>"
        f"<td>{r['mktcap']}亿</td><td>{r['score']}</td><td>{r['risk']}</td></tr>"
        for _, r in df_r.iterrows()
    )
    content = (
        f"{diff_html}"
        f"<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        f"<tr><th>代码</th><th>名称</th><th>现价</th><th>价值线</th><th>折价</th><th>扣非EPS</th><th>市值</th><th>分数</th><th>风险</th></tr>"
        f"{rows}</table>"
        f"<p>公式：最新BVPS + 年报扣非EPS×(1+最新同比)×10 | 市值>200亿 | 近3年扣非连续增长 | 排除周期/金融/地产等 | 分数=折价60%+扣非EPS25%+规模15%</p>"
    )
    title = f"{today_str} 基本价值线筛选({len(results)}只)"
    print("推送成功" if send_pushplus(title, content) else "推送失败")

if __name__ == '__main__':
    main()
