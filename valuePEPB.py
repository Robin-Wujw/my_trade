# -*- coding: utf-8 -*-
"""
PE/PB历史低估筛选
适用于基本价值线不适用的行业（周期、金融、公用事业等）

逻辑：
- PE低估：业绩平稳可预期的行业（银行、保险、家电、白酒等）
  → 当前PE ≤ 10年每年最低PE均值（剔除异常值）
- PB低估：利润受周期影响大的重资产行业（化工、有色、钢铁、煤炭、石油、电信等）
  → 当前PB ≤ 10年每年最低PB均值（剔除异常值）

异常值剔除：某年最低值与中位数差异超过2倍的剔除（防止极端年份干扰）
"""
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime

from trade_utils import build_diff_html, get_project_path, load_last_result, save_current_result, send_pushplus

LAST_RESULT_FILE = get_project_path('.valuePEPB_last.json')

# PE适用行业：业绩平稳可预期
PE_KEYWORDS = [
    '货币金融',  # 银行
    '保险',
    '酒、饮料和精制茶',  # 白酒饮料
    '食品制造',
    '电气机械',  # 家电（部分）
    '零售',
    '医药制造',
    '纺织服装',
]

# PB适用行业：利润受周期影响大的重资产行业
PB_KEYWORDS = [
    '钢铁', '煤炭', '有色', '化学原料', '化学纤维', '建材',
    '石油', '采矿', '非金属矿', '黑色金属', '燃料加工',
    '电信', '广播电视和卫星',  # 运营商
    '房地产',
    '电力', '热力', '燃气', '水的生产',  # 公用事业
    '资本市场',  # 券商
    '其他金融',
]


def remove_outliers(values):
    """剔除异常值：与中位数差异超过2倍的"""
    if len(values) < 3:
        return values
    median = np.median(values)
    if median <= 0:
        return values
    filtered = [v for v in values if 0.5 * median <= v <= 2.0 * median]
    return filtered if len(filtered) >= 3 else values

def clamp(value, low=0, high=100):
    try:
        return max(low, min(high, value))
    except Exception:
        return 0

def score_inverse(value, best, worst):
    if value is None or pd.isna(value) or best == worst:
        return 0
    return clamp((worst - value) / (worst - best) * 100)


def get_10yr_low_avg(code, field, start_date, end_date):
    """
    获取10年每年最低PE或PB的平均值（剔除异常值）
    field: 'peTTM' 或 'pbMRQ'
    返回: (当前值, 10年低估均值, 每年明细dict, 当前收盘价, 历史分位) 或空值
    """
    rs = bs.query_history_k_data_plus(
        code, f"date,close,{field}",
        start_date=start_date, end_date=end_date, frequency="d"
    )
    df = rs.get_data()
    if rs.error_code != '0' or df.empty:
        return None, None, None, None, None

    df[field] = pd.to_numeric(df[field], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=[field])
    if len(df) < 250:  # 至少1年数据
        return None, None, None, None, None

    # 只取正值
    df_pos = df[df[field] > 0]
    if df_pos.empty:
        return None, None, None, None, None

    current_val = df[field].iloc[-1]
    current_close = df['close'].iloc[-1]
    if pd.isna(current_val) or current_val <= 0:
        return None, None, None, None, None

    # 每年最低值
    df_pos = df_pos.copy()
    df_pos['year'] = df_pos['date'].str[:4]
    yearly_min = df_pos.groupby('year')[field].min()

    # 去掉当前不完整年份如果数据不足60天
    current_year = str(datetime.now().year)
    if current_year in yearly_min.index:
        days_this_year = len(df_pos[df_pos['year'] == current_year])
        if days_this_year < 60:
            yearly_min = yearly_min.drop(current_year)

    if len(yearly_min) < 3:
        return None, None, None, None, None

    # 剔除异常值
    values = yearly_min.values.tolist()
    filtered = remove_outliers(values)
    avg_low = np.mean(filtered)
    percentile = float((df_pos[field] <= current_val).mean())

    detail = {str(y): round(v, 2) for y, v in yearly_min.items()}
    return round(current_val, 2), round(avg_low, 2), detail, round(current_close, 2) if not pd.isna(current_close) else None, percentile

def calc_pepb_score(ratio, percentile):
    ratio_score = score_inverse(ratio, best=0.60, worst=1.35)
    percentile_score = score_inverse(percentile, best=0.05, worst=0.60)
    return round(ratio_score * 0.65 + percentile_score * 0.35, 1)


def main():
    lg = bs.login()
    if lg.error_code != '0':
        print('登录失败:', lg.error_msg)
        return

    # 获取最新交易日
    rs_dates = bs.query_trade_dates(
        start_date=(pd.Timestamp.now() - pd.DateOffset(days=15)).strftime('%Y-%m-%d'),
        end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
    )
    df_dates = rs_dates.get_data()
    df_dates.columns = rs_dates.fields
    df_dates = df_dates[df_dates['is_trading_day'] == '1']
    today_str, df_stocks = None, pd.DataFrame()
    for d in reversed(df_dates['calendar_date'].tolist()):
        rs = bs.query_all_stock(day=d)
        tmp = rs.get_data()
        if not tmp.empty:
            tmp.columns = rs.fields
            today_str = d
            df_stocks = tmp
            break
    if df_stocks.empty:
        print("无法获取股票列表")
        bs.logout()
        return
    print(f"最新交易日: {today_str}")

    # 过滤A股
    mask = (df_stocks['code'].str.startswith('sh.60') | df_stocks['code'].str.startswith('sz.00') |
            df_stocks['code'].str.startswith('sz.30') | df_stocks['code'].str.startswith('sh.68'))
    df_stocks = df_stocks[mask & ~df_stocks['tradeStatus'].eq('0')]
    df_stocks = df_stocks[~df_stocks['code_name'].str.contains(r'ST|\*ST')]
    print(f"过滤后A股: {len(df_stocks)}")

    # 获取行业分类
    rs_ind = bs.query_stock_industry()
    df_ind = rs_ind.get_data()
    df_ind.columns = rs_ind.fields

    # 分类：PE适用 / PB适用
    is_pe = df_ind['industry'].str.contains('|'.join(PE_KEYWORDS), na=False)
    is_pb = df_ind['industry'].str.contains('|'.join(PB_KEYWORDS), na=False)
    pe_codes = set(df_ind[is_pe]['code'].tolist())
    pb_codes = set(df_ind[is_pb]['code'].tolist())
    # PB优先（如果同时匹配PE和PB关键词，用PB）
    pe_codes = pe_codes - pb_codes

    df_pe = df_stocks[df_stocks['code'].isin(pe_codes)]
    df_pb = df_stocks[df_stocks['code'].isin(pb_codes)]
    print(f"PE适用: {len(df_pe)} 只, PB适用: {len(df_pb)} 只")

    # 10年数据范围
    start_10yr = (pd.Timestamp.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    end_date = today_str

    results = []

    # === PE低估筛选 ===
    print(f"\n--- PE低估筛选 ({len(df_pe)}只) ---")
    for idx, (_, row) in enumerate(df_pe.iterrows()):
        code, name = row['code'], row['code_name']
        try:
            current_pe, avg_low_pe, detail, close, percentile = get_10yr_low_avg(code, 'peTTM', start_10yr, end_date)
            if current_pe is None or avg_low_pe is None or close is None or percentile is None:
                continue
            ratio = round(current_pe / avg_low_pe, 2)
            if current_pe > avg_low_pe and percentile > 0.15:
                continue

            # 查行业
            ind_row = df_ind[df_ind['code'] == code]
            ind_name = ind_row.iloc[0]['industry'] if not ind_row.empty else ''

            score = calc_pepb_score(ratio, percentile)
            results.append({
                'code': code, 'name': name, 'close': close,
                'method': 'PE', 'current': current_pe, 'low_avg': avg_low_pe,
                'ratio': ratio, 'percentile': round(percentile * 100, 1), 'score': score, 'industry': ind_name
            })
            print(f"  ✅ {code} {name} | PE={current_pe} 低估均值={avg_low_pe} 比值={ratio} 分位={percentile:.1%} 分数={score}")
        except Exception as e:
            continue
        if (idx + 1) % 100 == 0:
            print(f"  PE进度: {idx+1}/{len(df_pe)}")

    # === PB低估筛选 ===
    print(f"\n--- PB低估筛选 ({len(df_pb)}只) ---")
    for idx, (_, row) in enumerate(df_pb.iterrows()):
        code, name = row['code'], row['code_name']
        try:
            current_pb, avg_low_pb, detail, close, percentile = get_10yr_low_avg(code, 'pbMRQ', start_10yr, end_date)
            if current_pb is None or avg_low_pb is None or close is None or percentile is None:
                continue
            ratio = round(current_pb / avg_low_pb, 2)
            if current_pb > avg_low_pb and percentile > 0.15:
                continue

            ind_row = df_ind[df_ind['code'] == code]
            ind_name = ind_row.iloc[0]['industry'] if not ind_row.empty else ''

            score = calc_pepb_score(ratio, percentile)
            results.append({
                'code': code, 'name': name, 'close': close,
                'method': 'PB', 'current': current_pb, 'low_avg': avg_low_pb,
                'ratio': ratio, 'percentile': round(percentile * 100, 1), 'score': score, 'industry': ind_name
            })
            print(f"  ✅ {code} {name} | PB={current_pb} 低估均值={avg_low_pb} 比值={ratio} 分位={percentile:.1%} 分数={score}")
        except Exception as e:
            continue
        if (idx + 1) % 100 == 0:
            print(f"  PB进度: {idx+1}/{len(df_pb)}")

    bs.logout()
    print(f"\n共筛选出 {len(results)} 只")

    # 对比上次结果
    last_dict = load_last_result(LAST_RESULT_FILE)
    diff_html = build_diff_html(last_dict, results) if last_dict else "<p>首次运行，无历史对比</p>"
    save_current_result(LAST_RESULT_FILE, today_str, results)

    if not results:
        content = "今日无PE/PB低估股票" + diff_html
        send_pushplus(f"{today_str} PE/PB低估筛选", content)
        return

    # 按因子分排序（比值越低、历史分位越低，分数越高）
    results.sort(key=lambda x: x['score'], reverse=True)

    # 分PE和PB两个表
    pe_results = [r for r in results if r['method'] == 'PE']
    pb_results = [r for r in results if r['method'] == 'PB']

    content_parts = [diff_html]

    if pe_results:
        rows = "".join(
            f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r['close']}</td>"
            f"<td>{r['current']}</td><td>{r['low_avg']}</td><td>{r['ratio']}</td>"
            f"<td>{r['percentile']}%</td><td>{r['score']}</td><td>{r['industry']}</td></tr>"
            for r in pe_results
        )
        content_parts.append(
            f"<h3>PE低估({len(pe_results)}只)</h3>"
            f"<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            f"<tr><th>代码</th><th>名称</th><th>现价</th><th>当前PE</th><th>10年低估PE</th><th>比值</th><th>历史分位</th><th>分数</th><th>行业</th></tr>"
            f"{rows}</table>"
        )

    if pb_results:
        rows = "".join(
            f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r['close']}</td>"
            f"<td>{r['current']}</td><td>{r['low_avg']}</td><td>{r['ratio']}</td>"
            f"<td>{r['percentile']}%</td><td>{r['score']}</td><td>{r['industry']}</td></tr>"
            for r in pb_results
        )
        content_parts.append(
            f"<h3>PB低估({len(pb_results)}只)</h3>"
            f"<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            f"<tr><th>代码</th><th>名称</th><th>现价</th><th>当前PB</th><th>10年低估PB</th><th>比值</th><th>历史分位</th><th>分数</th><th>行业</th></tr>"
            f"{rows}</table>"
        )

    content_parts.append(
        "<p>PE低估：当前PE ≤ 10年每年最低PE均值，或历史分位≤15% | 适用银行/保险/消费等业绩平稳行业</p>"
        "<p>PB低估：当前PB ≤ 10年每年最低PB均值，或历史分位≤15% | 适用周期/金融/公用事业等重资产行业</p>"
    )

    content = ''.join(content_parts)
    title = f"{today_str} PE/PB低估筛选({len(results)}只)"
    print("推送成功" if send_pushplus(title, content) else "推送失败")


if __name__ == '__main__':
    main()
