# -*- coding: utf-8 -*-
import baostock as bs
import pandas as pd
from datetime import datetime
import time
import os
import multiprocessing
import akshare as ak

from factorStock import comparable_excl_eps, get_excl_eps_yoy, infer_excl_eps
from trade_utils import build_diff_html, get_project_path, load_last_result, save_current_result, send_pushplus

"""
说明：
【最终版】 - 按指定代码筛选
核心修改：
1. 遵照您的最新指示，调整了股票代码的筛选规则，只保留以 60, 00, 30 开头的股票，
   即沪市主板、深市主板/中小板、创业板。
"""

LAST_RESULT_FILE = get_project_path('.selectStock_last.json')

# --- 初始化函数，将在每个子进程启动时被调用一次 ---
def init_worker():
    bs.login()

def get_latest_quarter(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year, month = d.year, d.month
    if month <= 4: return year - 1, 4
    elif month <= 8: return year, 1
    elif month <= 10: return year, 2
    else: return year, 3

def parse_yi(s):
    """解析'14.96亿'或'3500万'格式为float"""
    try:
        s = str(s).strip()
        if '亿' in s: return float(s.replace('亿', '')) * 1e8
        if '万' in s: return float(s.replace('万', '')) * 1e4
        return float(s)
    except:
        return None

def get_value_line(symbol):
    try:
        df = ak.stock_financial_abstract_ths(symbol=symbol, indicator='按报告期')
        df = df[df['扣非净利润'] != False].copy()
        if df.empty: return None
        df['报告期_dt'] = pd.to_datetime(df['报告期'], errors='coerce')
        df = df.dropna(subset=['报告期_dt']).sort_values('报告期_dt')
        if df.empty: return None
        latest = df.iloc[-1]
        bvps = parse_yi(latest['每股净资产'])
        yoy_metrics = get_excl_eps_yoy(df, latest)
        if bvps is None or not yoy_metrics: return None
        yoy = yoy_metrics['yoy']
        df_a = df[df['报告期'].astype(str).str.endswith('12-31')]
        if df_a.empty: return None
        annual = df_a.iloc[-1]
        raw_eps_excl = infer_excl_eps(annual)
        eps_excl, _ = comparable_excl_eps(symbol, annual['报告期'], latest['报告期'], raw_eps_excl)
        if eps_excl is None: return None
        # 扣非EPS为负（亏损）时价值线公式会失真（负负得正导致虚高），直接跳过
        if eps_excl <= 0: return None
        vl = round(bvps + eps_excl * (1 + yoy) * 10, 2)
        return vl if vl > 0 else None
    except:
        return None

def get_consecutive_profit_data(code, start_year, start_quarter, count=5):
    all_profit_data, all_fields = [], []
    current_year, current_quarter = start_year, start_quarter
    for _ in range(count):
        rs = bs.query_profit_data(code=code, year=current_year, quarter=current_quarter)
        if rs.error_code == '0' and rs.next():
            all_profit_data.append(rs.get_row_data())
            if not all_fields: all_fields = rs.fields
        if current_quarter == 1:
            current_quarter, current_year = 4, current_year - 1
        else:
            current_quarter -= 1
    if all_profit_data: return pd.DataFrame(all_profit_data, columns=all_fields)
    return pd.DataFrame()

def save_to_excel(dataframe):
    if dataframe.empty:
        print("\n没有筛选到任何股票，不创建Excel文件。")
        return
    output_dir = "选股结果"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"stock_selection_{timestamp}.xlsx")
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='SelectedStocks')
        worksheet = writer.sheets['SelectedStocks']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except: pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    print(f"\n选股结果已成功保存到文件: {filename}")

def process_single_stock(args):
    code, stock_name, all_industries_df, year, quarter, today_str = args
    try:
        kline_fields = "date,code,close,preclose,volume"
        start_date = (pd.to_datetime(today_str) - pd.DateOffset(years=1, days=30)).strftime('%Y-%m-%d')
        rs_kline = bs.query_history_k_data_plus(code, kline_fields, start_date=start_date, end_date=today_str, frequency="d", adjustflag="2")
        df_kline = rs_kline.get_data()
        if rs_kline.error_code != '0' or len(df_kline) < 250: return None
        df_kline[['close', 'preclose', 'volume']] = df_kline[['close', 'preclose', 'volume']].apply(pd.to_numeric, errors='coerce')
        df_kline.dropna(inplace=True)
        if len(df_kline) < 250: return None

        vol_30d = df_kline['volume'].tail(30).mean()
        vol_250d = df_kline['volume'].tail(250).mean()
        if vol_30d <= vol_250d * 1.2: return None

        recent_df = df_kline.tail(60).copy()
        recent_df['change'] = recent_df['close'] - recent_df['preclose']
        vol_up = recent_df[recent_df['change'] > 0]['volume'].mean()
        vol_down = recent_df[recent_df['change'] < 0]['volume'].mean()
        if not (pd.notna(vol_up) and pd.notna(vol_down) and vol_up > vol_down): return None

        industry_info = all_industries_df[all_industries_df['code'] == code]
        if industry_info.empty or not industry_info.iloc[0]['industry']: return None

        profit_df = get_consecutive_profit_data(code, year, quarter, count=5)
        if len(profit_df) < 5: return None

        profit_df['epsTTM'] = pd.to_numeric(profit_df['epsTTM'], errors='coerce')
        profit_df['roeAvg'] = pd.to_numeric(profit_df['roeAvg'], errors='coerce')
        profit_df.dropna(subset=['epsTTM', 'roeAvg'], inplace=True)
        if len(profit_df) < 5: return None

        latest_eps = profit_df['epsTTM'].iloc[0]
        yoy_eps = profit_df['epsTTM'].iloc[4]
        latest_roe = profit_df['roeAvg'].iloc[0]

        if latest_eps <= 0 or latest_eps <= yoy_eps: return None
        if latest_roe < 0.10: return None

        symbol = code.replace('sh.', '').replace('sz.', '')
        value_line = get_value_line(symbol)
        print(f"  {code} {stock_name} 入选")
        return {
            'code': code, 'name': stock_name,
            'ROE(%)': f"{latest_roe*100:.2f}",
            'EPS(TTM)': f"{latest_eps:.2f}",
            '基本价值线': value_line if value_line else 'N/A',
            '现价vs价值线': f"{df_kline['close'].iloc[-1]:.2f}/{value_line}" if value_line else 'N/A'
        }
    except Exception as e:
        print(f"  {code} 处理出错: {e}")
        return None

def main():
    print("开始准备数据...")
    lg = bs.login()
    if lg.error_code != '0': print('Baostock主进程登录失败:', lg.error_msg); return
    
    rs_trade_dates = bs.query_trade_dates(start_date=(pd.Timestamp.now() - pd.DateOffset(days=15)).strftime('%Y-%m-%d'), end_date=pd.Timestamp.now().strftime('%Y-%m-%d'))
    df_trade_dates = rs_trade_dates.get_data()
    if rs_trade_dates.error_code != '0' or df_trade_dates.empty:
        print("错误：未能找到有效的最近交易日。"); bs.logout(); return
    df_trade_dates.columns = rs_trade_dates.fields
    if 'is_trading_day' in df_trade_dates.columns:
        df_trade_dates = df_trade_dates[df_trade_dates['is_trading_day'] == '1']
    if df_trade_dates.empty:
        print("错误：在返回的日期中未能找到有效的交易日。"); bs.logout(); return

    trade_dates = df_trade_dates['calendar_date'].tolist()
    stock_df = pd.DataFrame()
    today_str = None
    for d in reversed(trade_dates):
        rs_all_stock = bs.query_all_stock(day=d)
        stock_df = rs_all_stock.get_data()
        if not stock_df.empty:
            stock_df.columns = rs_all_stock.fields
            today_str = d
            break
    if stock_df.empty or today_str is None:
        print("错误：无法获取股票列表数据。"); bs.logout(); return
    print(f"找到最新交易日为: {today_str}，将使用此日期进行查询。")
    
    is_sh_60 = stock_df['code'].str.startswith('sh.60')
    is_sz_00 = stock_df['code'].str.startswith('sz.00')
    is_sz_30 = stock_df['code'].str.startswith('sz.30')
    is_sh_68 = stock_df['code'].str.startswith('sh.68')
    stock_df = stock_df[is_sh_60 | is_sz_00 | is_sz_30 | is_sh_68]

    stock_df = stock_df[~stock_df['tradeStatus'].eq('0')]
    stock_df = stock_df[~stock_df['code_name'].str.contains(r'ST|\*ST')]
    all_stocks_df = stock_df
    print(f"精确过滤后，获取到 {len(all_stocks_df)} 只正常交易的A股。")

    print("正在一次性获取所有股票的行业分类信息...")
    rs_all_industries = bs.query_stock_industry()
    all_industries_df = rs_all_industries.get_data(); all_industries_df.columns = rs_all_industries.fields
    print("全市场行业数据获取完毕！")
    
    year, quarter = get_latest_quarter(today_str)
    print(f"将统一查询以 {year} 年第 {quarter} 季度为终点的财报数据（保守策略）。")
    print(f"筛选条件: ROE(计算值)>10%, EPS同比增长, 放量上涨, 低于行业估值")

    tasks = []
    for _, row in all_stocks_df.iterrows():
        tasks.append((row['code'], row['code_name'], all_industries_df, year, quarter, today_str))

    bs.logout() 

    print(f"\n开始并行处理 {len(tasks)} 只股票...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=4, initializer=init_worker) as pool:
        results = pool.map(process_single_stock, tasks)

    end_time = time.time()
    print(f"\n处理完成！总耗时: {end_time - start_time:.2f} 秒")
    
    selected_stocks = [res for res in results if res is not None]

    # 对比上次结果
    last_dict = load_last_result(LAST_RESULT_FILE)
    diff_html = build_diff_html(last_dict, selected_stocks) if last_dict else "<p>首次运行，无历史对比</p>"
    save_current_result(LAST_RESULT_FILE, today_str, selected_stocks)

    print("\n\n===================== 选股策略执行完毕 =====================")
    if not selected_stocks:
        print("本次未筛选出符合所有条件的股票。")
        content = "今日未筛选出符合所有条件的股票。" + diff_html
        send_pushplus(f"{today_str} 选股报告", content)
    else:
        result_df = pd.DataFrame(selected_stocks)
        print(f"共筛选出 {len(selected_stocks)} 只符合条件的股票：")
        print(result_df.to_string())

        title = f"{today_str} 选股结果: {len(selected_stocks)}只"
        content = diff_html + result_df.to_html(index=False, justify='center')
        send_pushplus(title, content)

    print("==========================================================")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
