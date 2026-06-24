import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

from trade_utils import build_diff_html, call_with_retry, get_project_path, load_last_result, save_current_result, send_pushplus

LAST_RESULT_FILE = get_project_path('.stockpush_last.json')

def analyze_stock(stock_code):
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=240*3)).strftime('%Y%m%d')
    df = call_with_retry(
        ak.stock_zh_a_hist,
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
        label=f"{stock_code} 行情",
    )
    if df is None or df.empty:
        return None
    required_columns = {'收盘', '成交额'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{stock_code} 行情字段缺失: {required_columns - set(df.columns)}")
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
    df = df.dropna(subset=['收盘', '成交额'])
    if df.empty:
        return None

    ma_days = [5, 10, 20, 60, 120, 240]
    for ma in ma_days:
        df[f'MA{ma}'] = df['收盘'].rolling(window=ma).mean()
        df[f'VolMA{ma}'] = df['成交额'].rolling(window=ma).mean()

    latest = df.iloc[-1]
    close_today = latest['收盘']
    vol_today = latest['成交额']

    # 均量扣抵筛选：今日收盘 > 扣抵价（ma+1天前），今日成交额 > 扣抵量
    passed = []
    for ma in ma_days:
        deduct_idx = -(ma + 1)
        if len(df) < ma + 2:
            continue
        deduct_price = df.iloc[deduct_idx]['收盘']
        deduct_vol = df.iloc[deduct_idx]['成交额']
        price_up = close_today > deduct_price
        vol_up = vol_today > deduct_vol
        if price_up and vol_up:
            passed.append(f'MA{ma}')

    if not passed:
        return None

    name = get_stock_name(stock_code)
    lines = [f"<b>{name}({stock_code})</b>", f"收盘: {close_today:.2f}  成交额: {vol_today/1e8:.2f}亿", f"均量扣抵双上扬: {', '.join(passed)}"]
    for ma in ma_days:
        deduct_idx = -(ma + 1)
        if len(df) < ma + 2:
            continue
        lines.append(f"MA{ma}={latest[f'MA{ma}']:.2f} | 扣抵价={df.iloc[deduct_idx]['收盘']:.2f} | 扣抵量={df.iloc[deduct_idx]['成交额']/1e8:.2f}亿")
    return {"code": stock_code, "name": name, "html": "<br>".join(lines)}

def get_stock_name(stock_code):
    info = call_with_retry(ak.stock_individual_info_em, stock_code, label=f"{stock_code} 个股信息")
    if info is None or info.empty:
        return stock_code
    if {'item', 'value'}.issubset(info.columns):
        for key in ('股票简称', '证券简称', '股票名称', '简称'):
            row = info[info['item'].astype(str) == key]
            if not row.empty:
                return str(row.iloc[0]['value'])
    try:
        return str(info.iloc[1]['value'])
    except Exception:
        return stock_code

if __name__ == '__main__':
    stock_codes = ["300558","300115","603650","600763","603659","000063","300003","603290","603986","002294","002555","603259"]
    results = []
    for code in stock_codes:
        try:
            r = analyze_stock(code)
            if r:
                results.append(r)
                print(f"{code} 通过筛选")
            else:
                print(f"{code} 未通过均量扣抵筛选")
        except Exception as e:
            print(f"{code} 出错: {e}")

    # 对比上次结果
    last_dict = load_last_result(LAST_RESULT_FILE)
    diff_html = build_diff_html(last_dict, results) if last_dict else "<p>首次运行，无历史对比</p>"
    today_str = datetime.today().strftime('%Y-%m-%d')
    save_current_result(LAST_RESULT_FILE, today_str, results)

    if results:
        content = diff_html + "<hr>" + "<hr>".join([r['html'] for r in results])
        title = f"{today_str} 均量扣抵筛选结果({len(results)}只)"
        print("推送成功" if send_pushplus(title, content) else "推送失败")
    else:
        content = "今日无符合条件股票" + diff_html
        print("无符合条件股票")
        send_pushplus(f"{today_str} 均量扣抵", content)
