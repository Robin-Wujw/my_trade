"""Direct Eastmoney HTTP fallbacks for sector data."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import requests

from stock_research.core.paths import PATHS


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}

def _get(url, *, params, timeout=15):
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


@lru_cache(maxsize=1)
def stock_board_industry_name_em() -> pd.DataFrame:
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "5000",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:90 t:2 f:!50",
        "fields": "f3,f4,f6,f8,f12,f14,f20,f104,f105,f128,f136",
    }
    payload = _get(url, params=params)
    rows = payload.get("data", {}).get("diff", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "f12": "板块代码",
            "f14": "板块名称",
            "f3": "涨跌幅",
            "f4": "涨跌额",
            "f6": "成交额",
            "f8": "换手率",
            "f20": "总市值",
            "f104": "上涨家数",
            "f105": "下跌家数",
            "f128": "领涨股票",
            "f136": "领涨股票-涨跌幅",
        }
    )
    frame.insert(0, "排名", range(1, len(frame) + 1))
    for column in ["涨跌幅", "涨跌额", "成交额", "换手率", "总市值", "上涨家数", "下跌家数", "领涨股票-涨跌幅"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        [
            "排名",
            "板块名称",
            "板块代码",
            "涨跌额",
            "涨跌幅",
            "总市值",
            "换手率",
            "上涨家数",
            "下跌家数",
            "领涨股票",
            "领涨股票-涨跌幅",
        ]
    ]


def _cached_board_code(symbol: str) -> str | None:
    candidates = [
        PATHS.cache / "sector_stats" / "board_names",
        PATHS.cache / "sector_watch" / "board_names",
    ]
    for folder in candidates:
        for path in folder.glob("*.csv"):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            name_col = "board" if "board" in frame.columns else "board_name" if "board_name" in frame.columns else "板块名称"
            if name_col not in frame.columns or "板块代码" not in frame.columns:
                continue
            matched = frame[frame[name_col] == symbol]
            if not matched.empty:
                return str(matched.iloc[0]["板块代码"])
    return None


def stock_board_industry_hist_em(
    symbol: str,
    start_date: str,
    end_date: str,
    period: str = "日k",
    adjust: str = "",
) -> pd.DataFrame:
    board_code = str(symbol)
    if not board_code.startswith("BK"):
        try:
            listing = stock_board_industry_name_em()
            matched = listing[listing["板块名称"] == symbol]
            if not matched.empty:
                board_code = str(matched.iloc[0]["板块代码"])
        except Exception:
            cached_code = _cached_board_code(symbol)
            if cached_code:
                board_code = cached_code
        if not board_code.startswith("BK"):
            cached_code = _cached_board_code(symbol)
            if cached_code:
                board_code = cached_code
            else:
                raise KeyError(f"unknown Eastmoney board: {symbol}")

    period_map = {"日k": "101", "周k": "102", "月k": "103"}
    adjust_map = {"": "0", "qfq": "1", "hfq": "2"}
    payload = _get(
        "https://7.push2his.eastmoney.com/api/qt/stock/kline/get",
        params={
            "secid": f"90.{board_code}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": period_map[period],
            "fqt": adjust_map[adjust],
            "beg": start_date,
            "end": end_date,
            "smplmt": "10000",
            "lmt": "1000000",
        },
    )
    klines = payload.get("data", {}).get("klines") or []
    frame = pd.DataFrame([item.split(",") for item in klines])
    if frame.empty:
        return frame
    frame.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        [
            "日期",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "换手率",
        ]
    ]
