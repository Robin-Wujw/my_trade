"""Tonghuashun industry-board HTTP adapter.

All public frames use English canonical columns. Percentage fields are decimal
ratios (for example, 5% is ``0.05``), matching the storage-layer contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from io import StringIO
import json
import re
from typing import Optional

from bs4 import BeautifulSoup
import pandas as pd
import requests

from stock_research.api.retry import call_with_retry


DEFAULT_TIMEOUT = (5, 10)
REQUEST_ATTEMPTS = 2
REQUEST_RETRY_DELAY = 0.25
MIN_BOARD_COUNT = 90
MAX_PAGE_COUNT = 100

_BOARD_LIST_URL = "https://q.10jqka.com.cn/thshy/detail/code/881272/"
_SUMMARY_URL = (
    "http://q.10jqka.com.cn/thshy/index/field/199112/order/desc/page/{page}/ajax/1/"
)
_HISTORY_URL = "https://d.10jqka.com.cn/v4/line/bk_{code}/01/{year}.js"
_CONSTITUENTS_URL = "http://q.10jqka.com.cn/thshy/detail/code/{code}/page/{page}/"
_CONSTITUENTS_SORT_URL = (
    "http://q.10jqka.com.cn/thshy/detail/code/{code}/field/{field}/"
    "order/{order}/page/{page}/"
)
_ANONYMOUS_PAGE_LIMIT = 5
_CONSTITUENT_SORT_FIELDS = (
    "10",
    "199112",
    "19",
    "3475914",
    "2034120",
    "407",
    "1968584",
    "1771976",
    "526792",
    "48",
    "264648",
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HISTORY_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
]
SUMMARY_COLUMNS = [
    "rank",
    "name",
    "pct_chg",
    "volume",
    "amount",
    "net_inflow",
    "advancers",
    "decliners",
    "avg_price",
    "leader_name",
    "leader_price",
    "leader_pct_chg",
]
CONSTITUENT_COLUMNS = [
    "rank",
    "code",
    "name",
    "price",
    "pct_chg",
    "price_chg",
    "speed",
    "turnover",
    "volume_ratio",
    "amplitude",
    "amount",
    "float_shares",
    "float_market_cap",
    "pe",
]


class THSResponseError(ValueError):
    """Raised when Tonghuashun returns a successful but unusable response."""


@lru_cache(maxsize=1)
def _cookie_value() -> str:
    """Generate Tonghuashun's ``v`` cookie from AkShare's bundled JS."""
    from akshare.datasets import get_ths_js
    import py_mini_racer

    runtime = py_mini_racer.MiniRacer()
    with open(get_ths_js("ths.js"), encoding="utf-8") as source:
        runtime.eval(source.read())
    value = runtime.call("v")
    if not value:
        raise RuntimeError("Tonghuashun cookie generator returned an empty value")
    return str(value)


def _headers(*, history: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": _USER_AGENT,
        "Referer": "http://q.10jqka.com.cn/",
        "Cookie": f"v={_cookie_value()}",
    }
    if history:
        headers["Host"] = "d.10jqka.com.cn"
    return headers


@contextmanager
def _session_scope(session=None):
    if session is not None:
        session.trust_env = False
        yield session
        return

    owned = requests.Session()
    owned.trust_env = False
    try:
        yield owned
    finally:
        owned.close()


def _request_text(session, url: str, *, headers, timeout) -> str:
    def request_once():
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        if not isinstance(response.text, str):
            raise THSResponseError(f"Tonghuashun returned non-text content: {url}")
        return response.text

    return call_with_retry(
        request_once,
        retries=REQUEST_ATTEMPTS,
        delay=REQUEST_RETRY_DELAY,
        label="Tonghuashun",
    )


def _board_code(value) -> str:
    code = str(value).strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("Tonghuashun board code must be exactly six digits")
    return code


def _date_value(value, *, argument: str) -> pd.Timestamp:
    text = str(value).strip().replace("-", "")
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"{argument} must use YYYYMMDD or YYYY-MM-DD")
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{argument} is not a valid calendar date")
    return pd.Timestamp(parsed).normalize()


def _pagination(text: str, *, context: str) -> tuple[int, int]:
    soup = BeautifulSoup(text, "lxml")
    marker = soup.find("span", class_="page_info")
    if marker is None:
        return 1, 1
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", marker.get_text())
    if not match:
        raise THSResponseError(
            f"Tonghuashun {context} response has invalid pagination metadata"
        )
    current, count = (int(match.group(1)), int(match.group(2)))
    if count < 1 or count > MAX_PAGE_COUNT or current < 1 or current > count:
        raise THSResponseError(
            f"Tonghuashun {context} pagination is outside 1..{MAX_PAGE_COUNT}: "
            f"{current}/{count}"
        )
    return current, count


def _column_name(value) -> str:
    if isinstance(value, tuple):
        parts = [
            str(part).strip()
            for part in value
            if str(part).strip() and not str(part).startswith("Unnamed:")
        ]
        return parts[-1] if parts else ""
    return str(value).strip()


def _html_table(text: str, *, required, context: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(text))
    except Exception as exc:
        raise THSResponseError(
            f"Tonghuashun {context} response contains no readable table"
        ) from exc

    required_set = set(required)
    for table in tables:
        candidate = table.copy()
        candidate.columns = [_column_name(column) for column in candidate.columns]
        if required_set.issubset(candidate.columns):
            return candidate
    raise THSResponseError(
        f"Tonghuashun {context} response is missing columns: {sorted(required_set)}"
    )


def _plain_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": pd.NA, "-": pd.NA, "--": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _cn_number(value):
    if value is None or pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "None", "nan"}:
        return float("nan")
    multipliers = (("万亿", 1e12), ("亿", 1e8), ("万", 1e4))
    multiplier = 1.0
    for suffix, factor in multipliers:
        if suffix in text:
            multiplier = factor
            text = text.replace(suffix, "")
            break
    text = text.replace("元", "").replace("股", "").replace("手", "").strip()
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(number) else float(number) * multiplier


def _stock_code(value) -> Optional[str]:
    match = re.search(r"(?<!\d)(\d{1,6})(?:\.0)?(?!\d)", str(value).strip())
    return match.group(1).zfill(6) if match else None


def load_board_list(*, session=None, timeout=DEFAULT_TIMEOUT) -> pd.DataFrame:
    """Return the complete Tonghuashun industry taxonomy as ``code/name``."""
    with _session_scope(session) as active:
        text = _request_text(
            active,
            _BOARD_LIST_URL,
            headers=_headers(),
            timeout=timeout,
        )

    soup = BeautifulSoup(text, "lxml")
    container = soup.find("div", class_="cate_inner")
    if container is None:
        raise THSResponseError(
            "Tonghuashun board-list response is missing the industry container"
        )

    rows = []
    for anchor in container.find_all("a", href=True):
        match = re.search(r"/code/(\d{6})(?:/|$)", anchor["href"])
        name = anchor.get_text(" ", strip=True)
        if match and name:
            rows.append({"code": match.group(1), "name": name})
    frame = pd.DataFrame(rows, columns=["code", "name"])
    frame = frame.drop_duplicates("code", keep="first").reset_index(drop=True)
    if len(frame) < MIN_BOARD_COUNT:
        raise THSResponseError(
            "Tonghuashun board-list response is incomplete: "
            f"expected at least {MIN_BOARD_COUNT}, got {len(frame)}"
        )
    if frame["name"].duplicated().any():
        raise THSResponseError("Tonghuashun board-list response contains duplicate names")
    return frame


def _history_payload(text: str, *, code: str, year: int) -> list[list[str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise THSResponseError(
            f"Tonghuashun history response is not valid JS data: {code}/{year}"
        )
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise THSResponseError(
            f"Tonghuashun history response has invalid JSON: {code}/{year}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), str):
        raise THSResponseError(
            f"Tonghuashun history response is missing data: {code}/{year}"
        )
    if not payload["data"].strip():
        return []
    records = [record.split(",") for record in payload["data"].split(";") if record]
    if any(len(record) < 7 for record in records):
        raise THSResponseError(
            f"Tonghuashun history response has short records: {code}/{year}"
        )
    return records


def load_board_history(
    code,
    *,
    start_date,
    end_date,
    session=None,
    timeout=DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Load daily OHLCV history; ``pct_chg`` is a decimal close return."""
    board_code = _board_code(code)
    start = _date_value(start_date, argument="start_date")
    end = _date_value(end_date, argument="end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    records = []
    headers = _headers(history=True)
    with _session_scope(session) as active:
        for year in range(start.year, end.year + 1):
            text = _request_text(
                active,
                _HISTORY_URL.format(code=board_code, year=year),
                headers=headers,
                timeout=timeout,
            )
            records.extend(_history_payload(text, code=board_code, year=year))

    if not records:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.DataFrame(
        [record[:7] for record in records],
        columns=["date", "open", "high", "low", "close", "volume", "amount"],
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[["date", *numeric_columns]].isna().any(axis=1)
    if invalid.any():
        raise THSResponseError(
            f"Tonghuashun history response contains {int(invalid.sum())} malformed rows"
        )

    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["pct_chg"] = frame["close"].pct_change(fill_method=None)
    frame = frame.loc[frame["date"].between(start, end), HISTORY_COLUMNS]
    return frame.reset_index(drop=True)


def _parse_summary_page(text: str) -> pd.DataFrame:
    aliases = {
        "rank": ("序号",),
        "name": ("板块",),
        "pct_chg": ("涨跌幅(%)", "涨跌幅"),
        "volume": ("总成交量（万手）", "总成交量(万手)", "总成交量"),
        "amount": ("总成交额（亿元）", "总成交额(亿元)", "总成交额"),
        "net_inflow": ("净流入（亿元）", "净流入(亿元)", "净流入"),
        "advancers": ("上涨家数",),
        "decliners": ("下跌家数",),
        "avg_price": ("均价",),
        "leader_name": ("领涨股",),
        "leader_price": ("领涨股-最新价", "最新价"),
        "leader_pct_chg": (
            "领涨股-涨跌幅",
            "涨跌幅(%).1",
            "涨跌幅.1",
        ),
    }
    source = _html_table(
        text,
        required=("序号", "板块", "领涨股"),
        context="board-summary",
    )
    selected = {}
    selected_names = {}
    for canonical, candidates in aliases.items():
        source_name = next(
            (candidate for candidate in candidates if candidate in source.columns),
            None,
        )
        if source_name is None:
            raise THSResponseError(
                "Tonghuashun board-summary response is missing field "
                f"{canonical}: expected one of {candidates}"
            )
        selected[canonical] = source[source_name]
        selected_names[canonical] = source_name

    frame = pd.DataFrame(selected, columns=SUMMARY_COLUMNS)
    frame["name"] = frame["name"].astype("string").str.strip()
    frame["leader_name"] = frame["leader_name"].astype("string").str.strip()
    numeric_columns = [
        column
        for column in SUMMARY_COLUMNS
        if column not in {"name", "leader_name"}
    ]
    for column in numeric_columns:
        frame[column] = _plain_numeric(frame[column])
    frame["pct_chg"] /= 100.0
    frame["leader_pct_chg"] /= 100.0
    if "万" in selected_names["volume"]:
        frame["volume"] *= 1e4
    for column in ("amount", "net_inflow"):
        if "亿" in selected_names[column]:
            frame[column] *= 1e8
    return frame


def load_board_summary(*, session=None, timeout=DEFAULT_TIMEOUT) -> pd.DataFrame:
    """Return all pages of the current Tonghuashun industry overview."""
    headers = _headers()
    with _session_scope(session) as active:
        first_text = _request_text(
            active,
            _SUMMARY_URL.format(page=1),
            headers=headers,
            timeout=timeout,
        )
        frames = [_parse_summary_page(first_text)]
        first_page, page_count = _pagination(first_text, context="board-summary")
        if first_page != 1:
            raise THSResponseError(
                f"Tonghuashun board-summary pagination started at page {first_page}"
            )
        for page in range(2, page_count + 1):
            text = _request_text(
                active,
                _SUMMARY_URL.format(page=page),
                headers=headers,
                timeout=timeout,
            )
            current_page, reported_count = _pagination(
                text,
                context="board-summary",
            )
            if current_page != page or reported_count != page_count:
                raise THSResponseError(
                    "Tonghuashun board-summary pagination returned "
                    f"{current_page}/{reported_count} while requesting {page}/{page_count}"
                )
            frames.append(_parse_summary_page(text))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["name"])
    if frame.empty:
        raise THSResponseError("Tonghuashun board-summary response has no data rows")
    if frame["name"].duplicated().any():
        raise THSResponseError(
            "Tonghuashun board-summary pagination returned duplicate boards"
        )
    if len(frame) < MIN_BOARD_COUNT:
        raise THSResponseError(
            "Tonghuashun board-summary response is incomplete: "
            f"expected at least {MIN_BOARD_COUNT}, got {len(frame)}"
        )
    return frame.reset_index(drop=True)[SUMMARY_COLUMNS]


def _parse_constituent_page(text: str) -> pd.DataFrame:
    source_columns = [
        "序号",
        "代码",
        "名称",
        "现价",
        "涨跌幅(%)",
        "涨跌",
        "涨速(%)",
        "换手(%)",
        "量比",
        "振幅(%)",
        "成交额",
        "流通股",
        "流通市值",
        "市盈率",
    ]
    frame = _html_table(
        text,
        required=source_columns,
        context="board-constituents",
    ).rename(columns=dict(zip(source_columns, CONSTITUENT_COLUMNS)))
    return frame[CONSTITUENT_COLUMNS]


def load_board_constituents(
    code,
    *,
    session=None,
    timeout=DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Return every constituent page for one Tonghuashun industry board."""
    board_code = _board_code(code)
    headers = _headers()
    with _session_scope(session) as active:
        first_text = _request_text(
            active,
            _CONSTITUENTS_URL.format(code=board_code, page=1),
            headers=headers,
            timeout=timeout,
        )
        frames = [_parse_constituent_page(first_text)]
        first_page, page_count = _pagination(
            first_text,
            context="board-constituents",
        )
        if first_page != 1:
            raise THSResponseError(
                f"Tonghuashun board-constituents pagination started at page {first_page}"
            )
        if page_count <= _ANONYMOUS_PAGE_LIMIT:
            for page in range(2, page_count + 1):
                text = _request_text(
                    active,
                    _CONSTITUENTS_URL.format(code=board_code, page=page),
                    headers=headers,
                    timeout=timeout,
                )
                current_page, reported_count = _pagination(
                    text,
                    context="board-constituents",
                )
                if current_page != page or reported_count != page_count:
                    raise THSResponseError(
                        "Tonghuashun board-constituents pagination returned "
                        f"{current_page}/{reported_count} while requesting {page}/{page_count}"
                    )
                frames.append(_parse_constituent_page(text))
        else:
            for page in range(2, _ANONYMOUS_PAGE_LIMIT + 1):
                text = _request_text(
                    active,
                    _CONSTITUENTS_URL.format(code=board_code, page=page),
                    headers=headers,
                    timeout=timeout,
                )
                frames.append(_parse_constituent_page(text))

            minimum_expected = (page_count - 1) * 20 + 1
            maximum_expected = page_count * 20
            previous_count = len(
                {
                    identity
                    for frame in frames
                    for identity in frame["code"].map(_stock_code)
                    if identity is not None
                }
            )
            converged = False
            for field in _CONSTITUENT_SORT_FIELDS:
                for order in ("asc", "desc"):
                    for page in range(1, _ANONYMOUS_PAGE_LIMIT + 1):
                        text = _request_text(
                            active,
                            _CONSTITUENTS_SORT_URL.format(
                                code=board_code,
                                field=field,
                                order=order,
                                page=page,
                            ),
                            headers=headers,
                            timeout=timeout,
                        )
                        frames.append(_parse_constituent_page(text))
                current_count = len(
                    {
                        identity
                        for frame in frames
                        for identity in frame["code"].map(_stock_code)
                        if identity is not None
                    }
                )
                if current_count > maximum_expected:
                    raise THSResponseError(
                        "Tonghuashun board-constituents sort union exceeds "
                        f"pagination bound: {current_count}>{maximum_expected}"
                    )
                if current_count >= minimum_expected and current_count == previous_count:
                    converged = True
                    break
                previous_count = current_count
            if not converged:
                raise THSResponseError(
                    "Tonghuashun board-constituents sort union did not converge "
                    f"within pagination bounds: rows={previous_count}, pages={page_count}"
                )

    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].map(_stock_code)
    frame["name"] = frame["name"].astype("string").str.strip()
    if frame["code"].isna().any() or frame["name"].isna().any():
        raise THSResponseError(
            "Tonghuashun board-constituents response contains malformed identities"
        )

    plain_columns = ["rank", "price", "price_chg", "volume_ratio", "pe"]
    percentage_columns = ["pct_chg", "speed", "turnover", "amplitude"]
    scaled_columns = ["amount", "float_shares", "float_market_cap"]
    for column in plain_columns:
        frame[column] = _plain_numeric(frame[column])
    for column in percentage_columns:
        frame[column] = _plain_numeric(frame[column]) / 100.0
    for column in scaled_columns:
        frame[column] = frame[column].map(_cn_number)

    frame = frame.drop_duplicates("code", keep="first")
    if frame.empty:
        raise THSResponseError("Tonghuashun board-constituents response has no rows")
    return frame.reset_index(drop=True)[CONSTITUENT_COLUMNS]


__all__ = [
    "CONSTITUENT_COLUMNS",
    "DEFAULT_TIMEOUT",
    "HISTORY_COLUMNS",
    "SUMMARY_COLUMNS",
    "THSResponseError",
    "load_board_constituents",
    "load_board_history",
    "load_board_list",
    "load_board_summary",
]
