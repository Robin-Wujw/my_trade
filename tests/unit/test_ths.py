from __future__ import annotations

import re

import pandas as pd
import pytest
import requests

from stock_research.api import ths


class FakeResponse:
    def __init__(self, text: str, *, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


class FakeSession:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responder(url)


@pytest.fixture(autouse=True)
def fixed_cookie(monkeypatch):
    monkeypatch.setattr(ths, "_cookie_value", lambda: "unit-test-cookie")


def _table(headers, rows, *, page=None, pages=None):
    head = "".join(f"<th>{value}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    pagination = (
        f'<span class="page_info">{page}/{pages}</span>'
        if page is not None and pages is not None
        else ""
    )
    return (
        f"<html><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>{pagination}</html>"
    )


def test_board_list_returns_all_industries_and_disables_ambient_proxy():
    anchors = "".join(
        f'<a href="/thshy/detail/code/{881100 + index}/">行业{index:02d}</a>'
        for index in range(90)
    )
    session = FakeSession(
        lambda _url: FakeResponse(f'<div class="cate_inner">{anchors}</div>')
    )

    frame = ths.load_board_list(session=session, timeout=(1, 2))

    assert frame.columns.tolist() == ["code", "name"]
    assert len(frame) == 90
    assert frame["code"].is_unique
    assert frame.iloc[0].to_dict() == {"code": "881100", "name": "行业00"}
    assert session.trust_env is False
    assert len(session.calls) == 1
    assert session.calls[0][1]["timeout"] == (1, 2)
    assert session.calls[0][1]["headers"]["Cookie"] == "v=unit-test-cookie"


def test_board_history_returns_canonical_daily_frame_and_filters_dates():
    payload = (
        'quotebridge_callback({"data":"'
        "20260701,10,11,9,10,100,1000,,,,0;"
        "20260702,10,12,10,11,110,1210,,,,0;"
        "20260710,11,13,10,12,120,1440,,,,0"
        '"})'
    )
    session = FakeSession(lambda _url: FakeResponse(payload))

    frame = ths.load_board_history(
        "881121",
        start_date="20260702",
        end_date="20260710",
        session=session,
    )

    assert frame.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
    ]
    assert frame["date"].tolist() == [
        pd.Timestamp("2026-07-02"),
        pd.Timestamp("2026-07-10"),
    ]
    assert frame.iloc[0]["pct_chg"] == pytest.approx(0.10)
    assert frame.iloc[1]["pct_chg"] == pytest.approx(1 / 11)
    assert session.calls[0][0].endswith("/bk_881121/01/2026.js")


def test_board_history_fetches_only_years_in_requested_range():
    session = FakeSession(
        lambda _url: FakeResponse('callback({"data":"20250102,1,1,1,1,1,1,,,,0"})')
    )

    ths.load_board_history(
        "881121",
        start_date="20250101",
        end_date="20261231",
        session=session,
    )

    assert [re.search(r"/(\d{4})\.js$", url).group(1) for url, _ in session.calls] == [
        "2025",
        "2026",
    ]


def test_board_summary_reads_every_page_into_canonical_columns():
    headers = [
        "序号",
        "板块",
        "涨跌幅(%)",
        "总成交量（万手）",
        "总成交额（亿元）",
        "净流入（亿元）",
        "上涨家数",
        "下跌家数",
        "均价",
        "领涨股",
        "最新价",
        "涨跌幅(%).1",
    ]
    first_rows = [
        [1, "医疗服务", 5.59, 1635.08, 373.43, 28.33, 53, 3, 22.84, "益诺思", 77.87, 20.0]
    ] + [
        [rank, f"行业{rank:02d}", 1, 10, 20, 1, 10, 2, 8, "领涨股", 9, 2]
        for rank in range(2, 51)
    ]
    second_rows = [
        [51, "影视院线", 3.94, 1195.59, 92.17, 8.89, 20, 0, 7.71, "幸福蓝海", 15.9, 10.8]
    ] + [
        [rank, f"行业{rank:02d}", 1, 10, 20, 1, 10, 2, 8, "领涨股", 9, 2]
        for rank in range(52, 91)
    ]
    pages = {
        1: _table(
            headers,
            first_rows,
            page=1,
            pages=2,
        ),
        2: _table(
            headers,
            second_rows,
            page=2,
            pages=2,
        ),
    }

    def respond(url):
        page = int(re.search(r"/page/(\d+)/", url).group(1))
        return FakeResponse(pages[page])

    session = FakeSession(respond)
    frame = ths.load_board_summary(session=session)

    assert frame.columns.tolist() == [
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
    assert len(frame) == 90
    assert frame.loc[frame["name"].eq("医疗服务"), "pct_chg"].item() == pytest.approx(0.0559)
    assert frame.loc[frame["name"].eq("影视院线"), "pct_chg"].item() == pytest.approx(0.0394)
    assert frame.loc[0, "volume"] == pytest.approx(16_350_800)
    assert frame.loc[0, "amount"] == pytest.approx(37_343_000_000)
    assert [int(re.search(r"/page/(\d+)/", url).group(1)) for url, _ in session.calls] == [1, 2]


def test_board_summary_rejects_an_incomplete_snapshot():
    headers = [
        "序号", "板块", "涨跌幅(%)", "总成交量（万手）", "总成交额（亿元）",
        "净流入（亿元）", "上涨家数", "下跌家数", "均价", "领涨股", "最新价", "涨跌幅(%).1",
    ]
    text = _table(
        headers,
        [[1, "仅一个板块", 1, 1, 1, 1, 1, 1, 1, "股票", 1, 1]],
        page=1,
        pages=1,
    )

    with pytest.raises(ths.THSResponseError, match="incomplete"):
        ths.load_board_summary(session=FakeSession(lambda _url: FakeResponse(text)))


def test_board_constituents_reads_every_page_and_normalizes_codes():
    headers = [
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
    pages = {
        1: _table(
            headers,
            [[1, "000001", "平安银行", 10, 1, 0.1, 0, 1, 1, 2, "1.2亿", "10亿", "100亿", 8]],
            page=1,
            pages=3,
        ),
        2: _table(
            headers,
            [[2, "600000", "浦发银行", 11, 2, 0.2, 0, 1, 1, 2, "2亿", "20亿", "200亿", 9]],
            page=2,
            pages=3,
        ),
        3: _table(
            headers,
            [
                [3, "000001", "平安银行", 10, 1, 0.1, 0, 1, 1, 2, "1.2亿", "10亿", "100亿", 8],
                [4, "300750", "宁德时代", 12, 3, 0.3, 0, 1, 1, 2, "3亿", "30亿", "300亿", 10],
            ],
            page=3,
            pages=3,
        ),
    }

    def respond(url):
        page = int(re.search(r"/page/(\d+)/", url).group(1))
        return FakeResponse(pages[page])

    session = FakeSession(respond)
    frame = ths.load_board_constituents("881175", session=session)

    assert frame.columns.tolist() == [
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
    assert frame["code"].tolist() == ["000001", "600000", "300750"]
    assert frame["code"].str.len().eq(6).all()
    assert frame["code"].is_unique
    assert frame.loc[0, "amount"] == pytest.approx(120_000_000)
    assert frame.loc[0, "pct_chg"] == pytest.approx(0.01)
    assert [int(re.search(r"/page/(\d+)/", url).group(1)) for url, _ in session.calls] == [1, 2, 3]


def test_board_constituents_uses_sort_union_when_pages_above_login_limit():
    headers = [
        "序号", "代码", "名称", "现价", "涨跌幅(%)", "涨跌", "涨速(%)",
        "换手(%)", "量比", "振幅(%)", "成交额", "流通股", "流通市值", "市盈率",
    ]

    def rows(codes):
        return [
            [index + 1, code, f"股票{code}", 10, 1, 0.1, 0, 1, 1, 2, "1亿", "10亿", "100亿", 8]
            for index, code in enumerate(codes)
        ]

    ascending = [f"{index:06d}" for index in range(101)]
    descending = list(reversed(ascending))

    def respond(url):
        page = int(re.search(r"/page/(\d+)/", url).group(1))
        ordered = descending if "/order/desc/" in url else ascending
        start = (page - 1) * 20
        page_codes = ordered[start : start + 20]
        return FakeResponse(_table(headers, rows(page_codes), page=page, pages=6))

    session = FakeSession(respond)
    frame = ths.load_board_constituents("881117", session=session)

    assert len(frame) == 101
    assert frame["code"].is_unique
    assert all("/page/6/" not in url for url, _ in session.calls)


def test_board_constituents_rejects_a_repeated_first_page():
    headers = [
        "序号", "代码", "名称", "现价", "涨跌幅(%)", "涨跌", "涨速(%)",
        "换手(%)", "量比", "振幅(%)", "成交额", "流通股", "流通市值", "市盈率",
    ]
    first_page = _table(
        headers,
        [[1, "000001", "平安银行", 10, 1, 0.1, 0, 1, 1, 2, "1亿", "10亿", "100亿", 8]],
        page=1,
        pages=2,
    )
    session = FakeSession(lambda _url: FakeResponse(first_page))

    with pytest.raises(ths.THSResponseError, match="pagination"):
        ths.load_board_constituents("881175", session=session)


def test_default_session_is_closed_and_does_not_use_environment_proxy(monkeypatch):
    anchors = "".join(
        f'<a href="/thshy/detail/code/{881100 + index}/">行业{index:02d}</a>'
        for index in range(90)
    )

    class OwnedSession(FakeSession):
        def __init__(self):
            super().__init__(
                lambda _url: FakeResponse(f'<div class="cate_inner">{anchors}</div>')
            )
            self.closed = False

        def close(self):
            self.closed = True

    owned = OwnedSession()
    monkeypatch.setattr(ths.requests, "Session", lambda: owned)

    ths.load_board_list()

    assert owned.trust_env is False
    assert owned.closed is True


def test_request_retries_are_bounded(monkeypatch):
    class FailingSession(FakeSession):
        def __init__(self):
            super().__init__(lambda _url: None)

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            raise requests.ConnectionError("offline")

    session = FailingSession()
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _seconds: None)

    with pytest.raises(requests.ConnectionError, match="offline"):
        ths.load_board_list(session=session)

    assert len(session.calls) == 2


@pytest.mark.parametrize(
    ("loader", "response_text"),
    [
        (lambda session: ths.load_board_list(session=session), "<html>blocked</html>"),
        (
            lambda session: ths.load_board_history(
                "881121",
                start_date="20260701",
                end_date="20260710",
                session=session,
            ),
            "not a javascript payload",
        ),
        (lambda session: ths.load_board_summary(session=session), "<html>blocked</html>"),
        (
            lambda session: ths.load_board_constituents("881175", session=session),
            "<html>blocked</html>",
        ),
    ],
)
def test_malformed_responses_raise_clear_errors(loader, response_text):
    session = FakeSession(lambda _url: FakeResponse(response_text))

    with pytest.raises(ths.THSResponseError, match="Tonghuashun"):
        loader(session)


@pytest.mark.parametrize("code", ["", "88112", "881121/../../x"])
def test_board_code_is_validated_before_request(code):
    session = FakeSession(lambda _url: FakeResponse("unused"))

    with pytest.raises(ValueError, match="six digits"):
        ths.load_board_constituents(code, session=session)

    assert session.calls == []
