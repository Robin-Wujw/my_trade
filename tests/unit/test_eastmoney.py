import os

from stock_research.api import eastmoney


def test_get_uses_an_isolated_session_without_mutating_environment(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"ok": True}}

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, url, **kwargs):
            calls.append((self.trust_env, url, kwargs))
            return Response()

    monkeypatch.setattr(eastmoney.requests, "Session", Session)
    monkeypatch.setattr(
        eastmoney.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global requests.get used")
        ),
    )
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:7897")
    before = dict(os.environ)

    payload = eastmoney._get("https://example.invalid/data", params={"q": "x"})

    assert payload == {"data": {"ok": True}}
    assert calls[0][0] is False
    assert calls[0][1] == "https://example.invalid/data"
    assert dict(os.environ) == before


def test_board_history_uses_https(monkeypatch):
    urls = []

    def fake_get(url, **kwargs):
        urls.append(url)
        return {"data": {"klines": []}}

    monkeypatch.setattr(eastmoney, "_get", fake_get)

    result = eastmoney.stock_board_industry_hist_em(
        "BK0001",
        start_date="20260701",
        end_date="20260710",
    )

    assert result.empty
    assert urls == ["https://7.push2his.eastmoney.com/api/qt/stock/kline/get"]
