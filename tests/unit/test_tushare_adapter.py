from types import SimpleNamespace

import pytest

from stock_research.api import tushare


def test_query_builds_standard_frame_without_exposing_token_in_result(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "code": 0,
            "data": {"fields": ["ts_code", "close"], "items": [["000001.SZ", 10.5]]},
        },
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(
        tushare.requests,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or response,
    )

    result = tushare.query("daily", ts_code="000001.SZ", retries=1)

    assert result.to_dict("records") == [{"ts_code": "000001.SZ", "close": 10.5}]
    assert captured["json"]["token"] == "secret-token"
    assert "secret-token" not in repr(result)


def test_permission_error_is_not_retried(monkeypatch):
    calls = []
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"code": 2002, "msg": "没有权限", "data": None},
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(
        tushare.requests,
        "post",
        lambda *_args, **_kwargs: calls.append(1) or response,
    )

    with pytest.raises(tushare.TushareAPIError, match="2002"):
        tushare.query("adj_factor", retries=5, retry_delay=0)

    assert calls == [1]


def test_empty_success_response_returns_empty_frame(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"code": 0, "data": {"fields": [], "items": []}},
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(tushare.requests, "post", lambda *_args, **_kwargs: response)

    assert tushare.query("daily", retries=1).empty
