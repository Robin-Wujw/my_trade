import ssl
from types import SimpleNamespace

import pytest
import requests

from stock_research.api import tushare


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_error_is_not_retried(monkeypatch, status):
    calls = []
    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(
        tushare._HTTP_SESSION, "post",
        lambda *_args, **_kwargs: calls.append(1) or response,
    )
    with pytest.raises(requests.HTTPError):
        tushare.query("daily", retries=4, retry_delay=0)
    assert calls == [1]


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_transient_http_errors_keep_retry_policy(status):
    response = requests.Response()
    response.status_code = status
    assert tushare._retryable(requests.HTTPError(response=response))


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
        tushare._HTTP_SESSION,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or response,
    )

    result = tushare.query("daily", ts_code="000001.SZ", retries=1)

    assert result.to_dict("records") == [{"ts_code": "000001.SZ", "close": 10.5}]
    assert captured["url"] == tushare.API_URL
    assert captured["headers"] == {"Accept-Encoding": "gzip"}
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
        tushare._HTTP_SESSION,
        "post",
        lambda *_args, **_kwargs: calls.append(1) or response,
    )

    with pytest.raises(tushare.TushareAPIError, match="2002"):
        tushare.query("adj_factor", retries=5, retry_delay=0)

    assert calls == [1]


def test_stk_limit_permission_error_is_not_misread_as_rate_limit(monkeypatch):
    calls = []
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "code": 40203,
            "msg": "没有接口(stk_limit)访问权限",
            "data": None,
        },
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(
        tushare._HTTP_SESSION,
        "post",
        lambda *_args, **_kwargs: calls.append(1) or response,
    )

    with pytest.raises(tushare.TushareAPIError, match="40203"):
        tushare.query("stk_limit", retries=5, retry_delay=0)

    assert calls == [1]


def test_empty_success_response_returns_empty_frame(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"code": 0, "data": {"fields": [], "items": []}},
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(tushare._HTTP_SESSION, "post", lambda *_args, **_kwargs: response)

    assert tushare.query("daily", retries=1).empty


def test_query_pads_short_rows_from_proxy(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "code": 0,
            "data": {"fields": ["ts_code", "close", "limit_status"], "items": [["000001.SZ", 10.5]]},
        },
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(tushare._HTTP_SESSION, "post", lambda *_args, **_kwargs: response)

    result = tushare.query("daily_basic", retries=1)

    assert result.to_dict("records") == [
        {"ts_code": "000001.SZ", "close": 10.5, "limit_status": None}
    ]


def test_query_truncates_long_rows_from_proxy(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "code": 0,
            "data": {"fields": ["ts_code", "close"], "items": [["000001.SZ", 10.5, "extra"]]},
        },
    )
    monkeypatch.setattr(tushare, "get_token", lambda: "secret-token")
    monkeypatch.setattr(tushare._RATE_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(tushare._HTTP_SESSION, "post", lambda *_args, **_kwargs: response)

    result = tushare.query("daily_basic", retries=1)

    assert result.to_dict("records") == [{"ts_code": "000001.SZ", "close": 10.5}]


def test_system_trust_context_keeps_verification_enabled(monkeypatch):
    captured = {}

    class FakeContext:
        check_hostname = False
        verify_mode = ssl.CERT_NONE

        def load_default_certs(self, *, purpose):
            captured["purpose"] = purpose

    monkeypatch.setattr(
        tushare.ssl,
        "SSLContext",
        lambda protocol: captured.update(protocol=protocol) or FakeContext(),
    )

    context = tushare._create_system_trust_context()

    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert captured == {
        "protocol": ssl.PROTOCOL_TLS_CLIENT,
        "purpose": ssl.Purpose.SERVER_AUTH,
    }
