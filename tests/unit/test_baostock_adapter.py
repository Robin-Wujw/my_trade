from types import SimpleNamespace

import pytest

from stock_research.api import baostock


def test_ensure_success_raises_for_provider_error_code():
    result = SimpleNamespace(error_code="10002007", error_msg="network error")

    with pytest.raises(baostock.BaostockResponseError, match="10002007"):
        baostock.ensure_success(result, "history")


def test_reconnect_logs_out_and_validates_login(monkeypatch):
    events = []
    sdk = SimpleNamespace(
        logout=lambda: events.append("logout"),
        login=lambda: events.append("login") or SimpleNamespace(
            error_code="0", error_msg="success"
        ),
    )
    monkeypatch.setattr(baostock, "_sdk", sdk)

    result = baostock.reconnect()

    assert result.error_code == "0"
    assert events == ["logout", "login"]


def test_query_methods_are_rate_limited(monkeypatch):
    calls = []
    monkeypatch.setattr(baostock._RATE_LIMITER, "wait", lambda: calls.append("wait"))
    monkeypatch.setattr(
        baostock,
        "_sdk",
        SimpleNamespace(query_stock_basic=lambda: "result"),
    )

    assert baostock.query_stock_basic() == "result"
    assert calls == ["wait"]
