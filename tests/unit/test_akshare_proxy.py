import threading
from types import SimpleNamespace

from stock_research.api import akshare


def test_akshare_adapter_clears_proxy_environment_during_calls(monkeypatch):
    seen = {}

    def sample_call():
        import os

        seen["HTTP_PROXY"] = os.environ.get("HTTP_PROXY")
        seen["https_proxy"] = os.environ.get("https_proxy")
        return "ok"

    monkeypatch.setattr(akshare, "_sdk", SimpleNamespace(sample_call=sample_call))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7897")

    assert akshare.sample_call() == "ok"

    assert seen == {"HTTP_PROXY": None, "https_proxy": None}
    assert akshare.os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert akshare.os.environ["https_proxy"] == "http://127.0.0.1:7897"


def test_akshare_adapter_does_not_change_requests_defaults():
    import requests

    session = requests.Session()

    assert session.trust_env is True


def test_akshare_adapter_serializes_proxy_environment_changes(monkeypatch):
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()
    seen = []

    def first_call():
        import os

        seen.append(os.environ.get("HTTP_PROXY"))
        first_inside.set()
        assert release_first.wait(timeout=2)

    def second_call():
        import os

        seen.append(os.environ.get("HTTP_PROXY"))
        second_inside.set()

    monkeypatch.setattr(
        akshare,
        "_sdk",
        SimpleNamespace(first_call=first_call, second_call=second_call),
    )
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:7897")

    first = threading.Thread(target=akshare.first_call)
    second = threading.Thread(target=akshare.second_call)
    first.start()
    assert first_inside.wait(timeout=2)
    second.start()

    try:
        assert not second_inside.wait(timeout=0.2)
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert seen == [None, None]
    assert akshare.os.environ["HTTP_PROXY"] == "http://proxy.invalid:7897"
