import pytest

from stock_research.api.retry import call_with_retry


def test_retry_uses_exact_attempt_count_and_raises_last_error(monkeypatch):
    calls = []
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _: None)

    def fail():
        calls.append(len(calls) + 1)
        raise RuntimeError(f"failure-{len(calls)}")

    with pytest.raises(RuntimeError, match="failure-3"):
        call_with_retry(fail, retries=3, delay=0)

    assert calls == [1, 2, 3]
