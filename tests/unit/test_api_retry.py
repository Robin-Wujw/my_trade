import pytest

from stock_research.api.retry import (
    FileRateLimiter,
    RateLimiter,
    call_with_backoff,
    call_with_retry,
)


def test_retry_uses_exact_attempt_count_and_raises_last_error(monkeypatch):
    calls = []
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _: None)

    def fail():
        calls.append(len(calls) + 1)
        raise RuntimeError(f"failure-{len(calls)}")

    with pytest.raises(RuntimeError, match="failure-3"):
        call_with_retry(fail, retries=3, delay=0)

    assert calls == [1, 2, 3]


def test_backoff_stops_immediately_for_permanent_error(monkeypatch):
    calls = []
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _: None)

    def fail():
        calls.append(1)
        raise ValueError("invalid symbol")

    with pytest.raises(ValueError, match="invalid symbol"):
        call_with_backoff(
            fail,
            retries=5,
            retry_delay=0,
            retry_if=lambda exc: not isinstance(exc, ValueError),
        )

    assert calls == [1]


def test_backoff_runs_recovery_hook_before_retry(monkeypatch):
    events = []
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _: None)

    def flaky():
        events.append("call")
        if events.count("call") == 1:
            raise ConnectionError("reset")
        return "ok"

    assert call_with_backoff(
        flaky,
        retries=2,
        retry_delay=0,
        on_retry=lambda _exc, attempt: events.append(f"recover-{attempt}"),
    ) == "ok"
    assert events == ["call", "recover-1", "call"]


def test_rate_limiter_waits_only_for_remaining_interval(monkeypatch):
    clock = iter([10.0, 10.0, 10.2, 10.5])
    sleeps = []
    monkeypatch.setattr("stock_research.api.retry.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("stock_research.api.retry.time.sleep", sleeps.append)
    limiter = RateLimiter(0.5)

    limiter.wait()
    limiter.wait()

    assert sleeps == pytest.approx([0.3])


def test_file_rate_limiter_persists_last_call_for_other_processes(monkeypatch, tmp_path):
    clock = iter([100.0, 100.0, 100.2, 100.5])
    sleeps = []
    monkeypatch.setattr("stock_research.api.retry.time.time", lambda: next(clock))
    monkeypatch.setattr("stock_research.api.retry.time.sleep", sleeps.append)
    path = tmp_path / "provider.lock"

    FileRateLimiter(0.5, path).wait()
    FileRateLimiter(0.5, path).wait()

    assert sleeps == pytest.approx([0.3])
    assert float(path.read_text(encoding="ascii")) == pytest.approx(100.5)
