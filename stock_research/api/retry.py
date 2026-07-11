"""Resilience helpers for external API calls.

The retry loop is intentionally provider-agnostic.  SDK adapters are
responsible for turning provider error codes into exceptions before returning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from threading import Lock
import time
from typing import Callable, Optional


TRANSIENT_ERROR_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "频繁",
    "限流",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


def is_transient_error(exc: Exception) -> bool:
    """Return whether an exception is safe and useful to retry."""
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError, OSError)):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_with_backoff(
    func: Callable,
    label: Optional[str] = None,
    *,
    retries: int = 3,
    retry_delay: float = 2.0,
    max_delay: float = 60.0,
    retry_if: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Call ``func`` with capped exponential full-jitter backoff.

    Unknown exceptions remain retryable by default for backward compatibility.
    Callers that can distinguish permanent failures should pass ``retry_if``.
    ``Retry-After`` is honoured when an HTTP exception exposes that header.
    """
    attempts = max(1, int(retries))
    base = max(0.0, float(retry_delay))
    cap = max(base, float(max_delay))
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            retryable = True if retry_if is None else bool(retry_if(exc))
            if attempt >= attempts or not retryable:
                raise
            if on_retry is not None:
                on_retry(exc, attempt)
            retry_after = _retry_after_seconds(exc)
            ceiling = min(cap, base * (2 ** (attempt - 1)))
            wait = retry_after if retry_after is not None else random.uniform(0, ceiling)
            prefix = f"{label} " if label else ""
            print(
                f"{prefix}请求失败: {exc} | 第 {attempt}/{attempts} 次，"
                f"{wait:.1f}s 后重试"
            )
            if wait:
                time.sleep(wait)
    raise last_exc  # pragma: no cover


def call_with_retry(func, *args, retries=3, delay=2, label=None, **kwargs):
    """Backward-compatible wrapper around :func:`call_with_backoff`."""
    return call_with_backoff(
        lambda: func(*args, **kwargs),
        label,
        retries=retries,
        retry_delay=delay,
    )


@dataclass
class RateLimiter:
    """Thread-safe, per-process minimum-interval limiter."""

    min_interval: float
    _last_call: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def wait(self) -> None:
        interval = max(0.0, float(self.min_interval))
        if not interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = interval - (now - self._last_call)
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()
