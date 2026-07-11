"""Resilience helpers for external API calls.

The retry loop is intentionally provider-agnostic.  SDK adapters are
responsible for turning provider error codes into exceptions before returning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import random
from threading import Lock
import time
from typing import Callable, Optional

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None


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


@contextmanager
def _locked_rate_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b"0\n")
            handle.flush()
        acquired = False
        while not acquired:
            try:
                handle.seek(0)
                if msvcrt is not None:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                elif fcntl is not None:  # pragma: no cover - POSIX path
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                acquired = True
            except OSError:
                time.sleep(0.01)
        yield handle
    finally:
        if msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:  # pragma: no cover - POSIX path
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@dataclass
class FileRateLimiter:
    """Cross-process minimum-interval limiter backed by a small lock file."""

    min_interval: float
    path: Path

    def wait(self) -> None:
        interval = max(0.0, float(self.min_interval))
        if not interval:
            return
        with _locked_rate_file(Path(self.path)) as handle:
            handle.seek(0)
            try:
                last_call = float(handle.read().decode("ascii").strip() or 0)
            except (UnicodeDecodeError, ValueError):
                last_call = 0.0
            now = time.time()
            delay = interval - (now - last_call)
            if delay > 0:
                time.sleep(delay)
            handle.seek(0)
            handle.truncate()
            handle.write(f"{time.time():.6f}\n".encode("ascii"))
            handle.flush()
