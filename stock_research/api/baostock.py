"""Single import boundary for the Baostock SDK."""
from __future__ import annotations

import os

import baostock as _sdk

from .retry import RateLimiter


_RATE_LIMITER = RateLimiter(float(os.environ.get("BAOSTOCK_MIN_INTERVAL", "0.02")))


class BaostockResponseError(RuntimeError):
    """BaoStock returned a non-zero provider error code."""


def ensure_success(result, operation="BaoStock request"):
    """Turn SDK error-code results into exceptions so retry logic can act."""
    code = str(getattr(result, "error_code", "0"))
    if code != "0":
        message = getattr(result, "error_msg", "unknown provider error")
        raise BaostockResponseError(f"{operation} failed [{code}]: {message}")
    return result


def reconnect():
    """Reset the process-local BaoStock session and verify the new login."""
    try:
        _sdk.logout()
    except Exception:
        pass
    return ensure_success(_sdk.login(), "BaoStock login")


def __getattr__(name):
    value = getattr(_sdk, name)
    if callable(value) and name.startswith("query_"):
        def rate_limited(*args, **kwargs):
            _RATE_LIMITER.wait()
            return value(*args, **kwargs)
        rate_limited.__name__ = getattr(value, "__name__", name)
        rate_limited.__doc__ = getattr(value, "__doc__", None)
        return rate_limited
    return value
