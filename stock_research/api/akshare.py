"""Single import boundary for the AkShare SDK."""
from __future__ import annotations

from functools import wraps
import os
from threading import RLock

import akshare as _sdk

from .retry import RateLimiter


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_PROXY_ENV_LOCK = RLock()
_RATE_LIMITER = RateLimiter(float(os.environ.get("AKSHARE_MIN_INTERVAL", "0.05")))


def _without_proxy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _PROXY_ENV_LOCK:
            _RATE_LIMITER.wait()
            keys = (*PROXY_ENV_KEYS, "NO_PROXY", "no_proxy")
            saved = {key: os.environ.get(key) for key in keys}
            try:
                for key in PROXY_ENV_KEYS:
                    os.environ.pop(key, None)
                os.environ["NO_PROXY"] = "*"
                os.environ["no_proxy"] = "*"
                return func(*args, **kwargs)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    return wrapper


def __getattr__(name):
    value = getattr(_sdk, name)
    if callable(value):
        return _without_proxy(value)
    return value
