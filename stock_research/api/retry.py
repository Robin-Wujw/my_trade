"""Retry helpers for external API calls."""
from __future__ import annotations

import time


def call_with_retry(func, *args, retries=3, delay=2, label=None, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            prefix = f"{label} " if label else ""
            print(f"{prefix}请求失败: {exc} | 第 {attempt}/{retries} 次")
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last_exc
