"""Minimal Tushare Pro HTTP adapter with safe token loading and throttling."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from stock_research.core.paths import PATHS

from .retry import FileRateLimiter, call_with_backoff, is_transient_error


API_URL = "https://api.tushare.pro"
# Keep a conservative cross-process default. Tushare permissions and limits
# are endpoint-specific; for example, daily and adj_factor use different tiers.
_RATE_LIMITER = FileRateLimiter(
    float(os.environ.get("TUSHARE_MIN_INTERVAL", "1.25")),
    PATHS.tmp / "tushare_rate_limit.lock",
)


class TushareAPIError(RuntimeError):
    """Tushare returned a non-zero API response code."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f"Tushare API error [{code}]: {message}")


def get_token() -> str:
    """Read the token from the environment or an ignored local secret file."""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    path = Path(
        os.environ.get("TUSHARE_TOKEN_FILE", str(PATHS.secrets / "tushare_token"))
    )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, TushareAPIError):
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("频繁", "每分钟", "limit", "timeout", "内部错误")
        )
    return is_transient_error(exc)


def query(
    api_name: str,
    *,
    fields: str = "",
    retries: int = 4,
    retry_delay: float = 2.0,
    timeout: float = 20.0,
    **params,
) -> pd.DataFrame:
    """Call a Tushare Pro endpoint and return its standard tabular response."""
    token = get_token()
    if not token:
        raise RuntimeError(
            "未配置 TUSHARE_TOKEN 或 TUSHARE_TOKEN_FILE/var/secrets/tushare_token"
        )

    def request_once():
        _RATE_LIMITER.wait()
        response = requests.post(
            API_URL,
            json={
                "api_name": str(api_name),
                "token": token,
                "params": params,
                "fields": fields,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        code = payload.get("code", -1)
        if code != 0:
            raise TushareAPIError(code, payload.get("msg") or "unknown error")
        data = payload.get("data") or {}
        columns = data.get("fields") or []
        items = data.get("items") or []
        return pd.DataFrame(items, columns=columns)

    return call_with_backoff(
        request_once,
        f"Tushare {api_name}",
        retries=retries,
        retry_delay=retry_delay,
        retry_if=_retryable,
    )
