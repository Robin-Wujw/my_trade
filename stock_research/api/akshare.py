"""Single import boundary for the AkShare SDK."""
from __future__ import annotations

import akshare as _sdk


def __getattr__(name):
    return getattr(_sdk, name)
