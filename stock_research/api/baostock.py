"""Single import boundary for the Baostock SDK."""
from __future__ import annotations

import baostock as _sdk


def __getattr__(name):
    return getattr(_sdk, name)
