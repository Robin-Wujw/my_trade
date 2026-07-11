"""Console encoding helpers for Windows command-line runs."""
from __future__ import annotations

import os
import sys


def configure_utf8_console() -> None:
    """Prefer UTF-8 output so Chinese logs do not become mojibake."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
