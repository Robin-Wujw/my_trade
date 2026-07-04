"""Shared pytest configuration for stock_research tests."""

from pathlib import Path


def pytest_sessionstart(session):
    del session
    Path("var/tmp").mkdir(parents=True, exist_ok=True)
