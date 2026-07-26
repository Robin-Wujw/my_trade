"""Canonical data-source roles for the local research system."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataSourceRole(str, Enum):
    PRIMARY_MARKET = "primary_market"
    REALTIME_MARKET = "realtime_market"
    LIVE_TRADING = "live_trading"
    FUNDAMENTAL = "fundamental"
    VALUATION = "valuation"
    SPECIAL_DATA = "special_data"
    FALLBACK_PATCH = "fallback_patch"


@dataclass(frozen=True)
class DataCapability:
    name: str
    provider: str
    role: DataSourceRole
    writable: bool
    description: str


class SourceRegistry:
    """Keep provider ownership explicit instead of scattering source strings."""

    def __init__(self, capabilities: tuple[DataCapability, ...]):
        self.capabilities = tuple(capabilities)

    def provider_for(self, role: DataSourceRole | str) -> str:
        normalized = DataSourceRole(role)
        matches = [item for item in self.capabilities if item.role == normalized]
        if not matches:
            raise KeyError(f"no provider registered for role: {role}")
        return matches[0].provider

    def by_provider(self, provider: str) -> tuple[DataCapability, ...]:
        return tuple(item for item in self.capabilities if item.provider == provider)


def default_source_registry() -> SourceRegistry:
    """MiniQMT owns market/execution, Tushare owns fundamentals, AKShare patches gaps."""
    return SourceRegistry((
        DataCapability(
            name="Historical OHLCV",
            provider="miniqmt",
            role=DataSourceRole.PRIMARY_MARKET,
            writable=False,
            description="MiniQMT xtdata daily/minute bars are the primary market-data source.",
        ),
        DataCapability(
            name="Realtime ticks and minute bars",
            provider="miniqmt",
            role=DataSourceRole.REALTIME_MARKET,
            writable=False,
            description="MiniQMT realtime quote APIs feed intraday monitoring.",
        ),
        DataCapability(
            name="XtTrader account and order boundary",
            provider="miniqmt",
            role=DataSourceRole.LIVE_TRADING,
            writable=False,
            description="Only MiniQMT may represent the broker boundary; this repo keeps it read-only.",
        ),
        DataCapability(
            name="Financial statements",
            provider="tushare",
            role=DataSourceRole.FUNDAMENTAL,
            writable=True,
            description="Tushare Pro financial statement tables feed point-in-time fundamentals.",
        ),
        DataCapability(
            name="Daily basic valuation",
            provider="tushare",
            role=DataSourceRole.VALUATION,
            writable=True,
            description="Tushare Pro daily_basic owns PE/PB/turnover/market-cap snapshots.",
        ),
        DataCapability(
            name="Events and feature datasets",
            provider="tushare",
            role=DataSourceRole.SPECIAL_DATA,
            writable=True,
            description="Tushare Pro owns unlocking, dividend, holder and other structured event tables.",
        ),
        DataCapability(
            name="Board, concept, news and Eastmoney specialty gaps",
            provider="akshare",
            role=DataSourceRole.FALLBACK_PATCH,
            writable=True,
            description="AKShare is a non-core fallback for missing board/concept/news/specialty data.",
        ),
    ))
