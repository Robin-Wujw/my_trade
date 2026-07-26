"""Data-source architecture contracts."""

from .sources import DataCapability, DataSourceRole, SourceRegistry, default_source_registry

__all__ = [
    "DataCapability",
    "DataSourceRole",
    "SourceRegistry",
    "default_source_registry",
]
