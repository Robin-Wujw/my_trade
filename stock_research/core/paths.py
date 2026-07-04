"""Single authority for repository and runtime paths."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())

    @property
    def runtime_root(self) -> Path:
        override = os.environ.get("STOCK_RESEARCH_VAR")
        return Path(override).resolve() if override else self.project_root / "var"

    @property
    def cache(self) -> Path:
        return self.runtime_root / "cache"

    @property
    def database(self) -> Path:
        return self.runtime_root / "data" / "my_trade.duckdb"

    @property
    def exports(self) -> Path:
        return self.runtime_root / "exports"

    @property
    def selection_exports(self) -> Path:
        return self.exports / "selection"

    @property
    def market_exports(self) -> Path:
        return self.exports / "market"

    @property
    def report_exports(self) -> Path:
        return self.exports / "reports"

    @property
    def logs(self) -> Path:
        return self.runtime_root / "logs"

    @property
    def state(self) -> Path:
        return self.runtime_root / "state"

    @property
    def secrets(self) -> Path:
        return self.runtime_root / "secrets"

    @property
    def tmp(self) -> Path:
        return self.runtime_root / "tmp"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths(PROJECT_ROOT)
