"""Typed daily-pipeline configuration with environment overrides."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class PipelineConfig:
    factor_workers: int
    formula33_workers: int
    formula33_sleep: float
    formula33_retries: int
    sector_sleep: float
    sector_retries: int
    financial_updates: int

    def __post_init__(self) -> None:
        if self.factor_workers < 1 or self.formula33_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.formula33_retries < 1 or self.sector_retries < 1:
            raise ValueError("retry counts must be positive")
        if self.formula33_sleep < 0 or self.sector_sleep < 0:
            raise ValueError("sleep durations cannot be negative")
        if self.financial_updates < 0:
            raise ValueError("financial_updates cannot be negative")


def load_pipeline_config(path: Path) -> PipelineConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return PipelineConfig(
        factor_workers=int(
            os.environ.get("FACTOR_WORKERS", raw["factor"]["workers"])
        ),
        formula33_workers=int(
            os.environ.get("FORMULA33_WORKERS", raw["formula33"]["workers"])
        ),
        formula33_sleep=float(
            os.environ.get("FORMULA33_SLEEP", raw["formula33"]["sleep"])
        ),
        formula33_retries=int(
            os.environ.get("FORMULA33_RETRIES", raw["formula33"]["retries"])
        ),
        sector_sleep=float(
            os.environ.get("SECTOR_SLEEP", raw["sector"]["sleep"])
        ),
        sector_retries=int(
            os.environ.get("SECTOR_RETRIES", raw["sector"]["retries"])
        ),
        financial_updates=int(
            os.environ.get(
                "FINANCIAL_UPDATES", raw["fundamental"]["max_updates"]
            )
        ),
    )
