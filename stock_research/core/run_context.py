"""Immutable identity and date boundaries for a pipeline run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class RunMode(str, Enum):
    """Supported execution modes persisted in ``ops.runs``."""

    PRODUCTION = "production"
    BACKTEST = "backtest"
    OFFLINE = "offline"


MARKET_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class RunContext:
    """All immutable cutoffs needed to reproduce one pipeline run."""

    run_id: str
    observation_date: date
    market_cutoff: date
    financial_cutoff: datetime
    report_period: date
    code_version: str
    created_at: datetime
    mode: RunMode

    @classmethod
    def create(
        cls,
        *,
        observation_date: date,
        market_cutoff: date,
        financial_cutoff: datetime,
        report_period: date,
        code_version: str,
        mode: RunMode,
        now: Optional[datetime] = None,
    ) -> "RunContext":
        """Validate time boundaries and create a traceable unique identity."""
        _require_aware(financial_cutoff, "financial_cutoff")
        created_at = now or datetime.now(timezone.utc)
        _require_aware(created_at, "now")

        if market_cutoff > observation_date:
            raise ValueError("market_cutoff cannot be later than observation_date")
        if financial_cutoff.astimezone(MARKET_TIMEZONE).date() > observation_date:
            raise ValueError("financial_cutoff cannot be later than observation_date")
        if report_period > observation_date:
            raise ValueError("report_period cannot be later than observation_date")
        if not str(code_version).strip():
            raise ValueError("code_version cannot be empty")

        created_at_utc = created_at.astimezone(timezone.utc)
        run_id = (
            f"{observation_date:%Y%m%d}-"
            f"{created_at_utc:%Y%m%dT%H%M%SZ}-"
            f"{uuid4().hex[:8]}"
        )
        return cls(
            run_id=run_id,
            observation_date=observation_date,
            market_cutoff=market_cutoff,
            financial_cutoff=financial_cutoff,
            report_period=report_period,
            code_version=str(code_version).strip(),
            created_at=created_at_utc,
            mode=RunMode(mode),
        )

    def to_record(self) -> dict[str, Any]:
        """Return values suitable for a parameterized DuckDB insert."""
        record = asdict(self)
        record["mode"] = self.mode.value
        return record
