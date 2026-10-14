"""R07's fixed growth cap, requiring independently captured annual EPS evidence."""
from __future__ import annotations

import math
import re

import pandas as pd

from stock_research.research.structure_events import REGISTERED_AT, _local_time


GROWTH_CAP = 0.20


def capped_value_observation(row, *, as_of):
    """Compare formulas on one source observation, without changing the whitelist.

    No quarterly EPS annualization, current industry backfill or reconstructed
    historical financial revision is accepted as the registered forward test.
    The source builder must adjust BPS and annual EPS to the same share basis
    as the observed raw price and bind the corporate-action source file.
    """
    cutoff = _local_time(as_of)
    required = {"observation_date", "captured_at", "book_value_per_share",
                "annual_eps_excl_comparable", "annual_report_period", "annual_announcement_date",
                "announcement_date", "annual_source_sha256", "eps_adjustment_source_sha256",
                "financial_source_sha256", "industry_source_sha256", "earnings_yoy", "close"}
    missing = sorted(required - set(row))
    if missing:
        return {"status": "data_limited", "missing_fields": missing, "value_line": None,
                "research_only": True}
    captured = _local_time(row["captured_at"])
    observed = pd.Timestamp(row["observation_date"])
    if pd.isna(observed):
        raise ValueError("Value observation date is invalid")
    day = cutoff.tz_localize(None).normalize()
    if (captured > cutoff or captured <= REGISTERED_AT or observed.normalize() != day
            or captured.tz_localize(None).normalize() != day):
        raise ValueError("R07 requires a visible real same-day capture after registration")
    for field in ("annual_source_sha256", "eps_adjustment_source_sha256",
                  "financial_source_sha256", "industry_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(row[field])):
            raise ValueError("Value experiment source fingerprint is invalid")
    period = pd.to_datetime(str(row["annual_report_period"]), errors="coerce")
    annual_ann = pd.to_datetime(row["annual_announcement_date"], errors="coerce")
    latest_ann = pd.to_datetime(row["announcement_date"], errors="coerce")
    if (pd.isna(period) or (period.month, period.day) != (12, 31)
            or pd.isna(annual_ann) or pd.isna(latest_ann) or period > annual_ann
            or annual_ann > day or latest_ann > day):
        raise ValueError("Actual visible annual and latest financial announcements are required")
    bps, eps, growth, close = (float(row[field]) for field in
                             ("book_value_per_share", "annual_eps_excl_comparable", "earnings_yoy", "close"))
    if not all(math.isfinite(value) for value in (bps, eps, growth, close)):
        raise ValueError("Value experiment needs finite comparable financial fields")
    if eps <= 0 or close <= 0 or growth <= -1:
        return {"status": "ineligible", "reason": "nonpositive_earnings_or_invalid_value_inputs",
                "value_line": None, "research_only": True}
    original = bps + eps * (1 + growth) * 10
    capped = bps + eps * (1 + min(growth, GROWTH_CAP)) * 10
    if min(original, capped) <= 0:
        return {"status": "ineligible", "reason": "nonpositive_value_line", "value_line": None,
                "research_only": True}
    return {"status": "ok", "research_only": True, "strict_point_in_time": False,
            "growth_cap": GROWTH_CAP, "observed_growth": growth,
            "original_value_line": original, "value_line": capped, "close": close,
            "original_price_to_value": close / original, "price_to_value": close / capped,
            "annual_report_period": str(period.date()), "captured_at": captured.isoformat(),
            "source_sha256": {field: row[field] for field in required if field.endswith("sha256")},
            "industry_whitelist_changed": False, "production_configuration_changed": False}
