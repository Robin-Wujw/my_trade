"""Candidate eligibility and portfolio classification rules, separate from execution."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from stock_research.backtesting.signals import _enabled


def risk_stock_name_reason(name):
    text = str(name or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    upper = text.upper()
    if "ST" in upper or "退" in text:
        return "risk_name_or_status"
    return ""


def candidate_risk_reason(candidate):
    if not candidate:
        return ""
    return risk_stock_name_reason(
        candidate.get("name") or "",
    )


def left_candidate_kind(candidate):
    sources = set(str((candidate or {}).get("candidate_source") or "").split("+"))
    return "pe_pb" if sources & {"pe_model", "pb_model"} else "value"


def candidate_industry_tags(candidate):
    raw = candidate.get("industry") or candidate.get("mainline_boards") or ""
    if pd.isna(raw):
        return set()
    aliases = {
        "通信网络设备及器件": "通信设备",
        "光通信设备": "通信设备",
        "通信终端及配件": "通信设备",
    }
    tags = {
        aliases.get(tag.strip(), tag.strip())
        for tag in re.split(r"[、,，;；/|+]", str(raw))
        if tag.strip()
    }
    return tags


def left_value_falsification_reason(candidate):
    if not candidate:
        return ""
    reason = str(candidate.get("value_falsification_reason") or "").strip()
    if reason:
        return reason
    if _enabled(candidate.get("value_falsified"), default=False):
        return str(
            candidate.get("candidate_failure_reason")
            or "value thesis falsified by financial snapshot"
        ).strip()
    return ""


def candidate_failure_reason(candidate):
    if not candidate:
        return ""
    return str(candidate.get("candidate_failure_reason") or "").strip()






def candidate_passes_hard_gate(candidate):
    if not candidate:
        return False
    quality = pd.to_numeric(candidate.get("quality_score"), errors="coerce")
    growth = pd.to_numeric(candidate.get("earnings_yoy"), errors="coerce")
    market_cap = pd.to_numeric(candidate.get("mktcap"), errors="coerce")
    return (
        pd.notna(quality)
        and pd.notna(growth)
        and pd.notna(market_cap)
        and float(quality) >= 70.0
        and float(growth) >= 0.10
        and float(market_cap) >= 100.0
    )


def right_candidate_can_evaluate(candidate):
    if not candidate:
        return False
    if candidate_risk_reason(candidate):
        return False
    if (
        _enabled(candidate.get("selected_for_trading"), default=True)
        and _enabled(candidate.get("signal_eligible"), default=True)
    ):
        return True
    # Ranking out of the daily Top10 is a report/display limit.  It should
    # not prevent a right-side model candidate from waiting for its own
    # high-quality technical setup.
    return (
        candidate_is_daily_quota_diagnostic(candidate)
        and _enabled(candidate.get("allow_right"), default=False)
        and candidate_has_exceptional_right_profile(candidate)
        and candidate_passes_hard_gate(candidate)
        and not left_value_falsification_reason(candidate)
    )


def left_candidate_can_add(candidate):
    if not candidate:
        return False
    if candidate_risk_reason(candidate):
        return False
    sources = set(str(candidate.get("candidate_source") or "").split("+"))
    quality = pd.to_numeric(candidate.get("quality_score"), errors="coerce")
    market_cap = pd.to_numeric(candidate.get("mktcap"), errors="coerce")
    if "pe_model" in sources or "pb_model" in sources:
        return pe_pb_candidate_can_add(
            candidate, falsification_reason=left_value_falsification_reason(candidate),
        )
    price_to_value = pd.to_numeric(candidate.get("price_to_value"), errors="coerce")
    growth = pd.to_numeric(candidate.get("earnings_yoy"), errors="coerce")
    return (
        _enabled(candidate.get("selected_for_trading"), default=True)
        and _enabled(candidate.get("signal_eligible"), default=True)
        and _enabled(candidate.get("allow_left"), default=False)
        and "value_model" in sources
        and _enabled(candidate.get("value_industry_allowed"), default=False)
        and pd.notna(price_to_value)
        and 0.88 <= float(price_to_value) <= 1.03
        and pd.notna(quality)
        and float(quality) >= 80.0
        and pd.notna(growth)
        and float(growth) >= 0.20
        and pd.notna(market_cap)
        and float(market_cap) >= 150.0
        and not left_value_falsification_reason(candidate)
    )

from stock_research.research.legacy_candidates import (
    candidate_has_exceptional_right_profile, candidate_is_daily_quota_diagnostic,
    pe_pb_candidate_can_add,
)
