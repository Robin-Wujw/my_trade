# -*- coding: utf-8 -*-
"""Point-in-time audit helpers shared by selection and backtest scripts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd


REPORT_DEADLINES = {
    (3, 31): (4, 30),
    (6, 30): (8, 31),
    (9, 30): (10, 31),
    (12, 31): (4, 30),
}


def statutory_visible_date(report_period):
    report = pd.Timestamp(report_period)
    month, day = REPORT_DEADLINES[(report.month, report.day)]
    year = report.year + 1 if report.month == 12 else report.year
    return pd.Timestamp(year=year, month=month, day=day)


def audit_dates(report_period, formation_date, market_cutoff=None):
    report = pd.Timestamp(report_period)
    formation = pd.Timestamp(formation_date)
    visible = statutory_visible_date(report)
    issues = []
    if formation < visible:
        issues.append(f"formation_date {formation.date()} is before statutory visibility {visible.date()}")
    if market_cutoff is not None and pd.Timestamp(market_cutoff) > formation:
        issues.append(f"market_cutoff {market_cutoff} is later than formation_date {formation.date()}")
    return {
        "report_period": report.strftime("%Y-%m-%d"),
        "statutory_visible_date": visible.strftime("%Y-%m-%d"),
        "formation_date": formation.strftime("%Y-%m-%d"),
        "date_status": "unsafe" if issues else "safe",
        "date_issues": issues,
    }


def read_metadata(data_path):
    path = data_path + ".meta.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata(data_path, metadata):
    payload = dict(metadata)
    payload.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    with open(data_path + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def audit_source(path, expected_kind=None):
    metadata = read_metadata(path)
    issues = []
    if not metadata:
        issues.append("missing provenance sidecar")
    if expected_kind and metadata.get("kind") != expected_kind:
        issues.append(f"expected kind={expected_kind}, got {metadata.get('kind') or 'missing'}")
    status = metadata.get("point_in_time_status", "warning" if issues else "safe")
    if status == "unsafe" and not issues:
        issues.append(metadata.get("point_in_time_note", "source marked unsafe"))
    return status, issues, metadata


def require_safe(audits, allow_unsafe=False):
    unsafe = [item for item in audits if item.get("status") == "unsafe"]
    if unsafe and not allow_unsafe:
        detail = "; ".join(f"{item['name']}: {', '.join(item.get('issues') or ['unsafe'])}" for item in unsafe)
        raise SystemExit("Point-in-time audit failed: " + detail + ". Use --allow-unsafe only for labelled research runs.")

