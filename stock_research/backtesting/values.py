"""Shared finite-number and industry-tag normalization."""
import math
import re
import pandas as pd


def _finite_float(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not math.isfinite(float(number)):
        return None
    return float(number)



def _split_industry_tags(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = re.split(r"[、,，|/;；]+", str(value))
    tags = []
    for item in raw_items:
        tag = str(item or "").strip()
        if tag and tag.lower() != "nan" and tag not in tags:
            tags.append(tag)
    return tags

