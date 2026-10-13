"""Build a PIT hybrid stream: QuantsPlaybook right lane plus value left lane."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.backtesting.workflow import (
    load_candidate_snapshots,
    validate_candidate_manifest_financial_point_in_time,
    validate_candidate_manifest_industry_point_in_time,
)
from stock_research.core.paths import PATHS
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


DEFAULT_FACTOR_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)
DEFAULT_LEFT_CANDIDATES = (
    PATHS.runtime_root / "backtests"
    / "tushare_official_candidates_2021_to_20260721_merged"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_hybrid_low_corr_value_2021_to_20260721"
)
HYBRID_FACTOR = "hybrid_low_corr_value"
RIGHT_INDUSTRY_DATA_SOURCE = "tushare/pro:sw_member"
RIGHT_INDUSTRY_LEVEL_COLUMN = "l2_name"
RIGHT_INVERSE_VOLATILITY_WINDOW = 252
RIGHT_RANK_PERSISTENCE_WINDOW = 20
INDUSTRY_MANIFEST_FIELDS = (
    "industry_source_path",
    "industry_source_sha256",
    "industry_source_modified_at",
    "industry_as_of_date",
    "industry_data_source",
    "industry_point_in_time_status",
    "industry_mapping_count",
    "industry_universe_count",
    "industry_coverage_ratio",
    "industry_point_in_time_note",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _sources(row: dict) -> set[str]:
    return {
        value.strip()
        for value in str(row.get("candidate_source") or "").split("+")
        if value.strip()
    }


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _normalize_stock_code(value) -> str:
    text = _clean_text(value).lower().replace("_", ".")
    if not text:
        return ""
    if "." not in text:
        return text
    left, right = text.split(".", 1)
    if left in {"sh", "sz", "bj"}:
        return f"{left}.{right.zfill(6)}"
    if right in {"sh", "sz", "bj"}:
        return f"{right}.{left.zfill(6)}"
    return text


def _industry_member_date(value) -> pd.Timestamp:
    text = _clean_text(value)
    if not text:
        return pd.NaT
    if text.endswith(".0"):
        text = text[:-2]
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _load_right_industry_lookup(manifest: dict, directory: Path):
    if manifest.get("industry_point_in_time") is not True:
        raise RuntimeError("right industry cap requires point-in-time industry data")
    if manifest.get("industry_point_in_time_status") != "safe":
        raise RuntimeError("right industry cap requires safe industry intervals")
    if manifest.get("industry_data_source") != RIGHT_INDUSTRY_DATA_SOURCE:
        raise RuntimeError(
            "right industry cap requires Tushare sw_member intervals"
        )
    expected_sha256 = _clean_text(manifest.get("industry_source_sha256"))
    source_value = _clean_text(manifest.get("industry_source_path"))
    if not expected_sha256 or not source_value:
        raise RuntimeError("right industry cap source fingerprint is missing")
    source_path = Path(source_value)
    candidates = [source_path]
    local_copy = directory / source_path.name
    if local_copy not in candidates:
        candidates.append(local_copy)
    source = next(
        (
            path for path in candidates
            if path.is_file() and _sha256(path) == expected_sha256
        ),
        None,
    )
    if source is None:
        raise RuntimeError("right industry cap source is missing or hash-mismatched")

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {
        "code", RIGHT_INDUSTRY_LEVEL_COLUMN, "in_date", "out_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"right industry cap source is missing columns: {missing}"
        )
    frame = frame.copy()
    frame["code"] = frame["code"].map(_normalize_stock_code)
    frame["industry_l2"] = frame[RIGHT_INDUSTRY_LEVEL_COLUMN].map(_clean_text)
    frame["in_date_ts"] = frame["in_date"].map(_industry_member_date)
    frame["out_date_ts"] = frame["out_date"].map(_industry_member_date)
    frame = frame[
        frame["code"].ne("")
        & frame["industry_l2"].ne("")
        & frame["in_date_ts"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError("right industry cap source has no usable intervals")
    grouped = {
        code: group.sort_values(
            ["in_date_ts", "out_date_ts"], na_position="last"
        )
        for code, group in frame.groupby("code")
    }
    cache = {}

    def lookup(code, observation_date) -> str:
        normalized_code = _normalize_stock_code(code)
        date = pd.Timestamp(observation_date).normalize()
        key = (normalized_code, date.strftime("%Y-%m-%d"))
        if key in cache:
            return cache[key]
        group = grouped.get(normalized_code)
        if group is None or group.empty:
            cache[key] = ""
            return ""
        active = group[
            (group["in_date_ts"] <= date)
            & (
                group["out_date_ts"].isna()
                | (group["out_date_ts"] >= date)
            )
        ]
        labels = sorted(set(active["industry_l2"].dropna().map(_clean_text)) - {""})
        if len(labels) > 1:
            raise RuntimeError(
                "conflicting active Shenwan level-2 memberships for "
                f"{normalized_code} on {date:%Y-%m-%d}: {labels}"
            )
        cache[key] = labels[0] if labels else ""
        return cache[key]

    return lookup


_LEFT_MODEL_SOURCES = ("value_model", "pe_model", "pb_model")


def _left_rows(rows: list[dict]) -> dict[str, dict]:
    selected = {}
    for source in rows:
        sources = _sources(source)
        lane_model = next(
            (model for model in _LEFT_MODEL_SOURCES if model in sources),
            None,
        )
        if lane_model is None:
            continue
        if not all(
            _truthy(source.get(field))
            for field in (
                "allow_left", "selected_for_trading", "signal_eligible",
                "financial_point_in_time", "industry_point_in_time",
            )
        ):
            continue
        # value_model 左侧依赖行业白名单；PE/PB 低估通道不使用行业白名单。
        if (
            lane_model == "value_model"
            and not _truthy(source.get("value_industry_allowed"))
        ):
            continue
        if _truthy(source.get("value_falsified")) or _clean_text(
            source.get("value_falsification_reason")
        ):
            continue
        row = dict(source)
        row.update({
            "candidate_source": lane_model,
            "selection_profile": "hybrid_value_left",
            "hybrid_lane": (
                "left_value" if lane_model == "value_model" else "left_pe_pb"
            ),
            "allow_left": True,
            "allow_right": False,
        })
        selected[str(row["code"])] = row
    return selected


def _right_rows(
    rows: list[dict],
    *,
    right_factor: str,
) -> dict[str, dict]:
    lane = f"right_{right_factor.removeprefix('playbook_')}"
    selected = {}
    for source in rows:
        if not all(
            _truthy(source.get(field))
            for field in (
                "selected_for_trading", "signal_eligible", "allow_right",
            )
        ):
            continue
        row = dict(source)
        row.update({
            "candidate_source": "quantsplaybook_factor",
            "selection_profile": "quantsplaybook_factor_only",
            "hybrid_lane": lane,
            "allow_left": False,
            "allow_right": True,
        })
        selected[str(row["code"])] = row
    return selected


def _candidate_score(row: dict) -> float:
    value = pd.to_numeric(row.get("candidate_score"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def merge_hybrid_rows(
    right_rows: list[dict],
    left_rows: list[dict],
    *,
    right_factor: str = "playbook_low_corr",
    left_extra_rows: list[dict] | None = None,
) -> list[dict]:
    right = _right_rows(right_rows, right_factor=right_factor)
    left = _left_rows(left_rows)
    extra = _left_rows(left_extra_rows or [])
    right_lane = f"right_{right_factor.removeprefix('playbook_')}"
    result = []
    for code in sorted(set(right) | set(left) | set(extra)):
        if code in right and (code in left or code in extra):
            row = dict(left[code] if code in left else extra[code])
            factor = right[code]
            for field in (
                "playbook_factor", "playbook_factor_score",
                "playbook_factor_rank", "candidate_score",
                "strategy_part", "selection_engine", "selection_profile",
                "factor_horizon_rule_cn", "factor_horizon_holding_days",
                "factor_horizon_channel_eligible",
                "right_inverse_volatility_multiplier", "right_volatility_252",
            ):
                if field in factor:
                    row[field] = factor[field]
            row.update({
                "candidate_source": (
                    f"{row.get('candidate_source')}+quantsplaybook_factor"
                ),
                "hybrid_lane": f"{row.get('hybrid_lane', 'left_value')}+{right_lane}",
                "allow_left": True,
                "allow_right": True,
            })
        elif code in right:
            row = right[code]
        else:
            row = left[code] if code in left else extra[code]
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            -_candidate_score(row),
            str(row.get("code") or ""),
        ),
    )


def _read_manifest(path: Path, *, lane: str) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != SNAPSHOT_VERSION:
        raise RuntimeError(f"{lane} candidate version is stale: {manifest_path}")
    if manifest.get("value_industry_rule_version") != VALUE_INDUSTRY_RULE_VERSION:
        raise RuntimeError(f"{lane} value-industry rule is stale: {manifest_path}")
    return manifest


def _manifest_dates(manifest: dict, *, lane: str) -> set[str]:
    rows = manifest.get("snapshots")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{lane} candidate manifest has no snapshots")
    dates = set()
    for row in rows:
        value = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(value):
            raise RuntimeError(f"{lane} candidate manifest has an invalid date")
        date = value.strftime("%Y-%m-%d")
        if date in dates:
            raise RuntimeError(
                f"{lane} candidate manifest has duplicate date: {date}"
            )
        dates.add(date)
    return dates


def _select_right_rows(
    rows,
    *,
    top_n=None,
    exclude_codes=(),
    industry_lookup=None,
    industry_max_per_day=None,
    observation_date=None,
):
    excluded = {str(code).strip() for code in exclude_codes if str(code).strip()}
    selected = [
        dict(row) for row in rows
        if str(row.get("code") or "").strip() not in excluded
    ]
    if top_n is None and industry_max_per_day is None:
        return selected
    if top_n is None:
        raise ValueError("right_top_n is required with an industry cap")
    limit = int(top_n)
    if limit <= 0:
        raise ValueError("right_top_n must be positive or None")

    def rank_key(row):
        rank = pd.to_numeric(row.get("playbook_factor_rank"), errors="coerce")
        score = pd.to_numeric(row.get("playbook_factor_score"), errors="coerce")
        if pd.isna(score):
            score = pd.to_numeric(row.get("candidate_score"), errors="coerce")
        return (
            float(rank) if pd.notna(rank) else float("inf"),
            -float(score) if pd.notna(score) else 0.0,
            str(row.get("code") or ""),
        )

    ranked = sorted(selected, key=rank_key)
    if industry_max_per_day is None:
        return ranked[:limit]
    industry_limit = int(industry_max_per_day)
    if industry_limit <= 0:
        raise ValueError("right_industry_max_per_day must be positive or None")
    if industry_lookup is None or observation_date is None:
        raise ValueError(
            "industry lookup and observation date are required with an industry cap"
        )
    result = []
    industry_counts = {}
    for source in ranked:
        code = str(source.get("code") or "").strip()
        industry = _clean_text(industry_lookup(code, observation_date))
        if not industry:
            raise RuntimeError(
                f"missing Shenwan level-2 membership for {code} on "
                f"{pd.Timestamp(observation_date):%Y-%m-%d}"
            )
        if industry_counts.get(industry, 0) >= industry_limit:
            continue
        row = dict(source)
        row["right_industry_l2"] = industry
        result.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(result) >= limit:
            break
    if len(result) != limit:
        raise RuntimeError(
            "right industry cap cannot refill the requested depth: "
            f"requested={limit} selected={len(result)} date="
            f"{pd.Timestamp(observation_date):%Y-%m-%d}"
        )
    return result


def _apply_right_inverse_volatility_sizing(
    snapshots: dict[str, list[dict]],
    *,
    price_directory: str | Path,
    window: int = RIGHT_INVERSE_VOLATILITY_WINDOW,
) -> dict:
    lookback = int(window)
    if lookback <= 1:
        raise ValueError("inverse volatility window must exceed one session")
    directory = Path(price_directory)
    if not directory.is_dir():
        raise RuntimeError(
            f"inverse volatility prepared-price directory is missing: {directory}"
        )
    codes = sorted({
        _normalize_stock_code(row.get("code"))
        for rows in snapshots.values()
        for row in rows
        if _normalize_stock_code(row.get("code"))
    })
    volatility_by_code = {}
    for code in codes:
        path = directory / f"{code.replace('.', '_')}.pkl"
        if not path.is_file():
            raise RuntimeError(
                f"inverse volatility prepared price is missing for {code}: {path}"
            )
        frame = pd.read_pickle(path)
        if not {"date", "close"}.issubset(frame.columns):
            raise RuntimeError(
                f"inverse volatility price frame lacks date/close: {path}"
            )
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        closes = pd.to_numeric(frame["close"], errors="coerce").where(
            pd.to_numeric(frame["close"], errors="coerce").gt(0)
        )
        work = pd.DataFrame({"date": dates, "close": closes}).dropna()
        work = work.drop_duplicates("date", keep="last").sort_values("date")
        returns = work["close"].pct_change(fill_method=None)
        volatility = returns.rolling(
            lookback, min_periods=lookback,
        ).std(ddof=1)
        volatility_by_code[code] = pd.Series(
            volatility.to_numpy(), index=work["date"], dtype=float,
        )

    available_count = 0
    fallback_count = 0
    multipliers = []
    for date, rows in snapshots.items():
        observation_date = pd.Timestamp(date).normalize()
        inverse = {}
        sigma_by_code = {}
        for row in rows:
            code = _normalize_stock_code(row.get("code"))
            series = volatility_by_code[code]
            sigma = series.get(observation_date, np.nan)
            if pd.notna(sigma) and np.isfinite(float(sigma)) and float(sigma) > 0:
                sigma_by_code[code] = float(sigma)
                inverse[code] = 1.0 / float(sigma)
        inverse_mean = float(np.mean(list(inverse.values()))) if inverse else None
        for row in rows:
            code = _normalize_stock_code(row.get("code"))
            if inverse_mean is None or code not in inverse:
                multiplier = 1.0
                sigma = None
                fallback_count += 1
            else:
                multiplier = inverse[code] / inverse_mean
                sigma = sigma_by_code[code]
                available_count += 1
            row["right_inverse_volatility_multiplier"] = float(multiplier)
            row["right_volatility_252"] = sigma
            multipliers.append(float(multiplier))
    return {
        "window_sessions": lookback,
        "price_directory": str(directory),
        "price_file_count": len(codes),
        "available_candidate_rows": available_count,
        "fallback_candidate_rows": fallback_count,
        "multiplier_min": min(multipliers) if multipliers else None,
        "multiplier_mean": float(np.mean(multipliers)) if multipliers else None,
        "multiplier_max": max(multipliers) if multipliers else None,
        "point_in_time_rule": (
            "adjusted close returns through observation date; no forward fill; "
            "insufficient 252-return history uses multiplier 1"
        ),
    }


def _apply_right_rank_persistence_selection(
    snapshots: dict[str, list[dict]],
    *,
    top_n: int,
    source_top_n: int,
    window: int = RIGHT_RANK_PERSISTENCE_WINDOW,
    exclude_codes=(),
) -> tuple[dict[str, list[dict]], dict]:
    """Select current candidates by their trailing point-in-time rank quality."""
    selected_top_n = int(top_n)
    saved_depth = int(source_top_n)
    lookback = int(window)
    if selected_top_n <= 0:
        raise ValueError("right rank persistence top_n must be positive")
    if saved_depth < selected_top_n:
        raise ValueError(
            "right rank persistence source_top_n must cover selected top_n"
        )
    if lookback <= 1:
        raise ValueError("right rank persistence window must exceed one session")

    dates = sorted(snapshots)
    filtered = {
        date: _select_right_rows(
            snapshots[date], exclude_codes=exclude_codes,
        )
        for date in dates
    }
    rank_quality_by_date = {}
    for date in dates:
        quality = {}
        for row in filtered[date]:
            rank = pd.to_numeric(
                row.get("playbook_factor_rank"), errors="coerce",
            )
            if pd.isna(rank) or float(rank) < 1 or float(rank) > saved_depth:
                continue
            quality[str(row.get("code") or "")] = (
                saved_depth - float(rank) + 1.0
            ) / saved_depth
        rank_quality_by_date[date] = quality

    result = {}
    fallback_dates = 0
    changed_dates = 0
    total_replacements = 0
    deepest_selected_rank = 0
    for position, date in enumerate(dates):
        baseline = _select_right_rows(
            filtered[date], top_n=selected_top_n,
        )
        if len(baseline) != selected_top_n:
            raise RuntimeError(
                "right rank persistence cannot fill requested depth: "
                f"date={date} requested={selected_top_n} "
                f"selected={len(baseline)}"
            )
        if position + 1 < lookback:
            chosen = baseline
            fallback_dates += 1
        else:
            history_dates = dates[position - lookback + 1:position + 1]
            ranked = []
            for source in filtered[date]:
                row = dict(source)
                code = str(row.get("code") or "")
                qualities = [
                    rank_quality_by_date[history_date].get(code, 0.0)
                    for history_date in history_dates
                ]
                current_rank = pd.to_numeric(
                    row.get("playbook_factor_rank"), errors="coerce",
                )
                score = float(np.mean(qualities))
                appearances = sum(value > 0 for value in qualities)
                row["right_rank_persistence_20_score"] = score
                row["right_rank_persistence_20_appearances"] = appearances
                ranked.append((
                    -score,
                    float(current_rank) if pd.notna(current_rank) else np.inf,
                    code,
                    row,
                ))
            chosen = [item[-1] for item in sorted(ranked)[:selected_top_n]]

        baseline_codes = {str(row.get("code") or "") for row in baseline}
        chosen_codes = {str(row.get("code") or "") for row in chosen}
        replacements = len(chosen_codes - baseline_codes)
        if replacements:
            changed_dates += 1
            total_replacements += replacements
        for row in chosen:
            rank = pd.to_numeric(
                row.get("playbook_factor_rank"), errors="coerce",
            )
            if pd.notna(rank):
                deepest_selected_rank = max(
                    deepest_selected_rank, int(rank),
                )
        result[date] = chosen

    return result, {
        "window_sessions": lookback,
        "source_top_n": saved_depth,
        "selected_top_n": selected_top_n,
        "warmup_fallback_dates": fallback_dates,
        "selection_changed_dates": changed_dates,
        "total_replacements": total_replacements,
        "deepest_selected_source_rank": deepest_selected_rank,
        "formula": (
            "mean over trailing 20 snapshots of "
            "(source_top_n-rank+1)/source_top_n; missing from saved "
            "Top100 contributes zero; current Top100 only"
        ),
        "point_in_time_rule": (
            "observation-date selection uses only source ranks dated on or "
            "before that observation date"
        ),
    }


def _manifest_research_only(manifest: dict) -> bool:
    if manifest.get("research_only") is True:
        return True
    factor_metadata = manifest.get("factor_metadata")
    return bool(
        isinstance(factor_metadata, dict)
        and factor_metadata.get("research_only") is True
    )


def build_hybrid_candidates(
    *,
    right_directory: str | Path,
    left_directory: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
    left_extra_directory: str | Path | None = None,
    allow_unsafe_financial: bool = False,
    allow_unsafe_industry: bool = False,
    right_top_n: int | None = None,
    right_industry_max_per_day: int | None = None,
    include_right_inverse_volatility_252: bool = False,
    include_right_rank_persistence_20: bool = False,
    right_inverse_volatility_price_directory: str | Path | None = None,
    exclude_right_codes=(),
) -> Path:
    right_directory = Path(right_directory)
    left_directory = Path(left_directory)
    output = Path(output_directory)
    right_manifest = _read_manifest(right_directory, lane="right")
    left_manifest = _read_manifest(left_directory, lane="left")
    extra_manifest = None
    if left_extra_directory is not None:
        left_extra_directory = Path(left_extra_directory)
        extra_manifest = _read_manifest(left_extra_directory, lane="left-extra")
    right_financial_status = validate_candidate_manifest_financial_point_in_time(
        right_directory,
        allow_unsafe_financial=allow_unsafe_financial,
    )
    left_financial_status = validate_candidate_manifest_financial_point_in_time(
        left_directory,
        allow_unsafe_financial=allow_unsafe_financial,
    )
    left_industry_status = validate_candidate_manifest_industry_point_in_time(
        left_directory,
        allow_unsafe_industry=allow_unsafe_industry,
    )
    extra_financial_status = None
    extra_industry_status = None
    if left_extra_directory is not None:
        extra_financial_status = validate_candidate_manifest_financial_point_in_time(
            left_extra_directory,
            allow_unsafe_financial=allow_unsafe_financial,
        )
        extra_industry_status = validate_candidate_manifest_industry_point_in_time(
            left_extra_directory,
            allow_unsafe_industry=allow_unsafe_industry,
        )
    right_uses_financial = right_manifest.get("uses_financial_data") is True
    right_research_only = _manifest_research_only(right_manifest)
    left_research_only = _manifest_research_only(left_manifest)
    extra_research_only = (
        _manifest_research_only(extra_manifest)
        if extra_manifest is not None else False
    )
    right_strict_financial = (
        right_manifest.get("strict_financial_point_in_time") is not False
        if right_uses_financial else True
    )
    strict_financial = bool(
        left_manifest.get("strict_financial_point_in_time") is True
        and right_strict_financial
        and (
            extra_manifest is None
            or extra_manifest.get("strict_financial_point_in_time") is True
        )
    )
    financial_point_in_time = all(
        status.get("financial_point_in_time") is True
        and not status.get("unsafe_dates")
        for status in (
            right_financial_status,
            left_financial_status,
            *(
                (extra_financial_status,)
                if extra_financial_status is not None else ()
            ),
        )
    )
    industry_point_in_time = all(
        status.get("industry_point_in_time") is True
        and not status.get("unsafe_dates")
        and not status.get("metadata_errors")
        for status in (
            left_industry_status,
            *((extra_industry_status,) if extra_industry_status is not None else ()),
        )
    )
    right_factor = str(
        right_manifest.get("factor") or right_directory.name
    ).strip()
    if not right_factor:
        raise RuntimeError(f"right factor is missing: {right_directory}")
    right = load_candidate_snapshots(right_directory, start_date, end_date)
    excluded_right_codes = tuple(sorted({
        str(code).strip()
        for code in exclude_right_codes
        if str(code).strip()
    }))
    industry_lookup = None
    source_top_n = pd.to_numeric(right_manifest.get("top_n"), errors="coerce")
    if include_right_rank_persistence_20:
        if right_top_n is None:
            raise ValueError(
                "right_top_n is required with right rank persistence"
            )
        if pd.isna(source_top_n):
            raise RuntimeError(
                "right rank persistence requires a saved source top_n"
            )
        if right_industry_max_per_day is not None:
            raise ValueError(
                "right rank persistence cannot be combined with an industry cap"
            )
        if include_right_inverse_volatility_252:
            raise ValueError(
                "right rank persistence cannot be combined with inverse-volatility sizing"
            )
    if right_industry_max_per_day is not None:
        if right_top_n is None:
            raise ValueError("right_top_n is required with an industry cap")
        if pd.isna(source_top_n) or int(source_top_n) < int(right_top_n):
            raise RuntimeError(
                "right industry cap requires a saved source depth at least as "
                "large as right_top_n"
            )
        industry_lookup = _load_right_industry_lookup(
            left_manifest, left_directory,
        )
    selection_changed_dates = 0
    total_replacements = 0
    deepest_selected_rank = 0
    rank_persistence_metadata = None
    if include_right_rank_persistence_20:
        right, rank_persistence_metadata = (
            _apply_right_rank_persistence_selection(
                right,
                top_n=right_top_n,
                source_top_n=int(source_top_n),
                exclude_codes=excluded_right_codes,
            )
        )
        selection_changed_dates = rank_persistence_metadata[
            "selection_changed_dates"
        ]
        total_replacements = rank_persistence_metadata["total_replacements"]
        deepest_selected_rank = rank_persistence_metadata[
            "deepest_selected_source_rank"
        ]
    else:
        selected_right = {}
        for date, rows in right.items():
            baseline = _select_right_rows(
                rows,
                top_n=right_top_n,
                exclude_codes=excluded_right_codes,
            )
            selected = _select_right_rows(
                rows,
                top_n=right_top_n,
                exclude_codes=excluded_right_codes,
                industry_lookup=industry_lookup,
                industry_max_per_day=right_industry_max_per_day,
                observation_date=date,
            )
            baseline_codes = {str(row.get("code") or "") for row in baseline}
            selected_codes = {str(row.get("code") or "") for row in selected}
            replacements = len(selected_codes - baseline_codes)
            if replacements:
                selection_changed_dates += 1
                total_replacements += replacements
            for row in selected:
                rank = pd.to_numeric(
                    row.get("playbook_factor_rank"), errors="coerce",
                )
                if pd.notna(rank):
                    deepest_selected_rank = max(
                        deepest_selected_rank, int(rank),
                    )
            selected_right[date] = selected
        right = selected_right
    inverse_volatility_metadata = None
    if include_right_inverse_volatility_252:
        inverse_volatility_metadata = _apply_right_inverse_volatility_sizing(
            right,
            price_directory=(
                right_inverse_volatility_price_directory
                or (DEFAULT_FACTOR_ROOT / "prepared_prices")
            ),
        )
    left = load_candidate_snapshots(left_directory, start_date, end_date)
    extra = (
        load_candidate_snapshots(left_extra_directory, start_date, end_date)
        if left_extra_directory is not None
        else {}
    )
    right_dates = set(right)
    left_dates = set(left)
    extra_dates = set(extra)
    if left_dates != right_dates:
        missing = sorted(right_dates - left_dates)
        surplus = sorted(left_dates - right_dates)
        raise RuntimeError(
            "left candidate coverage differs from right lane: "
            f"missing={missing[:5]} surplus={surplus[:5]}"
        )
    if extra_manifest is not None and extra_dates != right_dates:
        missing = sorted(right_dates - extra_dates)
        surplus = sorted(extra_dates - right_dates)
        raise RuntimeError(
            "left-extra candidate coverage differs from right lane: "
            f"missing={missing[:5]} surplus={surplus[:5]}"
        )
    dates = sorted(right_dates)
    if not dates:
        raise RuntimeError("hybrid candidate date range is empty")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    total_left = 0
    total_right = 0
    total_overlap = 0
    for date in dates:
        rows = merge_hybrid_rows(
            right.get(date, []),
            left.get(date, []),
            right_factor=right_factor,
            left_extra_rows=extra.get(date, []),
        )
        frame = pd.DataFrame(rows)
        path = output / f"candidates_{date}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        lanes = frame.get("hybrid_lane", pd.Series(dtype=str)).fillna("")
        left_count = int(lanes.str.startswith("left_").sum())
        right_count = int(lanes.str.contains("right_", regex=False).sum())
        overlap_count = int(lanes.str.contains("+", regex=False).sum())
        total_left += left_count
        total_right += right_count
        total_overlap += overlap_count
        snapshots.append({
            "date": date,
            "file": path.name,
            "sha256": _sha256(path),
            "candidate_count": len(frame),
            "signal_eligible_count": len(frame),
            "left_candidate_count": left_count,
            "right_candidate_count": right_count,
            "overlap_candidate_count": overlap_count,
            "financial_point_in_time": financial_point_in_time,
            "strict_financial_point_in_time": strict_financial,
            "industry_point_in_time": industry_point_in_time,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "snapshot_count": len(snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": dates[0],
        "end_date": dates[-1],
        "financial_point_in_time": financial_point_in_time,
        "strict_financial_point_in_time": strict_financial,
        "uses_financial_data": True,
        "unsafe_snapshot_count": (
            0 if financial_point_in_time and industry_point_in_time
            else len(snapshots)
        ),
        "industry_point_in_time": industry_point_in_time,
        **{
            field: left_manifest.get(field)
            for field in INDUSTRY_MANIFEST_FIELDS
        },
        "selection_engine": "quantsplaybook_right_plus_value_left",
        "selection_profile": output.name,
        "signal_timing": "close_t_signal_earliest_execution_t_plus_1",
        "right_lane": {
            "factor": right_factor,
            "candidate_directory": str(right_directory),
            "manifest_sha256": _sha256(right_directory / "manifest.json"),
            "industry_data_used": right_industry_max_per_day is not None,
            "industry_data_source": (
                left_manifest.get("industry_data_source")
                if right_industry_max_per_day is not None else None
            ),
            "industry_source_sha256": (
                left_manifest.get("industry_source_sha256")
                if right_industry_max_per_day is not None else None
            ),
            "industry_level": (
                "Shenwan level 2"
                if right_industry_max_per_day is not None else None
            ),
            "industry_level_column": (
                RIGHT_INDUSTRY_LEVEL_COLUMN
                if right_industry_max_per_day is not None else None
            ),
            "max_per_industry_per_day": (
                None if right_industry_max_per_day is None
                else int(right_industry_max_per_day)
            ),
            "selection_changed_dates": selection_changed_dates,
            "total_replacements": total_replacements,
            "deepest_selected_source_rank": deepest_selected_rank,
            "inverse_volatility_sizing": inverse_volatility_metadata,
            "rank_persistence_selection": rank_persistence_metadata,
            "uses_financial_data": right_uses_financial,
            "strict_financial_point_in_time": right_strict_financial,
            "research_only": right_research_only,
            "source_top_n": right_manifest.get("top_n"),
            "selected_top_n": (
                None if right_top_n is None else int(right_top_n)
            ),
            "excluded_codes": list(excluded_right_codes),
        },
        "left_lane": {
            "factor": "basic_value_line",
            "candidate_directory": str(left_directory),
            "manifest_sha256": _sha256(left_directory / "manifest.json"),
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
        },
        "left_extra_lane": (
            {
                "factor": "pe_pb_low_value",
                "candidate_directory": str(left_extra_directory),
                "manifest_sha256": _sha256(
                    left_extra_directory / "manifest.json"
                ),
                "uses_financial_data": extra_manifest.get(
                    "uses_financial_data"
                ),
                "strict_financial_point_in_time": extra_manifest.get(
                    "strict_financial_point_in_time"
                ),
                "industry_point_in_time": extra_manifest.get(
                    "industry_point_in_time"
                ),
                **{
                    field: extra_manifest.get(field)
                    for field in INDUSTRY_MANIFEST_FIELDS
                },
            }
            if extra_manifest is not None
            else None
        ),
        "lane_policy": (
            f"right candidates only from {right_factor}; left candidates "
            "from executable value_model"
            + (" plus PE/PB low-valuation (top-N/day)" if extra_manifest else "")
            + (
                "; right lane caps Shenwan level-2 membership at "
                f"{int(right_industry_max_per_day)} names per day and refills "
                "from saved source ranks"
                if right_industry_max_per_day is not None else ""
            )
            + (
                "; right Top50 is reranked by fixed trailing-20 source-rank "
                "persistence within the current Top100"
                if include_right_rank_persistence_20 else ""
            )
            + "; scores are not added across lanes"
        ),
        "realized_outcome_exclusions_used": bool(excluded_right_codes),
        "research_only": bool(
            not strict_financial
            or not financial_point_in_time
            or not industry_point_in_time
            or right_research_only
            or left_research_only
            or extra_research_only
            or right_industry_max_per_day is not None
            or include_right_inverse_volatility_252
            or include_right_rank_persistence_20
            or bool(excluded_right_codes)
        ),
        "total_left_candidate_rows": total_left,
        "total_right_candidate_rows": total_right,
        "total_overlap_candidate_rows": total_overlap,
        "snapshots": snapshots,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--right-directory",
        default=str(DEFAULT_FACTOR_ROOT / "candidates" / "playbook_low_corr"),
    )
    parser.add_argument("--left-directory", default=str(DEFAULT_LEFT_CANDIDATES))
    parser.add_argument(
        "--right-top-n",
        type=int,
        default=None,
        help=(
            "optional fixed right-lane depth selected from the saved source "
            "ranks; use 50 with a Top100 source for replacement robustness"
        ),
    )
    parser.add_argument(
        "--exclude-right-codes",
        default="",
        help=(
            "comma-separated right-lane codes removed before --right-top-n "
            "is applied, allowing deeper saved ranks to replace exclusions"
        ),
    )
    parser.add_argument(
        "--right-industry-max-per-day",
        type=int,
        default=None,
        help=(
            "research only: cap each Shenwan level-2 industry in the selected "
            "right lane and refill from saved source ranks; requires strict "
            "Tushare sw_member intervals"
        ),
    )
    parser.add_argument(
        "--include-right-inverse-volatility-252",
        action="store_true",
        help=(
            "research only: attach fixed cross-sectional inverse-volatility "
            "entry multipliers from 252 returns through each observation date"
        ),
    )
    parser.add_argument(
        "--include-right-rank-persistence-20",
        action="store_true",
        help=(
            "research only: rerank the current saved Top100 by the fixed "
            "trailing-20 mean source-rank quality before selecting Top50"
        ),
    )
    parser.add_argument(
        "--right-inverse-volatility-price-directory",
        default=str(DEFAULT_FACTOR_ROOT / "prepared_prices"),
        help="prepared point-in-time price frames used by inverse-volatility sizing",
    )
    parser.add_argument(
        "--left-extra-directory",
        default="",
        help=(
            "可选PE/PB左侧候选目录；默认关闭，输入必须通过严格财务、"
            "行业时点和完整交易日覆盖门禁"
        ),
    )
    parser.add_argument(
        "--allow-unsafe-financial",
        action="store_true",
        help="仅研究：允许非严格财务PIT输入，并在输出中保留research_only标记",
    )
    parser.add_argument(
        "--allow-unsafe-industry",
        action="store_true",
        help="仅研究：允许非严格行业PIT输入，并在输出中保留research_only标记",
    )
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--factor", default=HYBRID_FACTOR)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-21")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / "candidates" / args.factor
    build_hybrid_candidates(
        right_directory=args.right_directory,
        left_directory=args.left_directory,
        output_directory=output,
        start_date=args.start_date,
        end_date=args.end_date,
        left_extra_directory=(args.left_extra_directory or None),
        allow_unsafe_financial=args.allow_unsafe_financial,
        allow_unsafe_industry=args.allow_unsafe_industry,
        right_top_n=args.right_top_n,
        right_industry_max_per_day=args.right_industry_max_per_day,
        include_right_inverse_volatility_252=(
            args.include_right_inverse_volatility_252
        ),
        include_right_rank_persistence_20=(
            args.include_right_rank_persistence_20
        ),
        right_inverse_volatility_price_directory=(
            args.right_inverse_volatility_price_directory
        ),
        exclude_right_codes=tuple(
            value.strip()
            for value in str(args.exclude_right_codes).split(",")
            if value.strip()
        ),
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
