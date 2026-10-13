"""Build the pre-registered forward-only GP/A right-side candidate stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_chunked import (
    _apply_combination,
    _formula_snapshot_dates,
    _load_cached_panel,
    _merge_candidate_chunks,
    _one_network_price_only_low_volatility_spec,
    _year_periods,
)
from stock_research.research.quantsplaybook_compare import prepare_factor_candidates
from stock_research.research.quantsplaybook_liquidity_universe import (
    _rolling_average_amount,
    liquidity_factor_name,
)
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository
from stock_research.storage.tushare_repository import tushare_to_project_code


GROSS_PROFITABILITY_FACTOR = "gross_profitability_to_assets_forward_pit"
SOURCE_FACTOR = (
    "playbook_low_corr_network_cap1_price_only_residual_volatility_"
    "gross_profitability_forward_pit"
)
FINAL_FACTOR = liquidity_factor_name(30_000_000.0, source_factor=SOURCE_FACTOR)
LIQUIDITY_THRESHOLD_YUAN = 30_000_000.0
MARKET_TIMEZONE = "Asia/Shanghai"
FORWARD_RULE_FROZEN_AT = "2026-08-11T00:00:00+08:00"
FINANCIAL_SOURCE_DATASETS = {
    "eastmoney": ("eastmoney_income", "eastmoney_balancesheet"),
    "tushare": ("income", "balancesheet"),
}
FINANCIAL_INDUSTRY_KEYWORDS = (
    "银行", "非银金融", "证券", "保险", "多元金融", "金融",
)
DEFAULT_LOW_VOLATILITY_PANEL_ROOT = (
    PATHS.runtime_root / "backtests"
    / "one_network_price_only_low_residual_volatility_experiment"
    / "research_panels"
)
DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "forward"
    / "right_gross_profitability_to_assets"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize(MARKET_TIMEZONE).tz_convert("UTC")
    return result.tz_convert("UTC")


def _market_date(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()


def gross_profitability_factor_metadata() -> dict:
    return {
        "factor_family": "gross_profitability",
        "formula": (
            "(latest fully announced annual total_revenue - oper_cost) / "
            "same-period total_assets"
        ),
        "effective_time_rule": (
            "max(provider ann_date, provider f_ann_date when present, "
            "first local payload captured_at), assigned to Asia/Shanghai date"
        ),
        "market_timezone": MARKET_TIMEZONE,
        "annual_report_rule": "December-31 fiscal year; report_type=1",
        "revision_rule": (
            "each payload revision starts only at its own first captured_at; "
            "later captures never backfill an earlier observation date"
        ),
        "missing_data_rule": (
            "missing announcement, revenue, operating cost, assets, positive "
            "assets, or dated industry membership fails closed"
        ),
        "industry_rule": "financial industries fail closed",
        "rule_frozen_at": FORWARD_RULE_FROZEN_AT,
        "literature": {
            "title": "The Other Side of Value: The Gross Profitability Premium",
            "author": "Robert Novy-Marx",
            "doi": "10.1016/j.jfineco.2013.01.003",
        },
    }


def fixed_gross_profitability_combination_spec() -> dict:
    base = _one_network_price_only_low_volatility_spec()
    selected = [*base["selected_factors"], GROSS_PROFITABILITY_FACTOR]
    weight = 1.0 / len(selected)
    return {
        "method": (
            "fixed_equal_weight_seven_factor_low_residual_volatility_"
            "plus_forward_gross_profitability"
        ),
        "weights": {factor: weight for factor in selected},
        "selected_factors": selected,
        "require_complete_factors": True,
        "missing_factor_rule": (
            "all seven factors must be available before cross-sectional ranking"
        ),
        "future_return_used_for_candidate_selection": False,
        "uses_financial_data": True,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "financial_point_in_time_reason": (
            "append-only provider payload versions are effective no earlier "
            "than both ann_date and first local capture"
        ),
        "industry_point_in_time": True,
        "research_only": True,
        "research_only_reason": (
            "forward evidence collection; the factor is not approved for production"
        ),
        "gross_profitability": gross_profitability_factor_metadata(),
        "base_factor_spec": base,
    }


def _empty_statement_versions(metric_columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "code", "report_period", "effective_at", "captured_at",
        "payload_sha256", *metric_columns,
    ])


def _normalize_statement_versions(
    frame: pd.DataFrame,
    *,
    metric_columns: tuple[str, ...],
    rule_frozen_at: str = FORWARD_RULE_FROZEN_AT,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_statement_versions(metric_columns)
    required = {
        "ts_code", "ann_date", "end_date", "report_type", "_captured_at",
        "_payload_sha256", *metric_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"financial version fields are missing: {missing}")

    data = frame.copy()
    data["code"] = data["ts_code"].map(tushare_to_project_code)
    data["report_period"] = pd.to_datetime(
        data["end_date"], errors="coerce",
    ).dt.normalize()
    announced = pd.to_datetime(data["ann_date"], errors="coerce")
    announced_at = announced.dt.tz_localize(
        MARKET_TIMEZONE,
    ).dt.tz_convert("UTC")
    if "f_ann_date" in data:
        final_announced = pd.to_datetime(
            data["f_ann_date"], errors="coerce",
        ).dt.tz_localize(MARKET_TIMEZONE).dt.tz_convert("UTC")
        data["announced_at"] = pd.concat(
            [announced_at, final_announced], axis=1,
        ).max(axis=1)
    else:
        data["announced_at"] = announced_at
    data["captured_at"] = pd.to_datetime(
        data["_captured_at"], errors="coerce", utc=True,
    )
    data["effective_at"] = pd.concat(
        [data["announced_at"], data["captured_at"]], axis=1,
    ).max(axis=1)
    report_type = pd.to_numeric(data["report_type"], errors="coerce")
    annual = (
        data["report_period"].dt.month.eq(12)
        & data["report_period"].dt.day.eq(31)
    )
    data = data[
        data["code"].ne("")
        & data["announced_at"].notna()
        & data["captured_at"].ge(_utc_timestamp(rule_frozen_at))
        & data["effective_at"].notna()
        & report_type.eq(1)
        & annual
    ].copy()
    for column in metric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    update_flag = (
        data["update_flag"]
        if "update_flag" in data
        else pd.Series(0, index=data.index)
    )
    data["update_priority"] = pd.to_numeric(
        update_flag, errors="coerce",
    ).fillna(0)
    data["payload_sha256"] = data["_payload_sha256"].astype(str)
    columns = [
        "code", "report_period", "effective_at", "captured_at",
        "update_priority", "payload_sha256", *metric_columns,
    ]
    return data[columns].sort_values(
        [
            "code", "report_period", "effective_at", "captured_at",
            "update_priority", "payload_sha256",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _statement_pair_events(
    income: pd.DataFrame,
    balance: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "code", "report_period", "effective_at", "gpa",
        "income_payload_sha256", "balance_payload_sha256",
    ]
    if income.empty or balance.empty:
        return pd.DataFrame(columns=columns)

    income_groups = {
        key: group.to_dict("records")
        for key, group in income.groupby(["code", "report_period"], sort=False)
    }
    balance_groups = {
        key: group.to_dict("records")
        for key, group in balance.groupby(["code", "report_period"], sort=False)
    }
    rows = []
    for code, report_period in sorted(set(income_groups) | set(balance_groups)):
        income_rows = income_groups.get((code, report_period), [])
        balance_rows = balance_groups.get((code, report_period), [])
        event_times = sorted({
            row["effective_at"] for row in [*income_rows, *balance_rows]
        })
        income_position = balance_position = 0
        current_income = current_balance = None
        for effective_at in event_times:
            while (
                income_position < len(income_rows)
                and income_rows[income_position]["effective_at"] <= effective_at
            ):
                current_income = income_rows[income_position]
                income_position += 1
            while (
                balance_position < len(balance_rows)
                and balance_rows[balance_position]["effective_at"] <= effective_at
            ):
                current_balance = balance_rows[balance_position]
                balance_position += 1
            revenue = (
                current_income["total_revenue"]
                if current_income is not None else np.nan
            )
            operating_cost = (
                current_income["oper_cost"]
                if current_income is not None else np.nan
            )
            total_assets = (
                current_balance["total_assets"]
                if current_balance is not None else np.nan
            )
            valid = (
                current_income is not None
                and current_balance is not None
                and all(np.isfinite(value) for value in (
                    revenue, operating_cost, total_assets,
                ))
                and float(total_assets) > 0
            )
            gpa = (
                (float(revenue) - float(operating_cost)) / float(total_assets)
                if valid else np.nan
            )
            rows.append({
                "code": code,
                "report_period": report_period,
                "effective_at": effective_at,
                "gpa": gpa,
                "income_payload_sha256": (
                    current_income["payload_sha256"]
                    if current_income is not None else None
                ),
                "balance_payload_sha256": (
                    current_balance["payload_sha256"]
                    if current_balance is not None else None
                ),
            })
    return pd.DataFrame(rows, columns=columns)


def _latest_annual_report_events(statement_events: pd.DataFrame) -> pd.DataFrame:
    columns = ["code", "effective_date", "report_period", "gpa"]
    if statement_events.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for code, code_events in statement_events.groupby("code", sort=False):
        state: dict[pd.Timestamp, dict] = {}
        ordered = code_events.sort_values(
            ["effective_at", "report_period"], kind="mergesort",
        )
        for effective_at, changes in ordered.groupby("effective_at", sort=True):
            for change in changes.to_dict("records"):
                state[pd.Timestamp(change["report_period"])] = change
            report_period = max(state)
            selected = state[report_period]
            rows.append({
                "code": code,
                "effective_date": _market_date(effective_at),
                "report_period": report_period,
                "gpa": selected["gpa"],
            })
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["code", "effective_date", "report_period"], kind="mergesort")
        .drop_duplicates(["code", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def _align_event_values(
    index: pd.MultiIndex,
    events: pd.DataFrame,
) -> pd.Series:
    result = pd.Series(np.nan, index=index, dtype=float, name=GROSS_PROFITABILITY_FACTOR)
    if events.empty:
        return result
    work = index.to_frame(index=False)
    work["_position"] = np.arange(len(work))
    grouped_events = {
        str(code): group.sort_values("effective_date", kind="mergesort")
        for code, group in events.groupby("code", sort=False)
    }
    values = np.full(len(work), np.nan, dtype=float)
    for code, observations in work.groupby("code", sort=False):
        available = grouped_events.get(str(code))
        if available is None or available.empty:
            continue
        event_dates = available["effective_date"].to_numpy(dtype="datetime64[ns]")
        observation_dates = pd.to_datetime(
            observations["date"], errors="coerce",
        ).to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(event_dates, observation_dates, side="right") - 1
        eligible = positions >= 0
        if not eligible.any():
            continue
        target = observations["_position"].to_numpy()[eligible]
        source = positions[eligible]
        values[target] = pd.to_numeric(
            available["gpa"], errors="coerce",
        ).to_numpy()[source]
    result.iloc[:] = values
    return result


def _normalize_industry_history(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "code", "membership_identity", "in_date", "out_date",
        "captured_at", "captured_date", "industry_text",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    if "ts_code" not in frame:
        raise RuntimeError("industry observations are missing ts_code")
    if "in_date" not in frame:
        raise RuntimeError("industry observations are missing in_date")
    captured_column = (
        "_captured_at" if "_captured_at" in frame else "_updated_at"
    )
    if captured_column not in frame:
        raise RuntimeError("industry observations are missing local capture time")
    data = frame.copy()
    data["code"] = data["ts_code"].map(tushare_to_project_code)
    data["in_date"] = pd.to_datetime(
        data["in_date"], errors="coerce",
    ).dt.normalize()
    out_date = (
        data["out_date"]
        if "out_date" in data
        else pd.Series(pd.NaT, index=data.index)
    )
    data["out_date"] = pd.to_datetime(
        out_date, errors="coerce",
    ).dt.normalize()
    data["captured_date"] = pd.to_datetime(
        data[captured_column], errors="coerce", utc=True,
    ).dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).dt.normalize()
    data["captured_at"] = pd.to_datetime(
        data[captured_column], errors="coerce", utc=True,
    )
    labels = []
    for column in ("l1_name", "l2_name", "l3_name"):
        if column in data:
            labels.append(data[column].fillna("").astype(str).str.strip())
    data["industry_text"] = (
        pd.concat(labels, axis=1).agg("|".join, axis=1)
        if labels else ""
    )
    identity_columns = [
        column for column in (
            "ts_code", "in_date", "l1_code", "l2_code", "l3_code",
        ) if column in data
    ]
    data["membership_identity"] = data[identity_columns].fillna("").astype(
        str,
    ).agg("|".join, axis=1)
    return data[
        data["code"].ne("")
        & data["in_date"].notna()
        & data["captured_date"].notna()
    ][columns].sort_values(
        ["code", "membership_identity", "captured_at"], kind="mergesort",
    ).reset_index(drop=True)


def _industry_eligibility(
    index: pd.MultiIndex,
    industry_history: pd.DataFrame,
) -> pd.Series:
    normalized = _normalize_industry_history(industry_history)
    result = pd.Series(False, index=index, dtype=bool, name="industry_eligible")
    if normalized.empty:
        return result
    work = index.to_frame(index=False)
    work["_position"] = np.arange(len(work))
    groups = {
        str(code): group.to_dict("records")
        for code, group in normalized.groupby("code", sort=False)
    }
    values = np.zeros(len(work), dtype=bool)
    for code, observations in work.groupby("code", sort=False):
        membership_versions = groups.get(str(code), [])
        if not membership_versions:
            continue
        for row in observations.to_dict("records"):
            date = pd.Timestamp(row["date"]).normalize()
            latest_by_identity = {}
            for membership in membership_versions:
                if membership["captured_date"] > date:
                    continue
                identity = membership["membership_identity"]
                latest = latest_by_identity.get(identity, [])
                if (
                    not latest
                    or membership["captured_at"] > latest[0]["captured_at"]
                ):
                    latest_by_identity[identity] = [membership]
                elif membership["captured_at"] == latest[0]["captured_at"]:
                    latest.append(membership)
            active = []
            for versions in latest_by_identity.values():
                # Same-capture conflicts have no reliable provider ordering.
                if all(
                    membership["in_date"] <= date
                    and (
                        pd.isna(membership["out_date"])
                        or membership["out_date"] >= date
                    )
                    for membership in versions
                ):
                    active.extend(versions)
            labels = [
                str(membership["industry_text"]).strip("|")
                for membership in active
                if str(membership["industry_text"]).strip("|")
            ]
            if not labels:
                continue
            is_financial = any(
                keyword in label
                for label in labels
                for keyword in FINANCIAL_INDUSTRY_KEYWORDS
            )
            values[int(row["_position"])] = not is_financial
    result.iloc[:] = values
    return result


def calculate_forward_gross_profitability_factor(
    index: pd.MultiIndex,
    income_versions: pd.DataFrame,
    balance_versions: pd.DataFrame,
    industry_history: pd.DataFrame,
    *,
    rule_frozen_at: str = FORWARD_RULE_FROZEN_AT,
) -> pd.Series:
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("gross profitability factor requires a MultiIndex")
    if list(index.names) != ["date", "code"]:
        raise ValueError("gross profitability factor index must be date/code")
    if index.has_duplicates:
        raise ValueError("gross profitability factor index contains duplicates")
    income = _normalize_statement_versions(
        income_versions,
        metric_columns=("total_revenue", "oper_cost"),
        rule_frozen_at=rule_frozen_at,
    )
    balance = _normalize_statement_versions(
        balance_versions,
        metric_columns=("total_assets",),
        rule_frozen_at=rule_frozen_at,
    )
    statement_events = _statement_pair_events(income, balance)
    annual_events = _latest_annual_report_events(statement_events)
    factor = _align_event_values(index, annual_events)
    industry_eligible = _industry_eligibility(index, industry_history)
    observation_dates = pd.to_datetime(
        index.get_level_values("date"), errors="coerce",
    ).tz_localize(MARKET_TIMEZONE).tz_convert("UTC")
    after_freeze = observation_dates >= _utc_timestamp(rule_frozen_at)
    return factor.where(industry_eligible & after_freeze)


def add_gross_profitability_combination(
    low_volatility_panel: pd.DataFrame,
    income_versions: pd.DataFrame,
    balance_versions: pd.DataFrame,
    industry_history: pd.DataFrame,
    average_amount_yuan: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    if not isinstance(low_volatility_panel.index, pd.MultiIndex):
        raise ValueError("low-volatility panel requires a MultiIndex")
    spec = fixed_gross_profitability_combination_spec()
    missing = sorted(
        set(spec["selected_factors"])
        - {GROSS_PROFITABILITY_FACTOR}
        - set(low_volatility_panel.columns)
    )
    if missing:
        raise ValueError(f"low-volatility source factors are missing: {missing}")
    panel = low_volatility_panel.copy()
    panel[GROSS_PROFITABILITY_FACTOR] = (
        calculate_forward_gross_profitability_factor(
            panel.index,
            income_versions,
            balance_versions,
            industry_history,
        )
    )
    selected = list(spec["selected_factors"])
    average_amount = pd.to_numeric(
        average_amount_yuan.reindex(panel.index), errors="coerce",
    )
    complete = panel[selected].notna().all(axis=1)
    liquid = average_amount.ge(LIQUIDITY_THRESHOLD_YUAN)
    ranking_universe = complete & liquid
    ranking_panel = panel.copy()
    ranking_panel[selected] = ranking_panel[selected].where(
        ranking_universe,
    )
    panel[SOURCE_FACTOR] = _apply_combination(ranking_panel, spec)
    panel[FINAL_FACTOR] = panel[SOURCE_FACTOR]
    return panel, {
        "panel_rows": int(len(panel)),
        "income_version_rows": int(len(income_versions)),
        "balance_version_rows": int(len(balance_versions)),
        "industry_observation_rows": int(len(industry_history)),
        "gross_profitability_available_rows": int(
            panel[GROSS_PROFITABILITY_FACTOR].notna().sum()
        ),
        "complete_seven_factor_rows": int(complete.sum()),
        "liquid_complete_seven_factor_rows": int(ranking_universe.sum()),
        "combination_available_rows": int(panel[SOURCE_FACTOR].notna().sum()),
    }


def _load_average_amount(
    database: Database,
    *,
    start_date: str,
    end_date: str,
) -> pd.Series:
    connection = database.connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                ts_code,
                trade_date,
                json_extract(payload_json, '$.amount')
            FROM raw.tushare_dataset_rows
                 INDEXED BY idx_tushare_dataset_trade_date
            WHERE dataset = 'daily_kline'
              AND trade_date >= ?
              AND trade_date <= ?
            ORDER BY trade_date, ts_code, row_key
            """,
            [start_date, end_date],
        ).fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame(rows, columns=["ts_code", "date", "amount"])
    if frame.empty:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "code"]),
            name="average_amount_20_yuan",
        )
    frame["code"] = frame["ts_code"].map(tushare_to_project_code)
    frame = frame.drop_duplicates(["date", "code"], keep="last")
    return _rolling_average_amount(frame)


def _known_through(frame: pd.DataFrame, column: str, end_date: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame.copy()
    captured = pd.to_datetime(frame[column], errors="coerce", utc=True)
    cutoff = _utc_timestamp(pd.Timestamp(end_date) + pd.Timedelta(days=1))
    return frame[captured.lt(cutoff)].copy()


def _write_industry_audit(
    history: pd.DataFrame,
    *,
    output_root: Path,
    universe_codes: set[str],
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = output_root / "industry_source_tushare_sw_member.csv"
    history.to_csv(source_path, index=False, encoding="utf-8-sig")
    normalized = _normalize_industry_history(history)
    mapped_codes = set(normalized["code"].astype(str))
    captured_column = (
        "_captured_at" if "_captured_at" in history else "_updated_at"
    )
    captured = pd.to_datetime(
        history[captured_column], errors="coerce", utc=True,
    )
    as_of = captured.max()
    coverage = (
        len(mapped_codes & universe_codes) / len(universe_codes)
        if universe_codes else 0.0
    )
    return {
        "industry_point_in_time": True,
        "industry_point_in_time_status": "safe",
        "industry_data_source": "tushare/pro:sw_member interval snapshot",
        "industry_source_path": str(source_path.resolve()),
        "industry_source_sha256": _sha256(source_path),
        "industry_source_modified_at": pd.Timestamp(
            source_path.stat().st_mtime, unit="s", tz="UTC",
        ).isoformat(),
        "industry_as_of_date": (
            _market_date(as_of).strftime("%Y-%m-%d")
            if pd.notna(as_of) else None
        ),
        "industry_mapping_count": len(mapped_codes),
        "industry_universe_count": len(universe_codes),
        "industry_coverage_ratio": coverage,
    }


def _current_listed_codes(repository: TushareRepository, end_date: str) -> set[str]:
    observations = _known_through(
        repository.load_dataset_current_observations("stock_basic"),
        "_updated_at",
        end_date,
    )
    if observations.empty or "ts_code" not in observations:
        return set()
    if "list_status" in observations:
        observations = observations[
            observations["list_status"].fillna("").astype(str).eq("L")
        ]
    return {
        tushare_to_project_code(code)
        for code in observations["ts_code"].dropna()
        if tushare_to_project_code(code)
    }


def build_candidates(
    *,
    low_volatility_panel_root: str | Path,
    formula_history: str | Path,
    database_path: str | Path,
    output_root: str | Path,
    start_date: str,
    end_date: str,
    top_n: int = 100,
    financial_source: str = "eastmoney",
) -> Path:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    freeze_date = _market_date(FORWARD_RULE_FROZEN_AT)
    if start < freeze_date:
        raise RuntimeError(
            "GP/A is forward-only; start_date cannot precede 2026-08-11"
        )
    if end < start:
        raise ValueError("end_date must not precede start_date")
    today = pd.Timestamp.now(tz=MARKET_TIMEZONE).tz_localize(None).normalize()
    if end > today:
        raise RuntimeError("forward GP/A candidates cannot be generated for a future date")
    if int(top_n) != 100:
        raise ValueError("the pre-registered forward GP/A stream requires top_n=100")
    if financial_source not in FINANCIAL_SOURCE_DATASETS:
        raise ValueError(f"unsupported financial source: {financial_source}")

    panel_root = Path(low_volatility_panel_root)
    formula_path = Path(formula_history)
    database_path = Path(database_path)
    output = Path(output_root)
    if not formula_path.is_file():
        raise FileNotFoundError(f"Formula33 history does not exist: {formula_path}")
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {database_path}")
    periods = _year_periods(start_date, end_date)
    database = Database(database_path)
    repository = TushareRepository(database)
    captured_through = (
        _utc_timestamp(end + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1)
    ).isoformat()
    income_dataset, balance_dataset = FINANCIAL_SOURCE_DATASETS[financial_source]
    income_versions = repository.load_dataset_versions(
        income_dataset,
        captured_from=FORWARD_RULE_FROZEN_AT,
        captured_through=captured_through,
    )
    balance_versions = repository.load_dataset_versions(
        balance_dataset,
        captured_from=FORWARD_RULE_FROZEN_AT,
        captured_through=captured_through,
    )
    if income_versions.empty or balance_versions.empty:
        raise RuntimeError(
            "post-freeze financial payload versions are required: "
            f"source={financial_source} datasets={income_dataset},{balance_dataset}; "
            "historical current-value rows are intentionally not backfilled"
        )
    industry_history = _known_through(
        repository.load_dataset_versions("sw_member"),
        "_captured_at",
        end_date,
    )
    if industry_history.empty:
        raise RuntimeError(
            "an append-only versioned sw_member snapshot is required"
        )
    universe_codes = _current_listed_codes(repository, end_date)
    if not universe_codes:
        raise RuntimeError("a locally timestamped listed-stock universe is required")
    industry_metadata = _write_industry_audit(
        industry_history,
        output_root=output,
        universe_codes=universe_codes,
    )

    combination_spec = fixed_gross_profitability_combination_spec()
    combination_spec.update(industry_metadata)
    captured = pd.concat([
        pd.to_datetime(income_versions["_captured_at"], errors="coerce", utc=True),
        pd.to_datetime(balance_versions["_captured_at"], errors="coerce", utc=True),
    ], ignore_index=True).dropna()
    financial_metadata = {
        "financial_data_source": financial_source,
        "financial_income_dataset": income_dataset,
        "financial_balance_dataset": balance_dataset,
        "financial_income_version_rows": int(len(income_versions)),
        "financial_balance_version_rows": int(len(balance_versions)),
        "financial_first_captured_at_utc": captured.min().isoformat(),
        "financial_last_captured_at_utc": captured.max().isoformat(),
    }
    combination_spec.update(financial_metadata)
    final_spec = {
        **combination_spec,
        "method": "PIT_liquidity_prefilter_then_fixed_seven_factor_rank",
        "source_factor": SOURCE_FACTOR,
        "minimum_average_amount_yuan": LIQUIDITY_THRESHOLD_YUAN,
        "liquidity_window_trading_days": 20,
        "liquidity_minimum_observations": 20,
        "saved_top_n": 100,
        "trading_top_n": 50,
    }
    source_panels = []
    coverage_rows = []
    for year, period_start, period_end in periods:
        panel_path = panel_root / f"factor_panel_{year}.pkl"
        if not panel_path.is_file():
            raise RuntimeError(f"cached low-volatility panel is missing for {year}")
        print(
            f"[gross-profitability-forward] year={year} "
            f"period={period_start}..{period_end}",
            flush=True,
        )
        base_panel = _load_cached_panel(panel_path)
        history_start = (
            pd.Timestamp(period_start) - pd.Timedelta(days=60)
        ).strftime("%Y-%m-%d")
        average_amount = _load_average_amount(
            database,
            start_date=history_start,
            end_date=period_end,
        )
        panel, coverage = add_gross_profitability_combination(
            base_panel,
            income_versions,
            balance_versions,
            industry_history,
            average_amount,
        )
        snapshot_dates = _formula_snapshot_dates(
            formula_path,
            start_date=period_start,
            end_date=period_end,
        )
        if not snapshot_dates:
            raise RuntimeError(f"no Formula33 snapshot dates for year {year}")
        daily_counts = panel[FINAL_FACTOR].notna().groupby(
            level="date", sort=False,
        ).sum().reindex(snapshot_dates, fill_value=0)
        insufficient = daily_counts[daily_counts.lt(int(top_n))]
        if not insufficient.empty:
            sample = ", ".join(
                f"{date:%Y-%m-%d}={int(count)}"
                for date, count in insufficient.head(5).items()
            )
            raise RuntimeError(
                "forward GP/A complete seven-factor coverage is below Top100; "
                f"sample={sample}"
            )
        annual_coverage = {
            "year": int(year),
            **coverage,
            "factor": FINAL_FACTOR,
            "minimum_average_amount_yuan": LIQUIDITY_THRESHOLD_YUAN,
            "minimum_complete_candidates_on_snapshot_dates": int(
                daily_counts.min()
            ),
        }
        coverage_rows.append(annual_coverage)
        prepare_factor_candidates(
            panel,
            panel.index.to_frame(index=False),
            factors=[FINAL_FACTOR],
            snapshot_dates=snapshot_dates,
            output_root=output / "chunks" / str(year),
            top_n=int(top_n),
            start_date=period_start,
            end_date=period_end,
            factor_metadata_by_name={FINAL_FACTOR: final_spec},
            trading_top_n=50,
        )
        source_panels.append({
            "year": int(year),
            "low_volatility_panel": str(panel_path),
            "low_volatility_panel_sha256": _sha256(panel_path),
            **annual_coverage,
        })
        del base_panel, panel, average_amount
        gc.collect()

    manifest = {
        "version": 1,
        "selection_rule": "single_pre_registered_forward_gross_profitability",
        "rule_frozen_at": FORWARD_RULE_FROZEN_AT,
        "future_return_used_for_candidate_selection": False,
        "full_history_previously_observed": False,
        "historical_backfill_allowed": False,
        "research_only": True,
        "start_date": start_date,
        "end_date": end_date,
        "top_n": int(top_n),
        "formula_history": str(formula_path),
        "formula_history_sha256": _sha256(formula_path),
        "database_path": str(database_path),
        "database_size_bytes": database_path.stat().st_size,
        **financial_metadata,
        "combinations": {FINAL_FACTOR: final_spec},
        "source_panels": source_panels,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "combination_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(coverage_rows).to_csv(
        output / "forward_gross_profitability_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_index = _merge_candidate_chunks(
        periods=periods,
        output_root=output,
        start_date=start_date,
        end_date=end_date,
        combination_manifest=manifest,
        factors_to_merge=[FINAL_FACTOR],
    )
    candidate_index.to_csv(
        output / "candidate_streams.csv", index=False, encoding="utf-8-sig",
    )
    return output / "candidates" / FINAL_FACTOR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low-volatility-panel-root",
        default=str(DEFAULT_LOW_VOLATILITY_PANEL_ROOT),
    )
    parser.add_argument("--formula-history", required=True)
    parser.add_argument("--database-path", default=str(PATHS.database))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--financial-source",
        choices=sorted(FINANCIAL_SOURCE_DATASETS),
        default="eastmoney",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = build_candidates(
        low_volatility_panel_root=args.low_volatility_panel_root,
        formula_history=args.formula_history,
        database_path=args.database_path,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=100,
        financial_source=args.financial_source,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
