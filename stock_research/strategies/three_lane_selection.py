"""Point-in-time three-lane stock selection research model."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


THREE_LANE_MODEL = "fundamental_smooth_high_stage2_vcp"
LANE_ORDER = ("fundamental_momentum", "smooth_52week_high", "stage2_vcp")
FUNDAMENTAL_WEIGHTS = {
    "profit_yoy": 0.22,
    "profit_yoy_acceleration": 0.18,
    "revenue_yoy": 0.16,
    "roe": 0.14,
    "roe_change": 0.08,
    "cashflow_quality": 0.12,
    "equity_ratio": 0.05,
    "leverage_improvement": 0.05,
}


def _number(value):
    converted = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(converted) else float(converted)


def _metric(row: dict | None, key: str):
    return None if row is None else _number(row.get(key))


def _ratio(numerator, denominator):
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _normalize_statement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "report_period", "ann_date"])
    work = frame.copy()
    work["code"] = work["code"].astype(str)
    work["report_period"] = pd.to_datetime(work["report_period"], errors="coerce")
    work["ann_date"] = pd.to_datetime(work["ann_date"], errors="coerce")
    work = work.dropna(subset=["code", "report_period", "ann_date"])
    return (
        work.sort_values(["code", "ann_date", "report_period"], kind="mergesort")
        .drop_duplicates(["code", "report_period", "ann_date"], keep="last")
        .reset_index(drop=True)
    )


def build_fundamental_events(statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build announcement-visible fundamental observations.

    Every statement version enters state on its own announcement date. A later
    revision changes only that date and later observations.
    """
    normalized = {
        name: _normalize_statement_frame(statements.get(name, pd.DataFrame()))
        for name in ("fina_indicator", "income", "balancesheet", "cashflow")
    }
    codes = sorted(set().union(*(
        set(frame["code"].unique()) for frame in normalized.values()
    )))
    grouped = {
        dataset: {
            str(code): group.to_dict("records")
            for code, group in frame.groupby("code", sort=False)
        }
        for dataset, frame in normalized.items()
    }
    output = []
    for code in codes:
        dated_updates: dict[pd.Timestamp, list[tuple[str, pd.Timestamp, dict]]] = (
            defaultdict(list)
        )
        for dataset in normalized:
            for row in grouped[dataset].get(code, []):
                dated_updates[pd.Timestamp(row["ann_date"]).normalize()].append((
                    dataset,
                    pd.Timestamp(row["report_period"]).normalize(),
                    row,
                ))
        state: dict[str, dict[pd.Timestamp, dict]] = {
            name: {} for name in normalized
        }
        for ann_date in sorted(dated_updates):
            for dataset, period, row in dated_updates[ann_date]:
                state[dataset][period] = row
            if not state["fina_indicator"]:
                continue
            period = max(state["fina_indicator"])
            periods = sorted(state["fina_indicator"])
            previous_period = max((item for item in periods if item < period), default=None)
            prior_year_period = period - pd.DateOffset(years=1)
            fina = state["fina_indicator"].get(period)
            prior_fina = state["fina_indicator"].get(previous_period)
            income = state["income"].get(period)
            prior_year_income = state["income"].get(prior_year_period)
            balance = state["balancesheet"].get(period)
            prior_balance = state["balancesheet"].get(previous_period)
            cashflow = state["cashflow"].get(period)

            profit_yoy = _metric(fina, "dt_netprofit_yoy")
            if profit_yoy is None:
                profit_yoy = _metric(fina, "netprofit_yoy")
            previous_profit_yoy = _metric(prior_fina, "dt_netprofit_yoy")
            if previous_profit_yoy is None:
                previous_profit_yoy = _metric(prior_fina, "netprofit_yoy")
            roe = _metric(fina, "roe_dt")
            if roe is None:
                roe = _metric(fina, "roe")
            previous_roe = _metric(prior_fina, "roe_dt")
            if previous_roe is None:
                previous_roe = _metric(prior_fina, "roe")

            revenue_yoy = None
            current_revenue = _metric(income, "total_revenue")
            prior_revenue = _metric(prior_year_income, "total_revenue")
            if current_revenue is not None and prior_revenue not in (None, 0):
                revenue_yoy = (current_revenue / abs(prior_revenue) - 1.0) * 100.0
            net_income = _metric(income, "n_income_attr_p")
            operating_cashflow = _metric(cashflow, "n_cashflow_act")
            cashflow_quality = _ratio(operating_cashflow, abs(net_income or 0.0))

            equity_ratio = _ratio(
                _metric(balance, "total_hldr_eqy_exc_min_int"),
                _metric(balance, "total_assets"),
            )
            leverage = _ratio(
                _metric(balance, "total_liab"),
                _metric(balance, "total_assets"),
            )
            previous_leverage = _ratio(
                _metric(prior_balance, "total_liab"),
                _metric(prior_balance, "total_assets"),
            )
            component_count = sum(
                state[name].get(period) is not None for name in state
            )
            output.append({
                "effective_date": ann_date,
                "code": code,
                "report_period": period,
                "profit_yoy": profit_yoy,
                "profit_yoy_acceleration": (
                    None if profit_yoy is None or previous_profit_yoy is None
                    else profit_yoy - previous_profit_yoy
                ),
                "revenue_yoy": revenue_yoy,
                "roe": roe,
                "roe_change": (
                    None if roe is None or previous_roe is None else roe - previous_roe
                ),
                "cashflow_quality": cashflow_quality,
                "equity_ratio": equity_ratio,
                "leverage_improvement": (
                    None if leverage is None or previous_leverage is None
                    else previous_leverage - leverage
                ),
                "financial_component_count": component_count,
            })
    if not output:
        return pd.DataFrame()
    return (
        pd.DataFrame(output)
        .sort_values(["code", "effective_date", "report_period"], kind="mergesort")
        .drop_duplicates(["code", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def map_fundamental_lane(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    maximum_report_age_days: int = 460,
) -> pd.DataFrame:
    """Map the latest announced version to each eligible observation date."""
    if events.empty or membership.empty:
        return pd.DataFrame(columns=["date", "code", "fundamental_momentum"])
    observations = membership[["date", "code"]].copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations["code"] = observations["code"].astype(str)
    event_groups = {
        str(code): group.sort_values("effective_date").reset_index(drop=True)
        for code, group in events.groupby("code", sort=False)
    }
    mapped = []
    for code, group in observations.groupby("code", sort=False):
        available = event_groups.get(str(code))
        if available is None or available.empty:
            continue
        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        event_dates = available["effective_date"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(event_dates, dates, side="right") - 1
        valid = positions >= 0
        if not valid.any():
            continue
        chosen = available.iloc[positions[valid]].copy().reset_index(drop=True)
        chosen["date"] = dates[valid]
        mapped.append(chosen)
    if not mapped:
        return pd.DataFrame(columns=["date", "code", "fundamental_momentum"])
    result = pd.concat(mapped, ignore_index=True)
    age = (result["date"] - result["report_period"]).dt.days
    eligible = (
        age.between(0, int(maximum_report_age_days))
        & pd.to_numeric(result["profit_yoy"], errors="coerce").gt(-20.0)
    )
    features = list(FUNDAMENTAL_WEIGHTS)
    ranked = result[features].apply(pd.to_numeric, errors="coerce").groupby(
        result["date"], sort=False,
    ).rank(method="average", pct=True)
    weights = pd.Series(FUNDAMENTAL_WEIGHTS)
    available_weight = ranked.notna().mul(weights, axis=1).sum(axis=1)
    result["fundamental_momentum"] = (
        ranked.mul(weights, axis=1).sum(axis=1, min_count=1)
        .div(available_weight.where(available_weight > 0))
        .where(eligible)
    )
    result["financial_report_age_days"] = age
    return result[[
        "date", "code", "report_period", "effective_date",
        "financial_component_count", "financial_report_age_days",
        *features, "fundamental_momentum",
    ]]


def _cross_section_score(parts: dict[str, tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    weighted = None
    available = None
    for _, (values, weight) in parts.items():
        ranks = values.rank(axis=1, method="average", pct=True)
        contribution = ranks * float(weight)
        present = ranks.notna().astype(float) * float(weight)
        weighted = contribution if weighted is None else weighted.add(contribution, fill_value=0)
        available = present if available is None else available.add(present, fill_value=0)
    return weighted.div(available.where(available > 0))


def calculate_technical_lanes(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate smooth-high and Stage 2/VCP scores from current-or-prior bars."""
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["code"] = work["code"].astype(str)
    for column in ("close", "high", "low", "volume", "turnover_rate", "raw_to_qfq_factor"):
        if column not in work:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    factor = work["raw_to_qfq_factor"].where(work["raw_to_qfq_factor"].gt(0), 1.0)
    for column in ("close", "high", "low"):
        work[f"qfq_{column}"] = work[column] / factor
    work = work.dropna(subset=["date", "code", "qfq_close"])
    close = work.pivot(index="date", columns="code", values="qfq_close").sort_index()
    high = work.pivot(index="date", columns="code", values="qfq_high").reindex_like(close)
    low = work.pivot(index="date", columns="code", values="qfq_low").reindex_like(close)
    volume = work.pivot(index="date", columns="code", values="volume").reindex_like(close)
    turnover = work.pivot(index="date", columns="code", values="turnover_rate").reindex_like(close)
    returns = close.pct_change(fill_method=None)

    high252 = close.rolling(252, min_periods=180).max()
    low252 = close.rolling(252, min_periods=180).min()
    proximity = close.div(high252)
    ret252 = close.div(close.shift(252)) - 1.0
    ret126 = close.div(close.shift(126)) - 1.0
    ret63 = close.div(close.shift(63)) - 1.0
    ret20 = close.div(close.shift(20)) - 1.0
    positive_sum = returns.clip(lower=0).rolling(126, min_periods=80).sum()
    largest_day_share = returns.clip(lower=0).rolling(126, min_periods=80).max().div(
        positive_sum.where(positive_sum > 0)
    )
    positive_day_ratio = returns.gt(0).rolling(126, min_periods=80).mean()
    turnover_mean = turnover.rolling(60, min_periods=40).mean()
    turnover_instability = turnover.rolling(60, min_periods=40).std().div(
        turnover_mean.where(turnover_mean > 0)
    )
    volume_spike = volume.rolling(20, min_periods=15).max().div(
        volume.rolling(20, min_periods=15).median().where(lambda value: value > 0)
    )
    late_acceleration = (ret20 - ret126 * (20.0 / 126.0)).clip(lower=0)
    smooth_score = _cross_section_score({
        "proximity": (proximity, 0.20),
        "ret252": (ret252, 0.18),
        "ret126": (ret126, 0.15),
        "ret63": (ret63, 0.10),
        "positive_day_ratio": (positive_day_ratio, 0.12),
        "turnover_level_penalty": (-turnover_mean, 0.05),
        "turnover_instability_penalty": (-turnover_instability, 0.07),
        "volume_spike_penalty": (-volume_spike, 0.05),
        "return_concentration_penalty": (-largest_day_share, 0.05),
        "late_acceleration_penalty": (-late_acceleration, 0.03),
    }).where(proximity.ge(0.75) & ret126.gt(0))

    ma50 = close.rolling(50, min_periods=50).mean()
    ma150 = close.rolling(150, min_periods=150).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    previous_close = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - previous_close).abs().to_numpy(),
            (low - previous_close).abs().to_numpy(),
        ]),
        index=close.index,
        columns=close.columns,
    )
    atr20 = true_range.rolling(20, min_periods=15).mean().div(close)
    atr60 = true_range.rolling(60, min_periods=40).mean().div(close)
    range20 = high.rolling(20, min_periods=15).max().div(
        low.rolling(20, min_periods=15).min()
    ) - 1.0
    range60 = high.rolling(60, min_periods=40).max().div(
        low.rolling(60, min_periods=40).min()
    ) - 1.0
    volume_contraction = volume.rolling(20, min_periods=15).mean().div(
        volume.rolling(50, min_periods=35).mean()
    )
    stage2_gate = (
        close.gt(ma50)
        & ma50.gt(ma150)
        & ma150.gt(ma200)
        & ma200.gt(ma200.shift(20))
        & close.ge(low252 * 1.30)
        & proximity.ge(0.75)
    )
    stage2_score = _cross_section_score({
        "high_proximity": (proximity, 0.18),
        "ma50_slope": (ma50.div(ma50.shift(20)) - 1.0, 0.16),
        "ma200_slope": (ma200.div(ma200.shift(20)) - 1.0, 0.12),
        "atr_contraction": (-atr20.div(atr60), 0.18),
        "range_contraction": (-range20.div(range60), 0.18),
        "volume_contraction": (-volume_contraction, 0.10),
        "turnover_instability_penalty": (-turnover_instability, 0.05),
        "volume_spike_penalty": (-volume_spike, 0.03),
    }).where(stage2_gate)

    output = pd.concat({
        "smooth_52week_high": smooth_score.stack(),
        "stage2_vcp": stage2_score.stack(),
    }, axis=1)
    output.index.names = ["date", "code"]
    return output.reset_index()


def quota_union_candidates(
    lane_scores: pd.DataFrame,
    *,
    lane_top_n: int = 20,
    maximum_candidates: int = 50,
) -> list[dict]:
    """Round-robin lane leaders so one style cannot consume all rights."""
    queues: dict[str, list[tuple[str, float]]] = {}
    for lane in LANE_ORDER:
        values = lane_scores[["code", lane]].dropna().copy()
        values["code"] = values["code"].astype(str)
        values = values.sort_values([lane, "code"], ascending=[False, True], kind="mergesort")
        queues[lane] = list(values.head(int(lane_top_n)).itertuples(index=False, name=None))
    selected: dict[str, dict] = {}
    cursors = {lane: 0 for lane in LANE_ORDER}
    while len(selected) < int(maximum_candidates):
        advanced = False
        for lane in LANE_ORDER:
            queue = queues[lane]
            while cursors[lane] < len(queue):
                code, score = queue[cursors[lane]]
                cursors[lane] += 1
                advanced = True
                row = selected.setdefault(str(code), {
                    "code": str(code),
                    "selection_lanes": [],
                    "lane_scores": {},
                    "lane_ranks": {},
                })
                rank = cursors[lane]
                row["selection_lanes"].append(lane)
                row["lane_scores"][lane] = float(score)
                row["lane_ranks"][lane] = int(rank)
                break
            if len(selected) >= int(maximum_candidates):
                break
        if not advanced or all(cursors[lane] >= len(queues[lane]) for lane in LANE_ORDER):
            break
    rows = []
    for row in selected.values():
        best_score = max(row["lane_scores"].values())
        rows.append({
            "code": row["code"],
            "name": row["code"],
            "strategy_part": f"three_lane:{'+'.join(row['selection_lanes'])}",
            "selection_engine": "three_lane_factor_only",
            "selection_profile": "quantsplaybook_factor_only",
            "playbook_factor": THREE_LANE_MODEL,
            "playbook_factor_score": best_score,
            "playbook_factor_rank": min(row["lane_ranks"].values()),
            "candidate_score": round(best_score * 100.0, 6),
            "three_lane_membership": "+".join(row["selection_lanes"]),
            "three_lane_scores": ";".join(
                f"{lane}={row['lane_scores'][lane]:.8f}"
                for lane in row["selection_lanes"]
            ),
            "selected_for_trading": True,
            "signal_eligible": True,
            "allow_right": True,
            "allow_left": False,
            "candidate_failure_reason": "",
        })
    return rows
