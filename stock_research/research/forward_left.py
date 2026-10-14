"""Capture a real same-day value-model input, independently of report pools.

Announcement visibility and current industry membership are recorded as seen
now. They do not certify historical revisions or historical industry snapshots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from apps.tushare_financial_cache import _metrics_from_rows, _read_dataset
from stock_research.api import tushare
from stock_research.core.financial_period import latest_visible_report_period
from stock_research.data.forward_sync import DIVIDEND_FIELDS, verify_completed_receipts
from stock_research.data.zer0share_manifest import ZER0SHARE_DATASET_MAP
from stock_research.research.frozen_forward import (
    capture_clock, implementation_fingerprint, load_left_snapshot, sha256, validate_sync_receipt, write_json,
)
from stock_research.storage import Database, TushareRepository
from stock_research.strategies.candidate_interface import normalize_candidate
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION, is_value_industry_allowed, value_industry_allowlist_match,
)
from stock_research.strategies.historical_candidates import (
    _candidate_feature_frame, _leadership_from_feature_row, _trade_basis_from_feature_row,
)


def visible_financial_rows(frame, observation_date):
    """Both original and final announcement dates must already be visible."""
    required = {"ts_code", "end_date", "ann_date"}
    if not required.issubset(frame):
        raise ValueError("financial announcement schema missing")
    work = frame.copy()
    ann = pd.to_datetime(work["ann_date"].astype(str), errors="coerce")
    final = pd.to_datetime(work.get("f_ann_date", pd.Series(index=work.index, dtype=str)).astype(str), errors="coerce")
    visible = pd.concat([ann, final], axis=1).max(axis=1)
    period = pd.to_datetime(work["end_date"].astype(str), errors="coerce")
    work = work.loc[ann.notna() & visible.le(pd.Timestamp(observation_date)) & period.le(visible)].copy()
    work["_visible_date"] = visible.loc[work.index]
    # Distinct financial statement types must not silently overwrite each other.
    if "report_type" in work:
        work = work.loc[work["report_type"].astype(str).isin({"1", "1.0", "4", "4.0"})]
    work = work.sort_values(["ts_code", "_visible_date"])
    return work.drop_duplicates("ts_code", keep="last").set_index("ts_code")


def industry_mapping(frame, observation_date, required_codes):
    required = {"ts_code", "l2_name", "in_date", "out_date"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("current industry response missing membership schema")
    starts = pd.to_datetime(frame["in_date"].astype(str), errors="coerce")
    ends = pd.to_datetime(frame["out_date"].astype(str), errors="coerce")
    active = frame.loc[starts.notna() & starts.le(pd.Timestamp(observation_date)) &
                       (ends.isna() | ends.ge(pd.Timestamp(observation_date)))].copy()
    active["industry"] = ""
    for column in ("l3_name", "l2_name", "l1_name"):
        if column in active:
            labels = active[column].fillna("").astype(str).str.strip()
            active["industry"] = active["industry"].where(active["industry"].ne(""), labels)
    active = active.loc[active["industry"].ne("")]
    if active.groupby("ts_code")["industry"].nunique().gt(1).any():
        raise ValueError("conflicting current industry memberships")
    mapping = active.drop_duplicates("ts_code").set_index("ts_code")["industry"].to_dict()
    missing = sorted(set(required_codes) - set(mapping))
    if missing:
        raise ValueError(f"current industry coverage incomplete: {len(missing)} securities; {missing[:10]}")
    return mapping


def fetch_industry(query=None):
    query = query or tushare.query
    chunks, signatures = [], set()
    for offset in range(0, 100000, 2000):
        chunk = query("index_member_all", fields=ZER0SHARE_DATASET_MAP["sw_member"].fields_arg,
                      is_new="Y", limit=2000, offset=offset)
        if not {"ts_code", "l2_name", "in_date", "out_date"}.issubset(chunk):
            raise ValueError("industry endpoint returned an invalid schema")
        signature = tuple(map(tuple, chunk.fillna("").astype(str).to_numpy()))
        if signature in signatures and len(chunk):
            raise ValueError("industry pagination repeated a page")
        signatures.add(signature)
        chunks.append(chunk)
        if len(chunk) < 2000:
            return pd.concat(chunks, ignore_index=True)
    raise ValueError("industry pagination incomplete")


def annual_report_period(observation_date):
    day = pd.Timestamp(observation_date)
    year = day.year - (1 if day >= pd.Timestamp(day.year, 4, 30) else 2)
    return f"{year}-12-31"


def comparable_share_factors(distributions, annual_period, latest_period, observation_date):
    """Use implemented, already visible stock distributions; cash dividends do not dilute EPS."""
    if not set(DIVIDEND_FIELDS).issubset(distributions):
        raise ValueError("EPS adjustment distribution schema incomplete")
    annual, latest, day = map(pd.Timestamp, (annual_period, latest_period, observation_date))
    annual_to_latest = latest_to_observation = 1.0
    events = {}
    for row in distributions.to_dict("records"):
        ann = pd.to_datetime(str(row.get("ann_date")), errors="coerce")
        implemented = pd.to_datetime(str(row.get("imp_ann_date")), errors="coerce")
        if pd.isna(ann):
            raise ValueError("EPS adjustment announcement date missing")
        raw_implementation_date = row.get("imp_ann_date")
        if (pd.notna(raw_implementation_date) and str(raw_implementation_date).strip()
                and pd.isna(implemented)):
            raise ValueError("EPS adjustment implementation announcement date invalid")
        if ann > day or (pd.notna(implemented) and implemented > day):
            continue
        if row.get("div_proc") in {"预案", "股东大会通过", "不分配", "取消分红"}:
            continue
        if row.get("div_proc") != "实施":
            raise ValueError("EPS adjustment implementation status unknown")
        ex = pd.to_datetime(str(row.get("ex_date")), errors="coerce")
        if pd.isna(ex):
            raise ValueError("EPS adjustment ex-date missing")
        if ex > day or ex <= annual:
            continue
        if ann > ex or (pd.notna(implemented) and implemented > ex):
            raise ValueError("EPS adjustment announcement after implementation")
        total = pd.to_numeric(row.get("stk_div"), errors="coerce")
        bonus = pd.to_numeric(row.get("stk_bo_rate"), errors="coerce")
        transfer = pd.to_numeric(row.get("stk_co_rate"), errors="coerce")
        components = bonus + transfer if np.isfinite([bonus, transfer]).all() else None
        if not np.isfinite(total):
            total = components
        if total is None or not np.isfinite(total) or total < 0:
            raise ValueError("EPS adjustment stock distribution unknown")
        if components is not None and not np.isclose(total, components, atol=1e-9):
            raise ValueError("EPS adjustment conflicting stock distribution")
        key = ex.strftime("%Y-%m-%d")
        if key in events and not np.isclose(events[key], total):
            raise ValueError("EPS adjustment conflicting events on one ex-date")
        events[key] = float(total)
    for ex, ratio in sorted(events.items()):
        if pd.Timestamp(ex) <= latest:
            annual_to_latest *= 1 + ratio
        else:
            latest_to_observation *= 1 + ratio
    return {"annual_to_latest_factor": annual_to_latest,
            "latest_to_observation_factor": latest_to_observation,
            "annual_to_observation_factor": annual_to_latest * latest_to_observation,
            "basis": "visible_implemented_stock_distributions_to_observation_share_basis",
            "events": [{"ex_date": ex, "stock_ratio": ratio} for ex, ratio in sorted(events.items())]}


def capture_share_adjustments(query, codes, day_root, annual_period, latest_period, observation_date):
    """A complete per-security response is evidence; failures never become zero distributions."""
    query = query or tushare.query
    output = {}
    source_root = day_root / "share_adjustment_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    for ts_code in sorted(codes):
        source = source_root / f"{ts_code}.csv"
        receipt_path = source.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
        if receipt is not None:
            stamp = pd.to_datetime(receipt.get("captured_at"), utc=True, errors="coerce")
            if (receipt.get("ts_code") != ts_code or receipt.get("observation_date") != observation_date
                    or pd.isna(stamp) or stamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != observation_date
                    or stamp > pd.Timestamp(capture_clock(observation_date)) or not source.is_file()
                    or receipt.get("sha256") != sha256(source) or receipt.get("pagination_complete") is not True):
                raise ValueError("EPS adjustment saved source receipt invalid")
            frame = pd.read_csv(source, dtype=str)
        else:
            pages, signatures = [], set()
            for offset in range(0, 100000, 2000):
                chunk = query("dividend", ts_code=ts_code, fields=",".join(DIVIDEND_FIELDS), limit=2000, offset=offset)
                if chunk is None or not set(DIVIDEND_FIELDS).issubset(chunk):
                    raise ValueError(f"EPS adjustment dividend response incomplete: {ts_code}")
                if not chunk["ts_code"].eq(ts_code).all():
                    raise ValueError("EPS adjustment provider did not honor stock filter")
                signature = tuple(map(tuple, chunk.fillna("").astype(str).to_numpy()))
                if signature in signatures and len(chunk):
                    raise ValueError("EPS adjustment repeated dividend page")
                signatures.add(signature)
                pages.append(chunk)
                if len(chunk) < 2000:
                    break
            else:
                raise ValueError("EPS adjustment dividend pagination incomplete")
            frame = pd.concat(pages, ignore_index=True).drop_duplicates()
            # Validate before publishing a reusable successful source receipt.
            comparable_share_factors(frame, annual_period, latest_period, observation_date)
            frame.to_csv(source, index=False, encoding="utf-8-sig")
            receipt = {"ts_code": ts_code, "observation_date": observation_date,
                       "captured_at": capture_clock(observation_date), "sha256": sha256(source),
                       "pagination_complete": True, "row_count": len(frame)}
            write_json(receipt_path, receipt)
        factors = comparable_share_factors(frame, annual_period, latest_period, observation_date)
        output[ts_code] = {**factors, "receipt": receipt,
                           "raw_distributions": json.loads(frame.to_json(orient="records", date_format="iso"))}
    source = day_root / "eps_adjustments.json"
    write_json(source, output)
    return output, source


def build_value_rows(prices, basic, financial, income, industries, observation_date, captured_at, *,
                     st_codes=(), annual_financial=None, share_adjustments=None, annual_source_sha256=None,
                     eps_adjustment_source_sha256=None, annual_capital=None, latest_capital=None,
                     value_observations=None):
    """Reuse the value formula and candidate gates; preserve raw absolute prices."""
    day = pd.Timestamp(observation_date)
    history = prices.loc[pd.to_datetime(prices["date"]).le(day)].copy()
    current = history.loc[pd.to_datetime(history["date"]).eq(day)]
    if current.empty:
        raise ValueError("left model has no same-day prices")
    master = basic.set_index("code")
    rows, assessed = [], 0
    for source in current.to_dict("records"):
        code = source["code"]
        if source.get("volume", 0) <= 0 or str(source.get("tradestatus", "1")) not in {"1", "1.0"}:
            continue
        if not code.startswith(("sh.", "sz.")):
            continue
        if code not in master.index:
            raise ValueError(f"left security master coverage incomplete: {code}")
        security = master.loc[code]
        if isinstance(security, pd.DataFrame):
            raise ValueError("duplicate security master")
        listed = pd.to_datetime(security.get("list_date"), errors="coerce")
        if pd.isna(listed):
            raise ValueError(f"left security listing date missing: {code}")
        if listed > day:
            continue
        delist = pd.to_datetime(security.get("delist_date"), errors="coerce")
        if pd.notna(delist) and delist <= day:
            continue
        ts_code = code.split(".")[1] + "." + code.split(".")[0].upper()
        # New listings after the financial period cannot be required to have
        # that pre-listing financial universe; retain the exclusion in receipt.
        if listed > pd.Timestamp(latest_visible_report_period(day)):
            continue
        # Known ineligibility is an economic exclusion, never a missing-data substitution.
        if code in st_codes or "ST" in str(security.get("name", "")).upper() or "退" in str(security.get("name", "")):
            assessed += 1
            continue
        if ts_code not in industries:
            raise ValueError(f"current industry coverage incomplete: {code}")
        industry = industries[ts_code]
        if not is_value_industry_allowed(industry):
            assessed += 1
            continue
        market_cap = pd.to_numeric(source.get("total_mv"), errors="coerce")
        if not np.isfinite(market_cap) or market_cap <= 0:
            raise ValueError(f"left current market cap missing: {code}")
        if market_cap < 1_000_000.:
            assessed += 1
            continue
        if ts_code not in financial.index or ts_code not in income.index:
            raise ValueError(f"visible financial coverage incomplete: {code}")
        if annual_financial is None or ts_code not in annual_financial.index:
            raise ValueError(f"visible annual financial coverage incomplete: {code}")
        annual = annual_financial.loc[ts_code]
        expected_annual = annual_report_period(day)
        if pd.Timestamp(str(annual.get("end_date"))) != pd.Timestamp(expected_annual):
            raise ValueError(f"annual EPS report period mismatch: {code}")
        if pd.isna(annual.get("_visible_date")) or pd.Timestamp(annual["_visible_date"]) > day:
            raise ValueError(f"annual EPS announcement is not visible: {code}")
        raw_eps = pd.to_numeric(annual.get("dt_eps"), errors="coerce")
        latest = financial.loc[ts_code].copy()
        raw_bps = pd.to_numeric(latest.get("bps"), errors="coerce")
        latest_yoy = pd.to_numeric(latest.get("dt_netprofit_yoy"), errors="coerce")
        if not np.isfinite([raw_eps, raw_bps, latest_yoy]).all():
            raise ValueError(f"annual deducted EPS/latest BPS/deducted growth missing: {code}")
        assessed += 1
        if raw_eps <= 0:
            continue
        roe = pd.to_numeric(latest.get("roe_dt"), errors="coerce")
        if not np.isfinite(roe):
            roe = pd.to_numeric(latest.get("roe"), errors="coerce")
        if not np.isfinite(roe):
            raise ValueError(f"left quality ROE source missing: {code}")
        adjustment = (share_adjustments or {}).get(ts_code)
        if adjustment is None or not annual_source_sha256 or not eps_adjustment_source_sha256:
            raise ValueError(f"annual EPS comparable share sources missing: {code}")
        annual_factor = pd.to_numeric(adjustment.get("annual_to_observation_factor"), errors="coerce")
        latest_factor = pd.to_numeric(adjustment.get("latest_to_observation_factor"), errors="coerce")
        if not np.isfinite([annual_factor, latest_factor]).all() or min(annual_factor, latest_factor) < 1:
            raise ValueError(f"annual EPS comparable share factors invalid: {code}")
        if (annual_capital is None or latest_capital is None
                or ts_code not in annual_capital.index or ts_code not in latest_capital.index):
            raise ValueError(f"annual EPS share-basis financial sources missing: {code}")
        annual_shares = pd.to_numeric(annual_capital.loc[ts_code].get("total_share"), errors="coerce")
        latest_shares = pd.to_numeric(latest_capital.loc[ts_code].get("total_share"), errors="coerce")
        observation_shares = pd.to_numeric(source.get("total_share"), errors="coerce") * 10000.
        annual_profit = pd.to_numeric(annual.get("profit_dedt"), errors="coerce")
        latest_net_assets = pd.to_numeric(latest_capital.loc[ts_code].get("total_hldr_eqy_exc_min_int"), errors="coerce")
        if (not np.isfinite([annual_shares, latest_shares, observation_shares, annual_profit, latest_net_assets]).all()
                or min(annual_shares, latest_shares, observation_shares) <= 0):
            raise ValueError(f"annual EPS/BPS share-basis data incomplete: {code}")
        if (not np.isclose(raw_eps, annual_profit / annual_shares, atol=.005, rtol=0)
                or not np.isclose(raw_bps, latest_net_assets / latest_shares, atol=.005, rtol=0)):
            raise ValueError(f"annual EPS/BPS report-end share basis cannot be verified: {code}")
        if (not np.isclose(annual_shares * annual_factor / latest_factor, latest_shares, rtol=1e-5, atol=1.)
                or not np.isclose(latest_shares * latest_factor, observation_shares, rtol=1e-5, atol=1.)):
            raise ValueError(f"share-count change is not explained by captured distributions: {code}")
        # Never substitute ordinary earnings or annualize the current quarter.
        latest["dt_eps"] = float(raw_eps) / annual_factor
        latest["bps"] = float(raw_bps) / latest_factor
        metrics = _metrics_from_rows(latest, income.loc[ts_code],
                                     pd.Series(source), report_period=latest_visible_report_period(day))
        if metrics is None or not np.isfinite([metrics.get("mktcap", np.nan),
                                               metrics.get("quality_score", np.nan)]).all():
            raise ValueError(f"left financial fields incomplete: {code}")
        row = {"date": observation_date, "observation_date": observation_date, "code": code, "name": security.get("name", code),
               "candidate_source": "value_model", "industry": industry,
               "value_industry_allowed": is_value_industry_allowed(industry),
               "value_industry_allowlist_match": value_industry_allowlist_match(industry),
               "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
               "close": source["close"], "value_line": metrics["value_line"],
               "price_to_value": metrics["price_to_value"], "quality_score": metrics["quality_score"],
               "earnings_yoy": metrics["yoy"], "mktcap": metrics["mktcap"],
               "report_period": metrics["report_period"], "eps_excl": metrics["eps_excl"],
               "book_value_per_share": float(latest["bps"]), "book_value_per_share_raw": float(raw_bps),
               "annual_eps_excl_raw": float(raw_eps), "annual_eps_excl_comparable": metrics["eps_excl"],
               "annual_report_period": expected_annual,
               "annual_announcement_date": str(max(annual["_visible_date"], annual_capital.loc[ts_code, "_visible_date"]).date()),
               "annual_eps_announcement_date": str(annual["_visible_date"].date()),
               "annual_source_sha256": annual_source_sha256,
               "eps_adjustment_source_sha256": eps_adjustment_source_sha256,
               "eps_adjustment_factor": annual_factor,
               "book_value_adjustment_factor": latest_factor,
               "annual_report_total_shares": float(annual_shares),
               "latest_report_total_shares": float(latest_shares),
               "observation_total_shares": float(observation_shares),
               "annual_eps_share_basis": "verified_annual_deducted_profit_over_report_end_shares",
               "book_value_share_basis": "verified_latest_parent_equity_over_report_end_shares",
               "comparable_share_basis_date": observation_date,
               "eps_adjustment_basis": adjustment["basis"],
               "earnings_yoy_source": "latest_report_dt_netprofit_yoy_proxy_for_deducted_eps_yoy",
               "quality_score_source": "existing_tushare_annual_eps_growth_ROE_proxy_not_MiniQMT_three_year_score",
               "value_formula_source": "latest_BPS_plus_comparable_annual_deducted_EPS_times_latest_deducted_growth",
               "financial_point_in_time": True, "industry_point_in_time": True,
               "financial_point_in_time_source": "same_day_capture_and_visible_announcements",
               "announcement_date": str(max(financial.loc[ts_code, "_visible_date"], income.loc[ts_code, "_visible_date"],
                                             latest_capital.loc[ts_code, "_visible_date"]).date()),
               "industry_point_in_time_source": "same_day_index_member_all_snapshot",
               "captured_at": captured_at, "research_only": True, "strict_point_in_time": False,
               "signal_eligible": True, "selected_for_trading": True, "data_status": "traded"}
        if value_observations is not None:
            value_observations.append(dict(row))
        # Failed economic gates are genuine exclusions; missing data above are failures.
        row = normalize_candidate(row)
        if not row["allow_left"] or not row["signal_eligible"]:
            continue
        frame = history.loc[history["code"].eq(code)].sort_values("date").set_index("date").copy()
        factors = pd.to_numeric(frame["raw_to_qfq_factor"], errors="coerce")
        if not np.isfinite(factors).all() or factors.le(0).any():
            raise ValueError(f"left adjustment factors incomplete: {code}")
        # Repository factor is raw/qfq: rebase to today's raw absolute price.
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] / factors * factors.iloc[-1]
        frame["amount"] = frame["amount"] * 1000.0
        feature = _candidate_feature_frame(frame).iloc[-1]
        row.update(_trade_basis_from_feature_row(feature))
        row.update(_leadership_from_feature_row(feature))
        row = normalize_candidate(row)
        rows.append(row)
    if assessed == 0:
        raise ValueError("left financial universe was not assessed")
    return sorted(rows, key=lambda row: (-row["candidate_score"], row["code"])), assessed


def capture_left(*, database, sync_state, output_root, observation_date, query=None):
    captured_at = capture_clock(observation_date)
    root = Path(output_root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("implementation") != implementation_fingerprint():
            raise ValueError("left capture implementation changed; start a separate stream")
        if any(item["date"] == observation_date for item in manifest["snapshots"]):
            load_left_snapshot(root, observation_date, captured_at)
            return root
    else:
        manifest = {"version": "forward-value-left-v2", "snapshots": [], "research_only": True,
                    "strict_point_in_time": False, "historical_backfill_is_forward_evidence": False,
                    "implementation": implementation_fingerprint()}
    state = json.loads(Path(sync_state).read_text(encoding="utf-8"))
    validate_sync_receipt(state, database, observation_date, captured_at)
    repository = TushareRepository(Database(database))
    verify_completed_receipts(state, repository)
    period = latest_visible_report_period(observation_date)
    annual_period = annual_report_period(observation_date)
    for dataset, receipt_period in (("fina_indicator", period), ("income", period), ("fina_indicator", annual_period),
                                    ("balancesheet", period), ("balancesheet", annual_period)):
        receipt = state.get("completed", {}).get(f"{dataset}:{receipt_period}:as_of:{observation_date}", {})
        stamp = pd.to_datetime(receipt.get("captured_at"), utc=True, errors="coerce")
        if (pd.isna(stamp) or stamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != observation_date
                or stamp > pd.Timestamp(captured_at)):
            raise ValueError(f"left financial receipt is not same-day: {dataset}:{receipt_period}")
    with repository.database.connect(read_only=True) as connection:
        financial = visible_financial_rows(_read_dataset(connection, "fina_indicator", report_period=period), observation_date)
        income = visible_financial_rows(_read_dataset(connection, "income", report_period=period), observation_date)
        annual = visible_financial_rows(_read_dataset(connection, "fina_indicator", report_period=annual_period), observation_date)
        annual_capital = visible_financial_rows(_read_dataset(connection, "balancesheet", report_period=annual_period), observation_date)
        latest_capital = visible_financial_rows(_read_dataset(connection, "balancesheet", report_period=period), observation_date)
    basic = repository.load_stock_basic_universe()
    start = (pd.Timestamp(observation_date) - pd.Timedelta(days=750)).strftime("%Y-%m-%d")
    prices = repository.load_market_daily_frame(start_date=start, end_date=observation_date)
    daily_capital = repository.load_dataset_for_codes("daily_basic", basic["code"],
                                                      start_date=observation_date, end_date=observation_date)
    if "total_share" not in daily_capital:
        raise ValueError("observation share capital source missing")
    if daily_capital.groupby("code")["total_share"].nunique(dropna=False).gt(1).any():
        raise ValueError("conflicting observation share capital sources")
    capital_lookup = daily_capital.drop_duplicates("code").set_index("code")["total_share"]
    prices["total_share"] = prices["code"].map(capital_lookup)
    industry = fetch_industry(query)
    mapping = industry_mapping(industry, observation_date, set())
    st = repository.load_dataset_for_codes("stock_st", basic["code"], start_date=observation_date, end_date=observation_date)
    st_codes = set(st["code"]) if not st.empty else set()
    day_root = root / observation_date
    day_root.mkdir(parents=True, exist_ok=True)
    annual_path = day_root / "annual_financial.json"
    write_json(annual_path, {"report_period": annual_period,
                            "fina_indicator": json.loads(annual.reset_index().to_json(orient="records", date_format="iso")),
                            "balancesheet": json.loads(annual_capital.reset_index().to_json(orient="records", date_format="iso"))})
    current = prices.loc[pd.to_datetime(prices["date"]).eq(pd.Timestamp(observation_date))]
    eligible_market = (pd.to_numeric(current["volume"], errors="coerce").gt(0)
                       & pd.to_numeric(current["total_mv"], errors="coerce").ge(1_000_000.))
    active_codes = set(current.loc[eligible_market, "code"]) - st_codes
    inactive = basic.loc[
        basic["name"].fillna("").str.upper().str.contains("ST|退")
        | pd.to_datetime(basic["list_date"], errors="coerce").gt(pd.Timestamp(period))
        | pd.to_datetime(basic["delist_date"], errors="coerce").le(pd.Timestamp(observation_date)), "code"]
    active_codes -= set(inactive)
    adjustment_codes = {code.split(".")[1] + "." + code.split(".")[0].upper() for code in active_codes
                        if code.startswith(("sh.", "sz."))}
    adjustment_codes = {code for code in adjustment_codes if is_value_industry_allowed(mapping.get(code))}
    adjustments, adjustment_path = capture_share_adjustments(
        query, adjustment_codes, day_root, annual_period, period, observation_date,
    )
    value_observations = []
    rows, assessed = build_value_rows(prices, basic, financial, income, mapping, observation_date, captured_at,
                                      st_codes=st_codes, annual_financial=annual, share_adjustments=adjustments,
                                      annual_source_sha256=sha256(annual_path),
                                      eps_adjustment_source_sha256=sha256(adjustment_path),
                                      annual_capital=annual_capital, latest_capital=latest_capital,
                                      value_observations=value_observations)
    captured_at = capture_clock(observation_date)
    verify_completed_receipts(state, repository)
    if json.loads(Path(sync_state).read_text(encoding="utf-8")) != state:
        raise ValueError("sync state changed during left capture")
    csv_path = day_root / "candidates.csv"
    for row in [*rows, *value_observations]:
        row["captured_at"] = captured_at
    industry_path = day_root / "industry.csv"
    industry.to_csv(industry_path, index=False, encoding="utf-8-sig")
    financial_path = day_root / "financial.json"
    financial_records = {"fina_indicator": json.loads(financial.reset_index().to_json(orient="records", date_format="iso")),
                         "income": json.loads(income.reset_index().to_json(orient="records", date_format="iso")),
                         "balancesheet": json.loads(latest_capital.reset_index().to_json(orient="records", date_format="iso")),
                         "observation_share_capital": json.loads(daily_capital.to_json(orient="records", date_format="iso"))}
    write_json(financial_path, financial_records)
    for row in [*rows, *value_observations]:
        row["financial_source_sha256"] = sha256(financial_path)
        row["industry_source_sha256"] = sha256(industry_path)
    pd.DataFrame(rows, columns=None if rows else ["code", "candidate_source"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    from stock_research.research.value_policy import capped_value_observation
    value_source_path = day_root / "value_observations.json"
    write_json(value_source_path, {"observations": value_observations, "captured_at": captured_at,
                                  "scope": "verified_model_universe_before_quality_growth_price_to_value_filters"})
    value_policy_path = day_root / "r07_growth_cap_diagnostics.json"
    write_json(value_policy_path, {"research_id": "R07", "research_only": True,
                                  "strict_point_in_time": False, "production_candidates_changed": False,
                                  "observation_date": observation_date, "captured_at": captured_at,
                                  "source_sha256": sha256(value_source_path),
                                  "observations": [{"code": row["code"], **capped_value_observation(row, as_of=captured_at)}
                                                   for row in value_observations]})
    manifest["snapshots"].append({"date": observation_date, "file": csv_path.relative_to(root).as_posix(),
        "sha256": sha256(csv_path), "captured_at": captured_at, "candidate_count": len(rows),
        "assessed_count": assessed, "universe_rule": "current traded SH/SZ listed by visible report period",
        "industry_file": industry_path.relative_to(root).as_posix(), "industry_sha256": sha256(industry_path),
        "financial_file": financial_path.relative_to(root).as_posix(), "financial_sha256": sha256(financial_path),
        "annual_financial_file": annual_path.relative_to(root).as_posix(), "annual_financial_sha256": sha256(annual_path),
        "eps_adjustment_file": adjustment_path.relative_to(root).as_posix(), "eps_adjustment_sha256": sha256(adjustment_path),
        "annual_report_period": annual_period,
        "value_observations_file": value_source_path.relative_to(root).as_posix(), "value_observations_sha256": sha256(value_source_path),
        "r07_diagnostics_file": value_policy_path.relative_to(root).as_posix(), "r07_diagnostics_sha256": sha256(value_policy_path),
        "sync_state_sha256": sha256(sync_state), "research_only": True, "strict_point_in_time": False})
    write_json(manifest_path, manifest)
    return root
