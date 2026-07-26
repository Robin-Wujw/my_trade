"""Build q1_value-compatible financial cache files from local Tushare SQLite."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_research.core.as_of import write_metadata
from stock_research.core.paths import PATHS
from stock_research.indicators.factors import clamp, score_direct, score_inverse
from stock_research.storage import Database


CACHE_VERSION = "tushare-financial-cache-v1"


def _num(value):
    try:
        if value is None or pd.isna(value):
            return None
        converted = float(value)
        if math.isnan(converted):
            return None
        return converted
    except (TypeError, ValueError):
        return None


def _date_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    converted = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(converted):
        return None
    return converted.strftime("%Y-%m-%d")


def _symbol(ts_code: str) -> str:
    return str(ts_code or "").split(".", 1)[0].zfill(6)


def _local_code(ts_code: str) -> str:
    symbol = _symbol(ts_code)
    return ("sh." if symbol.startswith(("6", "9")) else "sz.") + symbol


def _read_dataset(connection, dataset: str, *, report_period: str | None = None) -> pd.DataFrame:
    if report_period:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM raw.tushare_dataset_rows
            WHERE dataset = ?
              AND report_period = ?
            """,
            [dataset, report_period],
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM raw.tushare_dataset_rows
            WHERE dataset = ?
            """,
            [dataset],
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(row[0]) for row in rows])


def _read_daily_basic(connection, *, as_of_date: str) -> pd.DataFrame:
    rows = connection.execute(
        """
        WITH latest AS (
            SELECT ts_code, MAX(trade_date) AS trade_date
            FROM raw.tushare_dataset_rows
            WHERE dataset = 'daily_basic'
              AND trade_date <= ?
            GROUP BY ts_code
        )
        SELECT r.payload_json
        FROM raw.tushare_dataset_rows r
        JOIN latest l
          ON r.ts_code = l.ts_code
         AND r.trade_date = l.trade_date
        WHERE r.dataset = 'daily_basic'
        """,
        [as_of_date],
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(row[0]) for row in rows])


def _dedupe_latest_by_announcement(frame: pd.DataFrame, *, as_of_date: str) -> pd.DataFrame:
    if frame.empty or "ts_code" not in frame:
        return pd.DataFrame()
    work = frame.copy()
    for column in ("ann_date", "f_ann_date"):
        if column not in work:
            work[column] = None
        work[column] = pd.to_datetime(work[column], errors="coerce")
    as_of = pd.Timestamp(as_of_date).normalize()
    visible = work["ann_date"].notna() & (work["ann_date"].dt.normalize() <= as_of)
    work = work.loc[visible].copy()
    if work.empty:
        return pd.DataFrame()
    work["_announce_sort"] = work["ann_date"].fillna(work["f_ann_date"])
    return (
        work.sort_values(["ts_code", "_announce_sort"])
        .drop_duplicates("ts_code", keep="last")
        .drop(columns=["_announce_sort"])
        .reset_index(drop=True)
    )


def _quality_score(eps_excl: float | None, yoy: float | None, roe: float | None) -> float:
    quality_yoy = None if yoy is None else min(max(yoy, -0.5), 1.0)
    roe_ratio = None if roe is None else roe / 100.0
    score = (
        score_direct(eps_excl, 0.10, 1.50) * 0.35
        + score_direct(quality_yoy, -0.10, 0.50) * 0.35
        + score_direct(roe_ratio, 0.04, 0.18) * 0.20
        + score_direct(yoy, 0.0, 0.30) * 0.10
    )
    return float(clamp(score))


def _metrics_from_rows(
    financial: pd.Series,
    income: pd.Series | None,
    market: pd.Series | None,
    *,
    report_period: str,
) -> dict | None:
    bps = _num(financial.get("bps"))
    eps_excl = _num(financial.get("dt_eps"))
    if eps_excl is None:
        eps_excl = _num(financial.get("eps"))
    if eps_excl is None and income is not None:
        eps_excl = _num(income.get("basic_eps"))
    yoy_pct = _num(financial.get("dt_netprofit_yoy"))
    if yoy_pct is None:
        yoy_pct = _num(financial.get("netprofit_yoy"))
    if bps is None or eps_excl is None or yoy_pct is None:
        return None

    yoy = yoy_pct / 100.0
    modeled_value_line = bps + eps_excl * (1 + yoy) * 10
    loss_maker = eps_excl <= 0
    value_line = modeled_value_line
    value_line_policy = "tushare_report_period_eps_value_line"
    if loss_maker or value_line <= 0:
        value_line = max(bps, 0.01)
        value_line_policy = "book_value_floor_for_loss_maker"

    close = _num(market.get("close")) if market is not None else None
    total_mv = _num(market.get("total_mv")) if market is not None else None
    circ_mv = _num(market.get("circ_mv")) if market is not None else None
    total_share = None
    if close and close > 0 and total_mv and total_mv > 0:
        total_share = total_mv * 10000.0 / close
    mktcap = total_mv / 10000.0 if total_mv is not None else None
    avg_amount20 = None
    liquidity_score = None
    if market is not None:
        amount = _num(market.get("amount"))
        if amount and amount > 0:
            avg_amount20 = amount * 1000.0
            liquidity_score = score_direct(math.log10(avg_amount20), 7.0, 9.5)

    ann_date = _date_text(financial.get("ann_date"))
    f_ann_date = _date_text(income.get("f_ann_date") if income is not None else None)
    quality = _quality_score(eps_excl, yoy, _num(financial.get("roe_dt")) or _num(financial.get("roe")))
    return {
        "cache_version": CACHE_VERSION,
        "value_line": float(value_line),
        "modeled_value_line": float(modeled_value_line),
        "value_line_policy": value_line_policy,
        "price_to_value": None if close is None or value_line <= 0 else float(close / value_line),
        "valuation_score": score_inverse(
            None if close is None or value_line <= 0 else close / value_line,
            best=0.55,
            worst=1.25,
        ),
        "quality_score": quality,
        "mktcap": None if mktcap is None else float(mktcap),
        "eps_excl": float(eps_excl),
        "yoy": float(yoy),
        "yoy_source": "tushare fina_indicator.dt_netprofit_yoy",
        "latest_excl_eps": float(eps_excl),
        "prev_excl_eps": None,
        "latest_report": report_period,
        "annual_report": report_period,
        "data_source": "tushare/sqlite",
        "total_share": None if total_share is None else float(total_share),
        "eps_excl_raw": float(eps_excl),
        "eps_adjustment_factor": 1.0,
        "eps_excl_source": "Tushare fina_indicator.dt_eps",
        "eps_bonus_detail": None,
        "loss_maker": bool(loss_maker),
        "report_period": report_period,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_history_available": False,
        "financial_point_in_time_source": "announce_time",
        "announcement_date": ann_date,
        "annual_announcement_date": ann_date or f_ann_date,
        "capital_announcement_date": ann_date or f_ann_date,
        "tushare_ts_code": financial.get("ts_code"),
        "market_date": _date_text(market.get("trade_date")) if market is not None else None,
        "market_total_mv_10k_yuan": total_mv,
        "market_circ_mv_10k_yuan": circ_mv,
        "avg_amount20": avg_amount20,
        "liquidity_score": liquidity_score,
        "strict_point_in_time_note": (
            "Financial rows are selected by ann_date <= as_of_date; daily_basic "
            "market fields use the latest trade_date not later than as_of_date."
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{random.randint(100000, 999999)}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_cache(
    *,
    report_period: str,
    as_of_date: str,
    market_as_of_date: str | None = None,
    output_directory: Path,
    database_path: Path,
) -> dict:
    database = Database(database_path, code_version="tushare-financial-cache-v1")
    database.initialize()
    connection = database.connect(read_only=True)
    try:
        financial = _dedupe_latest_by_announcement(
            _read_dataset(connection, "fina_indicator", report_period=report_period),
            as_of_date=as_of_date,
        )
        income = _dedupe_latest_by_announcement(
            _read_dataset(connection, "income", report_period=report_period),
            as_of_date=as_of_date,
        )
        market_cutoff = market_as_of_date or as_of_date
        market = _read_daily_basic(connection, as_of_date=market_cutoff)
    finally:
        connection.close()

    if financial.empty:
        raise RuntimeError(f"no visible fina_indicator rows for {report_period} as of {as_of_date}")
    income_by_code = {}
    if not income.empty:
        income_by_code = {
            str(row["ts_code"]): row
            for _, row in income.drop_duplicates("ts_code", keep="last").iterrows()
        }
    market_by_code = {}
    if not market.empty and "ts_code" in market:
        market_by_code = {
            str(row["ts_code"]): row
            for _, row in market.drop_duplicates("ts_code", keep="last").iterrows()
        }

    saved = skipped = 0
    for _, row in financial.iterrows():
        ts_code = str(row.get("ts_code") or "")
        metrics = _metrics_from_rows(
            row,
            income_by_code.get(ts_code),
            market_by_code.get(ts_code),
            report_period=report_period,
        )
        if metrics is None:
            skipped += 1
            continue
        path = output_directory / f"{_symbol(ts_code)}_{report_period.replace('-', '')}.json"
        _write_json_atomic(path, metrics)
        saved += 1

    metadata = {
        "kind": "tushare_financial_cache",
        "point_in_time_status": "safe",
        "report_period": report_period,
        "as_of_date": as_of_date,
        "market_as_of_date": market_as_of_date or as_of_date,
        "financial_rows": int(len(financial)),
        "income_rows": int(len(income)),
        "market_rows": int(len(market)),
        "saved_count": saved,
        "skipped_count": skipped,
        "output_directory": str(output_directory),
        "database": str(database_path),
        "data_source": "tushare/sqlite",
        "cache_version": CACHE_VERSION,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / f"tushare_financial_cache_{report_period.replace('-', '')}_{as_of_date.replace('-', '')}.json"
    _write_json_atomic(metadata_path, metadata)
    write_metadata(str(metadata_path), metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-period", required=True, help="Report period, YYYY-MM-DD.")
    parser.add_argument("--as-of-date", required=True, help="Visibility cutoff, YYYY-MM-DD.")
    parser.add_argument("--market-as-of-date", default="", help="Market/share cutoff, YYYY-MM-DD; defaults to as-of-date.")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--output-directory", default=str(PATHS.cache / "q1_value"))
    args = parser.parse_args(argv)

    metadata = build_cache(
        report_period=pd.Timestamp(args.report_period).strftime("%Y-%m-%d"),
        as_of_date=pd.Timestamp(args.as_of_date).strftime("%Y-%m-%d"),
        market_as_of_date=(
            pd.Timestamp(args.market_as_of_date).strftime("%Y-%m-%d")
            if args.market_as_of_date
            else None
        ),
        output_directory=Path(args.output_directory),
        database_path=Path(args.database),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
