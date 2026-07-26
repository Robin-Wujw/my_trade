"""Run a Tushare-SQLite point-in-time research backtest.

This entrypoint is intentionally self contained: it reads the unified
``raw_tushare_dataset_rows`` cache, builds Formula33 market phases and a
daily_basic-based right-side candidate pool, then runs the existing portfolio
execution engine.  It does not call network providers.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.indicators.formula33 import calc_kdj_k, calc_rsi, calc_wr
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION
from stock_research.strategies.historical_candidates import CANDIDATE_SNAPSHOT_COLUMNS, SNAPSHOT_VERSION
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2022-01-04")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "tushare_sqlite_2022"),
    )
    parser.add_argument("--daily-candidates", type=int, default=60)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-total-held-symbols", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    return parser


def _date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _ts_to_local(ts_code: str) -> str:
    symbol, suffix = str(ts_code).split(".", 1)
    return f"{suffix.lower()}.{symbol}"


def _local_to_symbol(code: str) -> str:
    return str(code).split(".")[-1].zfill(6)


def _read_dataset(
    conn: sqlite3.Connection,
    dataset: str,
    start_date: str,
    end_date: str,
    fields: dict[str, str],
) -> pd.DataFrame:
    projections = [
        "ts_code",
        "trade_date",
        *[
            f"json_extract(payload_json, '$.{payload_key}') AS {column}"
            for column, payload_key in fields.items()
        ],
    ]
    sql = f"""
        SELECT {", ".join(projections)}
        FROM raw_tushare_dataset_rows
        WHERE dataset = ?
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY ts_code, trade_date
    """
    frame = pd.read_sql_query(sql, conn, params=[dataset, start_date, end_date])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["code"] = frame["ts_code"].map(_ts_to_local)
    return frame.dropna(subset=["date", "code"])


def _read_basic(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    rows = conn.execute(
        """
        SELECT payload_json
        FROM raw_tushare_dataset_rows
        WHERE dataset IN ('basic', 'stock_basic')
        """
    ).fetchall()
    names: dict[str, str] = {}
    list_dates: dict[str, pd.Timestamp] = {}
    for (payload_json,) in rows:
        try:
            row = json.loads(payload_json)
        except (TypeError, ValueError):
            continue
        ts_code = row.get("ts_code")
        if not ts_code:
            continue
        code = _ts_to_local(ts_code)
        names[code] = str(row.get("name") or code)
        list_date = pd.to_datetime(row.get("list_date"), errors="coerce")
        if pd.notna(list_date):
            list_dates[code] = list_date.normalize()
    return names, list_dates


def _read_code_date_set(conn: sqlite3.Connection, dataset: str, start_date: str, end_date: str) -> set[tuple[str, str]]:
    frame = pd.read_sql_query(
        """
        SELECT ts_code, trade_date
        FROM raw_tushare_dataset_rows
        WHERE dataset = ?
          AND trade_date >= ?
          AND trade_date <= ?
        """,
        conn,
        params=[dataset, start_date, end_date],
    )
    if frame.empty:
        return set()
    return {
        (_ts_to_local(row.ts_code), _date_text(row.trade_date))
        for row in frame.itertuples(index=False)
        if row.ts_code and row.trade_date
    }


def _read_industry_intervals(conn: sqlite3.Connection) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    rows = conn.execute(
        """
        SELECT payload_json
        FROM raw_tushare_dataset_rows
        WHERE dataset = 'sw_member'
        """
    ).fetchall()
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = defaultdict(list)
    for (payload_json,) in rows:
        try:
            row = json.loads(payload_json)
        except (TypeError, ValueError):
            continue
        ts_code = row.get("ts_code")
        if not ts_code:
            continue
        start = pd.to_datetime(row.get("in_date"), errors="coerce")
        end = pd.to_datetime(row.get("out_date"), errors="coerce")
        industry = str(row.get("l3_name") or row.get("l2_name") or row.get("l1_name") or "").strip()
        if not industry:
            continue
        result[_ts_to_local(ts_code)].append((
            pd.Timestamp.min if pd.isna(start) else start.normalize(),
            pd.Timestamp.max if pd.isna(end) else end.normalize(),
            industry,
        ))
    return result


def _industry_for(intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]], code: str, date: pd.Timestamp) -> str:
    for start, end, industry in intervals.get(code, []):
        if start <= date.normalize() <= end:
            return industry
    return ""


def _score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    close = pd.to_numeric(frame["close_adj"], errors="coerce")
    high = pd.to_numeric(frame["high_adj"], errors="coerce")
    low = pd.to_numeric(frame["low_adj"], errors="coerce")
    volume = pd.to_numeric(frame["vol"], errors="coerce")
    amount_yuan = pd.to_numeric(frame["amount"], errors="coerce") * 1000.0
    frame["return_5d"] = close / close.shift(5) - 1
    frame["return_10d"] = close / close.shift(10) - 1
    frame["return_20d"] = close / close.shift(20) - 1
    frame["return_60d"] = close / close.shift(60) - 1
    frame["return_120d"] = close / close.shift(120) - 1
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    frame["ma20_slope"] = ma20 / ma20.shift(10) - 1
    frame["ma60_slope"] = ma60 / ma60.shift(20) - 1
    frame["above_ma20"] = close >= ma20
    frame["near_ma20"] = (close / ma20 - 1).abs() <= 0.05
    frame["avg_amount_20"] = amount_yuan.rolling(20, min_periods=5).mean()
    frame["volatility_20"] = close.pct_change().rolling(20).std()
    frame["drawdown_60"] = close / close.rolling(60, min_periods=20).max() - 1
    frame["distance_120d_high"] = close / close.rolling(120, min_periods=40).max() - 1
    high_21 = close.rolling(21, min_periods=10).max()
    low_21 = close.rolling(21, min_periods=10).min()
    frame["range_21_pct"] = high_21 / low_21 - 1
    frame["close_position_21"] = (close - low_21) / (high_21 - low_21)
    frame["volume_ratio_calc"] = volume / volume.rolling(10, min_periods=5).mean().shift(1)
    momentum = (
        frame["return_20d"].clip(lower=-0.2, upper=0.5).fillna(0.0) * 30.0
        + frame["return_60d"].clip(lower=-0.3, upper=0.8).fillna(0.0) * 35.0
        + frame["return_120d"].clip(lower=-0.4, upper=1.2).fillna(0.0) * 20.0
    )
    liquidity = (frame["avg_amount_20"].fillna(0.0) / 3_000_000_000.0).clip(0, 1) * 10.0
    trend = (
        frame["above_ma20"].fillna(False).astype(float) * 5.0
        + frame["ma20_slope"].clip(lower=-0.05, upper=0.12).fillna(0.0) * 80.0
        + frame["ma60_slope"].clip(lower=-0.08, upper=0.20).fillna(0.0) * 50.0
    )
    risk_penalty = frame["volatility_20"].fillna(0.0).clip(0, 0.08) * 120.0
    overheat_penalty = frame["return_20d"].fillna(0.0).sub(0.6).clip(lower=0.0) * 30.0
    frame["right_quant_score"] = (55.0 + momentum + liquidity + trend - risk_penalty - overheat_penalty).clip(0, 100)
    frame["trade_basis_score"] = (
        frame["above_ma20"].fillna(False).astype(float) * 3.0
        + frame["ma20_slope"].gt(0).fillna(False).astype(float) * 2.0
        + frame["ma60_slope"].gt(0).fillna(False).astype(float) * 2.0
        + frame["volume_ratio_calc"].ge(1.2).fillna(False).astype(float) * 2.0
    ).clip(0, 10)
    frame["leadership_score"] = frame["right_quant_score"].sub(70).clip(lower=0, upper=30)
    return frame


def _formula_rows(
    price_frames: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
    list_dates: dict[str, pd.Timestamp],
    st_set: set[tuple[str, str]],
    suspend_set: set[tuple[str, str]],
) -> pd.DataFrame:
    hits: dict[pd.Timestamp, set[str]] = defaultdict(set)
    traded: dict[pd.Timestamp, set[str]] = defaultdict(set)
    for code, frame in price_frames.items():
        work = frame.sort_values("date").copy()
        if len(work) < 30:
            continue
        k = calc_kdj_k(pd.DataFrame({
            "high": work["high_adj"],
            "low": work["low_adj"],
            "close": work["close_adj"],
        }))
        wr10 = calc_wr(pd.DataFrame({"high": work["high_adj"], "low": work["low_adj"], "close": work["close_adj"]}), 10)
        wr20 = calc_wr(pd.DataFrame({"high": work["high_adj"], "low": work["low_adj"], "close": work["close_adj"]}), 20)
        rsi9 = calc_rsi(work["close_adj"], 9)
        ipo = list_dates.get(code, pd.Timestamp("1900-01-01"))
        list_days_ok = (work["date"] - ipo).dt.days > 300
        base = (k > 80) & (wr10 < 20) & (wr20 < 20) & (rsi9 > 70) & list_days_ok
        xg = base.rolling(5, min_periods=5).sum().eq(5)
        for row in work.loc[xg.fillna(False), ["date"]].itertuples(index=False):
            hits[row.date.normalize()].add(code)
        for row in work.loc[pd.to_numeric(work["vol"], errors="coerce").fillna(0).gt(0), ["date"]].itertuples(index=False):
            traded[row.date.normalize()].add(code)

    rows = []
    phase = "waiting"
    up_streak = down_streak = 0
    previous_count = None
    for index, date in enumerate(calendar):
        window_pool: set[str] = set()
        for window_date in calendar[max(0, index - 20): index + 1]:
            window_pool.update(hits.get(window_date, set()))
        date_text = date.strftime("%Y-%m-%d")
        excluded = {
            code for code in window_pool
            if (code, date_text) in st_set or (code, date_text) in suspend_set
        }
        formal = (window_pool & traded.get(date, set())) - excluded
        count = len(formal)
        change = 0 if previous_count is None else count - previous_count
        up_streak = up_streak + 1 if change > 0 else 0
        down_streak = down_streak + 1 if change < 0 else 0
        if down_streak >= 5:
            phase = "exited"
        elif up_streak >= 5:
            phase = "active"
        elif up_streak >= 3 and phase != "active":
            phase = "watch"
        rows.append({
            "date": date_text,
            "window_unique_count": count,
            "window_change": change,
            "window_up_streak": up_streak,
            "window_down_streak": down_streak,
            "phase": phase,
            "reconstruction_version": "tushare-sqlite-formula33-v1",
        })
        previous_count = count
    return pd.DataFrame(rows)


def _candidate_rows_for_date(
    date: pd.Timestamp,
    rows: list[dict],
    *,
    daily_candidates: int,
) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["right_quant_score", "trade_basis_score", "avg_amount_20"],
        ascending=[False, False, False],
    ).head(max(1, daily_candidates)).reset_index(drop=True)
    result = []
    for rank, row in enumerate(frame.to_dict("records"), start=1):
        score = float(row["right_quant_score"])
        quality_proxy = max(70.0, min(95.0, 70.0 + score * 0.25))
        result.append({
            "date": date.strftime("%Y-%m-%d"),
            "code": row["code"],
            "name": row.get("name") or row["code"],
            "industry": row.get("industry") or "",
            "value_industry_allowed": False,
            "value_industry_allowlist_match": "",
            "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
            "industry_point_in_time": True,
            "close": row["close"],
            "value_line": None,
            "quality_score": quality_proxy,
            "earnings_yoy": 0.10,
            "mktcap": row["mktcap"],
            "report_period": date.strftime("%Y-%m-%d"),
            "snapshot_version": SNAPSHOT_VERSION,
            "financial_point_in_time": True,
            "financial_point_in_time_source": "tushare_daily_basic_trade_date_proxy",
            "announcement_date": date.strftime("%Y-%m-%d"),
            "price_to_value": None,
            "mainline_snapshot_date": None,
            "mainline_snapshot_fresh": False,
            "mainline_boards": "",
            "trade_basis_score": row["trade_basis_score"],
            "trade_basis_reason": "Tushare daily_basic liquidity/trend proxy",
            "technical_alignment": "trade_ready" if row["trade_basis_score"] >= 7 else "watch",
            "ma20_rising": bool(row.get("ma20_slope", 0) > 0),
            "ma60_rising": bool(row.get("ma60_slope", 0) > 0),
            "above_ma20": bool(row.get("above_ma20", False)),
            "near_ma20": bool(row.get("near_ma20", False)),
            "known_volume_ratio": row.get("volume_ratio_calc"),
            "return_5d": row.get("return_5d"),
            "return_10d": row.get("return_10d"),
            "return_20d": row.get("return_20d"),
            "return_60d": row.get("return_60d"),
            "return_120d": row.get("return_120d"),
            "distance_120d_high": row.get("distance_120d_high"),
            "leadership_score": row.get("leadership_score"),
            "leadership_reason": "Tushare right-side proxy",
            "right_strength_score": row.get("right_quant_score"),
            "right_strength_reason": "Tushare daily_basic + adjusted-price trend",
            "right_quant_score": row.get("right_quant_score"),
            "right_quant_reason": "point-in-time daily_basic/momentum/liquidity score",
            "right_quant_rank": rank,
            "volatility_20": row.get("volatility_20"),
            "drawdown_60": row.get("drawdown_60"),
            "ma20_slope": row.get("ma20_slope"),
            "ma60_slope": row.get("ma60_slope"),
            "range_21_pct": row.get("range_21_pct"),
            "close_position_21": row.get("close_position_21"),
            "avg_amount_20": row.get("avg_amount_20"),
            "data_status": "traded",
            "tradestatus": 1,
            "is_traded_bar": True,
            "amount_source": "tushare_daily_amount",
            "price_source": "tushare_sqlite_adj_factor_point_in_time",
            "valid_price_bar": True,
            "strategy_part": "right_quant_proxy",
            "candidate_score": score,
            "historical_adjustment_check": "Tushare adj_factor through observation date only",
            "qfq_anchor_date": date.strftime("%Y-%m-%d"),
            "candidate_source": "factor_quant",
            "signal_eligible": True,
            "selected_for_trading": True,
            "candidate_failure_reason": "",
            "value_falsified": False,
            "value_falsification_reason": "",
            "selection_reason": "Tushare SQLite daily_basic point-in-time quant proxy",
            "selection_rank": rank,
            "right_quant_setup": "high_payoff_proxy" if score >= 75 else "watch",
            "allow_left": False,
            "allow_right": True,
        })
    return result


def _write_candidate_snapshots(output: Path, snapshots: dict[str, list[dict]], start_date: str, end_date: str) -> None:
    directory = output / "candidate_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for date, rows in sorted(snapshots.items()):
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=CANDIDATE_SNAPSHOT_COLUMNS)
        else:
            columns = CANDIDATE_SNAPSHOT_COLUMNS + [column for column in frame.columns if column not in CANDIDATE_SNAPSHOT_COLUMNS]
            frame = frame.reindex(columns=columns)
        path = directory / f"candidates_{date}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append({
            "date": date,
            "report_period": date,
            "candidate_count": len(rows),
            "signal_eligible_count": len(rows),
            "mainline_snapshot_date": None,
            "mainline_snapshot_fresh": False,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
            "file": path.name,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "requested_start": start_date,
        "requested_end": end_date,
        "snapshot_count": len(manifest_rows),
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "unsafe_snapshot_count": 0,
        "unsafe_snapshot_sample": [],
        "candidate_pool_formula": "Tushare SQLite daily_basic + adjusted-price right-side proxy",
        "point_in_time_note": (
            "Rows use only trade_date-bounded Tushare daily, adj_factor, daily_basic, "
            "stock_st and suspend_d records. quality_score and earnings_yoy are "
            "proxy compatibility fields, not financial-statement metrics."
        ),
        "industry_point_in_time": True,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "industry_data_source": "tushare sw_member intervals",
        "industry_point_in_time_status": "interval_filtered",
        "snapshots": manifest_rows,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    warmup = max(pd.Timestamp("2022-01-01"), start - pd.Timedelta(days=700))
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.database)
    print("[tushare-sqlite-backtest] loading cached datasets", flush=True)
    names, list_dates = _read_basic(conn)
    st_set = _read_code_date_set(conn, "stock_st", _date_text(warmup), _date_text(end))
    suspend_set = _read_code_date_set(conn, "suspend_d", _date_text(warmup), _date_text(end))
    industry_intervals = _read_industry_intervals(conn)
    daily = _read_dataset(
        conn,
        "daily_kline",
        _date_text(warmup),
        _date_text(end),
        {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "vol",
            "amount": "amount",
        },
    )
    adj = _read_dataset(
        conn,
        "adj_factor",
        _date_text(warmup),
        _date_text(end),
        {"adj_factor": "adj_factor"},
    )[["ts_code", "trade_date", "adj_factor"]]
    basic = _read_dataset(
        conn,
        "daily_basic",
        _date_text(warmup),
        _date_text(end),
        {
            "turnover_rate": "turnover_rate",
            "volume_ratio": "volume_ratio",
            "pe": "pe",
            "pb": "pb",
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
        },
    )[["ts_code", "trade_date", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"]]
    conn.close()
    print(
        "[tushare-sqlite-backtest] loaded rows "
        f"daily={len(daily)} adj={len(adj)} daily_basic={len(basic)}",
        flush=True,
    )

    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="inner").merge(
        basic,
        on=["ts_code", "trade_date"],
        how="left",
    )
    for column in (
        "open", "high", "low", "close", "vol", "amount", "adj_factor",
        "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    for column in ("open", "high", "low", "close"):
        merged[f"{column}_adj"] = merged[column] * merged["adj_factor"]
    merged = merged.dropna(subset=["date", "code", "open", "high", "low", "close", "close_adj"])
    calendar = sorted(
        date.normalize()
        for date in merged.loc[(merged["date"] >= start) & (merged["date"] <= end), "date"].drop_duplicates()
    )

    price_frames: dict[str, pd.DataFrame] = {}
    feature_frames: dict[str, pd.DataFrame] = {}
    candidate_pool: dict[str, list[dict]] = defaultdict(list)
    print("[tushare-sqlite-backtest] building features and candidates", flush=True)
    for code, frame in merged.groupby("code", sort=False):
        frame = _score_frame(frame)
        feature_frames[code] = frame
        price_frames[code] = pd.DataFrame({
            "date": frame["date"].dt.strftime("%Y-%m-%d"),
            "code": code,
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["vol"],
            "amount": frame["amount"],
            "tradestatus": 1,
        }).dropna(subset=["date", "high", "low", "close"])
        for row in frame[(frame["date"] >= start) & (frame["date"] <= end)].itertuples(index=False):
            date = row.date.normalize()
            date_text = date.strftime("%Y-%m-%d")
            if (code, date_text) in st_set or (code, date_text) in suspend_set:
                continue
            mktcap = getattr(row, "total_mv", math.nan)
            mktcap_yi = float(mktcap) / 10000.0 if pd.notna(mktcap) else math.nan
            if not math.isfinite(mktcap_yi) or mktcap_yi < 100.0:
                continue
            score = getattr(row, "right_quant_score", math.nan)
            avg_amount = getattr(row, "avg_amount_20", math.nan)
            if not math.isfinite(float(score)) or float(score) < 68.0:
                continue
            if not math.isfinite(float(avg_amount)) or float(avg_amount) < 200_000_000.0:
                continue
            candidate_pool[date_text].append({
                "code": code,
                "name": names.get(code, code),
                "industry": _industry_for(industry_intervals, code, date),
                "close": float(getattr(row, "close")),
                "mktcap": mktcap_yi,
                "right_quant_score": float(score),
                "trade_basis_score": float(getattr(row, "trade_basis_score", 0.0)),
                "avg_amount_20": float(avg_amount),
                "return_5d": getattr(row, "return_5d", None),
                "return_10d": getattr(row, "return_10d", None),
                "return_20d": getattr(row, "return_20d", None),
                "return_60d": getattr(row, "return_60d", None),
                "return_120d": getattr(row, "return_120d", None),
                "distance_120d_high": getattr(row, "distance_120d_high", None),
                "leadership_score": float(getattr(row, "leadership_score", 0.0)),
                "volatility_20": getattr(row, "volatility_20", None),
                "drawdown_60": getattr(row, "drawdown_60", None),
                "ma20_slope": getattr(row, "ma20_slope", None),
                "ma60_slope": getattr(row, "ma60_slope", None),
                "above_ma20": bool(getattr(row, "above_ma20", False)),
                "near_ma20": bool(getattr(row, "near_ma20", False)),
                "range_21_pct": getattr(row, "range_21_pct", None),
                "close_position_21": getattr(row, "close_position_21", None),
                "volume_ratio_calc": getattr(row, "volume_ratio_calc", None),
            })

    print("[tushare-sqlite-backtest] rebuilding Formula33 phase", flush=True)
    formula = _formula_rows(feature_frames, calendar, list_dates, st_set, suspend_set)
    formula_path = output / "formula33_phase_tushare_sqlite.csv"
    formula.to_csv(formula_path, index=False, encoding="utf-8-sig")
    snapshots = {
        date: _candidate_rows_for_date(pd.Timestamp(date), rows, daily_candidates=args.daily_candidates)
        for date, rows in candidate_pool.items()
        if rows
    }
    _write_candidate_snapshots(output, snapshots, args.start_date, args.end_date)
    print(
        "[tushare-sqlite-backtest] running portfolio engine "
        f"snapshots={len(snapshots)} codes={len({row['code'] for rows in snapshots.values() for row in rows})}",
        flush=True,
    )
    phases = {
        row["date"]: {
            "phase": row["phase"],
            "window_up_streak": int(row.get("window_up_streak") or 0),
            "window_down_streak": int(row.get("window_down_streak") or 0),
        }
        for row in formula.to_dict("records")
    }
    candidate_codes = {
        row["code"]
        for rows in snapshots.values()
        for row in rows
    }
    price_subset = {code: frame for code, frame in price_frames.items() if code in candidate_codes}
    result = run_portfolio_backtest(
        price_subset,
        snapshots,
        phases,
        requested_start=args.start_date,
        end_date=args.end_date,
        max_positions=args.max_positions,
        max_total_held_symbols=args.max_total_held_symbols,
        initial_capital=args.initial_capital,
        signals_effective_next_day=True,
    )
    equity = pd.DataFrame(result.get("equity_curve") or [])
    trades = pd.DataFrame(result.get("trades") or [])
    events = pd.DataFrame(result.get("events") or [])
    trade_events = (
        events[events.get("trade_side").notna()].copy()
        if not events.empty and "trade_side" in events
        else pd.DataFrame()
    )
    if trades.empty and not trade_events.empty:
        trades = trade_events
    equity_path = output / f"portfolio_{args.start_date}_{args.end_date}_equity.csv"
    trades_path = output / f"portfolio_{args.start_date}_{args.end_date}_trades.csv"
    events_path = output / f"portfolio_{args.start_date}_{args.end_date}_events.csv"
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    final_equity = None
    if not equity.empty and "equity" in equity:
        final_equity = float(equity["equity"].iloc[-1])
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "database": str(Path(args.database).resolve()),
        "formula_rows": int(len(formula)),
        "snapshot_count": int(len(snapshots)),
        "candidate_code_count": int(len(candidate_codes)),
        "equity_rows": int(len(equity)),
        "trade_count": int(len(trade_events) if not trade_events.empty else len(trades)),
        "final_equity": final_equity,
        "total_return": None if final_equity is None else final_equity - 1.0,
        "note": (
            "Tushare SQLite research backtest. Candidate quality/growth fields "
            "are compatibility proxies derived from daily_basic/price data, not "
            "financial statement PIT metrics."
        ),
        "outputs": {
            "formula": str(formula_path),
            "candidate_directory": str(output / "candidate_snapshots"),
            "equity": str(equity_path),
            "trades": str(trades_path),
            "events": str(events_path),
        },
    }
    summary_path = output / f"portfolio_{args.start_date}_{args.end_date}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
