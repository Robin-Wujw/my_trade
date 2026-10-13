"""Local input loading. No data refresh or strategy execution occurs here."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from stock_research.core.paths import PATHS
from stock_research.storage import Database, KlineRepository, TushareRepository
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.strategies.fundamental_selection import VALUE_INDUSTRY_RULE_VERSION


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _attach_raw_adjustment_factor(frame, raw_directory, code):
    """Attach raw/qfq factor so raw-price portfolios can handle ex-rights."""
    if Path(raw_directory).name != "akshare_raw":
        return frame
    qfq_path = Path(raw_directory).parent / "akshare" / f"{str(code).replace('.', '_')}.csv"
    try:
        qfq = pd.read_csv(qfq_path, usecols=["date", "close"])
    except (OSError, ValueError):
        return frame
    result = frame.copy()
    qfq["date"] = pd.to_datetime(qfq["date"], errors="coerce")
    qfq["close"] = pd.to_numeric(qfq["close"], errors="coerce")
    raw_dates = pd.to_datetime(result.get("date"), errors="coerce")
    raw_close = pd.to_numeric(result.get("close"), errors="coerce")
    qfq_close = qfq.dropna(subset=["date"]).drop_duplicates("date").set_index("date")["close"]
    aligned_qfq = pd.Series(qfq_close.reindex(raw_dates).to_numpy(), index=result.index)
    factor = raw_close.div(aligned_qfq.where(aligned_qfq > 0))
    result["raw_to_qfq_factor"] = factor.where(factor > 0)
    return result



def load_candidate_snapshots(directory, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"candidate manifest cannot be read: {manifest_path}"
            ) from exc
        if manifest.get("version") != SNAPSHOT_VERSION:
            raise RuntimeError(
                "candidate manifest version is incompatible with the current strategy: "
                f"expected={SNAPSHOT_VERSION} actual={manifest.get('version')} "
                f"manifest={manifest_path}"
            )
        if manifest.get("value_industry_rule_version") != VALUE_INDUSTRY_RULE_VERSION:
            raise RuntimeError(
                "candidate manifest value-industry rule version is missing or stale: "
                f"expected={VALUE_INDUSTRY_RULE_VERSION} "
                f"actual={manifest.get('value_industry_rule_version')} "
                f"manifest={manifest_path}"
            )
        rows = manifest.get("snapshots")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"candidate manifest snapshots must be a list: {manifest_path}"
            )
        dated_paths = []
        seen_dates = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"candidate manifest contains an invalid snapshot row: {manifest_path}"
                )
            date = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(date):
                raise RuntimeError(
                    f"candidate manifest contains an invalid snapshot date: {manifest_path}"
                )
            date = date.normalize()
            if date in seen_dates:
                raise RuntimeError(
                    f"candidate manifest contains duplicate date {date:%Y-%m-%d}: "
                    f"{manifest_path}"
                )
            seen_dates.add(date)
            filename = str(row.get("file") or f"candidates_{date:%Y-%m-%d}.csv")
            if Path(filename).name != filename:
                raise RuntimeError(
                    f"candidate manifest snapshot file must be a basename: {filename}"
                )
            dated_paths.append((
                date,
                directory / filename,
                row.get("candidate_count"),
                str(row.get("sha256") or "").strip().lower() or None,
            ))
        listed_paths = {path.name for _, path, _, _ in dated_paths}
        unlisted = []
        for path in sorted(directory.glob("candidates_*.csv")):
            date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
            if (
                pd.notna(date)
                and start <= date.normalize() <= end
                and path.name not in listed_paths
            ):
                unlisted.append(path.name)
        if unlisted:
            raise RuntimeError(
                "candidate directory contains snapshots outside manifest in requested "
                f"range: count={len(unlisted)} sample={unlisted[:5]}"
            )
    else:
        dated_paths = []
        for path in sorted(directory.glob("candidates_*.csv")):
            date = pd.to_datetime(path.stem.removeprefix("candidates_"), errors="coerce")
            if pd.notna(date):
                dated_paths.append((date.normalize(), path, None, None))
    snapshots = {}
    for date, path, expected_count, expected_sha256 in dated_paths:
        if not start <= date <= end:
            continue
        if not path.exists():
            raise RuntimeError(f"candidate snapshot listed by manifest is missing: {path}")
        if expected_sha256 is not None:
            actual_sha256 = file_sha256(path).lower()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "candidate snapshot SHA256 differs from manifest: "
                    f"{path} expected={expected_sha256} actual={actual_sha256}"
                )
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except EmptyDataError:
            frame = pd.DataFrame()
        if expected_count is not None and len(frame) != int(expected_count):
            raise RuntimeError(
                f"candidate snapshot row count differs from manifest: {path} "
                f"expected={int(expected_count)} actual={len(frame)}"
            )
        snapshots[date.strftime("%Y-%m-%d")] = frame.to_dict("records")
    return snapshots



def load_price_frames(
    codes,
    directory,
    *,
    start_date=None,
    end_date=None,
    source="akshare",
    prefer_database=True,
    database_source="auto",
    attach_daily_limits=True,
    database=None,
):
    """Load OHLC frames for execution and valuation.

    SQLite is the fast authority for the normal qfq cache.  Raw historical
    execution tests must be able to bypass it, otherwise a raw CSV directory is
    silently overwritten by qfq database rows.
    """
    normalized_codes = sorted({str(code) for code in codes})
    frames = {}
    requested_database_source = str(database_source or "auto")
    if prefer_database and (database is not None or PATHS.database.is_file()) and normalized_codes:
        database = database or Database(PATHS.database)
        if requested_database_source in {"auto", "miniqmt", "akshare"}:
            repository = KlineRepository(database)
            loaded = repository.load_stock_klines(
                source,
                normalized_codes,
                start_date=str(start_date or "1900-01-01"),
                end_date=str(end_date or "2999-12-31"),
                max_qfq_anchor_date=None if end_date is None else str(end_date),
            )
            if not loaded.empty:
                frames.update({
                    str(code): group.drop(columns=["code"]).reset_index(drop=True)
                    for code, group in loaded.groupby("code", sort=False)
                })
                for frame in frames.values():
                    if "raw_close" not in frame and "raw_to_qfq_factor" not in frame:
                        frame.attrs["execution_price_basis"] = "qfq_only"

        if requested_database_source in {"auto", "tushare"}:
            missing_codes = [code for code in normalized_codes if code not in frames]
            if missing_codes:
                loaded = TushareRepository(database).load_daily_kline_frames(
                    missing_codes,
                    start_date=str(start_date or "1900-01-01"),
                    end_date=str(end_date or "2999-12-31"),
                )
                if not loaded.empty:
                    frames.update({
                        str(code): group.drop(columns=["code"]).reset_index(drop=True)
                        for code, group in loaded.groupby("code", sort=False)
                    })
    for code in (item for item in normalized_codes if item not in frames):
        stem = str(code).replace(".", "_")
        csv_path = Path(directory) / f"{stem}.csv"
        pickle_path = Path(directory) / f"{stem}.pkl"
        try:
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
            elif pickle_path.exists():
                frame = pd.read_pickle(pickle_path)
            else:
                continue
            if start_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(start_date)]
            if end_date and "date" in frame:
                frame = frame[pd.to_datetime(frame["date"], errors="coerce") <= pd.Timestamp(end_date)]
            frame = _attach_raw_adjustment_factor(frame, directory, code)
            if Path(directory).name == "akshare" and "raw_close" not in frame:
                frame.attrs["execution_price_basis"] = "qfq_only"
            frames[str(code)] = frame
        except (OSError, ValueError):
            continue
    if (
        attach_daily_limits
        and frames
        and start_date is not None
        and end_date is not None
        and (database is not None or PATHS.database.is_file())
    ):
        limits = TushareRepository(database or Database(PATHS.database)).load_dataset_for_codes(
            "stk_limit",
            normalized_codes,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        if not limits.empty and {"date", "code"}.issubset(limits.columns):
            for column in ("up_limit", "down_limit"):
                if column not in limits:
                    limits[column] = pd.NA
                limits[column] = pd.to_numeric(limits[column], errors="coerce")
            limits["date"] = pd.to_datetime(limits["date"], errors="coerce")
            limits = limits.dropna(subset=["date", "code"]).drop_duplicates(
                ["date", "code"], keep="last",
            )
            for code, frame in frames.items():
                code_limits = limits[limits["code"].astype(str).eq(str(code))]
                if code_limits.empty:
                    continue
                by_date = code_limits.set_index("date")
                dates = pd.to_datetime(frame.get("date"), errors="coerce")
                enriched = frame.copy()
                for column in ("up_limit", "down_limit"):
                    loaded_values = pd.Series(
                        by_date[column].reindex(dates).to_numpy(),
                        index=enriched.index,
                    )
                    existing_values = (
                        pd.to_numeric(enriched[column], errors="coerce")
                        if column in enriched
                        else pd.Series(
                            float("nan"), index=enriched.index, dtype="float64",
                        )
                    )
                    enriched[column] = loaded_values.combine_first(existing_values)
                verified_dates = by_date.index[
                    by_date["up_limit"].gt(by_date["down_limit"]) & by_date["down_limit"].gt(0)]
                enriched["daily_limit_source"] = enriched.get("daily_limit_source", "unverified_file")
                enriched.loc[dates.isin(verified_dates), "daily_limit_source"] = "local_tushare_stk_limit"
                frames[code] = enriched
    incompatible = [code for code, frame in frames.items()
                    if frame.attrs.get("execution_price_basis") == "qfq_only"]
    if incompatible:
        raise ValueError("Raw execution prices are required; qfq-only prices cannot be combined with raw limits: "
                         + ", ".join(incompatible[:10]))
    return frames



def load_corporate_actions(codes, *, start_date, end_date):
    if not PATHS.database.is_file():
        return {}
    database = Database(PATHS.database)
    try:
        return TushareRepository(database).load_dividend_actions(
            codes,
            start_date=str(start_date),
            end_date=str(end_date),
        )
    except Exception as exc:
        print(f"[portfolio-backtest] skip local dividend actions: {exc}")
        return {}



def load_security_end_dates(codes):
    """Load known delisting dates for lifecycle validation and terminal write-offs."""
    if not PATHS.database.is_file():
        return {}
    database = Database(PATHS.database)
    try:
        stock_basic = TushareRepository(database).load_stock_basic_universe()
    except Exception as exc:
        print(f"[portfolio-backtest] skip local security end dates: {exc}")
        return {}
    requested = {str(code) for code in codes}
    rows = stock_basic[
        stock_basic["code"].astype(str).isin(requested)
        & stock_basic["delist_date"].notna()
    ]
    return {
        str(row["code"]): pd.Timestamp(row["delist_date"]).normalize()
        for _, row in rows.iterrows()
    }



def load_security_suspension_dates(codes, *, start_date, end_date):
    """Load explicit Tushare suspension dates; network failures are never inferred."""
    if not PATHS.database.is_file():
        return {}
    database = Database(PATHS.database)
    try:
        frame = TushareRepository(database).load_dataset_for_codes(
            "suspend_d",
            codes,
            start_date=str(start_date),
            end_date=str(end_date),
        )
    except Exception as exc:
        print(f"[portfolio-backtest] skip local suspension dates: {exc}")
        return {}
    if frame.empty:
        return {}
    return {
        str(code): {
            pd.Timestamp(value).normalize()
            for value in group["date"].dropna()
        }
        for code, group in frame.groupby("code", sort=False)
    }



def first_candidate_dates(snapshots):
    result = {}
    for date_text, rows in sorted(snapshots.items()):
        date = pd.Timestamp(date_text).normalize()
        for row in rows:
            code = str(row.get("code") or "").strip()
            if code and code not in result:
                result[code] = date
    return result

