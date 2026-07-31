"""Backfill historical ST and suspension evidence into the unified cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.api.retry import call_with_backoff, is_transient_error
from stock_research.storage import Database, TushareRepository
from stock_research.storage.tushare_repository import normalize_tushare_code


DEFAULT_CANDIDATE_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
    / "candidates"
)
DEFAULT_PREPARED_PRICES = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
    / "prepared_prices"
)
DEFAULT_FORMULA_HISTORY = (
    PATHS.runtime_root / "backtests"
    / "formula33_tushare_2021_to_20260721.csv"
)
DEFAULT_PROVIDER_CACHE = (
    PATHS.tmp / "baostock_risk_history_2021.pkl"
)
BAOSTOCK_CODE_PATTERN = re.compile(r"^(?:sh|sz)\.\d{6}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest_verified_candidate_codes(
    candidate_root: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> set[str]:
    root = Path(candidate_root)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    codes: set[str] = set()
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"no candidate manifests found: {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("snapshots") or []:
            date = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(date) or not start <= date <= end:
                continue
            path = manifest_path.parent / str(row["file"])
            if not path.is_file() or _sha256(path) != str(row["sha256"]):
                raise RuntimeError(f"candidate snapshot fingerprint mismatch: {path}")
            frame = pd.read_csv(path, usecols=["code"], dtype={"code": str})
            codes.update(frame["code"].dropna().astype(str).str.strip())
    return {code for code in codes if code}


def required_suspension_dates(
    market_dates,
    *,
    last_trade_date,
    end_date,
    security_end_date=None,
) -> set[pd.Timestamp]:
    last = pd.Timestamp(last_trade_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    security_end = pd.to_datetime(security_end_date, errors="coerce")
    return {
        pd.Timestamp(value).normalize()
        for value in market_dates
        if last < pd.Timestamp(value).normalize() <= end
        and (
            pd.isna(security_end)
            or pd.Timestamp(value).normalize() < security_end.normalize()
        )
    }


def _terminal_dates(
    prepared_price_cache: Path,
    codes: set[str],
    *,
    start_date: str,
    end_date: str,
) -> dict[str, pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    result = {}
    for code in sorted(codes):
        path = prepared_price_cache / f"{code.replace('.', '_')}.pkl"
        if not path.is_file():
            raise RuntimeError(f"prepared price frame is missing: {path}")
        frame = pd.read_pickle(path)
        dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
        dates = dates[(dates >= start) & (dates <= end)]
        if dates.empty:
            raise RuntimeError(f"prepared price frame has no requested bars: {code}")
        result[code] = dates.max().normalize()
    return result


def _query_baostock(codes: set[str], *, start_date: str, end_date: str):
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    stock_st_rows = []
    suspension_rows = []
    try:
        for index, code in enumerate(
            sorted(code for code in codes if BAOSTOCK_CODE_PATTERN.fullmatch(code)),
            1,
        ):
            result = bs.query_history_k_data_plus(
                code,
                "date,code,tradestatus,isST",
                start_date,
                end_date,
                frequency="d",
                adjustflag="3",
            )
            if result.error_code != "0":
                raise RuntimeError(
                    f"BaoStock history failed code={code}: {result.error_msg}",
                )
            while result.next():
                date, _, trade_status, is_st = result.get_row_data()
                common = {
                    "ts_code": normalize_tushare_code(code),
                    "trade_date": date.replace("-", ""),
                }
                if is_st == "1":
                    stock_st_rows.append({
                        **common,
                        "type": "ST",
                        "source_evidence": "baostock_isST",
                    })
                if trade_status == "0":
                    suspension_rows.append({
                        **common,
                        "suspend_type": "S",
                        "suspend_timing": "D",
                        "source_evidence": "baostock_tradestatus_0",
                    })
            if index % 500 == 0:
                print(f"[baostock-risk] queried {index} symbols", flush=True)
    finally:
        bs.logout()
    return pd.DataFrame(stock_st_rows), pd.DataFrame(suspension_rows)


def _query_beijing_confirmed_gaps(
    codes: set[str],
    terminal_dates: dict[str, pd.Timestamp],
    market_dates: list[pd.Timestamp],
    security_end_dates: dict[str, pd.Timestamp],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    import akshare as ak

    rows = []
    end = pd.Timestamp(end_date).normalize()
    for code in sorted(code for code in codes if code.startswith("bj.")):
        last = terminal_dates[code]
        required = required_suspension_dates(
            market_dates,
            last_trade_date=last,
            end_date=end,
            security_end_date=security_end_dates.get(code),
        )
        if not required:
            continue
        remote = call_with_backoff(
            lambda: ak.stock_zh_a_daily(
                symbol=f"bj{code.split('.', 1)[1]}",
                start_date=(last - pd.Timedelta(days=45)).strftime("%Y%m%d"),
                end_date=(end + pd.Timedelta(days=45)).strftime("%Y%m%d"),
                adjust="",
            ),
            f"AKShare Beijing suspension evidence {code}",
            retries=4,
            retry_delay=2.0,
            retry_if=is_transient_error,
        )
        if remote.empty or "date" not in remote:
            raise RuntimeError(f"AKShare returned no Beijing history: {code}")
        remote_dates = set(
            pd.to_datetime(remote["date"], errors="coerce").dropna().dt.normalize()
        )
        if last not in remote_dates:
            raise RuntimeError(
                f"AKShare does not confirm the local terminal bar: {code} {last:%Y-%m-%d}",
            )
        unexpected = sorted(required & remote_dates)
        if unexpected:
            raise RuntimeError(
                f"prepared cache misses AKShare-traded Beijing bars: "
                f"{code} {unexpected[:5]}",
            )
        if not any(value > end for value in remote_dates):
            security_end = security_end_dates.get(code)
            if security_end is None or pd.Timestamp(security_end) > end:
                raise RuntimeError(
                    f"AKShare has no post-gap resumption evidence: {code}",
                )
        for date in sorted(required):
            rows.append({
                "ts_code": normalize_tushare_code(code),
                "trade_date": date.strftime("%Y%m%d"),
                "suspend_type": "S",
                "suspend_timing": "D",
                "source_evidence": (
                    "akshare_sina_confirmed_gap_between_trading_bars"
                ),
            })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument(
        "--prepared-price-cache", default=str(DEFAULT_PREPARED_PRICES),
    )
    parser.add_argument("--formula-history", default=str(DEFAULT_FORMULA_HISTORY))
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument(
        "--provider-cache", default=str(DEFAULT_PROVIDER_CACHE),
    )
    parser.add_argument(
        "--all-stock-basic",
        action="store_true",
        help="query every Shanghai/Shenzhen symbol in the local security master",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    candidate_codes = load_manifest_verified_candidate_codes(
        args.candidate_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    formula = pd.read_csv(args.formula_history)
    market_dates = sorted({
        pd.Timestamp(value).normalize()
        for value in pd.to_datetime(formula.get("date"), errors="coerce").dropna()
        if pd.Timestamp(args.start_date) <= value <= pd.Timestamp(args.end_date)
    })
    if not market_dates:
        raise RuntimeError("formula history has no market dates in the requested range")
    terminal_dates = _terminal_dates(
        Path(args.prepared_price_cache),
        candidate_codes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    repository = TushareRepository(Database(args.database))
    basic = repository.load_stock_basic_universe()
    security_end_dates = {
        str(row["code"]): pd.Timestamp(row["delist_date"]).normalize()
        for _, row in basic[basic["delist_date"].notna()].iterrows()
    }
    query_codes = set(candidate_codes)
    if args.all_stock_basic:
        query_codes.update(
            code for code in basic["code"].dropna().astype(str)
            if BAOSTOCK_CODE_PATTERN.fullmatch(code)
        )

    provider_cache = Path(args.provider_cache)
    query_codes_sha256 = hashlib.sha256(
        "\n".join(sorted(query_codes)).encode("utf-8"),
    ).hexdigest()
    if provider_cache.is_file():
        cached = pd.read_pickle(provider_cache)
        if (
            str(cached.get("start_date")) != str(args.start_date)
            or str(cached.get("end_date")) != str(args.end_date)
            or cached.get("query_codes_sha256") != query_codes_sha256
        ):
            raise RuntimeError(
                f"provider cache does not match requested symbols/range: "
                f"{provider_cache}",
            )
        stock_st = cached["stock_st"]
        suspensions = cached["suspensions"]
        print(f"[baostock-risk] reuse provider cache {provider_cache}", flush=True)
    else:
        stock_st, suspensions = _query_baostock(
            query_codes,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        provider_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "symbol_count": len(query_codes),
                "query_codes_sha256": query_codes_sha256,
                "stock_st": stock_st,
                "suspensions": suspensions,
            },
            provider_cache,
        )
    beijing = _query_beijing_confirmed_gaps(
        candidate_codes,
        terminal_dates,
        market_dates,
        security_end_dates,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not beijing.empty:
        suspensions = pd.concat([suspensions, beijing], ignore_index=True)

    suspension_dates = {
        normalize_tushare_code(code): set(
            pd.to_datetime(group["trade_date"], errors="coerce").dropna().dt.normalize()
        )
        for code, group in suspensions.groupby("ts_code", sort=False)
    } if not suspensions.empty else {}
    failures = {}
    for code, last in terminal_dates.items():
        required = required_suspension_dates(
            market_dates,
            last_trade_date=last,
            end_date=args.end_date,
            security_end_date=security_end_dates.get(code),
        )
        if not required:
            continue
        missing = sorted(
            required - suspension_dates.get(normalize_tushare_code(code), set()),
        )
        if missing:
            failures[code] = [value.strftime("%Y-%m-%d") for value in missing[:10]]
    if failures:
        raise RuntimeError(
            f"cross-provider suspension coverage is incomplete: {failures}",
        )

    stock_st_count = repository.upsert_dataset(
        "stock_st",
        stock_st,
        source="baostock/history",
        key_columns=("ts_code", "trade_date"),
        params={"start_date": args.start_date, "end_date": args.end_date},
    )
    suspension_count = repository.upsert_dataset(
        "suspend_d",
        suspensions,
        source="baostock+akshare/history",
        key_columns=("ts_code", "trade_date"),
        params={"start_date": args.start_date, "end_date": args.end_date},
    )
    print(
        f"risk history backfill complete symbols={len(query_codes)} "
        f"stock_st_rows={stock_st_count} suspension_rows={suspension_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
