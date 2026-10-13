"""Attach the pre-registered market-volatility entry sizing overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.research.quantsplaybook_chunked import _load_cached_panel
from stock_research.core.paths import PATHS


MARKET_VOLATILITY_WINDOW = 20
DEFAULT_SOURCE_CANDIDATES = (
    PATHS.runtime_root / "backtests"
    / "one_network_price_only_low_residual_volatility_liquidity_30m_experiment"
    / "hybrid" / "candidates"
    / "hybrid_low_residual_volatility_liquidity_30m_top50"
)
DEFAULT_PANEL_ROOT = (
    PATHS.runtime_root / "backtests"
    / "one_network_price_only_low_residual_volatility_experiment"
    / "research_panels"
)
DEFAULT_PRICE_DIRECTORY = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
    / "prepared_prices"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PATHS.runtime_root / "backtests" / "right_market_volatility_20_train"
    / "hybrid" / "candidates"
    / "hybrid_low_residual_volatility_liquidity_30m_top50_market_volatility_20"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def calculate_market_volatility_scaling(
    market_returns: pd.Series,
    *,
    window: int = MARKET_VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Use current RV20 versus its strictly prior expanding mean."""
    lookback = int(window)
    if lookback <= 1:
        raise ValueError("market volatility window must exceed one session")
    if not isinstance(market_returns.index, pd.DatetimeIndex):
        raise ValueError("market returns require a DatetimeIndex")
    if market_returns.index.has_duplicates:
        raise ValueError("market returns contain duplicate dates")
    returns = pd.to_numeric(market_returns, errors="coerce").sort_index()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    if returns.isna().any():
        missing = returns.index[returns.isna()]
        raise RuntimeError(
            "market return coverage is incomplete: "
            f"sample={[value.strftime('%Y-%m-%d') for value in missing[:5]]}"
        )

    realized_variance = returns.pow(2).rolling(
        lookback, min_periods=lookback,
    ).mean()
    historical_baseline = realized_variance.expanding(
        min_periods=1,
    ).mean().shift(1)
    multiplier = historical_baseline.div(realized_variance).clip(upper=1.0)
    warmup = realized_variance.isna() | historical_baseline.isna()
    multiplier = multiplier.where(~warmup, 1.0)
    zero_variance = realized_variance.eq(0.0)
    multiplier = multiplier.where(~zero_variance, 1.0)
    invalid = (
        multiplier.isna()
        | ~np.isfinite(multiplier)
        | multiplier.le(0.0)
        | multiplier.gt(1.0)
    )
    if invalid.any():
        dates = multiplier.index[invalid]
        raise RuntimeError(
            "market volatility multiplier is invalid: "
            f"sample={[value.strftime('%Y-%m-%d') for value in dates[:5]]}"
        )
    return pd.DataFrame({
        "market_return": returns,
        "realized_variance_20": realized_variance,
        "historical_realized_variance_mean": historical_baseline,
        "right_market_volatility_multiplier": multiplier.astype(float),
        "warmup_fallback": warmup.astype(bool),
    })


def load_equal_weight_market_returns(
    *,
    panel_root: str | Path,
    price_directory: str | Path,
    dates,
) -> tuple[pd.Series, dict]:
    """Build PIT equal-weight market returns over each saved panel universe."""
    panels = Path(panel_root)
    prices = Path(price_directory)
    target_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(value).normalize() for value in dates}),
        name="date",
    )
    if target_dates.empty:
        raise ValueError("market return dates are empty")
    if not panels.is_dir() or not prices.is_dir():
        raise RuntimeError("market volatility source directory is missing")

    pair_frames = []
    source_panels = []
    for year in sorted(set(target_dates.year)):
        panel_path = panels / f"factor_panel_{year}.pkl"
        if not panel_path.is_file():
            raise RuntimeError(f"market universe panel is missing: {panel_path}")
        panel = _load_cached_panel(panel_path)
        pairs = panel.index.to_frame(index=False)[["date", "code"]]
        pairs["date"] = pd.to_datetime(pairs["date"], errors="coerce").dt.normalize()
        pairs["code"] = pairs["code"].astype(str)
        pairs = pairs[pairs["date"].isin(target_dates)].dropna()
        pair_frames.append(pairs)
        source_panels.append({
            "year": int(year),
            "path": str(panel_path),
            "sha256": _sha256(panel_path),
            "eligible_rows": int(len(pairs)),
        })
    eligible = pd.concat(pair_frames, ignore_index=True)
    eligible = eligible.drop_duplicates(["date", "code"])
    if eligible.empty:
        raise RuntimeError("market universe panels have no requested rows")
    eligible_by_code = {
        code: pd.DatetimeIndex(group["date"].drop_duplicates())
        for code, group in eligible.groupby("code", sort=True)
    }

    sums = np.zeros(len(target_dates), dtype=float)
    counts = np.zeros(len(target_dates), dtype=np.int64)
    aggregate_hash = hashlib.sha256()
    price_file_count = 0
    available_rows = 0
    for code, eligible_dates in eligible_by_code.items():
        path = prices / f"{code.replace('.', '_')}.pkl"
        if not path.is_file():
            raise RuntimeError(f"market return price frame is missing: {path}")
        aggregate_hash.update(code.encode("ascii"))
        aggregate_hash.update(b"\0")
        aggregate_hash.update(bytes.fromhex(_sha256(path)))
        price_file_count += 1
        frame = pd.read_pickle(path)
        if not {"date", "close"}.issubset(frame.columns):
            raise RuntimeError(f"market return price frame lacks date/close: {path}")
        work = pd.DataFrame({
            "date": pd.to_datetime(frame["date"], errors="coerce").dt.normalize(),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
        })
        work["close"] = work["close"].where(work["close"].gt(0.0))
        work = work.dropna().drop_duplicates("date", keep="last").sort_values("date")
        returns = pd.Series(
            work["close"].pct_change(fill_method=None).to_numpy(),
            index=work["date"],
            dtype=float,
        )
        values = returns.reindex(target_dates).to_numpy(dtype=float)
        member = target_dates.isin(eligible_dates)
        valid = member & np.isfinite(values)
        sums[valid] += values[valid]
        counts[valid] += 1
        available_rows += int(valid.sum())

    if (counts == 0).any():
        missing = target_dates[counts == 0]
        raise RuntimeError(
            "market return has no valid constituents: "
            f"sample={[value.strftime('%Y-%m-%d') for value in missing[:5]]}"
        )
    market_returns = pd.Series(
        sums / counts,
        index=target_dates,
        name="market_return",
        dtype=float,
    )
    return market_returns, {
        "universe_rule": "saved PIT low-residual-volatility panel date/code rows",
        "return_rule": "equal mean of valid adjusted-close returns on each date",
        "missing_return_rule": "skip invalid stock return; fail if date has none",
        "eligible_rows": int(len(eligible)),
        "available_return_rows": int(available_rows),
        "coverage_ratio": float(available_rows / len(eligible)),
        "minimum_constituent_count": int(counts.min()),
        "maximum_constituent_count": int(counts.max()),
        "price_file_count": int(price_file_count),
        "price_files_aggregate_sha256": aggregate_hash.hexdigest(),
        "source_panels": source_panels,
    }


def build_market_volatility_candidates(
    *,
    source_candidate_directory: str | Path,
    panel_root: str | Path,
    price_directory: str | Path,
    output_directory: str | Path,
    start_date: str,
    end_date: str,
) -> Path:
    source = Path(source_candidate_directory)
    output = Path(output_directory)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    snapshots = [
        row for row in manifest.get("snapshots", [])
        if start <= pd.Timestamp(row.get("date")).normalize() <= end
    ]
    if not snapshots:
        raise RuntimeError("source candidate range is empty")
    dates = [str(row["date"]) for row in snapshots]
    market_returns, coverage = load_equal_weight_market_returns(
        panel_root=panel_root,
        price_directory=price_directory,
        dates=dates,
    )
    scaling = calculate_market_volatility_scaling(market_returns)
    output.mkdir(parents=True, exist_ok=True)
    output_snapshots = []
    right_rows = 0
    for item in snapshots:
        date = str(item["date"])
        source_path = source / str(item["file"])
        expected = str(item.get("sha256") or "")
        if expected and _sha256(source_path) != expected:
            raise RuntimeError(f"source candidate hash mismatch: {source_path}")
        frame = pd.read_csv(source_path, dtype={"code": str}).copy()
        allow_right = _truthy_series(
            frame.get("allow_right", pd.Series(False, index=frame.index))
        )
        values = scaling.loc[pd.Timestamp(date)]
        frame.loc[allow_right, "right_market_volatility_multiplier"] = float(
            values["right_market_volatility_multiplier"]
        )
        frame.loc[allow_right, "right_market_realized_variance_20"] = values[
            "realized_variance_20"
        ]
        frame.loc[
            allow_right, "right_market_historical_variance_mean"
        ] = values["historical_realized_variance_mean"]
        output_path = output / source_path.name
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        right_rows += int(allow_right.sum())
        copied = dict(item)
        copied.update({
            "file": output_path.name,
            "sha256": _sha256(output_path),
            "candidate_count": int(len(frame)),
            "market_volatility_multiplier": float(
                values["right_market_volatility_multiplier"]
            ),
        })
        output_snapshots.append(copied)

    daily_path = output / "market_volatility_daily.csv"
    scaling.reset_index().to_csv(daily_path, index=False, encoding="utf-8-sig")
    result_manifest = {
        **manifest,
        "snapshot_count": len(output_snapshots),
        "requested_start": start_date,
        "requested_end": end_date,
        "start_date": dates[0],
        "end_date": dates[-1],
        "selection_profile": output.name,
        "source_candidate_directory": str(source),
        "source_candidate_manifest_sha256": _sha256(manifest_path),
        "future_return_used_for_candidate_selection": False,
        "full_history_previously_observed": True,
        "research_only": True,
        "right_market_volatility_sizing": {
            "method": "entry_only_inverse_market_realized_variance",
            "window_sessions": MARKET_VOLATILITY_WINDOW,
            "formula": "min(1, prior_expanding_mean(RV20) / current_RV20)",
            "warmup_rule": "multiplier=1 until RV20 and prior baseline exist",
            "leverage_rule": "multiplier is greater than 0 and at most 1",
            "application_rule": "ordinary right new-campaign first entries only",
            "rebalancing": False,
            "price_directory": str(Path(price_directory)),
            "daily_file": daily_path.name,
            "daily_file_sha256": _sha256(daily_path),
            "right_candidate_rows": int(right_rows),
            "warmup_date_count": int(scaling["warmup_fallback"].sum()),
            "scaled_date_count": int(
                scaling["right_market_volatility_multiplier"].lt(1.0).sum()
            ),
            "multiplier_min": float(
                scaling["right_market_volatility_multiplier"].min()
            ),
            "multiplier_mean": float(
                scaling["right_market_volatility_multiplier"].mean()
            ),
            "multiplier_max": float(
                scaling["right_market_volatility_multiplier"].max()
            ),
            "point_in_time_rule": (
                "current RV20 uses returns through the observation close; "
                "baseline uses strictly prior RV20 values"
            ),
            "literature": {
                "title": "Volatility-Managed Portfolios",
                "authors": "Moreira and Muir",
                "doi": "10.1111/jofi.12513",
                "adaptation": "entry-only, no leverage, no rebalancing",
            },
            "coverage": coverage,
        },
        "snapshots": output_snapshots,
    }
    (output / "manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-candidate-directory", default=str(DEFAULT_SOURCE_CANDIDATES),
    )
    parser.add_argument("--panel-root", default=str(DEFAULT_PANEL_ROOT))
    parser.add_argument(
        "--price-directory", default=str(DEFAULT_PRICE_DIRECTORY),
    )
    parser.add_argument(
        "--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2023-12-29")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = build_market_volatility_candidates(
        source_candidate_directory=args.source_candidate_directory,
        panel_root=args.panel_root,
        price_directory=args.price_directory,
        output_directory=args.output_directory,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
