"""Audit point-in-time selection snapshots with forward return labels.

The audit deliberately stays separate from the execution engine.  It labels a
frozen candidate snapshot with close-to-close returns and reports ranking
quality, hit rate, cross-sectional excess return, and concentration.  Labels
are never read by candidate generation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (1, 5, 10, 20, 60)
DEFAULT_BUCKETS = (10, 20, 50, 100)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _read_price(path: Path) -> pd.Series:
    if path.suffix.lower() == ".pkl":
        frame = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    return price_series(frame, source_label=str(path))


def price_series(frame: pd.DataFrame, *, source_label="in-memory frame") -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or "date" not in frame:
        raise ValueError(f"price file has no date column: {source_label}")
    close_column = "close"
    # ``close`` is the prepared qfq series in this repository.  Prefer an
    # explicit signal close only when the source does not provide close.
    if close_column not in frame and "signal_close" in frame:
        close_column = "signal_close"
    if close_column not in frame:
        raise ValueError(f"price file has no close column: {source_label}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame[close_column] = pd.to_numeric(frame[close_column], errors="coerce")
    valid = frame["date"].notna() & frame[close_column].gt(0)
    if "tradestatus" in frame:
        status = frame["tradestatus"].astype(str).str.strip()
        valid &= status.isin({"1", "1.0", "True", "true"})
    series = (
        frame.loc[valid, ["date", close_column]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .set_index("date")[close_column]
    )
    return series.astype(float)


def load_price_series(price_directory: str | Path, codes: set[str]) -> tuple[dict[str, pd.Series], list[str]]:
    """Load only prices needed by the manifest and return missing codes."""
    directory = Path(price_directory)
    prices: dict[str, pd.Series] = {}
    missing: list[str] = []
    for code in sorted({str(value) for value in codes if str(value).strip()}):
        stem = code.replace(".", "_")
        candidates = [directory / f"{stem}{suffix}" for suffix in (".pkl", ".csv", ".parquet")]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            missing.append(code)
            continue
        try:
            prices[code] = _read_price(path)
        except (OSError, ValueError, TypeError):
            missing.append(code)
    return prices, missing


def _forward_return(series: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    """Return the h-th available traded bar after the observation bar."""
    if series.empty or date not in series.index:
        return np.nan
    position = int(series.index.get_loc(date))
    target = position + int(horizon)
    if target >= len(series):
        return np.nan
    base = float(series.iloc[position])
    future = float(series.iloc[target])
    return future / base - 1.0 if base > 0 else np.nan


def _candidate_group(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("candidate_source", pd.Series(index=frame.index, dtype=object)).copy()
    if "playbook_factor" in frame:
        source = source.fillna(frame["playbook_factor"])
    if "selection_profile" in frame:
        source = source.fillna(frame["selection_profile"])
    source = source.astype(str).str.strip()
    source = source.where(~source.isin({"", "nan", "None"}), "unknown")
    return source


def load_manifest_candidates(
    candidate_directory: str | Path,
    *,
    include_ineligible: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read only files listed by manifest and attach stable daily ranks."""
    directory = Path(candidate_directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest.get("snapshots") or []:
        snapshot_date = pd.Timestamp(item["date"]).normalize()
        if start_date and snapshot_date < pd.Timestamp(start_date).normalize():
            continue
        if end_date and snapshot_date > pd.Timestamp(end_date).normalize():
            continue
        path = directory / str(item["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if "date" not in frame:
            frame["date"] = snapshot_date
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["code"] = frame["code"].astype(str)
        if not include_ineligible and "signal_eligible" in frame:
            frame = frame[frame["signal_eligible"].map(_as_bool)]
        rows.append(frame)
    if not rows:
        raise ValueError("manifest selected no candidate snapshots")
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates.dropna(subset=["date", "code"]).copy()
    candidates["candidate_group"] = _candidate_group(candidates)
    score_columns = ("candidate_score", "playbook_factor_score", "right_quant_score", "selection_rank")
    score = pd.Series(np.nan, index=candidates.index, dtype=float)
    for column in score_columns:
        if column in candidates:
            score = score.fillna(pd.to_numeric(candidates[column], errors="coerce"))
    candidates["audit_score"] = score
    candidates = candidates.sort_values(
        ["date", "candidate_group", "audit_score", "code"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    candidates["audit_rank"] = candidates.groupby(
        ["date", "candidate_group"], sort=False,
    )["audit_score"].rank(method="first", ascending=False)
    return candidates.reset_index(drop=True), manifest


def attach_forward_returns(
    candidates: pd.DataFrame,
    prices: dict[str, pd.Series],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    result = candidates.copy()
    for horizon in horizons:
        result[f"forward_{horizon}d"] = [
            _forward_return(prices.get(str(code), pd.Series(dtype=float)), date, horizon)
            for code, date in zip(result["code"], result["date"])
        ]
        if benchmark is not None:
            result[f"benchmark_forward_{horizon}d"] = [
                _forward_return(benchmark, date, horizon)
                for date in result["date"]
            ]
            result[f"excess_{horizon}d"] = (
                result[f"forward_{horizon}d"] - result[f"benchmark_forward_{horizon}d"]
            )
    return result


def _spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 3 or frame.iloc[:, 1].nunique() < 3:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def summarize_audit(
    observations: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """Aggregate overall and annual cross-sectional ranking diagnostics."""
    rows = []
    for (group, year), daily_group in observations.assign(
        year=observations["date"].dt.year,
    ).groupby(["candidate_group", "year"], sort=True):
        for horizon in horizons:
            return_column = f"forward_{horizon}d"
            valid = daily_group.dropna(subset=[return_column]).copy()
            if valid.empty:
                continue
            daily = []
            for date, day in valid.groupby("date", sort=True):
                daily_mean = float(day[return_column].mean())
                ic = _spearman(day["audit_score"], day[return_column])
                row = {
                    "candidate_group": group,
                    "period": str(int(year)),
                    "horizon_days": horizon,
                    "bucket": "universe",
                    "observation_count": len(day),
                    "date_count": 1,
                    "mean_forward_return": daily_mean,
                    "median_forward_return": float(day[return_column].median()),
                    "win_rate": float(day[return_column].gt(0).mean()),
                    "rank_ic": ic,
                    "benchmark_excess": float(day[f"excess_{horizon}d"].mean()) if f"excess_{horizon}d" in day else np.nan,
                }
                daily.append(row)
                for bucket in buckets:
                    selected = day[day["audit_rank"].le(bucket)]
                    if selected.empty:
                        continue
                    row = dict(row)
                    row.update({
                        "bucket": f"top{bucket}",
                        "observation_count": len(selected),
                        "mean_forward_return": float(selected[return_column].mean()),
                        "median_forward_return": float(selected[return_column].median()),
                        "win_rate": float(selected[return_column].gt(0).mean()),
                        "benchmark_excess": float(selected[f"excess_{horizon}d"].mean()) if f"excess_{horizon}d" in selected else np.nan,
                        "universe_mean_forward_return": daily_mean,
                        "top_minus_universe": float(selected[return_column].mean() - daily_mean),
                    })
                    daily.append(row)
            if not daily:
                continue
            daily_frame = pd.DataFrame(daily)
            for bucket, bucket_frame in daily_frame.groupby("bucket", sort=False):
                rows.append({
                    "candidate_group": group,
                    "period": str(int(year)),
                    "horizon_days": horizon,
                    "bucket": bucket,
                    "observation_count": int(bucket_frame["observation_count"].sum()),
                    "date_count": int(len(bucket_frame)),
                    "mean_forward_return": float(bucket_frame["mean_forward_return"].mean()),
                    "median_daily_forward_return": float(bucket_frame["mean_forward_return"].median()),
                    "median_observation_forward_return": float(valid[return_column].median()),
                    "win_rate": float(bucket_frame["win_rate"].mean()),
                    "mean_rank_ic": float(bucket_frame["rank_ic"].mean()),
                    "positive_rank_ic_rate": float(bucket_frame["rank_ic"].gt(0).mean()),
                    "mean_benchmark_excess": float(bucket_frame["benchmark_excess"].mean()),
                    "mean_top_minus_universe": float(bucket_frame.get("top_minus_universe", pd.Series(dtype=float)).mean()),
                })
    return pd.DataFrame(rows)


def audit_candidates(
    candidate_directory: str | Path,
    price_directory: str | Path,
    *,
    benchmark_code: str | None = None,
    output_directory: str | Path | None = None,
    include_ineligible: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    candidates, manifest = load_manifest_candidates(
        candidate_directory,
        include_ineligible=include_ineligible,
        start_date=start_date,
        end_date=end_date,
    )
    codes = set(candidates["code"].astype(str))
    if benchmark_code:
        codes.add(str(benchmark_code))
    prices, missing_codes = load_price_series(price_directory, codes)
    if end_date:
        prices = {code: series.loc[series.index <= pd.Timestamp(end_date).normalize()]
                  for code, series in prices.items()}
    benchmark = prices.get(str(benchmark_code)) if benchmark_code else None
    observations = attach_forward_returns(candidates, prices, benchmark=benchmark)
    metrics = summarize_audit(observations)
    limitations = [
        "close_to_close_labels_use_prepared_price_cache",
        "candidate_effectiveness_is_not_execution_engine_performance",
        "corporate_action_and_exact_limit_coverage_must_be_checked_separately",
    ]
    if missing_codes:
        limitations.append("missing_price_files_excluded_from_labels")
    payload = {
        "research_only": True,
        "strict_point_in_time": False,
        "qualification_reason": "manifest_flags_do_not_prove_revision_vintage",
        "label_end_date": end_date,
        "financial_point_in_time": bool(manifest.get("financial_point_in_time", False)),
        "industry_point_in_time": bool(manifest.get("industry_point_in_time", False)),
        "data_limited": True,
        "candidate_directory": str(Path(candidate_directory)),
        "price_directory": str(Path(price_directory)),
        "benchmark_code": benchmark_code,
        "candidate_snapshot_count": int(candidates["date"].nunique()),
        "candidate_row_count": int(len(candidates)),
        "candidate_group_counts": {str(k): int(v) for k, v in candidates["candidate_group"].value_counts().items()},
        "price_code_count": int(len(prices)),
        "missing_price_code_count": int(len(missing_codes)),
        "missing_price_code_sample": missing_codes[:20],
        "data_limitations": limitations,
    }
    if output_directory:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        observations.to_csv(output / "observations.csv", index=False, encoding="utf-8-sig")
        metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
        (output / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _write_markdown(output / "report.md", payload, metrics)
        payload["output_directory"] = str(output)
    return {"summary": payload, "observations": observations, "metrics": metrics}


def _write_markdown(path: Path, payload: dict, metrics: pd.DataFrame) -> None:
    lines = [
        "# 选股前向收益审计",
        "",
        "> 本报告只评估冻结候选快照的选股标签，不代表组合买卖回测或实盘收益。",
        "",
        f"- 候选快照：{payload['candidate_snapshot_count']} 个",
        f"- 候选行数：{payload['candidate_row_count']}",
        f"- 价格文件：{payload['price_code_count']} 个，可用基准：{payload.get('benchmark_code') or '未提供'}",
        f"- 研究标记：research_only={payload['research_only']}，data_limited={payload['data_limited']}",
        "",
        "## 指标",
        "",
    ]
    if metrics.empty:
        lines.append("没有足够的前向价格标签生成指标。")
    else:
        display = metrics.copy()
        for column in ("mean_forward_return", "median_daily_forward_return", "median_observation_forward_return", "win_rate", "mean_rank_ic", "positive_rank_ic_rate", "mean_benchmark_excess", "mean_top_minus_universe"):
            if column in display:
                display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        lines.append(display.to_markdown(index=False))
    lines.extend(["", "## 数据限制", ""])
    lines.extend(f"- {item}" for item in payload["data_limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计候选快照的多周期前向收益与排名有效性")
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--price-directory", required=True)
    parser.add_argument(
        "--benchmark-code",
        default=None,
        help="可选的已验证指数价格文件代码；缺失时使用候选横截面均值作为对照",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_candidates(
        args.candidate_directory,
        args.price_directory,
        benchmark_code=args.benchmark_code,
        output_directory=args.output_directory,
        include_ineligible=args.include_ineligible,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
