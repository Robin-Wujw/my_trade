"""Validate and summarize the complete QuantsPlaybook portfolio matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apps.quantsplaybook_compare import _portfolio_summary_row
from stock_research.core.paths import PATHS


DEFAULT_OUTPUT_ROOT = (
    PATHS.runtime_root / "backtests"
    / "quantsplaybook_factor_only_2021_to_20260721"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_periods(
    start_year: int,
    end_year: int,
    final_end_date: str,
) -> list[tuple[str, str, str]]:
    periods = []
    for year in range(int(start_year), int(end_year) + 1):
        end = (
            final_end_date
            if year == pd.Timestamp(final_end_date).year
            else f"{year}-12-31"
        )
        periods.append((str(year), f"{year}-01-01", end))
    periods.append((
        f"{start_year}_to_date",
        f"{start_year}-01-01",
        final_end_date,
    ))
    return periods


def load_validated_matrix(
    output_root: str | Path,
    *,
    start_year: int,
    end_year: int,
    final_end_date: str,
) -> pd.DataFrame:
    root = Path(output_root)
    factors = sorted(
        path.name
        for path in (root / "candidates").iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if not factors:
        raise RuntimeError("no factor candidate manifests are available")
    rows = []
    for factor in factors:
        manifest = root / "candidates" / factor / "manifest.json"
        manifest_hash = _sha256(manifest)
        for label, start, end in expected_periods(
            start_year, end_year, final_end_date,
        ):
            path = (
                root / "portfolio" / factor
                / f"portfolio_{start}_{end}_summary.json"
            )
            if not path.is_file():
                raise RuntimeError(
                    f"portfolio matrix summary is missing: {factor} {label}",
                )
            summary = json.loads(path.read_text(encoding="utf-8"))
            audit = summary.get("trade_audit_summary") or {}
            fingerprints = summary.get("input_fingerprints") or {}
            failures = []
            if summary.get("coverage_complete") is not True:
                failures.append("coverage_incomplete")
            if audit.get("violation_count") != 0:
                failures.append("trade_audit_violation")
            if fingerprints.get("candidate_manifest_sha256") != manifest_hash:
                failures.append("candidate_manifest_mismatch")
            if str(summary.get("requested_start")) != start:
                failures.append("requested_start_mismatch")
            if str(summary.get("end_date")) != end:
                failures.append("end_date_mismatch")
            if failures:
                raise RuntimeError(
                    f"portfolio matrix summary is invalid: "
                    f"{factor} {label} {failures}",
                )
            row = _portfolio_summary_row(factor, path)
            row.update({
                "period": label,
                "requested_start": start,
                "end_date": end,
                "coverage_complete": True,
                "trade_audit_violation_count": 0,
            })
            rows.append(row)
    expected_count = len(factors) * len(
        expected_periods(start_year, end_year, final_end_date),
    )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"portfolio matrix count mismatch: {len(rows)} != {expected_count}",
        )
    return pd.DataFrame(rows)


def _compound_percent(values) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return round(float((1.0 + clean / 100.0).prod() - 1.0) * 100.0, 3)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (
        not isinstance(value, (str, bool))
        and pd.isna(value)
    ):
        return None
    return value


def strategy_comparison(
    matrix: pd.DataFrame,
    *,
    validation_year: int = 2024,
    oos_start_year: int = 2025,
) -> pd.DataFrame:
    rows = []
    for factor, group in matrix.groupby("factor", sort=True):
        annual = group[
            group["period"].astype(str).str.fullmatch(r"\d{4}")
        ].copy()
        annual["year"] = annual["period"].astype(int)
        validation = annual[annual["year"].eq(int(validation_year))]
        oos = annual[annual["year"].ge(int(oos_start_year))]
        full = group[
            group["period"].astype(str).str.endswith("_to_date")
        ]
        if len(validation) != 1 or oos.empty or len(full) != 1:
            raise RuntimeError(f"strategy periods are incomplete: {factor}")
        validation_row = validation.iloc[0]
        full_row = full.iloc[0]
        annual_returns = pd.to_numeric(
            annual["final_return_pct"], errors="coerce",
        )
        rows.append({
            "factor": factor,
            "validation_return_pct": validation_row["final_return_pct"],
            "validation_ex_top1_return_pct": validation_row[
                "exclude_top1_approx_final_return_pct"
            ],
            "validation_ex_top3_return_pct": validation_row[
                "exclude_top3_approx_final_return_pct"
            ],
            "validation_max_drawdown_pct": validation_row[
                "maximum_drawdown_pct"
            ],
            "validation_profit_loss_ratio": validation_row[
                "profit_loss_ratio"
            ],
            "oos_compound_independent_period_return_pct": _compound_percent(
                oos["final_return_pct"],
            ),
            "oos_ex_top1_compound_independent_period_return_pct": _compound_percent(
                oos["exclude_top1_approx_final_return_pct"],
            ),
            "oos_ex_top3_compound_independent_period_return_pct": _compound_percent(
                oos["exclude_top3_approx_final_return_pct"],
            ),
            "oos_worst_period_return_pct": pd.to_numeric(
                oos["final_return_pct"], errors="coerce",
            ).min(),
            "oos_positive_period_count": int(
                pd.to_numeric(oos["final_return_pct"], errors="coerce").gt(0).sum()
            ),
            "annual_mean_return_pct": round(float(annual_returns.mean()), 3),
            "annual_median_return_pct": round(float(annual_returns.median()), 3),
            "annual_worst_return_pct": round(float(annual_returns.min()), 3),
            "annual_positive_period_count": int(annual_returns.gt(0).sum()),
            "full_period_return_pct": full_row["final_return_pct"],
            "full_period_max_drawdown_pct": full_row["maximum_drawdown_pct"],
            "full_period_profit_loss_ratio": full_row["profit_loss_ratio"],
            "full_period_ex_top1_return_pct": full_row[
                "exclude_top1_approx_final_return_pct"
            ],
            "full_period_ex_top3_return_pct": full_row[
                "exclude_top3_approx_final_return_pct"
            ],
            "full_period_top1_positive_profit_share_pct": full_row[
                "top1_positive_profit_share_pct"
            ],
            "full_period_top3_positive_profit_share_pct": full_row[
                "top3_positive_profit_share_pct"
            ],
        })
    result = pd.DataFrame(rows)
    result = result.sort_values(
        [
            "validation_ex_top1_return_pct",
            "validation_return_pct",
            "validation_max_drawdown_pct",
            "validation_profit_loss_ratio",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "validation_rank", range(1, len(result) + 1))
    result["selected_by_validation"] = result["validation_rank"].eq(1)
    result["oos_robust"] = (
        pd.to_numeric(
            result["oos_compound_independent_period_return_pct"],
            errors="coerce",
        ).gt(0)
        & pd.to_numeric(
            result["oos_ex_top1_compound_independent_period_return_pct"],
            errors="coerce",
        ).gt(0)
    )
    result["oos_top3_robust"] = (
        pd.to_numeric(
            result["oos_compound_independent_period_return_pct"],
            errors="coerce",
        ).gt(0)
        & pd.to_numeric(
            result["oos_ex_top3_compound_independent_period_return_pct"],
            errors="coerce",
        ).gt(0)
    )
    return result


def render_markdown(
    matrix: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    winner = comparison.iloc[0]
    annual = matrix[
        matrix["period"].astype(str).str.fullmatch(r"\d{4}")
    ].pivot(index="period", columns="factor", values="final_return_pct")
    annual = annual.reindex(sorted(annual.columns), axis=1)
    display_columns = [
        "validation_rank",
        "factor",
        "validation_return_pct",
        "validation_ex_top1_return_pct",
        "validation_ex_top3_return_pct",
        "oos_compound_independent_period_return_pct",
        "oos_ex_top1_compound_independent_period_return_pct",
        "oos_ex_top3_compound_independent_period_return_pct",
        "annual_median_return_pct",
        "annual_worst_return_pct",
        "full_period_return_pct",
        "full_period_max_drawdown_pct",
        "full_period_profit_loss_ratio",
        "full_period_ex_top1_return_pct",
        "full_period_ex_top3_return_pct",
        "oos_robust",
        "oos_top3_robust",
    ]
    ranking = comparison[display_columns].copy()
    ranking.columns = [
        "验证排名", "因子/组合", "2024收益%", "2024剔除头部1近似收益%",
        "2024剔除头部3近似收益%", "2025-2026独立期复合收益%",
        "样本外剔除头部1复合收益%", "样本外剔除头部3复合收益%",
        "年度收益中位数%", "最差年度收益%", "全周期收益%",
        "全周期最大回撤%", "全周期盈亏比", "全周期剔除头部1近似收益%",
        "全周期剔除头部3近似收益%", "样本外Top1稳健", "样本外Top3稳健",
    ]
    return "\n".join([
        "# QuantsPlaybook 因子组合回测比较",
        "",
        "所有策略使用相同买入、退出、仓位、T+1、整手、费用、涨跌停、分红送转和退市口径。",
        "最终策略只按 2024 验证集选择；2025-2026 仅作样本外验收，不参与拟合或重新排名。",
        "",
        "## 验证集选择",
        "",
        f"- 验证集选择：`{winner['factor']}`。",
        f"- 2024 收益：`{winner['validation_return_pct']:.3f}%`。",
        (
            "- 2024 剔除头部第一贡献后的近似收益："
            f"`{winner['validation_ex_top1_return_pct']:.3f}%`。"
        ),
        (
            "- 2025-2026 独立年度复合收益："
            f"`{winner['oos_compound_independent_period_return_pct']:.3f}%`。"
        ),
        (
            "- 样本外剔除头部第一贡献后的复合近似收益："
            f"`{winner['oos_ex_top1_compound_independent_period_return_pct']:.3f}%`。"
        ),
        (
            "- 样本外剔除头部前三贡献后的复合近似收益："
            f"`{winner['oos_ex_top3_compound_independent_period_return_pct']:.3f}%`。"
        ),
        f"- 样本外 Top1 稳健验收：`{bool(winner['oos_robust'])}`。",
        f"- 样本外 Top3 稳健验收：`{bool(winner['oos_top3_robust'])}`。",
        "",
        "## 全策略排名",
        "",
        ranking.to_markdown(index=False),
        "",
        "## 独立年度收益",
        "",
        annual.to_markdown(),
        "",
        "剔除头部贡献是集中度诊断，不是重新分配现金后的独立回测。",
        "2025 与 2026 年至 7 月 21 日为独立起始资金回测，复合值不代表跨年持仓连续回放。",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--oos-start-year", type=int, default=2025)
    return parser


def write_results(
    output_root: str | Path,
    *,
    start_year: int,
    end_year: int,
    end_date: str,
    validation_year: int = 2024,
    oos_start_year: int = 2025,
) -> pd.DataFrame:
    root = Path(output_root)
    matrix = load_validated_matrix(
        root,
        start_year=start_year,
        end_year=end_year,
        final_end_date=end_date,
    )
    comparison = strategy_comparison(
        matrix,
        validation_year=validation_year,
        oos_start_year=oos_start_year,
    )
    matrix.to_csv(
        root / "portfolio_metrics_all_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        root / "strategy_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winner = _json_safe(comparison.iloc[0].to_dict())
    (root / "selection_decision.json").write_text(
        json.dumps(
            {
                "selection_data": f"{validation_year}_validation_only",
                "oos_used_for_selection": False,
                "selected_strategy": winner["factor"],
                "selected_strategy_oos_robust": bool(winner["oos_robust"]),
                "selected_strategy_oos_top3_robust": bool(
                    winner["oos_top3_robust"]
                ),
                "metrics": winner,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (root / "portfolio_comparison.md").write_text(
        render_markdown(matrix, comparison),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)
    return comparison


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    write_results(
        args.output_root,
        start_year=args.start_year,
        end_year=args.end_year,
        end_date=args.end_date,
        validation_year=args.validation_year,
        oos_start_year=args.oos_start_year,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
