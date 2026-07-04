"""Single, explicit orchestration for the daily research run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from stock_research.core.config import PipelineConfig
from stock_research.pipelines import (
    daily_report,
    factor_selection,
    formula33,
    fundamental_selection,
    fundamental_update,
    sector_statistics,
    sector_watch,
)


STEP_NAMES = (
    "formula33",
    "sector_stats",
    "sector_watch",
    "factor_selection",
    "fundamental_update",
    "fundamental_selection",
    "daily_report",
)


@dataclass(frozen=True)
class DailyRunResult:
    failed_steps: tuple[str, ...]
    skipped_steps: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed_steps else 0


def _run_step(name: str, step: Callable[[], int]) -> int:
    try:
        return int(step() or 0)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if exc.code not in (None, 0):
            print(f"{name} failed: {exc.code}")
        return int(code or 0)
    except Exception as exc:
        print(f"{name} failed: {exc}")
        return 1


def run_daily_pipeline(
    *,
    steps: Mapping[str, Callable[[], int]],
    no_push: bool = False,
    alert: Optional[Callable[[str], object]] = None,
) -> DailyRunResult:
    statuses = {}
    failed = []
    skipped = []
    for name in STEP_NAMES:
        if name == "fundamental_selection" and statuses.get("fundamental_update") != 0:
            statuses[name] = None
            skipped.append(name)
            continue
        if name == "daily_report" and any(
            statuses.get(required) != 0
            for required in ("formula33", "sector_watch", "fundamental_selection")
        ):
            statuses[name] = None
            skipped.append(name)
            continue
        code = _run_step(name, steps[name])
        statuses[name] = code
        if code:
            failed.append(name)
    result = DailyRunResult(tuple(failed), tuple(skipped))
    if failed and alert is not None and not no_push:
        try:
            alert("FAILED STEPS: " + ", ".join(failed))
        except Exception as exc:
            print(f"pipeline alert failed: {exc}")
    return result


def build_default_steps(
    config: PipelineConfig,
    *,
    no_push: bool,
    report_period: str = "",
):
    factor_args = [
        "--top", "200", "--core-min-score", "80", "--low-min-score", "75",
        "--quality-min-score", "80", "--value-min-mktcap", "100",
        "--workers", str(config.factor_workers), "--value-watch-ratio", "1.08",
        "--value-watch-top", "20", "--akshare-cache-only", "--allow-login-fail",
        "--no-push",
    ]
    report_args = ["--top", "10", "--selection-top", "30", "--max-chars", "12000"]
    if no_push:
        report_args.append("--no-push")
    update_args = [
        "--max-updates", str(config.financial_updates), "--workers", "2",
        "--min-price-coverage", "0.90", "--min-financial-coverage", "0.35",
        "--target-financial-coverage", "0.95", "--alert",
    ]
    selection_args = ["--value-ratio", "1.08", "--normal-top", "30"]
    if report_period:
        update_args.extend(["--report-period", report_period])
        selection_args.extend(["--report-period", report_period])
    return {
        "formula33": lambda: formula33.main([
            "--lookback", "21", "--history-days", "420",
            "--workers", str(config.formula33_workers),
            "--sleep", str(config.formula33_sleep),
            "--retries", str(config.formula33_retries),
            "--retry-delay", "5", "--capital-workers", "1",
            "--require-end-trade", "--price-source", "akshare",
            "--metadata-source", "akshare", "--missing-mktcap-policy", "pass",
            "--market-cap-source", "none",
        ]),
        "sector_stats": lambda: sector_statistics.main([
            "--lookback", "10", "--history-days", "90", "--top-amount", "50",
            "--sleep", str(config.sector_sleep), "--retries", str(config.sector_retries),
            "--retry-delay", "5",
        ]),
        "sector_watch": lambda: sector_watch.main([
            "--top", "30", "--workers", "4", "--days", "80",
            "--limit-up-days", "5", "--sleep", str(config.sector_sleep),
            "--retries", str(config.sector_retries), "--retry-delay", "5",
        ]),
        "factor_selection": lambda: factor_selection.main(factor_args),
        "fundamental_update": lambda: fundamental_update.main(update_args),
        "fundamental_selection": lambda: fundamental_selection.main(selection_args),
        "daily_report": lambda: daily_report.run(report_args),
    }
