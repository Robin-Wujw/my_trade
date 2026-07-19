"""Single, explicit orchestration for the daily research run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

from stock_research.core.config import PipelineConfig
from stock_research.core.completion_manifest import CompletionManifest
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


def _output_snapshot(directory, prefix, suffix=".csv"):
    root = Path(directory)
    if not root.is_dir():
        return {}
    snapshot = {}
    for path in root.iterdir():
        if (
            not path.is_file()
            or not path.name.startswith(prefix)
            or not path.name.endswith(suffix)
        ):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _capture_single_current_output(before, directory, prefix, suffix=".csv"):
    after = _output_snapshot(directory, prefix, suffix)
    changed = [
        path
        for path, signature in after.items()
        if before.get(path) != signature
    ]
    sample_paths = [
        path for path in changed if "_sample" in Path(path).name.lower()
    ]
    current = [path for path in changed if path not in sample_paths]
    if sample_paths:
        raise RuntimeError(
            f"{prefix} 本轮生成了样例产物，拒绝用于生产日报: {sample_paths}"
        )
    if len(current) != 1:
        raise RuntimeError(
            f"{prefix} 本轮产物数量必须为1，实际为{len(current)}: {current}"
        )
    if Path(current[0]).stat().st_size <= 0:
        raise RuntimeError(f"{prefix} 本轮产物为空: {current[0]}")
    return current[0]


def _formula_csv_from_completion_manifest():
    payload = CompletionManifest(formula33.FORMULA33_MANIFEST_FILE).read()
    if payload.get("status") != "completed":
        raise RuntimeError("Formula33 completion manifest 未完成或不存在")
    expected_version = formula33.FORMULA33_CODE_VERSION
    if payload.get("code_version") != expected_version:
        raise RuntimeError(
            "Formula33 completion manifest 版本不匹配: "
            f"expected={expected_version} actual={payload.get('code_version')}"
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("Formula33 completion manifest outputs 无效")
    csv_outputs = [
        str(Path(path).resolve())
        for path in outputs
        if isinstance(path, str)
        and Path(path).suffix.lower() == ".csv"
        and Path(path).name.startswith("formula33_stats_")
        and "_sample" not in Path(path).name.lower()
        and Path(path).is_file()
    ]
    if len(csv_outputs) != 1:
        raise RuntimeError(
            "Formula33 completion manifest 必须包含唯一有效CSV: "
            f"{csv_outputs}"
        )
    observation_date = str(payload.get("observation_date") or "").strip()
    try:
        observation_date = date.fromisoformat(observation_date).isoformat()
    except ValueError as exc:
        raise RuntimeError(
            "Formula33 completion manifest observation_date 无效: "
            f"{observation_date!r}"
        ) from exc
    return csv_outputs[0], observation_date


class _ArtifactTrackingSteps(dict):
    """Default production steps plus per-run output capture hooks."""

    def __init__(self, *args, report_args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_args = list(report_args)
        self.artifacts = {}
        self._snapshots = {}

    def before_step(self, name):
        if name == "sector_stats":
            self._snapshots[name] = {
                ".xlsx": _output_snapshot(
                    sector_statistics.OUTPUT_DIR,
                    "sector_stats_",
                    ".xlsx",
                ),
                ".md": _output_snapshot(
                    sector_statistics.OUTPUT_DIR,
                    "sector_stats_",
                    ".md",
                ),
            }
        elif name == "sector_watch":
            self._snapshots[name] = _output_snapshot(
                sector_watch.OUTPUT_DIR,
                "sector_watch_",
            )
        elif name == "factor_selection":
            self._snapshots[name] = _output_snapshot(
                factor_selection.OUTPUT_DIR,
                "factor_selection_",
            )
        elif name == "fundamental_selection":
            self._snapshots[name] = _output_snapshot(
                fundamental_selection.OUTPUT_DIR,
                "daily_fundamental_selection_",
            )

    def after_step(self, name):
        if name == "formula33":
            formula_path, observation_date = _formula_csv_from_completion_manifest()
            self.artifacts["formula_path"] = formula_path
            self.artifacts["observation_date"] = observation_date
        elif name == "sector_stats":
            snapshots = self._snapshots.get(name, {})
            xlsx_path = _capture_single_current_output(
                snapshots.get(".xlsx", {}),
                sector_statistics.OUTPUT_DIR,
                "sector_stats_",
                ".xlsx",
            )
            md_path = _capture_single_current_output(
                snapshots.get(".md", {}),
                sector_statistics.OUTPUT_DIR,
                "sector_stats_",
                ".md",
            )
            if Path(xlsx_path).stem != Path(md_path).stem:
                raise RuntimeError(
                    "sector_stats 本轮 Excel 和 Markdown 产物不属于同一次输出: "
                    f"{xlsx_path}, {md_path}"
                )
            self.artifacts["sector_stats_xlsx_path"] = xlsx_path
            self.artifacts["sector_stats_md_path"] = md_path
        elif name == "sector_watch":
            self.artifacts["sector_path"] = _capture_single_current_output(
                self._snapshots.get(name, {}),
                sector_watch.OUTPUT_DIR,
                "sector_watch_",
            )
        elif name == "factor_selection":
            self.artifacts["factor_path"] = _capture_single_current_output(
                self._snapshots.get(name, {}),
                factor_selection.OUTPUT_DIR,
                "factor_selection_",
            )
        elif name == "fundamental_selection":
            self.artifacts["fundamental_path"] = _capture_single_current_output(
                self._snapshots.get(name, {}),
                fundamental_selection.OUTPUT_DIR,
                "daily_fundamental_selection_",
            )

    def fundamental_selection_args(self, selection_args):
        observation_date = self.artifacts.get("observation_date")
        if not observation_date:
            raise RuntimeError("基本面选股缺少 Formula33 本轮 observation_date")
        return list(selection_args) + ["--observation-date", observation_date]

    def daily_report_args(self):
        required = (
            "formula_path",
            "observation_date",
            "sector_stats_xlsx_path",
            "sector_stats_md_path",
            "sector_path",
            "factor_path",
            "fundamental_path",
        )
        missing = [
            name
            for name in required
            if not self.artifacts.get(name)
            or (name.endswith("_path") and not Path(self.artifacts[name]).is_file())
        ]
        if missing:
            raise RuntimeError(f"日报缺少本轮捕获产物: {missing}")
        return self.report_args + [
            "--fundamental-path", self.artifacts["fundamental_path"],
            "--formula-path", self.artifacts["formula_path"],
            "--sector-path", self.artifacts["sector_path"],
        ]


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
) -> DailyRunResult:
    statuses = {}
    failed = []
    skipped = []
    for name in STEP_NAMES:
        if name == "fundamental_selection" and any(
            statuses.get(required) != 0
            for required in ("formula33", "fundamental_update")
        ):
            statuses[name] = None
            skipped.append(name)
            print(
                f"[daily_pipeline][{name}] skip "
                "dependency=formula33/fundamental_update"
            )
            continue
        if name == "daily_report" and any(
            statuses.get(required) != 0
            for required in (
                "formula33",
                "sector_stats",
                "sector_watch",
                "factor_selection",
                "fundamental_update",
                "fundamental_selection",
            )
        ):
            statuses[name] = None
            skipped.append(name)
            print(
                "[daily_pipeline][daily_report] skip "
                "dependency=formula33/sector_stats/sector_watch/factor_selection/"
                "fundamental_update/fundamental_selection"
            )
            continue
        print(f"[daily_pipeline][{name}] start")
        before_step = getattr(steps, "before_step", None)
        after_step = getattr(steps, "after_step", None)
        try:
            if before_step is not None:
                before_step(name)
            code = _run_step(name, steps[name])
            if code == 0 and after_step is not None:
                after_step(name)
        except Exception as exc:
            print(f"{name} artifact capture failed: {exc}")
            code = 1
        statuses[name] = code
        print(f"[daily_pipeline][{name}] finish code={code}")
        if code:
            failed.append(name)
    return DailyRunResult(tuple(failed), tuple(skipped))


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
        "--value-watch-top", "20", "--akshare-cache-only",
        "--no-push",
    ]
    report_args = ["--top", "10", "--selection-top", "30", "--max-chars", "18000"]
    if no_push:
        report_args.append("--no-push")
    update_args = [
        "--max-updates", str(config.financial_updates), "--workers", "2",
        "--min-price-coverage", "0.90", "--min-financial-coverage", "0.35",
        "--target-financial-coverage", "0.95",
        "--require-target-financial-coverage",
    ]
    selection_args = ["--value-ratio", "1.08", "--normal-top", "30"]
    if report_period:
        update_args.extend(["--report-period", report_period])
        selection_args.extend(["--report-period", report_period])
    tracked_steps = None
    step_functions = {
        "formula33": lambda: formula33.main([
            "--lookback", "21", "--history-days", "420",
            "--workers", str(config.formula33_workers),
            "--maxtasksperchild", "1000",
            "--sleep", str(config.formula33_sleep),
            "--retries", str(config.formula33_retries),
            "--retry-delay", "5", "--capital-workers", "1",
            "--require-end-trade", "--price-source", "akshare",
            "--metadata-source", "akshare", "--missing-mktcap-policy", "exclude",
            "--market-cap-source", "auto",
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
        "fundamental_selection": lambda: fundamental_selection.main(
            tracked_steps.fundamental_selection_args(selection_args)
        ),
        "daily_report": lambda: daily_report.run(tracked_steps.daily_report_args()),
    }
    tracked_steps = _ArtifactTrackingSteps(
        step_functions,
        report_args=report_args,
    )
    return tracked_steps
