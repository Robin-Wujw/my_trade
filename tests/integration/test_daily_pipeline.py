import json

import pytest

from stock_research.core.config import PipelineConfig
from stock_research.pipelines import daily as daily_pipeline
from stock_research.pipelines.daily import (
    STEP_NAMES,
    build_default_steps,
    run_daily_pipeline,
)


def test_daily_pipeline_runs_steps_in_declared_order():
    calls = []
    steps = {name: lambda name=name: calls.append(name) or 0 for name in STEP_NAMES}

    result = run_daily_pipeline(steps=steps, no_push=True)

    assert result.failed_steps == ()
    assert result.skipped_steps == ()
    assert calls == list(STEP_NAMES)


def test_daily_pipeline_prints_step_lifecycle(capsys):
    steps = {name: lambda: 0 for name in STEP_NAMES}

    run_daily_pipeline(steps=steps, no_push=True)

    output = capsys.readouterr().out
    assert "[daily_pipeline][formula33] start" in output
    assert "[daily_pipeline][formula33] finish code=0" in output


def test_daily_pipeline_skips_dependent_selection_and_report():
    calls = []

    def step(name, code=0):
        return lambda: calls.append(name) or code

    steps = {name: step(name) for name in STEP_NAMES}
    steps["fundamental_update"] = step("fundamental_update", 1)

    result = run_daily_pipeline(steps=steps, no_push=True)

    assert result.failed_steps == ("fundamental_update",)
    assert result.skipped_steps == ("fundamental_selection", "daily_report")
    assert "fundamental_selection" not in calls
    assert "daily_report" not in calls


def test_daily_pipeline_converts_system_exit_and_continues_independent_steps():
    calls = []

    def exits():
        calls.append("formula33")
        raise SystemExit(2)

    steps = {name: lambda name=name: calls.append(name) or 0 for name in STEP_NAMES}
    steps["formula33"] = exits

    result = run_daily_pipeline(steps=steps, no_push=True)

    assert result.failed_steps == ("formula33",)
    assert "sector_stats" in calls
    assert "daily_report" not in calls


def test_daily_pipeline_skips_report_when_sector_statistics_fails():
    calls = []

    def step(name, code=0):
        return lambda: calls.append(name) or code

    steps = {name: step(name) for name in STEP_NAMES}
    steps["sector_stats"] = step("sector_stats", 2)

    result = run_daily_pipeline(steps=steps, no_push=True)

    assert result.failed_steps == ("sector_stats",)
    assert "daily_report" in result.skipped_steps
    assert "daily_report" not in calls


@pytest.mark.parametrize(
    "failed_step",
    [
        "formula33",
        "sector_stats",
        "sector_watch",
        "factor_selection",
        "fundamental_update",
        "fundamental_selection",
    ],
)
def test_daily_report_requires_every_production_step(failed_step):
    calls = []
    steps = {
        name: lambda name=name: calls.append(name) or (1 if name == failed_step else 0)
        for name in STEP_NAMES
    }

    result = run_daily_pipeline(steps=steps, no_push=True)

    assert failed_step in result.failed_steps
    assert "daily_report" in result.skipped_steps
    assert "daily_report" not in calls


def test_default_steps_only_allow_final_report_to_push(monkeypatch, tmp_path):
    captured = {}
    config = PipelineConfig(
        factor_workers=1,
        formula33_workers=1,
        formula33_sleep=0.2,
        formula33_retries=5,
        sector_sleep=0.3,
        sector_retries=5,
        financial_updates=100,
    )
    monkeypatch.setattr(
        daily_pipeline.factor_selection,
        "main",
        lambda args: captured.setdefault("factor", args) or 0,
    )
    monkeypatch.setattr(
        daily_pipeline.daily_report,
        "run",
        lambda args: captured.setdefault("report", args) or 0,
    )
    monkeypatch.setattr(
        daily_pipeline.fundamental_update,
        "main",
        lambda args: captured.setdefault("update", args) or 0,
    )

    steps = build_default_steps(config, no_push=False)
    artifact_paths = {
        "fundamental_path": tmp_path / "fundamental.csv",
        "formula_path": tmp_path / "formula.csv",
        "sector_stats_xlsx_path": tmp_path / "sector_stats.xlsx",
        "sector_stats_md_path": tmp_path / "sector_stats.md",
        "sector_path": tmp_path / "sector.csv",
        "factor_path": tmp_path / "factor.csv",
    }
    for path in artifact_paths.values():
        path.write_text("current run", encoding="utf-8")
    steps.artifacts.update(
        {name: str(path) for name, path in artifact_paths.items()}
    )
    steps.artifacts["observation_date"] = "2026-07-10"
    steps["factor_selection"]()
    steps["fundamental_update"]()
    steps["daily_report"]()

    assert "--no-push" in captured["factor"]
    assert "--allow-login-fail" not in captured["factor"]
    assert "--alert" not in captured["update"]
    assert "--no-push" not in captured["report"]
    assert captured["report"][captured["report"].index("--max-chars") + 1] == "18000"
    assert captured["report"][captured["report"].index("--formula-path") + 1] == str(artifact_paths["formula_path"])
    assert captured["report"][captured["report"].index("--sector-path") + 1] == str(artifact_paths["sector_path"])
    assert captured["report"][captured["report"].index("--fundamental-path") + 1] == str(artifact_paths["fundamental_path"])


def test_default_steps_forward_actions_report_period(monkeypatch):
    captured = {}
    config = PipelineConfig(
        factor_workers=1,
        formula33_workers=1,
        formula33_sleep=0.2,
        formula33_retries=5,
        sector_sleep=0.3,
        sector_retries=5,
        financial_updates=100,
    )
    monkeypatch.setattr(
        daily_pipeline.fundamental_update,
        "main",
        lambda args: captured.setdefault("update", args) or 0,
    )
    monkeypatch.setattr(
        daily_pipeline.fundamental_selection,
        "main",
        lambda args: captured.setdefault("selection", args) or 0,
    )

    steps = build_default_steps(
        config,
        no_push=True,
        report_period="2026-03-31",
    )
    steps.artifacts["observation_date"] = "2026-07-10"
    steps["fundamental_update"]()
    steps["fundamental_selection"]()

    assert captured["update"][-2:] == ["--report-period", "2026-03-31"]
    assert captured["selection"][captured["selection"].index("--report-period") + 1] == "2026-03-31"
    assert captured["selection"][-2:] == ["--observation-date", "2026-07-10"]


def test_default_steps_use_production_formula_filters_and_no_push_alerts(monkeypatch):
    captured = {}
    config = PipelineConfig(
        factor_workers=1,
        formula33_workers=1,
        formula33_sleep=0.2,
        formula33_retries=2,
        sector_sleep=0.3,
        sector_retries=2,
        financial_updates=100,
    )
    monkeypatch.setattr(
        daily_pipeline.formula33,
        "main",
        lambda args: captured.setdefault("formula33", args) or 0,
    )
    monkeypatch.setattr(
        daily_pipeline.fundamental_update,
        "main",
        lambda args: captured.setdefault("update", args) or 0,
    )

    steps = build_default_steps(config, no_push=True)
    steps["formula33"]()
    steps["fundamental_update"]()

    formula_args = captured["formula33"]
    assert formula_args[formula_args.index("--market-cap-source") + 1] == "auto"
    assert formula_args[formula_args.index("--missing-mktcap-policy") + 1] == "exclude"
    assert formula_args[formula_args.index("--maxtasksperchild") + 1] == "1000"
    assert "--alert" not in captured["update"]


def test_default_pipeline_captures_only_outputs_from_current_run(
    monkeypatch,
    tmp_path,
):
    market_dir = tmp_path / "market"
    selection_dir = tmp_path / "selection"
    state_dir = tmp_path / "state"
    market_dir.mkdir()
    selection_dir.mkdir()
    state_dir.mkdir()
    stale_sector = market_dir / "sector_watch_stale.csv"
    stale_factor = selection_dir / "factor_selection_stale.csv"
    stale_fundamental = selection_dir / "daily_fundamental_selection_stale.csv"
    stale_sector.write_text("date\n2026-07-03\n", encoding="utf-8")
    stale_factor.write_text("date\n2026-07-03\n", encoding="utf-8")
    stale_fundamental.write_text("date\n2026-07-03\n", encoding="utf-8")
    formula_csv = market_dir / "formula33_stats_current.csv"
    sector_stats_xlsx = market_dir / "sector_stats_current.xlsx"
    sector_stats_md = market_dir / "sector_stats_current.md"
    sector_csv = market_dir / "sector_watch_current.csv"
    factor_csv = selection_dir / "factor_selection_current.csv"
    fundamental_csv = selection_dir / "daily_fundamental_selection_current.csv"
    manifest_path = state_dir / "formula33_completion.json"
    captured = {}

    monkeypatch.setattr(daily_pipeline.formula33, "FORMULA33_MANIFEST_FILE", str(manifest_path))
    monkeypatch.setattr(daily_pipeline.sector_statistics, "OUTPUT_DIR", str(market_dir))
    monkeypatch.setattr(daily_pipeline.sector_watch, "OUTPUT_DIR", str(market_dir))
    monkeypatch.setattr(daily_pipeline.factor_selection, "OUTPUT_DIR", str(selection_dir))
    monkeypatch.setattr(
        daily_pipeline.fundamental_selection,
        "OUTPUT_DIR",
        str(selection_dir),
    )

    def formula_main(args):
        formula_csv.write_text("date\n2026-07-10\n", encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "code_version": daily_pipeline.formula33.FORMULA33_CODE_VERSION,
                    "observation_date": "2026-07-10",
                    "outputs": [str(formula_csv)],
                }
            ),
            encoding="utf-8",
        )
        return 0

    def sector_statistics_main(args):
        sector_stats_xlsx.write_text("xlsx", encoding="utf-8")
        sector_stats_md.write_text("markdown", encoding="utf-8")
        return 0

    def sector_main(args):
        sector_csv.write_text("date\n2026-07-10\n", encoding="utf-8")
        return 0

    def factor_main(args):
        factor_csv.write_text("date\n2026-07-10\n", encoding="utf-8")
        return 0

    def fundamental_main(args):
        captured["fundamental_args"] = args
        fundamental_csv.write_text("date\n2026-07-10\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(daily_pipeline.formula33, "main", formula_main)
    monkeypatch.setattr(daily_pipeline.sector_statistics, "main", sector_statistics_main)
    monkeypatch.setattr(daily_pipeline.sector_watch, "main", sector_main)
    monkeypatch.setattr(daily_pipeline.factor_selection, "main", factor_main)
    monkeypatch.setattr(daily_pipeline.fundamental_update, "main", lambda args: 0)
    monkeypatch.setattr(
        daily_pipeline.fundamental_selection,
        "main",
        fundamental_main,
    )
    monkeypatch.setattr(
        daily_pipeline.daily_report,
        "run",
        lambda args: captured.setdefault("report_args", args) and 0,
    )
    config = PipelineConfig(
        factor_workers=1,
        formula33_workers=1,
        formula33_sleep=0,
        formula33_retries=1,
        sector_sleep=0,
        sector_retries=1,
        financial_updates=1,
    )

    steps = build_default_steps(config, no_push=True)
    result = run_daily_pipeline(
        steps=steps,
        no_push=True,
    )

    assert result.failed_steps == ()
    report_args = captured["report_args"]
    assert report_args[report_args.index("--formula-path") + 1] == str(formula_csv.resolve())
    assert report_args[report_args.index("--sector-path") + 1] == str(sector_csv.resolve())
    assert report_args[report_args.index("--fundamental-path") + 1] == str(fundamental_csv.resolve())
    assert captured["fundamental_args"][-2:] == ["--observation-date", "2026-07-10"]
    assert steps.artifacts["sector_stats_xlsx_path"] == str(sector_stats_xlsx.resolve())
    assert steps.artifacts["sector_stats_md_path"] == str(sector_stats_md.resolve())
    assert steps.artifacts["factor_path"] == str(factor_csv.resolve())
    assert str(stale_sector) not in report_args
    assert str(stale_factor) not in steps.artifacts.values()
    assert str(stale_fundamental) not in report_args


def test_current_run_capture_rejects_unchanged_stale_output(tmp_path):
    stale = tmp_path / "sector_watch_stale.csv"
    stale.write_text("date\n2026-07-03\n", encoding="utf-8")
    before = daily_pipeline._output_snapshot(tmp_path, "sector_watch_")

    with pytest.raises(RuntimeError, match="本轮产物数量必须为1"):
        daily_pipeline._capture_single_current_output(
            before,
            tmp_path,
            "sector_watch_",
        )


def test_sector_statistics_capture_requires_matching_xlsx_and_markdown(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(daily_pipeline.sector_statistics, "OUTPUT_DIR", str(tmp_path))
    steps = daily_pipeline._ArtifactTrackingSteps({}, report_args=[])
    steps.before_step("sector_stats")
    (tmp_path / "sector_stats_20260711_120000.xlsx").write_text(
        "xlsx",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="本轮产物数量必须为1"):
        steps.after_step("sector_stats")
