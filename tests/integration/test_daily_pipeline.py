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


def test_daily_pipeline_alerts_once_after_failures():
    alerts = []
    steps = {name: lambda: 0 for name in STEP_NAMES}
    steps["fundamental_update"] = lambda: 1

    result = run_daily_pipeline(
        steps=steps,
        no_push=False,
        alert=lambda message: alerts.append(message),
    )

    assert result.failed_steps == ("fundamental_update",)
    assert alerts == ["FAILED STEPS: fundamental_update"]


def test_default_steps_only_allow_final_report_to_push(monkeypatch):
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

    steps = build_default_steps(config, no_push=False)
    steps["factor_selection"]()
    steps["daily_report"]()

    assert "--no-push" in captured["factor"]
    assert "--no-push" not in captured["report"]


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
    steps["fundamental_update"]()
    steps["fundamental_selection"]()

    assert captured["update"][-2:] == ["--report-period", "2026-03-31"]
    assert captured["selection"][-2:] == ["--report-period", "2026-03-31"]
