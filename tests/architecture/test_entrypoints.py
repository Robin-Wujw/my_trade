from pathlib import Path

from apps.daily_pipeline import build_parser
from stock_research.pipelines.daily import STEP_NAMES


ROOT = Path(__file__).parents[2]


def test_daily_entrypoint_declares_the_seven_production_steps():
    assert STEP_NAMES == (
        "formula33",
        "sector_stats",
        "sector_watch",
        "factor_selection",
        "fundamental_update",
        "fundamental_selection",
        "daily_report",
    )


def test_powershell_entrypoint_is_under_scripts():
    assert (ROOT / "scripts" / "run_daily_analysis.ps1").is_file()


def test_daily_entrypoint_honors_actions_no_push_environment(monkeypatch):
    monkeypatch.setenv("NO_PUSH", "1")
    assert build_parser().parse_args([]).no_push is True
