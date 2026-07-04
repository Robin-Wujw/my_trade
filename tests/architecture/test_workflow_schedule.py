from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_stock_selection_workflow_runs_full_pipeline_daily():
    text = (ROOT / ".github/workflows/stock-selection.yml").read_text(
        encoding="utf-8"
    )

    assert 'cron: "30 8 * * *"' in text
    assert "Three-day schedule gate" not in text
    assert "steps.gate.outputs.run" not in text
    assert "Run tests" in text
    assert "Run full-market selection" in text


def test_strategy_uses_only_rolling_21_day_formula33_breadth():
    text = (ROOT / "STRATEGY.md").read_text(encoding="utf-8-sig")

    assert "最近21个交易日三浪三命中股票去重数" in text
    assert "当日命中数" not in text
    assert "较前日变化和连续扩张/收缩" not in text
    assert "新进入、退出和分区变化" in text
