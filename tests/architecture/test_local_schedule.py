from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_local_scheduled_task_is_daily_and_checkout_portable():
    text = (ROOT / "scripts/setup_scheduled_task.bat").read_text(
        encoding="utf-8"
    )

    assert "New-ScheduledTaskTrigger -Daily -At '20:30'" in text
    assert "%~dp0" in text
    assert r"D:\MyCodes\my_trade" not in text


def test_strategy_uses_only_rolling_21_day_formula33_breadth():
    text = (ROOT / "STRATEGY.md").read_text(encoding="utf-8-sig")

    assert "最近21个交易日三浪三命中股票去重数" in text
    assert "当日命中数" not in text
    assert "较前日变化和连续扩张/收缩" not in text
    assert "新进入、退出和分区变化" in text
