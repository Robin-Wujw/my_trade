import pandas as pd

from stock_research.reporting import daily_report
from stock_research.reporting.daily_report import (
    render_formula_status,
    right_side_conclusion,
)


def test_formula_status_only_reports_21_day_breadth_and_trend():
    result = render_formula_status(
        {
            "window_unique_count": 193,
            "window_trend_slope": 2.14,
            "trend_up_streak": 3,
            "trend_down_streak": 0,
            "tradable_unique_count": 191,
            "suspended_count": 2,
            "unavailable_count": 6,
            "count": 15,
            "change": 3,
        }
    )

    assert "近21个交易日三浪三技术去重193只" in result
    assert "趋势斜率+2.14" in result
    assert "连续正趋势3日" in result
    assert "正式191只" in result
    assert "观察日无交易排除2只" in result
    assert "数据不可用6只" in result
    assert "当日XG" not in result
    assert "较前一交易日" not in result


def test_right_side_conclusion_uses_rolling_trend_streaks_only():
    assert right_side_conclusion(
        {"trend_up_streak": 5, "trend_down_streak": 0}
    )[0] == "可以右侧交易"
    assert right_side_conclusion(
        {"trend_up_streak": 3, "trend_down_streak": 0}
    )[0] == "可以谨慎右侧"
    assert right_side_conclusion(
        {"trend_up_streak": 0, "trend_down_streak": 5}
    )[0] == "暂停右侧交易"
    assert right_side_conclusion(
        {"trend_up_streak": 0, "trend_down_streak": 3}
    )[0] == "谨慎或暂停右侧"


def test_right_side_conclusion_ignores_single_day_change():
    status, reason = right_side_conclusion(
        {
            "trend_up_streak": 0,
            "trend_down_streak": 0,
            "window_trend_slope": -0.2,
            "change": 100,
            "up_streak": 9,
        }
    )

    assert status == "等待右侧确认"
    assert "当日" not in reason


def test_daily_report_pushes_the_final_selection_html(monkeypatch, tmp_path):
    pushed = []
    stocks = pd.DataFrame([{"code": "sz.000001", "name": "平安银行"}])
    final_html = "<h1>最终选股结果</h1><p>平安银行</p>"
    monkeypatch.setattr(daily_report, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(daily_report, "SELECTION_DIR", str(tmp_path))
    monkeypatch.setattr(
        daily_report,
        "build_reports",
        lambda *args: (
            "2026-07-03",
            final_html,
            final_html,
            stocks,
            "fundamental.csv",
            "formula.csv",
            "sector.csv",
        ),
    )
    monkeypatch.setattr(
        daily_report,
        "send_pushplus",
        lambda title, content: pushed.append((title, content)) or True,
    )

    daily_report.main([])

    assert pushed == [("2026-07-03 每日四项分析", final_html)]
