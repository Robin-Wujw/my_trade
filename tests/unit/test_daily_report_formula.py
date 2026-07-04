import pandas as pd

from stock_research.reporting import daily_report
from stock_research.reporting.daily_report import render_formula_status


def test_formula_status_reports_window_and_observation_diagnostics():
    text = render_formula_status(
        {
            "count": 12,
            "change": -28,
            "up_streak": 0,
            "down_streak": 2,
            "window_unique_count": 186,
            "tradable_unique_count": 184,
            "suspended_count": 2,
            "unavailable_count": 0,
        }
    )

    assert "当日XG 12只" in text
    assert "近21日技术去重186只" in text
    assert "正式184只" in text
    assert "观察日无交易排除2只" in text
    assert "数据不可用0只" in text


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
