import pandas as pd
import pytest

from stock_research.reporting import daily_report
from stock_research.reporting.daily_report import (
    DailyReportBundle,
    build_push_reports,
    render_formula_status,
    right_side_conclusion,
)
from stock_research.reporting.diff import SelectionDiff


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
    full_html = "<h1>最终选股结果</h1><p>平安银行</p>"
    push_parts = ("<h1>第一部分</h1>", "<h1>第二部分</h1>")
    monkeypatch.setattr(daily_report, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(daily_report, "SELECTION_DIR", str(tmp_path))
    monkeypatch.setattr(daily_report, "HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setattr(
        daily_report,
        "build_reports",
        lambda *args: DailyReportBundle(
            report_date="2026-07-03",
            full_html=full_html,
            push_parts=push_parts,
            stocks=stocks,
            fundamental_path="fundamental.csv",
            formula_path="formula.csv",
            sector_path="sector.csv",
            selection_diff=SelectionDiff((), (), ()),
        ),
    )
    monkeypatch.setattr(
        daily_report,
        "send_pushplus",
        lambda title, content: pushed.append((title, content)) or True,
    )

    daily_report.main([])

    assert pushed == [
        ("[1/2] 2026-07-03 市场状态与价值线池", push_parts[0]),
        ("[2/2] 2026-07-03 基本面候选与主线", push_parts[1]),
    ]


def make_stocks(prefix, count, strategy_part):
    rows = []
    for index in range(count):
        rows.append(
            {
                "date": "2026-07-03",
                "code": f"sz.{index:06d}{prefix}",
                "name": f"{prefix}股票{index:02d}",
                "strategy_part": strategy_part,
                "strategy_layer": "主流板块基本面优秀·右侧确认",
                "close": 10 + index,
                "value_line": 20 + index,
                "price_to_value": 0.5,
                "quality_score": 90,
                "industry": "半导体",
                "mainline_boards": "半导体",
                "wave_level_50": 10,
                "wave_level_625": 12,
                "wave_level_75": 14,
                "wave_pct": 70,
                "wave_zone": "62.5%以上确认",
                "earnings_yoy": 0.3,
                "selection_reason": "质量、增长、流动性和右侧位置均通过。",
                "value_applicable": True,
            }
        )
    return pd.DataFrame(rows)


def test_build_push_reports_keeps_every_stock_and_bounds_each_part():
    values = make_stocks("V", 55, "1.基本价值线或附近")
    normal = make_stocks("N", 30, "2.正常基本面选股")
    formula = pd.Series(
        {
            "window_unique_count": 193,
            "window_trend_slope": 2.14,
            "trend_up_streak": 3,
            "trend_down_streak": 0,
            "tradable_unique_count": 191,
            "suspended_count": 2,
            "unavailable_count": 6,
        }
    )
    sectors = pd.DataFrame(
        [{"board": "半导体", "final_score": 60.5, "ret3": 0.03,
          "ret5": 0.05, "ret20": 0.20, "amount_5_20": 1.2,
          "limit_up_count": 3}]
    )

    part1, part2 = build_push_reports(
        "2026-07-03",
        values,
        normal,
        formula,
        sectors,
        SelectionDiff((), (), ()),
        18000,
    )

    assert len(part1) <= 18000
    assert len(part2) <= 18000
    assert all(code in part1 for code in values["code"])
    assert all(code in part2 for code in normal["code"])
    assert "结论" in part1 and "风险" in part1
    assert "结论" in part2 and "风险" in part2


def test_daily_report_fails_when_either_push_fails(monkeypatch, tmp_path):
    stocks = pd.DataFrame([{"code": "sz.000001", "name": "平安银行"}])
    bundle = DailyReportBundle(
        report_date="2026-07-03",
        full_html="<h1>完整报告</h1>",
        push_parts=("<h1>第一部分</h1>", "<h1>第二部分</h1>"),
        stocks=stocks,
        fundamental_path="fundamental.csv",
        formula_path="formula.csv",
        sector_path="sector.csv",
        selection_diff=SelectionDiff((), (), ()),
    )
    results = iter([True, False])
    monkeypatch.setattr(daily_report, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(daily_report, "SELECTION_DIR", str(tmp_path))
    monkeypatch.setattr(daily_report, "HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setattr(daily_report, "build_reports", lambda *args: bundle)
    monkeypatch.setattr(
        daily_report,
        "send_pushplus",
        lambda *args: next(results),
    )

    with pytest.raises(SystemExit) as exc:
        daily_report.main([])

    assert exc.value.code == 2
