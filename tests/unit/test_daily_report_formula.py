import os

import pandas as pd
import pytest

from stock_research.reporting import daily_report
from stock_research.reporting.daily_report import (
    DailyReportBundle,
    RIGHT_SIDE_ACTIVE,
    RIGHT_SIDE_EXITED,
    RIGHT_SIDE_WATCH,
    advance_formula_phase,
    build_reports,
    build_push_reports,
    latest_file,
    render_formula_status,
    render_normal_section,
    render_selection_changes,
    render_stock_section,
)
from stock_research.reporting.diff import SelectionDiff


def test_formula_status_only_reports_formal_21_day_result():
    result = render_formula_status(
        {
            "window_unique_count": 188,
            "technical_unique_count": 191,
            "window_trend_slope": 2.14,
            "trend_up_streak": 3,
            "trend_down_streak": 0,
            "tradable_unique_count": 188,
            "market_cap_unique_count": 145,
            "suspended_count": 3,
            "unavailable_count": 6,
            "count": 15,
            "change": 3,
        }
    )

    assert result == (
        "三浪三正式结果：188只（技术命中191只；观察日停牌或无交易3只已排除；"
        "数据不可用6只）。含义：近21个交易日内曾连续5日满足强势技术条件，"
        "且观察日仍可交易的去重股票。"
    )


def test_formula_phase_evaluates_every_21_day_node_and_ignores_daily_xg_counts():
    state = None
    positive_phases = []
    for index, date in enumerate(
        pd.bdate_range("2026-07-01", periods=5).strftime("%Y-%m-%d"),
        start=1,
    ):
        state = advance_formula_phase(
            [
                {
                    "date": date,
                    "window_up_streak": index,
                    "window_down_streak": 0,
                    # Deliberately contradict the 21-day-node trend. These
                    # single-day compatibility fields must have no effect.
                    "count": 100 - index,
                    "change": -1,
                    "up_streak": 0,
                    "down_streak": index,
                    "trend_up_streak": 0,
                    "trend_down_streak": index,
                }
            ],
            state,
        )
        positive_phases.append(state["phase"])

    assert positive_phases == [
        daily_report.RIGHT_SIDE_WAITING,
        daily_report.RIGHT_SIDE_WAITING,
        RIGHT_SIDE_WATCH,
        RIGHT_SIDE_WATCH,
        RIGHT_SIDE_ACTIVE,
    ]

    negative_phases = []
    for index, date in enumerate(
        pd.bdate_range("2026-07-08", periods=5).strftime("%Y-%m-%d"),
        start=1,
    ):
        state = advance_formula_phase(
            [
                {
                    "date": date,
                    "window_up_streak": 0,
                    "window_down_streak": index,
                    "count": 100 + index,
                    "change": 1,
                    "up_streak": index,
                    "down_streak": 0,
                    "trend_up_streak": index,
                    "trend_down_streak": 0,
                }
            ],
            state,
        )
        negative_phases.append(state["phase"])

    assert negative_phases == [
        RIGHT_SIDE_ACTIVE,
        RIGHT_SIDE_ACTIVE,
        RIGHT_SIDE_ACTIVE,
        RIGHT_SIDE_ACTIVE,
        RIGHT_SIDE_EXITED,
    ]
    assert state["transition_date"] == "2026-07-14"
    assert state["trigger"] == "连续5日负趋势"


def test_latest_file_excludes_newer_sample_artifact(tmp_path):
    current = tmp_path / "formula33_stats_current.csv"
    sample = tmp_path / "formula33_stats_current_sample.csv"
    current.write_text("date\n2026-07-10\n", encoding="utf-8")
    sample.write_text("date\n2026-07-11\n", encoding="utf-8")
    os.utime(current, (1, 1))
    os.utime(sample, (2, 2))

    selected = latest_file(str(tmp_path), "formula33_stats_", ".csv")

    assert selected == str(current)


def test_daily_report_rejects_mismatched_explicit_observation_dates(tmp_path):
    fundamental = tmp_path / "daily_fundamental_selection_current.csv"
    formula = tmp_path / "formula33_stats_current.csv"
    sector = tmp_path / "sector_watch_current.csv"
    pd.DataFrame([{"date": "2026-07-10"}]).to_csv(fundamental, index=False)
    pd.DataFrame([{"date": "2026-07-10"}]).to_csv(formula, index=False)
    pd.DataFrame([{"date": "2026-07-09"}]).to_csv(sector, index=False)

    with pytest.raises(ValueError, match="observation date mismatch"):
        build_reports(
            10,
            30,
            18000,
            str(fundamental),
            str(formula),
            str(sector),
        )


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
    assert "三浪三正式结果：193只" in part1 and "风险" in part1
    assert "结论" in part2 and "风险" in part2
    assert "30秒结论" in part1 and "今天怎么用" in part1
    assert "数据是否可用" in part1 and "现价÷价值线" in part1
    assert "70.0% · 右侧确认" in part1 and "低点=0%" in part1
    assert "30秒结论" in part2 and "核验顺序" in part2
    assert "70.0% · 右侧确认" in part2 and "两种分位不可混用" in part2


def test_large_value_pool_keeps_codes_and_exact_wave_percentile_in_minimal_table():
    values = make_stocks("V", 180, "1.基本价值线或附近")
    normal = make_stocks("N", 30, "2.正常基本面选股")

    part1, _part2 = build_push_reports(
        "2026-07-03",
        values,
        normal,
        pd.Series({"window_unique_count": 188, "unavailable_count": 0}),
        pd.DataFrame(),
        SelectionDiff((), (), ()),
        18000,
    )

    assert len(part1) <= 18000
    assert all(code in part1 for code in values["code"])
    assert part1.count("70.0% · 右侧确认") == len(values)


def test_selection_changes_are_separated_by_strategy_part():
    value_part = "1.基本价值线或附近"
    normal_part = "2.正常基本面选股"
    selection_diff = SelectionDiff(
        added=(
            {"code": "VALUE-ADD", "name": "价值新增", "strategy_part": value_part},
            {"code": "NORMAL-ADD", "name": "基本面新增", "strategy_part": normal_part},
        ),
        removed=(
            {"code": "VALUE-OUT", "name": "价值退出", "strategy_part": value_part},
            {"code": "NORMAL-OUT", "name": "基本面退出", "strategy_part": normal_part},
        ),
        moved=(
            {
                "code": "MOVE-TO-NORMAL",
                "name": "转入基本面",
                "strategy_part": normal_part,
                "from_part": value_part,
                "to_part": normal_part,
            },
            {
                "code": "MOVE-TO-VALUE",
                "name": "转入价值线",
                "strategy_part": value_part,
                "from_part": normal_part,
                "to_part": value_part,
            },
        ),
    )

    value_changes = render_selection_changes(selection_diff, value_part)
    normal_changes = render_selection_changes(selection_diff, normal_part)

    assert "VALUE-ADD" in value_changes
    assert "VALUE-OUT" in value_changes
    assert "NORMAL-ADD" not in value_changes
    assert "NORMAL-OUT" not in value_changes
    assert "MOVE-TO-NORMAL" not in value_changes
    assert "MOVE-TO-VALUE" in value_changes
    assert "NORMAL-ADD" in normal_changes
    assert "NORMAL-OUT" in normal_changes
    assert "MOVE-TO-NORMAL" in normal_changes
    assert "MOVE-TO-VALUE" not in normal_changes
    assert "分区变化 1只" in normal_changes
    assert normal_changes.count("MOVE-TO-NORMAL") == 1


def test_full_report_sections_place_their_own_changes_below_each_heading():
    values = make_stocks("V", 1, "1.基本价值线或附近")
    normal = make_stocks("N", 1, "2.正常基本面选股")
    value_changes = "<div>VALUE-CHANGES</div>"
    normal_changes = "<div>NORMAL-CHANGES</div>"

    value_html = render_stock_section(
        "1. 基本价值线或附近（适用股票全量）",
        values,
        changes_html=value_changes,
    )
    normal_html = render_normal_section(normal, changes_html=normal_changes)

    assert value_html.index("<h2>") < value_html.index("VALUE-CHANGES") < value_html.index("<ol>")
    assert normal_html.index("<h2>") < normal_html.index("NORMAL-CHANGES") < normal_html.index("<h3>")


def test_push_reports_do_not_mix_strategy_change_lists():
    value_part = "1.基本价值线或附近"
    normal_part = "2.正常基本面选股"
    values = make_stocks("V", 1, value_part)
    normal = make_stocks("N", 1, normal_part)
    selection_diff = SelectionDiff(
        added=(
            {"code": "VALUE-ADD", "name": "价值新增", "strategy_part": value_part},
            {"code": "NORMAL-ADD", "name": "基本面新增", "strategy_part": normal_part},
        ),
        removed=(
            {"code": "VALUE-OUT", "name": "价值退出", "strategy_part": value_part},
            {"code": "NORMAL-OUT", "name": "基本面退出", "strategy_part": normal_part},
        ),
        moved=(
            {
                "code": "MOVE-TO-NORMAL",
                "name": "转入基本面",
                "strategy_part": normal_part,
                "from_part": value_part,
                "to_part": normal_part,
            },
        ),
    )

    part1, part2 = build_push_reports(
        "2026-07-03",
        values,
        normal,
        pd.Series({"window_unique_count": 188}),
        pd.DataFrame(),
        selection_diff,
        18000,
    )

    assert "VALUE-ADD" in part1
    assert "VALUE-OUT" in part1
    assert "NORMAL-ADD" not in part1
    assert "NORMAL-OUT" not in part1
    assert "MOVE-TO-NORMAL" not in part1
    assert "NORMAL-ADD" in part2
    assert "NORMAL-OUT" in part2
    assert "VALUE-ADD" not in part2
    assert "VALUE-OUT" not in part2
    assert part2.count("MOVE-TO-NORMAL") == 1


def test_build_reports_has_no_global_mixed_change_list(monkeypatch, tmp_path):
    value_part = "1.基本价值线或附近"
    normal_part = "2.正常基本面选股"
    values = make_stocks("V", 2, value_part)
    normal = make_stocks("N", 1, normal_part)
    values.loc[:, "code"] = ["VALUE-ADD", "MOVE-TO-VALUE"]
    values.loc[:, "name"] = ["价值新增", "转入价值线"]
    normal.loc[:, "code"] = ["NORMAL-ADD"]
    normal.loc[:, "name"] = ["基本面新增"]
    stocks = pd.concat([values, normal], ignore_index=True)
    fundamental = tmp_path / "daily_fundamental_selection_current.csv"
    formula = tmp_path / "formula33_stats_current.csv"
    sector = tmp_path / "sector_watch_current.csv"
    stocks.to_csv(fundamental, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-07-03",
                "window_unique_count": 188,
                "window_up_streak": 5,
                "window_down_streak": 0,
            }
        ]
    ).to_csv(formula, index=False)
    pd.DataFrame(
        [{"date": "2026-07-03", "board": "半导体", "final_score": 60}]
    ).to_csv(sector, index=False)

    class History:
        def previous_before(self, _date):
            return [
                {"code": "VALUE-OUT", "name": "价值退出", "strategy_part": value_part},
                {"code": "NORMAL-OUT", "name": "基本面退出", "strategy_part": normal_part},
                {"code": "MOVE-TO-VALUE", "name": "转入价值线", "strategy_part": normal_part},
            ]

    monkeypatch.setattr(daily_report, "load_history", lambda _path: History())
    monkeypatch.setattr(daily_report, "load_formula_phase_state", lambda _path: None)

    bundle = build_reports(
        10,
        30,
        18000,
        str(fundamental),
        str(formula),
        str(sector),
    )

    first_heading = bundle.full_html.index("<h2>1.")
    second_heading = bundle.full_html.index("<h2>2.")
    third_heading = bundle.full_html.index("<h2>3.")
    value_section = bundle.full_html[first_heading:second_heading]
    normal_section = bundle.full_html[second_heading:third_heading]
    assert "与上一交易日相比" not in bundle.full_html[:first_heading]
    assert "VALUE-ADD" in value_section and "VALUE-OUT" in value_section
    assert "MOVE-TO-VALUE" in value_section
    assert "NORMAL-ADD" not in value_section and "NORMAL-OUT" not in value_section
    assert "NORMAL-ADD" in normal_section and "NORMAL-OUT" in normal_section
    assert "VALUE-ADD" not in normal_section and "VALUE-OUT" not in normal_section
    assert "MOVE-TO-VALUE" not in normal_section
    assert bundle.full_html.count("MOVE-TO-VALUE") == 2


def test_value_pool_keeps_stock_above_recovery_50():
    values = make_stocks("V", 1, "1.基本价值线或附近")
    values.loc[0, "price_to_value"] = 0.8
    values.loc[0, "wave_pct"] = 75
    normal = make_stocks("N", 1, "2.正常基本面选股")

    part1, _ = build_push_reports(
        "2026-07-03",
        values,
        normal,
        pd.Series({"window_unique_count": 188}),
        pd.DataFrame(),
        SelectionDiff((), (), ()),
        18000,
    )

    assert values.iloc[0]["code"] in part1


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
