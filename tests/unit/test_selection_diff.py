from stock_research.reporting.diff import (
    compare_snapshots,
    load_history,
    save_snapshot,
)


def rows(*items):
    return [
        {"code": code, "name": name, "strategy_part": part}
        for code, name, part in items
    ]


def test_compare_snapshots_separates_enter_exit_and_moves():
    previous = rows(
        ("A", "甲", "1.基本价值线或附近"),
        ("B", "乙", "2.正常基本面选股"),
        ("C", "丙", "1.基本价值线或附近"),
    )
    current = rows(
        ("B", "乙", "1.基本价值线或附近"),
        ("C", "丙", "1.基本价值线或附近"),
        ("D", "丁", "2.正常基本面选股"),
    )

    result = compare_snapshots(previous, current)

    assert [item["code"] for item in result.added] == ["D"]
    assert [item["code"] for item in result.removed] == ["A"]
    assert [
        (item["code"], item["from_part"], item["to_part"])
        for item in result.moved
    ] == [("B", "2.正常基本面选股", "1.基本价值线或附近")]


def test_snapshot_history_uses_previous_distinct_date_on_same_day_rerun(tmp_path):
    path = tmp_path / "history.json"
    save_snapshot(
        path,
        "2026-07-02",
        rows(("A", "甲", "1.基本价值线或附近")),
    )
    save_snapshot(
        path,
        "2026-07-03",
        rows(("B", "乙", "2.正常基本面选股")),
    )
    save_snapshot(
        path,
        "2026-07-03",
        rows(("C", "丙", "2.正常基本面选股")),
    )

    history = load_history(path)

    assert history.previous_before("2026-07-03")[0]["code"] == "A"
    assert history.snapshot_for("2026-07-03")[0]["code"] == "C"


def test_snapshot_history_has_no_false_baseline_on_first_run(tmp_path):
    path = tmp_path / "history.json"
    save_snapshot(
        path,
        "2026-07-03",
        rows(("A", "甲", "1.基本价值线或附近")),
    )

    history = load_history(path)

    assert history.previous_before("2026-07-03") is None


def test_compare_snapshots_reports_no_change_without_duplicates():
    snapshot = rows(
        ("A", "甲", "1.基本价值线或附近"),
        ("B", "乙", "2.正常基本面选股"),
    )

    result = compare_snapshots(snapshot, list(reversed(snapshot)))

    assert result.added == ()
    assert result.removed == ()
    assert result.moved == ()
