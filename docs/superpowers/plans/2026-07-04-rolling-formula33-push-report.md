# Rolling Formula33 Push Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-day Formula33 reporting with a rolling 21-trading-day breadth trend, add previous-trading-day selection changes, send a readable complete report in two bounded PushPlus messages, and run the self-hosted Action every day.

**Architecture:** Pure rolling calculations and selection-diff rules live in `strategies`/`reporting` helpers and are covered before orchestration changes. The Formula33 pipeline fetches 61 calculation dates and exports 21 complete rolling rows; the daily report consumes only rolling fields, builds full local HTML plus two independently bounded push messages, persists dated selection snapshots, and sends both parts. The workflow removes its three-day gate while preserving the daily cron and manual entrypoint.

**Tech Stack:** Python 3.12/3.13, pandas, NumPy, pytest, HTML, PowerShell, GitHub Actions YAML, PushPlus HTTP API

---

### Task 1: Rolling Formula33 pure calculations

**Files:**
- Modify: `stock_research/strategies/formula33.py`
- Modify: `tests/unit/test_formula33_status.py`

- [ ] **Step 1: Write failing tests for rolling unique counts and linear trends**

Add imports and tests that define the desired pure API:

```python
from stock_research.strategies.formula33 import build_window_trend


def test_build_window_trend_uses_distinct_codes_in_each_21_day_window():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame(
        [
            {"date": date, "code": f"sz.{index:06d}"}
            for index, date in enumerate(dates)
        ]
        + [{"date": dates[-1], "code": "sz.000060"}]
    )

    result = build_window_trend(hits, dates, window=21, output_days=21)

    assert len(result) == 21
    assert result["window_unique_count"].tolist() == [21] * 21
    assert result["window_trend_slope"].abs().max() < 1e-12
    assert result.iloc[-1]["trend_up_streak"] == 0
    assert result.iloc[-1]["trend_down_streak"] == 0


def test_build_window_trend_counts_consecutive_positive_and_negative_slopes():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    rows = []
    for index, date in enumerate(dates):
        for code_index in range(index + 1):
            rows.append({"date": date, "code": f"sz.{code_index:06d}"})

    rising = build_window_trend(pd.DataFrame(rows), dates, window=21, output_days=21)
    assert rising.iloc[-1]["window_trend_slope"] > 0
    assert rising.iloc[-1]["trend_up_streak"] >= 5
    assert rising.iloc[-1]["trend_down_streak"] == 0

    falling_hits = pd.DataFrame(
        [row for row in rows if int(row["code"].split(".")[1]) >= 61 - dates.index(row["date"])]
    )
    falling = build_window_trend(falling_hits, dates, window=21, output_days=21)
    assert falling.iloc[-1]["window_trend_slope"] < 0
    assert falling.iloc[-1]["trend_down_streak"] >= 5
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_formula33_status.py -q
```

Expected: collection fails because `build_window_trend` does not exist.

- [ ] **Step 3: Implement the pure rolling calculation**

Add a function with this contract:

```python
def build_window_trend(xg_hits, trade_dates, window=21, output_days=21):
    dates = [str(value) for value in trade_dates]
    required = window * 2 + output_days - 2
    if len(dates) < required:
        return pd.DataFrame(columns=[
            "date", "window_unique_count", "window_trend_slope",
            "trend_up_streak", "trend_down_streak", "trend_signal",
        ])
    hit_codes = {}
    if xg_hits is not None and not xg_hits.empty:
        normalized = xg_hits[["date", "code"]].dropna().copy()
        normalized["date"] = normalized["date"].astype(str)
        for date, group in normalized.groupby("date"):
            hit_codes[date] = set(group["code"].astype(str))
    unique_rows = []
    for end_index in range(window - 1, len(dates)):
        codes = set()
        for date in dates[end_index - window + 1 : end_index + 1]:
            codes.update(hit_codes.get(date, set()))
        unique_rows.append({"date": dates[end_index], "window_unique_count": len(codes)})
    rolling = pd.DataFrame(unique_rows)
    values = rolling["window_unique_count"].astype(float)
    x_axis = np.arange(window, dtype=float)
    rolling["window_trend_slope"] = values.rolling(window).apply(
        lambda sample: float(np.polyfit(x_axis, sample, 1)[0]), raw=True
    )
    up = down = 0
    up_values, down_values, signals = [], [], []
    for slope in rolling["window_trend_slope"]:
        if pd.notna(slope) and slope > 0:
            up, down = up + 1, 0
        elif pd.notna(slope) and slope < 0:
            up, down = 0, down + 1
        else:
            up = down = 0
        up_values.append(up)
        down_values.append(down)
        signals.append("up" if up else "down" if down else "neutral")
    rolling["trend_up_streak"] = up_values
    rolling["trend_down_streak"] = down_values
    rolling["trend_signal"] = signals
    return rolling.tail(output_days).reset_index(drop=True)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all Formula33 status tests pass.

- [ ] **Step 5: Commit the pure calculation**

```powershell
git add stock_research/strategies/formula33.py tests/unit/test_formula33_status.py
git commit -m "feat: calculate rolling formula33 breadth trend"
```

### Task 2: Integrate 61-day calculation into the Formula33 pipeline

**Files:**
- Modify: `stock_research/pipelines/formula33.py`
- Modify: `tests/unit/test_formula33_status.py`

- [ ] **Step 1: Add a failing integration-level unit test for summary construction**

Expose a pure `build_formula_summary(hits_df, trade_dates, output_days=21)` helper and test:

```python
def test_formula_summary_populates_all_21_rolling_rows():
    dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d").tolist()
    hits = pd.DataFrame([
        {"signal_type": "XG", "date": date, "code": f"sz.{index:06d}"}
        for index, date in enumerate(dates)
    ])

    summary = formula33.build_formula_summary(hits, dates, output_days=21)

    assert len(summary) == 21
    assert summary["window_unique_count"].notna().all()
    assert summary["window_trend_slope"].notna().all()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the test by node id. Expected: `AttributeError` for `build_formula_summary`.

- [ ] **Step 3: Implement summary integration and calculation-date selection**

Implement `build_formula_summary` by calculating diagnostic daily counts for compatibility, calling `build_window_trend`, and merging on `date`. In `main`:

```python
calculation_days = args.lookback * 3 - 2
raw_trade_dates = get_trade_dates(calculation_days + 5, args.history_days + 90)
calculation_dates = select_trade_dates(
    raw_trade_dates, args.start_date, effective_end_date, calculation_days
)
output_dates = calculation_dates[-args.lookback:]
date_set = set(calculation_dates)
```

Use `calculation_dates` for hit collection and `build_formula_summary`, but export only `output_dates`. Add rolling fields to both workbook sheets and retain latest observation diagnostics only on the latest row.

- [ ] **Step 4: Run Formula33 tests and verify GREEN**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_formula33_status.py tests/regression/test_output_baseline.py -q
```

Expected: pass without changing legacy fixture hashes.

- [ ] **Step 5: Commit pipeline integration**

```powershell
git add stock_research/pipelines/formula33.py tests/unit/test_formula33_status.py
git commit -m "feat: export complete rolling formula33 history"
```

### Task 3: Replace right-side report logic with rolling fields

**Files:**
- Modify: `stock_research/reporting/daily_report.py`
- Modify: `tests/unit/test_daily_report_formula.py`

- [ ] **Step 1: Replace the old report test with failing rolling-trend tests**

```python
from stock_research.reporting.daily_report import render_formula_status, right_side_conclusion


def test_formula_status_only_reports_21_day_breadth_and_trend():
    text = render_formula_status({
        "window_unique_count": 193,
        "window_trend_slope": 2.14,
        "trend_up_streak": 3,
        "trend_down_streak": 0,
        "tradable_unique_count": 191,
        "suspended_count": 2,
        "unavailable_count": 6,
        "count": 15,
        "change": 3,
    })
    assert "近21个交易日三浪三技术去重193只" in text
    assert "趋势斜率+2.14" in text
    assert "连续正趋势3日" in text
    assert "当日XG" not in text
    assert "较前一交易日" not in text


def test_right_side_conclusion_uses_rolling_trend_streaks_only():
    assert right_side_conclusion({"trend_up_streak": 5, "trend_down_streak": 0})[0] == "可以右侧交易"
    assert right_side_conclusion({"trend_up_streak": 3, "trend_down_streak": 0})[0] == "可以谨慎右侧"
    assert right_side_conclusion({"trend_up_streak": 0, "trend_down_streak": 5})[0] == "暂停右侧交易"
    assert right_side_conclusion({"trend_up_streak": 0, "trend_down_streak": 3})[0] == "谨慎或暂停右侧"
```

- [ ] **Step 2: Run focused report tests and verify RED**

Expected: assertions fail because the implementation still prints single-day values.

- [ ] **Step 3: Implement rolling-only rendering and conclusions**

Update `render_formula_status` and `right_side_conclusion` to read only `window_unique_count`, `window_trend_slope`, `trend_up_streak`, and `trend_down_streak`; slope-positive streaks under three return “轻仓观察右侧”, missing/neutral data returns “等待右侧确认”.

- [ ] **Step 4: Run focused report tests and verify GREEN**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_daily_report_formula.py -q
```

- [ ] **Step 5: Commit the report logic**

```powershell
git add stock_research/reporting/daily_report.py tests/unit/test_daily_report_formula.py
git commit -m "fix: base right-side switch on rolling breadth"
```

### Task 4: Persist and compare final selection snapshots

**Files:**
- Modify: `stock_research/reporting/diff.py`
- Create: `tests/unit/test_selection_diff.py`

- [ ] **Step 1: Write failing tests for added, removed, moved, baseline, and rerun behavior**

```python
from stock_research.reporting.diff import compare_snapshots, load_history, save_snapshot


def rows(*items):
    return [{"code": code, "name": name, "strategy_part": part} for code, name, part in items]


def test_compare_snapshots_separates_enter_exit_and_moves():
    previous = rows(("A", "甲", "1.基本价值线或附近"), ("B", "乙", "2.正常基本面选股"), ("C", "丙", "1.基本价值线或附近"))
    current = rows(("B", "乙", "1.基本价值线或附近"), ("C", "丙", "1.基本价值线或附近"), ("D", "丁", "2.正常基本面选股"))
    diff = compare_snapshots(previous, current)
    assert [item["code"] for item in diff.added] == ["D"]
    assert [item["code"] for item in diff.removed] == ["A"]
    assert [(item["code"], item["from_part"], item["to_part"]) for item in diff.moved] == [
        ("B", "2.正常基本面选股", "1.基本价值线或附近")
    ]


def test_snapshot_history_uses_previous_distinct_date_on_same_day_rerun(tmp_path):
    path = tmp_path / "history.json"
    save_snapshot(path, "2026-07-02", rows(("A", "甲", "1.基本价值线或附近")))
    save_snapshot(path, "2026-07-03", rows(("B", "乙", "2.正常基本面选股")))
    save_snapshot(path, "2026-07-03", rows(("C", "丙", "2.正常基本面选股")))
    history = load_history(path)
    assert history.previous_before("2026-07-03")[0]["code"] == "A"
    assert history.snapshot_for("2026-07-03")[0]["code"] == "C"
```

- [ ] **Step 2: Run the new tests and verify RED**

Expected: imports fail because the snapshot API does not exist.

- [ ] **Step 3: Implement dated history and disjoint diff categories**

Use dataclasses `SelectionDiff(added, removed, moved)` and `SelectionHistory(snapshots)`. Normalize rows by code, sort outputs by code, save JSON atomically, retain the five latest distinct dates, and return `None` when no earlier baseline exists.

- [ ] **Step 4: Run selection diff tests and verify GREEN**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_selection_diff.py -q
```

- [ ] **Step 5: Commit snapshot support**

```powershell
git add stock_research/reporting/diff.py tests/unit/test_selection_diff.py
git commit -m "feat: track daily selection changes"
```

### Task 5: Build two complete bounded PushPlus reports

**Files:**
- Modify: `stock_research/reporting/daily_report.py`
- Modify: `stock_research/api/pushplus.py`
- Modify: `stock_research/pipelines/daily.py`
- Modify: `tests/unit/test_daily_report_formula.py`
- Modify: `tests/unit/test_pushplus.py`
- Modify: `tests/integration/test_daily_pipeline.py`

- [ ] **Step 1: Write failing tests for two messages, full coverage, length, and send failures**

Create small value/normal frames and assert:

```python
def test_build_push_reports_keeps_every_stock_and_bounds_each_part():
    values = make_stocks("V", 55, "1.基本价值线或附近")
    normal = make_stocks("N", 30, "2.正常基本面选股")
    part1, part2 = daily_report.build_push_reports(
        "2026-07-03", values, normal, formula_row(), sectors(), empty_diff(), 18000
    )
    assert len(part1) <= 18000
    assert len(part2) <= 18000
    assert all(name in part1 for name in values["name"])
    assert all(name in part2 for name in normal["name"])
    assert "结论" in part1 and "风险" in part1
    assert "结论" in part2 and "风险" in part2


def test_daily_report_sends_two_parts_in_order(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(daily_report, "build_reports", fake_two_part_report)
    monkeypatch.setattr(daily_report, "send_pushplus", lambda title, body: sent.append((title, body)) or True)
    daily_report.main([])
    assert [title for title, _ in sent] == [
        "[1/2] 2026-07-03 市场状态与价值线池",
        "[2/2] 2026-07-03 基本面候选与主线",
    ]


def test_daily_report_fails_when_either_push_fails(monkeypatch):
    results = iter([True, False])
    monkeypatch.setattr(daily_report, "send_pushplus", lambda *args: next(results))
    with pytest.raises(SystemExit) as exc:
        daily_report.main([])
    assert exc.value.code == 2
```

Add a PushPlus unit test that captures the JSON payload and asserts the fallback API guard never sends more than 18,000 characters.

- [ ] **Step 2: Run report, API, and pipeline tests and verify RED**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_daily_report_formula.py tests/unit/test_pushplus.py tests/integration/test_daily_pipeline.py -q
```

- [ ] **Step 3: Implement two structured message builders**

Use simple PushPlus-safe HTML (`h1/h2/h3`, `p`, `ol`, `table`, `b`, `hr`) and helper functions for:

- overall conclusion and rolling breadth evidence;
- selection change HTML with names, codes, and move directions;
- value-pool aggregate analysis and complete compact rows;
- normal-pool aggregate analysis, complete compact rows, and risk labels;
- sector evidence and freshness;
- a `validate_push_report(content, expected_codes, max_chars)` guard that raises before sending if content is too long or omits a code.

Return `(full_html, push_part_1, push_part_2)` from report construction. Set both CLI and API defaults to `18000`, update the daily pipeline argument, save/update the dated snapshot after local artifacts are written, then send part 1 and part 2 sequentially.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all pass.

- [ ] **Step 5: Commit two-part reporting**

```powershell
git add stock_research/reporting/daily_report.py stock_research/api/pushplus.py stock_research/pipelines/daily.py tests/unit/test_daily_report_formula.py tests/unit/test_pushplus.py tests/integration/test_daily_pipeline.py
git commit -m "feat: send complete two-part daily report"
```

### Task 6: Update strategy wording and run the Action every day

**Files:**
- Modify: `STRATEGY.md`
- Modify: `.github/workflows/stock-selection.yml`
- Create: `tests/architecture/test_workflow_schedule.py`

- [ ] **Step 1: Write a failing workflow structure test**

```python
def test_stock_selection_workflow_runs_full_pipeline_daily():
    text = (ROOT / ".github/workflows/stock-selection.yml").read_text(encoding="utf-8")
    assert 'cron: "30 8 * * *"' in text
    assert "Three-day schedule gate" not in text
    assert "steps.gate.outputs.run" not in text
    assert "Run tests" in text
    assert "Run full-market selection" in text
```

Add a strategy wording assertion that the document contains “最近21个交易日三浪三命中股票去重数” and does not contain the obsolete sentence beginning “日报必须同时报告‘当日命中数’”.

- [ ] **Step 2: Run the workflow test and verify RED**

Expected: failure because the three-day gate remains.

- [ ] **Step 3: Remove the gate and update strategy language**

Delete the gate step and `if: steps.gate.outputs.run == 'true'` conditions. Keep `if: always()` on artifact upload without the gate suffix. Replace Strategy sections 7 and 9 with the approved rolling-only wording and add the previous-trading-day enter/exit/move requirement.

- [ ] **Step 4: Run workflow and architecture tests and verify GREEN**

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/architecture -q
```

- [ ] **Step 5: Commit workflow and documentation**

```powershell
git add .github/workflows/stock-selection.yml STRATEGY.md tests/architecture/test_workflow_schedule.py
git commit -m "ci: run stock selection every day"
```

### Task 7: Full verification and real delivery

**Files:**
- Runtime outputs only: `var/exports`, `var/state`, `var/logs`

- [ ] **Step 1: Run syntax, full test, and baseline verification**

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m compileall -q apps stock_research tests
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

Expected: zero compile errors, all tests pass, and `6 baselines verified`.

- [ ] **Step 2: Generate the current report without sending and inspect both parts**

Run the report with `--no-push --max-chars 18000`. Parse the result to verify four full-HTML headings, both push part lengths, every current selection code, rolling-only wording, and selection-diff section.

- [ ] **Step 3: Send the final two-part PushPlus report**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_report --top 10 --selection-top 30 --max-chars 18000
```

Expected: `PUSH_RESULT_1 True`, `PUSH_RESULT_2 True`; each reported length is at most 18,000.

- [ ] **Step 4: Verify Action and runner readiness**

Check the Windows service is `Running/Automatic`, the GitHub runner is `online/idle` with `self-hosted,Windows,X64,my-trade`, `PUSHPLUS_TOKEN` exists as a repository secret, and `gh workflow view stock-selection.yml` parses the updated workflow.

- [ ] **Step 5: Review final diff and commit any verification-only test adjustment**

Run `git status --short`, `git diff --check`, and `git log --oneline`. No runtime artifact under `var/` may be staged.
