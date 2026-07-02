# 阶段一基础边界 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有顶层脚本输出口径的前提下，建立可测试的 Python 包、不可变运行上下文、DuckDB schema 迁移和运行记录仓储。

**Architecture:** 新代码进入 `src/my_trade`，领域对象不依赖数据库；`storage` 只负责显式 DuckDB 连接、迁移和运行记录事务。第一批只建立四个 schema 与 `ops` 基础表，财务和策略业务表留给各自后续计划。

**Tech Stack:** Python 3.9+、dataclasses、DuckDB Python API、pytest

---

### Task 1: 测试与包骨架

**Files:**
- Create: `pyproject.toml`
- Create: `src/my_trade/__init__.py`
- Create: `src/my_trade/domain/__init__.py`
- Create: `src/my_trade/storage/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements-actions.txt`
- Modify: `.gitignore`

- [ ] **Step 1: 声明运行与测试依赖**

在 `requirements-actions.txt` 增加 `duckdb>=1.5,<2` 与 `pytest>=8,<10`；在 `pyproject.toml` 声明 `src` 包目录和 pytest 的 `pythonpath = ["src"]`、`testpaths = ["tests"]`。

- [ ] **Step 2: 建立最小包结构**

```python
# src/my_trade/__init__.py
"""Core package for the my_trade research pipeline."""
```

三个包的 `__init__.py` 只暴露后续任务创建的稳定接口，不包含启动副作用。

- [ ] **Step 3: 忽略运行数据库**

在 `.gitignore` 增加 `.data/`，防止生产 DuckDB 文件进入 Git。

- [ ] **Step 4: 验证测试发现机制**

Run: `python -m pytest --collect-only -q`

Expected: pytest 正常启动；在尚未增加测试时报告 `no tests collected`，而不是导入错误。

### Task 2: 不可变 RunContext 与日期门控

**Files:**
- Create: `src/my_trade/domain/run_context.py`
- Modify: `src/my_trade/domain/__init__.py`
- Test: `tests/unit/test_run_context.py`

- [ ] **Step 1: 写失败测试**

```python
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from my_trade.domain import RunContext, RunMode


def test_create_builds_traceable_run_id_and_freezes_context():
    context = RunContext.create(
        observation_date=date(2026, 6, 30),
        market_cutoff=date(2026, 6, 30),
        financial_cutoff=datetime(2026, 6, 30, 15, tzinfo=timezone.utc),
        report_period=date(2026, 3, 31),
        mode=RunMode.PRODUCTION,
        code_version="abc123",
        now=datetime(2026, 7, 2, 1, 2, 3, tzinfo=timezone.utc),
    )
    assert context.run_id.startswith("20260630-")
    with pytest.raises(FrozenInstanceError):
        context.code_version = "changed"


def test_rejects_cutoff_after_observation_date():
    with pytest.raises(ValueError, match="market_cutoff"):
        RunContext.create(
            observation_date=date(2026, 6, 30),
            market_cutoff=date(2026, 7, 1),
            financial_cutoff=datetime(2026, 6, 30, tzinfo=timezone.utc),
            report_period=date(2026, 3, 31),
            mode=RunMode.BACKTEST,
            code_version="abc123",
        )
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/test_run_context.py -q`

Expected: FAIL because `my_trade.domain.RunContext` does not exist.

- [ ] **Step 3: 实现最小领域对象**

用 `@dataclass(frozen=True, slots=True)` 定义 `RunContext`，用字符串枚举定义 `production/backtest/offline`。`create()` 必须拒绝晚于观察日的市场截止日、财务截止日和报告期，并要求带时区的时间；`run_id` 使用 `YYYYMMDD-<UTC时间>-<8位随机标识>`。

- [ ] **Step 4: 运行单元测试**

Run: `python -m pytest tests/unit/test_run_context.py -q`

Expected: PASS.

### Task 3: DuckDB 有序迁移

**Files:**
- Create: `src/my_trade/storage/database.py`
- Create: `src/my_trade/storage/migrations.py`
- Modify: `src/my_trade/storage/__init__.py`
- Test: `tests/integration/test_database_migrations.py`

- [ ] **Step 1: 写失败测试**

```python
from my_trade.storage import Database


def test_initialize_creates_schemas_and_is_idempotent(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    database.initialize()
    with database.connect(read_only=True) as connection:
        schemas = {row[0] for row in connection.execute(
            "select schema_name from information_schema.schemata"
        ).fetchall()}
        versions = connection.execute(
            "select version from ops.schema_migrations order by version"
        ).fetchall()
    assert {"raw", "core", "derived", "ops"} <= schemas
    assert versions == [(1,)]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/integration/test_database_migrations.py -q`

Expected: FAIL because `my_trade.storage.Database` does not exist.

- [ ] **Step 3: 实现显式连接与迁移**

`Database.connect()` 必须创建独立连接，禁止使用 DuckDB 全局连接；写连接先创建父目录，读连接不得创建文件。迁移 1 在一个事务中创建 `raw/core/derived/ops`、`ops.schema_migrations`、`ops.runs` 和 `ops.run_steps`，成功后登记版本；异常时回滚。

- [ ] **Step 4: 运行迁移测试**

Run: `python -m pytest tests/integration/test_database_migrations.py -q`

Expected: PASS and applying migration twice leaves one migration row.

### Task 4: 运行记录仓储

**Files:**
- Create: `src/my_trade/storage/run_repository.py`
- Modify: `src/my_trade/storage/__init__.py`
- Test: `tests/integration/test_run_repository.py`

- [ ] **Step 1: 写失败测试**

```python
from datetime import date, datetime, timezone

from my_trade.domain import RunContext, RunMode
from my_trade.storage import Database, RunRepository


def test_run_and_step_lifecycle_is_persisted(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb")
    database.initialize()
    context = RunContext.create(
        observation_date=date(2026, 6, 30),
        market_cutoff=date(2026, 6, 30),
        financial_cutoff=datetime(2026, 6, 30, tzinfo=timezone.utc),
        report_period=date(2026, 3, 31),
        mode=RunMode.OFFLINE,
        code_version="test",
    )
    repository = RunRepository(database)
    repository.start_run(context)
    repository.start_step(context.run_id, "market", context.market_cutoff)
    repository.finish_step(context.run_id, "market", status="succeeded", row_count=42, coverage=0.99)
    repository.finish_run(context.run_id, gate_status="passed")
    run = repository.get_run(context.run_id)
    assert run["gate_status"] == "passed"
    assert run["steps"][0]["row_count"] == 42
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/integration/test_run_repository.py -q`

Expected: FAIL because `RunRepository` does not exist.

- [ ] **Step 3: 实现事务化生命周期**

`start_run` 插入 `running` 运行；`start_step` 插入 `running` 步骤；两个 `finish_*` 方法只允许终态值并更新时间。重复 `run_id` 或重复 `(run_id, step_name)` 由数据库主键拒绝，未知运行更新必须抛 `LookupError`。

- [ ] **Step 4: 运行完整测试**

Run: `python -m pytest -q`

Expected: all tests PASS without network access and without creating `.data/my_trade.duckdb`.

### Task 5: 历史结果回归基线

**Files:**
- Create: `src/my_trade/regression/__init__.py`
- Create: `src/my_trade/regression/output_baseline.py`
- Create: `tests/regression/legacy-output-v1.json`
- Create: `tests/regression/test_output_baseline.py`

- [ ] **Step 1: 写失败测试**

用临时 CSV 验证审计器会检查行数、原文件 SHA-256 和按结果类型选取的关键列语义摘要；修改一个 `count`、股票代码或关键评分后必须返回可定位的差异。另一个测试读取 `legacy-output-v1.json`，确认其中固定了 3 次三浪三、2 次合并选股和 1 次因子选股。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/regression/test_output_baseline.py -q`

Expected: FAIL because `my_trade.regression.output_baseline` does not exist.

- [ ] **Step 3: 实现确定性摘要与审计**

使用标准库 `csv/json/hashlib`，按类型选择稳定业务列、按日期与代码排序、把空值保留为空字符串，再计算 canonical JSON 的 SHA-256。审计结果必须区分文件缺失、行数变化、原文件变化和语义结果变化；不依赖 pandas 或网络。

- [ ] **Step 4: 固化真实历史基线并执行本地回放审计**

基线使用以下历史产物：

```text
板块观察/formula33_stats_20260630_235623.csv
板块观察/formula33_stats_20260628_114828.csv
板块观察/formula33_stats_20260626_013200.csv
选股结果/daily_consolidated_selection_2026-06-26_181827.csv
选股结果/daily_consolidated_selection_2026-06-26_174939.csv
选股结果/factor_selection_2026-06-26_165743.csv
```

Run: `python -m my_trade.regression.output_baseline verify tests/regression/legacy-output-v1.json`

Expected: `6 baselines verified` and exit code 0.

### Task 6: 文档与兼容性验收

**Files:**
- Modify: `docs/knowledge/project-architecture.md`
- Modify: `docs/knowledge/operations-runbook.md`

- [ ] **Step 1: 记录阶段一真实状态**

说明新包目前提供运行身份和运维存储边界，现有顶层脚本尚未切换到新仓储，因此生产行为保持不变；列出后续接入入口时必须调用的 `Database.initialize()`、`RunRepository.start_run()` 和终态更新顺序。

- [ ] **Step 2: 编译与测试验收**

Run: `python -m compileall -q src tests`

Expected: exit code 0.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 3: 检查工作区范围**

Run: `git diff --check`

Expected: exit code 0; no whitespace errors. Do not commit unrelated pre-existing changes.
