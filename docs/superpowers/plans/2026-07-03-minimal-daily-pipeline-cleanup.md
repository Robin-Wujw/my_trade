# 最小每日推送链路 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固定三浪三观察日停牌规则，删除非生产代码和重复产物，并证明每日推送入口与 DuckDB SQL 基础层可正常运行。

**Architecture:** 保留当前 Windows 每日编排入口和全部直接依赖，把停牌状态作为每次运行的临时状态而非永久排除。先用纯函数和回放测试固定 2026-07-02 的技术集合 186、正式集合 184，再清理文件；DuckDB 通过临时库和本地 `.data` 两层 smoke test 验收。

**Tech Stack:** Python 3.12、pandas、AkShare/Baostock、DuckDB 1.5、pytest、PowerShell、GitHub Actions

---

### Task 1: 固定观察日状态与 21 日集合语义

**Files:**
- Create: `tests/unit/test_formula33_status.py`
- Modify: `formula33Stats.py`

- [ ] **Step 1: 写失败测试**

测试必须覆盖：窗口技术去重、观察日停牌排除、数据不可用诊断、下一观察日复牌恢复，以及固定回放集合 `186 → 184`。

```python
def test_suspended_stock_is_excluded_only_for_current_observation():
    hits = pd.DataFrame([
        {"code": "sh.688072", "date": "2026-06-26"},
        {"code": "sz.000001", "date": "2026-07-02"},
    ])
    statuses = pd.DataFrame([
        {"code": "sh.688072", "observation_status": "suspended_or_no_trade"},
        {"code": "sz.000001", "observation_status": "traded"},
    ])
    technical, formal = select_window_unique_hits(hits, statuses)
    assert set(technical.code) == {"sh.688072", "sz.000001"}
    assert set(formal.code) == {"sz.000001"}
```

- [ ] **Step 2: 运行并确认测试因接口缺失而失败**

Run: `python -m pytest tests/unit/test_formula33_status.py -q`

Expected: FAIL because status helpers do not exist.

- [ ] **Step 3: 实现最小状态接口**

新增 `classify_observation_status()` 和 `select_window_unique_hits()`。状态只允许 `traded`、`suspended_or_no_trade`、`data_unavailable`；正式集合只接受 `traded`，技术集合不因观察日停牌丢失历史命中。

- [ ] **Step 4: 运行单元测试**

Run: `python -m pytest tests/unit/test_formula33_status.py -q`

Expected: PASS.

### Task 2: 接入全市场三浪三汇总和诊断

**Files:**
- Modify: `formula33Stats.py`
- Modify: `dailyReportPush.py`
- Modify: `tests/unit/test_formula33_status.py`

- [ ] **Step 1: 增加失败测试**

测试 `fetch_one_stock` 的返回中包含观察日状态；缺观察日日线不再丢弃历史 XG；抓取异常返回 `data_unavailable`。测试日报摘要能读取 `window_unique_count`、`tradable_unique_count`、`suspended_count` 和 `unavailable_count`。

- [ ] **Step 2: 改造结果流**

每只股票返回命中行和一条状态行。汇总先计算全部历史日统计，再生成技术去重集合与观察日正式集合；工作簿增加 `21日技术XG去重`、`观察日状态`，CSV 最新行增加四个覆盖字段。现有 `count/change/streak/signal` 口径不变。

- [ ] **Step 3: 回放 2026-07-02**

用已有工作簿和缓存验证技术集合 186、停牌排除 2、正式集合 184；拓荆科技和日科化学只在本观察日被排除。

### Task 3: 精简运行入口与运维文件

**Files:**
- Delete: `install_daily_task.ps1`
- Delete: `run_daily_analysis.sh`
- Move: `install_github_runner.ps1` → `scripts/admin/install_github_runner.ps1`
- Modify: `README.md`
- Modify: `docs/knowledge/operations-runbook.md`
- Modify: `.github/workflows/stock-selection.yml`

- [ ] **Step 1: 验证 Windows 编排引用清单**

确认 `run_daily_analysis.ps1` 引用的八个 Python 入口全部存在，且共享模块仍可导入。

- [ ] **Step 2: 删除或移动非生产入口**

使用补丁删除本地计划任务和 Linux 入口；保留 GitHub runner 重建脚本但移出根目录。

- [ ] **Step 3: 增加 Actions 测试门**

运行时验证加入 `duckdb`、`pytz`、`pytest`，并在正式选股前执行 `python -m pytest -q`。测试失败时后续生产步骤不得执行。

### Task 4: 清理可再生产文件

**Files:**
- Delete: `回测结果/*`
- Delete: `.cache/**/*.tmp`
- Delete: `__pycache__/`, `.pytest_cache/`
- Prune: `板块观察/`, `选股结果/`, `logs/`

- [ ] **Step 1: 生成显式保留清单**

保留 6 份回归基线、2026-07-02 最新三浪三 CSV/XLSX，以及日报、板块、基本面、合并选股、因子诊断各自最新一份。

- [ ] **Step 2: 验证所有候选路径位于仓库根目录**

PowerShell 对每个绝对路径执行 `StartsWith($workspaceRoot)` 检查，任何越界立即停止。

- [ ] **Step 3: 按显式路径删除并报告释放空间**

不删除 `.cache/formula33_kline`、财务缓存、股票池、股本缓存、凭证或基线文件。删除后再次运行基线审计。

### Task 5: DuckDB 和 SQL 双层验收

**Files:**
- Modify: `tests/integration/test_database_migrations.py`
- Modify: `tests/integration/test_run_repository.py`

- [ ] **Step 1: 增加约束和回滚失败测试**

验证非法日期、非法覆盖率、重复步骤、失败事务和只读连接不会留下脏数据。

- [ ] **Step 2: 安装运行器依赖并运行全部测试**

Run: `D:\ActionsRunner\my-trade\python\python.exe -m pip install -r requirements-actions.txt`

Run: `D:\ActionsRunner\my-trade\python\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 3: 初始化本地生产库并执行无残留 smoke test**

初始化 `.data/my_trade.duckdb`，确认四个 schema 和三个 `ops` 表；在事务中插入 smoke run、查询、回滚，最后确认 smoke `run_id` 行数为 0。

### Task 6: 最终验证与文档同步

**Files:**
- Modify: `README.md`
- Modify: `docs/knowledge/project-architecture.md`
- Modify: `docs/knowledge/operations-runbook.md`

- [ ] **Step 1: 更新当前真实架构**

文档明确：每日生产仍使用文件缓存；DuckDB 基础层和 SQL 已验证，但尚未接入全部顶层脚本。

- [ ] **Step 2: 完整验证**

Run: `python -m compileall -q src tests *.py`

Run: `python -m pytest -q`

Run: `python -m my_trade.regression.output_baseline verify tests/regression/legacy-output-v1.json`

Run: `git diff --check`

Expected: 全部退出码为 0，6 份基线通过，Windows 编排引用无缺失。
