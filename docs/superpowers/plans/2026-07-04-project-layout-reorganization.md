# Project Layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库重构为 `apps/ + stock_research/ + config/ + scripts/ + tests/ + docs/ + var/`，删除旧根目录脚本和重复产物，并以完整测试与回归证明业务行为未发生未批准变化。

**Architecture:** `apps` 是唯一进程入口，`stock_research` 是唯一可导入核心包；外部通信进入 `api`，路径与运行上下文进入 `core`，纯计算进入 `indicators`，业务筛选进入 `strategies`，执行顺序进入 `pipelines`，格式化与推送进入 `reporting`。运行数据统一由 `ProjectPaths` 定位到 `var`，旧路径只在一次性迁移清单中出现。

**Tech Stack:** Python 3.12/3.13、pandas、AkShare、Baostock、DuckDB 1.5、pytest 9、PowerShell、GitHub Actions

---

## File structure locked by this plan

```text
apps/
  __init__.py
  daily_pipeline.py
  formula33.py
  sector_analysis.py
  factor_selection.py
  fundamental_update.py
  fundamental_selection.py
  daily_report.py
  pipeline_alert.py
stock_research/
  __init__.py
  api/{__init__,akshare,baostock,pushplus,retry}.py
  core/{__init__,as_of,config,errors,paths,run_context}.py
  storage/{__init__,database,migrations,run_repository}.py
  market/{__init__,fundamentals,prices,sectors,universe}.py
  indicators/{__init__,factors,formula33,sector_metrics,waves}.py
  strategies/{__init__,factor_selection,formula33,fundamental_selection,sector_watch}.py
  pipelines/{__init__,daily,factor_selection,formula33,fundamental_selection,fundamental_update,sector_analysis}.py
  reporting/{__init__,alerts,daily_report,diff,exports}.py
  regression/{__init__,output_baseline}.py
config/
  pipeline.toml
  requirements/actions.txt
scripts/
  run_daily_analysis.ps1
  admin/install_github_runner.ps1
tests/
  architecture/test_module_boundaries.py
  fixtures/regression/*.csv
  integration/*.py
  regression/*.py
  unit/*.py
var/  # ignored; runtime only
```

### Task 1: Lock the current baseline and new architecture rules

**Files:**
- Create: `tests/architecture/test_module_boundaries.py`
- Modify: `tests/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing root-layout and import-boundary tests**

```python
# tests/architecture/test_module_boundaries.py
import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
OLD_MODULES = {
    "dailyFundamentalSelect", "dailyReportPush", "factorStock",
    "formula33Stats", "fullMarketFundamentalUpdate", "pipelineAlert",
    "point_in_time", "sectorStats", "sectorWatch", "trade_utils", "wave_utils",
}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_root_contains_no_business_python_files():
    assert [path.name for path in ROOT.glob("*.py")] == []


def test_old_module_names_are_not_imported():
    offenders = {}
    for path in [*ROOT.glob("apps/**/*.py"), *ROOT.glob("stock_research/**/*.py")]:
        stale = {name for name in imports(path) if name.split(".")[0] in OLD_MODULES}
        if stale:
            offenders[path.relative_to(ROOT).as_posix()] = sorted(stale)
    assert offenders == {}


def test_core_never_depends_on_outer_layers():
    forbidden = ("apps", "stock_research.pipelines", "stock_research.reporting", "stock_research.strategies")
    offenders = {}
    for path in ROOT.glob("stock_research/core/*.py"):
        bad = {name for name in imports(path) if name.startswith(forbidden)}
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}


def test_api_never_depends_on_strategy_or_pipeline():
    forbidden = ("stock_research.strategies", "stock_research.pipelines")
    offenders = {}
    for path in ROOT.glob("stock_research/api/*.py"):
        bad = {name for name in imports(path) if name.startswith(forbidden)}
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}
```

- [ ] **Step 2: Run the tests and verify they fail for the current root scripts**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/architecture/test_module_boundaries.py -q --basetemp .test-tmp/plan-task-1 -p no:cacheprovider
```

Expected: FAIL at `test_root_contains_no_business_python_files` and before the new package exists.

- [ ] **Step 3: Make pytest use the project runtime directory**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = [".", "src"]
testpaths = ["tests"]
addopts = "--basetemp=var/tmp/pytest -p no:cacheprovider"
```

Keep `src` only during Tasks 1-2 so the existing `my_trade` tests remain runnable. Task 3 removes `src` from `pythonpath` and package discovery immediately after imports switch to `stock_research`. Keep `tests/conftest.py` free of path mutation; it may contain only shared fixtures.

- [ ] **Step 4: Re-run the existing baseline independently of the expected architecture failure**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit tests/integration tests/regression -q --basetemp .test-tmp/plan-task-1-baseline -p no:cacheprovider
```

Expected: `27 passed`.

- [ ] **Step 5: Commit the guard tests**

```powershell
git add tests/architecture/test_module_boundaries.py tests/conftest.py pyproject.toml
git commit -m "test: lock project module boundaries"
```

### Task 2: Establish paths, configuration, and the renamed foundation package

**Files:**
- Create: `stock_research/__init__.py`
- Create: `stock_research/core/__init__.py`
- Create: `stock_research/core/paths.py`
- Create: `stock_research/core/config.py`
- Create: `config/pipeline.toml`
- Create: `tests/unit/test_paths.py`
- Create: `tests/unit/test_config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing path tests**

```python
# tests/unit/test_paths.py
from pathlib import Path
from stock_research.core.paths import ProjectPaths


def test_project_paths_put_all_runtime_files_under_var(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    assert paths.runtime_root == tmp_path / "var"
    assert paths.cache == tmp_path / "var" / "cache"
    assert paths.database == tmp_path / "var" / "data" / "my_trade.duckdb"
    assert paths.selection_exports == tmp_path / "var" / "exports" / "selection"
    assert paths.market_exports == tmp_path / "var" / "exports" / "market"
    assert paths.report_exports == tmp_path / "var" / "exports" / "reports"


def test_runtime_root_can_be_overridden_without_moving_project(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("STOCK_RESEARCH_VAR", str(runtime))
    paths = ProjectPaths(project_root=tmp_path / "project")
    assert paths.runtime_root == runtime.resolve()
```

- [ ] **Step 2: Run and verify missing-package failure**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_paths.py -q`

Expected: FAIL with `ModuleNotFoundError: stock_research`.

- [ ] **Step 3: Implement the path authority**

```python
# stock_research/core/paths.py
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())

    @property
    def runtime_root(self) -> Path:
        override = os.environ.get("STOCK_RESEARCH_VAR")
        return Path(override).resolve() if override else self.project_root / "var"

    @property
    def cache(self) -> Path: return self.runtime_root / "cache"
    @property
    def database(self) -> Path: return self.runtime_root / "data" / "my_trade.duckdb"
    @property
    def exports(self) -> Path: return self.runtime_root / "exports"
    @property
    def selection_exports(self) -> Path: return self.exports / "selection"
    @property
    def market_exports(self) -> Path: return self.exports / "market"
    @property
    def report_exports(self) -> Path: return self.exports / "reports"
    @property
    def logs(self) -> Path: return self.runtime_root / "logs"
    @property
    def state(self) -> Path: return self.runtime_root / "state"
    @property
    def secrets(self) -> Path: return self.runtime_root / "secrets"
    @property
    def tmp(self) -> Path: return self.runtime_root / "tmp"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths(PROJECT_ROOT)
```

- [ ] **Step 4: Write failing configuration precedence tests**

```python
# tests/unit/test_config.py
from stock_research.core.config import load_pipeline_config


def test_environment_overrides_toml(monkeypatch, tmp_path):
    path = tmp_path / "pipeline.toml"
    path.write_text(
        '[factor]\nworkers = 1\n'
        '[formula33]\nworkers = 2\nsleep = 0.2\nretries = 5\n'
        '[sector]\nsleep = 0.3\nretries = 5\n'
        '[fundamental]\nmax_updates = 100\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("FORMULA33_WORKERS", "4")
    config = load_pipeline_config(path)
    assert config.formula33_workers == 4
```

- [ ] **Step 5: Implement typed config and committed defaults**

Use `tomllib` on Python 3.11+ and `tomli` on older supported interpreters. `PipelineConfig` must validate positive worker/retry values and coverage values in `[0, 1]`. `config/pipeline.toml` must reproduce the current PowerShell defaults: factor workers 1, formula workers 1, formula sleep 0.2, retries 5, sector sleep 0.3, sector retries 5, financial updates 100.

```python
from dataclasses import dataclass
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class PipelineConfig:
    factor_workers: int
    formula33_workers: int
    formula33_sleep: float
    formula33_retries: int
    sector_sleep: float
    sector_retries: int
    financial_updates: int

    def __post_init__(self):
        if self.factor_workers < 1 or self.formula33_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.formula33_retries < 1 or self.sector_retries < 1:
            raise ValueError("retry counts must be positive")


def load_pipeline_config(path: Path) -> PipelineConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return PipelineConfig(
        factor_workers=int(os.environ.get("FACTOR_WORKERS", raw["factor"]["workers"])),
        formula33_workers=int(os.environ.get("FORMULA33_WORKERS", raw["formula33"]["workers"])),
        formula33_sleep=float(os.environ.get("FORMULA33_SLEEP", raw["formula33"]["sleep"])),
        formula33_retries=int(os.environ.get("FORMULA33_RETRIES", raw["formula33"]["retries"])),
        sector_sleep=float(os.environ.get("SECTOR_SLEEP", raw["sector"]["sleep"])),
        sector_retries=int(os.environ.get("SECTOR_RETRIES", raw["sector"]["retries"])),
        financial_updates=int(os.environ.get("FINANCIAL_UPDATES", raw["fundamental"]["max_updates"])),
    )
```

- [ ] **Step 6: Switch package discovery to the repository root**

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["apps*", "stock_research*"]
exclude = ["tests*"]
```

Add `tomli>=2; python_version < '3.11'` to project dependencies.

- [ ] **Step 7: Run focused tests and commit**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_paths.py tests/unit/test_config.py -q`

Expected: PASS.

```powershell
git add stock_research config/pipeline.toml tests/unit/test_paths.py tests/unit/test_config.py pyproject.toml
git commit -m "feat: establish stock research package foundation"
```

### Task 3: Rename domain, storage, regression, and controlled fixtures

**Files:**
- Create: `stock_research/core/run_context.py`
- Create: `stock_research/storage/database.py`
- Create: `stock_research/storage/migrations.py`
- Create: `stock_research/storage/run_repository.py`
- Create: `stock_research/regression/output_baseline.py`
- Create: `tests/fixtures/regression/*.csv`
- Modify: `tests/unit/test_run_context.py`
- Modify: `tests/integration/test_database_migrations.py`
- Modify: `tests/integration/test_run_repository.py`
- Modify: `tests/regression/test_output_baseline.py`
- Modify: `tests/regression/legacy-output-v1.json`
- Delete: `src/my_trade/`

- [ ] **Step 1: Change tests to import `stock_research` and point the manifest at fixtures**

All imports become:

```python
from stock_research.core import RunContext, RunMode
from stock_research.storage import Database, RunRepository
from stock_research.regression.output_baseline import build_entry, compare_entry
```

Update the manifest to `"root": "../fixtures/regression"` and use file basenames below that root.

- [ ] **Step 2: Copy the six baseline files into `tests/fixtures/regression` without transforming bytes**

The six source paths are exactly those listed in `tests/regression/legacy-output-v1.json`. Verify each copied file SHA-256 against the manifest before changing the manifest.

- [ ] **Step 3: Run and verify tests fail because renamed modules are absent**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_run_context.py tests/integration tests/regression -q`

Expected: collection FAIL for missing `stock_research.core`/`storage`/`regression` implementations.

- [ ] **Step 4: Move implementations with only import and default-path changes**

Preserve the current implementations byte-for-byte except:

```python
# stock_research/storage/database.py
from stock_research.core.paths import PATHS

class Database:
    def __init__(self, path=PATHS.database, code_version="unknown"):
        self.path = Path(path)
        self.code_version = str(code_version)
```

and:

```python
# stock_research/storage/run_repository.py
from stock_research.core import RunContext
```

- [ ] **Step 5: Verify all foundation tests and six historical outputs**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_run_context.py tests/integration tests/regression -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

Expected: tests PASS and `6 baselines verified`.

- [ ] **Step 6: Delete `src/my_trade` only after the new tests pass, then commit**

```powershell
git add stock_research tests pyproject.toml
git add -u src/my_trade
git commit -m "refactor: rename core package to stock research"
```

### Task 4: Split shared utilities, time rules, external APIs, and pure wave calculations

**Files:**
- Create: `stock_research/core/as_of.py`
- Create: `stock_research/api/retry.py`
- Create: `stock_research/api/pushplus.py`
- Create: `stock_research/reporting/diff.py`
- Create: `stock_research/indicators/waves.py`
- Create: `tests/unit/test_api_retry.py`
- Create: `tests/unit/test_pushplus.py`
- Create: `tests/unit/test_waves.py`
- Modify: existing users of `trade_utils.py`, `point_in_time.py`, and `wave_utils.py`

- [ ] **Step 1: Add unit tests for retry, token precedence, diff rendering, and wave functions**

```python
# tests/unit/test_api_retry.py
import pytest
from stock_research.api.retry import call_with_retry


def test_retry_uses_exact_attempt_count_and_raises_last_error(monkeypatch):
    calls = []
    monkeypatch.setattr("stock_research.api.retry.time.sleep", lambda _: None)
    def fail():
        calls.append(len(calls) + 1)
        raise RuntimeError(f"failure-{len(calls)}")
    with pytest.raises(RuntimeError, match="failure-3"):
        call_with_retry(fail, retries=3, delay=0)
    assert calls == [1, 2, 3]


# tests/unit/test_pushplus.py
from stock_research.api.pushplus import get_pushplus_token
from stock_research.core.paths import ProjectPaths


def test_pushplus_token_prefers_environment_then_local_secret(monkeypatch, tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.secrets.mkdir(parents=True)
    (paths.secrets / "pushplus_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("PUSHPLUS_TOKEN", "env-token")
    assert get_pushplus_token(paths) == "env-token"
    monkeypatch.delenv("PUSHPLUS_TOKEN")
    assert get_pushplus_token(paths) == "file-token"


# tests/unit/test_waves.py
from stock_research.indicators.waves import calc_wave_pct, wave_levels


def test_wave_levels_are_deterministic():
    assert calc_wave_pct(10, 20, 15) == 50.0
    assert wave_levels(10, 20, current=16)["level_625"] == 16.25
```

- [ ] **Step 2: Extract exact function groups without behavior changes**

Move:

- `point_in_time.py:20-77` → `core/as_of.py`;
- `trade_utils.py:105-116` and duplicate `call_with_backoff` helpers → `api/retry.py`;
- `trade_utils.py:18-28,68-102` → `api/pushplus.py`;
- `trade_utils.py:31-65` → `reporting/diff.py` and state paths from `ProjectPaths`;
- `wave_utils.py:12-166` → `indicators/waves.py`.

Use these public names unchanged so business functions can be moved independently.

- [ ] **Step 3: Run focused tests**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_api_retry.py tests/unit/test_pushplus.py tests/unit/test_waves.py tests/unit/test_run_context.py -q`

Expected: PASS.

- [ ] **Step 4: Commit the shared boundaries**

```powershell
git add stock_research tests/unit
git commit -m "refactor: separate shared API and indicator utilities"
```

### Task 5: Split and migrate the Formula33 flow

**Files:**
- Create: `stock_research/api/akshare.py`
- Create: `stock_research/api/baostock.py`
- Create: `stock_research/market/prices.py`
- Create: `stock_research/market/universe.py`
- Create: `stock_research/indicators/formula33.py`
- Create: `stock_research/strategies/formula33.py`
- Create: `stock_research/pipelines/formula33.py`
- Create: `stock_research/reporting/exports.py`
- Create: `apps/formula33.py`
- Modify: `tests/unit/test_formula33_status.py`
- Delete: `formula33Stats.py`

- [ ] **Step 1: Redirect tests to the target boundaries**

```python
from stock_research.indicators.formula33 import calc_kdj_k, calc_rsi, calc_wr
from stock_research.strategies.formula33 import (
    classify_observation_status,
    select_window_unique_hits,
)
from stock_research.pipelines import formula33 as formula33_pipeline
```

Monkeypatch `formula33_pipeline.load_kline_with_cache` in fetch tests.

- [ ] **Step 2: Run and verify missing-module failures**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_formula33_status.py -q`

Expected: collection FAIL until the modules exist.

- [ ] **Step 3: Extract pure and strategy functions**

Move exact functions:

- `tdx_sma`, `calc_kdj_k`, `calc_wr`, `calc_rsi` → `indicators/formula33.py`;
- `classify_observation_status`, `select_window_unique_hits`, `calc_streaks` → `strategies/formula33.py`.

These modules may import pandas/numpy but must not import AkShare, Baostock, filesystem paths, or reporting.

- [ ] **Step 4: Extract provider and market access**

Move AkShare-specific functions (`get_trade_dates_akshare`, `get_universe_akshare`, `load_stock_basic_akshare`, market-capital and kline AkShare loaders) to `api/akshare.py`; move `to_bs_code` and `load_kline_baostock` to `api/baostock.py`. Place date selection, cache normalization, cached kline loading and universe fallback in `market/prices.py` and `market/universe.py` using `PATHS.cache`.

- [ ] **Step 5: Create pipeline and thin app**

`pipelines/formula33.py` owns `fetch_one_stock`, task fan-out, diagnostics, and `run(args)`. `reporting/exports.py` owns workbook/CSV writing. `apps/formula33.py` contains the former argument definitions and:

```python
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return formula33_pipeline.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Pass Formula33 unit tests and CLI smoke**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_formula33_status.py -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.formula33 --help
```

Expected: tests PASS; help exits 0.

- [ ] **Step 7: Delete old file, scan imports, and commit**

Run: `Get-ChildItem -Recurse -File -Include *.py | Select-String 'formula33Stats'`

Expected: no production or test matches.

```powershell
git add apps stock_research tests/unit/test_formula33_status.py
git add -u formula33Stats.py
git commit -m "refactor: modularize formula33 pipeline"
```

### Task 6: Split and migrate sector analysis

**Files:**
- Create: `stock_research/market/sectors.py`
- Create: `stock_research/indicators/sector_metrics.py`
- Create: `stock_research/strategies/sector_watch.py`
- Create: `stock_research/pipelines/sector_analysis.py`
- Create: `apps/sector_analysis.py`
- Create: `tests/unit/test_sector_metrics.py`
- Delete: `sectorStats.py`
- Delete: `sectorWatch.py`

- [ ] **Step 1: Add fixed-frame tests for sector metrics and mainline scoring**

```python
# tests/unit/test_sector_metrics.py
import pandas as pd
import pytest

from stock_research.indicators.sector_metrics import candle_label, pct_change
from stock_research.market.sectors import classify_group, normalize_board_name
from stock_research.strategies.sector_watch import score_direct


def test_sector_helpers_keep_current_classification_and_candle_rules():
    assert classify_group("半导体设备") == "半导体"
    assert normalize_board_name("通信行业板块") == "通信"
    assert pct_change(pd.Series([100.0, 105.0]), 1) == pytest.approx(0.05)
    assert candle_label({"pct_chg": 0.06}) == "长阳"
    assert candle_label({"pct_chg": -0.03}) == "中阴"
    assert score_direct(5, 0, 10) == 50.0
```

- [ ] **Step 2: Verify tests fail for missing modules**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_sector_metrics.py -q`

Expected: collection FAIL.

- [ ] **Step 3: Extract exact boundaries**

- Data/cache loaders and board normalization from both scripts → `market/sectors.py`;
- `pct_change`, `candle_label`, count matrices, open patterns and board metric calculations → `indicators/sector_metrics.py`;
- `score_direct`, mainline row selection and constituent selection → `strategies/sector_watch.py`;
- both current `main` flows → `pipelines/sector_analysis.py` as `run_statistics(args)` and `run_watch(args)`.

`apps/sector_analysis.py` exposes subcommands `stats` and `watch` and contains all CLI option definitions.

- [ ] **Step 4: Pass tests and both CLI help paths**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_sector_metrics.py -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.sector_analysis stats --help
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.sector_analysis watch --help
```

- [ ] **Step 5: Delete old files and commit**

```powershell
git add apps stock_research tests/unit/test_sector_metrics.py
git add -u sectorStats.py sectorWatch.py
git commit -m "refactor: modularize sector analysis"
```

### Task 7: Decompose the factor-selection monolith

**Files:**
- Create: `stock_research/market/fundamentals.py`
- Create: `stock_research/indicators/factors.py`
- Create: `stock_research/strategies/factor_selection.py`
- Create: `stock_research/pipelines/factor_selection.py`
- Create: `stock_research/reporting/factor_report.py`
- Create: `apps/factor_selection.py`
- Create: `tests/unit/test_factor_indicators.py`
- Create: `tests/unit/test_factor_strategy.py`
- Delete: `factorStock.py`

- [ ] **Step 1: Pin existing pure calculations before extraction**

```python
# tests/unit/test_factor_indicators.py
from factorStock import parse_float, parse_pct, parse_yi, remove_outliers, score_direct


def test_factor_parsers_and_scalers_keep_current_behavior():
    assert parse_yi("1.5亿") == 150_000_000
    assert parse_pct("12.5%") == 0.125
    assert parse_float("1,234.5") == 1234.5
    assert remove_outliers([10, 11, 12, 100]) == [10, 11, 12]
    assert score_direct(5, 0, 10) == 50


# tests/unit/test_factor_strategy.py
import pytest
from factorStock import (
    build_risk_flags, calc_high_quality_score, calc_low_value_score,
    calc_total_score, classify_selection_bucket,
)


def test_factor_bucket_and_scores_keep_current_rules():
    row = {
        "method": "VALUE", "price_to_value": 0.80, "quality_score": 80,
        "liquidity_score": 60, "trend_score": 70,
    }
    assert classify_selection_bucket(row) == ("低估且高质量", "深度低估且高质量")
    assert calc_low_value_score("VALUE", 80, 70, 60, 50) == pytest.approx(72.0)
    assert calc_high_quality_score("VALUE", 80, 70, 60, 50) == pytest.approx(64.0)
    score, mode = calc_total_score("RIGHT", 0, 80, 70, 60)
    assert score == pytest.approx(69.5)
    assert mode == "右侧趋势"


def test_factor_risk_flags_remain_explainable():
    row = {
        "method": "VALUE", "selection_bucket": "观察池", "valuation_score": 40,
        "quality_score": 40, "trend_score": 40, "liquidity_score": 30,
        "price_to_value": 0.20, "technical_flags_list": ["量能异常"],
    }
    assert build_risk_flags(row) == "估值优势弱、质量偏弱、趋势未确认、流动性偏弱、价值线折价异常需复核、量能异常"
```

- [ ] **Step 2: Run the tests against current imports to establish green characterization**

Run: `& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_factor_indicators.py tests/unit/test_factor_strategy.py -q`

Expected: PASS while imports still target `factorStock`.

- [ ] **Step 3: Move provider-neutral calculations**

Move these exact functions to `indicators/factors.py`: `parse_yi`, `parse_pct`, `parse_float`, `clamp`, `score_direct`, `score_inverse`, `remove_outliers`, `cn_sma`, `calc_kd_lines`, `calc_rsi_999`, `detect_kd_divergence`, `get_technical_metrics`, `calc_mainline_metrics`, and `get_history_metrics`.

- [ ] **Step 4: Move strategy decisions**

Move `classify_method`, candidate predicates, `classify_selection_bucket`, score gates, risk flags, score calculators, `get_block_reason`, `get_value_watch_rows`, and `score_stock` to `strategies/factor_selection.py`. Inject market/fundamental values into `score_stock`; it must not call AkShare directly.

- [ ] **Step 5: Move data access and provider-specific valuation**

Move report-date, bonus, EPS, financial cache, benchmark, industry, profit, PE/PB and value-line fetch/normalization functions to `market/fundamentals.py`; provider-specific calls go through `api/akshare.py`. Preserve the existing fallback order and cache payload fields.

- [ ] **Step 6: Move execution and presentation**

Move worker initialization, task fan-out, cache fallback and diagnostics to `pipelines/factor_selection.py`. Move HTML tables, theme summaries, push content, compact rows and console report functions to `reporting/factor_report.py`. `apps/factor_selection.py` contains former CLI options and calls `pipeline.run(args)`.

- [ ] **Step 7: Redirect characterization tests and run complete factor tests**

Tests must now import only `stock_research.indicators.factors` and `stock_research.strategies.factor_selection`.

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_factor_indicators.py tests/unit/test_factor_strategy.py -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.factor_selection --help
```

Expected: PASS and exit 0.

- [ ] **Step 8: Delete the monolith only after no imports remain, then commit**

```powershell
git add apps stock_research tests/unit
git add -u factorStock.py
git commit -m "refactor: decompose factor selection pipeline"
```

### Task 8: Migrate fundamental update, fundamental selection, reporting, and alerting

**Files:**
- Create: `stock_research/strategies/fundamental_selection.py`
- Create: `stock_research/pipelines/fundamental_update.py`
- Create: `stock_research/pipelines/fundamental_selection.py`
- Create: `stock_research/pipelines/daily_report.py`
- Create: `stock_research/reporting/daily_report.py`
- Create: `stock_research/reporting/alerts.py`
- Create: `apps/fundamental_update.py`
- Create: `apps/fundamental_selection.py`
- Create: `apps/daily_report.py`
- Create: `apps/pipeline_alert.py`
- Modify: `tests/unit/test_daily_report_formula.py`
- Create: `tests/unit/test_fundamental_selection.py`
- Delete: `fullMarketFundamentalUpdate.py`
- Delete: `dailyFundamentalSelect.py`
- Delete: `dailyReportPush.py`
- Delete: `pipelineAlert.py`
- Delete: `point_in_time.py`
- Delete: `trade_utils.py`
- Delete: `wave_utils.py`

- [ ] **Step 1: Redirect the existing report test and add selection characterization tests**

```python
from stock_research.reporting.daily_report import render_formula_status
```

Add the following fixed-input tests:

```python
# tests/unit/test_fundamental_selection.py
import pytest

from stock_research.pipelines.daily_report import ensure_same_observation_date
from stock_research.strategies.fundamental_selection import (
    growth_risk, quality_detail, value_method_reason,
)


def test_fundamental_explanations_keep_current_wording():
    detail = quality_detail(1.50, 0.50, 100)
    assert "扣非EPS为1.50元，盈利能力较强" in detail
    assert "扣非利润同比50.0%，增长较强" in detail
    assert "近年扣非盈利稳定性较高" in detail
    reason = value_method_reason("计算机、通信", 120, 1.20, 0.20)
    assert "属于制造业" in reason
    assert "市值120.0亿元" in reason
    assert "超过300%" in growth_risk(3.01)
    assert "同比为负" in growth_risk(-0.01)


def test_report_rejects_mixed_observation_dates():
    with pytest.raises(ValueError, match="observation date mismatch"):
        ensure_same_observation_date({
            "formula33": "2026-07-02",
            "selection": "2026-07-03",
        })
```

`ensure_same_observation_date` returns the one shared date for non-empty inputs and raises before rendering or pushing when dates differ.

- [ ] **Step 2: Extract update and selection boundaries**

Move snapshot building, coverage calculation, industry map and incremental update coordination to `pipelines/fundamental_update.py`. Move pure row eligibility/reason functions to `strategies/fundamental_selection.py`; file/cache loading and row coordination belong in `pipelines/fundamental_selection.py`.

- [ ] **Step 3: Extract daily report and alerting**

Move formatting/render functions from `dailyReportPush.py` into `reporting/daily_report.py`, input resolution and push decision into `pipelines/daily_report.py`, and alert content into `reporting/alerts.py`. PushPlus transport must only come from `api.pushplus`.

- [ ] **Step 4: Add four thin app modules**

Each app uses the same pattern:

```python
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return pipeline.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests and CLI smoke**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/unit/test_daily_report_formula.py tests/unit/test_fundamental_selection.py -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.fundamental_update --help
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.fundamental_selection --help
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_report --help
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.pipeline_alert --help
```

Expected: all PASS/exit 0.

- [ ] **Step 6: Delete replaced files and commit**

```powershell
git add apps stock_research tests
git add -u fullMarketFundamentalUpdate.py dailyFundamentalSelect.py dailyReportPush.py pipelineAlert.py point_in_time.py trade_utils.py wave_utils.py
git commit -m "refactor: migrate fundamentals and reporting"
```

### Task 9: Replace the PowerShell chain with one application entry

**Files:**
- Create: `stock_research/pipelines/daily.py`
- Create: `apps/daily_pipeline.py`
- Create: `scripts/run_daily_analysis.ps1`
- Create: `config/requirements/actions.txt`
- Modify: `.github/workflows/stock-selection.yml`
- Create: `tests/integration/test_daily_pipeline.py`
- Create: `tests/architecture/test_entrypoints.py`
- Delete: `run_daily_analysis.ps1`
- Delete: `requirements-actions.txt`

- [ ] **Step 1: Add orchestration tests with fake step callables**

Test this contract:

```python
result = run_daily_pipeline(steps=fake_steps, config=config, no_push=True)
assert result.failed_steps == ()
assert result.skipped_steps == ()
assert [call.name for call in calls] == [
    "formula33", "sector_stats", "sector_watch", "factor_selection",
    "fundamental_update", "fundamental_selection", "daily_report",
]
```

Also test that fundamental selection is skipped after fundamental update failure and report is skipped when any required input step fails.

- [ ] **Step 2: Implement structured daily orchestration**

`stock_research.pipelines.daily` defines immutable `StepResult` and `DailyRunResult`, executes the existing order, records failures, and invokes alerting once at the end. It calls pipeline functions directly; it does not import `apps` or spawn a second Python interpreter.

- [ ] **Step 3: Implement the single production app and PowerShell wrapper**

`apps.daily_pipeline` loads `config/pipeline.toml`, applies environment/CLI overrides, configures UTF-8 logging in `PATHS.logs`, and calls `run_daily_pipeline`.

Its parser includes `--no-push` and `--dry-run`. `--dry-run` loads and validates configuration, imports every registered pipeline callable, prints these names in order, and returns 0 without invoking a callable:

```python
STEP_NAMES = (
    "formula33", "sector_stats", "sector_watch", "factor_selection",
    "fundamental_update", "fundamental_selection", "daily_report",
)

if args.dry_run:
    print("\n".join(STEP_NAMES))
    return 0
```

`scripts/run_daily_analysis.ps1` is limited to interpreter/proxy environment setup and:

```powershell
& $PythonBin -u -m apps.daily_pipeline @Arguments
exit $LASTEXITCODE
```

- [ ] **Step 4: Update Actions paths and dependencies**

Use `config/requirements/actions.txt`, seed `var/cache`, call `scripts/run_daily_analysis.ps1`, and upload `var/logs`, `var/exports`, and coverage metadata under `var/cache`.

- [ ] **Step 5: Verify entry references and orchestration tests**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest tests/integration/test_daily_pipeline.py tests/architecture/test_entrypoints.py -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --help
```

Expected: PASS/exit 0.

- [ ] **Step 6: Delete old root files and commit**

```powershell
git add apps stock_research scripts config .github tests
git add -u run_daily_analysis.ps1 requirements-actions.txt
git commit -m "refactor: unify daily production entrypoint"
```

### Task 10: Migrate runtime data and delete generated clutter safely

**Files:**
- Modify: `.gitignore`
- Runtime move: `.cache/` → `var/cache/`
- Runtime move: `.data/` → `var/data/`
- Runtime move: `logs/` → `var/logs/`
- Runtime move: latest `选股结果/` → `var/exports/selection/`
- Runtime move: latest `板块观察/` → `var/exports/market/` and `var/exports/reports/`
- Runtime move: `.factorStock_last.json` → `var/state/factor_selection_last.json`
- Runtime move: `.pushplus_token` → `var/secrets/pushplus_token`
- Delete: `.test-tmp/`, `.pytest_cache/`, `__pycache__/`, `.agents/`, `.claude/`, duplicate legacy exports

- [ ] **Step 1: Generate an explicit manifest before any move**

The manifest records absolute source, absolute destination, file count and byte count for each directory. Resolve every path and assert it begins with the resolved workspace root. Abort on a destination collision with unequal content.

- [ ] **Step 2: Update ignore rules before creating `var`**

`.gitignore` must contain:

```gitignore
var/
__pycache__/
.pytest_cache/
.test-tmp/
```

Remove obsolete individual runtime-path rules after migration.

- [ ] **Step 3: Move persistent data and verify exact counts/bytes**

Move cache and database first, then compare pre/post counts and byte totals. Preserve the newest valid output of each category and the single existing log. Regression files are already checked into fixtures and do not need runtime copies.

- [ ] **Step 4: Delete only explicit disposable paths**

Delete test temp files, bytecode, empty tool directories and outputs not selected by the manifest. Do not use a repository-wide wildcard delete.

- [ ] **Step 5: Verify old runtime paths are absent and new paths are readable**

Run a script that asserts old paths do not exist, `PATHS.cache` contains the original cache count, `PATHS.database` opens read-only, and the latest report/selection files exist below `var/exports`.

- [ ] **Step 6: Commit tracked ignore changes**

```powershell
git add .gitignore
git commit -m "chore: consolidate runtime data under var"
```

### Task 11: Synchronize documentation and run final acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/knowledge/project-architecture.md`
- Modify: `docs/knowledge/operations-runbook.md`
- Modify: `docs/knowledge/database-schema.md`
- Modify: `STRATEGY.md` only where commands or paths are stale

- [ ] **Step 1: Rewrite docs to describe only the new structure**

README quick start becomes:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --no-push
```

Docs must use `stock_research`, `scripts/run_daily_analysis.ps1`, `var/data/my_trade.duckdb`, `var/cache`, `var/logs`, and `var/exports`. Remove all statements that present root scripts or `src/my_trade` as current architecture.

- [ ] **Step 2: Run stale-reference scans**

Run scans for every deleted filename, `src/my_trade`, `.data/`, `.cache/`, `选股结果/`, `板块观察/`, and root `run_daily_analysis.ps1`. Only historical design/plan documents may mention old names, and those mentions must be explicitly historical.

- [ ] **Step 3: Run Python compilation**

Run:

```powershell
python -m compileall -q apps stock_research tests
```

Expected: exit 0.

- [ ] **Step 4: Run the complete test suite**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
```

Expected: all tests PASS with no collection errors.

- [ ] **Step 5: Run regression verification**

Run:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

Expected: `6 baselines verified`.

- [ ] **Step 6: Run every CLI smoke test**

Run `& 'D:\ActionsRunner\my-trade\python\python.exe' -m <module> --help` for all eight app modules. Expected: each exits 0 without import warnings.

- [ ] **Step 7: Verify final root and Git diff**

Assert the only first-level directories are `.git`, `.github`, `apps`, `stock_research`, `config`, `scripts`, `tests`, `docs`, and ignored `var`, plus no root `*.py`. Run `git diff --check`; expected exit 0.

- [ ] **Step 8: Run a no-push application dry run or controlled offline smoke**

Run the configuration-and-step dry path, which imports every pipeline, validates paths and settings, but does not call external endpoints:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --dry-run --no-push
```

Expected: exit 0 and print the seven ordered step names. The `--dry-run` flag is implemented in Task 9 and covered by `test_entrypoints.py`; it is structural evidence only and does not replace the six historical output regressions.

- [ ] **Step 9: Commit documentation and final corrections**

```powershell
git add README.md STRATEGY.md docs
git commit -m "docs: document reorganized project architecture"
```

- [ ] **Step 10: Produce the acceptance report**

Report exact commands, exit codes, final test count, `6 baselines verified`, app smoke count, migrated cache file/byte counts, deleted path summary, and any external-network limitation. Do not state completion unless every non-network acceptance gate is green.
