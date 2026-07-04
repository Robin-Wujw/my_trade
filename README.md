# my_trade

沪深 A 股收盘后研究与选股系统。系统依次完成三浪三市场结构、板块主线、因子选股、基本面截面、基本面筛选和日报推送；不连接券商，也不自动下单。策略口径见 [STRATEGY.md](STRATEGY.md)。

## 目录

```text
apps/             命令行入口
stock_research/   核心 Python 包
config/           流水线与运行依赖配置
scripts/          生产和运维脚本
tests/            单元、集成、架构与回归测试
docs/             架构和运维文档
var/              缓存、数据库、日志、导出和本地凭证（不提交）
```

`stock_research` 内部按 `api`、`core`、`storage`、`market`、`indicators`、`strategies`、`pipelines`、`reporting` 和 `regression` 分层。仓库根目录不放业务 Python 脚本。

## 每日运行

```powershell
.\scripts\run_daily_analysis.ps1
```

也可以直接运行唯一生产入口：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --no-push
```

只检查配置、导入和七步顺序而不访问网络：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --dry-run --no-push
```

单步调试入口位于 `apps/`。例如：

```powershell
python -m apps.formula33 --help
python -m apps.sector_analysis stats --help
python -m apps.factor_selection --help
```

## 运行数据

- `var/cache/`：行情、财务和板块增量缓存；
- `var/data/my_trade.duckdb`：DuckDB 数据库；
- `var/exports/`：选股、市场和日报导出；
- `var/logs/`：运行日志；
- `var/state/`：断点和上次结果；
- `var/secrets/`：本地凭证，GitHub Actions 优先使用 Secrets；
- `var/tmp/`：测试临时数据。

## 验证

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m compileall -q apps stock_research tests
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

## GitHub Actions

[stock-selection.yml](.github/workflows/stock-selection.yml) 在自托管 Windows runner 上先执行完整测试，再调用 `scripts/run_daily_analysis.ps1`。缓存保留在 `var/cache/`，Artifact 上传 `var/logs/`、`var/exports/` 和覆盖率元数据。runner 重建脚本位于 `scripts/admin/install_github_runner.ps1`。
