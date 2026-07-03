# 最小每日推送链路与停牌状态重构设计

- 状态：待用户书面复核
- 日期：2026-07-03
- 目标：只保留稳定运行每日选股、三浪三、板块分析、日报推送、告警、测试和数据仓库所需的代码与数据边界
- 非目标：本轮不改变三浪三技术公式、基本面评分、价值线公式或选股排序

## 1. 已确认业务口径

三浪三统计窗口固定为最近 21 个市场交易日。股票是否进入正式结果，按本次观察日重新判断，不保存永久黑名单：

1. 观察日有有效日线，且 21 日窗口内曾命中 XG：可以进入 21 日正式去重结果。
2. 观察日确认停牌：本次不进入正式去重结果；复牌后的下一次运行重新判断，只要原命中仍在 21 日窗口内即可重新进入。
3. 行情接口失败、缓存写入失败和真实停牌必须分别记录。接口失败采取失败关闭，不进入正式结果，但只能标记为 `data_unavailable`，不能标记为停牌或写入永久排除名单。
4. 报告同时显示技术窗口去重数、观察日可交易去重数、停牌排除数和数据不可用数。面向用户的“三浪三 21 日数量”使用观察日可交易去重数。

2026-07-02 的回放结果为：技术窗口去重 186，只因观察日停牌排除拓荆科技和日科化学后，正式数量为 184。现有工作表显示 185 行，是 184 行股票数据加 1 行表头。

## 2. 最小生产链路

保留以下入口和共享模块：

```text
.github/workflows/stock-selection.yml
run_daily_analysis.ps1
formula33Stats.py
sectorStats.py
sectorWatch.py
factorStock.py
fullMarketFundamentalUpdate.py
dailyFundamentalSelect.py
dailyReportPush.py
pipelineAlert.py
point_in_time.py
trade_utils.py
wave_utils.py
requirements-actions.txt
pyproject.toml
src/my_trade/
tests/
docs/
STRATEGY.md
README.md
```

`factorStock.py` 虽然职责过多，但每日链路仍直接调用其中的行情、估值和评分函数。本轮不凭静态搜索删除其内部函数；拆分与删除只能在后续模块迁移、回归结果一致后进行。

## 3. 项目整理与删除边界

### 删除

- 已被 GitHub Actions Windows 自托管运行替代的 `install_daily_task.ps1`；
- 当前生产环境不调用的 Linux 入口 `run_daily_analysis.sh`；
- `.cache` 中失败写入遗留的 `*.tmp`；
- `__pycache__`、`.pytest_cache` 和可重新生成的字节码；
- `回测结果` 中的历史实验导出；
- `板块观察`、`选股结果` 和 `logs` 中不再作为回归基线或最新生产结果使用的重复导出。

### 保留或移动

- `.cache/formula33_kline`、财务缓存、股票池和股本缓存必须保留，它们是每日增量运行的输入；
- `tests/regression/legacy-output-v1.json` 引用的 6 份历史文件必须保留，除非先把基线迁移到受控测试夹具；
- 2026-07-02 最新三浪三 CSV/XLSX 保留用于本轮停牌规则验收；
- `install_github_runner.ps1` 移到 `scripts/admin/`，作为灾难恢复和重建自托管运行器的运维工具；
- `.test-deps` 在运行器 Python 完成测试依赖安装且验证成功后删除。

所有删除都使用显式路径清单。执行前后记录文件数量与磁盘占用，不使用覆盖整个工作区的通配递归删除。

## 4. 三浪三实现调整

`fetch_one_stock` 不再因观察日缺少 K 线而立即丢弃整只股票的 21 日历史。它返回：

- 窗口内 BASE/XG 命中；
- `latest_data_date`；
- `observation_status`：`traded`、`suspended` 或 `data_unavailable`；
- 实际数据源和错误信息。

汇总层先生成 21 日技术去重集合，再按 `observation_status == "traded"` 生成正式集合。状态只属于当前 `run_id` 和观察日，下一次运行重新计算。

当前数据源没有可靠的单一停牌接口，因此状态判定采用失败关闭：任一行情源成功返回历史数据但没有观察日日线时记为 `suspended_or_no_trade`；所有行情源均失败时记为 `data_unavailable`。报告不得把后者写成停牌。后续接入交易所停复牌清单后，再把 `suspended_or_no_trade` 精确拆成停牌和其他无成交状态。

## 5. DuckDB 与 SQL 验收

本轮不宣称生产流水线已经切换到 DuckDB。验收分两层：

1. **基础设施层**：使用运行器 Python 安装 `duckdb`、`pytest` 和 `pytz`，创建临时数据库并验证迁移幂等、四个 schema、`ops.schema_migrations`、`ops.runs`、`ops.run_steps`、约束、事务回滚、只读连接和运行生命周期查询。
2. **本地生产库层**：初始化 `.data/my_trade.duckdb`，查询 schema 版本和表结构，插入后删除一条受控 smoke-test 运行，确认没有残留测试记录。

GitHub Actions 在每日任务前增加快速测试步骤；测试失败时不得运行正式选股或推送。现有顶层脚本仍以文件缓存通信，只有完成后续接入计划后才允许声称 DuckDB 是生产唯一事实源。

## 6. 测试与回归

新增测试覆盖：

- 观察日成交且窗口内命中时进入正式集合；
- 观察日停牌时保留技术命中但不进入正式集合；
- 次日复牌后自动恢复资格，不存在永久黑名单；
- 数据源失败进入诊断，不伪装成停牌；
- 2026-07-02 固定回放得到技术窗口 186、正式结果 184；
- 原有 6 份历史输出基线继续通过；
- DuckDB 初始化、迁移、约束和运行生命周期集成测试通过；
- 删除文件后每日 PowerShell 编排引用的每个入口仍存在。

## 7. 完成标准

- 每日生产链路只引用保留文件；
- 三浪三停牌规则由自动测试固定，2026-07-02 正式数量为 184；
- 所有测试、Python 编译、PowerShell 静态入口检查和 SQL smoke test 通过；
- 清理未删除任何回归基线、生产缓存、凭证或最新生产结果；
- 文档准确区分“当前文件缓存生产链”和“已验证但尚未接入生产的 DuckDB 基础层”。
