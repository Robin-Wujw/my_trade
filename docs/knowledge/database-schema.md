# DuckDB 实际结构

## 1. 当前边界

数据库文件固定为 `var/data/my_trade.duckdb`。迁移代码当前创建4个schema和13张应用表；`core`保留为后续规范化实体层，`derived`已保存候选、Formula33阶段与回测结果。

生产仍使用 `var/cache` 的增量文件缓存，并输出CSV、Excel、HTML和Markdown。DuckDB是批量历史查询和结构化研究结果的事实源；报告正文、完成清单与可恢复的接口缓存仍保留在文件系统。

## 2. 实际13张表

| 表 | 用途 | 主键 |
|---|---|---|
| `ops.schema_migrations` | 已应用迁移版本 | `version` |
| `ops.runs` | 可选的运行生命周期基础记录 | `run_id` |
| `ops.run_steps` | 可选的单次运行步骤基础记录 | `run_id, step_name` |
| `ops.pipeline_events` | 板块等分步事件和进度日志 | 无 |
| `raw.sector_boards` | 板块名称、代码、分组和来源 | `board_name` |
| `raw.sector_board_history` | 按来源保存的板块日线 | `board_name, trade_date, source` |
| `raw.stock_kline_daily` | 按来源保存的股票日线 | `source, code, trade_date` |
| `raw.fundamental_metrics` | 按报告期保存财务选股指标及原始载荷 | `code, report_period` |
| `derived.candidate_snapshots` | 逐观察日统一候选与排名 | `observation_date, snapshot_version, code` |
| `derived.formula33_phase` | 逐观察日Formula33阶段与连续计数 | `observation_date, version` |
| `derived.backtest_runs` | 回测区间、资金、收益、回撤及摘要 | `run_id` |
| `derived.backtest_trades` | 回测逐笔真实成交 | `run_id, sequence` |
| `derived.backtest_positions` | 回测期末持仓 | `run_id, code` |

除这13张表外，文档不约定任何尚未落地的表或视图。

## 3. 表字段

### `ops.schema_migrations`

```text
version, name, applied_at, code_version
```

`Database.initialize()` 在一个事务中按版本顺序执行迁移。已经记录的版本不会重复执行；任一语句失败时整批回滚。

### `ops.runs`

```text
run_id, observation_date, market_cutoff, financial_cutoff, report_period,
mode, code_version, started_at, finished_at, status, gate_status, error_message
```

约束包括：

- `market_cutoff <= observation_date`；
- `report_period <= observation_date`；
- `mode` 只能是 `production/backtest/offline`；
- `status` 只能是 `running/succeeded/failed`；
- `gate_status` 只能是 `pending/passed/failed`。

这张表和 `ops.run_steps` 是已落地的基础设施，但当前顶层七步流水线尚未承诺每次运行都有完整记录。运维判断仍以进程退出码、完成清单、产物门禁和日志为准。

### `ops.run_steps`

```text
run_id, step_name, input_cutoff, status, started_at, finished_at,
row_count, coverage, elapsed_seconds, error_message, retry_count
```

`run_id` 外键引用 `ops.runs`。行数、覆盖率、耗时和重试次数均有非负约束。

### `ops.pipeline_events`

```text
created_at, run_id, step_name, part_name, event_type, status,
message, rows, elapsed_seconds, context_json
```

用于记录分步开始、完成、失败、覆盖率和进度事件。它不是完整运行事实表，也不能替代完成清单和输出门禁。

### `raw.sector_boards`

```text
board_name, group_name, source, updated_at, board_code
```

板块名称是当前主键，`board_code` 用于真实提供方的历史和成分请求。

### `raw.sector_board_history`

```text
board_name, trade_date, open, close, high, low,
amount, volume, pct_chg, source, updated_at
```

同一板块、交易日允许保存不同来源，读取时必须携带或明确选择 `source`。

### `raw.stock_kline_daily`

```text
source, code, trade_date, open, high, low, close,
volume, tradestatus, updated_at
```

Formula33 将前复权日线按股票和交易日增量写入该表。复权和提供方口径必须由 `source` 明确区分，不能把未复权数据写成同一来源覆盖。

### 财务与派生研究表

`raw.fundamental_metrics`保存`quality_score/earnings_yoy/market_cap/value_line`及完整`payload_json`。`derived.candidate_snapshots`保存统一候选接口、排名、报告期和原始载荷；`derived.formula33_phase`保存每日阶段与上下行连续计数。

`derived.backtest_runs`保存一次回测摘要；`derived.backtest_trades`按序号保存股票、买卖方向、成交数量、价格、金额、费用、盈亏与原因；`derived.backtest_positions`保存期末股数、成本、市值和浮盈。可读的CSV/Markdown报告仍同时写入`var/backtests`。

`derived.candidate_snapshot_coverage`保存每日候选快照覆盖，即使当日候选数为0也会入库，便于核对长区间回测是否真正覆盖所有交易日。

## 4. 写入规则

- 网络抓取可以并发，DuckDB写入通过仓储集中提交；
- 日线组合读取使用单连接批量扫描，日线写入使用DataFrame批量事务，不逐行开连接；
- 日线和板块历史按主键增量upsert，不因一次接口失败删除已有数据；
- 板块读取必须检查实际最后交易日、最少历史行数和来源；
- Formula33 文件缓存与 DuckDB 日线必须保持相同前复权口径；
- 测试使用独立临时数据库，不能写入生产库；
- 一个写事务失败时必须回滚，不能留下半次迁移。

## 5. 检查

可通过 DuckDB 的 `information_schema.tables` 核对应用表数量和名称。结构变更必须保持迁移幂等、事务回滚和既有数据兼容测试通过，不能直接手工修改生产数据库代替迁移。

## 6. 2026-07-14 架构评估与优化

当前持久化边界总体合理：`raw` 保存可复用事实数据，`derived` 保存候选、Formula33阶段和回测结果，`var/backtests` 保留 CSV/Markdown 作为人工审阅入口。此次不拆表，只在候选和成交表补充买卖依据字段：

```text
derived.candidate_snapshots:
trade_basis_score, trade_basis_reason, technical_alignment,
ima_web_validation, validation_sources_json

derived.backtest_trades:
selection_reason, trade_basis_reason, technical_alignment

derived.candidate_snapshot_coverage:
observation_date, snapshot_version, candidate_count, signal_eligible_count
```

这样 2024-09-24 至今的候选、依据、IMA/网页对照和最终成交可以在 DuckDB 中按日期、版本和 run_id 追溯；CSV 仍作为可读产物，不再承担唯一事实源。

取数速度方面新增索引：

```text
raw.stock_kline_daily(source, trade_date, code)
derived.candidate_snapshots(snapshot_version, observation_date)
derived.formula33_phase(version, observation_date)
derived.backtest_trades(run_id, trade_date)
```

组合回测已经使用 `KlineRepository.load_stock_klines()` 一次批量读取全组合日线，避免逐股票开连接。后续如继续提速，优先方向是把候选重建所需的历史行情也改成 DuckDB 批量读，并把主线成分代理的历史声明写入单独审计表。
