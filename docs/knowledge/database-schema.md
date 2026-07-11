# DuckDB 实际结构

## 1. 当前边界

数据库文件固定为 `var/data/my_trade.duckdb`。迁移代码当前只创建 4 个 schema 和 7 张应用表；其中 `core`、`derived` schema 目前没有表。

生产仍使用 `var/cache` 的增量文件缓存，并输出 CSV、Excel 和 HTML。DuckDB 当前只覆盖运行基础、板块和股票日线持久化，不是全部业务数据的唯一事实源。

## 2. 实际 7 张表

| 表 | 用途 | 主键 |
|---|---|---|
| `ops.schema_migrations` | 已应用迁移版本 | `version` |
| `ops.runs` | 可选的运行生命周期基础记录 | `run_id` |
| `ops.run_steps` | 可选的单次运行步骤基础记录 | `run_id, step_name` |
| `ops.pipeline_events` | 板块等分步事件和进度日志 | 无 |
| `raw.sector_boards` | 板块名称、代码、分组和来源 | `board_name` |
| `raw.sector_board_history` | 按来源保存的板块日线 | `board_name, trade_date, source` |
| `raw.stock_kline_daily` | 按来源保存的股票日线 | `source, code, trade_date` |

除这 7 张表外，文档不约定任何尚未落地的表或视图。

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

## 4. 写入规则

- 网络抓取可以并发，DuckDB 写入通过仓储集中提交；
- 日线和板块历史按主键增量 upsert，不因一次接口失败删除已有数据；
- 板块读取必须检查实际最后交易日、最少历史行数和来源；
- Formula33 文件缓存与 DuckDB 日线必须保持相同前复权口径；
- 测试使用独立临时数据库，不能写入生产库；
- 一个写事务失败时必须回滚，不能留下半次迁移。

## 5. 检查

可通过 DuckDB 的 `information_schema.tables` 核对应用表数量和名称。结构变更必须保持迁移幂等、事务回滚和既有数据兼容测试通过，不能直接手工修改生产数据库代替迁移。
