# 统一数据层架构

本文记录 2026-07-21 起新的数据源与持久化边界。

## 目标分工

```text
MiniQMT
  -> 历史行情、实时行情、只读账户诊断
  -> SQLite raw_stock_kline_daily

Tushare Pro
  -> 财报、daily_basic、估值、解禁、分红、股本等结构化特色数据
  -> SQLite raw_tushare_dataset_rows

AKShare
  -> 非核心补洞：板块、概念、新闻、东方财富特色接口
  -> 只有在 MiniQMT/Tushare 缺口明确时使用

SQLite
  -> 全仓库统一缓存层
  -> raw/core/derived/ops 逻辑分层用表名前缀表达
```

## 硬边界

- MiniQMT 是主行情源，也是唯一允许表示实盘交易边界的数据源；当前代码仍保持只读，不自动下单。
- Tushare 是基本面、估值和结构化特色数据的主源，token 只允许来自 `TUSHARE_TOKEN`、`TUSHARE_TOKEN_FILE` 或 `var/secrets/tushare_token`。
- AKShare 不作为核心依赖，只用于补洞，补洞结果必须记录 source。
- SQLite 数据库固定为 `var/data/my_trade.sqlite3`，`var/` 不进入 Git。

## Tushare 同步入口

```powershell
$env:TUSHARE_TOKEN_FILE = "D:\secure\tushare_token"
python -m apps.data_sync stock_basic
python -m apps.data_sync daily_basic --trade-date 20260720
python -m apps.data_sync fina_indicator --period 20260630
```

默认 Tushare API 地址为 `https://ts.gyzcloud.top/api`，可用 `TUSHARE_API_URL` 覆盖。

## 表设计

- `raw_stock_kline_daily`：MiniQMT/AKShare/Tushare 等行情源的日线缓存，保留 `source`、`adjustment`、`qfq_anchor_date` 和 `cache_version`。
- `raw_sector_boards`、`raw_sector_board_history`：板块和板块历史，AKShare/同花顺等补洞数据必须保留 `source`。
- `raw_tushare_dataset_rows`：通用 Tushare 数据集缓存，按 `dataset + row_key` 去重，保留原始 payload，并抽取 `ts_code`、`trade_date`、`ann_date`、`end_date`、`report_period` 用于时点查询。
- `raw_tushare_sync_state`：每个 Tushare dataset 最近一次成功同步的参数、行数和摘要。
- `derived_*`：候选、Formula33、回测等策略产物。
- `ops_*`：运行、步骤、事件和迁移记录。

## 参考项目落地取舍

参考 `zer0quant/zer0share` 的数据集同步与本地查询分层，采用 dataset 级缓存和同步状态表；参考 `zer0quant/zer0factor` 的契约化输入思路，将数据源职责集中到 `stock_research.data.sources`。本仓库不直接引入外部项目代码，也不把 Parquet/DuckDB 作为核心持久化格式。
