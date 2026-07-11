# 项目架构

## 1. 系统定位

`my_trade` 是沪深 A 股收盘后研究与选股系统，负责数据增量更新、Formula33 市场结构、板块主线、因子和基本面候选、完整日报与 PushPlus 摘要。系统不维护真实持仓，不连接券商，也不自动下单。

## 2. 目录

```text
apps/             薄命令行入口
stock_research/   唯一核心 Python 包
config/           流水线参数
scripts/          Windows 生产和计划任务脚本
tests/            自动测试与固定回归夹具
docs/             当前架构、数据和运维文档
var/              本地运行数据根目录，整体忽略提交
```

`stock_research/` 的职责分层：

- `api`：外部数据接口、PushPlus 和连接处理；
- `core`：路径、配置、完成清单、控制台和分步日志；
- `storage`：DuckDB 初始化、迁移、K 线和板块仓储；
- `market`：行情、板块和股票数据访问；
- `indicators`：不联网的指标计算；
- `strategies`：Formula33、基本面和板块规则；
- `pipelines`：七步执行、缓存、覆盖率和产物门禁；
- `reporting`：日报、差异、HTML/CSV 和推送正文；
- `regression`：历史输出基线校验。

## 3. 生产调用链

```text
scripts/run_daily_analysis.ps1
              |
              v
      apps.daily_pipeline
              |
              v
 stock_research.pipelines.daily
              |
              +-- 1. formula33
              +-- 2. sector_stats
              +-- 3. sector_watch
              +-- 4. factor_selection
              +-- 5. fundamental_update
              +-- 6. fundamental_selection
              +-- 7. daily_report
```

完整生产入口只有 `apps.daily_pipeline`。其他 `apps.*` 命令用于单步诊断和故障恢复，不能改变生产顺序或绕过最终门禁。

日报步骤接收本轮明确捕获的 Formula33、板块观察和基本面筛选路径。它拒绝样例产物、缺失产物、同类型多产物和观察日不一致的输入，不从目录中任意挑选旧结果。

## 4. Formula33 数据流

```text
AkShare 股票清单/交易日
          |
          v
按股票读取 qfq-cache-v2
          |
          +-- 缺历史或缺新增交易日 --> AkShare 前复权增量抓取
          |
          v
本地 CSV 缓存 + raw.stock_kline_daily
          |
          v
21 日技术全量 --> 观察日交易资格 --> 188 口径正式结果
          |
          v
Excel/CSV + formula33_completion.json
```

完成清单保存结果相关参数、观察日、股票池摘要和输出路径。同一观察日、相同参数和相同股票池已完整成功时可以直接复用；周末请求仍映射到最近交易日，因此命中后 `network_fetch=0`。中途取消时已写入的单股缓存保留，重跑只处理缺失部分。

## 5. 板块数据流

`sector_stats` 负责板块行情统计和持久化，`sector_watch` 负责主线评分、涨停扩散和主线成分映射。板块名称、代码、历史和来源写入 DuckDB；文件缓存用于接口波动时的受控复用。

板块步骤必须满足历史新鲜度、最少行数和覆盖率。主线板块成分缺失时不生成可供日报消费的正式产物。

## 6. 基本面与报告

`factor_selection` 生成独立因子候选；`fundamental_update` 增量补齐财务缓存并生成观察日截面；`fundamental_selection` 从截面生成价值线或附近、正常基本面两组结果。

`daily_report` 合并：

1. 价值线或附近；
2. 正常基本面；
3. Formula33 市场结构；
4. 板块主线。

完整结果写入 HTML 和 CSV，PushPlus 只发送有长度和代码完整性校验的两部分摘要。

## 7. 运行数据

`ProjectPaths` 统一管理运行路径：

- `var/cache`：可复用的增量数据和快照；
- `var/data/my_trade.duckdb`：当前 7 张实际表；
- `var/state`：完成清单、断点和日报比较基线；
- `var/exports`：市场、选股和日报导出；
- `var/logs`：运行日志；
- `var/secrets`：本地凭证。

当前生产是“文件缓存与导出 + DuckDB 部分持久化”的混合状态。DuckDB 已持久化 Formula33 日线、板块和基础运行表，但没有财务事实、派生选股、报告正文或完整全链路运行记录表。不得把目标架构描述成现状。

## 8. 生产不变量

- Formula33 正式结果必须排除观察日无交易股票；
- 数据源错误与停牌必须分开；
- AkShare 前复权缓存只能增量补齐，不能因单次失败覆盖有效历史；
- 同一日报的必需输入必须属于同一观察日；
- 六个上游生产步骤没有全部成功时不得推送；
- 样例产物不能进入正式日报；
- 敏感值只来自环境变量或 `var/secrets`；
- 测试夹具位于 `tests/fixtures`，不依赖 `var/exports`。
