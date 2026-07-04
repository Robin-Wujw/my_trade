# 项目架构知识库

## 1. 系统定位

`my_trade` 是沪深 A 股收盘后研究与选股系统，负责数据采集、市场结构、板块主线、因子与基本面候选、日报和 PushPlus 摘要。它不维护真实持仓，不连接券商，也不自动下单。

## 2. 一级目录

```text
apps/             可执行入口，只负责参数和退出码
stock_research/   唯一可导入核心包
config/           可提交配置和运行依赖
scripts/          PowerShell 生产与运维脚本
tests/            自动测试与受控回归夹具
docs/             当前架构、运维和历史设计
var/              唯一运行数据根目录，整体忽略提交
```

## 3. 调用方向

```text
scripts/run_daily_analysis.ps1
              ↓
      apps.daily_pipeline
              ↓
 stock_research.pipelines.daily
              ↓
 formula33 → sector_stats → sector_watch → factor_selection
           → fundamental_update → fundamental_selection → daily_report
```

模块依赖方向固定为：

```text
apps
  → pipelines
      → market / indicators / strategies / reporting
          → api / storage / core
```

- `api`：外部接口、PushPlus 和重试；
- `core`：路径、配置、运行上下文和时点规则；
- `storage`：DuckDB、迁移和运行仓储；
- `market`：行情、股票池、板块和基本面访问；
- `indicators`：可离线验证的纯计算；
- `strategies`：三浪三、板块、因子和基本面规则；
- `pipelines`：执行顺序、并发、门控和诊断；
- `reporting`：日报、导出、差异和告警；
- `regression`：历史输出原始与语义哈希。

## 4. 入口边界

正式生产入口只有 `apps.daily_pipeline`。`apps.formula33`、`apps.sector_analysis`、`apps.factor_selection`、`apps.fundamental_update`、`apps.fundamental_selection`、`apps.daily_report` 和 `apps.pipeline_alert` 只供单步调试、回放和故障恢复。根目录没有兼容脚本。

## 5. 运行数据

`ProjectPaths` 是路径唯一来源：缓存位于 `var/cache`，DuckDB 位于 `var/data/my_trade.duckdb`，导出位于 `var/exports`，日志位于 `var/logs`，状态位于 `var/state`，本地凭证位于 `var/secrets`。生产模块不得自行推断仓库根目录或恢复旧路径。

当前生产策略仍读取增量文件缓存并输出 CSV/HTML；DuckDB 的 schema、事务、运行记录和回归基础已可用，但尚未承诺所有策略数据都只从 DuckDB 读取。文档必须保持这一区分。

## 6. 架构不变量

- 任何历史计算必须有观察日；
- 数据源失败与停牌状态不得混淆；
- 报告必需输入失败时不得推送；
- 指标模块不联网、不写文件；
- 敏感值只来自环境变量或 `var/secrets`；
- 测试夹具位于 `tests/fixtures`，不依赖运行输出；
- 外部接口波动不作为结构验收依据；
- 新代码不得导入已删除的旧根模块名。

## 7. 添加能力

新数据源加入 `api` 并提供字段映射、来源和失败分类测试；新指标先加入 `indicators` 并使用固定输入测试；新筛选规则加入 `strategies`；执行顺序只在 `pipelines` 中修改；新增命令只在 `apps` 中提供薄入口。

## 8. 三浪三观察日状态

最近 21 个交易日的技术命中与观察日资格分开。停牌或无交易股票保留历史技术命中，但不进入本次正式集合；复牌后重新判断，不保存永久黑名单。所有行情源失败记录为 `data_unavailable`，不得伪装为停牌。
