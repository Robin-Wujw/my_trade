# my_trade

沪深 A 股收盘后研究与选股系统。它负责行情与财务增量更新、Formula33 市场结构、板块主线、因子和基本面筛选、日报生成与 PushPlus 摘要推送；不连接券商，也不自动下单。选股口径见 [STRATEGY.md](STRATEGY.md)，架构和运维索引见 [docs/README.md](docs/README.md)。

## 项目结构

```text
apps/             命令行入口
stock_research/   核心 Python 包
config/           生产参数
scripts/          Windows 生产与计划任务脚本
tests/            单元、集成、架构和回归测试
docs/             当前架构、数据和运维文档
var/              缓存、数据库、状态、日志和导出（不提交）
```

## 生产流水线

唯一完整生产入口是：

```powershell
.\scripts\run_daily_analysis.ps1
```

也可以直接调用 Python：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --no-push
```

七步顺序固定为：

```text
1. formula33
2. sector_stats
3. sector_watch
4. factor_selection
5. fundamental_update
6. fundamental_selection
7. daily_report
```

`--no-push` 会完整生成结果但不发送 PushPlus，适合生产前复核。正式运行不带该参数；只有六个上游步骤全部成功、日报输入属于同一观察日且必需产物有效时，才允许生成分页摘要。页数由完整名单和 PushPlus 字符上限决定，不固定为两条；任一必需步骤或任一分页发送失败都视为推送失败。

完整名单推荐使用一封 HTML 邮件投递。配置 `REPORT_EMAIL_TO` 后，日报默认切换为 `email`，不再发送多条 PushPlus；`REPORT_DELIVERY=both` 可同时发送邮件和 PushPlus，`REPORT_DELIVERY=pushplus` 保持旧模式。SMTP 密码必须使用邮箱授权码并放在 `SMTP_PASSWORD` 或被忽略的 `var/secrets/smtp_password`，不得提交仓库。

只检查配置、导入和步骤顺序，不访问行情接口：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --dry-run --no-push
```

单步入口位于 `apps/`，只用于调试和故障恢复，不能另行拼接一套生产顺序。

## 持仓与观察股快速提醒

在 `config/watch_stocks.json` 中维护少量持仓或观察股票，在 `config/trade_plans.json` 中维护需要执行的显式网格和仓位计划。快速分析不运行全市场选股：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.quick_watch
```

命令优先读取本地前复权日线；缓存缺失或早于预期最新交易日时，只增量补抓观察清单中的股票，不运行全市场选股。输出中文波段价位、均线、量能和操作意见，并单独发送一条 PushPlus。补抓失败时会标记行情过期并屏蔽买卖意见；复核但不推送时使用 `--no-push`。

## 历史候选与组合回测

先回建并保存逐交易日候选与 Formula33 阶段，再运行最多三只持仓的组合回测：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.rebuild_candidate_history --start-date 2026-01-01 --end-date 2026-07-10
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.rebuild_formula_history --start-date 2026-01-01 --end-date 2026-07-10
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.portfolio_backtest --start-date 2026-01-01 --end-date 2026-07-10
```

冻结第一个交易日的选股结果、不再加入新候选时，增加 `--candidate-mode fixed-first`；默认 `rolling` 使用逐日快照。默认总持股最多5只、右侧最多3只且总仓不超过100%；`--max-positions` 是用于研究的更严格总持股上限，不应误作右侧上限。

组合回测默认先断点补齐区间内每日主流板块排名并重建候选快照；已有合格日期直接跳过。历史涨停扩散缺失时只生成明确标记的基础主线分，历史成分使用当前接口代理并保留非严格时点声明。只有已人工验证全部输入时才使用 `--no-refresh-inputs`。持仓与观察股快速分析同样先补个股行情并校验对应数据日的主流快照，门禁失败时不生成买卖意见。

研究快照保存在 `var/backtests/candidate_snapshots/mainline-left-manual-v2/`，每个交易日一个 CSV，并由 `manifest.json` 记录报告期、右侧可用数、左侧观察数、主流快照日期和时点限制。右侧候选为“标准基本面选股与新鲜主流成分的交集”，再并入 `config/watch_stocks.json` 人工观察清单；价值线候选默认只有左侧权限，只有 `config/trade_plans.json` 中经过复权校准的显式网格才能成交。收盘后形成的候选和Formula33状态从下一交易日才生效。财务缓存没有完整公告修订历史，因此结果仍标记为研究回测。

## Formula33 缓存与断点

生产固定使用 AkShare 股票清单、交易日和前复权日线。普通 A 股优先使用东方财富前复权端点，接口明确失败时才回退新浪；同一股票的缓存必须保持单一复权序列。日线同时持久化到本地 QFQ 缓存和 DuckDB，缓存版本为 `qfq-cache-v2`。

- 已缓存股票通常只补抓缺少的交易日；重叠日价格变化表示复权因子改变，此时必须完整刷新并原子替换窗口。
- 中途取消后重跑，已完成股票直接复用缓存，从未完成位置继续。
- 同一观察日和同一有效参数已完整成功时，完成清单直接命中。
- 周末复跑最近交易日结果时输出 `network_fetch=0`，不访问网络。
- 数据源失败不能伪装成停牌；存在可重试的数据不可用股票时，不写完成清单。

2026-06-11 至 2026-07-10 的 21 个交易日是固定回归锚点：上市超过 300 天后的技术全量 191 只，观察日无交易排除 3 只，正式结果 188 只；总市值大于 100 亿元的独立池为 145 只。股票 `001331` 的 2026-05-27 前复权收盘价固定为 `48.08`。

## 运行数据

- `var/cache/`：AkShare 行情、财务、板块缓存和 Formula33 快照。
- `var/data/my_trade.duckdb`：当前已落地的迁移、运行基础表、板块和日线持久化。
- `var/state/`：完成清单、断点和上一交易日结果。
- `var/exports/market/`：Formula33 与市场结构结果。
- `var/exports/selection/`：因子和基本面选股结果。
- `var/exports/reports/`：完整 HTML 日报。
- `var/logs/`：生产运行日志。
- `var/secrets/`：本地凭证；生产优先读取环境变量。

DuckDB 当前只有文档列出的 7 张实际表，生产仍同时使用文件缓存和 CSV/Excel/HTML 产物；不能把尚未落地的目标表或查询服务当成现状。

## 验证

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m compileall -q apps stock_research tests
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

Formula33 的 188 只截图清单另有逐代码回归夹具，修改指标、复权、观察日资格或缓存逻辑后必须一并运行全量测试。

## 日报技术层

日报为前两组选股计算收盘 KD、盘中高低点 KD、RSI999、MACD、ENE、WR、BIAS 和 5/10 日量能基准，并输出可操作性、风险和置信度。原始数值保留在 CSV，PushPlus 只展开实际警讯。

自动波浪划分已停用，不再用50%、62.5%或75%生成选股、买卖和提醒。当前自动右侧只保留放量突破21日收盘高点，以及MA20、MA60均明确上扬时的MA20拉回；其他结构必须来自人工显式计划。

`var/state/two_month_breakout_watch.json` 独立跟踪最近两个月曾进入前两组选股的股票。回调修复 45%～50% 为强提醒区，50%～60% 为突破跟踪区；超过 60% 停止提醒。突破前高且脱离当前全部筛选后才从追踪池剔除。

## Windows 计划任务

`scripts/setup_scheduled_task.bat` 创建每天 20:30 的 Windows 计划任务，调用 `scripts/run_daily_analysis.ps1`。运行脚本优先使用 `PYTHON_BIN`，否则使用项目固定解释器；它会设置 UTF-8、自动选择财报期，并在 Python 子进程运行期间按配置处理代理环境变量。
