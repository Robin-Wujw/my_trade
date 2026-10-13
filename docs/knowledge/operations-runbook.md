# 每日流水线运维

## 1. 解释器与入口

推荐使用项目生产解释器：

```powershell
$Python = 'D:\ActionsRunner\my-trade\python\python.exe'
```

完整生产入口：

```powershell
.\scripts\run_daily_analysis.ps1
```

脚本设置 UTF-8、选择财报期、处理代理环境变量并调用 `python -m apps.daily_pipeline`。不要手工串联多个单步命令替代生产入口。

## 2. 七步顺序

```text
1. formula33             21 日市场结构和观察日资格
2. sector_stats          板块行情统计与持久化
3. sector_watch          主线评分、涨停扩散和成分
4. factor_selection      独立因子候选
5. fundamental_update    财务增量与动态截面
6. fundamental_selection 价值线和正常基本面候选
7. daily_report          四项日报、CSV/HTML 和可选推送
```

`fundamental_update` 失败时必须跳过 `fundamental_selection`。六个上游步骤任一失败时必须跳过正式日报推送。日报还会校验 Formula33 完成清单、样例标记、必需产物数量和三个输入的观察日。

## 3. 推荐运行流程

先做离线结构检查：

```powershell
& $Python -m apps.daily_pipeline --dry-run --no-push
```

再完整跑一遍但不推送：

```powershell
& $Python -m apps.daily_pipeline --no-push
```

逐项确认七步退出码、结果数量、日期和产物后，才执行正式推送：

```powershell
& $Python -m apps.daily_pipeline
```

正式推送成功的日志必须对本次生成的每一页都出现成功结果，例如六页时：

```text
PUSH_RESULT_1 True
PUSH_RESULT_2 True
...
PUSH_RESULT_6 True
```

只看到日报文件生成不代表推送成功。任何一步失败或任一 PushPlus 返回失败，都不能报告整次生产成功。

使用 HTML 邮件时，成功日志为 `EMAIL_RESULT True`。配置 `REPORT_EMAIL_TO` 后默认投递模式是 `email`；显式设置 `REPORT_DELIVERY=both` 才会同时检查邮件与所有 PushPlus 分页。邮件或任一分页失败，日报步骤均返回失败。

SMTP 必需变量为 `SMTP_HOST`、`SMTP_USERNAME`、`SMTP_PASSWORD` 和 `REPORT_EMAIL_TO`。SSL 默认端口 465；STARTTLS 通常使用 587，并设置 `SMTP_SECURITY=starttls`。授权码可以存放在 `var/secrets/smtp_password`，不得写入脚本、日志或 Git。

## 4. Formula33 增量与断点

生产参数固定使用 AkShare 清单、交易日和前复权行情，并带 `--require-end-trade`。日线缓存版本是 `qfq-cache-v2`。

运行行为：

- 单股已有完整历史时直接从缓存读取；
- 新交易日通常只请求缺少的增量区间；若重叠日 OHLC 变化，说明前复权锚点变化，必须整窗刷新并原子替换；
- 运行在第 N 只股票中断后，前面已原子落盘的缓存继续有效；
- 重跑时已完成股票快速通过，未完成或缺日期的股票才访问网络；
- 全部必要股票、覆盖率和输出均成功后才写 `var/state/formula33_completion.json`；
- 同一观察日、有效参数和股票池完全匹配时直接复用完成清单；
- 周末复跑最近交易日时完成清单命中，日志应显示 `network_fetch=0`。

不要删除有效缓存来处理网络故障。先确认请求、限流、代理和提供方状态；缓存版本或复权口径确实错误时才做有记录的定向重建。

## 5. Formula33 固定验收

复核命令：

```powershell
& $Python -m apps.formula33 --start-date 2026-06-11 --end-date 2026-07-10 --history-days 420 --workers 2 --maxtasksperchild 1000 --sleep 0.5 --retries 5 --retry-delay 5 --capital-workers 1 --require-end-trade --price-source akshare --metadata-source akshare --market-cap-source auto --missing-mktcap-policy exclude
```

验收值：

```text
技术条件原始去重       193
上市不足 300 天排除      2
技术全量               191
观察日无交易排除          3
正式结果               188
市值大于 100 亿元独立池 145
```

还必须核对 188 只逐代码回归，而不只核对总数。`001331` 的 2026-05-27 前复权收盘价应为 `48.08`。

## 6. 板块验收

`sector_stats` 和 `sector_watch` 必须都成功：

- 板块列表来自真实提供方，不使用样例回退；
- 每个板块历史满足最少行数；
- 实际最后交易日符合新鲜度要求；
- 整体新鲜覆盖率达到门槛；
- 排名前列板块有有效板块代码；
- 主线板块成分全部成功获取；
- 涨停池能够按股票代码与板块成分交叉核对。

接口失败时保留已验证的板块缓存。缓存过期、缺行或来源不匹配时必须失败关闭，不能用旧文件改名伪装本轮结果。

## 7. 基本面与选股验收

`fundamental_update` 输出本轮观察日、报告期、行情覆盖率和财务覆盖率。生产门槛由七步入口传入：

```text
最低行情覆盖率       90%
最低财务覆盖率       35%
目标财务覆盖率       95%
```

`fundamental_selection` 必须生成：

- 基本价值线或附近全量；
- 正常基本面候选；
- 每只股票的报告期、技术截止日、方法、质量、流动性、市值、主流板块和右侧位置。

价值线和正常基本面数量可以随观察日变化，不以固定数量为成功标准。成功标准是数据覆盖、硬条件、日期和产物完整。

## 8. 产物检查

主要输出目录：

- `var/exports/market`：Formula33 Excel/CSV 和市场结果；
- `var/exports/selection`：因子、基本面和合并选择 CSV；
- `var/exports/reports`：完整四项 HTML 日报；
- `var/logs`：PowerShell transcript 和步骤日志；
- `var/state`：Formula33 完成清单、财务断点和日报比较基线。

日报只消费本轮明确捕获的产物。不要把 `_sample` 文件、旧观察日文件或手工导出文件放进正式输入。

## 9. 常见故障

### 网络或限流

区分连接错误、限流、空数据和字段变化。保留有效缓存，按已配置重试和退避处理。Formula33 的 `data_unavailable` 不能改记为停牌。

### 观察日无交易

股票仍保留在 Formula33 技术全量和停牌诊断，但从正式名单排除。下一观察日重新判断，不写永久黑名单。

### Formula33 完成清单未命中

检查观察日、有效参数、股票池、代码版本和输出文件是否变化。参数不同或产物缺失时重新计算是正确行为；不能手工修改清单骗过门禁。

### 板块覆盖失败

查看 `ops.pipeline_events` 和控制台中的 expected、fresh、stale、missing、coverage。修复数据源或补齐缓存后重跑板块步骤，再从完整生产入口验证。

### 财务覆盖不足

保留 `var/state` 中的断点，后续从缺失股票继续补齐。不得为了推送降低生产门槛或把请求失败当成无财务数据。

### 两个月突破追踪异常

状态文件为 `var/state/two_month_breakout_watch.json`，历史名单基线为 `var/state/daily_selection_history.json`。同一观察日重跑不会重复累计突破次数；只有新交易日从 50% 下方重新上穿才加一次。不要通过删除状态文件消除提醒，除非明确要重新建立追踪基线。

### 日期不一致

检查 Formula33 观察日、板块实际日期和基本面技术截止日。日报拒绝混合旧产物；不要通过改文件名或修改时间绕过。

## 10. 代码验证

```powershell
& $Python -m compileall -q apps stock_research tests
& $Python -m pytest -q
& $Python -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

历史基线成功输出应为 `6 baselines verified`。Formula33 的 188 只逐代码回归包含在测试套件中。

## 11. DuckDB 检查边界

当前迁移只包含 7 张应用表：3 张基础运维表、1 张事件表、2 张板块表和 1 张股票日线表。顶层七步生产成功不能通过查询一个不存在的全链路运行表来判断；以退出码、完成清单、覆盖率、产物和 PushPlus 返回值共同验收。

数据库备份应在没有活跃写事务时进行。由于财务和部分生产数据仍在文件缓存中，当前灾备范围还必须包含 `var/cache` 和 `var/state`。

## 12. 计划任务

`scripts/setup_scheduled_task.bat` 创建每天 20:30 的 `StockSelection-Daily` 任务，工作目录为项目根目录。定时任务应使用与手工验证相同的解释器和环境变量，并保留 `var/logs` 中的完整日志。

## 13. GP/A 独立前向研究采集

GP/A 不属于七步生产流水线，不参与日报推送，也不能替换当前右侧候选。它从 `2026-08-11` 以后真实捕获的财报版本开始独立积累证据。

先同步当时最新完整年报。下面以2025年报为例；后续年度必须显式修改 `--period`：

```powershell
& $Python -m apps.data_sync income --period 20251231
& $Python -m apps.data_sync balancesheet --period 20251231
```

每次同步都会保留新payload的首次 `captured_at`；相同payload重复同步不会复制版本。网络失败、空表或限流时保留已有版本，不能把当前值表回填到追加版本表。申万成员发生更新时，通过独立状态文件刷新快照，避免旧的断点状态跳过请求：

```powershell
& $Python -m apps.zer0share_sync --table sw_member --end-date 20260811 --state-file var/state/zer0share_sw_member_20260811.json
```

候选生成前必须先有覆盖目标观察日的六因子全市场面板和 Formula33 交易日历史，然后显式传入实际文件：

```powershell
& $Python -m apps.quantsplaybook_gross_profitability_forward `
  --start-date 2026-08-11 `
  --end-date 2026-08-11 `
  --formula-history var/backtests/formula33_forward.csv `
  --low-volatility-panel-root var/backtests/forward_factor_panels
```

以下任一条件会失败关闭：观察日早于冻结日、没有冻结日后捕获的两张财务表、缺带本地时间的行业快照、缺上市股票全集、没有当日面板或Formula33日期、ADV20后完整七因子股票少于100只。成功产物写入 `var/forward/right_gross_profitability_to_assets`，必须与同期冻结六因子对照一起保存。MiniQMT仍只读，候选不得发送到券商下单接口。
