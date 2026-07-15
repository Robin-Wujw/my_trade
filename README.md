# my_trade

沪深 A 股收盘后研究与选股系统。项目只做数据更新、候选生成、日报、提醒和研究回测；不连接券商，不自动下单。

当前主线是统一候选池：价值线、主流基本面、右侧强势成长观察都先输出同一套候选字段，再交给同一套买卖和回测引擎处理。观察名单和交易计划只能用于提醒或显式结构锚点，不能直接注入候选池。

## 先跑什么

生产入口只有一个：

```powershell
.\scripts\run_daily_analysis.ps1
```

只复核、不推送：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --dry-run --no-push
```

完整跑但不推送：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --no-push
```

七步顺序固定：

```text
1. formula33
2. sector_stats
3. sector_watch
4. factor_selection
5. fundamental_update
6. fundamental_selection
7. daily_report
```

日报只在前六步全部成功、输入属于同一观察日、必需产物有效时生成。配置 `REPORT_EMAIL_TO` 后默认用 HTML 邮件投递完整名单；`REPORT_DELIVERY=both` 同时发邮件和 PushPlus，`REPORT_DELIVERY=pushplus` 保持旧模式。

## 常用入口

快速看持仓和观察股：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.quick_watch
```

重建历史候选和 Formula33 阶段，并做组合回测：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.rebuild_candidate_history --start-date 2026-01-01 --end-date 2026-07-10
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.rebuild_formula_history --start-date 2026-01-01 --end-date 2026-07-10
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.portfolio_backtest --start-date 2026-01-01 --end-date 2026-07-10
```

用 vectorbt 重放同一批成交做现金账本交叉验证：

```powershell
python -m pip install -e ".[vectorbt]"
python -m apps.portfolio_backtest --start-date 2026-01-01 --end-date 2026-07-10 --vectorbt-cross-check
```

正常回测会先刷新输入；只有确认要复用冻结数据时才加 `--no-refresh-inputs`。

## 选股口径

候选快照目录：

```text
var/backtests/candidate_snapshots/unified-selection-v4/
```

统一硬门槛：

- 质量分不低于 70；
- 扣非增长不低于 10%；
- 总市值不低于 100 亿元；
- 增长得分最多按 100% 增长计；
- 每日最多 10 只候选，其中至少 5 个核心名额留给价值线或主流基本面候选。

左右侧自动新买入都必须有观察日可见的 100 亿元以上总市值。缺失市值不按“尽量满足”放行，而是停止新买入；已有仓位继续按各自止盈、止损、价值证伪和组合限额处理。

三类候选：

- 价值线或附近：`0.80 <= 现价 / 价值线 <= 1.08`，再过统一硬门槛。
- 主流基本面：统一硬门槛通过，且属于新鲜主流板块快照。
- 右侧强势成长观察：统一硬门槛通过，`trade_basis_score >= 4`，并且 20/60/120 日强度与距 120 日高点形成 `leadership_score >= 15`。

IMA 和网页交叉验证吸收为 `trade_basis_score`，只做买点质量证据，不绕过统一候选和买卖模型。它检查：

- MA20/MA60 是否上扬；
- 收盘是否站上或贴近 MA20；
- 是否距离 21 日收盘高点 3% 以内；
- 量能是否高于 5/10 日基准；
- 5/10/20 日均量扣抵是否改善。

v4 新增的强势成长观察不再只看短线热度，而是把 20/60/120 日强度和距 120 日高点位置写入候选快照与 DuckDB，避免尚未加速的价值/主线核心候选被短期强势股全部挤出。

## 回测口径

组合回测使用收盘后形成的候选和 Formula33 状态，并从下一交易日生效。主交易名额默认 3 只，全部持仓标的硬上限 5 只，日常目标约 3 只；所有行情下左侧价值标的最多 1 只。左仓涨离价值线（收盘价/价值线 > 1.08）且出现合格右侧买点后，原左仓由右侧规则接管，左侧网格暂停，右侧周期结束前不再重开左侧网格。若已有多只左侧标的，只保留最优一只，其余按组合限额退出。单票总仓位不超过 62.5%，组合总仓位不超过 100%。

买卖引擎统一使用：

- 结构锚点和 50%/62.5%/75%价位；
- 上扬 MA20/MA60 拉回；
- 21 日收盘高点突破；
- 分批止盈、成本上方保护、峰值回撤和空间止损；
- 普通 A 股 100 股一手，科创板 `688` 为 200 股一手；
- 佣金、最低佣金、卖出印花税和估算滑点。

右侧名额满时，不会因为新候选出现就机械卖出已有趋势仓。只有已有仓跌破 MA20、持有满 5 日、收益不足 10%，且新信号明显更强，才进入汰弱换强候选。

左侧价值网格不会因为日榜落选清仓；只有候选诊断行给出明确价值/财务证伪原因时，才在下一交易日开盘清掉左侧计划。缺少未入选失败原因的日期只停止新增左侧买入。

## 输出在哪

```text
var/cache/                 行情、财务、板块和 Formula33 缓存
var/data/my_trade.duckdb   结构化研究数据和回测结果
var/exports/market/        Formula33 与市场结构输出
var/exports/selection/     因子和基本面选股输出
var/exports/reports/       完整 HTML 日报
var/backtests/             候选快照、回测净值、成交流水和摘要
var/logs/                  运行日志
var/secrets/               本地凭据，不提交
```

`var/` 是运行数据目录，不应提交到仓库。代码、配置、测试和文档才是仓库主体。

## 项目结构

```text
apps/             命令行入口
stock_research/   核心 Python 包
config/           生产参数、观察股和显式交易计划
scripts/          Windows 运行与计划任务脚本
tests/            单元、集成、架构和回归测试
docs/             当前架构、数据、输出和运维说明
```

更多规则：

- 策略口径：[STRATEGY.md](STRATEGY.md)
- 文档索引：[docs/README.md](docs/README.md)
- vectorbt 验证：[docs/vectorbt-cross-validation.md](docs/vectorbt-cross-validation.md)

## 验证

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m compileall -q apps stock_research tests
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
```

Formula33 固定回归区间为 2026-06-11 至 2026-07-10：技术全量 191 只、观察日无交易排除 3 只、正式结果 188 只、市值大于 100 亿元独立池 145 只。`001331` 在 2026-05-27 的前复权收盘价固定为 `48.08`。

## 计划任务

`scripts/setup_scheduled_task.bat` 创建每天 20:30 的 Windows 计划任务，调用 `scripts/run_daily_analysis.ps1`。运行脚本优先使用 `PYTHON_BIN`，否则使用项目固定解释器，并会设置 UTF-8、选择财报期、处理代理环境变量。
