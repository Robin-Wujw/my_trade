# my_trade

A股收盘后分析、选股、历史截面回测与 PushPlus 汇总工具。

## 先读这里

- [策略规范](STRATEGY.md)：所有选股口径、基本价值线适用条件、右侧50%/62.5%、板块和三浪三规则。它是策略实现的唯一准则。
- [分析与验证记录](ANALYSIS.md)：当前程序状态、历史回测、分层表现、已知偏差和下一步验证任务。
- [原始资料](baseInfo.md)：未经改写的学习与复盘资料，只用于追溯原意，不直接作为程序参数说明。

发生冲突时，以 `STRATEGY.md` 为准；代码和日报应向该规范对齐。

## 每日流程

```powershell
.\run_daily_analysis.ps1
```

```bash
bash ./run_daily_analysis.sh
```

流程依次运行：

1. `formula33Stats.py`：AkShare前复权日线，统计最近21个交易日三浪三XG。
2. `sectorStats.py`、`sectorWatch.py`：判断成交额、量能、强弱和涨停扩散形成的主流板块。
3. `factorStock.py`：更新全市场技术、均线均量和下跌波段字段。
4. `dailyFundamentalSelect.py`：生成“基本价值线或附近全量”和“正常基本面选股”两个独立部分。
5. `dailyReportPush.py`：汇总为价值线、正常基本面、33右侧开关、主流板块四栏，限制PushPlus正文长度，同时保存完整HTML和CSV。

## 主要工具

- `q1Backtest.py`：按指定财报期和历史截止日重建候选池。
- `rightSideBacktest.py`：候选池固定后，逐日等待下跌波段50%/62.5%右侧信号。
- `dailyFundamentalSelect.py`：从持久化财报截面与最新AkShare行情生成每日基本面分层。
- `portfolioSelect.py`：按策略分层收敛组合，不用短期涨幅补齐名额。
- `snapshotReturnUpdate.py`：用持久化AkShare缓存更新历史快照终点收益。
- `wave_utils.py`：计算前高到后低的下跌波段恢复位置。

## 数据和输出

- 行情优先使用AkShare并写入 `.cache/formula33_kline/akshare`。
- 财务截面缓存写入 `.cache/q1_value`。
- 回测结果写入 `回测结果`。
- 每日报告写入 `板块观察`，最终名单写入 `选股结果`。
- PushPlus token 使用环境变量 `PUSHPLUS_TOKEN` 或 `.pushplus_token`。

## 验证

## 动态选股与时点审计

正式日报默认使用 `dynamic` 模式：每个交易日按当时可见的最新财报、当日及以前行情、主流板块、估值和技术结构重新筛选。新财报公司、跌入基本价值线附近的公司以及多因素状态改善的公司都可以动态进入。

固定池只用于研究对照，不是正式日报默认口径：

```powershell
python fixedFundamentalPool.py `
  --report-period 2026-03-31 `
  --formation-date 2026-05-19 `
  --snapshot "回测结果/2026Q1_refactored_to_2026-06-26.csv"
```

使用固定池对照时需显式传入 `--selection-mode fixed`。未指定时，`dailyFundamentalSelect.py` 始终运行动态模式。

`point_in_time.py` 负责审计财报可见日、市场截止日和数据来源。`quarterlyFundamentalBacktest.py` 默认执行严格审计：旧历史板块成分缺少真实时点来源时会停止；只有研究复现才允许显式传入 `--allow-unsafe`，且输出旁会生成 `.meta.json` 标明风险。

## 无人值守全市场更新

`fullMarketFundamentalUpdate.py` 直接从全市场股票清单、行情缓存和对应报告期财务缓存生成动态基础截面，不再把旧回测CSV当成生产输入。每日脚本默认补抓100只缺失财务并保存断点：

```powershell
python fullMarketFundamentalUpdate.py --max-updates 100 --workers 2
```

数据接口统一优先级为 `AkShare → Baostock → 本地缓存`。AkShare成功时不会再请求Baostock；发生回退时，快照元数据会记录实际来源。生产财务缓存当前全部来自AkShare东方财富指标接口。

当前沪深支持市场基线：5534只，行情缓存4986只（90.1%），2026Q1财务覆盖2132只（38.5%），行业映射由AkShare覆盖全部已有财务样本。迁移期硬下限为行情90%、财务35%，95%才是生产完整率目标；未达到目标时输出标记为 `warning` 并发送覆盖率提醒。

完整流程：

```powershell
.\run_daily_analysis.ps1
```

可选安装Windows每日任务（默认16:30）：

```powershell
.\install_daily_task.ps1 -Time "16:30"
```

流水线会记录失败步骤、返回非零退出码，并通过 `pipelineAlert.py` 尝试发送PushPlus告警。定时任务安装脚本已提供，但不会自动修改系统任务计划。

## GitHub Actions每三天选股

工作流文件为 `.github/workflows/stock-selection.yml`：

- 定时：GitHub每天北京时间16:30唤醒一次，由工作流门控保证每连续3天实际选股一次，避免`*/3`在跨月时失真；GitHub定时任务可能有少量排队延迟。
- 手动：进入GitHub仓库的 `Actions → Stock Selection → Run workflow`，可随时执行当天选股。
- 手动参数可指定财报期、财务补抓数量以及是否跳过PushPlus。
- `.cache` 通过GitHub Actions Cache跨运行保存；首次云端运行没有缓存时自动执行冷启动全量财务构建，后续默认每次增量补100只。
- CSV、HTML、覆盖率状态和日志保存为Actions Artifact，保留14天。

如需PushPlus，在GitHub仓库 `Settings → Secrets and variables → Actions` 中添加：

```text
PUSHPLUS_TOKEN
```

没有配置Secret时，工作流仍会完成选股，只跳过推送。工作流只有提交并推送到GitHub默认分支后才会生效。

```powershell
python -m py_compile wave_utils.py factorStock.py q1Backtest.py portfolioSelect.py rightSideBacktest.py
python factorStock.py --akshare-cache-only --top 30
```

回测必须使用当时可见的财报和K线，不得用当前名单倒推历史，也不得用未来收益参与参数排序。
