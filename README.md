# my_trade

A股收盘后分析、选股、历史截面回测与 PushPlus 汇总工具。

## 先读这里

- [策略规范](STRATEGY.md)：所有选股口径、基本价值线适用条件、右侧50%/62.5%、板块和三浪三规则。它是策略实现的唯一准则。
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

```powershell
python -m py_compile wave_utils.py factorStock.py q1Backtest.py portfolioSelect.py rightSideBacktest.py
python factorStock.py --akshare-cache-only --top 30
```

回测必须使用当时可见的财报和K线，不得用当前名单倒推历史，也不得用未来收益参与参数排序。
