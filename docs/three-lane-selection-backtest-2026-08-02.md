# 三路选股与低优先级补位回测（2026-08-02）

## 1. 研究目的

本轮只替换右侧候选来源，不修改现有买点、仓位、止损、止盈、T+1、整手、涨跌停、费用、复权或分红送转逻辑。正式基线保持：

```text
右侧：playbook_low_corr 每日 Top50
左侧：严格 PIT value_model
执行：现有统一买卖引擎
```

研究模型把右侧拆成三条独立候选路，各取 Top20 后轮转去重，最多50只，不跨路加分：

1. 公告日可见的基本面动量；
2. 平滑52周新高动量；
3. Stage 2 / VCP 准备度。

## 2. 外部研究依据

- 基本面动量：中国市场论文 [Fundamental momentum in the Chinese stock market](https://doi.org/10.1016/j.iref.2022.02.012)。论文摘要报告基本面动量与价格动量具有互补性；本项目未复刻其 LASSO 训练，只实现透明、可审计的公告事件特征。
- 52周新高：[The 52-Week High and Momentum Investing](https://doi.org/10.1111/j.1540-6261.2004.00695.x)。
- 收益路径平滑度：[Frog in the Pan: Continuous Information and Momentum](https://doi.org/10.1093/rfs/hhu003)。
- 换手不稳定惩罚：中国市场研究 [Turnover volatility and momentum](https://www.sciopen.com/article/10.26599/CJE.2022.9300405)。
- Stage 2 规则只把 Minervini 趋势模板当实现参考，不当作学术证据；参考实现为 [mark_minervini_stock_screener](https://github.com/icedevil2001/mark_minervini_stock_screener)。

SSRN 2024 中国52周新高预印本只作弱证据背景，没有用于参数拟合。Qlib 也未引入；本轮不使用 Alpha158 默认因子或机器学习黑箱。

## 3. 实现口径

### 3.1 基本面动量

使用本地 Tushare `fina_indicator`、`income`、`balancesheet`、`cashflow` 历史行。每个版本只能从其 `ann_date` 起进入公司状态，逐日使用观察日当时最新可见报告。特征包括：

- 扣非净利润同比及其环比加速度；
- 同报告期营业收入同比；
- 扣非 ROE 及其变化；
- 经营现金流/归母净利润；
- 权益资产比和杠杆改善。

本地库覆盖约5520只股票，四表分别约25万至30万行。候选审计未发现 `financial_effective_date > candidate_date`。但本地缓存无法证明供应商历史修订版本绝对完整，因此三路模型标记为“公告日 PIT、研究回测”，`strict_financial_point_in_time=false`。

### 3.2 平滑52周新高

使用观察日及以前的前复权信号价，综合252/126/63日收益、距252日高点、上涨日连续性；惩罚高换手、换手不稳定、20日单日量能尖峰、126日收益集中于单日和20日后段加速。

### 3.3 Stage 2 / VCP

硬门要求：

```text
close > MA50 > MA150 > MA200
MA200 高于20日前
close >= 1.30 * 252日低点
close >= 0.75 * 252日高点
```

准备度再结合 ATR、20/60日区间和20/50日成交量收缩，并惩罚换手不稳定和量能尖峰。

### 3.4 时点与执行

- 候选共1343个交易日，2021-01-04至2026-07-21；平均49.99只，无重复代码、缺文件或公告越界。
- 三路候选权分别为23958、23544、23091行，配额均衡。
- 技术信号使用时点前复权序列，成交使用 raw OHLC。
- 候选在观察日收盘后冻结，最早下一交易日参与执行。
- 两次回测的成交价均在当日 raw 高低价内，候选日期均严格早于买入执行日，审计违规为0。

## 4. 全周期结果

区间统一为2021-01-01至2026-07-21。

| 模型 | 收益 | 最大回撤 | 卖出胜率 | 平均已实现R | 买入/卖出 | 费用占初始资金 | 审计违规 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 正式 `low_corr + value` | +90.015% | -13.328% | 63.699% | 2.322 | 106 / 146 | 3.077% | 0 |
| 三路等配额 + value | -41.921% | -67.650% | 39.852% | 0.699 | 186 / 271 | 4.590% | 0 |
| `low_corr Top50 + smooth Top5` 补位 + value | +45.346% | -17.817% | 53.646% | 0.748 | 143 / 192 | 3.682% | 0 |

三路模型中，按仅由单一路触发过买入的股票做组合路径归因：

- `stage2_vcp`：35只，合计约亏39.36万元；
- `fundamental_momentum`：17只，合计约亏11.42万元；
- `smooth_52week_high`：32只，合计约赚5.33万元。

该归因受资金占用和交易路径影响，不能当作独立回测。因此又做了不挤占任何基线候选的保守复核：完整保留 `low_corr Top50`，每天追加最多5只 smooth 候选，且补位分数严格低于当日主流最低分。结果仍从 +90.015% 降到 +45.346%，回撤从 -13.328% 扩大到 -17.817%。

## 5. 结论

三路等配额模型和 smooth 低优先级补位均不启用。失败不是撮合、未来函数或复权混乱造成：两轮成交审计均为0违规；主要问题是新候选显著增加低质量试错、条件退出和硬止损，降低胜率、平均R并占用原 `low_corr` 机会。

正式模型继续保持 `playbook_low_corr Top50 + strict PIT value_model`。本轮代码保留为研究工具和反例审计，不接入生产候选或日报。

## 6. 产物

- 三路候选：`var/backtests/three_lane_2021_to_20260721/candidates/fundamental_smooth_high_stage2_vcp/manifest.json`
- 三路混合摘要：`var/backtests/three_lane_2021_to_20260721/portfolio/hybrid_three_lane_value/portfolio_2021-01-01_2026-07-21_summary.json`
- smooth补位候选：`var/backtests/three_lane_2021_to_20260721/candidates/playbook_low_corr_plus_smooth5/manifest.json`
- smooth补位混合摘要：`var/backtests/three_lane_2021_to_20260721/portfolio/hybrid_low_corr_smooth5_value/portfolio_2021-01-01_2026-07-21_summary.json`
