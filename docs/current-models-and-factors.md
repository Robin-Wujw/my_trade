# 当前模型与因子清单

> 当前参数和审计收益统一见[当前冻结基线](current-baseline.md)。本文第9节及带日期报告中的其他收益只作历史对照。财务修订版本和历史行业缺口仍在，左侧不能宣称严格PIT；前向记录只能从真实运行日起积累。

本文是当前分支对“用了哪些模型、实现了哪些因子、哪些模型真正进入选股”的权威记录。代码行为与本文冲突时，以代码和候选池 `manifest.json` 为准；带日期的 QuantsPlaybook 文档仅保留为研究过程和审计记录。

## 1. 当前系统模型

```text
观察日可交易 A 股全集
├─ 右侧：价格型低残差波动率组合、ADV20不低于3000万元、每日 Top50
│        └─ 技术结构、风险距离和盈亏比确认后买入
└─ 左侧：公告日可见 value_model（历史修订版本不完整）
         └─ 基本价值线网格建仓，可在独立右侧信号成立后不可逆转右

两路候选取并集，不叠加左右分数
        ↓
统一持仓、成交、止损、分仓止盈和退出引擎
```

当前正式候选接口位于 `apps/quantsplaybook_hybrid.py`。右侧候选来自 `playbook_low_corr_network_cap1_price_only_residual_volatility_liquidity_30m`，左侧候选来自 `value_model`。旧候选池、人工牛股清单和历史复盘结论不参与自动选股。

组合硬约束：

- 全部持仓标的不超过 5 只，左侧标的不超过 2 只。
- 两只左侧同时持有时，普通右侧最多再持有 3 只。
- 左转右后立即释放左侧名额，但仍占总持仓标的名额；转换不可逆。
- 同行业或高相关主题最多 2 只。
- MiniQMT 当前只读，只用于数据和回测执行画像，不连接实盘下单 API。

## 2. 现行右侧研究模型：价格型低残差波动率组合

现行组合沿用 2021 至 2023 年训练区间的低相关框架，以纯行情的20日市场残差波动率替换重复网络因子，并对六个因子的每日截面百分位等权求和。每个因子权重均为 `1/6`：

| 因子 | 类别 | 当前方向 | 核心含义 |
|---|---|---:|---|
| `network_cc` | 网络 | 越高越优 | SCC 与 TCC 的 1:1 综合中心度 |
| `ma_convergence` | 价量结构 | 越高越优 | 价格均线与成交量均线共同收敛 |
| `disposition_reversal` | 行为 | 越高越优 | 处置效应 CGO 的反向值 |
| `high_quality_momentum` | 风险与动量 | 越高越优 | 极端收益、换手波动和风险修正动量的组合 |
| `coin_team` | 行为与价量 | 越高越优 | 收盘、日内、隔夜三类收益的反向行为组合 |
| `low_market_residual_volatility_20` | 风险 | 越高越优 | 用截至前一日的60个有效样本估计市场模型，取最近20个残差收益波动率的负值 |

计算与入选过程：

1. 使用观察日当时可交易的 A 股全集。
2. 每个因子按观察日做横截面百分位排名。
3. 六个百分位等权合成组合分；个别因子缺失时按当日该股票可用因子的权重重新归一，全缺失则不产生组合分。
4. 再要求观察日可见20日平均成交额不低于3000万元，按组合分保留Top50，不在流动性过滤后重算单因子分位。`playbook_factor_score`保存组合分，`candidate_score`保存组合分在当日合格截面的百分位乘100；不叠加旧基本面评分。
5. 候选不等于买入；只有技术结构、流动性、止损距离和计划盈亏比均合格才执行。

候选池记录的 QuantsPlaybook 来源提交为 `87163521c75629a3466564c017ac734a236a9ce4`。该组合是完整历史揭盲后提出的研究替换，`selected_before_oos=false`；其历史数据计算满足行情时点规则，但全部历史结果只能标记为 `research_only`，真正未见样本从2026-08-11之后开始。

## 3. 已实现的13个原始源因子和1个价格型替换因子

全部可执行源因子实现于 `stock_research/strategies/quantsplaybook_selection.py`。下面的“方向”是项目写入因子面板后的统一方向，数值越大越优。

### 3.1 high_quality_momentum

```text
风险修正动量 = close / close.shift(61) - 1
             - 3000 * Var(前置日收益, 60日)

因子 = mean(
    20日内最大前置日收益的反向截面排名,
    60日前置换手率标准差的反向截面排名,
    风险修正动量的正向截面排名
)
```

使用 `shift(1)` 的日收益和换手波动，避免把观察日之后信息放进回看窗口。

### 3.2 shadow_reversal

```text
上影线 = high - max(open, close)
Williams 下影线 = close - low
```

分别用前 5 日均值归一化，取 20 日上影线波动与下影线均值，横截面剔除市值影响并标准化后得到 UBL。原研究做多低 UBL，因此项目存储 `-UBL`。

### 3.3 coin_team

对收盘到收盘、开盘到收盘、前收盘到开盘三类收益分别计算 20 日均值，并结合波动率相对市场的符号及换手变化相对市场的符号。三类分量求和后取负，作为项目统一的“越高越优”方向。

### 3.4 salience_str

```text
sigma = abs(个股收益 - 市场平均收益)
        / (abs(个股收益) + abs(市场平均收益) + 0.1)
显著性权重 = 0.7 ** 当日显著性排名
因子 = Cov(显著性权重, 个股收益, 20日)
```

### 3.5 low_idiosyncratic_volatility

使用本地截面构造市值加权市场、SMB 和 HML，做 20 日滚动多元 OLS，得到残差波动率。在已完成月末截面上，再剔除与前 6 个已完成月度特质波动率截面的相关部分并向日频前向填充。项目存储纯残差波动率的负值。

### 3.6 buying_pressure

```text
30日成交量加权价格 = sum(close * volume) / sum(volume)
APB = 30日简单滚动 VWAP / 30日成交量加权滚动 VWAP
因子 = mean(log(APB), 30日)
```

### 3.7 ma_convergence

价格组为 `close、MA5、MA10、MA20、MA60、MA120`，成交量组为 `volume、VMA5、VMA10、VMA20、VMA60、VMA120`。

```text
价格收敛 = -log1p(std(价格组) / 复权因子)
成交量收敛 = -log1p(std(成交量组))
因子 = mean(价格收敛截面 z-score, 成交量收敛截面 z-score)
```

### 3.8 amplitude_structure

```text
振幅 = high / low - 1
```

使用 22 日价格排序窗口和 20 日因子窗口。高价区观察赋正振幅，低价区观察赋负振幅，再取 20 日均值；前一日停牌或一字跌停导致的无效观察不参与有效振幅贡献。

### 3.9 disposition_reversal

使用成交均价和换手率构造 100 日有限窗口、换手衰减的参考价格：

```text
CGO = close / reference_price - 1
因子 = -CGO
```

### 3.10 chip_loss_overhang

使用收盘价和换手率构造 60 日有限窗口、换手衰减的筹码参考价格：

```text
ARC = 1 - reference_price / close
因子 = -ARC
```

### 3.11 network_scc

在 20 日完整收益样本上计算每只股票与其他可用股票的平均相关系数：

```text
SCC = 1 / (2 * (1 - min(平均相关系数, 0.999999)))
```

### 3.12 network_tcc

先按日把个股收益相对全市场做横截面标准化，再计算 20 日标准化距离平方均值：

```text
TCC = 1 / mean(标准化收益 ** 2, 20日)
```

### 3.13 network_cc

```text
network_cc = (network_scc + network_tcc) / 2
```

### 3.14 low_market_residual_volatility_20

对观察日时点可交易股票的有效收益做等权市场收益；股票在日期 `d` 的市场模型参数只使用截至 `d-1` 的最近60个有效样本，因子值为最近20个市场残差收益标准差的负值。停牌或缺失收益不填零，样本不足时返回缺失并关闭入选资格。

### 3.15 playbook_ensemble

`playbook_ensemble` 不是独立原始公式，而是上述可执行源因子的每日横截面百分位等权平均。它已实现并参与模型比较，但不是当前正式右侧模型。

## 4. 已实现的组合模型

组合构建位于 `apps/quantsplaybook_chunked.py`：

| 模型 | 组合方式 | 当前状态 |
|---|---|---|
| `playbook_ensemble` | 全部源因子截面百分位等权 | 研究对照 |
| `playbook_positive_ic_top5` | 2021-2023 正 Rank IC 前五等权 | 研究对照 |
| `playbook_train_ic_weighted` | 正 Rank IC 比例加权 | 研究对照 |
| `playbook_category_balanced` | 类别等权，类别内因子等权 | 研究对照 |
| `playbook_capped_ic_weighted` | 正 IC 加权，类别上限 30%、单因子上限 15% | 研究对照 |
| `playbook_low_corr` | 原始正 IC 因子中贪心选6个低相关因子并等权 | 历史基线 |
| `playbook_low_corr_network_cap1_price_only_residual_volatility_liquidity_30m` | 单一网络综合因子、五个低相关价格因子等权，加入低残差波动率和ADV20门槛 | **现行右侧研究模型** |
| `playbook_factor_quota` | 正 IC 因子间轮流分配候选名额 | 研究对照 |
| `playbook_strategy_aligned` | 动量、均线收敛、买压三因子等权 | 研究对照，回测未通过，不启用 |
| `fundamental_smooth_high_stage2_vcp` | 基本面动量、平滑52周新高、Stage 2/VCP 三路配额 | 研究对照，回测未通过，不启用 |
| `playbook_low_corr_plus_smooth5` | 保留 low_corr Top50，追加低优先级 smooth Top5 | 稳健性复核未通过，不启用 |
| `playbook_low_corr_network_cap1_price_only_residual_volatility_gross_profitability_forward_pit_liquidity_30m` | 原六因子加严格前向 GP/A，七因子等权并保留ADV20门槛 | **前向研究采集，未进入生产** |

因子类别固定为：

- `price_structure`：`shadow_reversal`、`salience_str`、`ma_convergence`、`amplitude_structure`
- `behavior_and_flow`：`coin_team`、`buying_pressure`、`disposition_reversal`、`chip_loss_overhang`
- `risk_and_momentum`：`low_idiosyncratic_volatility`、`high_quality_momentum`
- `network`：`network_cc`、`network_scc`、`network_tcc`

`playbook_strategy_aligned` 是在现行模型冻结后新增的交易语义对照，不参与原 2024 选优。其全周期收益和回撤均差于 `playbook_low_corr`，正式模型保持不变；结果见 `docs/strategy-aligned-selection-backtest-2026-08-02.md`。

三路外部研究对照和 smooth 补位在当时的研究中均未通过。三路等配额全周期收益为 -41.921%，`low_corr Top50 + smooth Top5` 为 +45.346%，落后当时 +90.015% 的历史基线；不与修复后的审计冻结结果直接比较。详细公式、时点限制和结果见 `docs/three-lane-selection-backtest-2026-08-02.md`。

### 4.1 严格前向 GP/A 研究

`apps/quantsplaybook_gross_profitability_forward.py` 实现了唯一仍在采集证据的新右侧因子：

```text
GP/A = (最新已完整公告年报营业收入 - 营业成本) / 同期总资产
effective_date = max(供应商公告日, 实际公告日, 本地payload首次捕获时间)
```

它只读取 migration 12 的追加版本表中 `2026-08-11` 以后实际捕获的 `income` 和 `balancesheet` payload，不用当前值表倒填旧观察日。公告边界取 `ann_date`、可用时的 `f_ann_date` 和首次捕获时间三者最晚值，并按 `Asia/Shanghai` 交易日归属。标准合并年报固定为12月31日报告期和 `report_type=1`；收入表和资产表都可见后才生效，后来修订只从自身首次捕获日起生效。较旧年报的迟到修订不能覆盖当时已经可见的较新年报。

金融行业、公告日缺失、营业收入/成本/总资产缺失、总资产非正、行业区间缺失或原六因子任一缺失时，该股票当日失败关闭。完整股票中七因子各占 `1/7`，继续要求 `ADV20 >= 3000万元`，保存Top100并由现有接口取Top50。买点、退出、次日开盘、T+1、整手、涨跌停、成本、左侧2%/10%和右侧3.0倍均未改变。

该模型没有历史训练收益，也没有生产资格。只有从冻结日以后逐日形成的候选、成交和同期冻结六因子对照才属于有效证据；任何用后来抓取的年报重建2021至2026结果都必须拒绝。

## 5. 现行左侧模型：value_model

```text
基本价值线 = 最新报告期每股净资产
           + 最新年报扣非每股收益
             * (1 + 最新报告期扣非每股收益同比增速)
             * 10
```

自动左侧候选还必须同时满足：

- 财务公告日严格时点可见，行业使用 `sw_member` 的 `in_date/out_date` 历史快照。
- 命中基本价值线行业白名单，且估值方法适用于该公司。
- 观察日总市值不低于 100 亿元。
- 进入当前组合回测接口时，质量分不低于 70、扣非同比不低于 10%。
- `price / value_line <= 1.08`，且不存在明确的价值或财务证伪。

左侧按 10 份价值网格管理：每格占初始资金 2%，单只左侧标的成本暴露上限 10%。左仓只有在涨离价值线且出现独立合格右侧结构时才转右；转换后不恢复成原左仓。

注意：`STRATEGY.md` 的通用“价值线或附近”展示门槛仍写有质量分 50、扣非同比非负；进入当前组合回测的候选接口会再执行更严格的 70 分和 10% 门槛。两者分别是展示层与执行层，不应混用。

## 6. 买卖引擎不是选股因子

右侧 Top50 和左侧价值线只回答“观察哪些股票”。真正成交还由 `stock_research/strategies/portfolio_backtest.py` 的统一引擎判断：

- 整理平台放量突破、有效支撑拉回、回调波段突破等可审计结构。
- MA20/MA60/MA120 位置与趋势、量能、结构距离、流动性和 Formula33 市场状态。
- 原始价格止损、计划盈亏比、首仓比例、加仓批次和组合名额。
- T+1、100 股整手、停牌、涨跌停、手续费、印花税和滑点。
- 独立止损、分仓止盈、最大浮盈回撤及剩余利润跟踪仓。

完整买卖定义见 `docs/current-selection-entry-exit-strategy.md` 和 `docs/knowledge/position-entry-exit-system.md`。选股因子得分不能绕过这些成交条件。

## 7. 时点、复权与公司行动边界

- 候选只使用观察日及以前数据，候选快照必须严格早于执行日。
- 当日收盘后生成的候选只能从下一交易日起使用。
- 技术信号使用截至信号日的前复权序列；回测成交使用原始未复权 OHLC。
- 复权因子按请求终点锚定技术序列，不把终点之后公司行动倒灌进历史信号。
- 分红、送转和拆并股通过公司行动记录调整持股数量与成本，不用复权价格虚构现金成交。
- 财务数据按公告日可见；行业按观察日成员关系重建；当前简称仅在回测结束后补充展示，不参与历史 ST 或风险判断。
- 日 K 的收盘确认成交采用 14:55/收盘价代理，不能据此声称已经复现真实盘中排队成交质量。

## 8. 未复现的方法

QuantsPlaybook 盘点包含 27 个研究入口，但本项目只把数据条件与公式均可核验的 13 条源因子实现为可执行流。未满足条件的方法没有用日频或当前字段强行近似，例如：

- APM、聪明钱 2.0、高频价量 CPV：缺完整全市场分钟数据。
- A 股反转 W：缺日内收益序列。
- 隔夜与日间网络、多因子指数增强：缺来源项目训练模型。
- 基金重仓、优秀基金经理、金股增强：缺严格时点的历史持仓、净值或分析师推荐数据。
- 行业轮动、因子择时：属于行业或元模型，不是当前个股候选因子。

完整入口与阻断原因见 `docs/quantsplaybook-factor-selection-backtest-2026-07-31.md`。

## 9. 历史 low_corr 回测（非当前基线）

连续区间 `2021-01-01` 至 `2026-07-21`：

| 指标 | 结果 |
|---|---:|
| 总收益 | +90.015% |
| 最大回撤 | -13.328% |
| 盈亏比 | 1.346 |
| 买入笔数 | 106 |
| 卖出笔数 | 146 |
| 交易审计违规 | 0 |
| 候选覆盖 | 完整 |

历史产物：

- 候选清单：`var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/candidates/hybrid_low_corr_value/manifest.json`
- 回测摘要：`var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_summary.json`
- 买卖报告：`var/backtests/quantsplaybook_hybrid_low_corr_value_2021_to_20260721/portfolio/hybrid_low_corr_value/portfolio_2021-01-01_2026-07-21_买卖报告.md`

## 10. 代码与研究记录

- 源因子定义与计算：`stock_research/strategies/quantsplaybook_selection.py`
- 组合训练与冻结：`apps/quantsplaybook_chunked.py`
- 三路研究模型：`stock_research/strategies/three_lane_selection.py`
- 三路候选生成：`apps/three_lane_candidates.py`
- 主流加低优先级补位：`apps/quantsplaybook_supplement.py`
- 左右候选合并：`apps/quantsplaybook_hybrid.py`
- 候选接口与价值门槛：`stock_research/strategies/candidate_interface.py`
- 组合成交引擎：`stock_research/strategies/portfolio_backtest.py`
- 当前完整策略：`docs/current-selection-entry-exit-strategy.md`
- 因子复现记录：`docs/quantsplaybook-factor-selection-backtest-2026-07-31.md`
- 组合比较记录：`docs/quantsplaybook-diversified-combinations-backtest-2026-07-31.md`
- 双通道回测记录：`docs/quantsplaybook-hybrid-low-corr-value-backtest-2026-07-31.md`
- 三路与 smooth 补位记录：`docs/three-lane-selection-backtest-2026-08-02.md`
