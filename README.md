# my_trade 选股系统

当前项目只把 `factorStock.py` 作为主选股场景。其他旧脚本保留为历史参考或手动诊断，不再进入定时任务。

## 当前入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `factorStock.py` | 主入口 | 融合基本价值线、PE/PB 历史低估、质量、趋势、流动性、主线强弱 |
| `targetBacktest.py` | 手动验收 | 验证指定财报节点能否选出目标股票 |
| `trade_utils.py` | 公共工具 | PushPlus 推送、历史结果保存、接口重试 |
| `run_1630.sh` | 定时入口 | 每个交易日 16:30 执行 `factorStock.py` |
| `valueStock.py` / `valuePEPB.py` / `selectStock.py` / `stockpush.py` | 旧逻辑 | 核心思想已并入 `factorStock.py`，不参与定时 |

当前 crontab 只保留：

```bash
30 16 * * 1-5 /root/my_trades/my_trade-main/run_1630.sh
```

服务器提交后建议使用统一入口：

```bash
30 16 * * 1-5 /root/my_trades/my_trade-main/run_1630.sh
```

`run_1630.sh` 会调用 `run_daily_analysis.sh`，依次生成：

- `选股结果/factor_selection_*.csv`：每日因子选股和价值线附近观察。
- `选股结果/factor_diagnostic_*.csv`：全市场诊断候选。
- `板块观察/formula33_stats_*.xlsx/csv`：33公式全市场结构统计。
- `板块观察/sector_stats_*.xlsx/md`：板块横向统计和文字复盘。
- `板块观察/sector_watch_*.csv`：板块主线强度排行。

日志位置：

```bash
/root/my_trades/my_trade-main/daily_analysis.log
/root/my_trades/my_trade-main/logs/daily_analysis_YYYYMMDD.log
```

服务器上提交后先做一次语法和入口检查：

```bash
cd /root/my_trades/my_trade-main
python3 -m py_compile factorStock.py formula33Stats.py sectorStats.py sectorWatch.py strategyResearch.py q1Backtest.py fixBacktestReturns.py wave_utils.py
bash -n run_daily_analysis.sh run_1630.sh
bash ./run_daily_analysis.sh
```

`run_17.sh`、`run_1730.sh`、`run_1745.sh`、`run_18.sh` 已禁用，手动运行也只写跳过日志。

## 策略主线

`factorStock.py` 先按行业决定估值体系，再把股票放入互斥的结果桶：

1. `低估且高质量`：同时满足低估硬条件和高质量硬条件，优先级最高。
2. `低估价值`：只满足低估逻辑，强调估值安全垫。
3. `高质量趋势`：质量、趋势、流动性达标，且估值不过分离谱。
4. `观察池`：不推送、不保存为入选结果。

同一只股票不会重复出现在多个桶里。若同时满足低估和高质量，会进入 `低估且高质量`，不再重复进入后两组。

## 行业估值分流

### VALUE 股票

VALUE 指使用“基本价值线”的股票，主要覆盖非金融、非地产、非周期资源、非公用事业的公司，例如科技硬件、通信设备、电子、医药、消费、高端制造、汽车等。

基本价值线公式：

```text
基本价值线 = 最新报告期每股净资产
           + 最新年报扣非每股收益 * (1 + 最新报告期扣非每股收益同比增速) * 10
```

VALUE 只适合产业趋势向上、利润可外推、财务质量相对可信的公司。因此主流程额外加了硬门槛：

- 市值必须 `>= 100 亿`，避免基本价值线误用到太小的公司。
- 扣非 EPS 必须为正。
- 最新年报扣非 EPS 优先使用财务指标里的直接披露字段，不用“扣非净利润 / 期末总股本”反推；如果最新报告期是次年一季报，且最新年报分配方案含送转股，则把年报原始扣非 EPS 按送转实施月份做月度加权可比口径调整。
- 最新报告期扣非同比优先使用扣非 EPS 同比，缺失时用同报告期本期/上年同期扣非净利润同比近似。
- 价值线公式使用原始扣非同比，避免偏离复盘公式；质量评分会把同比截断到 `[-50%, 100%]`，避免低基数暴增把质量分打满。
- 近年扣非利润稳定性会进入质量分。

新易盛 2025 一季报后的标准样例：

```text
接口：
1. ak.stock_financial_analysis_indicator_em("300502.SZ", "按报告期")
   2025-03-31 BPS = 14.0070216
   2025-03-31 KCFJCXSYJLRTZ = 383.0965939%
   2024-12-31 EPSKCJB = 3.99
2. ak.stock_fhps_detail_em("300502")
   2024年报分配方案 10转4，除权除息日 2025-05-28

年报扣非 EPS 可比口径：
送转月度加权因子 = 1 + 4/10 * 8/12 = 1.2666667
扣非 EPS = 3.99 / 1.2666667 = 3.15

基本价值线：
14.0070216 + 3.15 * (1 + 3.830965939) * 10 = 166.18，约 166
```

低估硬条件：

```text
现价 / 基本价值线 <= 1.00
```

高质量趋势硬条件：

```text
现价 / 基本价值线 <= 1.80
质量分 >= 70
趋势分 >= 60
流动性分 >= 40
```

### PE/PB 股票

PE/PB 指不适合用基本价值线外推的股票。

- PE：银行、保险、食品饮料、批发零售、医药、纺织服装等盈利相对稳定行业。
- PB：周期资源、重资产、公用事业、地产、券商、电信运营、建筑工程、交运仓储、农林牧渔等利润周期波动大或资产属性更重的行业。

低估硬条件：

```text
当前估值 <= 10 年低估均值
或
历史分位 <= 15%
```

这里的 10 年低估均值取每年最低 PE/PB，剔除明显异常值后求均值。历史回测时，会按行情数据最后日期判断当年是否需要剔除未充分交易的年份。

### RIGHT 行业

RIGHT 指轻资产、强预期、左侧估值容易失真的行业，例如软件、互联网、游戏、部分信息技术服务。

RIGHT 行业不进入 `低估价值` 组，只进入 `高质量趋势` 或 `观察池`。原因是这类股票通常不是靠低 PE/PB 或基本价值线给买点，而是靠趋势、资金、景气度和右侧确认。

## 打分

低估价值：

| 类型 | 估值 | 质量 | 趋势 | 流动性 |
|------|------|------|------|--------|
| VALUE | 50% | 30% | 10% | 10% |
| PE/PB | 55% | 25% | 10% | 10% |

高质量趋势：

| 类型 | 估值 | 质量 | 趋势 | 流动性 |
|------|------|------|------|--------|
| RIGHT | 0% | 35% | 45% | 20% |
| VALUE / PE / PB | 5% | 45% | 35% | 15% |

低估且高质量：

```text
核心分 = 低估分 * 50% + 高质量分 * 50%
```

默认门槛：

```text
低估且高质量 >= 80
低估价值 >= 75
高质量趋势 >= 80
```

## 主线判断

主线不是单独的买入条件，而是用于判断当前市场资金更偏向哪些方向。输出中会显示 `主题`、`主线分`、`主线状态` 和 `主线参考`。

主线分构成：

| 因子 | 权重 | 含义 |
|------|------|------|
| 相对上证 60 日收益 | 35% | 是否长期强于大盘 |
| 相对上证 20 日收益 | 20% | 是否短期仍强 |
| 20 日 / 120 日成交额 | 20% | 是否持续有量 |
| 大盘下跌日胜率 | 15% | 大盘跌时是否更抗跌 |
| 早修复 | 10% | 是否早于大盘结束调整或重新站上短均线 |

标签：

```text
主线强势：主线分 >= 80
主线观察：主线分 >= 65
普通：主线分 < 65
```

这套指标用于识别类似 2025 年 AI 产业链/CPO、2025 年 9 月和 2026 年 4-5 月半导体这类主流方向：持续有量、相对大盘强、大盘回撤时抗跌或先修复。

## 提速设计

- `--workers` 支持多进程并行，定时任务默认 4 个进程。
- akshare 财报请求设置 12 秒超时，避免单票卡死全局。
- VALUE 基本价值线优先使用 `akshare` 东方财富/新浪财务指标里的直接扣非 EPS 字段，失败后回退 `adata` 东方财富 F10 和 `akshare` 同花顺财务摘要；送转股调整使用 `ak.stock_fhps_detail_em`。
- VALUE 基本价值线结果缓存到 `.cache/factor_value`，缓存 24 小时；日内重复运行主要只用新价格重算折价和市值。
- `--limit` 可做小样本调试，不影响生产参数。

推荐生产命令：

```bash
python3 -u factorStock.py --top 30 --core-min-score 80 --low-min-score 75 --quality-min-score 80 --value-min-mktcap 100 --workers 4
```

调试命令：

```bash
python3 factorStock.py --limit 300 --no-push --value-min-mktcap 100 --workers 4
python3 targetBacktest.py
```

## 指定节点验收

`targetBacktest.py` 只验证当前要求的财报节点：

| 股票 | 代码 | 财报节点 | 要求 |
|------|------|----------|------|
| 新易盛 | `sz.300502` | 2025 一季报 | 披露后必须能入选 |
| 中际旭创 | `sz.300308` | 2025 一季报 | 披露后必须能入选 |
| 工业富联 | `sh.601138` | 2025 中报 | 披露后必须能入选 |
| 洛阳钼业 | `sh.603993` | 2025 中报 | 披露后必须能入选 |
| 紫金矿业 | `sh.601899` | 2025 中报 | 披露后必须能入选 |

这个脚本只做选股节点验收，不做买卖回测。买入、卖出、调仓、交易成本、涨跌停成交约束后续单独建模。

## 输出

运行后会：

- 控制台打印入选股票、三组结果、主线观察。
- 保存 CSV 到 `选股结果/factor_selection_日期_时间.csv`。
- 保存上一日对比到 `.factorStock_last.json`。
- 配置 PushPlus 后推送压缩摘要、三组前排结果和新增/移除对比；全量字段以 CSV 为准。

## 数据源

- 股票列表、交易日、行情、PE/PB、利润表：baostock
- 财报摘要、基本价值线所需扣非数据：akshare / adata
- 推送：PushPlus

## PushPlus 配置

脚本不硬编码 token。任选一种方式配置：

```bash
export PUSHPLUS_TOKEN="你的PushPlus token"
```

或在项目根目录创建 `.pushplus_token`，文件内容只放 token。

cron 默认不会加载交互 shell 的环境变量，定时任务建议使用 `.pushplus_token` 文件方式。

## 依赖安装

```bash
python3 -m pip install baostock pandas numpy openpyxl requests akshare adata -i https://pypi.doubanio.com/simple
```

## 策略研究与验证

`strategyResearch.py` 用来做多策略、多参数扫描，不再只凭单一质量分或趋势分下结论。当前默认训练集为：

- 2025 一季报后：`回测结果/q1_backtest_2025-05-06_2026-05-18_wide_merged_returns_fixed.csv`
- 2025 中报后：`回测结果/q1_backtest_2025-09-01_2026-05-19_162431_returns_fixed.csv`
- 2026 一季报后验证：`回测结果/q1_backtest_2026-05-19_2026-05-19_151928_returns_fixed.csv`

先修复验证集收益率，再扫描 300/688：

```bash
python fixBacktestReturns.py "回测结果\q1_backtest_2026-05-19_2026-05-19_151928.csv" --buy-date 2026-05-19 --end-date 2026-06-22 --source baostock --code-prefixes sz.300,sh.688 --sleep 0
python strategyResearch.py --prefixes sz.300,sh.688 --out "回测结果\strategy_research_300688_latest.csv" --report-out "回测结果\strategy_report_300688_latest.csv"
```

当前 300/688 验证结论：

- `value_line_only` 在 2025 训练期收益极高，但 2026 一季报验证弱，属于明显过拟合，不能直接作为每日推送主策略。
- 更稳的方向是右侧/主线动量：`all` 或 `right_momentum` + `momentum` 排序，验证 top20 均值约 39%，中位数约 22%，胜率约 85%。
- 小样本弹性最好的是 `value_pullback` + 60 日涨幅过滤，验证 top5 均值约 48%，胜率约 80%，但样本少，需要继续扩展节点验证。

这意味着每日推送应分成两层：一层做“右侧主线候选”，一层做“价值线附近观察”，不要让价值线股票被趋势分直接过滤掉。

## 每日输出样式

`factorStock.py` 的控制台和 PushPlus 都按同一套日报顺序展示：

```text
每日交易观察
交易日: 2026-06-24
阅读顺序: 主线主题 -> 右侧主线候选 -> 价值线附近观察 -> 左侧低估组合 -> 风险提示

1. 主线主题
主题        入选数  平均主线分  20日均涨幅  60日均涨幅  代表股票
半导体        8      78.5       18%       42%      中船特气、华特气体、莱特光电
AI硬件        5      73.2       12%       35%      仕佳光子、联瑞新材、长盈通

2. 右侧主线候选
代码        名称      主题    现价   综合分  质量  趋势  主线分  20日涨幅  60日涨幅  风险
sh.688xxx  中船特气  半导体  xx.xx  88.0   82.0  91.0  86.0   22%     65%     -

3. 价值线附近观察
代码        名称      主题      现价   现价/价值线  价值线  质量  趋势  未入选原因
sh.600699  均胜电子  汽车电子  23.79    1.02      23.30  67.6  6.6   趋势分不足，质量分不足

4. 左侧低估组合
4.1 低估且高质量
4.2 低估价值

5. 风险提示
1. 先看板块/主题是否持续有量，再看个股是否进入右侧或回到价值线附近。
2. 价值线附近观察不等于正式买入，趋势未确认时只适合盯盘和等待右侧信号。
3. 右侧主线候选波动通常更大，追高前需要结合量能、扣抵价和波段分位复核。
```

## 板块主线观察

`sectorWatch.py` 用来展示和复盘板块主线，不直接替代个股选股。建议每天生成一张板块观察表：

```bash
python sectorWatch.py --top 30 --days 80 --limit-up-days 5
```

输出字段包括：

- `ret3/ret5/ret20`：短中期板块涨幅。
- `amount_5_20`、`amount_20_60`：近 5 日、20 日成交额相对基准量能。
- `weak_resilience`：大盘弱时板块是否抗跌或先修复。
- `strong_attack`：大盘强时板块是否更强。
- `limit_up_count`：最近涨停股数量，用来观察板块情绪扩散。
- `mainline_score/final_score`：综合主线强度。

评分思路是：涨幅确认趋势，量能确认资金，弱市抗跌确认韧性，强市进攻确认弹性，涨停数量确认情绪扩散。运行依赖 akshare/东方财富网络接口；如果接口或代理不可用，脚本会失败，需要换网络环境后重新生成。

## 板块横向统计

`sectorStats.py` 用来生成类似横向 Excel 复盘表的统计文件，适合每天收盘后看“主线是否延续、是否扩散、是否有量”。

真实数据命令：

```bash
python sectorStats.py --lookback 10 --history-days 90 --top-amount 50
```

离线样例命令：

```bash
python sectorStats.py --sample --lookback 8 --top-amount 5
```

输出目录：

```text
板块观察/sector_stats_日期_时间.xlsx
板块观察/sector_stats_日期_时间.md
```

Excel 第一张表按日期横向展开：

```text
一、成交额Top板块数量
分类        06.15  06.16  06.17  06.18  06.19  06.22  06.23  06.24
有色资源类      2      1      3      2      2      1      2      2
半导体          1      1      1      1      1      1      1      1
元器件          1      2      1      2      1      2      1      2
通信设备        1      1      1      1      1      1      1      1
电气设备        0      1      1      0      1      1      1      1
其他板块        0      0      0      0      0      0      0      0
合计            5      6      7      6      6      6      6      7

二、涨停扩散数量
三、三浪/放量候选数量
四、开盘八法/盘面状态
```

第二张表是文字分析，形式类似：

```text
一）板块复盘分析（2026-06-24）

1. 有色资源类：成交额靠前板块：贵金属、能源金属；涨停扩散数：0；三浪候选：能源金属、贵金属。
2. 半导体：成交额靠前板块：半导体；涨停扩散数：2；三浪候选：半导体。
3. 元器件：成交额靠前板块：电子元件、消费电子；涨停扩散数：2；三浪候选：无。

结论：优先观察连续进入成交额前列、同时涨停扩散和三浪候选都增加的方向；只有价值回归但无量能的板块，放入观察不追。
```

## 33公式市场结构统计

`formula33Stats.py` 用来统计通达信 33 公式最近 21 个交易日的全市场命中数量，范围为沪深A股，K线使用前复权。

公式：

```text
KD:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
K:=SMA(KD,3,1);
D:=SMA(K,3,1);
KD80:=K>80;
WR1:=100*(HHV(HIGH,10)-CLOSE)/(HHV(HIGH,10)-LLV(LOW,10));
WR2:=100*(HHV(HIGH,20)-CLOSE)/(HHV(HIGH,20)-LLV(LOW,20));
WR3:=WR1<20 AND WR2<20;
RSI70:=SMA(MAX(CLOSE-REF(CLOSE,1),0),9,1)/SMA(ABS(CLOSE-REF(CLOSE,1)),9,1)*100>70;
MKT_CAP:=FINANCE(40)/10000>100;
LIST_DAYS:=FINANCE(42)>300;
BASE:=KD80 AND WR3 AND RSI70 AND MKT_CAP AND LIST_DAYS;
XG:COUNT(BASE,5)=5;
```

真实数据命令：

```bash
python formula33Stats.py --lookback 21 --history-days 90 --workers 12 --price-source akshare --market-cap-source akshare-capital
```

数据源说明：

- `--price-source akshare`：使用 akshare 前复权日 K，避免 baostock 临时限流。
- `--market-cap-source akshare-capital`：用 akshare 股票列表、股本结构和现价估算总市值，绕开东方财富总市值列表接口。
- `--market-cap-source tushare`：如已配置 Tushare token，可用 Tushare `daily_basic.total_mv`。
- `--market-cap-source none`：只用于临时复核技术指标，会跳过 `FINANCE(40)` 市值过滤，不作为正式统计。

离线样例命令：

```bash
python formula33Stats.py --sample --lookback 21
```

输出文件：

```text
板块观察/formula33_stats_日期_时间.xlsx
板块观察/formula33_stats_日期_时间.csv
```

判断规则：

```text
连续上升 3 天：结构初步转好，右侧交易成功率提升
连续上升 5 天：结构转好确认
连续下降 3 天：结构初步转坏，右侧交易成功率下降
连续下降 5 天：结构转坏确认
```

Excel 包含三张表：

- `33公式日统计`：日期、命中数量、较前日变化、连续上升/下降天数、结构信号。
- `横向统计`：最近 21 个交易日横向展开，便于像截图那样复盘。
- `命中股票`：每天具体命中的股票、收盘价、市值、上市天数和 KDJ/WR/RSI 数值。

## 波段分位

`wave_utils.py` 把上涨波段拆成 50%、62.5%、75% 分位，`retracement_gui(1).py` 已接入这些计算，用于右侧持仓时观察回撤位置：

- 回撤到 50% 附近：观察是否有承接。
- 回撤到 62.5% 附近：趋势仍可修复，但需要更强确认。
- 跌破 75%：右侧波段结构明显变弱。

这部分更适合持仓管理和复盘，不建议单独作为买入条件。

## 价值线附近观察

`factorStock.py` 新增“价值线附近观察”，默认把 `现价 / 基本价值线 <= 1.08` 且质量不过差的股票单独列出。它不等同于正式入选，而是解决类似均胜电子、海康威视这类“跌回基本价值区但趋势分暂时不高”的观察需求。

常用参数：

```bash
python factorStock.py --value-watch-ratio 1.08 --value-watch-top 20
```

## 注意事项

- 当前策略只解决“选股”，不解决买卖点。
- VALUE 折价异常大时仍需人工复核，可能是利润增速或财报字段导致价值线失真。
- RIGHT 行业不做左侧低估判断，强趋势也可能高波动。
- PE/PB 低估更适合行业均值回归，不代表短期马上上涨。
- 小市值问题当前对 VALUE 做硬过滤；其他类型主要通过流动性和趋势约束过滤。
