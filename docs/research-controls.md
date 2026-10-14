# 同股票池选股对照实验

`python -m apps.selection_controls` 只读取本地输入、生成候选与核对已完成回放，不调用网络、通知或券商。输出始终是 `research_only`，不认证严格时点数据。没有完成至少 100 次随机回放时，不输出基线分位或前 5% 判断。

## 输入契约

提供两个 CSV，股票代码使用 `sh.600000`、`sz.000001`、`bj.920001`，日期使用 `YYYY-MM-DD`，每个日期/代码唯一：

- 因子表：`date,code,data_cutoff,average_amount_20_yuan,turnover_rate,close`，以及 `network_cc,ma_convergence,disposition_reversal,high_quality_momentum,coin_team,low_market_residual_volatility_20`。可选 `market_cap_yuan` 用于规模暴露。成交额单位为元，原 Tushare amount 若为千元须在上游明确转换。`close` 的复权口径写入来源声明。
- 成员表：`date,code,membership_as_of,is_listed,is_traded,is_st,history_days`。必须来自显式历史成员资格、历史ST和当日交易状态，不能用当前股票列表替代。布尔字段只接受 true/false 或 1/0。历史至少60个交易日，已上市、当日交易、非ST。
- 所有 cutoff/as_of 必须不晚于观察日。该检查不能证明原始缓存具有真实历史版本；外部数据修订限制仍适用。因子历史窗口/复权、财务公告与行业快照时点由上游构建者保留证据。

来源 JSON 至少包含 `observation_dates`（所有应有观察日的显式数组）、`panel_sha256`、`membership_sha256`、`universe_definition`、`adjustment_policy`、`availability_policy`。后三项为非空说明文字。日期缺失、重复、未知标志、缺少必要因子列或ADV会报错，不静默缩小共同股票池。单个因子因预热窗口而缺值时，六因子按原模型的可用因子权重重新归一化；不要求六因子同时有值。随机对照保留整个共同ADV股票池，简单因子对照仅对该因子可用值排序，当日不足50只则失败。

另提供冻结 profile（如 `config/strategy_baseline.json`，必须含完整 `engine_parameters`）和共享左侧候选目录（带当前版本manifest、逐日文件SHA256及日期覆盖）。生成工具核对文件，输出绑定其hash。左侧源必须明确覆盖各日；无左侧持仓策略也需要显式的空左侧快照证据。

## 生成

PowerShell示例（路径替换为已准备的本地文件）：

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.selection_controls generate --panel var/research_inputs/factors.csv --membership var/research_inputs/membership.csv --source-manifest var/research_inputs/source.json --profile config/strategy_baseline.json --left-candidates var/research_inputs/shared_left --output-directory var/research_controls/example --seed-start 0 --seed-count 100
```

生成103个**右侧**候选目录：六因子基线、低换手、低残差波动、100个随机种子。六因子先在完整的当日PIT可交易截面计算等权因子分位，再筛ADV20≥3000万元、取Top50，保持冻结模型顺序；不在流动性筛选后重算六因子分位。低换手按换手率升序；低残差波动字段是**负残差标准差**，按其降序。随机排名使用 `SHA256(seed|date|code)`，不受行顺序或未来日期追加影响。相同分数以代码排序。

每天全部对照使用同一ADV股票池，至少50只，否则失败。`style_exposures.csv` 给出换手率、ADV、残差分数、价格及可选市值的中位数和共同股票池内分位中位数。若市值列不存在则不报告规模暴露，不能据此宣称不存在规模偏向。

`experiment.json` 保存源文件、profile、左侧、候选及暴露hash，种子、观察日和共同股票池。此时 `completed_random_replays=0`。输出目录必须不存在，避免覆盖旧实验证据。

## 实际回放与评价

生成器没有执行交易回测。外部回放执行器必须用各右侧候选替换右侧输入，原样保留共同左侧、价格、公式阶段、计划、公司行动、退市日期与完整engine参数；保留成本、T+1、整手、停牌及涨跌停规则。不要把右侧目录直接当作完整双侧输入使用。

候选没有历史行业来源认证，`industry_point_in_time=false`。现有严格冻结入口因此可能拒绝；不得修改标志绕过门禁。研究回放如使用既有 `--allow-unsafe-industry`，必须保留该限制及独立账目审计。工具不会自动开启这些选项。旧118.712%产物不能直接作为新实验基线；候选和所有共同输入不完全相同就必须重跑对应基线。

每个实际完成的summary需要 `frozen_run`：`status=finished`、`accounting_status=passed`或`data_limited`、有效acceptance、完整 `engine_parameters`、`engine_sha256`、`code_sha256`、`input_bundle_sha256`。独立账目报告必须存在，`error_count=0`且无errors；`data_limited`不能改写为passed，整体结果也必须保留data_limited。它还必须包含 `selection_control` 对象，其值严格等于执行器收据中的以下四项：

```json
{
  "control_id": "random_0000",
  "candidate_manifest_sha256": "<实际候选manifest的SHA256>",
  "profile_sha256": "<生成实验时profile的SHA256>",
  "shared_input_sha256": {
    "prices": "<共同价格输入hash>",
    "formula_phases": "<共同阶段输入hash>",
    "trade_plans": "<共同计划输入hash>",
    "corporate_actions": "<共同公司行动输入hash>",
    "security_end_dates": "<共同退市日期输入hash>",
    "left_candidates": "<共享左侧manifest的SHA256>"
  }
}
```

这些摘要必须由实际执行器在冻结/执行时计算并嵌入，而非根据希望比较的控制名称事后补写。评价器核对收据和summary，但不反序列化pickle、不独立认证外部执行器真实性。

收据 JSON 格式为 `{"experiment_sha256":"...", "replays":[...]}`；每个replay包含上述四项，再加 `summary_path`（相对收据文件）、`summary_sha256`、`input_bundle_sha256`，以及 `accounting_status`、`accounting_error_count`、`accounting_audit_path`、`accounting_audit_sha256`。各summary必须绑定不同输入bundle，不能把同一份完成结果计作多个随机种子。引擎、代码、共同输入、acceptance和数据限制不一致则拒绝比较。

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.selection_controls evaluate --experiment var/research_controls/example/experiment.json --receipts var/research_controls/receipts.json --output var/research_controls/comparison.json
```

默认以累计收益比较。全部预声明控制（含≥100个随机种子）完成后才输出：基线在随机分布的并列中位分位，以及 `(1+随机收益不低于基线的次数)/(随机次数+1)` 的上尾经验概率；后者≤5%才标记前5%。这是描述性研究结果，不能单凭它证明六因子贡献；还需结合简单因子对照、风格暴露及选模/时点限制。

## 截至2026-07-21的历史执行器

`python -m apps.historical_selection_controls` 将既有完整市场因子缓存投影为纯观察日因子，丢弃 `forward_return`、`label_date`；保持原冻结基线的六因子可用权重算法和共享左侧。共同价格扩展到全部ADV股票池，原冻结股票沿用原价格及公司行动，新增股票使用本地已准备价格和只读SQLite快照；不下载或覆盖缓存。Tushare成交额从千元转为元，市值从万元转为元。日期必须不晚于2026-07-21；未满足ADV覆盖时失败，不把缺数当作无交易。

准备阶段生成103个紧凑的候选pickle和hash清单，并归档实际执行源码及依赖版本。`run`从该归档目录启动顺序子进程，屏蔽SQLite和网络连接，执行共同引擎参数与独立账目审计；只复用hash及绑定检查通过的已完成收据。生成候选、现金组合测试或程序计时不计作完成随机实验。研究数据的财务修订、行业历史、价格限制覆盖及公司行动支付日缺口保留在结果中。

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.historical_selection_controls prepare --panel-directory var/backtests/one_network_price_only_low_residual_volatility_experiment/research_panels --baseline-bundle var/backtests/execution_timing_audit_20260921/frozen_verified --price-directory var/backtests/quantsplaybook_factor_only_2021_to_20260721/prepared_prices --database var/data/my_trade.sqlite3 --profile config/strategy_baseline.json --output-directory var/research_controls/historical_pilot --start-date 2021-01-01 --end-date 2021-12-31
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.historical_selection_controls run --experiment var/research_controls/historical_pilot/experiment.json --controls baseline_six_factor random_0000 --profile-execution
```

先用2021基线和第0个预声明种子记录实际耗时、峰值内存与cProfile热点，再决定完整区间的执行规模；不能依据收益选择种子。`--profile-execution`会增加计时开销。正常完整运行省略该标志与 `--controls`。每个完成回放输出权益、成交、事件、summary、独立账目审计和绑定收据；`comparison.json`在不足100个完整随机回放时保持 `incomplete_replays`，不输出显著性结论。

## 2026-10-14实际试跑与等价优化

本轮历史验收截止2026-07-21，选股对照先实际执行2021年的243个交易日。`var/research_controls/historical_2021_pilot_20261014_v2` 已生成103个预声明控制，**实际完成基线及random_0000，只有1个随机回放**；还有101个控制未完成，不能报告随机分位、前5%或六因子超额贡献。2021试跑不是2021—2026完整区间的100次实验。

准备共同ADV股票池得到774,152条成员观察，耗时494.11秒、峰值1867.44MiB。`baseline_selection_comparison.json` 证明243日基线候选集合、排名及candidate_score均与原冻结基线一致，原候选的左右重叠数为0，左侧输入完整保留。试跑前将区间外公式阶段、公司行动及未来终止日期裁剪，并记录裁剪前后共同输入hash；因子选择输入不含未来收益和标签日期。

| 2021实际回放 | 累计收益 | 最大回撤 | Sharpe | 带cProfile耗时 | 峰值内存 | 独立账目审计 |
|---|---:|---:|---:|---:|---:|---|
| 原代码六因子基线 | +12.193% | -9.811% | 0.818 | 728.665秒 | 944.58MiB | passed，0错误 |
| 原代码random_0000 | +9.210% | -22.920% | 0.434 | 862.088秒 | 1021.78MiB | data_limited，0错误 |
| 优化代码六因子基线 | +12.193% | -9.811% | 0.818 | 140.213秒 | 942.61MiB | passed，0错误 |

random_0000的账目限制为 `sz.300642 / 2021-07-14 / pay_date_missing_or_before_ex_date`，保留真实`data_limited`状态；没有把它改成passed。两个研究回放整体均保持`research_only / data_limited`，原历史财务/行业版本、复权与交易限制覆盖条件不因零账目错误而升级。

最初的旧归档执行器只接受passed退出码，已修正为接受实际独立审计`passed`或`data_limited`且错误为0。旧试跑的summary和真实audit未改；补充收据绑定audit文件hash/状态/错误数，保留 `receipt_before_accounting_attestation.json` 及补证源码hash，属于同一次已完成回放的证据补充。

cProfile定位到大量混合DataFrame行的重复标量读取：结构锚点推断515.33秒，量价节点检查100.58秒。优化只在这两个函数预取列数组，保持原窗口、遍历顺序、比较、浮点运算及同分处理。旧归档函数与新函数对128只固定抽样股票、17个固定历史日期的1995个前缀逐项差分，所有完整结构与节点结果一致；相同、不带cProfile的函数计时分别为42.30→8.58秒、6.56→1.62秒。

优化版单独保存在 `var/research_controls/historical_2021_array_optimized_20261014`。它复用逐字节相同的共同输入、profile、左侧及右侧候选；2021完整权益、成交和事件三份CSV与原代码SHA256全部一致，保持15笔买入、21笔卖出。两次全程均开启cProfile，记录耗时约缩短为1/5.20；该优化验证不计作新的随机策略实验。

差分与逐笔回放证据分别在 `var/research_controls/signal_array_equivalence_20261014/differential.json` 和 `portfolio_parity_2021.json`。函数差分检查脚本及源码、价格文件hash一并记录；执行器恢复顺序、收据补证、输入复制与产物一致性专项测试9项通过。截止此处，100次完整区间随机回放尚未执行，归因结论继续保持未完成。
