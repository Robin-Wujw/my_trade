# my_trade 文档索引

本目录只保留与当前代码一致的知识文档。业务口径以仓库根目录的 [STRATEGY.md](../STRATEGY.md) 为准；历史设计稿、实施计划和未落地目标不再作为现行文档保存。

## 当前文档

- [项目架构](knowledge/project-architecture.md)：目录、模块职责、七步调用链和运行数据边界。
- [DuckDB 实际结构](knowledge/database-schema.md)：当前迁移真正创建的 7 张表及使用限制。
- [观察日与时点规则](knowledge/point-in-time-data.md)：行情截断、观察日交易资格、财报和板块时点边界。
- [每日流水线运维](knowledge/operations-runbook.md)：完整运行、增量续跑、结果验收、故障处理和正式推送门禁。
- [数据源连接与容错改进](knowledge/data-source-resilience.md)：AkShare/BaoStock 的节流、退避、重连、异常结果和参考项目对照。
- [结果输出阅读指南](knowledge/output-guide.md)：Formula33、每日选股和板块结果先看什么、字段是什么意思、哪些不能直接当买入信号。
- [多 agent 白大协作协议](knowledge/multi-agent-baida-protocol.md)：主 agent、白大 agent、量化 agent 的职责、调用顺序和策略落地检查清单。
- [均线均量扣抵思想](knowledge/ma-volume-deduction.md)：扣抵方向、支撑压力、周期分工和项目量化映射。
- [持仓建仓、止损与分仓止盈体系](knowledge/position-entry-exit-system.md)：显式网格、左转右、独立止损、分仓止盈和提醒配置。
- [前复权未来函数与右侧选股漏斗审计](qfq-lookahead-and-right-funnel-audit-2026-07-16.md)：解释长周期回测为什么必须先处理复权锚点和时点数据。
- [右侧低频量化选股模型更新](right-side-low-frequency-quant-model-2026-07-16.md)：解释当前右侧漏斗的趋势效率、盈亏比代理、结构位置和量价确认。

日报新增的技术量化、双波段关键价和两个月突破追踪均以 [结果输出阅读指南](knowledge/output-guide.md) 为展示口径，以根目录 [STRATEGY.md](../STRATEGY.md) 为计算口径。

## 已验证基准

Formula33 固定回归区间为 2026-06-11 至 2026-07-10：

- 上市超过 300 天后的技术全量 191 只；
- 观察日无交易排除 3 只；
- 正式结果 188 只；
- 总市值大于 100 亿元的独立池 145 只；
- `001331` 在 2026-05-27 的前复权收盘价为 `48.08`。

观察日无交易股票只进入诊断，不进入正式结果。这是后续 Formula33 修改必须保持的回归标准。
