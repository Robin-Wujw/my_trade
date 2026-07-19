# my_trade 文档索引

本目录只保留与当前代码口径一致、仍能指导生产或审计的文档。旧参数扫表、低收益诊断、临时候选分析和单票复盘已清理，避免和最终策略混在一起。

业务口径以仓库根目录的 [STRATEGY.md](../STRATEGY.md) 为总纲；当前自动候选和买卖细则以 [当前选股策略（完全版）与买卖策略（详细版）](current-selection-entry-exit-strategy.md) 为准。

## 当前策略

- [当前选股策略（完全版）与买卖策略（详细版）](current-selection-entry-exit-strategy.md)：最终候选池、右侧多因子模型、语义高盈亏比买入门、仓位、止损、止盈和白大符合度审查。
- [右侧低频量化选股模型更新](right-side-low-frequency-quant-model-2026-07-16.md)：右侧因子模型的研究说明。
- [前复权未来函数与右侧选股漏斗审计](qfq-lookahead-and-right-funnel-audit-2026-07-16.md)：时点数据和前复权风险审计。
- [Tushare 数据清洗与差异复盘](tushare-data-cleaning-and-divergence-review-2026-07-16.md)：外部数据差异的处理记录。
- [VectorBT 交叉验证](vectorbt-cross-validation.md)：组合回放和向量化验证边界。

## 知识库

- [项目架构](knowledge/project-architecture.md)：目录、模块职责、七步调用链和运行数据边界。
- [DuckDB 实际结构](knowledge/database-schema.md)：当前迁移真正创建的表和使用限制。
- [观察日与时点规则](knowledge/point-in-time-data.md)：行情截止、观察日交易资格、财报和板块时点边界。
- [每日流水线运维](knowledge/operations-runbook.md)：完整运行、增量续跑、结果验收、故障处理和正式推送门禁。
- [数据源连接与容错改进](knowledge/data-source-resilience.md)：AkShare/BaoStock 的节流、退避、重连和异常处理。
- [结果输出阅读指南](knowledge/output-guide.md)：Formula33、每日选股和板块结果如何阅读。
- [多 agent 白大协作协议](knowledge/multi-agent-baida-protocol.md)：主 agent、白大 agent、量化 agent 的职责和检查清单。
- [均线均量扣抵思想](knowledge/ma-volume-deduction.md)：扣抵方向、支撑压力、周期分工和量化映射。
- [持仓建仓、止损与分仓止盈体系](knowledge/position-entry-exit-system.md)：显式网格、左转右、独立止损、分仓止盈和提醒配置。
- [FinHack 与 MiniQMT 学习记录](knowledge/finhack-miniqmt-study-2026-07-18.md)：外部框架只作为研究参考，未引入生产依赖。

## 固定验收基准

Formula33 固定回归区间为 `2026-06-11` 至 `2026-07-10`：

- 上市超过 300 天后的技术全量：191 只
- 观察日无交易排除：3 只
- 正式结果：188 只
- 总市值大于 100 亿元的独立池：145 只
- `001331` 在 `2026-05-27` 的前复权收盘价：`48.08`

观察日无交易股票只进入诊断，不进入正式结果。
