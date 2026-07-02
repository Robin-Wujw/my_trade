# my_trade 文档索引

本目录记录系统当前结构、目标架构、数据时点规则和运行维护方式。策略业务口径仍以仓库根目录的 `STRATEGY.md` 为准。

## 设计规范

- [数据基础与工程分层重构设计](superpowers/specs/2026-07-01-data-architecture-refactor-design.md)：本轮已经确认的完整设计、边界、迁移顺序和验收条件。
- [阶段一基础边界实施计划](superpowers/plans/2026-07-02-phase-one-foundation.md)：包骨架、运行上下文、DuckDB 迁移、运行记录和历史回归门的实施与验证步骤。

## 知识文档

- [项目架构知识库](knowledge/project-architecture.md)：当前系统、目标模块、模块职责和扩展规则。
- [DuckDB 数据库结构](knowledge/database-schema.md)：数据库分层、核心表、约束、索引和备份边界。
- [时点数据与财报版本规范](knowledge/point-in-time-data.md)：观察日、披露、修订、行情截断和版本选择的统一语义。
- [数据回填与每日流水线运维手册](knowledge/operations-runbook.md)：历史回填、每日运行、故障处理、覆盖率和切换流程。

## 本轮范围

本轮处理数据日期一致性、观察日截断、完整历史财务数据、DuckDB 单一事实源、三浪三市值过滤和工程模块化。持仓、止盈止损及券商执行明确留到后续独立设计。
