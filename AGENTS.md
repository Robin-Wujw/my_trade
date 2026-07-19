# Agent Workflow

本仓库涉及策略、选股、买卖、日报、回测或生产流水线时，默认采用多 agent 协作习惯。

执行策略相关任务前，先读：

1. `STRATEGY.md`
2. `docs/README.md`
3. `docs/knowledge/multi-agent-baida-protocol.md`
4. `docs/knowledge/local-environment-runbook.md`
5. `docs/knowledge/` 下与本次任务直接相关的文档

## 本机环境硬规则

- 当前默认 shell 是 Windows PowerShell，不要假设 bash、zsh、Linux shell 或新版 PowerShell 语法可用。
- 不要用 `&&` 串联命令；本机 PowerShell 版本会报 `The token '&&' is not a valid statement separator`。需要分步执行，或使用 PowerShell 原生命令块。
- `rg.exe` 来自 Codex WindowsApps 目录，可能出现 `Access is denied`。遇到后不要反复重试 `rg`，改用 `git grep` 搜 tracked 文件，或用 `Select-String` 限定目录搜索。
- 避免对仓库根目录做无边界递归搜索；`var/pytest-tmp-*`、`var/` 历史产物和权限文件容易造成误报或访问失败。
- 网络数据源经常有 AkShare SSL、限流、空表和字段变化问题。不要删除有效缓存来“修复”网络问题；先检查代理、证书、重试、缓存命中和数据源状态。
- 涉及 AkShare 抓取失败时，优先采用已有缓存、重试退避、单股补抓或备用数据源；不得把网络失败记成停牌、无交易或无财务数据。
- MiniQMT 当前只读，不能打开自动实盘下单。任何买卖逻辑修改都只能进入回测、提醒或计划，不得连接券商下单 API。

详细处理见 `docs/knowledge/local-environment-runbook.md`。

## 必需角色

- 主 agent：负责用户请求、主流程运行、代码修改、命令执行、验证和最终交付。除非用户明确要求诊断单步运行，否则保持七步生产流程不被绕过。
- 白大 agent：负责研究与思想建议。任务依赖当前市场、业务场景、技术细节、买卖逻辑或选股逻辑时，优先调用 IMA 和网页查询；如果当前环境没有 IMA 工具，必须明确说明，并用网页原文、仓库文档和可复现数据替代。
- 量化 agent：负责纠错与一致性审查。检查代码、规则、模型、回测和输出是否冲突，是否偏离白大思想、时点数据规则和项目生产不变量。

主 agent 必须整合白大 agent 和量化 agent 的建议，但最终代码与生产决策仍以 `STRATEGY.md`、时点数据、测试和当前仓库不变量为准。

如果当前环境无法实际创建子 agent，主 agent 也必须按这三个角色视角自检，并说明哪些部分由主 agent 代做。
