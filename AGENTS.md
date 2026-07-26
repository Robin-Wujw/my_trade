# Agent Workflow

本仓库涉及策略、选股、买卖、日报、回测或生产流水线时，默认采用直接执行式工作流。

执行策略相关任务前，先读：

1. `STRATEGY.md`
2. `docs/README.md`
3. `docs/knowledge/local-environment-runbook.md`
4. `docs/knowledge/` 下与本次任务直接相关的文档

## 本机环境硬规则

- 当前默认 shell 是 Windows PowerShell，不要假设 bash、zsh、Linux shell 或新版 PowerShell 语法可用。
- 不要用 `&&` 串联命令；本机 PowerShell 版本会报 `The token '&&' is not a valid statement separator`。需要分步执行，或使用 PowerShell 原生命令块。
- `rg.exe` 来自 Codex WindowsApps 目录，可能出现 `Access is denied`。遇到后不要反复重试 `rg`，改用 `git grep` 搜 tracked 文件，或用 `Select-String` 限定目录搜索。
- 避免对仓库根目录做无边界递归搜索；`var/pytest-tmp-*`、`var/` 历史产物和权限文件容易造成误报或访问失败。
- 网络数据源经常有 AkShare SSL、限流、空表和字段变化问题。不要删除有效缓存来“修复”网络问题；先检查代理、证书、重试、缓存命中和数据源状态。
- 涉及 AkShare 抓取失败时，优先采用已有缓存、重试退避、单股补抓或备用数据源；不得把网络失败记成停牌、无交易或无财务数据。
- MiniQMT 当前只读，不能打开自动实盘下单。任何买卖逻辑修改都只能进入回测、提醒或计划，不得连接券商下单 API。

详细处理见 `docs/knowledge/local-environment-runbook.md`。

## 策略任务要求

- 以 `STRATEGY.md`、当前代码、时点数据规则和测试结果为准。
- 涉及买卖、选股、回测和生产流水线时，必须检查未来函数、复权口径、财务公告时点、行业/板块快照时点、成交成本、T+1、整手和停牌/涨跌停处理。
- 当前 MiniQMT 只读；任何买卖逻辑只能进入回测、提醒或计划，不得连接券商下单 API。
- 如果数据源或缓存不满足严格时点规则，必须明确标记为研究回测，不得合并或宣称为严格无未来函数口径。
