# 项目目录与模块边界重构设计

- 状态：已由用户确认
- 日期：2026-07-04
- 目标：让仓库一级目录、生产入口、业务模块、第三方接口和运行产物各自只有一个清晰归属
- 验收原则：不以“文件已移动”为完成标准，只以编译、测试、回归、入口和路径检查全部通过为完成标准
- 非目标：本轮不修改三浪三、因子选股、基本面筛选、板块判断和日报展示的业务口径

## 1. 已确认的架构选择

采用“根目录核心包 + 独立应用入口”，不使用 `src/` 布局，也不把多个可导入包直接平铺到仓库根目录。核心包固定命名为 `stock_research`，生产编排、GitHub Actions 和测试同步迁移，旧根目录脚本名不保留兼容壳。

目标一级目录如下：

```text
my_trade/
├─ .github/              # GitHub Actions
├─ apps/                 # 可执行入口，只解析参数和调用流水线
├─ stock_research/       # 唯一核心 Python 包
├─ config/               # 可提交的配置与样例
├─ scripts/              # PowerShell 运行与运维脚本
├─ tests/                # 单元、集成、回归与受控夹具
├─ docs/                 # 当前架构、运维、设计与决策文档
├─ var/                  # 唯一运行数据根目录，整体忽略提交
├─ pyproject.toml
├─ README.md
└─ STRATEGY.md
```

根目录不再保存业务 Python 脚本、运行日志、选股导出、板块导出、缓存、数据库、临时测试库或本地凭证。

## 2. 核心包职责

```text
stock_research/
├─ api/          # AkShare、Baostock、PushPlus 等外部接口与字段映射
├─ core/         # 配置、路径、运行上下文、时点规则和错误类型
├─ storage/      # DuckDB、迁移、仓储、缓存读写与运行记录
├─ market/       # 股票池、行情、板块、股本、市值和基本面访问服务
├─ indicators/   # 三浪三、波段、均线、量能和因子等纯计算
├─ strategies/   # 三浪三资格、因子、基本面和板块主线规则
├─ pipelines/    # 每日流程、单步流程、门控和步骤状态
├─ reporting/    # 报表组装、导出、PushPlus 内容和告警
└─ regression/   # 历史输出语义哈希与基线验证
```

依赖方向固定为：

```text
apps
  → pipelines
      → market / indicators / strategies / reporting
          → api / storage / core
```

约束如下：

1. `apps` 不包含业务计算，只做参数解析和进程退出码转换。
2. `pipelines` 负责运行顺序、步骤状态和门控，不实现指标公式。
3. `api` 只处理外部通信、重试、限速和源字段映射，不执行策略评分。
4. `indicators` 必须可离线测试，不联网、不读取全局路径、不写文件。
5. `strategies` 消费明确输入并返回结果，不寻找“最新文件”。
6. `reporting` 只格式化同一运行的结果，不重新选股。
7. `core` 不反向依赖任何业务模块。
8. 任一生产模块都不得导入 `apps`。

## 3. 应用入口

`apps/` 保留一个正式入口和必要的单步调试入口：

```text
apps/
├─ daily_pipeline.py
├─ formula33.py
├─ sector_analysis.py
├─ factor_selection.py
├─ fundamental_update.py
├─ fundamental_selection.py
├─ daily_report.py
└─ pipeline_alert.py
```

`scripts/run_daily_analysis.ps1` 只调用 `apps.daily_pipeline`。单步入口供故障定位、历史回放和人工补跑使用，不被 PowerShell 再次串成第二套生产编排。GitHub Actions 调用 `scripts/run_daily_analysis.ps1`，上传路径统一改为 `var/logs/` 和 `var/exports/`。

## 4. 现有代码迁移映射

| 现有文件 | 目标职责 |
|---|---|
| `formula33Stats.py` | `api` 的行情/元数据访问，`market` 的股票池与市值服务，`indicators.formula33` 的纯计算，`strategies.formula33` 的观察日资格，`pipelines.formula33` 的步骤协调，`reporting.exports` 的导出 |
| `sectorStats.py` | `api` 的板块数据访问，`market.sectors` 的标准化，`indicators.sector_metrics` 的计算，`pipelines.sector_analysis` 的协调 |
| `sectorWatch.py` | `strategies.sector_watch` 的主线规则和 `pipelines.sector_analysis` 的观察日运行 |
| `factorStock.py` | `api` 的财务/行情访问，`market.fundamentals`，`indicators.factors`，`strategies.factor_selection`，`pipelines.factor_selection` 和 `reporting.exports` |
| `fullMarketFundamentalUpdate.py` | `pipelines.fundamental_update` 与相应市场、存储服务 |
| `dailyFundamentalSelect.py` | `strategies.fundamental_selection` 与 `pipelines.fundamental_selection` |
| `dailyReportPush.py` | `reporting.daily_report`、`api.pushplus` 与 `pipelines.daily_report` |
| `pipelineAlert.py` | `reporting.alerts` 与 `apps.pipeline_alert` |
| `point_in_time.py` | `core.as_of` |
| `trade_utils.py` | 拆入 `core.paths`、`api.pushplus`、`reporting.diff` 和通用重试工具；不保留万能工具文件 |
| `wave_utils.py` | `indicators.waves` |
| `src/my_trade/domain` | 迁入 `stock_research/core` |
| `src/my_trade/storage` | 迁入 `stock_research/storage` |
| `src/my_trade/regression` | 迁入 `stock_research/regression` |

拆分按行为边界进行，不创建仅为缩短文件而存在的转发模块。共享能力只有一个实现位置，禁止新旧实现并存。

## 5. 运行数据目录

所有非源码运行数据统一进入被忽略的 `var/`：

```text
var/
├─ cache/       # 原 .cache，生产增量输入
├─ data/        # my_trade.duckdb
├─ exports/
│  ├─ selection/
│  ├─ market/
│  └─ reports/
├─ logs/
├─ state/       # 上次结果和断点状态
├─ secrets/     # 本地 token；Actions 仍优先使用环境变量
└─ tmp/         # 测试临时目录，可安全清空
```

路径由 `stock_research.core.paths.ProjectPaths` 统一产生。可通过一个明确环境变量覆盖 `var` 根目录，其他模块不得自行拼接仓库路径。生产缓存迁移后必须核对文件数量和总字节数；数量或字节数不一致时停止删除旧位置。

## 6. 保留、夹具化与删除

### 6.1 迁移保留

- `.cache` 中 113.37 MiB 生产缓存迁入 `var/cache`；
- `.data/my_trade.duckdb` 迁入 `var/data`；
- 最新有效选股、板块和日报产物迁入 `var/exports`；
- 当前日志迁入 `var/logs`；
- `.factorStock_last.json` 迁入 `var/state` 并采用蛇形命名；
- `.pushplus_token` 迁入 `var/secrets/pushplus_token`；
- 当前工作区未提交的三浪三状态、日报、数据库测试、Actions 和文档改动融合进对应新模块，不被覆盖或丢弃。

### 6.2 转为受控测试夹具

`tests/regression/legacy-output-v1.json` 引用的六份历史 CSV 迁入 `tests/fixtures/regression/`。清单根目录随之更新，文件哈希和语义哈希不得改变。完成后回归验证不再依赖 `板块观察` 或 `选股结果` 运行目录。

### 6.3 明确删除

- `__pycache__`、`.pytest_cache` 和 `.test-tmp`；
- 回归夹具与最新有效产物以外的重复旧导出；
- 空的 `.agents` 和非项目运行所需的 `.claude` 本地助手配置；
- 迁移完成后的 11 个根目录 Python 文件；
- 迁移完成后的 `src/my_trade`；
- 根目录旧 PowerShell 入口；
- 已被当前架构文档完整取代、会造成事实冲突的重复说明。

删除只依据显式路径清单。每个候选绝对路径必须先验证位于仓库根目录内；不使用覆盖整个工作区的递归通配删除。任一回归测试、导入检查或数据数量校验失败时都停止删除。

## 7. 配置与命名

Python 文件、模块和状态文件统一使用小写蛇形命名。业务默认参数进入 `config/pipeline.toml`，敏感值不进入配置文件。环境变量可覆盖配置；命令行参数优先级最高。配置加载由 `core.config` 负责并在启动时一次性验证，业务模块不直接读取环境变量。

`requirements-actions.txt` 迁入 `config/requirements/actions.txt`，`pyproject.toml` 继续定义项目元数据、可安装包和测试设置。README 只描述当前有效入口和目录，不保留旧命令。

## 8. 错误处理与运行门控

1. 外部接口错误由 `api` 转换为有来源、重试次数和类别的明确异常。
2. 单股抓取失败进入诊断结果；关键市场步骤、覆盖率或日期门控失败则阻止日报推送。
3. 每个流水线步骤返回结构化状态，正式入口按步骤状态决定继续、跳过或失败。
4. 报告只消费同一运行上下文的产物；观察日、运行标识或必需结果不一致时失败关闭。
5. `apps` 将已知业务错误转换为稳定退出码并写入日志，未知错误保留堆栈并触发告警。
6. 结构验收不依赖网络，避免把数据源波动误判为代码迁移失败。

## 9. 测试与架构约束

当前测试基线在项目专用临时目录下为 `27 passed`。重构采用以下验证层级：

1. **单元测试**：纯指标、策略规则、路径、配置、时点和报告格式。
2. **集成测试**：DuckDB 迁移、运行仓储、流水线门控、缓存/导出路径和入口退出码。
3. **回归测试**：六份历史输出文件哈希与语义哈希保持一致；已确认的三浪三状态语义继续通过。
4. **架构测试**：根目录无业务 Python 文件；不存在旧模块导入；`api` 不依赖策略；生产模块不依赖 `apps`；运行路径只由 `ProjectPaths` 提供。
5. **入口冒烟**：全部 `apps` 模块的 `--help` 可执行；PowerShell 和 Actions 引用的文件、模块与上传目录全部存在。
6. **静态验证**：全部 Python 文件可编译，Git diff 无空白错误，文档搜索不到失效生产命令。

Windows 权限环境下，pytest 固定使用仓库内 `var/tmp/pytest` 作为 `--basetemp`，并避免依赖不可写的系统临时目录。

## 10. 实施顺序与停止条件

1. 固定架构测试、路径测试和当前行为回归。
2. 建立 `stock_research`、`apps`、`config` 和新的测试夹具边界。
3. 先迁移无副作用的领域、存储、时点和指标模块。
4. 再按公式三、板块、因子、基本面、日报顺序迁移外部接口、策略和流水线。
5. 更新 PowerShell、Actions、配置和文档，只引用新入口。
6. 在旧路径仍存在时完成编译、测试、回归和入口冒烟。
7. 生成并校验显式数据迁移与删除清单。
8. 迁移运行数据，核对文件数量与字节数后删除旧路径。
9. 从干净进程重新运行全部验收。

出现下列任一情况立即停止最终删除：

- 测试或编译失败；
- 六份回归基线发生无法解释的语义差异；
- 生产缓存迁移前后数量或字节数不一致；
- PowerShell、Actions 或任一应用入口仍引用旧路径；
- 当前工作区已有改动无法明确映射到新模块。

## 11. 完成标准

- 根目录符合第 1 节结构，没有业务 Python 散文件和生成物目录；
- 核心包唯一命名为 `stock_research`，不存在 `my_trade` Python 包或旧模块导入；
- PowerShell、Actions、README 和运维文档只使用新入口与新路径；
- 生产缓存、数据库、最新产物、本地状态和凭证均迁移到 `var` 对应目录；
- 六份回归基线成为受控测试夹具；
- 所有编译、单元、集成、回归、架构、CLI 冒烟和路径检查全部通过；
- 删除动作未覆盖或丢失用户当前未提交成果；
- 最终验收报告列出执行命令、退出码、测试数量、回归数量、迁移文件数和删除清单摘要。
