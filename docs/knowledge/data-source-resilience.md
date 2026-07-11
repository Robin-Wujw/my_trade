# 数据源连接与容错改进

## 范围与参考基线

本文只讨论 Tushare、AkShare、BaoStock 等外部数据源的连接、限流、异常结果、缓存和恢复，不评价或引入参考项目的选股策略。

- [tianjingle/zMain](https://github.com/tianjingle/zMain)，检查提交 `327b5c5ab1ce075f419471a002ebdb393594b148`
- [sngyai/Sequoia-X](https://github.com/sngyai/Sequoia-X)，检查提交 `444c0db69ff36b46ef2b22ab265051d60c16029d`

## 参考项目对照

### zMain

zMain 会检查本地最后交易日，只补缺失区间；盘中数据暂存、收盘后再入正式库，并在 BaoStock 历史行情与腾讯实时行情之间按时间场景切换。核心价值是减少重复请求，并避免把盘中未定稿数据混入历史数据。

其 BaoStock 查询会按股票重新登录，未校验登录错误码，也没有对查询失败、空结果和限流做系统性重试，因此本项目不照搬连接方式。

### Sequoia-X

Sequoia-X 的回填流程包含：

1. 从本地最后日期增量续传，已完成股票直接跳过。
2. 单只股票失败重试三次，等待时间按 2、4、8 秒增加。
3. 查询失败后 logout/login 重建 BaoStock 会话。
4. 每处理 200 只股票主动重连，降低长连接失效概率。
5. 多进程 worker 各自登录，避免跨进程共享 SDK 会话。
6. 单只失败计数并继续，任务可再次运行续传。

不足之处是日常批量同步 worker 没有同等级重试，非零错误码会被直接跳过，空结果统一视作非交易日，并行数固定最高 8 且缺少显式节流。其写入前仅做基础数值过滤，完整性验证弱于本项目。

## 本项目原有能力

- DuckDB 与 CSV 双层缓存，按缺口增量请求，可中断续跑。
- 前复权变化时整窗刷新并事务替换，防止新旧复权数据混合。
- 用交易日、最少历史行数、重叠日期检查空表和部分返回，异常数据不会覆盖好缓存。
- AkShare 新浪日线失败后降级到东方财富；腾讯端点用于特殊 CDR。
- 多进程 BaoStock worker 独立登录，并用 `maxtasksperchild` 定期回收进程。
- 修改代理环境时加线程锁，调用结束恢复原环境。

## 本次已实施改进

### Tushare 标准化补充源

Formula33 的 `--price-source` 支持 `tushare/akshare/baostock`，默认仍为 AkShare。Tushare 保留为显式的小批量或高权限账号选项。启动时只检查 token，不做能力探测，以免探测本身消耗 `adj_factor` 的低频额度。

Tushare 返回的不复权 OHLC 与复权因子在本地按“价格 × 当日因子 ÷ 请求截止日因子”计算前复权，保持当前 Formula33 的 QFQ 口径。增量请求造成锚点变化时，既有重叠检测会触发整窗刷新。

本项目直接使用 Tushare 官方标准 HTTP 协议，不依赖额外 SDK。token 只从 `TUSHARE_TOKEN`、`TUSHARE_TOKEN_FILE` 或被 Git 忽略的 `var/secrets/tushare_token` 读取，不写入日志、缓存元数据和提交。

官方权限表显示 120 积分档为每分钟 50 次、每天 8000 次且只开放不复权日线；复权行情还需要复权因子权限。接口自身还可能有更低限制：当前账号实测 `adj_factor` 返回 40203，提示 1 次/分钟，个别请求提示 1 次/小时。因此不能把账号总频次直接套到每个接口。一次股票前复权窗口需要 `daily` 和 `adj_factor` 两次调用，当前逐股架构不适合用此账号做全市场 Tushare K 线。

Tushare 仍优先用于 `daily_basic` 总市值：一次请求可返回全市场，本账号实测返回 5599 行，标准化程度和调用效率都优于逐股抓取。前复权 K 线使用更快的 AkShare，并继续由本地缓存和完整性规则保护。

相关官方说明：

- [积分与频次权限对应表](https://tushare.pro/document/1?doc_id=290)
- [A 股复权行情与计算口径](https://tushare.pro/document/2?doc_id=146)
- [HTTP API 标准返回结构](https://www.tushare.pro/document/2?doc_id=130)

### 统一退避

三个 pipeline 的重复线性退避已统一到 `stock_research.api.retry.call_with_backoff`：

- 指数增长并封顶，避免故障期间持续高频请求。
- 使用 full jitter，让多个 worker 的重试时间错开。
- HTTP 异常带 `Retry-After` 时优先服从服务端等待时间。
- 支持 `retry_if`，可对参数错误等永久失败立即停止。
- 支持 `on_retry`，可在下一次尝试前恢复连接。

兼容入口 `call_with_retry` 保留，现有 THS 调用无需迁移。

### SDK 边界节流

AkShare 和 BaoStock adapter 增加线程安全、进程内最小请求间隔：

- `TUSHARE_MIN_INTERVAL`，默认 `1.25` 秒，跨进程共享。
- `AKSHARE_MIN_INTERVAL`，默认 `0.05` 秒。
- `BAOSTOCK_MIN_INTERVAL`，默认 `0.02` 秒，仅限制 `query_*` 方法。

可通过环境变量调大。AkShare/BaoStock 是进程内限流，总吞吐近似“单进程速率 × worker 数”；Tushare 使用跨进程锁，worker 增加不会突破账号总频率。

### BaoStock 错误码与会话恢复

BaoStock 常把失败放在返回对象的 `error_code` 中而不抛异常。新增 `ensure_success` 将非零错误码转成异常，使统一重试真正生效。

K 线查询失败后会 logout/login，再进行下一次重试；重新登录也校验错误码。多进程 worker 初始化登录不再静默忽略失败。

### 异常空表降级

AkShare 新浪日线即使没有抛异常，也可能返回空 DataFrame。现在空表会触发东方财富备用端点。备用端点最终仍为空时，不在 adapter 层武断认定为限流，而交给交易日、停牌状态、缓存重叠和历史行数规则判断。

空表既可能是限流或上游格式变化，也可能是合法停牌、未上市或非交易区间，不能统一自动重试。

## 运维建议

遇到大量 429、连接重置或连续空结果时：

1. 将公式任务 `workers` 降到 1～2。
2. 将 `AKSHARE_MIN_INTERVAL` 调到 `0.2`～`1.0`，BaoStock 同理按需调整。
3. 保留已有缓存，不要删除 DuckDB/CSV 后立即全市场重拉。
4. 重新执行同一日期任务，增量逻辑会跳过已经完整的股票。
5. 根据“新浪失败，回退东方财富”和缺失交易日日志，区分端点故障与数据本身缺失。

PowerShell 示例：

```powershell
$env:AKSHARE_MIN_INTERVAL = "0.3"
$env:BAOSTOCK_MIN_INTERVAL = "0.1"
$env:TUSHARE_TOKEN_FILE = "D:\secure\tushare_token"
python -m apps.formula33 --price-source akshare --workers 2 --retries 5 --retry-delay 2
```

## 暂未实施

- AkShare/BaoStock 暂未做跨进程全局令牌桶；Tushare 因官方账号级 50 次/分钟限制，已经使用跨进程文件限流器。
- 未对所有空表自动重试：没有交易、停牌、未上市也会产生合法空表。
- 未引入长期熔断器：数据源存在多个端点和日期差异，进程级长时间熔断可能误伤恢复后的接口。
- 未照搬固定“每 200 只重连”：本项目 `maxtasksperchild` 默认 200，会回收整个 BaoStock worker，连接清理更彻底。

## 验证覆盖

新增测试覆盖永久错误立即停止、重试前恢复钩子、限流器等待时间、BaoStock 非零错误码、重连登录校验和查询节流。既有 K 线测试覆盖缓存增量、复权刷新、空/部分结果拒绝、停牌标记和数据源降级。
