# 前复权未来函数与右侧选股漏斗审计（2026-07-16）

## 结论

当前长周期回测不能继续当作严格时点收益目标使用。

核心原因不是止盈止损，而是历史 K 线的前复权锚点。代码虽然按 `date <= observation_date` 截断了行，但本地 CSV / DuckDB 里的历史 OHLC 可能已经被晚于观察日的除权因子重写。绝对价格相关逻辑会受影响，包括：

- 现价 / 基本价值线；
- 左侧网格价；
- 回测成交价；
- 条件止损价；
- 基于绝对价位的结构价。

比例和形态类指标，例如均线斜率、涨跌幅、RSI、KDJ、WR 背离，若只是整段统一缩放，影响相对小；但它们和价值线、成交执行混在一起后，回测仍不能称为严格无未来函数。

## 样本证据

审计脚本：

```powershell
python scripts/audit_qfq_lookahead.py `
  --symbols sz.002594 `
  --observation-dates 2024-09-24 2025-04-30 2026-01-05 2026-07-14 `
  --start-date 2024-01-01 `
  --end-date 2026-07-14 `
  --provider-json-directory var/cache/tushare_qfq_audit_json
```

结果：

| 股票 | 观察日 | 当日锚定前复权收盘 | 2026-07-14 锚定前复权收盘 | 差异 |
|---|---:|---:|---:|---:|
| 比亚迪 | 2024-09-24 | 254.29 | 83.764978 | -67.059272% |
| 比亚迪 | 2025-04-30 | 353.09 | 116.310417 | -67.059272% |
| 比亚迪 | 2026-01-05 | 98.11 | 98.11 | 0 |
| 比亚迪 | 2026-07-14 | 90.18 | 90.18 | 0 |

本地 AkShare CSV 的 2024-09-24 收盘是 `83.76`，几乎贴合 Tushare 的“2026-07-14 锚定前复权价”。这说明当前缓存是晚锚点价格，不是观察日当时可见的绝对价格。

Tushare `adj_factor` 当前账号被限频到 `1次/小时`，所以这次只完成了一个强证据样本。这个样本已经足够证明风险存在。

## 已修防线

### 1. CSV 元数据增加复权锚点

`save_kline_cache_metadata()` 现在写入：

- `adjustment = qfq`
- `qfq_anchor_date`
- `provider_actual_end_date`
- `cache_version`

读取缓存时，若 `qfq_anchor_date > end_date`，不走 CSV fast path。

旧 CSV 没有元数据时，只在“CSV 最大日期不晚于本次 end_date”时允许推断锚点；如果已有元数据明确是未来锚点，不允许重推断覆盖。

### 2. DuckDB 增加复权锚点列

`raw.stock_kline_daily` 新增：

- `adjustment`
- `qfq_anchor_date`
- `cache_version`

`KlineRepository.load_stock_kline()` 和 `load_stock_klines()` 支持 `max_qfq_anchor_date`。传入后，晚锚点和缺锚点行不会返回。

### 3. 回测主入口接入锚点过滤

`apps/portfolio_backtest.py::load_price_frames()` 从 DuckDB 批量读取时传入 `max_qfq_anchor_date=end_date`，先挡住“请求较早 end_date，却读到更晚锚点行”的情况。

### 4. 候选快照暴露复权状态

`historical_candidates.py` 会读取 CSV `.meta.json`，输出：

- `qfq_anchor_date`
- `historical_adjustment_check`

如果某行观察日早于复权锚点，会标记：

```text
前复权锚点晚于观察日：绝对价格含未来复权因子
```

### 5. AkShare 不复权价反推观察日锚定价

历史候选重建新增 `raw_kline_directory`。如果目录中存在 AkShare / 东方财富不复权日线，候选层会用下面的缩放关系反推观察日锚定价格：

```text
观察日重锚比例 = 观察日不复权收盘价 / 当前锚点前复权收盘价
观察日锚定前复权价 = 当前锚点前复权价 × 观察日重锚比例
```

对观察日当天来说，重锚后的收盘价等于当日不复权收盘价，因此可以用于 `现价 / 基本价值线`、左侧网格、结构价和候选层绝对价格判断。

新增脚本：

```powershell
python scripts/fetch_akshare_raw_kline.py `
  --start-date 2024-01-01 `
  --end-date 2026-07-14 `
  --output-directory var/cache/formula33_kline/akshare_raw `
  --allow-insecure
```

重建候选默认读取：

```text
var/cache/formula33_kline/akshare_raw
```

若不复权价缺失，候选行不会伪装成严格时点，而会标记：

```text
缺少AkShare不复权价，绝对价格仍含未来复权因子
```

## 仍未彻底解决

历史候选重建现在已经支持用 AkShare 不复权价修正候选层观察日价格；但组合回测成交撮合仍读取统一前复权 K 线。收益率本身多数情况下不受统一缩放影响，绝对成交价、止损价和价值线触发仍应继续推进到完整 as-of K 线视图。

要做到严格无未来函数，下一步必须重构成：

```text
每个观察日
→ 只使用观察日及以前的原始行情 / 当前前复权行情
→ 用 AkShare 不复权价反推观察日锚定比例
→ 生成当日可见的前复权视图
→ 再计算价值线比值、结构价、买点、成交价
```

也就是说，不能再用一份 `2024-09-24 ~ 2026-07-14` 的全区间前复权 CSV 来证明 2024 或 2025 的严格历史买卖。候选层已经开始改，成交层还需要继续收口。

## 右侧漏斗审计

白大/右侧视角和量化审计视角一致认为，下一步优先级不是继续调止盈止损，而是先清理右侧候选的时点和漏斗边界。

### 硬问题

1. 底层 `run_portfolio_backtest()` 默认仍是 `signals_effective_next_day=False`。生产入口已经显式传了次日生效，但底层默认有误用风险。
2. `order_type == close` 的右侧信号使用当日收盘确认，又按当日收盘成交，属于同 bar 执行污染。严格口径应改为次日开盘或条件单。
3. Top10 展示池和交易候选池没有完全分开。某些 Top10 外诊断行仍可交易，容易造成口径冲突。
4. 财务缓存仍标记 `financial_point_in_time=False`，不能声称严格公告时点回测。

### 模型效果问题

这些不是未来函数，但会影响能否选中右侧牛股：

- 强势成长候选被 Top10 核心名额挤出；
- 缺失成交额曾被默认成高流动性；
- 右侧分数偏“涨得强就排得高”，需要更专业地拆成动量、低波动、量价确认、回撤控制、相对强度、主线共振；
- 背离指标更适合风险提示，不适合作为收益增强主因。

## 下一步建议

1. 先做“严格 as-of 前复权视图”，再重跑 `2024-09-24 至今` 和 `2026 年初至今`。
2. 严格回测里把收盘确认买点改成次日执行；当前同日成交结果只能叫研究代理。
3. 拆分“展示 Top10”和“可交易右侧池”，不要让展示名额决定交易宇宙。
4. 再继续优化右侧漏斗：用横截面多因子模型，但必须先保证数据时点干净。

## 当前抓数状态

本机 Python 直连 AkShare 会触发 OpenSSL Applink 错误，因此脚本改为 Node 调东方财富 HTTP 端点。当前测试抓 `002594` 时端点返回 `socket hang up`，非沙箱网络下也一样。代码和解析路径已经就绪，仍需要后续网络恢复后补齐：

```powershell
python scripts/fetch_akshare_raw_kline.py `
  --codes 002594 `
  --start-date 2024-01-01 `
  --end-date 2026-07-14 `
  --output-directory var/cache/formula33_kline/akshare_raw `
  --force `
  --allow-insecure
```

## 已验证

```text
tests/integration/test_kline_repository.py
tests/unit/test_formula33_kline_persistence.py
tests/unit/test_historical_candidates_asof_prices.py
tests/unit/test_portfolio_backtest.py

375 passed, 1 skipped
```
