# my_trade

沪深A股收盘后全市场动态选股系统。生产链只使用当时可见的财报和行情，数据源顺序固定为：

```text
AkShare → Baostock兜底 → 本地缓存
```

策略规则见 [STRATEGY.md](STRATEGY.md)。

## 每日流程

```powershell
.\run_daily_analysis.ps1
```

```bash
bash run_daily_analysis.sh
```

流水线依次执行：

1. `formula33Stats.py`：更新全市场行情缓存和三浪三市场结构。
2. `sectorStats.py`、`sectorWatch.py`：更新板块趋势、量能和主流板块。
3. `factorStock.py`：更新全市场技术因子。
4. `fullMarketFundamentalUpdate.py`：增量补齐财务缓存并生成全市场动态截面。
5. `dailyFundamentalSelect.py`：动态生成基本价值线池和正常基本面池。
6. `dailyReportPush.py`：生成CSV、HTML并按配置推送PushPlus。

缓存全部保存在 `.cache`，每日仅做增量更新。运行失败会写入 `logs` 并由 `pipelineAlert.py` 尝试告警。

## GitHub Actions

工作流：[stock-selection.yml](.github/workflows/stock-selection.yml)

- 每天北京时间16:30唤醒，由门控保证每连续3天实际选股一次。
- 在GitHub仓库 `Actions → Stock Selection → Run workflow` 可手动执行当天选股。
- 首次云端运行自动冷启动缓存，后续通过Actions Cache增量更新。
- 报告、CSV、覆盖率和日志以Artifact保留14天。
- 如需推送，在仓库Actions Secrets中添加 `PUSHPLUS_TOKEN`；未配置时自动跳过推送。

工作流提交并推送到GitHub默认分支后生效。

## 本地定时任务

```powershell
.\install_daily_task.ps1 -Time "16:30"
```

## 当前运行文件

- `dailyFundamentalSelect.py`
- `dailyReportPush.py`
- `factorStock.py`
- `formula33Stats.py`
- `fullMarketFundamentalUpdate.py`
- `pipelineAlert.py`
- `point_in_time.py`
- `sectorStats.py`
- `sectorWatch.py`
- `trade_utils.py`
- `wave_utils.py`

## 验证

```powershell
python -m py_compile dailyFundamentalSelect.py dailyReportPush.py factorStock.py `
  formula33Stats.py fullMarketFundamentalUpdate.py pipelineAlert.py point_in_time.py `
  sectorStats.py sectorWatch.py trade_utils.py wave_utils.py
```
