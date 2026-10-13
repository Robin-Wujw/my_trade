# Windows 研究工作台

项目提供一个本地 Streamlit 界面，用来查看回测摘要、净值与回撤、交易流水和数据资格。它借鉴 QuantsPlaybook 的研究结果组织方式和 OSkhQuant 的参数/日志体验，但没有复制两个外部仓库的代码；OSkhQuant 的 CC BY-NC 许可证也不适合直接作为生产依赖。

## 安装

在仓库根目录打开 PowerShell：

```powershell
.\scripts\install_dashboard.ps1
```

安装器会在项目根目录创建并复用 `.venv-dashboard`，所有 dashboard 依赖都安装到该环境，不会修改系统 Python 或用户级 site-packages。安装器默认创建桌面快捷方式；不需要快捷方式时：

```powershell
.\scripts\install_dashboard.ps1 -CreateShortcut:$false
```

也可以双击 `scripts\install_dashboard.bat`，或在命令提示符中运行它。安装器要求 Python 3.10 或更新版本；可通过 `-PythonBin` 或 `PYTHON_BIN` 指定解释器。

如果有多个 Python，可以显式指定创建虚拟环境所用的解释器：

```powershell
.\scripts\install_dashboard.ps1 -PythonBin 'D:\ActionsRunner\my-trade\python\python.exe'
```

脚本安装的是 `.[dashboard]` 可选依赖，不会让无界面 CLI 强制安装 Streamlit 和 Plotly。重复运行会升级虚拟环境中的 pip 并同步项目依赖。

## 启动

```powershell
.\scripts\start_dashboard.ps1
```

也可以双击或在命令提示符中使用：

```text
scripts\start_dashboard.bat
```

启动脚本只使用 `.venv-dashboard` 中的 Python，并在启动前检查 Python 版本和端口。默认自动打开 `http://127.0.0.1:8501`；服务器仍只监听本机地址，不提供实盘或下单入口。需要手动打开浏览器时：

```powershell
.\scripts\start_dashboard.ps1 -NoBrowser -Port 8502
```

启动脚本和 BAT 会把 Streamlit 的退出码返回给调用方，便于任务计划或其他启动器发现失败。

## 页面说明

- **回测产物**：扫描 `var/backtests/**/*_summary.json` 的最近 50 个文件，并关联同目录的 `*_equity.csv`、`*_trades.csv` 和 `*_events.csv`；大摘要包含完整交易账本，因此界面限制扫描数量以保持 Windows 启动速度。
- **净值与回撤**：Plotly 图使用摘要旁的净值 CSV，不重新计算交易账本。
- **交易流水**：可以按代码、动作和原因筛选，便于复盘买卖、成本和执行时序。
- **资格标签**：`研究回测`、`数据受限`、`严格时点/非严格时点` 和 `next_open/close_proxy` 会直接显示。
- **启动研究回测**：调用现有 `python -m apps.portfolio_backtest`，默认使用 `next_open` 和当前成本参数。勾选“允许使用已冻结本地输入”时会显式传递 `--no-refresh-inputs --allow-frozen-inputs`，结果仍按研究口径展示。

界面没有实盘账户、下单、撤单或券商连接入口。MiniQMT 仍只读；任何信号都只能进入回测、提醒或计划。

## 数据边界

如果摘要标记 `research_only`、`data_limited` 或 `strict_point_in_time=false`，它只能用于研究和敏感性比较，不能称为严格无未来函数的生产绩效。网络数据失败时应先检查缓存、重试和数据源状态，不能在界面中把失败改写成停牌或无数据。
