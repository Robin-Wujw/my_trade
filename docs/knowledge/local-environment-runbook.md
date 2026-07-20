# 本机环境与常见坑处理手册

本文记录当前 `D:\MyCodes\my_trade` 工作区的本机运行条件和反复踩坑点。策略、回测、数据抓取、文档整理和提交前都应先看这里。

## 1. 终端与命令习惯

当前默认终端是 Windows PowerShell。

容易犯错：

- 把环境当成 bash，使用 `cmd && cmd` 串联命令。
- 使用 Linux 路径、`grep`、`sed -n`、`cat <<EOF` 等 shell 习惯。
- 用一条长命令混合多个 shell 的语法。

正确做法：

- 多个命令分开执行。
- 需要条件执行时，用 PowerShell 原生写法。
- 文件编辑用 `apply_patch`，不要用 shell 重定向拼文件。
- Windows 文件操作优先用 `Get-ChildItem`、`Select-String`、`Remove-Item -LiteralPath`、`Move-Item -LiteralPath`。

示例：

```powershell
git status --short --branch
git add -A
git commit -m "message"
```

不要写：

```powershell
git add -A && git commit -m "message"
```

## 2. 搜索工具

首选仍然是 `rg`，但本机 `rg.exe` 位于 Codex WindowsApps 目录，已多次出现：

```text
Program 'rg.exe' failed to run: Access is denied
```

遇到后不要继续反复调用 `rg`。替代方案：

```powershell
git grep -n "keyword"
```

只搜 tracked 文件时，优先用 `git grep`。需要搜指定目录时：

```powershell
Get-ChildItem -Path docs,apps,stock_research,scripts,tests -Recurse -File |
  Select-String -Pattern "keyword"
```

不要对仓库根目录做无限制递归搜索。`var/` 里有历史回测产物、pytest 临时目录、权限异常目录，容易导致访问失败和噪声。

## 3. Python 解释器

生产运维文档里推荐的生产解释器是：

```powershell
$Python = 'D:\ActionsRunner\my-trade\python\python.exe'
```

日常开发测试可以直接用当前环境的：

```powershell
python -m pytest -q
```

如果出现包版本、SSL、路径或依赖差异，先确认使用的是哪个 Python：

```powershell
python -c "import sys; print(sys.executable)"
```

## 4. AkShare 与网络数据源

AkShare 常见问题：

- SSL 握手失败或证书验证失败
- 接口限流
- 返回空表
- 字段名变化
- 某只股票或某个日期单点失败
- 东方财富、新浪、Tushare、BaoStock 之间价格或成交额口径不同

处理顺序：

1. 先看是否已有有效缓存，避免重复打接口。
2. 检查代理和网络，不要把连接错误当作停牌或无数据。
3. 使用已有重试、退避、单股补抓参数。
4. 只对失败股票或缺失日期补抓，不要为了一个网络错删全量缓存。
5. 数据源差异必须记录来源和口径，不要把两个提供方的前复权历史拼成一条序列。
6. 如果需要备用源，必须在输出里标记 `price_source`、`amount_source` 或诊断原因。

不能做：

- 因 SSL 或限流失败清空有效缓存。
- 把请求失败登记为停牌。
- 把请求失败登记为无财务数据。
- 用当前数据源结果倒灌历史候选。

## 5. MiniQMT 边界

当前 MiniQMT 只允许：

- 读取行情
- 读取只读账户诊断
- 构建本地行情和财务缓存
- 作为回测执行画像

当前 MiniQMT 不允许：

- 自动实盘下单
- 自动撤单
- 把回测信号直接发送到券商
- 绕过 `LiveTradingDisabled`

真实盘中条件单和日 K 回测必须分开说：

- 事先已知的突破价、止损价，可以讨论 MiniQMT 条件单。
- 收盘后才确认的信号，只能收盘附近或次日执行。
- 日 K 回测没有分钟数据时，不能声称真实 14:55 成交质量。

## 6. 回测与产物

`var/` 已在 `.gitignore` 中，但历史上有一批回测产物已经被 tracked。整理策略代码时：

- 不要随手删除 `var/`，除非任务明确是清理历史产物。
- 不要把新回测大产物加入 Git。
- 需要引用回测结果时，写清楚路径、区间、候选池、执行画像和是否严格时点。
- 候选目录存在 `manifest.json` 时，回测只允许读取 manifest 明确列出的快照文件；目录里残留但未登记的 `candidates_*.csv` 不得参与回测。
- 候选 manifest 还必须匹配当前快照版本和价值行业规则版本。行业来源需记录路径、日期、SHA-256、覆盖率和 PIT 状态；`industry_point_in_time=false` 默认拒绝，仅研究回测可显式传 `--allow-unsafe-industry`。该开关与 `--allow-unsafe-financial` 独立，不能相互代替。
- MiniQMT 历史窗口补抓会与已有单股 CSV 合并，不能用较短刷新窗口覆盖较长历史；全市场刷新中断后可按实际候选代码做补抓和覆盖审计。
- 大回测前先确认没有另一个 Python 回测进程在跑。

检查进程：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -like 'python*' -or $_.Name -eq 'cmd.exe' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

## 7. 文档编码与中文化

文档应使用中文说明，必要英文术语只保留代码名、字段名、命令名和论文名。中文文档里避免用英文段落解释业务规则。

终端中看到中文乱码时，不要立即判断文件损坏。先确认：

- 文件是否为 UTF-8
- PowerShell 输出编码是否影响显示
- Git diff 中是否仍显示正常中文

如需重写文档，优先用 `apply_patch`，并保持文件内容可读、可审、可被后续 agent 直接使用。

## 8. 提交前检查

常规检查：

```powershell
git status --short --branch
git diff --check
python -m pytest -q
```

如果只改文档，也至少运行：

```powershell
git diff --check
```

策略、候选、买卖、回测相关代码改动后，必须跑相关单测或全量单测，并在最终答复里说明结果。
