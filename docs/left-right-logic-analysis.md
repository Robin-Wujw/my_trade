# 左侧和右侧买入逻辑分析

## 当前架构（已经是分离的！）

### 左侧网格买入（portfolio_backtest.py:1533-1574）

**特点：限价单网格买入，不需要技术信号**

```python
for planned in plan_by_slot.values():
    if not can_add_left:  # 只检查候选资格
        continue
    
    # 在网格点位放置限价单
    fill = fill_limit_order(
        row, side="buy", 
        limit_price=float(planned["buy_price"]),  # 网格价格
        ...
    )
    
    if fill["filled"]:
        execute_buy(...)  # 直接买入，无需技术信号
```

**网格结构（left_grid_plan）：**
- 槽位0-4：买入价 = 价值线
- 槽位5-9：买入价 = 价值线 × (1-5%)^n（逐级向下）

**买入条件：**
1. ✅ `left_candidate_can_add(candidate)` = True
2. ✅ 价格触及网格点位（限价单成交）
3. ✅ 持仓限制（左侧最多1只）

---

### 右侧技术买入（portfolio_backtest.py:1803-1824）

**特点：需要主结构信号 + 技术证据分≥8**

```python
signal = left_right_switch_signals.get(code) or _right_signal(
    data, index, plans.get(code),
    auto_price_structure=auto_price_structure,
    ...
)

if not signal:
    continue  # 没有信号，不买入

if float(signal.get("entry_evidence_score") or 0.0) < configured_min_entry_evidence_score:
    continue  # 技术证据分不够，不买入
```

**买入条件：**
1. ✅ `right_candidate_can_evaluate(candidate)` = True
2. ✅ 主结构信号（W底、平台突破、U50收复等）
3. ✅ 技术证据分 ≥ 8

---

## 中际旭创为什么没买入？

### 候选池情况（2024-09-24）

| 项目 | 值 |
|------|---|
| 排名 | 2/10 |
| 候选分 | 129.8 |
| 来源 | **value_model** |
| 股价 | 121.40元 |
| 价值线 | 124.05元 |
| 价格/价值线 | 0.98 |
| allow_left | **应该是True** |
| allow_right | **可能是False**（纯value_model来源）|

### 问题分析

**左侧网格应该能买入，但实际没买入！可能原因：**

#### 1. 限价单成交机制问题

```python
fill = fill_limit_order(
    row, side="buy", 
    limit_price=124.05,  # 网格点位（价值线）
    ...
)
```

- 当前价格121.40 < 限价124.05
- 理论上限价买单应该成交（价格更优）
- 但`fill_limit_order`可能只在价格"触及"时成交，而不是"更优时"也成交
- **需要检查这个函数的实现**

#### 2. left_candidate_can_add返回False

```python
def left_candidate_can_add(candidate):
    return (
        _enabled(candidate.get("selected_for_trading"), default=True)
        and _enabled(candidate.get("signal_eligible"), default=True)
        and _enabled(candidate.get("allow_left"), default=False)  # 关键！
        and not left_value_falsification_reason(candidate)
    )
```

**关键：`allow_left`的默认值是False！**

如果候选池中的中际旭创没有明确设置`allow_left=True`，就无法通过检查。

**需要验证：候选池CSV中，value_model来源的候选股是否有allow_left字段？**

#### 3. 持仓限制

- 左侧最多1只（代码中明确限制）
- 如果已有其他左侧持仓，中际旭创无法买入
- **需要检查那几天是否已有左侧持仓**

#### 4. 价格从未触及网格点位

- 网格槽位0-4的买入价都是124.05（价值线）
- 如果股价一直低于124.05，限价单永远不会成交
- **但2024-09-24股价是121.40，9-26是127.94（超过价值线）**
- 9-26应该能触及124.05的买入价

---

## 改进方向

### 方案1：修复左侧网格逻辑（推荐）

**问题定位：**
1. 检查`allow_left`字段是否正确设置
2. 检查`fill_limit_order`的成交机制
3. 检查持仓限制是否过于严格

**修改建议：**
```python
# 候选池生成时（historical_candidates.py:446）
if (价格在价值线0.80-1.08之间):
    value_rows.append({
        ...
        "candidate_source": "value_model",
        "signal_eligible": True,
        # 关键：明确设置allow_left=True
    })
```

### 方案2：放宽左侧网格买入条件

**当前问题：**
- 槽位0-4的买入价都等于价值线
- 当价格低于价值线时，限价单可能不成交

**改进建议：**
```python
def left_grid_plan(value_line):
    anchor = float(value_line)
    plan = []
    for slot in range(10):
        if slot < 5:
            # 改进：前5个槽位也略低于价值线
            buy_price = anchor * (0.98 + slot * 0.02 / 5)  # 0.98-1.00的渐进价格
        else:
            buy_price = anchor * (1.0 - configured_left_grid_step) ** (slot - 4)
        ...
```

### 方案3：增加左侧主动买入模式

**当前：** 只使用限价单被动等待

**改进：** 当价格在价值线0.80-0.98区间时，主动市价买入

```python
# 在左侧网格逻辑中添加
if (
    can_add_left 
    and not state.left_grid_started
    and 0.80 <= price_to_value <= 0.98  # 价格低于价值线
):
    # 主动买入前5个槽位
    execute_buy(...)
```

---

## 下一步行动

**优先级排序：**

1. **验证allow_left字段**
   - 检查候选池CSV：value_model来源的股票是否有allow_left字段
   - 如果没有，这就是根本原因

2. **检查fill_limit_order逻辑**
   - 理解限价单的成交机制
   - 确认为什么2024-09-24没有成交

3. **检查持仓限制**
   - 查看那几天是否已有其他左侧持仓

4. **修改代码**
   - 根据验证结果，修改候选池生成或买入逻辑
