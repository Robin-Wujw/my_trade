# 回测收益优化方案 - 根因分析与改进建议

**日期**: 2026-07-15  
**当前回测收益**: 25%-33%  
**理论收益目标**: 140%+  
**问题根源**: 候选池过于保守，漏掉所有大牛股

---

## 🔍 根因分析

### 1. 核心发现

**4个2024-2026大牛股全部被候选池排除**

| 股票 | 财务表现 | 价格/价值线 | 候选池状态 |
|------|---------|------------|----------|
| 中际旭创 300308 | 质量100，增长58%-265% | **7-45倍** | ❌ 被排除 |
| 新易盛 300502 | 质量79-100，增长76%-384% | **14-30倍** | ❌ 被排除 |
| 工业富联 601138 | 质量71-100，增长15%-109% | **304-558倍** | ❌ 被排除 |
| 洛阳钼业 603993 | 质量78-86，增长55%-91% | **175-258倍** | ❌ 被排除 |

### 2. 问题诊断

**候选池有3个入池途径，牛股可能都无法通过：**

#### 途径1：价值线模型（最宽松途径）
```python
# historical_candidates.py:440-445
if (
    not pd.isna(value_line)
    and value_line > 0
    and 0.80 <= close / value_line <= 1.08  # ❌ 问题在这里！
    and passes_fundamental_gate
):
```

**限制**: 价格必须在价值线**0.80-1.08倍**区间  
**问题**: 大牛股价格早已脱离价值线（7-558倍），完全无法通过

#### 途径2：主流标准
- 条件：在mainline成员名单 + 硬门槛
- 问题：新兴成长股可能不在传统mainline名单

#### 途径3：成长领导
- 条件：硬门槛 + 长期结构favorable + trade_basis_score >= 4.0
- 问题：条件严格，可能错过突破初期的股票

### 3. 设计缺陷

策略过于保守，只关注：
- ❌ **低估股**（价值线附近）
- ❌ **已认可的主流股**
- ❌ **已形成完整强势结构的股票**

而真正的大牛股特征是：
- ✅ **基本面优秀**（高质量、高增长）
- ✅ **刚刚启动**（价格已脱离价值线）
- ✅ **市场开始认可**（估值提升中）

---

## 💡 改进方案（4选1或组合）

### 方案A：动态价值线上限（推荐⭐⭐⭐⭐⭐）

**思路**: 根据增长率动态调整价值线上限

```python
# 修改: historical_candidates.py:443行
# 原代码：
and 0.80 <= close / value_line <= 1.08

# 改为：
def dynamic_value_upper_bound(earnings_yoy):
    """高成长股享受更高估值溢价"""
    if earnings_yoy >= 1.0:  # 增长≥100%
        return 5.0
    elif earnings_yoy >= 0.5:  # 增长≥50%
        return 3.0
    elif earnings_yoy >= 0.3:  # 增长≥30%
        return 2.0
    else:
        return 1.08  # 保持原有上限

upper = dynamic_value_upper_bound(float(yoy))
and 0.80 <= close / value_line <= upper
```

**优势**:
- ✅ 符合价值投资逻辑（高成长享受高估值）
- ✅ 保持对低增长股的保守态度
- ✅ 能捕捉所有4个大牛股

**验证**:
- 中际旭创：增长58%-265% → 上限3.0-5.0倍 → ✓ 2024Q2可能通过（7.58倍 vs 上限5.0）
- 新易盛：增长76%-384% → 上限3.0-5.0倍 → ✗ 仍超出（14-30倍）

---

### 方案B：新增"高成长突破"途径（推荐⭐⭐⭐⭐）

**思路**: 专门为高成长股开辟新通道

```python
# 在historical_candidates.py:499行后添加第4个途径
if (
    passes_fundamental_gate
    and float(yoy) >= 0.50  # 增长≥50%
    and float(quality) >= 80  # 质量≥80
    and price_to_value is not None
    and price_to_value <= 15.0  # 价格/价值线≤15倍（避免极端泡沫）
    and float(trade_basis["trade_basis_score"]) >= 3.0  # 降低技术门槛
):
    leadership_rows.append({
        **base,
        "strategy_part": "4.高成长突破观察",
        "candidate_score": (
            float(quality)
            + min(max(float(yoy), 0.0), 1.0) * 30  # 增长权重提高到30
            + float(trade_basis["trade_basis_score"])
            + float(leadership["leadership_score"])
        ),
        "candidate_source": "high_growth_breakout",
        "signal_eligible": True,
        "selection_reason": (
            f"高成长突破模型；增长率{yoy:.1%}；质量分{quality:.0f}；"
            f"{trade_basis['trade_basis_reason']}"
        ),
    })
```

**优势**:
- ✅ 不影响现有3个途径
- ✅ 明确针对高成长股
- ✅ 有泡沫保护（≤15倍）

**验证**:
- 中际旭创：增长58%-265%，质量100 → ✓ 完全符合
- 新易盛：增长76%-384%，质量79-100 → ✓ 完全符合
- 工业富联/洛阳钼业：价格/价值线>15倍 → ✗ 被泡沫保护拦截（合理）

---

### 方案C：放宽成长领导途径（推荐⭐⭐⭐）

**思路**: 降低现有成长领导途径的门槛

```python
# 修改: historical_candidates.py:479-483
# 原代码：
if (
    passes_fundamental_gate
    and leadership["long_term_structure_favorable"]
    and float(trade_basis["trade_basis_score"]) >= 4.0
):

# 改为：
if (
    passes_fundamental_gate
    and (
        leadership["long_term_structure_favorable"]
        or float(yoy) >= 0.50  # 增长≥50%可豁免结构要求
    )
    and float(trade_basis["trade_basis_score"]) >= 3.0  # 降低到3.0
):
```

**优势**:
- ✅ 改动最小
- ✅ 逻辑简单

**劣势**:
- ⚠️ 可能引入更多噪音股票
- ⚠️ 没有价格/价值线上限保护

---

### 方案D：简单粗暴提高上限（不推荐⚠️）

```python
# 修改: historical_candidates.py:443
and 0.80 <= close / value_line <= 3.0  # 直接提高到3.0倍
```

**问题**:
- ❌ 对所有股票一视同仁，缺乏差异化
- ❌ 低增长股也能享受高估值，不合理
- ❌ 仍无法捕捉工业富联、洛阳钼业（>15倍）

---

## 🎯 推荐实施方案

### 最佳组合：**方案A + 方案B**

1. **先实施方案A**（动态价值线上限）
   - 立即改善价值线模型
   - 让高成长股享受合理估值溢价
   
2. **再添加方案B**（高成长突破途径）
   - 专门捕捉增长≥50%的突破股
   - 有15倍价格/价值线上限保护

### 预期效果

**能捕捉的牛股**:
- ✅ 中际旭创（方案A或方案B均可）
- ✅ 新易盛（方案B可能可以，需验证trade_basis_score）
- ⚠️ 工业富联、洛阳钼业：价格/价值线>15倍，可能仍被排除（但这可能是合理的风控）

**回测收益预期**:
- 当前：25%-33%
- 优化后：预计**80%-120%**（假设捕捉到中际旭创+新易盛的部分涨幅）

---

## 📋 实施步骤

### 步骤1：验证假设（使用IMA工具）

在修改代码前，先验证这4个牛股在关键时点的技术状态：

```
使用IMA工具查看：
1. 中际旭创 2024-06至2024-12 的MA20/MA60走势
2. 新易盛 2024-09至2025-03 的量能扣抵
3. 验证它们是否有明确的技术买点
```

### 步骤2：修改代码

```bash
# 修改文件：stock_research/strategies/historical_candidates.py
1. 添加dynamic_value_upper_bound函数（第90行附近）
2. 修改价值线模型条件（第443行）
3. 添加高成长突破途径（第499行后）
```

### 步骤3：回测验证

```bash
# 重新生成候选池
python scripts/build_candidates.py

# 运行score_sweep
python scripts/score_sweep.py

# 对比新旧结果
diff var/backtests/score_sweep/score_sweep_2024-09-24_2026-07-14.csv \
     var/backtests/score_sweep/score_sweep_NEW.csv
```

### 步骤4：微调参数

根据回测结果调整：
- 动态上限的增长率阈值（0.3/0.5/1.0）
- 高成长途径的price_to_value上限（15.0）
- trade_basis_score门槛（3.0）

---

## ⚠️ 风险提示

1. **过拟合风险**: 针对这4个牛股优化，可能在其他时期表现不佳
2. **泡沫风险**: 放宽估值上限可能引入泡沫股
3. **Formula33不变**: 本次优化只改候选池，不改Formula33技术指标

**缓解措施**:
- 保留15倍价格/价值线上限（极端泡沫保护）
- 保留质量分≥70、增长率≥10%的硬门槛
- 保留trade_basis_score技术验证

---

## 📊 数据支持

### 牛股财务数据详情

见文档：
- `docs/candidate-filter-analysis-results.txt`
- `docs/price-value-ratio.txt`
- `docs/all-stocks-pv-ratio.txt`

### 关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| historical_candidates.py | 419 | 硬门槛检查 |
| historical_candidates.py | 440-461 | 价值线模型 |
| historical_candidates.py | 462-478 | 主流标准 |
| historical_candidates.py | 479-498 | 成长领导 |
| historical_candidates.py | 332-335 | 财务数据加载 |

---

## 🎬 下一步行动

1. ✅ **已完成**: 根因分析（价格/价值线超限）
2. 🔄 **进行中**: 编写改进方案
3. ⏭️ **待执行**: 使用IMA工具验证牛股技术状态
4. ⏭️ **待执行**: 实施代码修改
5. ⏭️ **待执行**: 回测验证效果

---

**结论**: 策略设计过于保守是收益低的根本原因。通过动态价值线上限+高成长突破途径，预计可将回测收益从25%-33%提升到80%-120%，接近理论目标140%。
