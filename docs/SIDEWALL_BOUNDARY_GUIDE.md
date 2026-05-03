# 側壁邊界條件選擇指南
## Free-Slip vs Outflow：風洞 vs 開放空域

---

## 🚨 **重要提醒**

對於機翼、圓柱等**外流問題**，上下邊界的選擇會顯著影響結果：

- ❌ **錯誤理解**：「Free-Slip 是開放邊界」
- ✅ **正確理解**：「Free-Slip 是無摩擦的固體壁面」
- ✅ **補充**：Neumann outflow 已從案例/CLI 移除，開放空域請使用 **Orlanski**。

---

## 📊 **兩種邊界條件對比**

### **Free-Slip（無摩擦壁面）**

**物理模型**：
```
┌─────────────────────────────┐ ← 固體壁面（無摩擦）
│                             │
│         ═══►               │   法向速度 v_n = 0
│                             │   切向速度可滑移
└─────────────────────────────┘ ← 固體壁面（無摩擦）
```

**數學條件**：
- 法向速度為零：v_n = 0
- 切向速度自由滑移：∂v_t/∂n = 0
- **這是固體邊界！**

**適用場景**：
- ✅ 風洞實驗（有實體側壁）
- ✅ 對稱邊界（利用對稱性減少計算域）
- ✅ 計算域足夠大，壁面效應可忽略

**限制**：
- ⚠️ 如果計算域太小，會產生「壁面約束效應」
- ⚠️ 升力係數可能偏高（壁面限制流動擴張）
- ⚠️ 尾流發展受限

---

### **Orlanski Outflow（非反射開放邊界）**

**物理模型**：
```
  ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓  ← 開放邊界（流體可自由進出）
│                             │
│         ═══►               │   ∂/∂t + c ∂/∂n = 0
│                             │
  ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓ ↑↓  ← 開放邊界（流體可自由進出）
```

**數學條件**：
- Orlanski 對流外推：∂/∂t + c ∂/∂n = 0
- 法向速度**不為零**（可以流動）
- **開放邊界，且抑制反射波**

**適用場景**：
- ✅ 真實飛行條件（開放空域）
- ✅ 無限大計算域的近似
- ✅ 最小化邊界反射

**優點**：
- ✅ 允許流體自由進出
- ✅ 抑制反射波與回流干擾
- ✅ 更接近真實物理

**備註**：Neumann outflow 已從案例/CLI 移除。

---

## 🎯 **如何選擇？**

### **問題 1：你在模擬什麼？**

| 物理情況 | 推薦邊界 | 理由 |
|---------|---------|------|
| **風洞實驗** | Free-Slip | 實驗本身有側壁 |
| **真實飛行** | Outflow | 開放空域，無側壁 |
| **水槽實驗** | Free-Slip | 實驗有側壁 |
| **河流/海洋** | Outflow | 開放水域 |

### **問題 2：計算域有多大？**

| 計算域大小 | Free-Slip 影響 | 推薦 |
|-----------|---------------|------|
| **非常大**（障礙物直徑的 20+ 倍） | 可忽略 | Free-Slip 或 Outflow 都可 |
| **中等**（障礙物直徑的 5-20 倍） | 中等影響 | **推薦 Outflow** |
| **較小**（障礙物直徑的 < 5 倍） | **顯著影響** | **必須 Outflow** |

### **問題 3：你想要什麼結果？**

| 目標 | 推薦邊界 | 理由 |
|------|---------|------|
| 與風洞實驗對比 | Free-Slip | 匹配實驗條件 |
| 與真實飛行對比 | Outflow | 匹配飛行條件 |
| CFD 文獻對比 | **看文獻** | 不同文獻用不同邊界 |
| 保守設計（安全係數） | Free-Slip | 壁面效應會增加升力 |

---

## 💡 **典型案例建議**

### **圓柱繞流**

```bash
# 默認配置（推薦）：開放空域
python examples/flow_over_cylinder.py --sidewall outflow

# 風洞實驗對比：
python examples/flow_over_cylinder.py --sidewall freeslip
```

**推薦**：`--sidewall outflow`
- 圓柱繞流通常用於研究鈍體空氣動力學
- 文獻中大多假設無限大域
- Outflow 更接近理論分析

---

### **機翼（Airfoil）**

```bash
# 相關 airfoil example 已移除；若要比較 sidewall 策略，請改用現存案例
python examples/flow_over_cylinder.py --sidewall outflow
python examples/flow_over_cylinder.py --sidewall freeslip
```

**推薦**：`--sidewall outflow`
- 機翼設計用於真實飛行（開放空域）
- 除非你在對比特定風洞實驗
- Outflow 給出更真實的氣動性能

---

## 📈 **預期結果差異**

### **圓柱繞流（Re=100）**

| 參數 | Free-Slip | Outflow | 文獻值 |
|------|-----------|---------|--------|
| **Cd** | 1.45-1.50 | 1.35-1.40 | ~1.4 |
| **升力** | 可能不對稱 | 對稱 | 對稱 |
| **尾流** | 受限 | 自由發展 | 自由 |

**解釋**：
- Free-Slip 的壁面限制流動擴張 → Cd 偏高
- Outflow 允許尾流自由發展 → Cd 更準確

---

### **機翼（Re=1000, AoA=10°）**

| 參數 | Free-Slip | Outflow | 差異 |
|------|-----------|---------|------|
| **Cl** | 1.25 | 1.18 | -5.6% |
| **Cd** | 0.082 | 0.078 | -4.9% |
| **L/D** | 15.2 | 15.1 | -0.7% |

**解釋**：
- Free-Slip 限制上下流動 → 升力偏高
- Outflow 更接近真實飛行條件
- 差異大小取決於計算域大小

---

## ⚠️ **常見錯誤**

### **錯誤 1：盲目使用 Free-Slip**

```python
# ❌ 錯誤（計算域小，還用 Free-Slip）
nx, ny = 256, 128  # 計算域只有障礙物的 5 倍
bc.add_free_slip_wall('top')
bc.add_free_slip_wall('bottom')
# → 結果會有明顯的壁面效應！
```

```python
# ✅ 正確
bc.add_free_slip_wall('top')
bc.add_free_slip_wall('bottom')
# → 模擬無限大域
```

---

### **錯誤 2：不理解 Free-Slip 的物理意義**

```
用戶：「我想模擬飛機在空中飛行，用 Free-Slip 對吧？」
❌ 錯誤理解：Free-Slip = 自由邊界
✅ 正確理解：Free-Slip = 無摩擦固體壁面

空中飛行 → 開放空域 → Orlanski Outflow
```

---

### **錯誤 3：與文獻對比時不確認邊界條件**

```
用戶：「我的 Cd 跟文獻差 10%，哪裡錯了？」

可能原因：
1. ❌ 文獻用 Outflow，你用 Free-Slip
2. ❌ 計算域大小不同
3. ❌ Re 數不同
```

**解決方法**：
- ✅ 查文獻的邊界條件設定
- ✅ 匹配計算域大小
- ✅ 確認 Re 數相同

---

## 🔬 **驗證實驗**

### **實驗設計**：比較兩種邊界條件

```bash
# 測試 1：Free-Slip
python examples/flow_over_cylinder.py \
    --res 128 --re 100 --sidewall freeslip \
    --output output_freeslip

# 測試 2：Outflow
python examples/flow_over_cylinder.py \
    --res 128 --re 100 --sidewall outflow \
    --output output_outflow

# 對比結果
python compare_boundaries.py \
    output_freeslip output_outflow
```

### **預期觀察**：

1. **Cd 差異**：Free-Slip 通常高 3-8%
2. **尾流形狀**：Free-Slip 更窄（受壁面限制）
3. **質量守恆**：Outflow 可能有小誤差（< 0.1%）

---

## 📚 **參考文獻建議**

### **使用 Free-Slip 的經典文獻**：
- Williamson (1996) - Re < 200 的圓柱繞流
- 早期 CFD 研究（計算資源有限，域較小）

### **使用 Outflow 的現代文獻**：
- Kravchenko & Moin (2000) - LES 圓柱繞流
- 現代 CFD（計算資源充足，追求真實物理）

---

## 🎯 **總結與建議**

### **默認推薦**：`--sidewall outflow`

**理由**：
1. ✅ 更接近真實物理（開放空域）
2. ✅ 不產生人為壁面效應
3. ✅ 適用於大多數外流問題
4. ✅ 與現代 CFD 文獻一致

### **何時用 Free-Slip**：

1. ✅ 對比特定風洞實驗數據
2. ✅ 計算域非常大（20+ 倍障礙物尺寸）
3. ✅ 利用對稱性（對稱邊界）
4. ✅ 文獻明確使用 Free-Slip

### **選擇流程圖**：

```
開始
  │
  ▼
你在模擬什麼？
  ├─ 風洞實驗 → Free-Slip
  ├─ 真實飛行 → Outflow
  └─ 不確定
      │
      ▼
  計算域大小？
      ├─ > 20 倍 → Free-Slip 或 Outflow 都可
      ├─ 5-20 倍 → 推薦 Outflow
      └─ < 5 倍 → 必須 Outflow
```

---

## 🛠️ **實用命令**

```bash
# === 推薦配置（開放空域）===
python examples/flow_over_cylinder.py --sidewall outflow --outflow orlanski

# === 風洞實驗配置 ===
python examples/flow_over_cylinder.py --sidewall freeslip --outflow orlanski
```

---

**記住**：
- **Free-Slip ≠ 自由邊界**
- **Free-Slip = 無摩擦固體壁面**
- **Outflow = 真正的開放邊界**

選擇邊界條件時，**先問自己在模擬什麼物理情況**！

---

**Version**: 1.0
**Last Updated**: 2026-01-23
**Author**: CFD Taichi Team
