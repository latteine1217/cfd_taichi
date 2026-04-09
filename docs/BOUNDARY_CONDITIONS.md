# 邊界條件完整指南
## LBM 邊界條件分類與使用場景

---

## 📋 **邊界條件總覽**

| 類型 | 邊界條件 | 物理條件 | 使用場景 | API |
|------|---------|---------|---------|-----|
| **流體邊界** | 固定速度入口 | u = u_in | 固定速度入口 | `bc.add_velocity_inlet(u, location)` |
| | Stable Outlet | 弱密度鬆弛 + 穩定外流 | 一般外流 | `bc.add_stable_outlet(rho_out, location)` |
| | Orlanski 出口 | 對流外推 | 非反射出口 | `bc.add_orlanski_outflow(location)` |
| | Free-Slip 壁面 | v_n = 0, ∂v_t/∂n = 0 | 風洞上下壁、對稱邊界 | `bc.add_free_slip_wall(location)` |
| | Moving Wall | u = u_wall(x), v = 0 | 運動壁面（Lid-Driven Cavity） | `bc.add_moving_wall(profile, location)` |
| **固體邊界** | No-Slip 壁面 | u = 0, v = 0 | 固體邊界 | `bc.add_no_slip_wall(location)` |
| | 障礙物 | u = 0, v = 0 | 圓柱、機翼等 | `solver.set_obstacle(mask)` |
| **特殊處理** | 角點 | mask = 1 | 兩個邊界交界處 | `bc.set_corners_solid()` |

---

## 🧠 **研究協議：邊界條件不是參數旋鈕**

當調整入口、出口、側壁或角點策略時，必須把這次改動視為一個可證偽的實驗，而不是一次模糊的「調參」。

- **核心假設（Hypothesis）**：這個邊界修改預期解決什麼具體問題？例如出口反射、角點污染、回流失穩。
- **預期證據（Expected Evidence）**：哪個指標、哪個物理場、哪個頻譜應明顯改善？
- **證偽條件（Falsifiability）**：若什麼現象出現，就代表這個邊界策略失敗？
- **風險標籤（Risk Tag）**：用簡短標籤指出這次設定的主要風險，例如 `[OPEN_BOUNDARY_REFLECTION]`、`[CORNER_CONTAMINATION]`、`[BACKFLOW_RISK]`。

建議輸出格式：

```text
=== Hypothesis ===
- Hypothesis: 降低 outlet relaxation 可減少尾流回傳波。
- Expected Evidence: 出口附近密度擾動下降，Cl 頻譜主峰更集中。
- Falsifiability: 若尾流被過度平滑、St 消失，或 mass_error 上升一個數量級，則假設失敗。
- Risk Tag: [OPEN_BOUNDARY_REFLECTION]
```

---

## 🎯 **典型案例配置**

### **1. Lid-Driven Cavity（方腔流）**

```python
bc = BoundaryConditions(solver)

# 三面固體壁面
bc.add_no_slip_wall('bottom')
bc.add_no_slip_wall('left')
bc.add_no_slip_wall('right')

# 上蓋運動壁面
u_wall_profile = create_lid_velocity_profile(nx, u_lid)
bc.add_moving_wall(u_wall_profile, location='top')

# 角點處理
bc.set_corners_solid()

# 施加初始邊界條件
solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)
```

**物理邊界**：
- 底、左、右：No-Slip（速度為零）
- 頂部：運動壁面（u = u_lid, v = 0）
- 四個角點：固體（避免未定義行為）

---

### **2. Flow Over Cylinder（圓柱繞流）**

```python
bc = BoundaryConditions(solver)

# 入口：固定速度
bc.add_velocity_inlet(u_in, location='left')

# 出口：Stable Outlet（推薦）
bc.add_stable_outlet(rho_out=1.0, location='right', relaxation=0.02)

# 上下：依場景選擇
bc.add_orlanski_outflow('top', relaxation=0.02)   # open sky
bc.add_orlanski_outflow('bottom', relaxation=0.02)
# 或：
# bc.add_free_slip_wall('top')                     # wind tunnel
# bc.add_free_slip_wall('bottom')

# 圓柱：通過 mask 設置
mask = create_circle_mask(nx, ny, center=(cx, cy), radius=R)
solver.set_obstacle(mask)

solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)
```

**物理邊界**：
- 左側：固定速度入口（u = u_in）
- 右側：Stable Outlet（弱密度鬆弛 + 穩定外流）
- 上下：Orlanski 開放邊界或 Free-Slip（依 open sky / wind tunnel 決定）
- 圓柱表面：No-Slip（通過 Bounce-Back）

---

### **3. Airfoil（機翼）**

```python
bc = BoundaryConditions(solver)

bc.add_velocity_inlet(u_in, location='left')
bc.add_stable_outlet(rho_out=1.0, location='right', relaxation=0.02)
bc.add_orlanski_outflow('top', relaxation=0.02)
bc.add_orlanski_outflow('bottom', relaxation=0.02)

# 機翼：通過 mask 設置
mask = create_airfoil_system(nx, ny, naca='2412', ...)
solver.set_obstacle(mask)

solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)
```

與圓柱繞流相同，只是障礙物幾何不同。

---

## 🔬 **邊界條件詳細說明**

### **1. 固定速度入口（Velocity Inlet）**

**物理原理**：
- 固定入口速度 u = u_in
- 通過非平衡外推法重建未知分佈函數
- 滿足質量守恆與動量守恆

**公式**（以左邊界為例）：
```
已知：f0, f2, f3, f4, f6, f7（從內部流傳來）
未知：f1, f5, f8（需要重建）

質量守恆：rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - u_in)
動量守恆：rho*u_in = f1 - f3 + f5 - f6 - f7 + f8

重建：
f1 = f3 + (2/3) * rho * u_in
f5 = f7 - 0.5*(f2 - f4) + (1/6) * rho * u_in
f8 = f6 + 0.5*(f2 - f4) + (1/6) * rho * u_in
```

**使用場景**：
- 已知入口速度的流動問題
- 風洞入口
- 管道流入口

---

### **2. Orlanski 非反射出口（Outflow）**

**物理原理**：
- 以對流外推形式輸出波動
- 減少出口反射
- 以弱密度鬆弛維持長時穩定

**使用場景**：
- 非定常尾流
- 剪切層不穩定性
- 高 Re 外流

#### **2.1 無反射邊界檢核協議（Non-Reflecting Validation Protocol）**

當使用 Orlanski、Stable Outlet 或其他開放邊界時，Agent 應主動檢查以下現象：

- **出口附近壓力 / 密度堆積**：代表波動沒有被乾淨帶出。
- **尾流突然變直或過早衰減**：可能是鬆弛太強，數值耗散吃掉了真實渦結構。
- **回流區振盪放大**：可能是開放邊界與局部回流不相容。
- **力係數頻譜污染**：若 `Cl` / `Cd` 主峰漂移、展寬或消失，優先懷疑出口配置。

建議最低檢查項：

1. **CLI 指標**：`mass_error`、`u_max`、`CFL`、`Cl/Cd` 是否與調整前相比惡化。
2. **場觀察**：出口前 5-10 個格點內是否有密度/渦度的回傳波紋。
3. **頻譜觀察**：對圓柱繞流等非定常案例，主峰 `St` 是否仍清晰存在。
4. **對照試驗**：只改一個邊界參數，不同時更改 LES、解析度、入口擾動。

判讀規則：

- **Evidence**：出口附近波紋下降，且 `St` 主峰保留。
- **Interpretation**：邊界反射降低，且沒有用過強人工耗散掩蓋真實尾流。
- **Fail Signal**：若 `mass_error` 上升一個數量級、`Cl` 振幅異常消失、或尾流明顯被壓平，視為失敗。

常用風險標籤：

- `[OPEN_BOUNDARY_REFLECTION]`
- `[BACKFLOW_RISK]`
- `[OVER_DAMPED_WAKE]`

---

### **3. Neumann 出口（Zero-Gradient Outflow）** ✨ Deprecated

**物理原理**：
- 零梯度外推：∂u/∂x = 0, ∂v/∂x = 0, ∂ρ/∂x = 0
- 假設流動在出口處 fully developed
- 不強制特定壓力值

**實作方式**：
```
直接複製內部鄰近節點的分佈函數：
f[boundary] = f[interior]

例如右邊界：f[nx-1, j] = f[nx-2, j]
```

**優點**：
- 最簡單的開放邊界條件
- 不具侵入性（不強制特定壓力）
- 最小化反射波（但比 Orlanski 弱）

**缺點**：
- 不嚴格保證質量守恆（可能有小誤差）
- 不適用於強回流情況
- 已從案例/CLI 移除，僅保留內部測試用途

**說明**：
此選項已移除（數值穩定性不足），請改用 Orlanski 穩定出口。

**對比 Zou-He Pressure Outlet**：
| | Zou-He Pressure Outlet | Orlanski Outflow |
|---|---|---|
| **質量守恆** | ✅ 嚴格保證 | ⚠️ 近似（小誤差）|
| **侵入性** | 中等（強制 ρ = ρ_out） | 低（非反射外推）|
| **適用場景** | 已知出口壓力 | 非定常外流 |
| **實作複雜度** | 中等 | 中等 |

---

### **4. Free-Slip 壁面（改進版 Zou-He 類型）** ✨ IMPROVED

**物理原理**：
- 法向速度為零：v_n = 0
- 切向速度自由滑移：∂v_t/∂n = 0
- 零法向應力

**改進**：
- ✅ **原版**：簡單鏡面反射（f2 = f4）
- ✅ **改進版**：Zou-He 類型（Krüger 2017），正確處理密度變化與切向動量耦合

**公式**（以底部為例）：
```
法向條件：v_y = 0
已知：f0, f1, f3, f4, f7, f8
未知：f2, f5, f6

質量守恆：rho = f0 + f1 + f3 + 2*(f4 + f7 + f8)
切向速度：u_x = u_x(inner)（從內部節點外推，對應 ∂u_x/∂y = 0）

重建：
f2 = f4
f5 = f7 + 0.5*(f1 - f3) + (1/6) * rho * u_x
f6 = f8 - 0.5*(f1 - f3) - (1/6) * rho * u_x
```

**優點（vs 簡單鏡面反射）**：
- ✅ 正確處理密度變化（壓力梯度）
- ✅ 滿足質量守恆與動量守恆
- ✅ 在非均勻流場中更精確

**使用場景**：
- 風洞上下壁面
- 對稱邊界（利用對稱性減少計算域）
- 開放通道的側壁

---

### **5. No-Slip 壁面（Bounce-Back）**

**物理原理**：
- 固體壁面邊界條件
- 速度為零：u = v = 0
- 通過設置 mask = 1，使用 Bounce-Back 自動處理

**實作方式**：
```python
bc.add_no_slip_wall('bottom')  # 設置 mask[i, 0] = 1

# 在 streaming 階段自動 Bounce-Back：
# 粒子撞到固體後，沿相反方向反彈
# f_反向[i, j] = f_正向[i, j]
```

**使用場景**：
- 所有固體壁面（管道壁、容器壁等）
- 靜止障礙物表面

---

### **6. Moving Wall（運動壁面）**

**物理原理**：
- 壁面本身有速度 u_wall(x)
- v_y = 0（法向速度為零）
- 通過 Zou-He 類型推導重建分佈函數

**公式**（以頂部為例）：
```
已知：f0, f1, f2, f3, f5, f6
未知：f4, f7, f8

質量守恆：rho = f0 + f1 + f3 + 2*(f2 + f5 + f6)

重建（考慮壁面速度）：
f4 = f2
f7 = f5 - (1/6) * rho * u_wall + 0.5*(f1 - f3)
f8 = f6 + (1/6) * rho * u_wall - 0.5*(f1 - f3)
```

**使用場景**：
- Lid-Driven Cavity 的上蓋
- 運動的傳送帶
- 旋轉圓柱表面

---

### **7. 角點處理** ✨ NEW

**問題**：
- 角點被兩個邊界條件同時影響
- 未定義優先級會導致數值不穩定

**解決方案**：
```python
bc.set_corners_solid()  # 將四個角點設為固體（mask=1）
```

**或者在設置邊界時排除角點**：
```python
bc.add_no_slip_wall('bottom', exclude_corners=True)
bc.add_no_slip_wall('left', exclude_corners=True)
bc.add_no_slip_wall('right', exclude_corners=True)

# 然後手動處理角點
bc.set_corners_solid()
```

---

## 📊 **邊界條件選擇指南**

### **入口邊界**
- ✅ **已知速度**：使用 Velocity Inlet
- ✅ **已知壓力**：使用壓力型入口（較少見；需明確驗證其守恆與穩定性）

### **出口邊界**
- ✅ **一般外流**：使用 Stable Outlet（**推薦**）
- ✅ **非定常外流 / 開放天空**：使用 Orlanski 出口
- ⚠️ **不要把「更穩定」誤判為「更正確」**：若尾流渦街被壓平，代表出口可能過度耗散

### **側壁邊界**
- ✅ **固體壁面**：使用 No-Slip
- ✅ **無摩擦壁面**：使用 Free-Slip（改進版）
- ✅ **對稱邊界**：使用 Free-Slip
- ✅ **開放天空 / 遠場**：使用 Orlanski

### **特殊壁面**
- ✅ **運動壁面**：使用 Moving Wall
- ✅ **角點**：使用 `set_corners_solid()`

---

## ⚠️ **常見錯誤與注意事項**

### **1. 忘記施加初始邊界條件**
```python
# ❌ 錯誤
bc.add_velocity_inlet(0.1, 'left')
# 忘記施加

# ✅ 正確
bc.add_velocity_inlet(0.1, 'left')
solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)
```

### **2. Lid-Driven Cavity 忘記設置三面壁面**
```python
# ❌ 錯誤（三面是開放邊界！）
bc.add_moving_wall(u_profile, 'top')

# ✅ 正確
bc.add_no_slip_wall('bottom')
bc.add_no_slip_wall('left')
bc.add_no_slip_wall('right')
bc.add_moving_wall(u_profile, 'top')
bc.set_corners_solid()
```

### **3. 角點未處理**
```python
# ✅ 總是處理角點
bc.set_corners_solid()
```

### **4. mask 與邊界條件衝突**
- **流體邊界**（Zou-He, Free-Slip）：要求 mask = 0
- **固體邊界**（No-Slip, 障礙物）：要求 mask = 1
- 不要在同一個節點同時施加兩種類型

### **5. 同時改太多參數，導致無法判讀**
- ❌ 一次同時改 `relaxation`、`Cs`、解析度、入口擾動
- ✅ 一次只改一個主要邊界參數，保留其他設定不變
- ✅ 先寫出 Hypothesis / Expected Evidence / Falsifiability，再跑案例

### **6. 結果太平順卻被誤判為更好**
- 若圓柱尾流突然非常乾淨、`Cl` 振幅顯著下降、渦街提早消失，這通常不是「邊界更準」，而是「耗散更強」
- 若機翼尾跡貼近出口前就被拉直，優先檢查 outlet relaxation 與計算域長度

---

## 📈 **性能與精度**

| 邊界條件 | 精度 | 穩定性 | 計算成本 |
|---------|------|--------|---------|
| Velocity Inlet | 二階 | 好 | 中等 |
| Stable Outlet | 二階 | 很好 | 中等 |
| Orlanski 出口 | 二階 | 很好 | 中等 |
| Free-Slip（改進版） | 二階 | 好 | 中等 |
| No-Slip（Bounce-Back） | 二階 | 極好 | 極低 |
| Moving Wall | 二階 | 好 | 中等 |

---

## 🧪 **測試與驗證**

### **邊界條件研究回報格式**

```text
=== Status ===
- Case: flow_over_cylinder, Re=150, outlet relaxation=0.02

=== Evidence ===
- mass_error = 3.2e-4
- Cl 主峰存在，出口附近密度波紋較前一版下降

=== Interpretation ===
- 反射波降低，且尾流未被過度耗散，屬於正向改進

=== Next Step ===
- 固定其他參數，只測 relaxation=0.01 / 0.03 做窄範圍對照

=== Risk Tag ===
- [OPEN_BOUNDARY_REFLECTION]
```

運行測試腳本驗證所有邊界條件：
```bash
python test_boundary_improvements.py
```

測試內容：
1. ✅ 角點處理（exclude_corners, set_corners_solid）
2. ✅ Lid-Driven Cavity（No-Slip + Moving Wall）
3. ✅ Free-Slip（法向速度為零，切向速度滑移）
4. ✅ 質量守恆驗證
5. ✅ 開放邊界反射檢查（建議搭配 cylinder / airfoil 非定常案例）

---

## 📚 **參考文獻**

1. Zou, Q., & He, X. (1997). *On pressure and velocity boundary conditions for the lattice Boltzmann BGK model*. Physics of Fluids, 9(6), 1591-1598.
2. Krüger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.
3. Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

---

**Version**: 2.1
**Last Updated**: 2026-04-07
**Status**: ✅ Production Ready
