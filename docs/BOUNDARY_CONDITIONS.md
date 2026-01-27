# 邊界條件完整指南
## LBM 邊界條件分類與使用場景

---

## 📋 **邊界條件總覽**

| 類型 | 邊界條件 | 物理條件 | 使用場景 | API |
|------|---------|---------|---------|-----|
| **流體邊界** | 固定速度入口 | u = u_in | 固定速度入口 | `bc.add_velocity_inlet(u, location)` |
| | Neumann 出口 | ∂u/∂n = 0 | 充分發展流出口 | `bc.add_neumann_outflow(location)` |
| | Orlanski 出口 | 對流外推 | 非反射出口 | `bc.add_orlanski_outflow(location)` |
| | Free-Slip 壁面 | v_n = 0, ∂v_t/∂n = 0 | 風洞上下壁、對稱邊界 | `bc.add_free_slip_wall(location, mode="symmetric")` |
| | Moving Wall | u = u_wall(x), v = 0 | 運動壁面（Lid-Driven Cavity） | `bc.add_moving_wall(profile, location)` |
| **固體邊界** | No-Slip 壁面 | u = 0, v = 0 | 固體邊界 | `bc.add_no_slip_wall(location)` |
| | 障礙物 | u = 0, v = 0 | 圓柱、機翼等 | `solver.set_obstacle(mask)` |
| **特殊處理** | 角點 | mask = 1 | 兩個邊界交界處 | `bc.set_corners_solid()` |

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

# 出口：Orlanski 非反射出口（推薦）或 Neumann
bc.add_orlanski_outflow(location='right')
# 或：bc.add_neumann_outflow('right')

# 上下：Free-Slip（對稱延拓）
bc.add_free_slip_wall('top', mode='symmetric')
bc.add_free_slip_wall('bottom', mode='symmetric')

# 圓柱：通過 mask 設置
mask = create_circle_mask(nx, ny, center=(cx, cy), radius=R)
solver.set_obstacle(mask)

solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)
```

**物理邊界**：
- 左側：固定速度入口（u = u_in）
- 右側：Orlanski 非反射出口或零梯度出口
- 上下：Free-Slip 對稱延拓（法向速度為零，切向自由滑移）
- 圓柱表面：No-Slip（通過 Bounce-Back）

---

### **3. Airfoil（機翼）**

```python
bc = BoundaryConditions(solver)

bc.add_velocity_inlet(u_in, location='left')
bc.add_orlanski_outflow(location='right')
bc.add_free_slip_wall('top', mode='symmetric')
bc.add_free_slip_wall('bottom', mode='symmetric')

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

---

### **3. Neumann 出口（Zero-Gradient Outflow）** ✨ NEW

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
- 最小化反射波

**缺點**：
- 不嚴格保證質量守恆（可能有小誤差）
- 不適用於強回流情況

**使用場景**：
- 長管道出口（充分發展流）
- 不確定出口壓力時
- 希望最小化邊界影響時

**對比 Zou-He Pressure Outlet**：
| | Zou-He Pressure Outlet | Neumann Outflow |
|---|---|---|
| **質量守恆** | ✅ 嚴格保證 | ⚠️ 近似（小誤差）|
| **侵入性** | 中等（強制 ρ = ρ_out） | 最小（零梯度）|
| **適用場景** | 已知出口壓力 | 充分發展流 |
| **實作複雜度** | 中等 | 簡單 |

---

### **4. Free-Slip 壁面（改進版 Zou-He 類型）** ✨ IMPROVED

**物理原理**：
- 法向速度為零：v_n = 0
- 切向速度自由滑移：∂v_t/∂n = 0
- 零法向應力

**改進**：
- ✅ **原版**：簡單鏡面反射（f2 = f4）
- ✅ **改進版**：Zou-He 類型，正確處理密度變化

**公式**（以底部為例）：
```
法向條件：v_y = 0
已知：f0, f1, f3, f4, f7, f8
未知：f2, f5, f6

質量守恆：rho = f0 + f1 + f3 + 2*(f4 + f7 + f8)
切向速度：u_x = (f1 - f3 + f8 - f7) / rho

重建：
f2 = f4
f5 = f7 + (1/6) * rho * u_x
f6 = f8 - (1/6) * rho * u_x
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
- ✅ **已知速度**：使用 Zou-He 速度入口
- ✅ **已知壓力**：使用 Zou-He 壓力入口（較少見）

### **出口邊界**
- ✅ **已知壓力**：使用 Zou-He 壓力出口（**推薦**）
- ✅ **充分發展流**：使用 Neumann 出口
- ✅ **不確定壓力**：使用 Neumann 出口

### **側壁邊界**
- ✅ **固體壁面**：使用 No-Slip
- ✅ **無摩擦壁面**：使用 Free-Slip（改進版）
- ✅ **對稱邊界**：使用 Free-Slip

### **特殊壁面**
- ✅ **運動壁面**：使用 Moving Wall
- ✅ **角點**：使用 `set_corners_solid()`

---

## ⚠️ **常見錯誤與注意事項**

### **1. 忘記施加初始邊界條件**
```python
# ❌ 錯誤
bc.add_zou_he_velocity_inlet(0.1, 'left')
# 忘記施加

# ✅ 正確
bc.add_zou_he_velocity_inlet(0.1, 'left')
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

---

## 📈 **性能與精度**

| 邊界條件 | 精度 | 穩定性 | 計算成本 |
|---------|------|--------|---------|
| Zou-He 入口/出口 | 二階 | 好 | 中等 |
| Neumann 出口 | 一階 | 很好 | 低 |
| Free-Slip（改進版） | 二階 | 好 | 中等 |
| Free-Slip（簡單版） | 一階 | 好 | 低 |
| No-Slip（Bounce-Back） | 二階 | 極好 | 極低 |
| Moving Wall | 二階 | 好 | 中等 |

---

## 🧪 **測試與驗證**

運行測試腳本驗證所有邊界條件：
```bash
python test_boundary_improvements.py
```

測試內容：
1. ✅ 角點處理（exclude_corners, set_corners_solid）
2. ✅ Lid-Driven Cavity（No-Slip + Moving Wall）
3. ✅ Free-Slip（法向速度為零，切向速度滑移）
4. ✅ 質量守恆驗證

---

## 📚 **參考文獻**

1. Zou, Q., & He, X. (1997). *On pressure and velocity boundary conditions for the lattice Boltzmann BGK model*. Physics of Fluids, 9(6), 1591-1598.
2. Krüger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.
3. Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

---

**Version**: 2.0
**Last Updated**: 2026-01-23
**Status**: ✅ Production Ready
