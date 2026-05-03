# Rayleigh-Bénard 熱對流模擬 - 設計文件

**日期**: 2026-02-26
**狀態**: 已確認，待實作

---

## 1. 目標

實作 DDF（Double Distribution Function）+ Boussinesq 近似的 Rayleigh-Bénard 對流模擬。

**驗收標準**：
- Ra = 1e5 → Nu ≈ 4.5–5.0
- Ra = 1e6 → Nu ≈ 8.0–10.0

---

## 2. 架構決策

### 不修改現有 LBMSolver

溫度場透過獨立模組管理，與流場鬆耦合：

```
LBMSolver (現有，不動)
    ↑ set_force_field_updater(thermal.compute_buoyancy)
ThermalModule (新增)
    ↓ 持有 g[nx_g, ny_g, 9]、T[nx_g, ny_g]
ThermalBoundaryConditions (新增)
```

### 理由

- 零侵入：不修改已驗證的核心求解器
- 可重用：ThermalModule 可套用於其他熱流 case（自然對流、散熱等）
- 模式一致：仿照現有 ch_lbm_solver 多相模組設計

---

## 3. 新增檔案

| 檔案 | 說明 |
|------|------|
| `src/lbm_taichi/core/thermal_module.py` | ThermalModule + ThermalBoundaryConditions |
| `examples/rayleigh_benard.py` | 主模擬腳本 |

---

## 4. ThermalModule 設計

### 4.1 欄位

```python
g     = ti.Vector.field(9, f32, (nx_g, ny_g))   # 溫度分佈函數
g_new = ti.Vector.field(9, f32, (nx_g, ny_g))   # 雙緩衝
T     = ti.field(f32, (nx_g, ny_g))             # 宏觀溫度
```

### 4.2 物理參數

```python
tau_g   # 熱鬆弛時間：tau_g = 0.5 + kappa / cs²
T_ref   # 參考溫度（= 0.5，中間值）
beta    # 熱膨脹係數（= 1.0 in dimensionless units）
g_grav  # 重力加速度（格子單位）
```

### 4.3 公開 API

```python
def step(self, f_dst)           # 推進溫度場一步（碰撞 + 串流）
def compute_buoyancy(self)      # 更新 solver.force_field（浮力）
def init_temperature(self, T_top, T_bot)  # 線性初始化
def get_nusselt(self)           # 計算並回傳 Nu 數
```

### 4.4 g 的碰撞方程

```
g_eq_α = T · w_α · (1 + (e_α · u) / cs²)
g_α*   = g_α - (g_α - g_eq_α) / tau_g      # BGK
```

### 4.5 浮力（Boussinesq）

```
F_y = rho * g_grav * beta * (T - T_ref)

# 透過 solver.force_field 施加，Guo scheme 已在 LBMSolver 內實作
```

---

## 5. ThermalBoundaryConditions 設計

### 5.1 公開 API

```python
def add_hot_wall(T_hot, location)    # Dirichlet：固定高溫壁
def add_cold_wall(T_cold, location)  # Dirichlet：固定低溫壁
def add_adiabatic_wall(location)     # Neumann：絕熱壁（零梯度）
def apply(g)                         # 每步施加所有溫度 BC
```

### 5.2 Dirichlet BC（Anti-Bounce-Back）

```
g_ᾱ(wall) = -g_α(wall) + 2 · w_α · T_wall
```

### 5.3 Neumann BC（Bounce-Back）

```
g_ᾱ(wall) = g_α(wall)
```

---

## 6. 無因次化策略

使用者指定 Ra、Pr，自動推導格子參數：

```python
u_ref = 0.05            # 參考速度（控制 Ma 數 < 0.3）
H     = ny              # 特徵長度
Re_eff = sqrt(Ra / Pr)  # 有效 Reynolds 數

nu    = u_ref * H / Re_eff
kappa = nu / Pr
tau_f = 0.5 + 3 * nu    # 動量鬆弛時間
tau_g = 0.5 + 3 * kappa # 熱鬆弛時間

# 浮力強度（保持 Ra 數一致）
g_lbm = Ra * nu * kappa / (H**3)  # beta * ΔT = 1
```

---

## 7. 主迴圈架構

```python
# 初始化
solver = LBMSolver(nx, ny, re=Re_eff, u_ref=u_ref, length_scale=H)
thermal = ThermalModule(solver, Pr=Pr, g_gravity=g_lbm)
thermal.init_temperature(T_top=0.0, T_bot=1.0)
solver.set_force_field_updater(thermal.compute_buoyancy)

bc = BoundaryConditions(solver)
bc.add_no_slip_wall('top')
bc.add_no_slip_wall('bottom')
bc.add_periodic_boundary('x')

thermal_bc = ThermalBoundaryConditions(thermal)
thermal_bc.add_hot_wall(T_hot=1.0, location='bottom')
thermal_bc.add_cold_wall(T_cold=0.0, location='top')
thermal_bc.add_adiabatic_wall('left')
thermal_bc.add_adiabatic_wall('right')

# 主迴圈
for step in range(1, steps + 1):
    f_src, f_dst = (solver.f, solver.f_new) if step % 2 == 1 else (solver.f_new, solver.f)
    g_src, g_dst = (thermal.g, thermal.g_new) if step % 2 == 1 else (thermal.g_new, thermal.g)

    solver.step(f_src, f_dst)          # ① 流場（含浮力）
    thermal.step(g_src, g_dst, f_dst)  # ② 溫度場
    thermal_bc.apply(g_dst)            # ③ 溫度 BC
```

---

## 8. 驗收流程

1. Ra = 1e4：確認對流胞成形（定性）
2. Ra = 1e5：Nu 數與文獻對比（定量）
3. Ra = 1e6：Nu 數與 Grossmann-Lohse 理論對比
4. 質量守恆誤差 < 1e-6（現有診斷系統）
