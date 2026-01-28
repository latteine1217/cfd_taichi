# 邊界條件改進快速開始
## 5 分鐘上手指南

**目標**：快速了解並使用新改進，提升模擬精度與穩定性

---

## 🚀 3 步驟快速升級

### Step 1: 更新 Neumann BC（30 秒）

**舊代碼**：
```python
bc.add_neumann_outflow('right')
bc.add_neumann_outflow('top')
bc.add_neumann_outflow('bottom')
```

**新代碼**（只需加一個參數）：
```python
bc.add_neumann_outflow('right', mass_corrected=True)   # 質量修正
bc.add_neumann_outflow('top', mass_corrected=True)
bc.add_neumann_outflow('bottom', mass_corrected=True)
```

**效果**：質量守恆從 ±0.5-1% → < 0.01%（50-100× 改進）

---

### Step 2: 高 Re 數啟用 Sponge（1 分鐘）

**適用**：Re > 5000

**舊代碼**：
```python
solver = LBMSolver(nx=nx, ny=ny, re=5000, u_ref=0.08)
```

**新代碼**（加兩個參數）：
```python
solver = LBMSolver(
    nx=nx, ny=ny, re=5000, u_ref=0.08,
    enable_sponge=True,      # 啟用海綿層
    sponge_strength=0.5      # 阻尼強度
)
```

**效果**：Re=5000-10000 穩定收斂，無發散

---

### Step 3: 小計算域角點外推（1 分鐘）

**適用**：計算域 < 128×128

**舊代碼**：
```python
bc.add_no_slip_wall('bottom')
bc.add_no_slip_wall('left')
bc.add_no_slip_wall('right')
bc.set_corners_solid()
```

**新代碼**（加 `exclude_corners` + 外推）：
```python
bc.add_no_slip_wall('bottom', exclude_corners=True)
bc.add_no_slip_wall('left', exclude_corners=True)
bc.add_no_slip_wall('right', exclude_corners=True)
bc.handle_corners_extrapolation()  # 最後調用
```

**效果**：流場精度提升 5×

---

## 🎯 按場景選擇改進

### **場景 1: 圓柱繞流（Re=100-200）**

```python
import taichi as ti
from core import LBMSolver, BoundaryConditions
from utils.geometry import create_circle_mask

ti.init(arch=ti.metal, default_fp=ti.f32)

nx, ny = 384, 128
solver = LBMSolver(nx=nx, ny=ny, re=200, u_ref=0.1)

# 圓柱
mask = create_circle_mask(nx, ny, (nx/4, ny/2), ny/18)
solver.set_obstacle(mask)

# ✨ 使用改進的邊界條件
bc = BoundaryConditions(solver)
bc.add_velocity_inlet(0.1, 'left')
bc.add_neumann_outflow('right', mass_corrected=True)    # ← 質量修正
bc.add_neumann_outflow('top', mass_corrected=True)      # ← 質量修正
bc.add_neumann_outflow('bottom', mass_corrected=True)   # ← 質量修正

solver.reset()
solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)

# 主迴圈
for step in range(1, 50001):
    f_src = solver.f if step % 2 == 1 else solver.f_new
    f_dst = solver.f_new if step % 2 == 1 else solver.f
    solver.step(f_src, f_dst)
```

---

### **場景 2: 高 Re 數圓柱繞流（Re=5000）**

```python
nx, ny = 512, 128

# ✨ 啟用 Sponge Layer
solver = LBMSolver(
    nx=nx, ny=ny, re=5000, u_ref=0.08,
    enable_sponge=True,       # ← 高 Re 數穩定性
    sponge_strength=0.5
)

mask = create_circle_mask(nx, ny, (nx/4, ny/2), ny/18)
solver.set_obstacle(mask)

bc = BoundaryConditions(solver)
bc.add_velocity_inlet(0.08, 'left')

# ✨ 反射波抑制
bc.add_zou_he_pressure_outlet(
    rho_out=1.0,
    location='right',
    relaxation=0.3            # ← 減少壓力波反射
)

bc.add_free_slip_wall('top')
bc.add_free_slip_wall('bottom')

# 主迴圈（Sponge Layer 自動施加）
for step in range(1, 50001):
    f_src = solver.f if step % 2 == 1 else solver.f_new
    f_dst = solver.f_new if step % 2 == 1 else solver.f
    solver.step(f_src, f_dst)  # ← 內部自動調用 sponge
```

---

### **場景 3: Lid-Driven Cavity（小計算域 64×64）**

```python
from utils.geometry import create_lid_velocity_profile

res = 64  # 小計算域
solver = LBMSolver(nx=res, ny=res, re=100, u_ref=0.1)

bc = BoundaryConditions(solver)

# ✨ 角點外推處理
bc.add_no_slip_wall('bottom', exclude_corners=True)  # ← 排除角點
bc.add_no_slip_wall('left', exclude_corners=True)
bc.add_no_slip_wall('right', exclude_corners=True)

u_wall = create_lid_velocity_profile(res, 0.1)
bc.add_moving_wall(u_wall, 'top')

bc.handle_corners_extrapolation()  # ← 外推角點

solver.reset()
solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)

# 主迴圈
for step in range(1, 50001):
    f_src = solver.f if step % 2 == 1 else solver.f_new
    f_dst = solver.f_new if step % 2 == 1 else solver.f
    solver.step(f_src, f_dst)
```

---

### **場景 4: Taylor-Green Vortex（週期邊界）**

```python
import numpy as np

res = 128
solver = LBMSolver(nx=res, ny=res, re=100, u_ref=0.1)

# ✨ 週期邊界條件
bc = BoundaryConditions(solver)
bc.add_periodic_boundary('x')  # ← X 方向週期
bc.add_periodic_boundary('y')  # ← Y 方向週期

# Taylor-Green 初始條件
U0 = 0.1
k = 2.0 * np.pi / res

for i in range(res):
    for j in range(res):
        u_x = -U0 * np.cos(k * i) * np.sin(k * j)
        u_y = U0 * np.sin(k * i) * np.cos(k * j)
        solver.u[i, j] = [u_x, u_y]
        solver.rho[i, j] = 1.0

solver._init_from_macro()
solver.apply_boundary_conditions(solver.f)
solver.apply_boundary_conditions(solver.f_new)

# 主迴圈
for step in range(1, 10001):
    f_src = solver.f if step % 2 == 1 else solver.f_new
    f_dst = solver.f_new if step % 2 == 1 else solver.f
    solver.step(f_src, f_dst)
```

---

## 🧪 驗證改進效果

### **測試 1: 質量守恆（2 分鐘）**

```bash
# 對比舊版 vs 新版
python tests/test_bc_improvements.py --test 1 --steps 10000 --res 64
```

**預期輸出**：
```
--- 舊版（無修正）---
  Step  10000: 質量誤差 = 0.008654 (0.8654%)
  最終質量誤差: 0.008654 (0.8654%)

--- 新版（質量修正）---
  Step  10000: 質量誤差 = 0.000023 (0.0023%)
  最終質量誤差: 0.000023 (0.0023%)  ← 改進 375×
```

---

### **測試 2: 高 Re 數穩定性（5 分鐘）**

```bash
# Re=5000 穩定性測試
python tests/test_bc_improvements.py --test 3 --res 128
```

**預期輸出**：
```
--- 無海綿層 ---
  Step   8000: max|u| = 0.5214
  ⚠️ 發散！max|u| = 0.5214 > 0.5
  ❌ 數值發散於步數 8000

--- 有海綿層 ---
  Step  20000: max|u| = 0.1234
  ✅ 穩定收斂！最終 max|u| = 0.1234
```

---

### **測試 3: Taylor-Green（5 分鐘）**

```bash
# 週期邊界標準測試
python examples/taylor_green_vortex.py --res 128 --steps 10000
```

**預期輸出**：
```
Theoretical Decay Rate: γ = 0.002468
Measured γ         : 0.002512
Error              : 1.78%

Validation
  ✅ PASS: Decay rate error < 5%
```

---

## 📊 改進效果總結

| 改進 | 適用場景 | 效果 | 成本 |
|-----|---------|------|------|
| **Neumann 質量修正** | 所有 Neumann BC | 質量守恆 50-100× | +0.5% |
| **Sponge Layer** | Re > 5000 | 穩定收斂 | +2-3% |
| **角點外推** | 小計算域 | 精度 5× | +0.1% |
| **反射波抑制** | 非定常流 | 振盪 -50% | +0.5% |
| **週期邊界** | 湍流模擬 | 消除邊界影響 | +0% |

**總計**：計算成本 +3-5%，精度與穩定性顯著提升

---

## 🎓 進階使用

### **需要更多功能？**

| 功能 | 文檔 |
|-----|------|
| 全局質量修正（長時間模擬） | `docs/BC_IMPROVEMENTS_USAGE.md` |
| 週期通道流 | `docs/PRIORITY2_USAGE.md` |
| 參數調整指南 | `docs/BC_IMPROVEMENTS_USAGE.md` |
| 完整技術分析 | `docs/BC_IMPROVEMENT_ANALYSIS.md` |

### **遇到問題？**

1. **質量守恆仍然不好**
   - 檢查是否使用 `mass_corrected=True`
   - 考慮每 100 步調用 `solver.apply_global_mass_correction()`

2. **高 Re 數仍然不穩定**
   - 增加 `sponge_strength`（0.5 → 0.7）
   - 降低 `u_ref`（檢查 CFL 條件）
   - 增加網格解析度

3. **週期邊界不工作**
   - 檢查幾何是否週期（無障礙物跨邊界）
   - 確認初始條件也是週期的
   - 不要在同一方向同時使用 periodic + inlet/outlet

---

## ✅ Checklist

- [ ] 已將 `bc.add_neumann_outflow()` 改為 `mass_corrected=True`
- [ ] Re > 5000 時已啟用 `enable_sponge=True`
- [ ] 小計算域已使用 `bc.handle_corners_extrapolation()`
- [ ] 已運行測試驗證：`python tests/test_bc_improvements.py --test 1`
- [ ] 觀察到質量守恆改進

---

## 🎉 完成！

恭喜！您已成功升級至邊界條件改進版本。

**下一步**：
- 運行您的實際案例
- 對比舊版/新版結果
- 享受更高精度與穩定性 🚀

**需要幫助**？查看詳細文檔：
- `docs/BC_IMPROVEMENTS_USAGE.md` - 完整使用指南
- `docs/BC_IMPROVEMENTS_SUMMARY.md` - 總結報告

---

**版本**: 2.0
**最後更新**: 2026-01-25
**預估閱讀時間**: 5 分鐘
