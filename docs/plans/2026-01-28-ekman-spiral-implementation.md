# Ekman 螺旋多層模擬器實現計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 實現多層 2D LBM 模擬器，捕捉科氏力驅動的 Ekman 螺旋現象

**Architecture:**
- 20 層獨立的 D2Q9 LBM 求解器垂直堆疊
- 層間通過顯式剪應力耦合
- 科氏力作為外力項施加於每層
- 使用現有 `LBMSolver` 核心架構

**Tech Stack:** Taichi (Metal backend), NumPy, Matplotlib

---

## 任務概覽

1. **Task 1**: 實現解析解函數（驗證基準）
2. **Task 2**: 實現 `MultiLayerEkmanSolver` 核心類
3. **Task 3**: 實現科氏力計算
4. **Task 4**: 實現層間剪應力耦合
5. **Task 5**: 實現診斷系統
6. **Task 6**: 實現主模擬腳本與 CLI
7. **Task 7**: 實現 Hodograph 可視化
8. **Task 8**: 單點驗證（1×1 網格 vs 解析解）
9. **Task 9**: 完整模擬與結果分析

---

## Task 1: 實現解析解函數

**Files:**
- Create: `examples/ekman_spiral.py` (開始架構)

### Step 1: 寫入檔案標頭與解析解函數

```python
"""
Ekman Spiral Multi-Layer Simulation
====================================

What: 多層 2D LBM 模擬風驅動的海洋 Ekman 螺旋
Why: 捕捉科氏力引起的垂直速度結構
When: 研究旋轉流體力學與海洋邊界層

物理現象：
- 表面流偏離風向 45°（北半球）
- 速度隨深度指數衰減
- Hodograph 呈順時針螺旋

參考：
- Ekman (1905), Cushman-Roisin & Beckers (2011)
- 設計文檔：docs/plans/2026-01-28-ekman-spiral-design.md
"""

import os
import sys
import taichi as ti
import numpy as np
import argparse
import time
from typing import Tuple, List

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lbm_taichi.core import LBMSolver

# === 物理常數 ===
RHO_WATER = 1025.0        # 海水密度 (kg/m³)
RHO_AIR = 1.225           # 空氣密度 (kg/m³)
NU_WATER = 1.0e-6         # 分子黏度 (m²/s)


def ekman_analytical_solution(
    z_depths: np.ndarray,
    f: float,
    nu_v: float,
    tau_wind: float,
    rho: float = RHO_WATER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算經典 Ekman 螺旋解析解

    What: 無限深海洋、恆定風應力的理論解
    Why: 驗證數值模擬的物理正確性
    When: 用於單點驗證（nx=1, ny=1）

    Physics:
        u(z) = u_0 * exp(-z/D_E) * cos(π/4 - z/D_E)
        v(z) = u_0 * exp(-z/D_E) * sin(π/4 - z/D_E)

        其中：
        - u_0 = τ_wind / (ρ √(ν_v f))
        - D_E = π √(2ν_v/f)
        - z: 深度（表面為 0，向下為正）

    Args:
        z_depths: 深度陣列 (m)，表面=0
        f: 科氏參數 (s⁻¹)
        nu_v: 垂直渦黏度 (m²/s)
        tau_wind: 風應力大小 (N/m²)
        rho: 海水密度 (kg/m³)

    Returns:
        (u, v): 東向與北向速度 (m/s)
    """
    # Ekman 深度
    D_E = np.pi * np.sqrt(2.0 * nu_v / f)

    # 表面速度幅值
    u_0 = tau_wind / (rho * np.sqrt(nu_v * f))

    # 無因次深度
    z_norm = z_depths / D_E

    # 速度分量
    u = u_0 * np.exp(-z_norm) * np.cos(np.pi / 4 - z_norm)
    v = u_0 * np.exp(-z_norm) * np.sin(np.pi / 4 - z_norm)

    return u, v


def compute_ekman_depth(f: float, nu_v: float) -> float:
    """計算 Ekman 深度 D_E = π√(2ν_v/f)"""
    return np.pi * np.sqrt(2.0 * nu_v / f)


def compute_wind_stress(U_10: float, C_d: float = 1.3e-3) -> float:
    """
    計算風應力 τ_w = ρ_air * C_d * U_10²

    Args:
        U_10: 10m 高度風速 (m/s)
        C_d: 阻力係數（典型值 1.3×10⁻³）

    Returns:
        風應力 (N/m²)
    """
    return RHO_AIR * C_d * U_10**2
```

### Step 2: 驗證解析解函數

手動測試解析解（加入腳本末尾，稍後會移除）：

```python
if __name__ == "__main__":
    # 快速測試解析解
    f = 1.0e-4
    nu_v = 1.0e-3
    tau_wind = 0.156

    z = np.linspace(0, 100, 21)
    u, v = ekman_analytical_solution(z, f, nu_v, tau_wind)

    D_E = compute_ekman_depth(f, nu_v)

    print("=== Ekman 解析解測試 ===")
    print(f"Ekman 深度: {D_E:.2f} m")
    print(f"表面速度: u={u[0]:.4f}, v={v[0]:.4f} m/s")
    print(f"表面偏角: {np.degrees(np.arctan2(v[0], u[0])):.1f}°")
    print(f"底層速度: u={u[-1]:.4e}, v={v[-1]:.4e} m/s")
```

### Step 3: 執行測試

```bash
cd /Users/latteine/Documents/coding/cfd_taichi
uv run python examples/ekman_spiral.py
```

**預期輸出**：
```
=== Ekman 解析解測試 ===
Ekman 深度: 99.67 m
表面速度: u=0.0558, v=0.0558 m/s
表面偏角: 45.0°
底層速度: u=2.4e-03, v=-2.4e-03 m/s
```

### Step 4: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: add Ekman analytical solution for validation

- Implement ekman_analytical_solution() function
- Add helper functions for Ekman depth and wind stress
- Include test output to verify correctness"
```

---

## Task 2: 實現 MultiLayerEkmanSolver 核心類

**Files:**
- Modify: `examples/ekman_spiral.py` (移除測試代碼，加入主類)

### Step 1: 寫入 MultiLayerEkmanSolver 類別架構

移除之前的測試代碼，加入：

```python
@ti.data_oriented
class MultiLayerEkmanSolver:
    """
    多層 Ekman 螺旋求解器

    What: 垂直離散化的 2D LBM 多層模型
    Why: 捕捉風驅動海洋流動的 Ekman 螺旋結構
    When: 需要模擬科氏力影響下的深度依賴性流動

    Architecture:
        - n_layers 層獨立的 LBMSolver
        - 每層通過週期性邊界條件（模擬無限海洋）
        - 層間通過顯式剪應力耦合
        - 科氏力作為外力項施加

    Attributes:
        n_layers: 垂直層數
        nx, ny: 水平網格尺寸
        dz: 層厚度 (m)
        f: 科氏參數 (s⁻¹)
        nu_v: 垂直渦黏度 (m²/s)
        tau_wind: 風應力向量 (N/m²)
        r_bottom: 底摩擦係數 (s⁻¹)
        layers: List[LBMSolver]
        shear_stress: 層間剪應力場
        coriolis_force: 科氏力場
    """

    def __init__(
        self,
        n_layers: int,
        nx: int,
        ny: int,
        dz: float,
        f: float,
        nu_v: float,
        tau_wind: Tuple[float, float],
        r_bottom: float,
        re: float = 1000.0,
        rho: float = RHO_WATER,
    ):
        """
        Args:
            n_layers: 垂直層數
            nx, ny: 水平網格尺寸
            dz: 層厚度 (m)
            f: 科氏參數 (s⁻¹)
            nu_v: 垂直渦黏度 (m²/s)
            tau_wind: 風應力向量 (τ_x, τ_y) (N/m²)
            r_bottom: 底摩擦係數 (s⁻¹)
            re: Reynolds 數（用於每層 LBM）
            rho: 海水密度 (kg/m³)
        """
        self.n_layers = n_layers
        self.nx = nx
        self.ny = ny
        self.dz = dz
        self.f = f
        self.nu_v = nu_v
        self.tau_wind = np.array(tau_wind, dtype=np.float32)
        self.r_bottom = r_bottom
        self.rho = rho
        self.re = re

        # 建立多層 LBM 求解器
        self.layers = []
        for k in range(n_layers):
            solver = LBMSolver(
                nx=nx,
                ny=ny,
                re=re,
                u_ref=0.05,  # 預估速度尺度
                length_scale=float(ny),
                cs=-1.0,  # 關閉 LES（使用外部渦黏度 nu_v）
                collision_model="mrt",
            )
            self.layers.append(solver)

        # === Taichi 場變數 ===
        # 剪應力：每層上下界面（n_layers+1 個界面）
        self.shear_stress_x = ti.field(dtype=ti.f32, shape=(n_layers + 1, nx, ny))
        self.shear_stress_y = ti.field(dtype=ti.f32, shape=(n_layers + 1, nx, ny))

        # 科氏力：每層
        self.coriolis_fx = ti.field(dtype=ti.f32, shape=(n_layers, nx, ny))
        self.coriolis_fy = ti.field(dtype=ti.f32, shape=(n_layers, nx, ny))

        # 總外力（科氏力 + 剪應力梯度）
        self.force_x = ti.field(dtype=ti.f32, shape=(n_layers, nx, ny))
        self.force_y = ti.field(dtype=ti.f32, shape=(n_layers, nx, ny))

        # 診斷量
        self.layer_u_mean = ti.field(dtype=ti.f32, shape=n_layers)
        self.layer_v_mean = ti.field(dtype=ti.f32, shape=n_layers)
        self.layer_ke = ti.field(dtype=ti.f32, shape=n_layers)

        # 初始化所有層為靜止
        self._init_layers()

    def _init_layers(self):
        """初始化所有層為靜止海洋"""
        for k, solver in enumerate(self.layers):
            # 設定週期性邊界（無邊界條件）
            # 初始速度 = 0，密度 = 1.0
            pass  # LBMSolver 預設已經是靜止

    def step(self):
        """
        單步時間推進

        順序：
        1. 計算科氏力
        2. 計算層間剪應力
        3. 計算總外力
        4. 各層 LBM step（含外力）
        """
        # 實現於 Task 3 & 4
        pass

    def get_velocity_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取垂直速度剖面（空間平均）

        Returns:
            (u_profile, v_profile): shape (n_layers,)
        """
        self._compute_layer_diagnostics()
        u_prof = self.layer_u_mean.to_numpy()
        v_prof = self.layer_v_mean.to_numpy()
        return u_prof, v_prof

    @ti.kernel
    def _compute_layer_diagnostics(self):
        """計算每層的空間平均診斷量"""
        for k in range(self.n_layers):
            u_sum = 0.0
            v_sum = 0.0
            ke_sum = 0.0
            count = 0

            for i, j in ti.ndrange(self.nx, self.ny):
                u_val = self.layers[k].u[i + 1, j + 1][0]
                v_val = self.layers[k].u[i + 1, j + 1][1]

                u_sum += u_val
                v_sum += v_val
                ke_sum += 0.5 * (u_val**2 + v_val**2)
                count += 1

            self.layer_u_mean[k] = u_sum / count
            self.layer_v_mean[k] = v_sum / count
            self.layer_ke[k] = ke_sum / count
```

### Step 2: 驗證類別初始化

測試代碼（暫時加入 `__main__`）：

```python
if __name__ == "__main__":
    ti.init(arch=ti.metal, default_fp=ti.f32)

    # 測試初始化
    solver = MultiLayerEkmanSolver(
        n_layers=20,
        nx=32,
        ny=32,
        dz=5.0,
        f=1.0e-4,
        nu_v=1.0e-3,
        tau_wind=(0.156, 0.0),  # 東風
        r_bottom=1.0e-4,
    )

    print("=== MultiLayerEkmanSolver 初始化測試 ===")
    print(f"層數: {solver.n_layers}")
    print(f"網格: {solver.nx}×{solver.ny}")
    print(f"層厚: {solver.dz} m")
    print(f"科氏參數: {solver.f} s⁻¹")
    print("✅ 初始化成功")
```

### Step 3: 執行測試

```bash
uv run python examples/ekman_spiral.py
```

**預期輸出**：
```
=== MultiLayerEkmanSolver 初始化測試 ===
層數: 20
網格: 32×32
層厚: 5.0 m
科氏參數: 0.0001 s⁻¹
✅ 初始化成功
```

### Step 4: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: implement MultiLayerEkmanSolver core class

- Create multi-layer structure with independent LBMSolver per layer
- Allocate Taichi fields for shear stress and Coriolis force
- Implement layer diagnostics (mean velocity, KE)
- Add initialization test"
```

---

## Task 3: 實現科氏力計算

**Files:**
- Modify: `examples/ekman_spiral.py:MultiLayerEkmanSolver`

### Step 1: 實現科氏力 kernel

在 `MultiLayerEkmanSolver` 類中加入：

```python
    @ti.kernel
    def _compute_coriolis_force(self):
        """
        計算所有層的科氏力

        Physics:
            F_coriolis = (f·v, -f·u)

            北半球（f > 0）：
            - 東向流（u > 0）產生南向力（Fy < 0）
            - 北向流（v > 0）產生東向力（Fx > 0）
            → 順時針偏轉
        """
        for k in range(self.n_layers):
            for i, j in ti.ndrange(self.nx, self.ny):
                # 從 LBMSolver 讀取速度（包含 ghost cells）
                u_val = self.layers[k].u[i + 1, j + 1][0]
                v_val = self.layers[k].u[i + 1, j + 1][1]

                # 科氏力
                self.coriolis_fx[k, i, j] = self.f * v_val
                self.coriolis_fy[k, i, j] = -self.f * u_val
```

### Step 2: 測試科氏力計算

在 `__main__` 中加入測試：

```python
    # 設定表層速度為純東向流
    solver.layers[0].u.fill(0.0)
    for i in range(solver.nx):
        for j in range(solver.ny):
            solver.layers[0].u[i + 1, j + 1] = [0.1, 0.0]  # 東向 0.1 m/s

    # 計算科氏力
    solver._compute_coriolis_force()

    # 檢查結果
    fx = solver.coriolis_fx.to_numpy()
    fy = solver.coriolis_fy.to_numpy()

    print("\n=== 科氏力測試（東向流） ===")
    print(f"Fx (應為 0): {fx[0, 0, 0]:.6f}")
    print(f"Fy (應為 -f*u = -1e-5): {fy[0, 0, 0]:.6e}")
    assert abs(fx[0, 0, 0]) < 1e-10, "Fx 應為 0"
    assert abs(fy[0, 0, 0] - (-1.0e-5)) < 1e-10, "Fy 應為 -1e-5"
    print("✅ 科氏力計算正確")
```

### Step 3: 執行測試

```bash
uv run python examples/ekman_spiral.py
```

**預期輸出**：
```
=== 科氏力測試（東向流） ===
Fx (應為 0): 0.000000
Fy (應為 -f*u = -1e-5): -1.000000e-05
✅ 科氏力計算正確
```

### Step 4: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: implement Coriolis force computation

- Add _compute_coriolis_force() kernel
- Coriolis force: F = (f·v, -f·u)
- Add test for eastward flow case
- Verify correct rightward deflection"
```

---

## Task 4: 實現層間剪應力耦合

**Files:**
- Modify: `examples/ekman_spiral.py:MultiLayerEkmanSolver`

### Step 1: 實現剪應力計算 kernel

在 `MultiLayerEkmanSolver` 中加入：

```python
    @ti.kernel
    def _compute_shear_stress(self):
        """
        計算層間剪應力（所有界面）

        Physics:
            τ(k→k+1) = ρ ν_v (u[k] - u[k+1]) / dz

            界面編號：
            - interface 0: 海洋表面（k=0 上方）
            - interface k: 第 k-1 層與第 k 層之間
            - interface n_layers: 海底（k=n_layers-1 下方）
        """
        # 頂層界面：風應力
        for i, j in ti.ndrange(self.nx, self.ny):
            self.shear_stress_x[0, i, j] = self.tau_wind[0]
            self.shear_stress_y[0, i, j] = self.tau_wind[1]

        # 中間界面：黏性剪應力
        for k in range(1, self.n_layers):
            for i, j in ti.ndrange(self.nx, self.ny):
                u_upper = self.layers[k - 1].u[i + 1, j + 1][0]
                v_upper = self.layers[k - 1].u[i + 1, j + 1][1]

                u_lower = self.layers[k].u[i + 1, j + 1][0]
                v_lower = self.layers[k].u[i + 1, j + 1][1]

                du_dz = (u_upper - u_lower) / self.dz
                dv_dz = (v_upper - v_lower) / self.dz

                self.shear_stress_x[k, i, j] = self.rho * self.nu_v * du_dz
                self.shear_stress_y[k, i, j] = self.rho * self.nu_v * dv_dz

        # 底層界面：底摩擦
        for i, j in ti.ndrange(self.nx, self.ny):
            u_bottom = self.layers[self.n_layers - 1].u[i + 1, j + 1][0]
            v_bottom = self.layers[self.n_layers - 1].u[i + 1, j + 1][1]

            # 線性拖曳：τ = -ρ r u dz
            self.shear_stress_x[self.n_layers, i, j] = -self.rho * self.r_bottom * u_bottom * self.dz
            self.shear_stress_y[self.n_layers, i, j] = -self.rho * self.r_bottom * v_bottom * self.dz

    @ti.kernel
    def _compute_total_force(self):
        """
        計算總外力 = 科氏力 + 剪應力梯度

        Physics:
            F_total = F_coriolis + (τ_top - τ_bottom) / (ρ dz)
        """
        for k in range(self.n_layers):
            for i, j in ti.ndrange(self.nx, self.ny):
                # 剪應力梯度（轉換為單位質量力）
                tau_top_x = self.shear_stress_x[k, i, j]
                tau_bottom_x = self.shear_stress_x[k + 1, i, j]
                F_shear_x = (tau_top_x - tau_bottom_x) / (self.rho * self.dz)

                tau_top_y = self.shear_stress_y[k, i, j]
                tau_bottom_y = self.shear_stress_y[k + 1, i, j]
                F_shear_y = (tau_top_y - tau_bottom_y) / (self.rho * self.dz)

                # 總外力
                self.force_x[k, i, j] = self.coriolis_fx[k, i, j] + F_shear_x
                self.force_y[k, i, j] = self.coriolis_fy[k, i, j] + F_shear_y
```

### Step 2: 實現完整的 step() 方法

更新 `step()` 方法：

```python
    def step(self):
        """
        單步時間推進

        順序：
        1. 計算科氏力
        2. 計算層間剪應力
        3. 計算總外力
        4. 各層 LBM step（含外力）
        """
        # 1. 科氏力
        self._compute_coriolis_force()

        # 2. 剪應力
        self._compute_shear_stress()

        # 3. 總外力
        self._compute_total_force()

        # 4. 各層時間推進
        for k, solver in enumerate(self.layers):
            # 將外力注入 LBM solver
            self._inject_force_to_layer(k)

            # LBM step
            if k % 2 == 0:
                solver.step(solver.f, solver.f_new)
            else:
                solver.step(solver.f_new, solver.f)

    @ti.kernel
    def _inject_force_to_layer(self, k: ti.i32):
        """
        將外力注入第 k 層的 LBM 求解器

        Method: 修改速度場（Guo's forcing scheme 簡化版）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            fx = self.force_x[k, i, j]
            fy = self.force_y[k, i, j]

            # 施加外力（動量更新）
            # Δu = F * Δt（LBM 中 Δt = 1）
            self.layers[k].u[i + 1, j + 1][0] += fx
            self.layers[k].u[i + 1, j + 1][1] += fy
```

### Step 3: 測試剪應力計算

在 `__main__` 中加入測試：

```python
    # 測試剪應力計算
    solver.layers[0].u.fill(0.0)
    solver.layers[1].u.fill(0.0)

    # 設定速度梯度：第 0 層 u=0.1，第 1 層 u=0.0
    for i in range(solver.nx):
        for j in range(solver.ny):
            solver.layers[0].u[i + 1, j + 1] = [0.1, 0.0]
            solver.layers[1].u[i + 1, j + 1] = [0.0, 0.0]

    solver._compute_shear_stress()

    tau_x = solver.shear_stress_x.to_numpy()

    print("\n=== 剪應力測試 ===")
    print(f"頂層界面（風應力）: {tau_x[0, 0, 0]:.4f} N/m²")
    print(f"第 1 界面（黏性）: {tau_x[1, 0, 0]:.4f} N/m²")

    # 理論值：τ = ρ ν_v Δu/dz = 1025 * 1e-3 * 0.1/5 = 0.0205
    expected = 1025.0 * 1.0e-3 * 0.1 / 5.0
    assert abs(tau_x[1, 0, 0] - expected) < 1e-6
    print(f"理論值: {expected:.4f} N/m²")
    print("✅ 剪應力計算正確")
```

### Step 4: 執行測試

```bash
uv run python examples/ekman_spiral.py
```

**預期輸出**：
```
=== 剪應力測試 ===
頂層界面（風應力）: 0.1560 N/m²
第 1 界面（黏性）: 0.0205 N/m²
理論值: 0.0205 N/m²
✅ 剪應力計算正確
```

### Step 5: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: implement vertical shear stress coupling

- Add _compute_shear_stress() kernel
- Handle top (wind stress), interior (viscous), bottom (friction)
- Add _compute_total_force() for Coriolis + shear gradient
- Implement complete step() method with force injection
- Add test for shear stress calculation"
```

---

## Task 5: 實現診斷系統

**Files:**
- Modify: `examples/ekman_spiral.py:MultiLayerEkmanSolver`

### Step 1: 加入診斷方法

在 `MultiLayerEkmanSolver` 中加入：

```python
    def compute_ekman_transport(self) -> Tuple[float, float, float]:
        """
        計算 Ekman 傳輸（垂直積分）

        Returns:
            (M_x, M_y, angle): 東向傳輸、北向傳輸、傳輸角度 (度)

        Physics:
            M_x = Σ u[k] * dz
            M_y = Σ v[k] * dz
            理論：傳輸方向垂直於風向（90°）
        """
        u_prof, v_prof = self.get_velocity_profile()

        M_x = np.sum(u_prof) * self.dz
        M_y = np.sum(v_prof) * self.dz
        angle = np.degrees(np.arctan2(M_y, M_x))

        return M_x, M_y, angle

    def compute_surface_angle(self) -> float:
        """
        計算表面流偏離東向的角度

        Returns:
            角度 (度)，理論值 ≈ 45°
        """
        u_prof, v_prof = self.get_velocity_profile()
        return np.degrees(np.arctan2(v_prof[0], u_prof[0]))

    def compute_total_ke(self) -> float:
        """計算總動能（所有層）"""
        self._compute_layer_diagnostics()
        ke = self.layer_ke.to_numpy()
        return np.sum(ke) * self.dz

    def check_mass_conservation(self) -> np.ndarray:
        """
        檢查每層質量守恆

        Returns:
            shape (n_layers,)，每層的質量誤差
        """
        errors = np.zeros(self.n_layers)
        for k, solver in enumerate(self.layers):
            solver._update_macro(solver.f if k % 2 == 0 else solver.f_new)
            solver._update_diagnostics()
            errors[k] = abs(solver.mass_error[None])
        return errors
```

### Step 2: 加入 CLI 輸出格式化

```python
    def print_diagnostics(self, step: int, physical_time_hr: float):
        """
        輸出診斷資訊（模擬 Diagnostics 格式）

        Args:
            step: 時間步數
            physical_time_hr: 物理時間（小時）
        """
        u_prof, v_prof = self.get_velocity_profile()
        u_surf = np.sqrt(u_prof[0]**2 + v_prof[0]**2)
        u_bot = np.sqrt(u_prof[-1]**2 + v_prof[-1]**2)

        angle_surf = self.compute_surface_angle()
        M_x, M_y, transport_angle = self.compute_ekman_transport()
        ke_total = self.compute_total_ke()

        mass_errors = self.check_mass_conservation()
        max_mass_error = np.max(mass_errors)

        print(
            f"| {step:5d} | {physical_time_hr:8.2f} | {u_surf:6.4f} | "
            f"{angle_surf:6.1f}° | {u_bot:6.4f} | {ke_total:8.4f} | "
            f"{transport_angle:6.1f}° | {max_mass_error:8.2e} |"
        )
```

### Step 3: 測試診斷系統

在 `__main__` 中加入測試：

```python
    # 測試診斷
    print("\n=== 診斷系統測試 ===")
    print("| step  | time(hr) | u_surf | angle_surf | u_bot  | KE_total | transport_angle | mass_err |")
    print("|-------|----------|--------|------------|--------|----------|-----------------|----------|")

    solver.print_diagnostics(0, 0.0)
    print("✅ 診斷系統正常")
```

### Step 4: 執行測試

```bash
uv run python examples/ekman_spiral.py
```

**預期輸出**：
```
=== 診斷系統測試 ===
| step  | time(hr) | u_surf | angle_surf | u_bot  | KE_total | transport_angle | mass_err |
|-------|----------|--------|------------|--------|----------|-----------------|----------|
|     0 |     0.00 | 0.0000 |    0.0° | 0.0000 |   0.0000 |    0.0° | 0.00e+00 |
✅ 診斷系統正常
```

### Step 5: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: add diagnostic system for Ekman solver

- Implement compute_ekman_transport() for vertical integration
- Add surface angle and total KE calculations
- Add mass conservation check for all layers
- Format CLI output to match project standards"
```

---

## Task 6: 實現主模擬腳本與 CLI

**Files:**
- Modify: `examples/ekman_spiral.py` (加入 `run_ekman_spiral()` 與 `main()`)

### Step 1: 寫入主模擬函數

移除 `__main__` 中的測試代碼，加入：

```python
def run_ekman_spiral(
    n_layers: int = 20,
    nx: int = 256,
    ny: int = 256,
    depth: float = 100.0,
    latitude: float = 45.0,
    U_10: float = 10.0,
    nu_v: float = 1.0e-3,
    r_bottom: float = 1.0e-4,
    steps: int = 50000,
    interval: int = 1000,
    output_dir: str = "output_ekman",
):
    """
    執行 Ekman 螺旋模擬

    Args:
        n_layers: 垂直層數
        nx, ny: 水平網格尺寸
        depth: 總深度 (m)
        latitude: 緯度 (度)
        U_10: 10m 風速 (m/s)
        nu_v: 垂直渦黏度 (m²/s)
        r_bottom: 底摩擦係數 (s⁻¹)
        steps: 總時間步數
        interval: 輸出間隔
        output_dir: 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" " * 15 + "EKMAN SPIRAL MULTI-LAYER SIMULATION")
    print("=" * 70)

    # === 物理參數 ===
    dz = depth / n_layers
    f = 2.0 * 7.2921e-5 * np.sin(np.radians(latitude))  # 科氏參數
    tau_wind_mag = compute_wind_stress(U_10)
    tau_wind = (tau_wind_mag, 0.0)  # 東風

    D_E = compute_ekman_depth(f, nu_v)
    T_i = 2.0 * np.pi / f  # 慣性週期 (s)
    T_i_hr = T_i / 3600.0  # 轉為小時

    print(f"\n=== 物理參數 ===")
    print(f"緯度: {latitude}°")
    print(f"科氏參數: {f:.6e} s⁻¹")
    print(f"慣性週期: {T_i_hr:.2f} hr")
    print(f"Ekman 深度: {D_E:.2f} m")
    print(f"10m 風速: {U_10} m/s")
    print(f"風應力: {tau_wind_mag:.4f} N/m²")
    print(f"垂直渦黏度: {nu_v:.6e} m²/s")
    print(f"底摩擦係數: {r_bottom:.6e} s⁻¹")

    print(f"\n=== 網格參數 ===")
    print(f"水平: {nx}×{ny}")
    print(f"垂直: {n_layers} 層")
    print(f"層厚: {dz:.2f} m")
    print(f"總深度: {depth} m")

    # === 初始化求解器 ===
    solver = MultiLayerEkmanSolver(
        n_layers=n_layers,
        nx=nx,
        ny=ny,
        dz=dz,
        f=f,
        nu_v=nu_v,
        tau_wind=tau_wind,
        r_bottom=r_bottom,
        re=1000.0,
    )

    # === 主迴圈 ===
    print(f"\n=== 開始模擬 ===")
    print("目標：3 個慣性週期（至穩態）")
    print(f"預估模擬時間：{3.0 * T_i_hr:.1f} hr")

    print("\n| step  | time(hr) | u_surf | angle_surf | u_bot  | KE_total | transport_angle | mass_err |")
    print("|-------|----------|--------|------------|--------|----------|-----------------|----------|")

    start_time = time.time()

    # 時間步長（LBM lattice time = 1）
    # 假設 u_ref = 0.05 lattice units ≈ 0.05 m/s
    # dt_physical ≈ dx_physical / u_ref = 1.0 / 0.05 = 20 s
    dt_physical = 20.0  # 秒

    for step in range(1, steps + 1):
        solver.step()

        if step % interval == 0:
            physical_time_hr = step * dt_physical / 3600.0
            solver.print_diagnostics(step, physical_time_hr)

            # 儲存狀態
            save_state(solver, step, output_dir)

    # === 總結 ===
    elapsed = time.time() - start_time
    print(f"\n--- 模擬完成，耗時 {elapsed:.2f} 秒 ---")

    # 最終對比解析解
    compare_with_analytical(solver, f, nu_v, tau_wind_mag, dz, output_dir)


def save_state(solver: MultiLayerEkmanSolver, step: int, output_dir: str):
    """儲存多層狀態"""
    u_prof, v_prof = solver.get_velocity_profile()

    state = {
        'step': step,
        'u_profile': u_prof,
        'v_profile': v_prof,
        'n_layers': solver.n_layers,
        'dz': solver.dz,
    }

    filename = os.path.join(output_dir, f"state_{step:06d}.npy")
    np.save(filename, state)


def compare_with_analytical(
    solver: MultiLayerEkmanSolver,
    f: float,
    nu_v: float,
    tau_wind: float,
    dz: float,
    output_dir: str,
):
    """與解析解對比"""
    u_num, v_num = solver.get_velocity_profile()

    z_depths = np.arange(solver.n_layers) * dz
    u_ana, v_ana = ekman_analytical_solution(z_depths, f, nu_v, tau_wind)

    # 相對誤差
    u_err = np.abs(u_num - u_ana) / (np.max(np.abs(u_ana)) + 1e-10)
    v_err = np.abs(v_num - v_ana) / (np.max(np.abs(v_ana)) + 1e-10)

    print(f"\n=== 與解析解對比 ===")
    print(f"平均相對誤差（u）: {np.mean(u_err):.2%}")
    print(f"平均相對誤差（v）: {np.mean(v_err):.2%}")
    print(f"最大相對誤差（u）: {np.max(u_err):.2%}")
    print(f"最大相對誤差（v）: {np.max(v_err):.2%}")

    # 儲存對比數據
    comparison = {
        'z_depths': z_depths,
        'u_numerical': u_num,
        'v_numerical': v_num,
        'u_analytical': u_ana,
        'v_analytical': v_ana,
        'u_error': u_err,
        'v_error': v_err,
    }

    filename = os.path.join(output_dir, "comparison.npy")
    np.save(filename, comparison)
    print(f"對比數據已儲存至 {filename}")


def main():
    parser = argparse.ArgumentParser(description="Ekman Spiral Multi-Layer Simulation")
    parser.add_argument('--n_layers', type=int, default=20, help='垂直層數')
    parser.add_argument('--nx', type=int, default=256, help='X 解析度')
    parser.add_argument('--ny', type=int, default=256, help='Y 解析度')
    parser.add_argument('--depth', type=float, default=100.0, help='總深度 (m)')
    parser.add_argument('--latitude', type=float, default=45.0, help='緯度 (度)')
    parser.add_argument('--U_10', type=float, default=10.0, help='10m 風速 (m/s)')
    parser.add_argument('--nu_v', type=float, default=1.0e-3, help='垂直渦黏度 (m²/s)')
    parser.add_argument('--r_bottom', type=float, default=1.0e-4, help='底摩擦係數 (s⁻¹)')
    parser.add_argument('--steps', type=int, default=50000, help='總步數')
    parser.add_argument('--interval', type=int, default=1000, help='輸出間隔')
    parser.add_argument('--output', type=str, default='output_ekman', help='輸出目錄')

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_ekman_spiral(
        n_layers=args.n_layers,
        nx=args.nx,
        ny=args.ny,
        depth=args.depth,
        latitude=args.latitude,
        U_10=args.U_10,
        nu_v=args.nu_v,
        r_bottom=args.r_bottom,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
```

### Step 2: 執行短時間測試（100 步）

```bash
uv run python examples/ekman_spiral.py --steps 100 --interval 50 --nx 32 --ny 32
```

**預期輸出**：應該顯示標頭、物理參數、並開始輸出診斷資訊。

### Step 3: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: implement main simulation function and CLI

- Add run_ekman_spiral() with full parameter control
- Implement save_state() for periodic output
- Add compare_with_analytical() for validation
- Add argparse CLI interface
- Calculate physical time from LBM time steps"
```

---

## Task 7: 實現 Hodograph 可視化

**Files:**
- Create: `examples/visualize_ekman.py`

### Step 1: 寫入可視化腳本

```python
"""
Ekman 螺旋可視化工具
====================

功能：
1. Hodograph（速度矢量圖）
2. 深度剖面對比（數值 vs 解析解）
3. 時間演化動畫
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from ekman_spiral import ekman_analytical_solution


def plot_hodograph(u_profile, v_profile, z_depths, output_path):
    """
    繪製 Hodograph（速度矢量圖）

    Args:
        u_profile: 東向速度 (m/s)
        v_profile: 北向速度 (m/s)
        z_depths: 深度 (m)
        output_path: 輸出檔案路徑
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 繪製數值解
    ax.plot(u_profile, v_profile, 'o-', linewidth=2, markersize=6,
            label='Numerical', color='blue')

    # 標記表面與底部
    ax.plot(u_profile[0], v_profile[0], 'ro', markersize=10, label='Surface')
    ax.plot(u_profile[-1], v_profile[-1], 'ks', markersize=10, label='Bottom')

    # 添加深度標記
    for i in [0, len(u_profile)//4, len(u_profile)//2, 3*len(u_profile)//4, -1]:
        ax.annotate(f'{z_depths[i]:.0f}m',
                   xy=(u_profile[i], v_profile[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)

    ax.set_xlabel('Eastward Velocity u (m/s)', fontsize=12)
    ax.set_ylabel('Northward Velocity v (m/s)', fontsize=12)
    ax.set_title('Ekman Spiral Hodograph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Hodograph 已儲存至 {output_path}")
    plt.close()


def plot_depth_profiles(u_num, v_num, u_ana, v_ana, z_depths, output_path):
    """
    繪製深度剖面對比

    Args:
        u_num, v_num: 數值解
        u_ana, v_ana: 解析解
        z_depths: 深度 (m)
        output_path: 輸出檔案路徑
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # 速度大小
    vel_num = np.sqrt(u_num**2 + v_num**2)
    vel_ana = np.sqrt(u_ana**2 + v_ana**2)

    axes[0].plot(vel_num, -z_depths, 'o-', label='Numerical', linewidth=2)
    axes[0].plot(vel_ana, -z_depths, '--', label='Analytical', linewidth=2)
    axes[0].set_xlabel('Velocity Magnitude (m/s)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title('Velocity Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 速度方向
    angle_num = np.degrees(np.arctan2(v_num, u_num))
    angle_ana = np.degrees(np.arctan2(v_ana, u_ana))

    axes[1].plot(angle_num, -z_depths, 'o-', label='Numerical', linewidth=2)
    axes[1].plot(angle_ana, -z_depths, '--', label='Analytical', linewidth=2)
    axes[1].set_xlabel('Flow Direction (°)')
    axes[1].set_ylabel('Depth (m)')
    axes[1].set_title('Flow Direction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 相對誤差
    u_err = np.abs(u_num - u_ana) / (np.max(np.abs(u_ana)) + 1e-10) * 100
    v_err = np.abs(v_num - v_ana) / (np.max(np.abs(v_ana)) + 1e-10) * 100

    axes[2].plot(u_err, -z_depths, 'o-', label='u error', linewidth=2)
    axes[2].plot(v_err, -z_depths, 's-', label='v error', linewidth=2)
    axes[2].set_xlabel('Relative Error (%)')
    axes[2].set_ylabel('Depth (m)')
    axes[2].set_title('Relative Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"深度剖面已儲存至 {output_path}")
    plt.close()


def visualize_ekman_results(output_dir: str):
    """
    可視化 Ekman 模擬結果

    Args:
        output_dir: 模擬輸出目錄
    """
    # 讀取最終狀態
    state_files = sorted([f for f in os.listdir(output_dir) if f.startswith('state_')])
    if not state_files:
        print("錯誤：找不到狀態檔案")
        return

    final_state = np.load(os.path.join(output_dir, state_files[-1]), allow_pickle=True).item()

    u_num = final_state['u_profile']
    v_num = final_state['v_profile']
    n_layers = final_state['n_layers']
    dz = final_state['dz']

    z_depths = np.arange(n_layers) * dz

    # 讀取對比數據
    comparison_file = os.path.join(output_dir, "comparison.npy")
    if os.path.exists(comparison_file):
        comp = np.load(comparison_file, allow_pickle=True).item()
        u_ana = comp['u_analytical']
        v_ana = comp['v_analytical']
    else:
        print("警告：找不到解析解對比數據")
        u_ana = v_ana = None

    # 繪製 Hodograph
    hodograph_path = os.path.join(output_dir, "hodograph.png")
    plot_hodograph(u_num, v_num, z_depths, hodograph_path)

    # 繪製深度剖面
    if u_ana is not None:
        profiles_path = os.path.join(output_dir, "depth_profiles.png")
        plot_depth_profiles(u_num, v_num, u_ana, v_ana, z_depths, profiles_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize Ekman Spiral Results")
    parser.add_argument('--output_dir', type=str, default='output_ekman',
                       help='模擬輸出目錄')

    args = parser.parse_args()

    visualize_ekman_results(args.output_dir)


if __name__ == "__main__":
    main()
```

### Step 2: 測試可視化（使用模擬數據）

先運行短時間模擬產生數據：

```bash
uv run python examples/ekman_spiral.py --steps 1000 --interval 500 --nx 32 --ny 32
```

然後可視化：

```bash
uv run python examples/visualize_ekman.py --output_dir output_ekman
```

### Step 3: 檢查輸出檔案

```bash
ls -lh output_ekman/*.png
```

**預期輸出**：
```
hodograph.png
depth_profiles.png
```

### Step 4: Commit

```bash
git add examples/visualize_ekman.py
git commit -m "feat: add Ekman spiral visualization tools

- Implement plot_hodograph() for velocity vector plot
- Implement plot_depth_profiles() for numerical vs analytical comparison
- Add relative error visualization
- Create CLI interface for post-processing"
```

---

## Task 8: 單點驗證（1×1 網格 vs 解析解）

**Files:**
- Modify: `examples/ekman_spiral.py` (加入驗證模式)

### Step 1: 加入單點驗證函數

在 `ekman_spiral.py` 中加入：

```python
def run_single_point_validation():
    """
    單點驗證：nx=1, ny=1，退化為垂直一維問題

    目標：
    - 驗證層間耦合正確性
    - 與解析解對比
    - 確認收斂性
    """
    print("=" * 70)
    print(" " * 15 + "SINGLE POINT VALIDATION (1×1 GRID)")
    print("=" * 70)

    # 物理參數（與設計文檔一致）
    n_layers = 20
    depth = 100.0
    dz = depth / n_layers
    latitude = 45.0
    f = 2.0 * 7.2921e-5 * np.sin(np.radians(latitude))
    U_10 = 10.0
    tau_wind_mag = compute_wind_stress(U_10)
    nu_v = 1.0e-3
    r_bottom = 1.0e-4

    print(f"\n物理參數：")
    print(f"  層數: {n_layers}")
    print(f"  科氏參數: {f:.6e} s⁻¹")
    print(f"  風應力: {tau_wind_mag:.4f} N/m²")

    # 初始化求解器（1×1 網格）
    solver = MultiLayerEkmanSolver(
        n_layers=n_layers,
        nx=1,
        ny=1,
        dz=dz,
        f=f,
        nu_v=nu_v,
        tau_wind=(tau_wind_mag, 0.0),
        r_bottom=r_bottom,
        re=100.0,  # 低 Re 確保穩定
    )

    # 運行至穩態（3 個慣性週期）
    T_i = 2.0 * np.pi / f
    T_i_hr = T_i / 3600.0
    dt_physical = 20.0  # 秒
    steps_per_Ti = int(T_i / dt_physical)
    total_steps = 3 * steps_per_Ti

    print(f"\n模擬設定：")
    print(f"  慣性週期: {T_i_hr:.2f} hr")
    print(f"  總步數: {total_steps} ({3} × 慣性週期)")
    print(f"  時間步長: {dt_physical} s")

    print(f"\n開始模擬...")
    print("| step  | time(hr) | u_surf | angle_surf | transport_angle | KE_total |")
    print("|-------|----------|--------|------------|-----------------|----------|")

    interval = total_steps // 20  # 輸出 20 次

    for step in range(1, total_steps + 1):
        solver.step()

        if step % interval == 0 or step == total_steps:
            physical_time_hr = step * dt_physical / 3600.0
            u_prof, v_prof = solver.get_velocity_profile()
            u_surf = np.sqrt(u_prof[0]**2 + v_prof[0]**2)
            angle_surf = np.degrees(np.arctan2(v_prof[0], u_prof[0]))
            _, _, transport_angle = solver.compute_ekman_transport()
            ke_total = solver.compute_total_ke()

            print(
                f"| {step:5d} | {physical_time_hr:8.2f} | {u_surf:6.4f} | "
                f"{angle_surf:6.1f}° | {transport_angle:7.1f}° | {ke_total:8.4f} |"
            )

    # 最終對比
    print(f"\n{'='*70}")
    print("最終對比（數值 vs 解析解）")
    print(f"{'='*70}")

    u_num, v_num = solver.get_velocity_profile()
    z_depths = np.arange(n_layers) * dz
    u_ana, v_ana = ekman_analytical_solution(z_depths, f, nu_v, tau_wind_mag)

    # 計算誤差
    u_err = np.abs(u_num - u_ana) / (np.max(np.abs(u_ana)) + 1e-10)
    v_err = np.abs(v_num - v_ana) / (np.max(np.abs(v_ana)) + 1e-10)

    print(f"\n平均相對誤差：")
    print(f"  u: {np.mean(u_err):.2%}")
    print(f"  v: {np.mean(v_err):.2%}")

    print(f"\n表面流驗證：")
    angle_surf_num = np.degrees(np.arctan2(v_num[0], u_num[0]))
    angle_surf_ana = np.degrees(np.arctan2(v_ana[0], u_ana[0]))
    print(f"  數值解偏角: {angle_surf_num:.1f}°")
    print(f"  理論偏角: {angle_surf_ana:.1f}° (45°)")
    print(f"  誤差: {abs(angle_surf_num - angle_surf_ana):.1f}°")

    print(f"\nEkman 傳輸驗證：")
    _, _, transport_angle = solver.compute_ekman_transport()
    print(f"  傳輸角度: {transport_angle:.1f}°")
    print(f"  理論值: 90° (垂直風向)")
    print(f"  誤差: {abs(transport_angle - 90.0):.1f}°")

    # 判定成功標準
    success = True
    print(f"\n{'='*70}")
    print("驗證結果")
    print(f"{'='*70}")

    checks = [
        ("平均相對誤差 < 10%", np.mean(u_err) < 0.10 and np.mean(v_err) < 0.10),
        ("表面偏角 = 45° ± 10°", abs(angle_surf_num - 45.0) < 10.0),
        ("傳輸角度 = 90° ± 5°", abs(transport_angle - 90.0) < 5.0),
    ]

    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {desc}")
        if not passed:
            success = False

    if success:
        print(f"\n{'='*70}")
        print("🎉 驗證成功！數值解與理論解一致")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("⚠️  驗證失敗，需要調整參數或檢查物理實現")
        print(f"{'='*70}")

    return success
```

### Step 2: 更新 main() 加入驗證模式

```python
def main():
    parser = argparse.ArgumentParser(description="Ekman Spiral Multi-Layer Simulation")
    parser.add_argument('--validate', action='store_true',
                       help='執行單點驗證（1×1 網格）')
    parser.add_argument('--n_layers', type=int, default=20, help='垂直層數')
    # ... 其他參數 ...

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    if args.validate:
        success = run_single_point_validation()
        sys.exit(0 if success else 1)
    else:
        run_ekman_spiral(
            # ... 參數 ...
        )
```

### Step 3: 執行單點驗證

```bash
uv run python examples/ekman_spiral.py --validate
```

**預期輸出**：應該看到模擬運行，最終顯示驗證結果（PASS/FAIL）。

### Step 4: Commit

```bash
git add examples/ekman_spiral.py
git commit -m "feat: add single-point validation mode

- Implement run_single_point_validation() for 1×1 grid test
- Compare numerical solution against analytical Ekman spiral
- Check surface angle (45°), transport angle (90°), and velocity errors
- Add success criteria checks with clear output
- Update CLI to support --validate flag"
```

---

## Task 9: 完整模擬與結果分析

**Files:**
- Modify: 無需修改程式碼，僅執行與分析

### Step 1: 執行完整模擬（256×256 網格）

```bash
uv run python examples/ekman_spiral.py \
  --n_layers 20 \
  --nx 256 \
  --ny 256 \
  --depth 100.0 \
  --latitude 45.0 \
  --U_10 10.0 \
  --steps 50000 \
  --interval 1000 \
  --output output_ekman_full
```

**預期執行時間**：約 10-20 分鐘（取決於硬體）

### Step 2: 監控模擬輸出

觀察 CLI 輸出，確認：
- 質量守恆誤差 < 1e-6
- 表面流偏角逐漸趨向 45°
- 傳輸角度逐漸趨向 90°
- 動能增加並趨於穩定

### Step 3: 可視化結果

```bash
uv run python examples/visualize_ekman.py --output_dir output_ekman_full
```

### Step 4: 分析結果

檢查輸出檔案：

```bash
ls -lh output_ekman_full/
open output_ekman_full/hodograph.png
open output_ekman_full/depth_profiles.png
```

**預期現象**：
- Hodograph 呈現順時針螺旋
- 速度隨深度指數衰減
- 數值解與解析解誤差 < 10%

### Step 5: 撰寫結果總結

在 `docs/plans/2026-01-28-ekman-spiral-results.md` 中記錄：

```markdown
# Ekman 螺旋模擬結果總結

## 執行日期
2026-01-28

## 模擬參數
- 層數：20
- 網格：256×256
- 深度：100 m
- 緯度：45°N
- 風速：10 m/s

## 驗證結果
- ✅ 質量守恆：< 1e-6
- ✅ 表面偏角：XX.X° (目標 45° ± 10°)
- ✅ 傳輸角度：XX.X° (目標 90° ± 5°)
- ✅ 速度誤差：< 10%

## 觀察到的物理現象
- 順時針 Hodograph 螺旋（北半球）
- 速度指數衰減
- Ekman 深度處速度 ≈ 4% 表面值

## 已知限制
- 層數解析度（20 層可能略粗）
- 數值擴散影響
- 非真實 3D 物理

## 後續改進方向
- 增加層數至 50
- 時變風場
- 完整 3D LBM
```

### Step 6: 最終 Commit

```bash
git add docs/plans/2026-01-28-ekman-spiral-results.md
git add output_ekman_full/*.png
git commit -m "docs: add Ekman spiral simulation results

- Complete full-scale simulation (256×256 grid, 20 layers)
- Generate Hodograph and depth profile comparisons
- Document validation results and physical observations
- Confirm Ekman spiral formation and theoretical consistency"
```

---

## 總結

完成本計劃後，你將擁有：

✅ **核心功能**：
- 多層 2D LBM 求解器
- 科氏力與剪應力耦合
- 風應力與底摩擦邊界條件

✅ **驗證系統**：
- 單點驗證（1×1 網格）
- 與解析解對比
- 物理一致性檢查

✅ **可視化工具**：
- Hodograph 速度矢量圖
- 深度剖面對比
- 誤差分析

✅ **文檔記錄**：
- 設計文檔
- 實現計劃
- 結果總結

---

**預計總開發時間**：4-6 小時（不含長時間模擬運行）

**關鍵里程碑**：
1. Task 1-2: 架構搭建（1 hr）
2. Task 3-4: 物理實現（2 hr）
3. Task 5-7: 診斷與可視化（1 hr）
4. Task 8-9: 驗證與分析（1-2 hr）

**最終交付物**：
- `examples/ekman_spiral.py`（~600 行）
- `examples/visualize_ekman.py`（~200 行）
- 設計與結果文檔
- 驗證數據與圖表
