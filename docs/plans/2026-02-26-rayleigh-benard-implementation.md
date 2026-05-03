# Rayleigh-Bénard 熱對流 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 實作 DDF + Boussinesq Rayleigh-Bénard 對流模擬，驗證 Ra=1e5～1e6 的 Nu 數。

**Architecture:** 新增獨立 `ThermalModule` 與 `ThermalBoundaryConditions` 類別（`core/thermal_module.py`）；透過 `solver.force_field_updater` 鬆耦合到現有 `LBMSolver`；新增 `examples/rayleigh_benard.py` 主腳本。零修改現有程式碼。

**Tech Stack:** Taichi 1.7.4, Metal backend, float32, D2Q9 BGK for thermal

---

## Task 1: ThermalModule 基礎結構

**Files:**
- Create: `src/lbm_taichi/core/thermal_module.py`
- Create: `tests/test_thermal_module.py`

### Step 1: 寫失敗測試

```python
# tests/test_thermal_module.py
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_thermal_module_init():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=32, ny=32, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.0, T_ref=0.5)

    assert thermal.g.shape == (34, 34)  # nx_g=34, ny_g=34
    assert thermal.T.shape == (34, 34)
    assert abs(thermal.kappa - solver.nu / 0.71) < 1e-6
    assert thermal.tau_g > 0.5


def test_init_temperature_linear():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=32, ny=32, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal.init_temperature(T_bot=1.0, T_top=0.0)

    T_np = thermal.T.to_numpy()
    # 底部（j=1）應接近 1.0，頂部（j=32）應接近 0.0
    assert abs(T_np[16, 1] - 1.0) < 0.05
    assert abs(T_np[16, 32] - 0.0) < 0.05
    # 中間層應接近 0.5
    assert abs(T_np[16, 16] - 0.5) < 0.1
```

### Step 2: 執行確認失敗

```bash
uv run python -m pytest tests/test_thermal_module.py::test_thermal_module_init -v
```
預期：`ImportError: cannot import name 'ThermalModule'`

### Step 3: 實作 ThermalModule 骨架

```python
# src/lbm_taichi/core/thermal_module.py
"""
Thermal LBM Module (DDF + Boussinesq)
======================================

What: 溫度場的 Double Distribution Function 求解器
Why:  標準 D2Q9 LBM 無法直接求解能量方程；DDF 以獨立分佈函數 g 求解
      溫度對流擴散方程，透過 Boussinesq 近似提供浮力給流場求解器。

適用場景: 自然對流 (Rayleigh-Bénard, 方腔自然對流)、散熱模擬
不適用: Ma > 0.3 的可壓縮流、Ra > 1e7（float32 精度限制）
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class ThermalModule:
    """
    溫度場 DDF 求解器

    What: 持有 g 分佈函數場，求解溫度對流擴散方程
    Why:  Boussinesq 近似下，溫度方程可獨立求解，
          再將浮力 F_y = ρ g β (T - T_ref) 注入速度求解器
    """

    def __init__(
        self,
        solver,
        Pr: float = 0.71,
        beta: float = 1.0,
        g_gravity: float = 0.0,
        T_ref: float = 0.5,
    ):
        """
        Args:
            solver:    現有 LBMSolver 實例（不修改）
            Pr:        Prandtl 數，決定熱擴散率 κ = ν/Pr
            beta:      熱膨脹係數（無因次化時通常 = 1.0）
            g_gravity: 重力加速度（格子單位，正值向 +y）
            T_ref:     參考溫度（Boussinesq 展開點，通常取 0.5）
        """
        self.solver = solver
        self.nx = solver.nx
        self.ny = solver.ny
        self.nx_g = solver.nx_g  # nx + 2（含 ghost cells）
        self.ny_g = solver.ny_g  # ny + 2
        self.Pr = Pr
        self.beta = beta
        self.g_gravity = g_gravity
        self.T_ref = T_ref

        # 熱擴散率與鬆弛時間
        self.kappa = solver.nu / Pr
        self.tau_g = 0.5 + 3.0 * self.kappa  # cs² = 1/3

        # D2Q9 常數（與 LBMSolver 相同）
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)
        self._init_constants()

        # 分佈函數場（雙緩衝）
        self.g = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.g_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.T = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        # Nusselt 診斷
        self.nu_sum = ti.field(dtype=ti.f32, shape=())

        self._fill_equilibrium(0.5)

    def _init_constants(self):
        """設定 D2Q9 格子常數（與 LBMSolver 一致）"""
        w_np = np.array(
            [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32
        )
        e_np = np.array(
            [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)],
            dtype=np.int32,
        )
        inv_np = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)

    @ti.kernel
    def _fill_equilibrium(self, T_init: ti.f32):
        """以均勻溫度 T_init 填充 g 為平衡分佈"""
        for i, j in self.g:
            self.T[i, j] = T_init
            for k in ti.static(range(9)):
                self.g[i, j][k] = self.w[k] * T_init
                self.g_new[i, j][k] = self.w[k] * T_init

    def init_temperature(self, T_bot: float, T_top: float):
        """
        線性溫度初始化

        Why: 線性分佈是純導熱的穩態解，減少暫態，加快收斂
        """
        self._init_temperature_kernel(float(T_bot), float(T_top))

    @ti.kernel
    def _init_temperature_kernel(self, T_bot: ti.f32, T_top: ti.f32):
        for i, j in self.g:
            inside = (j >= 1) and (j <= self.ny)
            if inside:
                y_frac = ti.cast(j - 1, ti.f32) / ti.cast(self.ny - 1, ti.f32)
                T_loc = T_bot + (T_top - T_bot) * y_frac
            else:
                T_loc = T_bot if j == 0 else T_top
            self.T[i, j] = T_loc
            for k in ti.static(range(9)):
                self.g[i, j][k] = self.w[k] * T_loc
                self.g_new[i, j][k] = self.w[k] * T_loc
```

### Step 4: 執行確認通過

```bash
uv run python -m pytest tests/test_thermal_module.py -v
```
預期：2 tests PASSED

### Step 5: Commit

```bash
git add src/lbm_taichi/core/thermal_module.py tests/test_thermal_module.py
git commit -m "feat: add ThermalModule skeleton with field init and linear temperature"
```

---

## Task 2: 熱 BGK 碰撞 + 串流 Kernel

**Files:**
- Modify: `src/lbm_taichi/core/thermal_module.py`

### Step 1: 新增測試

在 `tests/test_thermal_module.py` 追加：

```python
def test_thermal_step_preserves_uniform_temperature():
    """均勻溫度場在靜止流場下，經過多步後應保持不變"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=1000.0, u_ref=0.05)
    # 靜止流場（u=0）
    solver.u.fill(0.0)

    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.7)  # 均勻溫度 0.7

    # 執行 100 步
    for step in range(100):
        g_src = thermal.g if step % 2 == 0 else thermal.g_new
        g_dst = thermal.g_new if step % 2 == 0 else thermal.g
        thermal.step(g_src, g_dst)

    thermal._update_temperature(g_dst)
    T_np = thermal.T.to_numpy()
    # 物理域內溫度應保持 0.7 ± 0.01
    T_inner = T_np[1:17, 1:17]
    assert np.max(np.abs(T_inner - 0.7)) < 0.01
```

### Step 2: 執行確認失敗

```bash
uv run python -m pytest tests/test_thermal_module.py::test_thermal_step_preserves_uniform_temperature -v
```
預期：`AttributeError: 'ThermalModule' object has no attribute 'step'`

### Step 3: 實作碰撞串流

在 `thermal_module.py` 的 `ThermalModule` 類別內新增：

```python
    def step(self, g_src: ti.template(), g_dst: ti.template()):
        """
        執行溫度場單步推進（BGK 碰撞 + 串流）

        Why 先 update_temperature 再 collision?
        - 從 g_src 重算 T 確保每步一致性
        - 避免上一步 BC 修改造成的 T 不一致
        """
        self._update_temperature(g_src)
        self._thermal_collide_stream(g_src, g_dst)

    @ti.kernel
    def _update_temperature(self, g: ti.template()):
        """從 g 分佈函數重算宏觀溫度"""
        for i, j in ti.ndrange((1, self.nx + 1), (1, self.ny + 1)):
            T_loc = 0.0
            for k in ti.static(range(9)):
                T_loc += g[i, j][k]
            self.T[i, j] = T_loc

    @ti.kernel
    def _thermal_collide_stream(self, g_src: ti.template(), g_dst: ti.template()):
        """
        合併 BGK 碰撞與串流

        Why 合併?: 減少記憶體讀寫，省去 g_post 緩衝場
        Why BGK 而非 MRT?: 溫度方程只需 1 個鬆弛時間，MRT 不帶來額外收益

        碰撞方程:
            g_eq_α = T · w_α · (1 + e_α·u / cs²)
            g_α*   = g_α - (g_α - g_eq_α) / τ_g
        """
        cs2 = 1.0 / 3.0
        for i, j in ti.ndrange((1, self.nx + 1), (1, self.ny + 1)):
            if self.solver.mask[i, j] == 1:
                continue

            T_loc = 0.0
            for k in ti.static(range(9)):
                T_loc += g_src[i, j][k]

            ux = self.solver.u[i, j][0]
            uy = self.solver.u[i, j][1]

            for k in ti.static(range(9)):
                ex = ti.cast(self.e[k][0], ti.f32)
                ey = ti.cast(self.e[k][1], ti.f32)
                eu = ex * ux + ey * uy
                g_eq = self.w[k] * T_loc * (1.0 + eu / cs2)
                g_post = g_src[i, j][k] - (g_src[i, j][k] - g_eq) / self.tau_g

                # 串流至相鄰格點（ghost cells 作為緩衝，不會越界）
                ni = i + self.e[k][0]
                nj = j + self.e[k][1]
                g_dst[ni, nj][k] = g_post
```

### Step 4: 執行確認通過

```bash
uv run python -m pytest tests/test_thermal_module.py -v
```
預期：3 tests PASSED

### Step 5: Commit

```bash
git add src/lbm_taichi/core/thermal_module.py tests/test_thermal_module.py
git commit -m "feat: add thermal BGK collision-stream kernel"
```

---

## Task 3: 浮力力場計算

**Files:**
- Modify: `src/lbm_taichi/core/thermal_module.py`

### Step 1: 新增測試

```python
def test_buoyancy_zero_at_T_ref():
    """T = T_ref 時浮力應為零"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=1000.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.001, T_ref=0.5)
    thermal._fill_equilibrium(0.5)  # T = T_ref 均勻
    thermal._update_temperature(thermal.g)

    thermal.compute_buoyancy()
    F_np = solver.force_field.to_numpy()
    assert np.max(np.abs(F_np[1:17, 1:17, 1])) < 1e-6  # F_y ≈ 0


def test_buoyancy_positive_above_T_ref():
    """T > T_ref 時 F_y 應為正（向上浮力）"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=1000.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.001, T_ref=0.5)
    thermal._fill_equilibrium(1.0)  # T = 1.0 > T_ref = 0.5
    thermal._update_temperature(thermal.g)

    thermal.compute_buoyancy()
    F_np = solver.force_field.to_numpy()
    assert np.min(F_np[1:17, 1:17, 1]) > 0  # 全場 F_y > 0
```

### Step 2: 執行確認失敗

```bash
uv run python -m pytest tests/test_thermal_module.py::test_buoyancy_zero_at_T_ref -v
```
預期：`AttributeError: 'ThermalModule' object has no attribute 'compute_buoyancy'`

### Step 3: 實作浮力計算

在 `ThermalModule` 類別內新增：

```python
    def compute_buoyancy(self):
        """
        計算浮力並更新 solver.force_field

        Why 透過 force_field 而非直接修改?
        - 保持與 LBMSolver Guo forcing scheme 的相容性
        - solver.force_field_updater 在每步開頭自動呼叫此函數
        """
        self._compute_buoyancy_kernel()
        solver = self.solver
        solver.force_field_enabled[None] = 1
        solver._refresh_force_enabled()

    @ti.kernel
    def _compute_buoyancy_kernel(self):
        """
        Boussinesq 浮力: F_y = ρ · g · β · (T - T_ref)

        Why Boussinesq? 密度變化只在浮力項計入，流場仍視為不可壓縮
        Why ρ 乘？LBM 中 force_field 是加速度 × ρ（體積力密度）
        """
        for i, j in ti.ndrange((1, self.nx + 1), (1, self.ny + 1)):
            if self.solver.mask[i, j] == 1:
                self.solver.force_field[i, j] = ti.Vector([0.0, 0.0])
            else:
                T_loc = self.T[i, j]
                rho_loc = self.solver.rho[i, j]
                F_y = rho_loc * self.g_gravity * self.beta * (T_loc - self.T_ref)
                self.solver.force_field[i, j] = ti.Vector([0.0, F_y])

    def register_with_solver(self):
        """
        將 compute_buoyancy 註冊為 solver 的 force_field_updater

        使用方法:
            thermal.register_with_solver()
            # 之後 solver.step() 每步自動更新浮力
        """
        self.solver.set_force_field_updater(self.compute_buoyancy)
```

### Step 4: 執行確認通過

```bash
uv run python -m pytest tests/test_thermal_module.py -v
```
預期：5 tests PASSED

### Step 5: Commit

```bash
git add src/lbm_taichi/core/thermal_module.py tests/test_thermal_module.py
git commit -m "feat: add Boussinesq buoyancy force computation and solver registration"
```

---

## Task 4: ThermalBoundaryConditions

**Files:**
- Modify: `src/lbm_taichi/core/thermal_module.py`

### Step 1: 新增測試

```python
def test_hot_wall_bottom_sets_temperature():
    """底部熱壁應讓 j=1 的溫度趨近 T_hot"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions

    solver = LBMSolver(nx=16, ny=16, re=1000.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.5)

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location='bottom')

    # 施加 BC 多次讓溫度收斂
    for _ in range(50):
        tbc.apply(thermal.g)
        thermal._update_temperature(thermal.g)

    T_np = thermal.T.to_numpy()
    # j=1 的平均溫度應接近 1.0
    assert np.mean(T_np[1:17, 1]) > 0.9


def test_adiabatic_wall_left():
    """絕熱左壁：施加 BC 後溫度梯度不應從邊界引入熱量"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions

    solver = LBMSolver(nx=16, ny=16, re=1000.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.5)

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_adiabatic_wall('left')
    tbc.apply(thermal.g)

    # 施加後分佈函數應保持合理（不爆炸）
    g_np = thermal.g.to_numpy()
    assert not np.any(np.isnan(g_np))
    assert not np.any(np.isinf(g_np))
```

### Step 2: 執行確認失敗

```bash
uv run python -m pytest tests/test_thermal_module.py::test_hot_wall_bottom_sets_temperature -v
```
預期：`ImportError: cannot import name 'ThermalBoundaryConditions'`

### Step 3: 實作 ThermalBoundaryConditions

在 `thermal_module.py` 末尾新增（與 `ThermalModule` 同一檔案）：

```python
@ti.data_oriented
class ThermalBoundaryConditions:
    """
    溫度場邊界條件

    What: 管理 g 分佈函數在壁面的 Dirichlet/Neumann BC
    Why:
      - Dirichlet（固定溫度）: Anti-Bounce-Back scheme
        g_ᾱ = -g_α + 2·w_α·T_wall
      - Neumann（絕熱, ∂T/∂n=0）: Bounce-Back
        g_ᾱ = g_α
    """

    def __init__(self, thermal: ThermalModule):
        self.thermal = thermal
        self.nx = thermal.nx
        self.ny = thermal.ny

        # BC 類型: 0=none, 1=Dirichlet, 2=Neumann(adiabatic)
        self.bc_bottom = ti.field(dtype=ti.i32, shape=())
        self.bc_top    = ti.field(dtype=ti.i32, shape=())
        self.bc_left   = ti.field(dtype=ti.i32, shape=())
        self.bc_right  = ti.field(dtype=ti.i32, shape=())
        self.T_bottom  = ti.field(dtype=ti.f32, shape=())
        self.T_top     = ti.field(dtype=ti.f32, shape=())
        self.T_left    = ti.field(dtype=ti.f32, shape=())
        self.T_right   = ti.field(dtype=ti.f32, shape=())

        for f in [self.bc_bottom, self.bc_top, self.bc_left, self.bc_right]:
            f[None] = 0
        for f in [self.T_bottom, self.T_top, self.T_left, self.T_right]:
            f[None] = 0.0

    def add_hot_wall(self, T_hot: float, location: str):
        """固定高溫壁（Dirichlet）"""
        self._set_wall(location, bc_type=1, T_val=T_hot)

    def add_cold_wall(self, T_cold: float, location: str):
        """固定低溫壁（Dirichlet）"""
        self._set_wall(location, bc_type=1, T_val=T_cold)

    def add_adiabatic_wall(self, location: str):
        """絕熱壁（Neumann，零熱通量）"""
        self._set_wall(location, bc_type=2, T_val=0.0)

    def _set_wall(self, location: str, bc_type: int, T_val: float):
        if location == 'bottom':
            self.bc_bottom[None] = bc_type
            self.T_bottom[None] = T_val
        elif location == 'top':
            self.bc_top[None] = bc_type
            self.T_top[None] = T_val
        elif location == 'left':
            self.bc_left[None] = bc_type
            self.T_left[None] = T_val
        elif location == 'right':
            self.bc_right[None] = bc_type
            self.T_right[None] = T_val
        else:
            raise ValueError(f"Unknown location: {location}")

    def apply(self, g: ti.template()):
        """施加所有已設定的溫度邊界條件"""
        if self.bc_bottom[None] > 0:
            self._apply_bottom_bc(g)
        if self.bc_top[None] > 0:
            self._apply_top_bc(g)
        if self.bc_left[None] > 0:
            self._apply_left_bc(g)
        if self.bc_right[None] > 0:
            self._apply_right_bc(g)

    @ti.kernel
    def _apply_bottom_bc(self, g: ti.template()):
        """
        底部壁面 BC（壁在 j=0，流體在 j=1）

        未知方向（從壁面進入流體）: k=2(↑), k=5(↗), k=6(↖)
        Dirichlet: g_k = -g_inv[k] + 2·w[k]·T_wall
        Neumann:   g_k = g_inv[k]
        """
        w   = self.thermal.w
        inv = self.thermal.inv
        T_wall = self.T_bottom[None]
        bc    = self.bc_bottom[None]
        # 來自底部的方向: k=2,5,6
        for i in range(1, self.nx + 1):
            for k in ti.static([2, 5, 6]):
                ik = inv[k]
                if bc == 1:  # Dirichlet
                    g[i, 1][k] = -g[i, 1][ik] + 2.0 * w[k] * T_wall
                else:        # Neumann
                    g[i, 1][k] = g[i, 1][ik]

    @ti.kernel
    def _apply_top_bc(self, g: ti.template()):
        """
        頂部壁面 BC（壁在 j=ny+1，流體在 j=ny）

        未知方向（從壁面進入流體）: k=4(↓), k=7(↙), k=8(↘)
        """
        w   = self.thermal.w
        inv = self.thermal.inv
        T_wall = self.T_top[None]
        bc    = self.bc_top[None]
        ny    = self.ny
        for i in range(1, self.nx + 1):
            for k in ti.static([4, 7, 8]):
                ik = inv[k]
                if bc == 1:  # Dirichlet
                    g[i, ny][k] = -g[i, ny][ik] + 2.0 * w[k] * T_wall
                else:        # Neumann
                    g[i, ny][k] = g[i, ny][ik]

    @ti.kernel
    def _apply_left_bc(self, g: ti.template()):
        """
        左壁 BC（壁在 i=0，流體在 i=1）

        未知方向: k=1(→), k=5(↗), k=8(↘)
        """
        w   = self.thermal.w
        inv = self.thermal.inv
        T_wall = self.T_left[None]
        bc    = self.bc_left[None]
        for j in range(1, self.ny + 1):
            for k in ti.static([1, 5, 8]):
                ik = inv[k]
                if bc == 1:
                    g[1, j][k] = -g[1, j][ik] + 2.0 * w[k] * T_wall
                else:
                    g[1, j][k] = g[1, j][ik]

    @ti.kernel
    def _apply_right_bc(self, g: ti.template()):
        """
        右壁 BC（壁在 i=nx+1，流體在 i=nx）

        未知方向: k=3(←), k=6(↖), k=7(↙)
        """
        w   = self.thermal.w
        inv = self.thermal.inv
        T_wall = self.T_right[None]
        bc    = self.bc_right[None]
        nx    = self.nx
        for j in range(1, self.ny + 1):
            for k in ti.static([3, 6, 7]):
                ik = inv[k]
                if bc == 1:
                    g[nx, j][k] = -g[nx, j][ik] + 2.0 * w[k] * T_wall
                else:
                    g[nx, j][k] = g[nx, j][ik]
```

### Step 4: 執行確認通過

```bash
uv run python -m pytest tests/test_thermal_module.py -v
```
預期：7 tests PASSED

### Step 5: Commit

```bash
git add src/lbm_taichi/core/thermal_module.py tests/test_thermal_module.py
git commit -m "feat: add ThermalBoundaryConditions (anti-bounce-back Dirichlet, Neumann adiabatic)"
```

---

## Task 5: Nusselt 數計算

**Files:**
- Modify: `src/lbm_taichi/core/thermal_module.py`

### Step 1: 新增測試

```python
def test_nusselt_pure_conduction():
    """
    純導熱（Ra → 0，靜止流場）穩態下 Nu ≈ 1.0
    """
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions

    ny = 32
    solver = LBMSolver(nx=32, ny=ny, re=10000.0, u_ref=0.01)
    solver.u.fill(0.0)  # 靜止流場

    thermal = ThermalModule(solver, Pr=0.71)
    thermal.init_temperature(T_bot=1.0, T_top=0.0)

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location='bottom')
    tbc.add_cold_wall(T_cold=0.0, location='top')
    tbc.add_adiabatic_wall('left')
    tbc.add_adiabatic_wall('right')

    # 跑到穩態（純導熱，不需很多步）
    for step in range(5000):
        g_src = thermal.g if step % 2 == 0 else thermal.g_new
        g_dst = thermal.g_new if step % 2 == 0 else thermal.g
        thermal.step(g_src, g_dst)
        tbc.apply(g_dst)

    thermal._update_temperature(g_dst)
    nu = thermal.get_nusselt(T_bot=1.0, T_top=0.0)
    # 純導熱 Nu ≈ 1.0（允許 ±15% 誤差）
    assert abs(nu - 1.0) < 0.15, f"Nu = {nu:.3f}, expected ≈ 1.0"
```

### Step 2: 執行確認失敗

```bash
uv run python -m pytest tests/test_thermal_module.py::test_nusselt_pure_conduction -v
```
預期：`AttributeError: 'ThermalModule' object has no attribute 'get_nusselt'`

### Step 3: 實作 Nu 計算

在 `ThermalModule` 類別內新增：

```python
    def get_nusselt(self, T_bot: float, T_top: float) -> float:
        """
        計算 Nusselt 數

        Why 用中間截面梯度?
        - 避開壁面 BC 數值影響
        - 穩態下各截面熱通量應相等

        Nu = -H * mean(∂T/∂y)|_{j=ny//2} / ΔT
        純導熱: ∂T/∂y = -ΔT/H → Nu = 1
        """
        self.nu_sum[None] = 0.0
        self._compute_nu_kernel()
        mean_grad = self.nu_sum[None] / self.nx
        delta_T = T_bot - T_top
        H = self.ny
        return float(-H * mean_grad / delta_T)

    @ti.kernel
    def _compute_nu_kernel(self):
        j_mid = self.ny // 2
        for i in range(1, self.nx + 1):
            dT_dy = (self.T[i, j_mid + 1] - self.T[i, j_mid - 1]) / 2.0
            ti.atomic_add(self.nu_sum[None], dT_dy)
```

### Step 4: 執行確認通過

```bash
uv run python -m pytest tests/test_thermal_module.py -v
```
預期：8 tests PASSED

### Step 5: 更新 `__init__.py` 匯出

在 `src/lbm_taichi/core/__init__.py` 新增：

```python
from .thermal_module import ThermalModule, ThermalBoundaryConditions
```

並在 `__all__` 列表中加入：
```python
"ThermalModule",
"ThermalBoundaryConditions",
```

### Step 6: Commit

```bash
git add src/lbm_taichi/core/thermal_module.py src/lbm_taichi/core/__init__.py tests/test_thermal_module.py
git commit -m "feat: add Nusselt number calculation and export ThermalModule"
```

---

## Task 6: Rayleigh-Bénard 主腳本

**Files:**
- Create: `examples/rayleigh_benard.py`

### Step 1: 無需測試，直接實作（整合腳本）

```python
# examples/rayleigh_benard.py
"""
Rayleigh-Bénard 對流模擬
=========================

What: 底部加熱、頂部冷卻的封閉腔體對流
Why:  標準熱對流 benchmark，驗證 DDF+Boussinesq 的 Nu-Ra 關係

物理設定:
    Ra = g·β·ΔT·H³ / (ν·κ)
    Pr = ν / κ

驗收標準:
    Ra = 1e5 → Nu ≈ 4.5–5.0
    Ra = 1e6 → Nu ≈ 8.0–10.0
"""

import taichi as ti
import numpy as np
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lbm_taichi.core import LBMSolver, BoundaryConditions, Diagnostics
from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions


def compute_lbm_params(Ra: float, Pr: float, ny: int, u_ref: float):
    """
    由 Ra、Pr 推導格子 LBM 參數

    Why 用 Re_eff?
        Re_eff = sqrt(Ra/Pr) 是自然對流的特徵 Reynolds 數
    """
    H = ny
    Re_eff = (Ra / Pr) ** 0.5
    nu = u_ref * H / Re_eff
    kappa = nu / Pr
    # 確保 tau > 0.5（穩定性下界）
    assert nu > 1e-4, f"nu={nu:.2e} 太小，增加 u_ref 或減少 ny"
    # 浮力強度：由 Ra 定義反推 g_lbm
    # Ra = g·β·ΔT·H³/(ν·κ)，β=1，ΔT=1
    g_lbm = Ra * nu * kappa / (H ** 3)
    return nu, kappa, g_lbm, Re_eff


def run_rayleigh_benard(
    ny: int = 64,
    Ra: float = 1e5,
    Pr: float = 0.71,
    u_ref: float = 0.05,
    aspect: float = 2.0,
    steps: int = 100000,
    interval: int = 2000,
    tol: float = 1e-5,
    output_dir: str = "output_rb",
):
    """
    執行 Rayleigh-Bénard 模擬

    Args:
        ny:         Y 方向格點數（腔體高度）
        Ra:         Rayleigh 數
        Pr:         Prandtl 數（0.71=空氣, 7=水）
        u_ref:      LBM 參考速度（控制 Ma 數）
        aspect:     長寬比 nx = aspect * ny
        steps:      最大步數
        interval:   儲存間隔
        tol:        動量收斂容差
        output_dir: 輸出目錄
    """
    nx = int(aspect * ny)
    H = ny

    nu, kappa, g_lbm, Re_eff = compute_lbm_params(Ra, Pr, ny, u_ref)
    tau_f = 0.5 + 3.0 * nu
    tau_g = 0.5 + 3.0 * kappa

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("         RAYLEIGH-BÉNARD CONVECTION SIMULATION")
    print("=" * 65)
    print(f"  Grid:        {nx} x {ny}")
    print(f"  Ra:          {Ra:.2e}")
    print(f"  Pr:          {Pr:.2f}")
    print(f"  Re_eff:      {Re_eff:.1f}")
    print(f"  ν (nu):      {nu:.6f}    τ_f = {tau_f:.4f}")
    print(f"  κ (kappa):   {kappa:.6f}    τ_g = {tau_g:.4f}")
    print(f"  g_lbm:       {g_lbm:.2e}")
    print(f"  u_ref:       {u_ref:.3f}   Ma ≈ {u_ref * 1.732:.3f}")
    print("=" * 65)

    # === 求解器初始化 ===
    solver = LBMSolver(
        nx=nx, ny=ny,
        re=Re_eff, u_ref=u_ref,
        length_scale=float(H),
        cs=0.0,  # RB 對流不需要 Smagorinsky（低 Re_eff）
    )

    # === 熱模組 ===
    thermal = ThermalModule(
        solver, Pr=Pr, beta=1.0, g_gravity=g_lbm, T_ref=0.5
    )
    thermal.init_temperature(T_bot=1.0, T_top=0.0)
    thermal.register_with_solver()  # 每步自動更新浮力

    # === 流場邊界條件 ===
    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall('top')
    bc.add_no_slip_wall('bottom')
    bc.add_periodic_boundary('x')

    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # === 溫度邊界條件 ===
    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location='bottom')
    tbc.add_cold_wall(T_cold=0.0, location='top')
    # 左右：週期邊界（不設溫度 BC，保持平衡分佈）

    # 初始施加溫度 BC
    tbc.apply(thermal.g)
    tbc.apply(thermal.g_new)

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir)
    solver.reset_mass_baseline()

    print(f"\n{'step':>8} | {'mom_res':>10} | {'mass_err':>10} | {'T_mid':>7} | {'Nu':>6} | {'step/s':>8}")
    print("-" * 65)

    global_start = time.time()
    nu_history = []

    for step in range(1, steps + 1):
        f_src = solver.f     if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f
        g_src = thermal.g    if step % 2 == 1 else thermal.g_new
        g_dst = thermal.g_new if step % 2 == 1 else thermal.g

        # ① 流場推進（force_field_updater 在內部自動呼叫 compute_buoyancy）
        solver.step(f_src, f_dst)

        # ② 溫度場推進
        thermal.step(g_src, g_dst)

        # ③ 溫度邊界條件
        tbc.apply(g_dst)

        if step % 200 == 0:
            solver._update_macro(f_dst)
            solver._update_diagnostics()
            thermal._update_temperature(g_dst)
            ti.sync()

            elapsed = time.time() - global_start
            speed = step / elapsed

            mom_res = max(solver.mom_res_x[None], solver.mom_res_y[None])
            mass_err = abs(solver.mass_residual[None])
            T_mid = float(thermal.T.to_numpy()[nx // 2, ny // 2])
            nu_val = thermal.get_nusselt(T_bot=1.0, T_top=0.0)

            print(f"{step:>8} | {mom_res:>10.3e} | {mass_err:>10.3e} | "
                  f"{T_mid:>7.4f} | {nu_val:>6.2f} | {speed:>8.1f}")

            if step % 2000 == 0:
                nu_history.append((step, nu_val))

        if step % interval == 0:
            T_np = thermal.T.to_numpy()[1:nx+1, 1:ny+1]
            u_np = solver.u.to_numpy()[1:nx+1, 1:ny+1]
            rho_np = solver.rho.to_numpy()[1:nx+1, 1:ny+1]
            state = {
                'T': T_np, 'u': u_np, 'rho': rho_np,
                'step': step, 'Ra': Ra, 'Pr': Pr,
            }
            np.save(os.path.join(output_dir, f"state_{step:07d}.npy"), state)

    total_time = time.time() - global_start
    print(f"\n--- Completed in {total_time:.1f}s ---")

    # 最終 Nu
    solver._update_macro(f_dst)
    thermal._update_temperature(g_dst)
    final_nu = thermal.get_nusselt(T_bot=1.0, T_top=0.0)
    print(f"Final Nu = {final_nu:.3f}  (Ra={Ra:.1e}, Pr={Pr:.2f})")
    print(f"Expected: Nu ≈ 4.5-5.0 for Ra=1e5, Nu ≈ 8-10 for Ra=1e6")

    np.save(os.path.join(output_dir, "nu_history.npy"), nu_history)
    return final_nu


def main():
    parser = argparse.ArgumentParser(description="Rayleigh-Bénard Convection")
    parser.add_argument('--ny',       type=int,   default=64,       help='Y resolution')
    parser.add_argument('--Ra',       type=float, default=1e5,      help='Rayleigh number')
    parser.add_argument('--Pr',       type=float, default=0.71,     help='Prandtl number')
    parser.add_argument('--u_ref',    type=float, default=0.05,     help='Reference velocity')
    parser.add_argument('--aspect',   type=float, default=2.0,      help='Aspect ratio nx/ny')
    parser.add_argument('--steps',    type=int,   default=100000,   help='Max steps')
    parser.add_argument('--interval', type=int,   default=2000,     help='Save interval')
    parser.add_argument('--tol',      type=float, default=1e-5,     help='Convergence tolerance')
    parser.add_argument('--output',   type=str,   default='output_rb', help='Output directory')
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_rayleigh_benard(
        ny=args.ny, Ra=args.Ra, Pr=args.Pr, u_ref=args.u_ref,
        aspect=args.aspect, steps=args.steps, interval=args.interval,
        tol=args.tol, output_dir=args.output,
    )


if __name__ == "__main__":
    main()
```

### Step 2: 快速冒煙測試（小網格確認不崩潰）

```bash
uv run python examples/rayleigh_benard.py --ny 32 --Ra 1e4 --steps 1000 --output output_rb_smoke
```
預期：正常輸出表格，無 NaN，1000 步完成

### Step 3: Commit

```bash
git add examples/rayleigh_benard.py
git commit -m "feat: add Rayleigh-Benard simulation example with DDF+Boussinesq"
```

---

## Task 7: Nu 數驗證

### Step 1: 執行 Ra=1e5 驗證

```bash
uv run python examples/rayleigh_benard.py \
    --ny 64 --Ra 1e5 --Pr 0.71 --u_ref 0.05 \
    --steps 150000 --interval 5000 \
    --output output_rb_1e5
```

**驗收**: 最終輸出 `Final Nu = X.XX`，期待 **4.5 ≤ Nu ≤ 5.5**

### Step 2: 執行 Ra=1e6 驗證

```bash
uv run python examples/rayleigh_benard.py \
    --ny 96 --Ra 1e6 --Pr 0.71 --u_ref 0.05 \
    --steps 300000 --interval 10000 \
    --output output_rb_1e6
```

**驗收**: 最終 `Final Nu`，期待 **7.5 ≤ Nu ≤ 11.0**

### Step 3: 若 Nu 偏差 > 20%

常見原因與調整：

| 症狀 | 原因 | 解決 |
|------|------|------|
| Nu 太低（< 4.0，Ra=1e5）| 未達穩態 | 增加 steps |
| Nu 不穩定震盪 | tau_g 太小（τ < 0.6）| 降低 Ra 或增加 ny |
| NaN 出現 | tau_f 太小 | 降低 u_ref |
| Nu ≈ 1.0 | 浮力未作用 | 確認 register_with_solver 呼叫 |

### Step 4: Commit

```bash
git add docs/plans/2026-02-26-rayleigh-benard-implementation.md
git commit -m "docs: add Rayleigh-Benard implementation plan and validation results"
```

---

## 快速參考

### 驗收標準

| Ra | Pr | ny | Nu 預期 | 參考文獻 |
|----|----|----|---------|---------|
| 1e4 | 0.71 | 32 | ≈ 2.4 | de Vahl Davis (1983) |
| 1e5 | 0.71 | 64 | ≈ 4.5–5.0 | Grossmann-Lohse |
| 1e6 | 0.71 | 96 | ≈ 8–10 | Grossmann-Lohse |

### 穩定性限制（Metal float32）

- `tau_g > 0.52`（熱鬆弛時間過小會導致振盪）
- `tau_f > 0.52`（動量鬆弛同上）
- `u_ref * sqrt(3) < 0.3`（Ma 數限制）
- `Ra > 1e7` 建議增加解析度至 ny ≥ 128
