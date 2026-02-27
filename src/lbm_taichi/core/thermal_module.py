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
          再將浮力 F_y = rho * g * beta * (T - T_ref) 注入速度求解器
    """

    def __init__(self, solver, Pr=0.71, beta=1.0, g_gravity=0.0, T_ref=0.5):
        """
        Args:
            solver: LBMSolver 實例（提供 nx, ny, nx_g, ny_g, nu）
            Pr: Prandtl 數（空氣 0.71，水 7.0）
            beta: 熱膨脹係數（lattice units）
            g_gravity: 重力加速度（lattice units, 向下為負）
            T_ref: 參考溫度（Boussinesq 線性化中心）
        """
        assert solver.ny >= 2, "ThermalModule 需要 ny >= 2"

        self.solver = solver
        self.nx = solver.nx
        self.ny = solver.ny
        self.nx_g = solver.nx_g
        self.ny_g = solver.ny_g
        self.Pr = Pr
        self.beta = beta
        self.g_gravity = g_gravity
        self.T_ref = T_ref

        # 熱擴散率 kappa = nu / Pr，溫度鬆弛時間 tau_g = 0.5 + 3*kappa
        self.kappa = solver.nu / Pr
        self.tau_g = 0.5 + 3.0 * self.kappa
        if self.tau_g < 0.502:
            raise ValueError(
                f"tau_g={self.tau_g:.4f} 過小（< 0.502）。"
                f"請降低 Re 或提高 Pr。kappa={self.kappa:.6f}"
            )

        # D2Q9 常數：直接引用 solver 的 field，避免重複分配 GPU buffer
        self.w = solver.w
        self.e = solver.e
        self.inv = solver.inv

        # 溫度分佈函數 g 與宏觀溫度場 T
        self.g = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.g_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.T = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        # 診斷用累加器
        # TODO: Nu 數計算用（Task 5 實作 get_nusselt()）
        self.nu_sum = ti.field(dtype=ti.f32, shape=())

        # 以均勻溫度 T=0.5 填入平衡態
        self._fill_equilibrium(0.5)

    @ti.kernel
    def _fill_equilibrium(self, T_init: ti.f32):
        """以均勻溫度填入平衡態分佈 g_eq_k = w_k * T"""
        for i, j in self.g:
            self.T[i, j] = T_init
            for k in ti.static(range(9)):
                self.g[i, j][k] = self.w[k] * T_init
                self.g_new[i, j][k] = self.w[k] * T_init

    def init_temperature(self, T_bot: float, T_top: float):
        """
        線性溫度初始化：底部 T_bot，頂部 T_top

        What: 設定純導熱穩態解作為初始條件
        Why:  比均勻溫度初始化更接近物理穩態，減少初始暫態
        """
        self._init_temperature_kernel(float(T_bot), float(T_top))

    @ti.kernel
    def _init_temperature_kernel(self, T_bot: ti.f32, T_top: ti.f32):
        """
        kernel: 以線性插值設定溫度場及對應的平衡態分佈

        ghost cells (j=0, j=ny+1) 分別設為 T_bot, T_top
        內部 (1 <= j <= ny) 線性插值
        """
        for i, j in self.g:
            T_loc = T_bot  # 預設值（ghost cell 底部）
            if (j >= 1) and (j <= self.ny):
                y_frac = ti.cast(j - 1, ti.f32) / ti.cast(self.ny - 1, ti.f32)
                T_loc = T_bot + (T_top - T_bot) * y_frac
            elif j > self.ny:
                T_loc = T_top
            self.T[i, j] = T_loc
            for k in ti.static(range(9)):
                self.g[i, j][k] = self.w[k] * T_loc
                self.g_new[i, j][k] = self.w[k] * T_loc
