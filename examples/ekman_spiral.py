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

            self.layer_u_mean[k] = u_sum / ti.cast(count, ti.f32)
            self.layer_v_mean[k] = v_sum / ti.cast(count, ti.f32)
            self.layer_ke[k] = ke_sum / ti.cast(count, ti.f32)


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
