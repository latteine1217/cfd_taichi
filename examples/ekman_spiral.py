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

    @ti.kernel
    def _compute_coriolis_force_layer(
        self,
        k: ti.i32,
        u_field: ti.template(),
    ):
        """
        計算單層的科氏力

        Physics:
            F_coriolis = (f·v, -f·u)

            北半球（f > 0）：
            - 東向流（u > 0）產生南向力（Fy < 0）
            - 北向流（v > 0）產生東向力（Fx > 0）
            → 順時針偏轉

        Args:
            k: 層索引
            u_field: 速度場（來自 LBMSolver.u）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            # 從 LBMSolver 讀取速度（包含 ghost cells）
            u_val = u_field[i + 1, j + 1][0]
            v_val = u_field[i + 1, j + 1][1]

            # 科氏力
            self.coriolis_fx[k, i, j] = self.f * v_val
            self.coriolis_fy[k, i, j] = -self.f * u_val

    def _compute_coriolis_force(self):
        """計算所有層的科氏力（呼叫 kernel）"""
        for k in range(self.n_layers):
            self._compute_coriolis_force_layer(k, self.layers[k].u)

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
        self._set_wind_stress()

        # 中間界面：黏性剪應力
        for k in range(1, self.n_layers):
            self._compute_viscous_stress_layer(
                k, self.layers[k-1].u, self.layers[k].u
            )

        # 底層界面：底摩擦
        self._set_bottom_friction()

    @ti.kernel
    def _set_wind_stress(self):
        """設定頂層風應力"""
        for i, j in ti.ndrange(self.nx, self.ny):
            self.shear_stress_x[0, i, j] = self.tau_wind[0]
            self.shear_stress_y[0, i, j] = self.tau_wind[1]

    @ti.kernel
    def _compute_viscous_stress_layer(
        self, k: ti.i32, u_upper: ti.template(), u_lower: ti.template()
    ):
        """計算第 k 界面的黏性剪應力"""
        for i, j in ti.ndrange(self.nx, self.ny):
            u_u = u_upper[i + 1, j + 1][0]
            v_u = u_upper[i + 1, j + 1][1]

            u_l = u_lower[i + 1, j + 1][0]
            v_l = u_lower[i + 1, j + 1][1]

            du_dz = (u_u - u_l) / self.dz
            dv_dz = (v_u - v_l) / self.dz

            self.shear_stress_x[k, i, j] = self.rho * self.nu_v * du_dz
            self.shear_stress_y[k, i, j] = self.rho * self.nu_v * dv_dz

    @ti.kernel
    def _set_bottom_friction(self):
        """設定底層摩擦"""
        for i, j in ti.ndrange(self.nx, self.ny):
            u_bot = self.layers[self.n_layers - 1].u[i + 1, j + 1][0]
            v_bot = self.layers[self.n_layers - 1].u[i + 1, j + 1][1]

            # 線性拖曳：τ = -ρ r u dz
            self.shear_stress_x[self.n_layers, i, j] = -self.rho * self.r_bottom * u_bot * self.dz
            self.shear_stress_y[self.n_layers, i, j] = -self.rho * self.r_bottom * v_bot * self.dz

    def _compute_total_force(self):
        """
        計算總外力 = 科氏力 + 剪應力梯度

        Physics:
            F_total = F_coriolis + (τ_top - τ_bottom) / (ρ dz)
        """
        for k in range(self.n_layers):
            self._compute_force_layer(k)

    @ti.kernel
    def _compute_force_layer(self, k: ti.i32):
        """計算第 k 層的總外力"""
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
        for k in range(self.n_layers):
            # 將外力注入 LBM solver
            self._inject_force_to_layer(k, self.layers[k].u)

            # LBM step
            if k % 2 == 0:
                self.layers[k].step(self.layers[k].f, self.layers[k].f_new)
            else:
                self.layers[k].step(self.layers[k].f_new, self.layers[k].f)

    @ti.kernel
    def _inject_force_to_layer(self, k: ti.i32, u_field: ti.template()):
        """
        將外力注入第 k 層的 LBM 求解器

        Method: 修改速度場（Guo's forcing scheme 簡化版）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            fx = self.force_x[k, i, j]
            fy = self.force_y[k, i, j]

            # 施加外力（動量更新）
            # Δu = F * Δt（LBM 中 Δt = 1）
            u_field[i + 1, j + 1][0] += fx
            u_field[i + 1, j + 1][1] += fy

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
            # 計算質量誤差：|M(t) - M(0)| / M(0)
            mass_current = solver.total_mass[None]
            mass_initial = solver.initial_mass[None]
            if mass_initial > 1e-12:
                errors[k] = abs(mass_current - mass_initial) / mass_initial
            else:
                errors[k] = 0.0
        return errors

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

    def _compute_layer_diagnostics(self):
        """計算每層的空間平均診斷量"""
        for k in range(self.n_layers):
            self._compute_single_layer_diagnostics(k, self.layers[k].u)

    @ti.kernel
    def _compute_single_layer_diagnostics(self, k: ti.i32, u_field: ti.template()):
        """計算單層的空間平均診斷量"""
        u_sum = 0.0
        v_sum = 0.0
        ke_sum = 0.0
        count = 0

        for i, j in ti.ndrange(self.nx, self.ny):
            u_val = u_field[i + 1, j + 1][0]
            v_val = u_field[i + 1, j + 1][1]

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

    # 初始化各層的質量基準
    for k in range(solver.n_layers):
        f_src = solver.layers[k].f if k % 2 == 0 else solver.layers[k].f_new
        solver.layers[k]._update_macro(f_src)
        solver.layers[k]._update_diagnostics()
        solver.layers[k].initial_mass[None] = solver.layers[k].total_mass[None]

    # 測試診斷
    print("\n=== 診斷系統測試 ===")
    print("| step  | time(hr) | u_surf | angle_surf | u_bot  | KE_total | transport_angle | mass_err |")
    print("|-------|----------|--------|------------|--------|----------|-----------------|----------|")

    solver.print_diagnostics(0, 0.0)

    # 測試多步運行
    print("\n=== 短時間演化測試 ===")
    for step in range(1, 11):
        solver.step()
        if step % 5 == 0:
            solver.print_diagnostics(step, step * 1.0)  # 假設 dt=1 hr

    print("\n✅ 診斷系統測試完成")
