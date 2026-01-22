"""
邊界條件模組
============

提供各種 LBM 邊界條件的實作：
- Zou-He 速度邊界（固定速度入口）
- Zou-He 壓力邊界（固定壓力出口）
- Free-Slip 壁面（鏡面反射，無摩擦）
- No-Slip 壁面（Bounce-Back）
- Moving Wall（運動壁面，如 Lid-Driven Cavity 的上蓋）

Why Zou-He?
- 非平衡外推法，質量守恆
- 適用於入口/出口邊界
- 數值穩定性好

Why Bounce-Back?
- 最簡單的固體邊界實現
- 自動滿足 No-Slip 條件
- 與 LBM 格子完美配合
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class BoundaryConditions:
    """
    統一的邊界條件管理類別

    Why 這個設計?
    - 可組合：不同 case 可自由組合所需邊界條件
    - 可追蹤：清楚知道哪些邊界條件被啟用
    - 可驗證：每個邊界條件獨立，易於測試
    """

    def __init__(self, solver):
        """
        Args:
            solver: LBMSolver 實例
        """
        self.solver = solver
        self.nx = solver.nx
        self.ny = solver.ny

        # 儲存邊界條件參數
        self.u_inlet = ti.field(dtype=ti.f32, shape=())
        self.rho_outlet = ti.field(dtype=ti.f32, shape=())
        self.u_wall_field = ti.field(dtype=ti.f32, shape=self.nx)  # 用於 Moving Wall

    def add_zou_he_velocity_inlet(self, u_in: float, location: str = 'left'):
        """
        添加 Zou-He 固定速度入口邊界

        Why Zou-He?
        - 通過非平衡外推法重建未知分佈函數
        - 保證質量守恆：sum(f_i) = rho
        - 滿足固定速度：sum(f_i * e_i) / rho = u_target

        Args:
            u_in: 入口速度 (lattice units)
            location: 'left', 'right', 'top', 'bottom'
        """
        self.u_inlet[None] = u_in

        if location == 'left':
            self.solver.add_boundary_condition(self._zou_he_inlet_left, "Zou-He Inlet (Left)")
        elif location == 'right':
            self.solver.add_boundary_condition(self._zou_he_inlet_right, "Zou-He Inlet (Right)")
        else:
            raise NotImplementedError(f"Location '{location}' not implemented for velocity inlet")

    def add_zou_he_pressure_outlet(self, rho_out: float = 1.0, location: str = 'right'):
        """
        添加 Zou-He 固定壓力出口邊界

        Why 固定壓力?
        - 允許流體自由流出
        - 保證全局質量守恆
        - 避免出口反射波

        Args:
            rho_out: 出口密度（對應壓力）
            location: 'left', 'right', 'top', 'bottom'
        """
        self.rho_outlet[None] = rho_out

        if location == 'right':
            self.solver.add_boundary_condition(self._zou_he_outlet_right, "Zou-He Outlet (Right)")
        elif location == 'left':
            self.solver.add_boundary_condition(self._zou_he_outlet_left, "Zou-He Outlet (Left)")
        else:
            raise NotImplementedError(f"Location '{location}' not implemented for pressure outlet")

    def add_free_slip_wall(self, location: str):
        """
        添加 Free-Slip 壁面（鏡面反射）

        Why Free-Slip?
        - 模擬無摩擦壁面（風洞外壁）
        - 法向速度為零，切向速度保留
        - 鏡面反射：垂直分量反向，平行分量保留

        Args:
            location: 'top', 'bottom', 'left', 'right'
        """
        if location == 'top':
            self.solver.add_boundary_condition(self._free_slip_top, "Free-Slip (Top)")
        elif location == 'bottom':
            self.solver.add_boundary_condition(self._free_slip_bottom, "Free-Slip (Bottom)")
        elif location == 'left':
            self.solver.add_boundary_condition(self._free_slip_left, "Free-Slip (Left)")
        elif location == 'right':
            self.solver.add_boundary_condition(self._free_slip_right, "Free-Slip (Right)")
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_moving_wall(self, velocity_profile: np.ndarray, location: str = 'top'):
        """
        添加運動壁面（用於 Lid-Driven Cavity）

        Why 需要速度 profile?
        - 避免角點奇異性（速度不連續）
        - 平滑速度變化提高數值穩定性
        - Lid-Driven Cavity 的標準處理方式

        Args:
            velocity_profile: (nx,) 陣列，每個 x 位置的壁面速度
            location: 'top', 'bottom'
        """
        if velocity_profile.shape[0] != self.nx:
            raise ValueError(f"Velocity profile length {velocity_profile.shape[0]} != nx {self.nx}")

        self.u_wall_field.from_numpy(velocity_profile.astype(np.float32))

        if location == 'top':
            self.solver.add_boundary_condition(self._moving_wall_top, "Moving Wall (Top)")
        elif location == 'bottom':
            self.solver.add_boundary_condition(self._moving_wall_bottom, "Moving Wall (Bottom)")
        else:
            raise NotImplementedError(f"Location '{location}' not implemented for moving wall")

    # ==================== Zou-He 入口 ====================

    @ti.kernel
    def _zou_he_inlet_left(self, f_dst: ti.template()):
        """
        左邊界 Zou-He 固定速度入口

        已知：f0, f2, f3, f4, f6, f7 (從內部流傳來的)
        未知：f1, f5, f8 (需要重建)

        質量守恆：rho = (f0+f2+f4 + 2*(f3+f6+f7)) / (1 - u_in)
        動量守恆：rho*u_in = f1-f3 + f5-f6-f7+f8
        """
        for j in range(self.ny):
            if self.solver.mask[0, j] == 0:  # 流體節點
                f0 = f_dst[0, j][0]
                f2 = f_dst[0, j][2]
                f3 = f_dst[0, j][3]
                f4 = f_dst[0, j][4]
                f6 = f_dst[0, j][6]
                f7 = f_dst[0, j][7]

                u_in = self.u_inlet[None]
                rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - u_in)

                # 重建未知分佈函數
                f_dst[0, j][1] = f3 + (2.0/3.0) * rho_in * u_in
                f_dst[0, j][5] = f7 - 0.5*(f2 - f4) + (1.0/6.0) * rho_in * u_in
                f_dst[0, j][8] = f6 + 0.5*(f2 - f4) + (1.0/6.0) * rho_in * u_in

    @ti.kernel
    def _zou_he_inlet_right(self, f_dst: ti.template()):
        """右邊界 Zou-He 固定速度入口"""
        for j in range(self.ny):
            if self.solver.mask[self.nx-1, j] == 0:
                f0 = f_dst[self.nx-1, j][0]
                f1 = f_dst[self.nx-1, j][1]
                f2 = f_dst[self.nx-1, j][2]
                f4 = f_dst[self.nx-1, j][4]
                f5 = f_dst[self.nx-1, j][5]
                f8 = f_dst[self.nx-1, j][8]

                u_in = -self.u_inlet[None]  # 向左流入
                rho_in = (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / (1.0 + u_in)

                f_dst[self.nx-1, j][3] = f1 - (2.0/3.0) * rho_in * u_in
                f_dst[self.nx-1, j][7] = f5 + 0.5*(f2 - f4) - (1.0/6.0) * rho_in * u_in
                f_dst[self.nx-1, j][6] = f8 - 0.5*(f2 - f4) - (1.0/6.0) * rho_in * u_in

    # ==================== Zou-He 出口 ====================

    @ti.kernel
    def _zou_he_outlet_right(self, f_dst: ti.template()):
        """
        右邊界 Zou-He 固定壓力出口

        Why 固定 rho?
        - 出口邊界需要低壓（大氣壓）
        - rho = 1.0 對應標準大氣壓
        - 速度從質量守恆推導
        """
        for j in range(self.ny):
            if self.solver.mask[self.nx-1, j] == 0:
                f0 = f_dst[self.nx-1, j][0]
                f1 = f_dst[self.nx-1, j][1]
                f2 = f_dst[self.nx-1, j][2]
                f4 = f_dst[self.nx-1, j][4]
                f5 = f_dst[self.nx-1, j][5]
                f8 = f_dst[self.nx-1, j][8]

                rho_out = self.rho_outlet[None]
                u_x = -1.0 + (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / rho_out

                f_dst[self.nx-1, j][3] = f1 - (2.0/3.0) * rho_out * u_x
                f_dst[self.nx-1, j][7] = f5 - (1.0/6.0) * rho_out * u_x + 0.5*(f2 - f4)
                f_dst[self.nx-1, j][6] = f8 - (1.0/6.0) * rho_out * u_x - 0.5*(f2 - f4)

    @ti.kernel
    def _zou_he_outlet_left(self, f_dst: ti.template()):
        """左邊界 Zou-He 固定壓力出口"""
        for j in range(self.ny):
            if self.solver.mask[0, j] == 0:
                f0 = f_dst[0, j][0]
                f2 = f_dst[0, j][2]
                f3 = f_dst[0, j][3]
                f4 = f_dst[0, j][4]
                f6 = f_dst[0, j][6]
                f7 = f_dst[0, j][7]

                rho_out = self.rho_outlet[None]
                u_x = 1.0 - (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / rho_out

                f_dst[0, j][1] = f3 + (2.0/3.0) * rho_out * u_x
                f_dst[0, j][5] = f7 + (1.0/6.0) * rho_out * u_x - 0.5*(f2 - f4)
                f_dst[0, j][8] = f6 + (1.0/6.0) * rho_out * u_x + 0.5*(f2 - f4)

    # ==================== Free-Slip 壁面 ====================

    @ti.kernel
    def _free_slip_bottom(self, f_dst: ti.template()):
        """
        底部 Free-Slip 壁面（鏡面反射）

        Why Specular Reflection?
        - 法向速度反向：uy -> -uy
        - 切向速度保留：ux -> ux
        - 零法向應力，零摩擦

        映射關係（垂直方向鏡像）：
        - f2 (0,1) <- f4 (0,-1)
        - f5 (1,1) <- f8 (1,-1)
        - f6 (-1,1) <- f7 (-1,-1)
        """
        for i in range(self.nx):
            if self.solver.mask[i, 0] == 0:
                f_dst[i, 0][2] = f_dst[i, 0][4]
                f_dst[i, 0][5] = f_dst[i, 0][8]
                f_dst[i, 0][6] = f_dst[i, 0][7]

    @ti.kernel
    def _free_slip_top(self, f_dst: ti.template()):
        """頂部 Free-Slip 壁面"""
        for i in range(self.nx):
            if self.solver.mask[i, self.ny-1] == 0:
                f_dst[i, self.ny-1][4] = f_dst[i, self.ny-1][2]
                f_dst[i, self.ny-1][7] = f_dst[i, self.ny-1][6]
                f_dst[i, self.ny-1][8] = f_dst[i, self.ny-1][5]

    @ti.kernel
    def _free_slip_left(self, f_dst: ti.template()):
        """左側 Free-Slip 壁面"""
        for j in range(self.ny):
            if self.solver.mask[0, j] == 0:
                f_dst[0, j][1] = f_dst[0, j][3]
                f_dst[0, j][5] = f_dst[0, j][6]
                f_dst[0, j][8] = f_dst[0, j][7]

    @ti.kernel
    def _free_slip_right(self, f_dst: ti.template()):
        """右側 Free-Slip 壁面"""
        for j in range(self.ny):
            if self.solver.mask[self.nx-1, j] == 0:
                f_dst[self.nx-1, j][3] = f_dst[self.nx-1, j][1]
                f_dst[self.nx-1, j][6] = f_dst[self.nx-1, j][5]
                f_dst[self.nx-1, j][7] = f_dst[self.nx-1, j][8]

    # ==================== Moving Wall ====================

    @ti.kernel
    def _moving_wall_top(self, f_dst: ti.template()):
        """
        頂部運動壁面（Lid-Driven Cavity）

        Why 需要特殊處理?
        - 壁面本身有速度 u_wall
        - 需要通過 Bounce-Back 施加動量
        - 平滑速度 profile 避免角點奇異性

        未知：f4, f7, f8 (向下)
        已知：f2, f5, f6 (從下方來)

        Bounce-Back with wall velocity:
        f4 = f2 - (2/3) * rho * u_wall
        """
        j = self.ny - 1
        for i in range(self.nx):
            if self.solver.mask[i, j] == 0:
                u_wall = self.u_wall_field[i]

                # 計算當前密度（近似）
                f_vec = f_dst[i, j]
                rho_local = 0.0
                for k in ti.static(range(9)):
                    rho_local += f_vec[k]

                # Bounce-Back with wall velocity (Zou-He style)
                f0 = f_dst[i, j][0]
                f1 = f_dst[i, j][1]
                f2 = f_dst[i, j][2]
                f3 = f_dst[i, j][3]
                f5 = f_dst[i, j][5]
                f6 = f_dst[i, j][6]

                # 從動量守恆推導（簡化版）
                rho_w = (f0 + f1 + f3 + 2.0*(f2 + f5 + f6)) / (1.0 + 0.0)  # v_y=0 at wall

                f_dst[i, j][4] = f2
                f_dst[i, j][7] = f5 - (1.0/6.0) * rho_w * u_wall
                f_dst[i, j][8] = f6 + (1.0/6.0) * rho_w * u_wall

    @ti.kernel
    def _moving_wall_bottom(self, f_dst: ti.template()):
        """底部運動壁面"""
        j = 0
        for i in range(self.nx):
            if self.solver.mask[i, j] == 0:
                u_wall = self.u_wall_field[i]

                f0 = f_dst[i, j][0]
                f1 = f_dst[i, j][1]
                f3 = f_dst[i, j][3]
                f4 = f_dst[i, j][4]
                f7 = f_dst[i, j][7]
                f8 = f_dst[i, j][8]

                rho_w = (f0 + f1 + f3 + 2.0*(f4 + f7 + f8)) / (1.0 - 0.0)

                f_dst[i, j][2] = f4
                f_dst[i, j][5] = f7 + (1.0/6.0) * rho_w * u_wall
                f_dst[i, j][6] = f8 - (1.0/6.0) * rho_w * u_wall
