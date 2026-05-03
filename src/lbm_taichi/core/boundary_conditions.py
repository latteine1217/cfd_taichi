"""
邊界條件模組
============

提供各種 LBM 邊界條件的實作：
- 固定速度入口
- Free-Slip 壁面（鏡面反射，無摩擦）
- No-Slip 壁面（Bounce-Back）
- Moving Wall（運動壁面，如 Lid-Driven Cavity 的上蓋）

## 邊界條件分類與 mask 使用規則

### 類型 A：流體邊界條件（mask = 0）
這些邊界條件施加在**流體節點**上，通過重建分佈函數實現：
- 固定速度入口
- Moving Wall（運動壁面）
- Free-Slip（改進版）

Why mask = 0?
- 這些節點參與流體計算
- 通過邊界條件 kernel 重建未知分佈函數
- 在 `step()` 中的 BC 階段施加

### 類型 B：固體邊界條件（mask = 1）
這些邊界條件通過設置 mask 實現，使用 Bounce-Back：
- No-Slip 壁面
- 障礙物（圓柱、機翼等）

Why mask = 1?
- 這些節點不參與流體計算
- Bounce-Back 在 `_collide_and_stream()` 中自動處理
- 速度自動為零，滿足 No-Slip

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
        self.u_inlet_perturb = ti.field(dtype=ti.f32, shape=())
        self.u_inlet_omega = ti.field(dtype=ti.f32, shape=())
        self.u_inlet_time = ti.field(dtype=ti.f32, shape=())
        self.u_inlet_asym = ti.field(dtype=ti.f32, shape=())
        self.rho_outlet = ti.field(dtype=ti.f32, shape=())
        self.u_wall_field = ti.field(dtype=ti.f32, shape=self.nx)  # 用於 Moving Wall
        self.corner_extrapolation_enabled = False

        # 全域質量修正（低頻、弱鬆弛）
        self.mass_correction_interval = ti.field(dtype=ti.i32, shape=())
        self.mass_correction_strength = ti.field(dtype=ti.f32, shape=())
        self.mass_correction_counter = ti.field(dtype=ti.i32, shape=())
        self.mass_correction_enabled = False
        self._init_mass_correction()

        # Orlanski 出口緩衝區
        self.outflow_relax_right = ti.field(dtype=ti.f32, shape=())
        self.outflow_relax_left = ti.field(dtype=ti.f32, shape=())
        self.outflow_relax_top = ti.field(dtype=ti.f32, shape=())
        self.outflow_relax_bottom = ti.field(dtype=ti.f32, shape=())
        self.outflow_rho_target_right = ti.field(dtype=ti.f32, shape=())
        self.outflow_rho_target_left = ti.field(dtype=ti.f32, shape=())
        self.outflow_rho_target_top = ti.field(dtype=ti.f32, shape=())
        self.outflow_rho_target_bottom = ti.field(dtype=ti.f32, shape=())
        self.rho_outflow_right = ti.field(dtype=ti.f32, shape=self.ny)
        self.rho_outflow_left = ti.field(dtype=ti.f32, shape=self.ny)
        self.rho_outflow_top = ti.field(dtype=ti.f32, shape=self.nx)
        self.rho_outflow_bottom = ti.field(dtype=ti.f32, shape=self.nx)
        self.u_outflow_right = ti.Vector.field(2, dtype=ti.f32, shape=self.ny)
        self.u_outflow_left = ti.Vector.field(2, dtype=ti.f32, shape=self.ny)
        self.u_outflow_top = ti.Vector.field(2, dtype=ti.f32, shape=self.nx)
        self.u_outflow_bottom = ti.Vector.field(2, dtype=ti.f32, shape=self.nx)
        self.outflow_smooth_strength = ti.field(dtype=ti.f32, shape=())
        self._init_outflow_buffers()
        self.u_inlet_perturb[None] = 0.0
        self.u_inlet_omega[None] = 0.0
        self.u_inlet_time[None] = 0.0
        self.u_inlet_asym[None] = 0.0
        self._inlet_time_enabled = False

    def add_velocity_inlet(
        self,
        u_in: float,
        location: str = "left",
        epsilon: float = 0.02,
        omega=None,
        strouhal: float = 0.2,
        asymmetry: float = 0.0,
        method: str = "neq",
    ):
        """
        添加固定速度入口邊界

        Why:
        - 透過重建未知分佈函數施加入口速度
        - 保證質量守恆：sum(f_i) = rho
        - 滿足固定速度：sum(f_i * e_i) / rho = u_target

        Args:
            u_in: 入口速度 (lattice units)
            location: 'left', 'right', 'top', 'bottom'
            epsilon: 時間性正弦擾動幅度（相對值，1%~5%）
            omega: 擾動角頻率（rad/step），None 時用 Strouhal 估算
            strouhal: Strouhal 數（用於估算 omega）
            asymmetry: 入口速度上下非對稱擾動幅度（相對 u_in）
            method: 'neq' (Guo 非平衡外推) or 'zouhe' (標準 Zou-He)
        """
        self.u_inlet[None] = u_in
        self.u_inlet_perturb[None] = epsilon
        if omega is None:
            omega = 2.0 * np.pi * strouhal * u_in / self.solver.L_char
        self.u_inlet_omega[None] = omega
        self.u_inlet_asym[None] = asymmetry
        if not self._inlet_time_enabled:
            self.solver.add_boundary_condition(
                self._advance_inlet_time, "Inlet Time Advance"
            )
            self._inlet_time_enabled = True

        if method == "neq":
            left_kernel = self._velocity_inlet_left_neq
            right_kernel = self._velocity_inlet_right_neq
            tag = "Velocity Inlet NEQ"
        elif method == "zouhe":
            left_kernel = self._velocity_inlet_left
            right_kernel = self._velocity_inlet_right
            tag = "Velocity Inlet Zou-He"
        else:
            raise ValueError(f"Unknown inlet method: {method}")

        if location == "left":
            self.solver.add_boundary_condition(left_kernel, f"{tag} (Left)")
        elif location == "right":
            self.solver.add_boundary_condition(right_kernel, f"{tag} (Right)")
        else:
            raise NotImplementedError(
                f"Location '{location}' not implemented for velocity inlet"
            )

    def add_neumann_outflow(self, location: str = "right"):
        """
        添加 Neumann 出口邊界（零梯度外推，可選質量修正）

        Why Neumann Outflow?
        - 最簡單的開放邊界條件
        - ∂u/∂n = 0, ∂v/∂n = 0, ∂ρ/∂n = 0
        - 假設流動在出口處「fully developed」
        - 比固定壓力出口更不具侵入性（不強制特定壓力）

        Implementation Note:
        - 使用非平衡外推（Non-equilibrium extrapolation）
        - 宏觀量做弱平滑，減少 fp32 噪聲放大

        When to use?
        - 長管道出口（充分發展流）
        - 不確定出口壓力時
        - 希望最小化反射波時

        Limitation:
        - 不適用於強回流情況

        Args:
            location: 'right', 'left', 'top', 'bottom'
        """
        if location == "right":
            self.solver.add_boundary_condition(
                self._neumann_outflow_right, "Neumann Outflow (Right)"
            )
        elif location == "left":
            self.solver.add_boundary_condition(
                self._neumann_outflow_left, "Neumann Outflow (Left)"
            )
        elif location == "top":
            self.solver.add_boundary_condition(
                self._neumann_outflow_top, "Neumann Outflow (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._neumann_outflow_bottom, "Neumann Outflow (Bottom)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

        if not self.mass_correction_enabled:
            self.solver.add_boundary_condition(
                self._global_mass_correction_relaxed,
                "Global Mass Correction (Low-Frequency)",
            )
            self.mass_correction_enabled = True

    def add_stable_outlet(
        self,
        rho_out: float = 1.0,
        location: str = "right",
        relaxation: float = 0.02,
    ):
        """
        高雷諾數穩定出口（Orlanski + Weak Relaxation）

        What:
            - Orlanski 對流外推 (non-reflecting)
            - 以弱壓力鬆弛抑制密度漂移
        Why:
            - 高 Re 外流易產生反射與回流
            - Orlanski 更接近非反射邊界
        When:
            - 高 Re 圓柱/機翼外流
            - 非定常尾流

        Args:
            rho_out: 目標出口密度
            location: 'right' 或 'left'
            relaxation: Zou-He 壓力鬆弛係數
        """
        self.add_orlanski_outflow(
            location=location,
            rho_target=rho_out,
            relaxation=relaxation,
        )

    def add_no_slip_wall(self, location: str, exclude_corners: bool = False):
        """
        添加 No-Slip 壁面（Bounce-Back）

        Why No-Slip?
        - 固體壁面邊界條件
        - 速度為零：u = 0, v = 0
        - 通過 mask 設置，使用 Bounce-Back 自動處理

        Args:
            location: 'top', 'bottom', 'left', 'right'
            exclude_corners: 是否排除角點（用於 Moving Wall 情況）
        """
        if location == "top":
            self._set_mask_top(exclude_corners)
        elif location == "bottom":
            self._set_mask_bottom(exclude_corners)
        elif location == "left":
            self._set_mask_left(exclude_corners)
        elif location == "right":
            self._set_mask_right(exclude_corners)
        else:
            raise ValueError(f"Invalid location: {location}")

        # mask 已改變，自動重建索引列表。
        # Why: bulk/boundary/solid 分類依賴 mask，設完壁面即刻使其一致；
        #      無需呼叫端手動呼叫 solver._build_index_lists()。
        self.solver.rebuild_index_lists()

    def set_corners_solid(self):
        """
        將四個角點設為固體（mask=1）

        Why 角點需要特殊處理?
        - 角點被兩個邊界條件同時影響
        - 設為固體避免未定義行為
        - 對數值穩定性影響小（只有 4 個節點）

        ⚠️ Warning: 對於小計算域（< 128×128），固體角點會阻礙流動
        建議使用 `handle_corners_extrapolation()` 代替
        """
        self._set_corners_solid_kernel()

    def handle_corners_extrapolation(self):
        """
        角點外推處理（物理正確版）

        Why Extrapolation?
        - ✅ 不阻礙流動（mask=0，流體節點）
        - ✅ 從兩個邊界的內部節點外推
        - ✅ 適用於小計算域

        When to use?
        - 小計算域（< 128×128）
        - Cavity 等封閉流動
        - 需要精確流場（角點影響不可忽略）

        Note: 必須在每個時間步的邊界條件施加後調用
        """
        if not self.corner_extrapolation_enabled:
            self.solver.add_boundary_condition(
                self._handle_corners_extrapolation_kernel, "Corner Extrapolation"
            )
            self.corner_extrapolation_enabled = True

    @ti.kernel
    def _set_corners_solid_kernel(self):
        """設置四個角點為固體"""
        self.solver.mask[1, 1] = 1  # 左下
        self.solver.mask[self.nx, 1] = 1  # 右下
        self.solver.mask[1, self.ny] = 1  # 左上
        self.solver.mask[self.nx, self.ny] = 1  # 右上

    @ti.kernel
    def _handle_corners_extrapolation_kernel(self, f_dst: ti.template()):
        """
        角點外推處理 kernel

        實作方式：對角平均外推
        - 左下角 (0,0)：從 (1,0) 和 (0,1) 平均外推
        - 右下角 (nx-1,0)：從 (nx-2,0) 和 (nx-1,1) 平均外推
        - 左上角 (0,ny-1)：從 (1,ny-1) 和 (0,ny-2) 平均外推
        - 右上角 (nx-1,ny-1)：從 (nx-2,ny-1) 和 (nx-1,ny-2) 平均外推

        Why 平均外推?
        - 保持兩個邊界的影響平衡
        - 避免角點成為奇異點
        - 物理上更合理（流體可自由流動）
        """
        # 只在流體角點（mask=0）施加外推
        # 左下角 (0, 0)
        if self.solver.mask[1, 1] == 0:
            for k in ti.static(range(9)):
                f1 = f_dst[2, 1][k]  # 右側鄰居
                f2 = f_dst[1, 2][k]  # 上側鄰居
                f_dst[1, 1][k] = 0.5 * (f1 + f2)

        # 右下角 (nx-1, 0)
        if self.solver.mask[self.nx, 1] == 0:
            for k in ti.static(range(9)):
                f1 = f_dst[self.nx - 1, 1][k]  # 左側鄰居
                f2 = f_dst[self.nx, 2][k]  # 上側鄰居
                f_dst[self.nx, 1][k] = 0.5 * (f1 + f2)

        # 左上角 (0, ny-1)
        if self.solver.mask[1, self.ny] == 0:
            for k in ti.static(range(9)):
                f1 = f_dst[2, self.ny][k]  # 右側鄰居
                f2 = f_dst[1, self.ny - 1][k]  # 下側鄰居
                f_dst[1, self.ny][k] = 0.5 * (f1 + f2)

        # 右上角 (nx-1, ny-1)
        if self.solver.mask[self.nx, self.ny] == 0:
            for k in ti.static(range(9)):
                f1 = f_dst[self.nx - 1, self.ny][k]  # 左側鄰居
                f2 = f_dst[self.nx, self.ny - 1][k]  # 下側鄰居
                f_dst[self.nx, self.ny][k] = 0.5 * (f1 + f2)

    @ti.kernel
    def _handle_ldc_corners_kernel(self, f_dst: ti.template()):
        """
        LDC 專用角點處理：以局部平衡分佈重建四個角點，避免 generic average
        在閉合腔體中持續注入小的非物理質量誤差。

        策略：
        - top-left / top-right: 使用 lid 端點速度（通常平滑 profile 端點接近 0）
        - bottom-left / bottom-right: 使用靜止壁面速度 (0, 0)
        - rho 使用相鄰內部流體節點平均
        """
        # bottom-left
        if self.solver.mask[1, 1] == 0:
            rho_w = 0.5 * (self.solver.rho[2, 1] + self.solver.rho[1, 2])
            u = ti.Vector([0.0, 0.0])
            u_sq = u.norm_sqr()
            for k in ti.static(range(9)):
                eu = self.solver.e[k].dot(u)
                f_eq = self.solver.w[k] * rho_w * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                f_dst[1, 1][k] = f_eq

        # bottom-right
        if self.solver.mask[self.nx, 1] == 0:
            rho_w = 0.5 * (self.solver.rho[self.nx - 1, 1] + self.solver.rho[self.nx, 2])
            u = ti.Vector([0.0, 0.0])
            u_sq = u.norm_sqr()
            for k in ti.static(range(9)):
                eu = self.solver.e[k].dot(u)
                f_eq = self.solver.w[k] * rho_w * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                f_dst[self.nx, 1][k] = f_eq

        # top-left
        if self.solver.mask[1, self.ny] == 0:
            rho_w = 0.5 * (self.solver.rho[2, self.ny] + self.solver.rho[1, self.ny - 1])
            u = ti.Vector([self.u_wall_field[0], 0.0])
            u_sq = u.norm_sqr()
            for k in ti.static(range(9)):
                eu = self.solver.e[k].dot(u)
                f_eq = self.solver.w[k] * rho_w * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                f_dst[1, self.ny][k] = f_eq

        # top-right
        if self.solver.mask[self.nx, self.ny] == 0:
            rho_w = 0.5 * (self.solver.rho[self.nx - 1, self.ny] + self.solver.rho[self.nx, self.ny - 1])
            u = ti.Vector([self.u_wall_field[self.nx - 1], 0.0])
            u_sq = u.norm_sqr()
            for k in ti.static(range(9)):
                eu = self.solver.e[k].dot(u)
                f_eq = self.solver.w[k] * rho_w * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                f_dst[self.nx, self.ny][k] = f_eq

    @ti.kernel
    def _set_mask_top(self, exclude_corners: ti.i32):
        """設置頂部為固體壁面"""
        start = 1 if exclude_corners else 0
        end = self.nx - 1 if exclude_corners else self.nx
        for i in range(start, end):
            self.solver.mask[i + 1, self.ny] = 1

    @ti.kernel
    def _set_mask_bottom(self, exclude_corners: ti.i32):
        """設置底部為固體壁面"""
        start = 1 if exclude_corners else 0
        end = self.nx - 1 if exclude_corners else self.nx
        for i in range(start, end):
            self.solver.mask[i + 1, 1] = 1

    @ti.kernel
    def _set_mask_left(self, exclude_corners: ti.i32):
        """設置左側為固體壁面"""
        start = 1 if exclude_corners else 0
        end = self.ny - 1 if exclude_corners else self.ny
        for j in range(start, end):
            self.solver.mask[1, j + 1] = 1

    @ti.kernel
    def _set_mask_right(self, exclude_corners: ti.i32):
        """設置右側為固體壁面"""
        start = 1 if exclude_corners else 0
        end = self.ny - 1 if exclude_corners else self.ny
        for j in range(start, end):
            self.solver.mask[self.nx, j + 1] = 1

    def add_periodic_boundary(self, direction: str):
        """
        添加週期邊界條件（Periodic Boundary Conditions）

        Why Periodic BC?
        - ✅ 湍流模擬必須（空間週期性）
        - ✅ 減少計算域大小
        - ✅ 消除邊界影響
        - ✅ 研究充分發展流

        Physical Meaning:
        - 流體從右邊界流出 → 從左邊界流入
        - 保持流場週期性：f[0, j] = f[nx-1, j]

        When to use?
        - 湍流模擬（DNS, LES）
        - 管道充分發展流
        - 週期性幾何（重複單元）
        - Taylor-Green Vortex 等基準測試

        Critical:
        - 不要與其他 BC 同時施加在同一方向
        - 需要確保幾何也是週期的（無障礙物跨邊界）

        Args:
            direction: 'x' 或 'y'
            direction: 'x' 或 'y'
        """
        if direction == "x":
            self.solver.add_boundary_condition(
                self._periodic_x, "Periodic (X-direction)"
            )
        elif direction == "y":
            self.solver.add_boundary_condition(
                self._periodic_y, "Periodic (Y-direction)"
            )
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'x' or 'y'.")

    def add_free_slip_wall(self, location: str):
        """
        添加 Free-Slip 壁面（Zou-He 類型，改進版）

        Why Free-Slip?
        - 模擬無摩擦壁面（風洞外壁、對稱邊界）
        - 法向速度為零，切向速度自由滑移
        - 零法向應力：∂u_tangent/∂n = 0

        Why Zou-He 類型 vs 簡單鏡面反射?
        ✅ 正確處理密度變化（壓力梯度）
        ✅ 滿足質量守恆與動量守恆
        ✅ 在非均勻流場中更精確
        ❌ 簡單鏡面反射假設密度恆定，在有壓力梯度時會產生誤差

        Args:
            location: 'top', 'bottom', 'left', 'right'
        """
        if location == "top":
            self.solver.add_boundary_condition(self._free_slip_top, "Free-Slip (Top)")
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._free_slip_bottom, "Free-Slip (Bottom)"
            )
        elif location == "left":
            self.solver.add_boundary_condition(self._free_slip_left, "Free-Slip (Left)")
        elif location == "right":
            self.solver.add_boundary_condition(
                self._free_slip_right, "Free-Slip (Right)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_moving_wall(
        self,
        velocity_profile: np.ndarray,
        location: str = "top",
        handle_corners: bool = True,
        corner_mode: str = "generic",
    ):
        """
        添加運動壁面（用於 Lid-Driven Cavity）

        Why 需要速度 profile?
        - 避免角點奇異性（速度不連續）
        - 平滑速度變化提高數值穩定性
        - Lid-Driven Cavity 的標準處理方式

        Args:
            velocity_profile: (nx,) 陣列，每個 x 位置的壁面速度
            location: 'top', 'bottom'
            handle_corners: 是否處理角點
            corner_mode: 'generic' 使用既有平均外推；'ldc' 使用 lid-driven cavity
                專用角點平衡態重建（較一致，減少角點漏量）
        """
        if velocity_profile.shape[0] != self.nx:
            raise ValueError(
                f"Velocity profile length {velocity_profile.shape[0]} != nx {self.nx}"
            )

        self.u_wall_field.from_numpy(velocity_profile.astype(np.float32))

        if location == "top":
            self.solver.add_boundary_condition(
                self._moving_wall_top, "Moving Wall (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._moving_wall_bottom, "Moving Wall (Bottom)"
            )
        else:
            raise NotImplementedError(
                f"Location '{location}' not implemented for moving wall"
            )

        if handle_corners:
            if corner_mode == "ldc" and location == "top":
                self.solver.add_boundary_condition(
                    self._handle_ldc_corners_kernel, "LDC Corner Reconstruction"
                )
            else:
                self.handle_corners_extrapolation()

    @ti.kernel
    def _init_mass_correction(self):
        self.mass_correction_interval[None] = 0
        self.mass_correction_strength[None] = 0.0
        self.mass_correction_counter[None] = 0

    def add_orlanski_outflow(
        self,
        location: str = "right",
        rho_target: float = 1.0,
        relaxation: float = 0.02,
    ):
        """
        Orlanski 非反射出口

        What:
            - 對流型外推：phi^{n+1} = phi^{n} - c (phi^{n} - phi_i^{n})
        Why:
            - 減少出口反射
            - 保留非定常波的傳出
        When:
            - 尾流、剪切不穩定、渦脫落

        Args:
            location: 'right', 'left', 'top', 'bottom'
            rho_target: 目標密度（弱鬆弛用）
            relaxation: 鬆弛係數（0-0.1，建議 0.02）
        """
        if location == "right":
            self.outflow_relax_right[None] = relaxation
            self.outflow_rho_target_right[None] = rho_target
            self.solver.add_boundary_condition(
                self._orlanski_outflow_right, "Orlanski Outflow (Right)"
            )
        elif location == "left":
            self.outflow_relax_left[None] = relaxation
            self.outflow_rho_target_left[None] = rho_target
            self.solver.add_boundary_condition(
                self._orlanski_outflow_left, "Orlanski Outflow (Left)"
            )
        elif location == "top":
            self.outflow_relax_top[None] = relaxation
            self.outflow_rho_target_top[None] = rho_target
            self.solver.add_boundary_condition(
                self._orlanski_outflow_top, "Orlanski Outflow (Top)"
            )
        elif location == "bottom":
            self.outflow_relax_bottom[None] = relaxation
            self.outflow_rho_target_bottom[None] = rho_target
            self.solver.add_boundary_condition(
                self._orlanski_outflow_bottom, "Orlanski Outflow (Bottom)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    @ti.func
    def _compute_macro_from_f(self, f_vec: ti.template()):
        rho = 0.0
        u = ti.Vector([0.0, 0.0])
        for k in ti.static(range(9)):
            rho += f_vec[k]
            u += f_vec[k] * ti.cast(self.solver.e[k], ti.f32)
        if rho > 1e-12:
            u /= rho
        return rho, u

    @ti.func
    def _compute_equilibrium(self, rho, u):
        u_sq = u.dot(u)
        feq = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            e_k = ti.cast(self.solver.e[k], ti.f32)
            eu = e_k.dot(u)
            feq[k] = (
                self.solver.w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
            )
        return feq

    @ti.kernel
    def _global_mass_correction_relaxed(self, f_dst: ti.template()):
        interval = self.mass_correction_interval[None]
        step = self.mass_correction_counter[None]

        if interval > 0 and step % interval == 0:
            total_mass = 0.0
            for i, j in ti.ndrange(self.nx, self.ny):
                ig = i + 1
                jg = j + 1
                if self.solver.mask[ig, jg] == 0:
                    rho_local = 0.0
                    for k in ti.static(range(9)):
                        rho_local += f_dst[ig, jg][k]
                    ti.atomic_add(total_mass, rho_local)

            target_mass = self.solver.initial_mass[None]
            if total_mass > 1e-12:
                ratio = target_mass / total_mass
                strength = self.mass_correction_strength[None]
                correction = 1.0 + strength * (ratio - 1.0)

                for i, j in ti.ndrange(self.nx, self.ny):
                    ig = i + 1
                    jg = j + 1
                    if self.solver.mask[ig, jg] == 0:
                        for k in ti.static(range(9)):
                            f_dst[ig, jg][k] *= correction

        self.mass_correction_counter[None] = step + 1

    @ti.kernel
    def _init_outflow_buffers(self):
        for j in range(self.ny):
            self.rho_outflow_right[j] = 1.0
            self.rho_outflow_left[j] = 1.0
            self.u_outflow_right[j] = ti.Vector([0.0, 0.0])
            self.u_outflow_left[j] = ti.Vector([0.0, 0.0])

        for i in range(self.nx):
            self.rho_outflow_top[i] = 1.0
            self.rho_outflow_bottom[i] = 1.0
            self.u_outflow_top[i] = ti.Vector([0.0, 0.0])
            self.u_outflow_bottom[i] = ti.Vector([0.0, 0.0])

        self.outflow_relax_right[None] = 0.02
        self.outflow_relax_left[None] = 0.02
        self.outflow_relax_top[None] = 0.02
        self.outflow_relax_bottom[None] = 0.02
        self.outflow_rho_target_right[None] = 1.0
        self.outflow_rho_target_left[None] = 1.0
        self.outflow_rho_target_top[None] = 1.0
        self.outflow_rho_target_bottom[None] = 1.0
        self.outflow_smooth_strength[None] = 0.02

    @ti.kernel
    def _advance_inlet_time(self, f_dst: ti.template()):
        self.u_inlet_time[None] += 1.0

    @ti.func
    def _current_inlet_speed(self, j: ti.i32):
        t = self.u_inlet_time[None]
        base_u = self.u_inlet[None]
        omega = self.u_inlet_omega[None]
        epsilon = self.u_inlet_perturb[None]
        base = base_u * (1.0 + epsilon * ti.sin(omega * t))
        denom = ti.max(ti.cast(self.ny - 1, ti.f32), 1.0)
        y_norm = ti.cast(j, ti.f32) / denom
        asym = base_u * self.u_inlet_asym[None] * (y_norm - 0.5)
        return base + asym

    @ti.kernel
    def _orlanski_outflow_right(self, f_dst: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[self.nx - 1, jg])
                c_s = ti.sqrt(1.0 / 3.0)
                c = ti.max(u_i[0] + c_s, 0.0)
                c = ti.min(c, 1.0)

                rho_prev = self.rho_outflow_right[j]
                u_prev = self.u_outflow_right[j]

                rho_new = rho_prev - c * (rho_prev - rho_i)
                u_new = u_prev - c * (u_prev - u_i)

                if c < 1e-6:
                    rho_new = rho_i
                    u_new = u_i

                alpha = self.outflow_relax_right[None]
                rho_target = self.outflow_rho_target_right[None]
                rho_new = (1.0 - alpha) * rho_new + alpha * rho_target
                rho_new = ti.max(rho_new, 1e-6)

                beta = self.outflow_smooth_strength[None]
                rho_new = (1.0 - beta) * rho_new + beta * rho_i
                u_new = (1.0 - beta) * u_new + beta * u_i

                u_sq = u_new.dot(u_new)
                u_i_sq = u_i.dot(u_i)
                for k in ti.static(range(9)):
                    e_k = ti.cast(self.solver.e[k], ti.f32)
                    eu_new = e_k.dot(u_new)
                    eu_i = e_k.dot(u_i)
                    feq_new = (
                        self.solver.w[k]
                        * rho_new
                        * (1.0 + 3.0 * eu_new + 4.5 * eu_new * eu_new - 1.5 * u_sq)
                    )
                    feq_i = (
                        self.solver.w[k]
                        * rho_i
                        * (1.0 + 3.0 * eu_i + 4.5 * eu_i * eu_i - 1.5 * u_i_sq)
                    )
                    f_dst[self.nx, jg][k] = feq_new + (
                        f_dst[self.nx - 1, jg][k] - feq_i
                    )

                self.rho_outflow_right[j] = rho_new
                self.u_outflow_right[j] = u_new

    @ti.kernel
    def _orlanski_outflow_left(self, f_dst: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[2, jg])
                c_s = ti.sqrt(1.0 / 3.0)
                c = ti.max(-u_i[0] + c_s, 0.0)
                c = ti.min(c, 1.0)

                rho_prev = self.rho_outflow_left[j]
                u_prev = self.u_outflow_left[j]

                rho_new = rho_prev - c * (rho_prev - rho_i)
                u_new = u_prev - c * (u_prev - u_i)

                if c < 1e-6:
                    rho_new = rho_i
                    u_new = u_i

                alpha = self.outflow_relax_left[None]
                rho_target = self.outflow_rho_target_left[None]
                rho_new = (1.0 - alpha) * rho_new + alpha * rho_target
                rho_new = ti.max(rho_new, 1e-6)

                beta = self.outflow_smooth_strength[None]
                rho_new = (1.0 - beta) * rho_new + beta * rho_i
                u_new = (1.0 - beta) * u_new + beta * u_i

                u_sq = u_new.dot(u_new)
                u_i_sq = u_i.dot(u_i)
                for k in ti.static(range(9)):
                    e_k = ti.cast(self.solver.e[k], ti.f32)
                    eu_new = e_k.dot(u_new)
                    eu_i = e_k.dot(u_i)
                    feq_new = (
                        self.solver.w[k]
                        * rho_new
                        * (1.0 + 3.0 * eu_new + 4.5 * eu_new * eu_new - 1.5 * u_sq)
                    )
                    feq_i = (
                        self.solver.w[k]
                        * rho_i
                        * (1.0 + 3.0 * eu_i + 4.5 * eu_i * eu_i - 1.5 * u_i_sq)
                    )
                    f_dst[1, jg][k] = feq_new + (f_dst[2, jg][k] - feq_i)

                self.rho_outflow_left[j] = rho_new
                self.u_outflow_left[j] = u_new

    @ti.kernel
    def _orlanski_outflow_top(self, f_dst: ti.template()):
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, self.ny] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[ig, self.ny - 1])
                c_s = ti.sqrt(1.0 / 3.0)
                c = ti.max(u_i[1] + c_s, 0.0)
                c = ti.min(c, 1.0)

                rho_prev = self.rho_outflow_top[i]
                u_prev = self.u_outflow_top[i]

                rho_new = rho_prev - c * (rho_prev - rho_i)
                u_new = u_prev - c * (u_prev - u_i)

                if c < 1e-6:
                    rho_new = rho_i
                    u_new = u_i

                alpha = self.outflow_relax_top[None]
                rho_target = self.outflow_rho_target_top[None]
                rho_new = (1.0 - alpha) * rho_new + alpha * rho_target
                rho_new = ti.max(rho_new, 1e-6)

                beta = self.outflow_smooth_strength[None]
                rho_new = (1.0 - beta) * rho_new + beta * rho_i
                u_new = (1.0 - beta) * u_new + beta * u_i

                u_sq = u_new.dot(u_new)
                u_i_sq = u_i.dot(u_i)
                for k in ti.static(range(9)):
                    e_k = ti.cast(self.solver.e[k], ti.f32)
                    eu_new = e_k.dot(u_new)
                    eu_i = e_k.dot(u_i)
                    feq_new = (
                        self.solver.w[k]
                        * rho_new
                        * (1.0 + 3.0 * eu_new + 4.5 * eu_new * eu_new - 1.5 * u_sq)
                    )
                    feq_i = (
                        self.solver.w[k]
                        * rho_i
                        * (1.0 + 3.0 * eu_i + 4.5 * eu_i * eu_i - 1.5 * u_i_sq)
                    )
                    f_dst[ig, self.ny][k] = feq_new + (
                        f_dst[ig, self.ny - 1][k] - feq_i
                    )

                self.rho_outflow_top[i] = rho_new
                self.u_outflow_top[i] = u_new

    @ti.kernel
    def _orlanski_outflow_bottom(self, f_dst: ti.template()):
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[ig, 2])
                c_s = ti.sqrt(1.0 / 3.0)
                c = ti.max(-u_i[1] + c_s, 0.0)
                c = ti.min(c, 1.0)

                rho_prev = self.rho_outflow_bottom[i]
                u_prev = self.u_outflow_bottom[i]

                rho_new = rho_prev - c * (rho_prev - rho_i)
                u_new = u_prev - c * (u_prev - u_i)

                if c < 1e-6:
                    rho_new = rho_i
                    u_new = u_i

                alpha = self.outflow_relax_bottom[None]
                rho_target = self.outflow_rho_target_bottom[None]
                rho_new = (1.0 - alpha) * rho_new + alpha * rho_target
                rho_new = ti.max(rho_new, 1e-6)

                beta = self.outflow_smooth_strength[None]
                rho_new = (1.0 - beta) * rho_new + beta * rho_i
                u_new = (1.0 - beta) * u_new + beta * u_i

                u_sq = u_new.dot(u_new)
                u_i_sq = u_i.dot(u_i)
                for k in ti.static(range(9)):
                    e_k = ti.cast(self.solver.e[k], ti.f32)
                    eu_new = e_k.dot(u_new)
                    eu_i = e_k.dot(u_i)
                    feq_new = (
                        self.solver.w[k]
                        * rho_new
                        * (1.0 + 3.0 * eu_new + 4.5 * eu_new * eu_new - 1.5 * u_sq)
                    )
                    feq_i = (
                        self.solver.w[k]
                        * rho_i
                        * (1.0 + 3.0 * eu_i + 4.5 * eu_i * eu_i - 1.5 * u_i_sq)
                    )
                    f_dst[ig, 1][k] = feq_new + (f_dst[ig, 2][k] - feq_i)

                self.rho_outflow_bottom[i] = rho_new
                self.u_outflow_bottom[i] = u_new

    # ==================== Zou-He 入口 ====================

    @ti.kernel
    def _velocity_inlet_left(self, f_dst: ti.template()):
        """
        左邊界固定速度入口

        已知：f0, f2, f3, f4, f6, f7 (從內部流傳來的)
        未知：f1, f5, f8 (需要重建)

        質量守恆：rho = (f0+f2+f4 + 2*(f3+f6+f7)) / (1 - u_in)
        動量守恆：rho*u_in = f1-f3 + f5-f6-f7+f8

        Mask 檢查說明：
        - 只在流體節點（mask=0）施加此邊界條件
        - 如果 mask=1（固體），則由 Bounce-Back 處理
        """
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:  # 類型 A：流體邊界
                u_in = self._current_inlet_speed(j)
                f0 = f_dst[1, jg][0]
                f2 = f_dst[1, jg][2]
                f3 = f_dst[1, jg][3]
                f4 = f_dst[1, jg][4]
                f6 = f_dst[1, jg][6]
                f7 = f_dst[1, jg][7]
                rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)

                # 重建未知分佈函數
                f_dst[1, jg][1] = f3 + (2.0 / 3.0) * rho_in * u_in
                f_dst[1, jg][5] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * u_in
                f_dst[1, jg][8] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * u_in

    @ti.kernel
    def _velocity_inlet_left_neq(self, f_dst: ti.template()):
        """
        左邊界固定速度入口（Guo 非平衡外推）

        f_b = f_eq(rho_b, u_b) + (f_i - f_eq(rho_i, u_i))
        - rho_b 取鄰近流體格點密度
        - u_b 取指定入口速度
        """
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                u_in = self._current_inlet_speed(j)
                rho_i, u_i = self._compute_macro_from_f(f_dst[2, jg])
                u_b = ti.Vector([u_in, 0.0])
                rho_b = rho_i

                feq_b = self._compute_equilibrium(rho_b, u_b)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[1, jg][k] = feq_b[k] + (f_dst[2, jg][k] - feq_i[k])

    @ti.kernel
    def _velocity_inlet_right(self, f_dst: ti.template()):
        """右邊界固定速度入口"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                u_in = -self._current_inlet_speed(j)
                f0 = f_dst[self.nx, jg][0]
                f1 = f_dst[self.nx, jg][1]
                f2 = f_dst[self.nx, jg][2]
                f4 = f_dst[self.nx, jg][4]
                f5 = f_dst[self.nx, jg][5]
                f8 = f_dst[self.nx, jg][8]
                rho_in = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + u_in)

                f_dst[self.nx, jg][3] = f1 - (2.0 / 3.0) * rho_in * u_in
                f_dst[self.nx, jg][7] = (
                    f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_in * u_in
                )
                f_dst[self.nx, jg][6] = (
                    f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_in * u_in
                )

    @ti.kernel
    def _velocity_inlet_right_neq(self, f_dst: ti.template()):
        """右邊界固定速度入口（Guo 非平衡外推）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                u_in = self._current_inlet_speed(j)
                rho_i, u_i = self._compute_macro_from_f(f_dst[self.nx - 1, jg])
                u_b = ti.Vector([-u_in, 0.0])
                rho_b = rho_i

                feq_b = self._compute_equilibrium(rho_b, u_b)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[self.nx, jg][k] = feq_b[k] + (
                        f_dst[self.nx - 1, jg][k] - feq_i[k]
                    )

    # ==================== Free-Slip 壁面（改進版 Zou-He 類型）====================

    @ti.kernel
    def _free_slip_bottom(self, f_dst: ti.template()):
        """
        底部 Free-Slip 壁面（Zou-He 類型）

        Why Zou-He Free-Slip?
        - 正確處理密度變化（壓力梯度）
        - 允許切向速度滑移
        - 強制法向速度為零：v_y = 0
        - 比簡單鏡面反射更精確

        物理條件：
        - v_y = 0（法向速度為零）
        - ∂v_x/∂y = 0（零切向應力，通過外推實現）

        已知：f0, f1, f3, f4, f7, f8 (從內部流傳來)
        未知：f2, f5, f6 (需要重建，向上的分佈)
        """
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0:  # 類型 A：流體邊界
                f0 = f_dst[ig, 1][0]
                f1 = f_dst[ig, 1][1]
                f3 = f_dst[ig, 1][3]
                f4 = f_dst[ig, 1][4]
                f7 = f_dst[ig, 1][7]
                f8 = f_dst[ig, 1][8]

                # 質量守恆（v_y = 0）
                rho = f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)

                # 切向速度：使用內部節點外推以降低剪切噪聲
                _, u_inner = self._compute_macro_from_f(f_dst[ig, 2])
                u_x = u_inner[0]

                # Zou-He 重建未知分佈函數（Krüger 2017）
                f_dst[ig, 1][2] = f4
                f_dst[ig, 1][5] = (
                    f7 + 0.5 * (f1 - f3) + (1.0 / 6.0) * rho * u_x
                )
                f_dst[ig, 1][6] = (
                    f8 - 0.5 * (f1 - f3) - (1.0 / 6.0) * rho * u_x
                )

    @ti.kernel
    def _free_slip_top(self, f_dst: ti.template()):
        """頂部 Free-Slip 壁面（Zou-He 類型）"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, self.ny] == 0:
                f0 = f_dst[ig, self.ny][0]
                f1 = f_dst[ig, self.ny][1]
                f2 = f_dst[ig, self.ny][2]
                f3 = f_dst[ig, self.ny][3]
                f5 = f_dst[ig, self.ny][5]
                f6 = f_dst[ig, self.ny][6]

                # 質量守恆（v_y = 0）
                rho = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)

                # 切向速度：使用內部節點外推以降低剪切噪聲
                _, u_inner = self._compute_macro_from_f(f_dst[ig, self.ny - 1])
                u_x = u_inner[0]

                # Zou-He 重建（Krüger 2017）
                f_dst[ig, self.ny][4] = f2
                f_dst[ig, self.ny][7] = (
                    f5 - (1.0 / 6.0) * rho * u_x + 0.5 * (f1 - f3)
                )
                f_dst[ig, self.ny][8] = (
                    f6 + (1.0 / 6.0) * rho * u_x - 0.5 * (f1 - f3)
                )

    @ti.kernel
    def _free_slip_left(self, f_dst: ti.template()):
        """左側 Free-Slip 壁面（Zou-He 類型）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                f0 = f_dst[1, jg][0]
                f2 = f_dst[1, jg][2]
                f3 = f_dst[1, jg][3]
                f4 = f_dst[1, jg][4]
                f6 = f_dst[1, jg][6]
                f7 = f_dst[1, jg][7]

                # 質量守恆（v_x = 0）
                rho = f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)

                # 切向速度（y 方向）：使用內部節點外推
                _, u_inner = self._compute_macro_from_f(f_dst[2, jg])
                u_y = u_inner[1]

                # Zou-He 重建（Krüger 2017）
                f_dst[1, jg][1] = f3
                f_dst[1, jg][5] = (
                    f7 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_y
                )
                f_dst[1, jg][8] = (
                    f6 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * u_y
                )

    @ti.kernel
    def _free_slip_right(self, f_dst: ti.template()):
        """右側 Free-Slip 壁面（Zou-He 類型）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                f0 = f_dst[self.nx, jg][0]
                f1 = f_dst[self.nx, jg][1]
                f2 = f_dst[self.nx, jg][2]
                f4 = f_dst[self.nx, jg][4]
                f5 = f_dst[self.nx, jg][5]
                f8 = f_dst[self.nx, jg][8]

                # 質量守恆（v_x = 0）
                rho = f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)

                # 切向速度（y 方向）：使用內部節點外推
                _, u_inner = self._compute_macro_from_f(f_dst[self.nx - 1, jg])
                u_y = u_inner[1]

                # Zou-He 重建（Krüger 2017）
                f_dst[self.nx, jg][3] = f1
                f_dst[self.nx, jg][6] = (
                    f8 + 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * u_y
                )
                f_dst[self.nx, jg][7] = (
                    f5 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_y
                )

    @ti.kernel
    def _free_slip_bottom_symmetric(self, f_dst: ti.template()):
        """底部 Free-Slip（對稱延拓）"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0:
                f_dst[ig, 1][2] = f_dst[ig, 2][4]
                f_dst[ig, 1][5] = f_dst[ig, 2][7]
                f_dst[ig, 1][6] = f_dst[ig, 2][8]

    @ti.kernel
    def _free_slip_top_symmetric(self, f_dst: ti.template()):
        """頂部 Free-Slip（對稱延拓）"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, self.ny] == 0:
                f_dst[ig, self.ny][4] = f_dst[ig, self.ny - 1][2]
                f_dst[ig, self.ny][7] = f_dst[ig, self.ny - 1][5]
                f_dst[ig, self.ny][8] = f_dst[ig, self.ny - 1][6]

    @ti.kernel
    def _free_slip_left_symmetric(self, f_dst: ti.template()):
        """左側 Free-Slip（對稱延拓）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                f_dst[1, jg][1] = f_dst[2, jg][3]
                f_dst[1, jg][5] = f_dst[2, jg][7]
                f_dst[1, jg][8] = f_dst[2, jg][6]

    @ti.kernel
    def _free_slip_right_symmetric(self, f_dst: ti.template()):
        """右側 Free-Slip（對稱延拓）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                f_dst[self.nx, jg][3] = f_dst[self.nx - 1, jg][1]
                f_dst[self.nx, jg][6] = f_dst[self.nx - 1, jg][8]
                f_dst[self.nx, jg][7] = f_dst[self.nx - 1, jg][5]

    # ==================== Neumann Outflow（零梯度外推）====================

    @ti.kernel
    def _neumann_outflow_right(self, f_dst: ti.template()):
        """
        右邊界 Neumann 出口（零梯度外推）

        Why 零梯度？
        - 假設流動在出口處 fully developed
        - ∂u/∂x = 0 → u_boundary = u_interior
        - 最小化反射波

        實作方式：
        - 使用非平衡外推（Non-equilibrium extrapolation）
        - 以內部節點的非平衡量外推，並對宏觀量做弱平滑
        """
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:  # 類型 A：流體邊界
                rho_i, u_i = self._compute_macro_from_f(f_dst[self.nx - 1, jg])
                rho_prev = self.rho_outflow_right[j]
                u_prev = self.u_outflow_right[j]

                beta = self.outflow_smooth_strength[None]
                rho_target = (1.0 - beta) * rho_prev + beta * rho_i
                u_target = (1.0 - beta) * u_prev + beta * u_i
                rho_target = ti.max(rho_target, 1e-6)

                feq_target = self._compute_equilibrium(rho_target, u_target)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[self.nx, jg][k] = feq_target[k] + (
                        f_dst[self.nx - 1, jg][k] - feq_i[k]
                    )

                self.rho_outflow_right[j] = rho_target
                self.u_outflow_right[j] = u_target

    @ti.kernel
    def _neumann_outflow_left(self, f_dst: ti.template()):
        """左邊界 Neumann 出口"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[2, jg])
                rho_prev = self.rho_outflow_left[j]
                u_prev = self.u_outflow_left[j]

                beta = self.outflow_smooth_strength[None]
                rho_target = (1.0 - beta) * rho_prev + beta * rho_i
                u_target = (1.0 - beta) * u_prev + beta * u_i
                rho_target = ti.max(rho_target, 1e-6)

                feq_target = self._compute_equilibrium(rho_target, u_target)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[1, jg][k] = feq_target[k] + (f_dst[2, jg][k] - feq_i[k])

                self.rho_outflow_left[j] = rho_target
                self.u_outflow_left[j] = u_target

    @ti.kernel
    def _neumann_outflow_top(self, f_dst: ti.template()):
        """頂部 Neumann 出口"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, self.ny] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[ig, self.ny - 1])
                rho_prev = self.rho_outflow_top[i]
                u_prev = self.u_outflow_top[i]

                beta = self.outflow_smooth_strength[None]
                rho_target = (1.0 - beta) * rho_prev + beta * rho_i
                u_target = (1.0 - beta) * u_prev + beta * u_i
                rho_target = ti.max(rho_target, 1e-6)

                feq_target = self._compute_equilibrium(rho_target, u_target)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[ig, self.ny][k] = feq_target[k] + (
                        f_dst[ig, self.ny - 1][k] - feq_i[k]
                    )

                self.rho_outflow_top[i] = rho_target
                self.u_outflow_top[i] = u_target

    @ti.kernel
    def _neumann_outflow_bottom(self, f_dst: ti.template()):
        """底部 Neumann 出口"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0:
                rho_i, u_i = self._compute_macro_from_f(f_dst[ig, 2])
                rho_prev = self.rho_outflow_bottom[i]
                u_prev = self.u_outflow_bottom[i]

                beta = self.outflow_smooth_strength[None]
                rho_target = (1.0 - beta) * rho_prev + beta * rho_i
                u_target = (1.0 - beta) * u_prev + beta * u_i
                rho_target = ti.max(rho_target, 1e-6)

                feq_target = self._compute_equilibrium(rho_target, u_target)
                feq_i = self._compute_equilibrium(rho_i, u_i)

                for k in ti.static(range(9)):
                    f_dst[ig, 1][k] = feq_target[k] + (f_dst[ig, 2][k] - feq_i[k])

                self.rho_outflow_bottom[i] = rho_target
                self.u_outflow_bottom[i] = u_target

    # ==================== Neumann Outflow（質量修正版）====================

    @ti.kernel
    def _neumann_outflow_right_corrected(self, f_dst: ti.template()):
        """
        右邊界 Neumann 出口（零梯度 + 質量修正）

        Why Mass Correction?
        - 純零梯度外推不保證質量守恆：∑f_i ≠ ρ_target
        - 長時間模擬會累積質量誤差（±0.5-1% per 50k steps）
        - 修正方法：等比例縮放分佈函數

        實作方式：
        1. 零梯度外推：f[nx-1] = f[nx-2]
        2. 計算當前密度：ρ_current = ∑f_i
        3. 質量修正：f_i *= ρ_target / ρ_current

        Critical: 質量修正必須保持速度不變
        - 縮放所有 f_i 等比例 → 速度方向不變
        - 只調整密度（壓力）→ 物理合理
        """
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[self.nx, jg] == 0:
                # Step 1: 零梯度外推
                for k in ti.static(range(9)):
                    f_dst[self.nx, jg][k] = f_dst[self.nx - 1, jg][k]

                # Step 2: 計算當前密度
                rho_current = 0.0
                for k in ti.static(range(9)):
                    rho_current += f_dst[self.nx, jg][k]

                # Step 3: 質量修正（目標密度從內部推算）
                # 使用內部節點密度作為目標（避免強制特定值）
                rho_target = 0.0
                for k in ti.static(range(9)):
                    rho_target += f_dst[self.nx - 1, jg][k]

                # Step 4: 等比例縮放
                if rho_current > 1e-12:  # 避免除以零
                    correction = rho_target / rho_current
                    correction = ti.max(0.999, ti.min(1.001, correction))
                    for k in ti.static(range(9)):
                        f_dst[self.nx, jg][k] *= correction

    @ti.kernel
    def _neumann_outflow_left_corrected(self, f_dst: ti.template()):
        """左邊界 Neumann 出口（質量修正版）"""
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0:
                # 零梯度外推
                for k in ti.static(range(9)):
                    f_dst[1, jg][k] = f_dst[2, jg][k]

                # 質量修正
                rho_current = 0.0
                rho_target = 0.0
                for k in ti.static(range(9)):
                    rho_current += f_dst[1, jg][k]
                    rho_target += f_dst[2, jg][k]

                if rho_current > 1e-12:
                    correction = rho_target / rho_current
                    correction = ti.max(0.999, ti.min(1.001, correction))
                    for k in ti.static(range(9)):
                        f_dst[1, jg][k] *= correction

    @ti.kernel
    def _neumann_outflow_top_corrected(self, f_dst: ti.template()):
        """頂部 Neumann 出口（質量修正版）"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, self.ny] == 0:
                # 零梯度外推
                for k in ti.static(range(9)):
                    f_dst[ig, self.ny][k] = f_dst[ig, self.ny - 1][k]

                # 質量修正
                rho_current = 0.0
                rho_target = 0.0
                for k in ti.static(range(9)):
                    rho_current += f_dst[ig, self.ny][k]
                    rho_target += f_dst[ig, self.ny - 1][k]

                if rho_current > 1e-12:
                    correction = rho_target / rho_current
                    correction = ti.max(0.999, ti.min(1.001, correction))
                    for k in ti.static(range(9)):
                        f_dst[ig, self.ny][k] *= correction

    @ti.kernel
    def _neumann_outflow_bottom_corrected(self, f_dst: ti.template()):
        """底部 Neumann 出口（質量修正版）"""
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0:
                # 零梯度外推
                for k in ti.static(range(9)):
                    f_dst[ig, 1][k] = f_dst[ig, 2][k]

                # 質量修正
                rho_current = 0.0
                rho_target = 0.0
                for k in ti.static(range(9)):
                    rho_current += f_dst[ig, 1][k]
                    rho_target += f_dst[ig, 2][k]

                if rho_current > 1e-12:
                    correction = rho_target / rho_current
                    correction = ti.max(0.999, ti.min(1.001, correction))
                    for k in ti.static(range(9)):
                        f_dst[ig, 1][k] *= correction

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

        Zou-He for moving wall:
        - 質量守恆：rho = f0 + f1 + f3 + 2*(f2 + f5 + f6)
        - 動量守恆：rho*u_wall = f1 - f3 + f5 - f6 - f7 + f8
        """
        jg = self.ny
        for i in range(1, self.nx - 1):  # 跳過角點
            ig = i + 1
            if self.solver.mask[ig, jg] == 0:
                u_wall = self.u_wall_field[i]

                f0 = f_dst[ig, jg][0]
                f1 = f_dst[ig, jg][1]
                f2 = f_dst[ig, jg][2]
                f3 = f_dst[ig, jg][3]
                f5 = f_dst[ig, jg][5]
                f6 = f_dst[ig, jg][6]

                # 質量守恆推導密度（v_y = 0 at wall）
                rho_w = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)

                # Zou-He 重建未知分佈函數
                # 推導：f7 + f8 = f5 + f6（vy=0 約束），f8 - f7 = rho_w*u_wall - (f1-f3+f5-f6)
                # 解得：係數為 1/2（非 inlet BC 的 1/6；inlet 係數適用於法向速度，此處為切向）
                f_dst[ig, jg][4] = f2
                f_dst[ig, jg][7] = f5 - (0.5) * rho_w * u_wall + 0.5 * (f1 - f3)
                f_dst[ig, jg][8] = f6 + (0.5) * rho_w * u_wall - 0.5 * (f1 - f3)

    @ti.kernel
    def _moving_wall_bottom(self, f_dst: ti.template()):
        """底部運動壁面"""
        jg = 1
        for i in range(1, self.nx - 1):  # 跳過角點
            ig = i + 1
            if self.solver.mask[ig, jg] == 0:
                u_wall = self.u_wall_field[i]

                f0 = f_dst[ig, jg][0]
                f1 = f_dst[ig, jg][1]
                f3 = f_dst[ig, jg][3]
                f4 = f_dst[ig, jg][4]
                f7 = f_dst[ig, jg][7]
                f8 = f_dst[ig, jg][8]

                # 質量守恆推導密度（v_y = 0 at wall）
                rho_w = f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)

                # Zou-He 重建未知分佈函數（切向速度係數 1/2）
                f_dst[ig, jg][2] = f4
                f_dst[ig, jg][5] = f7 + (0.5) * rho_w * u_wall + 0.5 * (f1 - f3)
                f_dst[ig, jg][6] = f8 - (0.5) * rho_w * u_wall - 0.5 * (f1 - f3)

    # ==================== Periodic Boundary Conditions ====================

    @ti.kernel
    def _periodic_x(self, f_dst: ti.template()):
        """
        X 方向週期邊界條件（方向選擇性補缺）

        What: 補充左右邊界 push 串流後缺失的方向分量
        Why:  Push 串流後：
              - 左邊界 ig=1 缺少向東分量（k=1,5,8），因左側無 ghost 推送
              - 右邊界 ig=nx 缺少向西分量（k=3,6,7），因右側無 ghost 推送
              只補缺失分量，保留其餘正確分量，避免 full swap 汙染正確值。

        修正說明（vs. 舊版 full swap）：
            舊版 full swap 覆蓋所有 9 個分量，會把正確的西向分量（k=3,6,7）
            從右邊界「stale」值複製到左邊界，損害準確性。
            新版 direction-specific：只填寫確實缺失的方向分量，
            東向缺失（k=1,5,8）和西向缺失（k=3,6,7）是不相交集合，
            故無需保存舊值即可正確雙向填補。

        D2Q9 方向：
            東向 e_x=+1: k=1(E), k=5(NE), k=8(SE) → 左邊界 ig=1 缺失
            西向 e_x=-1: k=3(W), k=6(NW), k=7(SW) → 右邊界 ig=nx 缺失
        只在流體節點（mask=0）施加。
        """
        for j in range(self.ny):
            jg = j + 1
            # 左邊界：補填向東分量（k=1,5,8），來源：右側內部 ig=nx
            if self.solver.mask[1, jg] == 0:
                f_dst[1, jg][1] = f_dst[self.nx, jg][1]
                f_dst[1, jg][5] = f_dst[self.nx, jg][5]
                f_dst[1, jg][8] = f_dst[self.nx, jg][8]
            # 右邊界：補填向西分量（k=3,6,7），來源：左側內部 ig=1
            if self.solver.mask[self.nx, jg] == 0:
                f_dst[self.nx, jg][3] = f_dst[1, jg][3]
                f_dst[self.nx, jg][6] = f_dst[1, jg][6]
                f_dst[self.nx, jg][7] = f_dst[1, jg][7]

    @ti.kernel
    def _periodic_y(self, f_dst: ti.template()):
        """
        Y 方向週期邊界條件（方向選擇性補缺）

        What: 補充上下邊界 push 串流後缺失的方向分量
        Why:  與 _periodic_x 對稱，處理 Y 方向週期性。
              Push 串流後：
              - 底邊界 jg=1 缺少向北分量（k=2,5,6）
              - 頂邊界 jg=ny 缺少向南分量（k=4,7,8）
              只補缺失分量，不影響正確分量。

        D2Q9 方向：
            北向 e_y=+1: k=2(N), k=5(NE), k=6(NW) → 底邊界 jg=1 缺失
            南向 e_y=-1: k=4(S), k=7(SW), k=8(SE) → 頂邊界 jg=ny 缺失
        只在流體節點（mask=0）施加。
        """
        for i in range(self.nx):
            ig = i + 1
            # 底邊界：補填向北分量（k=2,5,6），來源：頂側內部 jg=ny
            if self.solver.mask[ig, 1] == 0:
                f_dst[ig, 1][2] = f_dst[ig, self.ny][2]
                f_dst[ig, 1][5] = f_dst[ig, self.ny][5]
                f_dst[ig, 1][6] = f_dst[ig, self.ny][6]
            # 頂邊界：補填向南分量（k=4,7,8），來源：底側內部 jg=1
            if self.solver.mask[ig, self.ny] == 0:
                f_dst[ig, self.ny][4] = f_dst[ig, 1][4]
                f_dst[ig, self.ny][7] = f_dst[ig, 1][7]
                f_dst[ig, self.ny][8] = f_dst[ig, 1][8]


@ti.data_oriented
class MultiphaseBoundaryConditions:
    """
    多相 LBM 邊界條件（雙組分）

    What:
    - 提供多相版本的速度入口、出口、壁面、週期邊界
    - 對 fA/fB 同時施加邊界條件

    Why:
    - 多相 solver 不可直接使用單相 Zou-He
    - 需要為兩組分佈函數重建邊界

    When:
    - 多相案例（RT、相分離）需要固定邊界條件
    """

    def __init__(self, solver):
        self.solver = solver
        self.nx = solver.nx
        self.ny = solver.ny

        self.u_inlet = ti.field(dtype=ti.f32, shape=())
        self.rho_inlet_a = ti.field(dtype=ti.f32, shape=())
        self.rho_inlet_b = ti.field(dtype=ti.f32, shape=())
        self.rho_inlet_a[None] = 0.0
        self.rho_inlet_b[None] = 0.0
        self.u_wall_field = ti.field(dtype=ti.f32, shape=self.nx)
        self.rho_target = ti.field(dtype=ti.f32, shape=())
        self.rho_target[None] = 1.0

        self.corner_extrapolation_enabled = False

    # ==================== Public API ====================

    def add_velocity_inlet(
        self,
        u_in: float,
        location: str = "left",
        method: str = "neq",
        rho_a: float | None = None,
        rho_b: float | None = None,
    ):
        """
        多相固定速度入口

        Note:
        - 對組分 A/B 分別重建
        - 使用 NEQ 或 Zou-He（簡化版）
        """
        self.u_inlet[None] = u_in
        self.rho_inlet_a[None] = 0.0 if rho_a is None else float(rho_a)
        self.rho_inlet_b[None] = 0.0 if rho_b is None else float(rho_b)
        if method == "neq":
            left_kernel = self._velocity_inlet_left_neq
            right_kernel = self._velocity_inlet_right_neq
        elif method == "zouhe":
            left_kernel = self._velocity_inlet_left
            right_kernel = self._velocity_inlet_right
        else:
            raise ValueError(f"Unknown inlet method: {method}")

        if location == "left":
            self.solver.add_boundary_condition(left_kernel, "MP Velocity Inlet (Left)")
        elif location == "right":
            self.solver.add_boundary_condition(right_kernel, "MP Velocity Inlet (Right)")
        else:
            raise NotImplementedError(f"Location '{location}' not implemented")

    def add_neumann_outflow(self, location: str = "right"):
        """多相零梯度出口"""
        if location == "right":
            self.solver.add_boundary_condition(
                self._neumann_outflow_right, "MP Neumann Outflow (Right)"
            )
        elif location == "left":
            self.solver.add_boundary_condition(
                self._neumann_outflow_left, "MP Neumann Outflow (Left)"
            )
        elif location == "top":
            self.solver.add_boundary_condition(
                self._neumann_outflow_top, "MP Neumann Outflow (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._neumann_outflow_bottom, "MP Neumann Outflow (Bottom)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_stable_outlet(
        self,
        rho_out: float = 1.0,
        location: str = "right",
        relaxation: float = 0.02,
    ):
        """多相穩定出口（弱鬆弛）"""
        self.add_orlanski_outflow(
            location=location,
            rho_target=rho_out,
            relaxation=relaxation,
        )

    def add_orlanski_outflow(
        self,
        location: str = "right",
        rho_target: float = 1.0,
        relaxation: float = 0.02,
    ):
        """
        多相非反射出口（簡化版）

        Note:
        - 以零梯度外推 + 弱密度鬆弛近似
        - 用於多相穩定性，不等同完整 Orlanski
        """
        self.rho_target[None] = rho_target
        if location == "right":
            self.solver.add_boundary_condition(
                lambda fA, fB: self._relaxed_outflow_right(fA, fB, relaxation),
                "MP Orlanski Outflow (Right)",
            )
        elif location == "left":
            self.solver.add_boundary_condition(
                lambda fA, fB: self._relaxed_outflow_left(fA, fB, relaxation),
                "MP Orlanski Outflow (Left)",
            )
        elif location == "top":
            self.solver.add_boundary_condition(
                lambda fA, fB: self._relaxed_outflow_top(fA, fB, relaxation),
                "MP Orlanski Outflow (Top)",
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                lambda fA, fB: self._relaxed_outflow_bottom(fA, fB, relaxation),
                "MP Orlanski Outflow (Bottom)",
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_no_slip_wall(self, location: str, exclude_corners: bool = False):
        """多相 No-Slip 壁面（Bounce-Back）"""
        # 固壁 -> 啟用濕潤性牆面旗標
        self.solver.enable_wall_boundary(location, enabled=True)
        if location == "top":
            self.solver.add_boundary_condition(
                self._no_slip_top, "MP No-Slip (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._no_slip_bottom, "MP No-Slip (Bottom)"
            )
        elif location == "left":
            self.solver.add_boundary_condition(
                self._no_slip_left, "MP No-Slip (Left)"
            )
        elif location == "right":
            self.solver.add_boundary_condition(
                self._no_slip_right, "MP No-Slip (Right)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_free_slip_wall(self, location: str):
        """多相 Free-Slip 壁面（鏡面反射）"""
        if location == "top":
            self.solver.add_boundary_condition(
                self._free_slip_top, "MP Free-Slip (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._free_slip_bottom, "MP Free-Slip (Bottom)"
            )
        elif location == "left":
            self.solver.add_boundary_condition(
                self._free_slip_left, "MP Free-Slip (Left)"
            )
        elif location == "right":
            self.solver.add_boundary_condition(
                self._free_slip_right, "MP Free-Slip (Right)"
            )
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_moving_wall(self, velocity_profile: np.ndarray, location: str = "top"):
        """多相運動壁面（Zou-He）"""
        if velocity_profile.shape[0] != self.nx:
            raise ValueError(
                f"Velocity profile length {velocity_profile.shape[0]} != nx {self.nx}"
            )
        self.u_wall_field.from_numpy(velocity_profile.astype(np.float32))
        self.solver.enable_wall_boundary(location, enabled=True)
        if location == "top":
            self.solver.add_boundary_condition(
                self._moving_wall_top, "MP Moving Wall (Top)"
            )
        elif location == "bottom":
            self.solver.add_boundary_condition(
                self._moving_wall_bottom, "MP Moving Wall (Bottom)"
            )
        else:
            raise NotImplementedError(
                f"Location '{location}' not implemented for moving wall"
            )

    def add_periodic_boundary(self, direction: str):
        """多相週期邊界（由 solver 控制）"""
        self.solver.set_periodic(direction, enabled=True)

    def set_wetting(self, g_wall_a: float, g_wall_b: float, psi_wall: float = 1.0):
        """
        設定濕潤性參數

        Args:
            g_wall_a: 組分 A 與壁面交互作用強度
            g_wall_b: 組分 B 與壁面交互作用強度
            psi_wall: 壁面 pseudo-potential
        """
        self.solver.set_wetting(g_wall_a, g_wall_b, psi_wall)

    def enable_wetting_wall(self, location: str, enabled: bool = True):
        """
        啟用/關閉指定邊界的濕潤性牆面旗標
        """
        self.solver.enable_wall_boundary(location, enabled=enabled)

    def handle_corners_extrapolation(self):
        """多相角點外推（平均外推）"""
        if not self.corner_extrapolation_enabled:
            self.solver.add_boundary_condition(
                self._handle_corners_extrapolation_kernel, "MP Corner Extrapolation"
            )
            self.corner_extrapolation_enabled = True

    # ==================== Helper Functions ====================

    @ti.func
    def _compute_equilibrium(self, rho, u):
        u_sq = u.dot(u)
        feq = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            e_k = ti.cast(self.solver.e[k], ti.f32)
            eu = e_k.dot(u)
            feq[k] = (
                self.solver.w[k]
                * rho
                * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
            )
        return feq

    @ti.func
    def _compute_macro_from_ab(self, fA_cell, fB_cell):
        rho_a = 0.0
        rho_b = 0.0
        momentum = ti.Vector([0.0, 0.0])
        for k in ti.static(range(9)):
            rho_a += fA_cell[k]
            rho_b += fB_cell[k]
            momentum += (fA_cell[k] + fB_cell[k]) * ti.cast(self.solver.e[k], ti.f32)
        rho = rho_a + rho_b
        u = ti.Vector([0.0, 0.0])
        if rho > 1e-12:
            u = momentum / rho
        return rho_a, rho_b, rho, u

    # ==================== Inlet (Zou-He / NEQ) ====================

    @ti.kernel
    def _velocity_inlet_left(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            u_in = self.u_inlet[None]

            # 組分 A
            f0 = fA[1, jg][0]
            f2 = fA[1, jg][2]
            f3 = fA[1, jg][3]
            f4 = fA[1, jg][4]
            f6 = fA[1, jg][6]
            f7 = fA[1, jg][7]
            rho_a = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
            fA[1, jg][1] = f3 + (2.0 / 3.0) * rho_a * u_in
            fA[1, jg][5] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_a * u_in
            fA[1, jg][8] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_a * u_in

            # 組分 B
            f0 = fB[1, jg][0]
            f2 = fB[1, jg][2]
            f3 = fB[1, jg][3]
            f4 = fB[1, jg][4]
            f6 = fB[1, jg][6]
            f7 = fB[1, jg][7]
            rho_b = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
            fB[1, jg][1] = f3 + (2.0 / 3.0) * rho_b * u_in
            fB[1, jg][5] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_b * u_in
            fB[1, jg][8] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_b * u_in

    @ti.kernel
    def _velocity_inlet_right(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            u_in = -self.u_inlet[None]

            f0 = fA[self.nx, jg][0]
            f1 = fA[self.nx, jg][1]
            f2 = fA[self.nx, jg][2]
            f4 = fA[self.nx, jg][4]
            f5 = fA[self.nx, jg][5]
            f8 = fA[self.nx, jg][8]
            rho_a = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + u_in)
            fA[self.nx, jg][3] = f1 - (2.0 / 3.0) * rho_a * u_in
            fA[self.nx, jg][7] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_a * u_in
            fA[self.nx, jg][6] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_a * u_in

            f0 = fB[self.nx, jg][0]
            f1 = fB[self.nx, jg][1]
            f2 = fB[self.nx, jg][2]
            f4 = fB[self.nx, jg][4]
            f5 = fB[self.nx, jg][5]
            f8 = fB[self.nx, jg][8]
            rho_b = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + u_in)
            fB[self.nx, jg][3] = f1 - (2.0 / 3.0) * rho_b * u_in
            fB[self.nx, jg][7] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_b * u_in
            fB[self.nx, jg][6] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_b * u_in

    @ti.kernel
    def _velocity_inlet_left_neq(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            u_in = self.u_inlet[None]
            rho_a_i, rho_b_i, _, u_i = self._compute_macro_from_ab(
                fA[2, jg], fB[2, jg]
            )
            u_b = ti.Vector([u_in, 0.0])
            rho_a = rho_a_i
            rho_b = rho_b_i
            if self.rho_inlet_a[None] > 0.0:
                rho_a = self.rho_inlet_a[None]
            if self.rho_inlet_b[None] > 0.0:
                rho_b = self.rho_inlet_b[None]

            feq_a_b = self._compute_equilibrium(rho_a, u_b)
            feq_a_i = self._compute_equilibrium(rho_a_i, u_i)
            feq_b_b = self._compute_equilibrium(rho_b, u_b)
            feq_b_i = self._compute_equilibrium(rho_b_i, u_i)

            for k in ti.static(range(9)):
                fA[1, jg][k] = feq_a_b[k] + (fA[2, jg][k] - feq_a_i[k])
                fB[1, jg][k] = feq_b_b[k] + (fB[2, jg][k] - feq_b_i[k])

    @ti.kernel
    def _velocity_inlet_right_neq(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            u_in = -self.u_inlet[None]
            rho_a_i, rho_b_i, _, u_i = self._compute_macro_from_ab(
                fA[self.nx - 1, jg], fB[self.nx - 1, jg]
            )
            u_b = ti.Vector([u_in, 0.0])
            rho_a = rho_a_i
            rho_b = rho_b_i
            if self.rho_inlet_a[None] > 0.0:
                rho_a = self.rho_inlet_a[None]
            if self.rho_inlet_b[None] > 0.0:
                rho_b = self.rho_inlet_b[None]

            feq_a_b = self._compute_equilibrium(rho_a, u_b)
            feq_a_i = self._compute_equilibrium(rho_a_i, u_i)
            feq_b_b = self._compute_equilibrium(rho_b, u_b)
            feq_b_i = self._compute_equilibrium(rho_b_i, u_i)

            for k in ti.static(range(9)):
                fA[self.nx, jg][k] = feq_a_b[k] + (
                    fA[self.nx - 1, jg][k] - feq_a_i[k]
                )
                fB[self.nx, jg][k] = feq_b_b[k] + (
                    fB[self.nx - 1, jg][k] - feq_b_i[k]
                )

    # ==================== Neumann Outflow ====================

    @ti.kernel
    def _neumann_outflow_right(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            for k in ti.static(range(9)):
                fA[self.nx, jg][k] = fA[self.nx - 1, jg][k]
                fB[self.nx, jg][k] = fB[self.nx - 1, jg][k]

    @ti.kernel
    def _neumann_outflow_left(self, fA: ti.template(), fB: ti.template()):
        for j in range(self.ny):
            jg = j + 1
            for k in ti.static(range(9)):
                fA[1, jg][k] = fA[2, jg][k]
                fB[1, jg][k] = fB[2, jg][k]

    @ti.kernel
    def _neumann_outflow_top(self, fA: ti.template(), fB: ti.template()):
        for i in range(self.nx):
            ig = i + 1
            for k in ti.static(range(9)):
                fA[ig, self.ny][k] = fA[ig, self.ny - 1][k]
                fB[ig, self.ny][k] = fB[ig, self.ny - 1][k]

    @ti.kernel
    def _neumann_outflow_bottom(self, fA: ti.template(), fB: ti.template()):
        for i in range(self.nx):
            ig = i + 1
            for k in ti.static(range(9)):
                fA[ig, 1][k] = fA[ig, 2][k]
                fB[ig, 1][k] = fB[ig, 2][k]

    # ==================== Relaxed Outflow ====================

    @ti.kernel
    def _relaxed_outflow_right(
        self, fA: ti.template(), fB: ti.template(), relax: ti.f32
    ):
        for j in range(self.ny):
            jg = j + 1
            for k in ti.static(range(9)):
                fA[self.nx, jg][k] = fA[self.nx - 1, jg][k]
                fB[self.nx, jg][k] = fB[self.nx - 1, jg][k]

            rho_a, rho_b, rho, _ = self._compute_macro_from_ab(
                fA[self.nx, jg], fB[self.nx, jg]
            )
            rho_target = (1.0 - relax) * rho + relax * self.rho_target[None]
            if rho > 1e-12:
                ratio_a = rho_a / rho
                ratio_b = rho_b / rho
                corr_a = (ratio_a * rho_target) / rho_a
                corr_b = (ratio_b * rho_target) / rho_b
                for k in ti.static(range(9)):
                    fA[self.nx, jg][k] *= corr_a
                    fB[self.nx, jg][k] *= corr_b

    @ti.kernel
    def _relaxed_outflow_left(
        self, fA: ti.template(), fB: ti.template(), relax: ti.f32
    ):
        for j in range(self.ny):
            jg = j + 1
            for k in ti.static(range(9)):
                fA[1, jg][k] = fA[2, jg][k]
                fB[1, jg][k] = fB[2, jg][k]

            rho_a, rho_b, rho, _ = self._compute_macro_from_ab(
                fA[1, jg], fB[1, jg]
            )
            rho_target = (1.0 - relax) * rho + relax * self.rho_target[None]
            if rho > 1e-12:
                ratio_a = rho_a / rho
                ratio_b = rho_b / rho
                corr_a = (ratio_a * rho_target) / rho_a
                corr_b = (ratio_b * rho_target) / rho_b
                for k in ti.static(range(9)):
                    fA[1, jg][k] *= corr_a
                    fB[1, jg][k] *= corr_b

    @ti.kernel
    def _relaxed_outflow_top(
        self, fA: ti.template(), fB: ti.template(), relax: ti.f32
    ):
        for i in range(self.nx):
            ig = i + 1
            for k in ti.static(range(9)):
                fA[ig, self.ny][k] = fA[ig, self.ny - 1][k]
                fB[ig, self.ny][k] = fB[ig, self.ny - 1][k]

            rho_a, rho_b, rho, _ = self._compute_macro_from_ab(
                fA[ig, self.ny], fB[ig, self.ny]
            )
            rho_target = (1.0 - relax) * rho + relax * self.rho_target[None]
            if rho > 1e-12:
                ratio_a = rho_a / rho
                ratio_b = rho_b / rho
                corr_a = (ratio_a * rho_target) / rho_a
                corr_b = (ratio_b * rho_target) / rho_b
                for k in ti.static(range(9)):
                    fA[ig, self.ny][k] *= corr_a
                    fB[ig, self.ny][k] *= corr_b

    @ti.kernel
    def _relaxed_outflow_bottom(
        self, fA: ti.template(), fB: ti.template(), relax: ti.f32
    ):
        for i in range(self.nx):
            ig = i + 1
            for k in ti.static(range(9)):
                fA[ig, 1][k] = fA[ig, 2][k]
                fB[ig, 1][k] = fB[ig, 2][k]

            rho_a, rho_b, rho, _ = self._compute_macro_from_ab(
                fA[ig, 1], fB[ig, 1]
            )
            rho_target = (1.0 - relax) * rho + relax * self.rho_target[None]
            if rho > 1e-12:
                ratio_a = rho_a / rho
                ratio_b = rho_b / rho
                corr_a = (ratio_a * rho_target) / rho_a
                corr_b = (ratio_b * rho_target) / rho_b
                for k in ti.static(range(9)):
                    fA[ig, 1][k] *= corr_a
                    fB[ig, 1][k] *= corr_b

    # ==================== Free-Slip ====================

    @ti.kernel
    def _free_slip_top(self, fA: ti.template(), fB: ti.template()):
        j = self.ny
        for i in range(self.nx):
            ig = i + 1
            fA[ig, j][2] = fA[ig, j][4]
            fA[ig, j][5] = fA[ig, j][8]
            fA[ig, j][6] = fA[ig, j][7]

            fB[ig, j][2] = fB[ig, j][4]
            fB[ig, j][5] = fB[ig, j][8]
            fB[ig, j][6] = fB[ig, j][7]

    @ti.kernel
    def _free_slip_bottom(self, fA: ti.template(), fB: ti.template()):
        j = 1
        for i in range(self.nx):
            ig = i + 1
            fA[ig, j][4] = fA[ig, j][2]
            fA[ig, j][7] = fA[ig, j][6]
            fA[ig, j][8] = fA[ig, j][5]

            fB[ig, j][4] = fB[ig, j][2]
            fB[ig, j][7] = fB[ig, j][6]
            fB[ig, j][8] = fB[ig, j][5]

    @ti.kernel
    def _free_slip_left(self, fA: ti.template(), fB: ti.template()):
        i = 1
        for j in range(self.ny):
            jg = j + 1
            fA[i, jg][3] = fA[i, jg][1]
            fA[i, jg][6] = fA[i, jg][5]
            fA[i, jg][7] = fA[i, jg][8]

            fB[i, jg][3] = fB[i, jg][1]
            fB[i, jg][6] = fB[i, jg][5]
            fB[i, jg][7] = fB[i, jg][8]

    @ti.kernel
    def _free_slip_right(self, fA: ti.template(), fB: ti.template()):
        i = self.nx
        for j in range(self.ny):
            jg = j + 1
            fA[i, jg][1] = fA[i, jg][3]
            fA[i, jg][5] = fA[i, jg][6]
            fA[i, jg][8] = fA[i, jg][7]

            fB[i, jg][1] = fB[i, jg][3]
            fB[i, jg][5] = fB[i, jg][6]
            fB[i, jg][8] = fB[i, jg][7]

    # ==================== No-Slip (Bounce-Back) ====================

    @ti.kernel
    def _no_slip_top(self, fA: ti.template(), fB: ti.template()):
        j = self.ny
        for i in range(self.nx):
            ig = i + 1
            fA[ig, j][2] = fA[ig, j][4]
            fA[ig, j][5] = fA[ig, j][7]
            fA[ig, j][6] = fA[ig, j][8]

            fB[ig, j][2] = fB[ig, j][4]
            fB[ig, j][5] = fB[ig, j][7]
            fB[ig, j][6] = fB[ig, j][8]

    @ti.kernel
    def _no_slip_bottom(self, fA: ti.template(), fB: ti.template()):
        j = 1
        for i in range(self.nx):
            ig = i + 1
            fA[ig, j][4] = fA[ig, j][2]
            fA[ig, j][7] = fA[ig, j][5]
            fA[ig, j][8] = fA[ig, j][6]

            fB[ig, j][4] = fB[ig, j][2]
            fB[ig, j][7] = fB[ig, j][5]
            fB[ig, j][8] = fB[ig, j][6]

    @ti.kernel
    def _no_slip_left(self, fA: ti.template(), fB: ti.template()):
        i = 1
        for j in range(self.ny):
            jg = j + 1
            fA[i, jg][3] = fA[i, jg][1]
            fA[i, jg][6] = fA[i, jg][8]
            fA[i, jg][7] = fA[i, jg][5]

            fB[i, jg][3] = fB[i, jg][1]
            fB[i, jg][6] = fB[i, jg][8]
            fB[i, jg][7] = fB[i, jg][5]

    @ti.kernel
    def _no_slip_right(self, fA: ti.template(), fB: ti.template()):
        i = self.nx
        for j in range(self.ny):
            jg = j + 1
            fA[i, jg][1] = fA[i, jg][3]
            fA[i, jg][5] = fA[i, jg][7]
            fA[i, jg][8] = fA[i, jg][6]

            fB[i, jg][1] = fB[i, jg][3]
            fB[i, jg][5] = fB[i, jg][7]
            fB[i, jg][8] = fB[i, jg][6]

    # ==================== Moving Wall ====================

    @ti.kernel
    def _moving_wall_top(self, fA: ti.template(), fB: ti.template()):
        jg = self.ny
        for i in range(1, self.nx - 1):
            ig = i + 1
            u_wall = self.u_wall_field[i]

            # 組分 A
            f0 = fA[ig, jg][0]
            f1 = fA[ig, jg][1]
            f2 = fA[ig, jg][2]
            f3 = fA[ig, jg][3]
            f5 = fA[ig, jg][5]
            f6 = fA[ig, jg][6]
            rho_a = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)
            fA[ig, jg][4] = f2
            fA[ig, jg][7] = f5 - (1.0 / 6.0) * rho_a * u_wall + 0.5 * (f1 - f3)
            fA[ig, jg][8] = f6 + (1.0 / 6.0) * rho_a * u_wall - 0.5 * (f1 - f3)

            # 組分 B
            f0 = fB[ig, jg][0]
            f1 = fB[ig, jg][1]
            f2 = fB[ig, jg][2]
            f3 = fB[ig, jg][3]
            f5 = fB[ig, jg][5]
            f6 = fB[ig, jg][6]
            rho_b = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)
            fB[ig, jg][4] = f2
            fB[ig, jg][7] = f5 - (1.0 / 6.0) * rho_b * u_wall + 0.5 * (f1 - f3)
            fB[ig, jg][8] = f6 + (1.0 / 6.0) * rho_b * u_wall - 0.5 * (f1 - f3)

    @ti.kernel
    def _moving_wall_bottom(self, fA: ti.template(), fB: ti.template()):
        jg = 1
        for i in range(1, self.nx - 1):
            ig = i + 1
            u_wall = self.u_wall_field[i]

            f0 = fA[ig, jg][0]
            f1 = fA[ig, jg][1]
            f3 = fA[ig, jg][3]
            f4 = fA[ig, jg][4]
            f7 = fA[ig, jg][7]
            f8 = fA[ig, jg][8]
            rho_a = f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)
            fA[ig, jg][2] = f4
            fA[ig, jg][5] = f7 + (1.0 / 6.0) * rho_a * u_wall + 0.5 * (f1 - f3)
            fA[ig, jg][6] = f8 - (1.0 / 6.0) * rho_a * u_wall - 0.5 * (f1 - f3)

            f0 = fB[ig, jg][0]
            f1 = fB[ig, jg][1]
            f3 = fB[ig, jg][3]
            f4 = fB[ig, jg][4]
            f7 = fB[ig, jg][7]
            f8 = fB[ig, jg][8]
            rho_b = f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)
            fB[ig, jg][2] = f4
            fB[ig, jg][5] = f7 + (1.0 / 6.0) * rho_b * u_wall + 0.5 * (f1 - f3)
            fB[ig, jg][6] = f8 - (1.0 / 6.0) * rho_b * u_wall - 0.5 * (f1 - f3)

    # ==================== Corner Extrapolation ====================

    @ti.kernel
    def _handle_corners_extrapolation_kernel(self, fA: ti.template(), fB: ti.template()):
        for k in ti.static(range(9)):
            # 左下角
            fA[1, 1][k] = 0.5 * (fA[2, 1][k] + fA[1, 2][k])
            fB[1, 1][k] = 0.5 * (fB[2, 1][k] + fB[1, 2][k])

            # 右下角
            fA[self.nx, 1][k] = 0.5 * (
                fA[self.nx - 1, 1][k] + fA[self.nx, 2][k]
            )
            fB[self.nx, 1][k] = 0.5 * (
                fB[self.nx - 1, 1][k] + fB[self.nx, 2][k]
            )

            # 左上角
            fA[1, self.ny][k] = 0.5 * (
                fA[2, self.ny][k] + fA[1, self.ny - 1][k]
            )
            fB[1, self.ny][k] = 0.5 * (
                fB[2, self.ny][k] + fB[1, self.ny - 1][k]
            )

            # 右上角
            fA[self.nx, self.ny][k] = 0.5 * (
                fA[self.nx - 1, self.ny][k] + fA[self.nx, self.ny - 1][k]
            )
            fB[self.nx, self.ny][k] = 0.5 * (
                fB[self.nx - 1, self.ny][k] + fB[self.nx, self.ny - 1][k]
            )

    # ==================== Periodic Boundary Conditions ====================

    @ti.kernel
    def _periodic_x(self, f_dst: ti.template()):
        """
        X 方向週期邊界條件

        Physical Meaning:
        - 左邊界 = 右邊界內部（流體從右流出，從左流入）
        - 右邊界 = 左邊界內部（流體從左流出，從右流入）

        Implementation:
        - f[0, j] = f[nx-2, j]（左邊界複製右側內部）
        - f[nx-1, j] = f[1, j]（右邊界複製左側內部）

        Why nx-2 and 1?
        - 邊界節點（0 和 nx-1）用於接收
        - 內部節點（1 和 nx-2）作為資料來源
        - 避免邊界與邊界直接複製（會產生時間延遲）

        Critical: 只在流體節點（mask=0）施加
        - 如果邊界有障礙物（mask=1），跳過
        """
        for j in range(self.ny):
            jg = j + 1
            # 左邊界：從右側內部複製
            if self.solver.mask[1, jg] == 0:
                for k in ti.static(range(9)):
                    f_dst[1, jg][k] = f_dst[self.nx - 1, jg][k]

            # 右邊界：從左側內部複製
            if self.solver.mask[self.nx, jg] == 0:
                for k in ti.static(range(9)):
                    f_dst[self.nx, jg][k] = f_dst[2, jg][k]

    @ti.kernel
    def _periodic_x_direct(self, f_dst: ti.template()):
        """
        X 方向週期邊界條件（直接交換）

        注意：直接交換左右邊界的分佈函數
        """
        for j in range(self.ny):
            jg = j + 1
            if self.solver.mask[1, jg] == 0 and self.solver.mask[self.nx, jg] == 0:
                for k in ti.static(range(9)):
                    left_val = f_dst[1, jg][k]
                    right_val = f_dst[self.nx, jg][k]
                    f_dst[1, jg][k] = right_val
                    f_dst[self.nx, jg][k] = left_val

    @ti.kernel
    def _periodic_y(self, f_dst: ti.template()):
        """
        Y 方向週期邊界條件

        Physical Meaning:
        - 底部邊界 = 頂部內部
        - 頂部邊界 = 底部內部

        Implementation:
        - f[i, 0] = f[i, ny-2]（底部複製頂部內部）
        - f[i, ny-1] = f[i, 1]（頂部複製底部內部）

        Use Case:
        - 垂直週期性管道流
        - Taylor-Green Vortex（雙向週期）
        - 湍流通道流
        """
        for i in range(self.nx):
            ig = i + 1
            # 底部邊界：從頂部內部複製
            if self.solver.mask[ig, 1] == 0:
                for k in ti.static(range(9)):
                    f_dst[ig, 1][k] = f_dst[ig, self.ny - 1][k]

            # 頂部邊界：從底部內部複製
            if self.solver.mask[ig, self.ny] == 0:
                for k in ti.static(range(9)):
                    f_dst[ig, self.ny][k] = f_dst[ig, 2][k]

    @ti.kernel
    def _periodic_y_direct(self, f_dst: ti.template()):
        """
        Y 方向週期邊界條件（直接交換）
        """
        for i in range(self.nx):
            ig = i + 1
            if self.solver.mask[ig, 1] == 0 and self.solver.mask[ig, self.ny] == 0:
                for k in ti.static(range(9)):
                    bottom_val = f_dst[ig, 1][k]
                    top_val = f_dst[ig, self.ny][k]
                    f_dst[ig, 1][k] = top_val
                    f_dst[ig, self.ny][k] = bottom_val
