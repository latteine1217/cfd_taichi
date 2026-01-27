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
        self.outflow_rho_target_right = ti.field(dtype=ti.f32, shape=())
        self.outflow_rho_target_left = ti.field(dtype=ti.f32, shape=())
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

    def add_velocity_inlet(
        self, u_in: float, location: str = "left", mode: str = "zouhe"
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
            mode: 'zouhe' 或 'neq'（Guo 非平衡外推）
        """
        self.u_inlet[None] = u_in

        if mode not in {"zouhe", "neq"}:
            raise ValueError(f"Invalid inlet mode: {mode}")

        if location == "left":
            kernel = (
                self._velocity_inlet_left
                if mode == "zouhe"
                else self._velocity_inlet_left_neq
            )
            self.solver.add_boundary_condition(kernel, f"Velocity Inlet (Left, {mode})")
        elif location == "right":
            kernel = (
                self._velocity_inlet_right
                if mode == "zouhe"
                else self._velocity_inlet_right_neq
            )
            self.solver.add_boundary_condition(
                kernel, f"Velocity Inlet (Right, {mode})"
            )
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
        relaxation: float = 0.2,
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

    def add_periodic_boundary(self, direction: str, mode: str = "buffered"):
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
            mode: 'buffered' 使用內部節點緩衝, 'direct' 直接對邊界交換
        """
        if direction == "x":
            if mode == "buffered":
                self.solver.add_boundary_condition(
                    self._periodic_x, "Periodic (X-direction, Buffered)"
                )
            elif mode == "direct":
                self.solver.add_boundary_condition(
                    self._periodic_x_direct, "Periodic (X-direction, Direct)"
                )
            else:
                raise ValueError(f"Invalid periodic mode: {mode}")
        elif direction == "y":
            if mode == "buffered":
                self.solver.add_boundary_condition(
                    self._periodic_y, "Periodic (Y-direction, Buffered)"
                )
            elif mode == "direct":
                self.solver.add_boundary_condition(
                    self._periodic_y_direct, "Periodic (Y-direction, Direct)"
                )
            else:
                raise ValueError(f"Invalid periodic mode: {mode}")
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'x' or 'y'.")

    def add_free_slip_wall(self, location: str, mode: str = "symmetric"):
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
            mode: 'symmetric' 對稱延拓, 'zouhe' Zou-He 重建
        """
        if mode not in {"symmetric", "zouhe"}:
            raise ValueError(f"Invalid free-slip mode: {mode}")

        if location == "top":
            kernel = (
                self._free_slip_top_symmetric
                if mode == "symmetric"
                else self._free_slip_top
            )
            self.solver.add_boundary_condition(kernel, f"Free-Slip (Top, {mode})")
        elif location == "bottom":
            kernel = (
                self._free_slip_bottom_symmetric
                if mode == "symmetric"
                else self._free_slip_bottom
            )
            self.solver.add_boundary_condition(kernel, f"Free-Slip (Bottom, {mode})")
        elif location == "left":
            kernel = (
                self._free_slip_left_symmetric
                if mode == "symmetric"
                else self._free_slip_left
            )
            self.solver.add_boundary_condition(kernel, f"Free-Slip (Left, {mode})")
        elif location == "right":
            kernel = (
                self._free_slip_right_symmetric
                if mode == "symmetric"
                else self._free_slip_right
            )
            self.solver.add_boundary_condition(kernel, f"Free-Slip (Right, {mode})")
        else:
            raise ValueError(f"Invalid location: {location}")

    def add_moving_wall(
        self,
        velocity_profile: np.ndarray,
        location: str = "top",
        handle_corners: bool = True,
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
            handle_corners: 是否啟用角點外推（避免角點奇異）
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
            self.handle_corners_extrapolation()

    @ti.kernel
    def _init_mass_correction(self):
        self.mass_correction_interval[None] = 200
        self.mass_correction_strength[None] = 0.1
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
            location: 'right' 或 'left'
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
        self.outflow_rho_target_right[None] = 1.0
        self.outflow_rho_target_left[None] = 1.0
        self.outflow_smooth_strength[None] = 0.2

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
                f0 = f_dst[1, jg][0]
                f2 = f_dst[1, jg][2]
                f3 = f_dst[1, jg][3]
                f4 = f_dst[1, jg][4]
                f6 = f_dst[1, jg][6]
                f7 = f_dst[1, jg][7]

                u_in = self.u_inlet[None]
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
                rho_i, u_i = self._compute_macro_from_f(f_dst[2, jg])
                u_b = ti.Vector([self.u_inlet[None], 0.0])
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
                f0 = f_dst[self.nx, jg][0]
                f1 = f_dst[self.nx, jg][1]
                f2 = f_dst[self.nx, jg][2]
                f4 = f_dst[self.nx, jg][4]
                f5 = f_dst[self.nx, jg][5]
                f8 = f_dst[self.nx, jg][8]

                u_in = -self.u_inlet[None]  # 向左流入
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
                rho_i, u_i = self._compute_macro_from_f(f_dst[self.nx - 1, jg])
                u_b = ti.Vector([-self.u_inlet[None], 0.0])
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

                # Zou-He 重建未知分佈函數
                f_dst[ig, 1][2] = f4
                f_dst[ig, 1][5] = f7 + (1.0 / 6.0) * rho * u_x
                f_dst[ig, 1][6] = f8 - (1.0 / 6.0) * rho * u_x

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

                # Zou-He 重建
                f_dst[ig, self.ny][4] = f2
                f_dst[ig, self.ny][7] = f6 - (1.0 / 6.0) * rho * u_x
                f_dst[ig, self.ny][8] = f5 + (1.0 / 6.0) * rho * u_x

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

                # Zou-He 重建
                f_dst[1, jg][1] = f3
                f_dst[1, jg][5] = f7 + (1.0 / 6.0) * rho * u_y
                f_dst[1, jg][8] = f6 - (1.0 / 6.0) * rho * u_y

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

                # Zou-He 重建
                f_dst[self.nx, jg][3] = f1
                f_dst[self.nx, jg][6] = f8 - (1.0 / 6.0) * rho * u_y
                f_dst[self.nx, jg][7] = f5 + (1.0 / 6.0) * rho * u_y

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
                f_dst[ig, jg][4] = f2
                f_dst[ig, jg][7] = f5 - (1.0 / 6.0) * rho_w * u_wall + 0.5 * (f1 - f3)
                f_dst[ig, jg][8] = f6 + (1.0 / 6.0) * rho_w * u_wall - 0.5 * (f1 - f3)

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

                # Zou-He 重建未知分佈函數
                f_dst[ig, jg][2] = f4
                f_dst[ig, jg][5] = f7 + (1.0 / 6.0) * rho_w * u_wall + 0.5 * (f1 - f3)
                f_dst[ig, jg][6] = f8 - (1.0 / 6.0) * rho_w * u_wall - 0.5 * (f1 - f3)

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
