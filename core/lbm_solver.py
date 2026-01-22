"""
核心 LBM 求解器
===============

統一的 D2Q9 MRT-LBM 求解器實作，支援：
- Multiple Relaxation Time (MRT) 碰撞算子
- Smagorinsky LES 湍流模型
- 可插拔的邊界條件系統
- 障礙物處理（Bounce-Back）
- 質量守恆與動量殘差檢查

設計原則：
- 模組化：邊界條件、碰撞模型分離
- 簡潔性：核心邏輯清晰，避免不必要複雜度
- 可驗證：物理量可追蹤，殘差可計算
"""

import taichi as ti
import numpy as np
from typing import Optional, Callable


@ti.data_oriented
class LBMSolver:
    """
    統一的 D2Q9 MRT-LBM 求解器

    Why MRT over BGK?
    - 更好的數值穩定性
    - 可獨立調整不同物理量的鬆弛時間
    - 支援 LES 湍流模型

    Why D2Q9?
    - 2D 問題的標準格子
    - 平衡精度與效能
    - 足夠捕捉 Navier-Stokes 物理
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        re: float = 100.0,
        u_ref: float = 0.1,
        length_scale: Optional[float] = None,
        cs: float = 0.16,
    ):
        """
        Args:
            nx: X 方向格點數
            ny: Y 方向格點數
            re: Reynolds 數 (U*L/ν)
            u_ref: 參考速度 (lattice units)
            length_scale: 特徵長度，若為 None 則使用 ny/9.0
            cs: Smagorinsky 常數 (0 表示不使用 LES)
        """
        self.nx = nx
        self.ny = ny
        self.re = re
        self.u_ref = u_ref
        self.cs = cs

        # 物理參數計算
        self.L_char = length_scale if length_scale is not None else ny / 9.0
        self.nu = self.u_ref * self.L_char / self.re
        self.tau = 3.0 * self.nu + 0.5

        # === D2Q9 常數 ===
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)

        # === 場變數（使用 AoS 佈局優化 cache locality）===
        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))

        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.mask = ti.field(dtype=ti.i32, shape=(nx, ny))  # 1=固體, 0=流體

        # === 診斷場 ===
        self.prev_rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.prev_u = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))

        # === 全局診斷量 ===
        self.total_mass = ti.field(dtype=ti.f32, shape=())
        self.initial_mass = ti.field(dtype=ti.f32, shape=())
        self.mass_residual = ti.field(dtype=ti.f32, shape=())
        self.mom_res_x = ti.field(dtype=ti.f32, shape=())
        self.mom_res_y = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_x = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_y = ti.field(dtype=ti.f32, shape=())
        self.max_u = ti.field(dtype=ti.f32, shape=())

        # === 粒子系統 (粒子追蹤可視化) ===
        self.num_particles = 500000
        self.px = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.py = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.p_active = ti.field(dtype=ti.i32, shape=self.num_particles)
        self.emitter_ptr = ti.field(dtype=ti.i32, shape=())

        # === MRT 矩陣 ===
        self.M = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.M_inv = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.S = ti.field(dtype=ti.f32, shape=9)

        self.bc_functions = []
        self._init_constants()
        self.reset()

    # --- 粒子系統 Kernels ---
    @ti.kernel
    def _init_particles(self):
        self.emitter_ptr[None] = 0
        for i in range(self.num_particles):
            self.p_active[i] = 0

    @ti.kernel
    def _emit_particles(self, num_lines: int):
        stride = self.ny / num_lines
        for l in range(num_lines):
            idx = (self.emitter_ptr[None] + l) % self.num_particles
            self.p_active[idx] = 1
            self.px[idx] = 2.0  # 入口稍微靠內
            self.py[idx] = (l + 0.5) * stride
        self.emitter_ptr[None] = (self.emitter_ptr[None] + num_lines) % self.num_particles

    @ti.func
    def _sample_u(self, x: float, y: float):
        """雙線性插值採樣速度場"""
        i, j = int(x), int(y)
        i = ti.max(0, ti.min(self.nx - 2, i))
        j = ti.max(0, ti.min(self.ny - 2, j))
        dx, dy = x - i, y - j
        u00, u10 = self.u[i, j], self.u[i+1, j]
        u01, u11 = self.u[i, j+1], self.u[i+1, j+1]
        return u00*(1-dx)*(1-dy) + u10*dx*(1-dy) + u01*(1-dx)*dy + u11*dx*dy

    @ti.kernel
    def _advect_particles(self):
        """更新粒子位置"""
        for i in range(self.num_particles):
            if self.p_active[i] == 1:
                vel = self._sample_u(self.px[i], self.py[i])
                self.px[i] += vel[0]
                self.py[i] += vel[1]
                # 邊界與障礙物檢查
                if (self.px[i] < 0 or self.px[i] >= self.nx or 
                    self.py[i] < 0 or self.py[i] >= self.ny):
                    self.p_active[i] = 0
                else:
                    if self.mask[int(self.px[i]), int(self.py[i])] == 1:
                        self.p_active[i] = 0

    def step_particles(self, emit: bool = True, num_lines: int = 32):
        if emit: self._emit_particles(num_lines)
        self._advect_particles()

    def reset(self):
        self._init_particles()
        self._reset_fields()

    def _init_constants(self):
        """初始化 D2Q9 常數與 MRT 矩陣"""
        # D2Q9 權重
        w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                         1/36, 1/36, 1/36, 1/36], dtype=np.float32)

        # D2Q9 格子速度向量
        e_np = np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ], dtype=np.int32)

        # 反向索引（用於 Bounce-Back）
        inv_np = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

        # MRT 變換矩陣 (速度空間 -> 矩量空間)
        # 對應矩量：rho, e, epsilon, j_x, q_x, j_y, q_y, p_xx, p_xy
        M_np = np.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1],
            [-4, -1, -1, -1, -1,  2,  2,  2,  2],
            [4, -2, -2, -2, -2,  1,  1,  1,  1],
            [0,  1,  0, -1,  0,  1, -1, -1,  1],
            [0, -2,  0,  2,  0,  1, -1, -1,  1],
            [0,  0,  1,  0, -1,  1,  1, -1, -1],
            [0,  0, -2,  0,  2,  1,  1, -1, -1],
            [0,  1, -1,  1, -1,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  1, -1,  1, -1]
        ], dtype=np.float32)

        M_inv_np = np.linalg.inv(M_np).astype(np.float32)

        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)
        self._set_mrt_matrices(M_np, M_inv_np)

    @ti.kernel
    def _set_mrt_matrices(self, m_arr: ti.types.ndarray(), minv_arr: ti.types.ndarray()):
        """設定 MRT 矩陣與鬆弛參數"""
        for i, j in ti.ndrange(9, 9):
            self.M[None][i, j] = m_arr[i, j]
            self.M_inv[None][i, j] = minv_arr[i, j]

        # 鬆弛參數設定
        # k=7,8 對應應力張量，使用黏度相關的 s_nu
        # 其他高階矩使用較大的 s_other (更快鬆弛，提高穩定性)
        s_nu = 1.0 / self.tau
        s_other = 1.2

        self.S[0] = 0.0       # 質量守恆（不鬆弛）
        self.S[1] = s_other   # 能量相關
        self.S[2] = s_other
        self.S[3] = 0.0       # 動量守恆（不鬆弛）
        self.S[4] = s_other
        self.S[5] = 0.0       # 動量守恆（不鬆弛）
        self.S[6] = s_other
        self.S[7] = s_nu      # 應力張量（黏度）
        self.S[8] = s_nu      # 應力張量（黏度）

    @ti.kernel
    def _reset_fields(self):
        """重置所有場到初始狀態"""
        for i, j in self.rho:
            self.mask[i, j] = 0
            self.rho[i, j] = 1.0
            self.prev_rho[i, j] = 1.0
            self.u[i, j] = ti.Vector([self.u_ref, 0.0])
            self.prev_u[i, j] = ti.Vector([self.u_ref, 0.0])

            # 初始化為平衡分佈
            u_vec = self.u[i, j]
            rho_val = self.rho[i, j]
            u_sq = u_vec.norm_sqr()

            for k in ti.static(range(9)):
                eu = self.e[k].dot(u_vec)
                f_eq = self.w[k] * rho_val * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)
                self.f[i, j][k] = f_eq
                self.f_new[i, j][k] = f_eq

    def reset(self):
        """公開介面：重置求解器"""
        self._reset_fields()

    def set_obstacle(self, mask_array: np.ndarray):
        """
        設定固體障礙物

        Args:
            mask_array: (nx, ny) 或 (ny, nx) 的 numpy 陣列，1=固體，0=流體
        """
        if mask_array.shape == (self.nx, self.ny):
            self.mask.from_numpy(mask_array.astype(np.int32))
        elif mask_array.shape == (self.ny, self.nx):
            self.mask.from_numpy(mask_array.T.astype(np.int32))
        else:
            raise ValueError(
                f"Mask shape {mask_array.shape} incompatible with grid ({self.nx}, {self.ny})"
            )

        # 修正固體區域的速度與分佈函數
        self._correct_solid_velocity()

    @ti.kernel
    def _correct_solid_velocity(self):
        """將固體區域的速度設為零並重新初始化分佈函數"""
        for i, j in self.rho:
            if self.mask[i, j] == 1:
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.prev_u[i, j] = ti.Vector([0.0, 0.0])

                rho_val = self.rho[i, j]
                for k in ti.static(range(9)):
                    self.f[i, j][k] = self.w[k] * rho_val
                    self.f_new[i, j][k] = self.w[k] * rho_val

    @ti.kernel
    def _collide_and_stream(self, f_src: ti.template(), f_dst: ti.template()):
        """
        MRT 碰撞與流傳核心

        Why MRT?
        - 分離不同物理量的鬆弛過程
        - 更好的數值穩定性
        - 支援 Smagorinsky LES 湍流模型

        流程：
        1. 計算巨觀量 (rho, u)
        2. 變換至矩量空間 m = M @ f
        3. 計算平衡矩量 meq(rho, u)
        4. MRT 鬆弛（應力項使用動態 tau_eff for LES）
        5. 變換回速度空間 f_post = M_inv @ m_star
        6. 流傳至鄰近格點（處理 Bounce-Back）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                continue  # 跳過固體節點

            # === 1. 計算巨觀量 ===
            f_vec = f_src[i, j]
            current_rho = 0.0
            current_u = ti.Vector([0.0, 0.0])

            for k in ti.static(range(9)):
                current_rho += f_vec[k]
                current_u += f_vec[k] * self.e[k]

            current_u /= current_rho

            # === 2. 變換至矩量空間 ===
            m = self.M[None] @ f_vec

            # === 3. 計算平衡矩量 ===
            ux, uy = current_u[0], current_u[1]
            u_sq = ux*ux + uy*uy

            meq = ti.Vector([0.0]*9)
            meq[0] = current_rho
            meq[1] = -2.0*current_rho + 3.0*current_rho*u_sq
            meq[2] = current_rho - 3.0*current_rho*u_sq
            meq[3] = current_rho*ux
            meq[4] = -current_rho*ux
            meq[5] = current_rho*uy
            meq[6] = -current_rho*uy
            meq[7] = current_rho*(ux*ux - uy*uy)
            meq[8] = current_rho*ux*uy

            # === 4. Smagorinsky LES 湍流模型 ===
            s_nu = self.S[7]
            if self.cs > 0.0:
                # 計算應力張量的非平衡部分
                noneq_pxx = m[7] - meq[7]
                noneq_pxy = m[8] - meq[8]
                Q = ti.sqrt(noneq_pxx*noneq_pxx + noneq_pxy*noneq_pxy)

                # Smagorinsky 渦黏度修正
                delta = 18.0 * (self.cs*self.cs) * Q / (current_rho + 1e-9)
                tau_eff = 0.5 * (self.tau + ti.sqrt(self.tau*self.tau + delta))
                s_nu = 1.0 / tau_eff

            # === 5. MRT 鬆弛 ===
            m_star = ti.Vector([0.0]*9)
            for k in ti.static(range(9)):
                rate = s_nu if (k == 7 or k == 8) else self.S[k]
                m_star[k] = m[k] - rate * (m[k] - meq[k])

            # === 6. 變換回速度空間 ===
            f_post = self.M_inv[None] @ m_star

            # === 7. 流傳（包含 Bounce-Back）===
            for k in ti.static(range(9)):
                dest_i = i + self.e[k][0]
                dest_j = j + self.e[k][1]

                in_bounds = (dest_i >= 0 and dest_i < self.nx and
                           dest_j >= 0 and dest_j < self.ny)

                if in_bounds:
                    if self.mask[dest_i, dest_j] == 1:
                        # Bounce-Back: 反向流傳回當前節點
                        f_dst[i, j][self.inv[k]] = f_post[k]
                    else:
                        # 正常流傳
                        f_dst[dest_i, dest_j][k] = f_post[k]

    @ti.kernel
    def _update_macro(self, f_src: ti.template()):
        """從分佈函數更新巨觀量 (rho, u)"""
        for i, j in ti.ndrange(self.nx, self.ny):
            f_vec = f_src[i, j]

            current_rho = 0.0
            current_u = ti.Vector([0.0, 0.0])

            for k in ti.static(range(9)):
                current_rho += f_vec[k]
                current_u += f_vec[k] * self.e[k]

            if current_rho > 0:
                current_u /= current_rho

            self.rho[i, j] = current_rho
            self.u[i, j] = current_u

    @ti.kernel
    def _update_diagnostics(self):
        """計算物理診斷量（質量、動量殘差）"""
        self.total_mass[None] = 0.0
        self.mass_residual[None] = 0.0
        self.mom_res_x[None] = 0.0
        self.mom_res_y[None] = 0.0
        self.mom_scale_x[None] = 0.0
        self.mom_scale_y[None] = 0.0
        self.max_u[None] = 0.0

        for i, j in self.rho:
            if self.mask[i, j] == 0:  # 只統計流體區域
                self.total_mass[None] += self.rho[i, j]
                self.mass_residual[None] += ti.abs(self.rho[i, j] - self.prev_rho[i, j])
                self.prev_rho[i, j] = self.rho[i, j]

                u_val = self.u[i, j]
                u_prev = self.prev_u[i, j]

                self.mom_res_x[None] += ti.abs(u_val[0] - u_prev[0])
                self.mom_res_y[None] += ti.abs(u_val[1] - u_prev[1])
                self.mom_scale_x[None] += ti.abs(u_val[0])
                self.mom_scale_y[None] += ti.abs(u_val[1])

                ti.atomic_max(self.max_u[None], u_val.norm())
                self.prev_u[i, j] = u_val

    def add_boundary_condition(self, bc_func: Callable, name: str = "custom_bc"):
        """
        添加邊界條件函數

        Args:
            bc_func: Taichi kernel，接受 (f_dst: ti.template()) 參數
            name: 邊界條件名稱（用於診斷）
        """
        self.bc_functions.append((bc_func, name))

    def step(self, f_src: ti.template(), f_dst: ti.template()):
        """
        執行單步時間推進

        Args:
            f_src: 來源分佈函數場
            f_dst: 目標分佈函數場
        """
        self._collide_and_stream(f_src, f_dst)

        # 應用所有註冊的邊界條件
        for bc_func, _ in self.bc_functions:
            bc_func(f_dst)

    def get_fields(self):
        """獲取當前場的 numpy 陣列"""
        return {
            'rho': self.rho.to_numpy(),
            'u': self.u.to_numpy(),
            'mask': self.mask.to_numpy(),
        }

    def get_diagnostics(self):
        """獲取診斷量"""
        return {
            'total_mass': self.total_mass[None],
            'initial_mass': self.initial_mass[None],
            'mass_residual': self.mass_residual[None],
            'mom_res_x': self.mom_res_x[None],
            'mom_res_y': self.mom_res_y[None],
            'mom_scale_x': self.mom_scale_x[None],
            'mom_scale_y': self.mom_scale_y[None],
            'max_u': self.max_u[None],
        }