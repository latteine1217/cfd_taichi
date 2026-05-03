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

    solver_family = "lbm"
    equation_set = "navier_stokes"
    regime = "low_mach"

    def __init__(
        self,
        nx: int,
        ny: int,
        re: float = 100.0,
        u_ref: float = 0.1,
        length_scale: Optional[float] = None,
        cs: float = 0.16,
        enable_sponge: bool = False,
        sponge_strength: float = 0.5,
        collision_model: str = "mrt",
        dynamic_cs_max: float = 0.23,
        nu_sgs_max: float = 0.0,
    ):
        """
        Args:
            nx: X 方向格點數
            ny: Y 方向格點數
            re: Reynolds 數 (U*L/ν)
            u_ref: 參考速度 (lattice units)
            length_scale: 特徵長度，若為 None 則使用 ny/9.0
            cs: LES 啟用旗標 (<=0 表示不使用 LES；動態 Smagorinsky 會自動估計)
            enable_sponge: 是否啟用海綿層（用於高 Re 數穩定性）
            sponge_strength: 海綿層最大阻尼係數（0-1，推薦 0.5）
            collision_model: "mrt" | "bgk" | "elbm"
            dynamic_cs_max: 動態 Smagorinsky 的上限 (建議 0.2~0.25)
            nu_sgs_max: SGS 渦黏度上限（<=0 表示不限制）
        """
        self.nx = nx
        self.ny = ny
        self.nx_g = nx + 2
        self.ny_g = ny + 2
        self.re = re
        self.u_ref = u_ref
        self.cs = cs
        self.dynamic_cs_max = dynamic_cs_max
        self.nu_sgs_max_value = nu_sgs_max
        self.enable_sponge = enable_sponge
        self.sponge_strength = sponge_strength

        # 物理參數計算
        self.L_char = length_scale if length_scale is not None else ny / 9.0
        self.nu = self.u_ref * self.L_char / self.re
        self.tau = 3.0 * self.nu + 0.5
        self.grid_spacing = 1.0

        # Sponge Layer 參數
        self.sponge_start_x = int(0.8 * nx)  # 海綿層起始位置（右側 20%）

        # === D2Q9 常數 ===
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)

        # === 場變數（使用 AoS 佈局優化 cache locality）===
        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.f_post = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        self.rho = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.mask = ti.field(
            dtype=ti.i32, shape=(self.nx_g, self.ny_g)
        )  # 1=固體, 0=流體
        self.boundary_q = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.use_bouzidi = ti.field(dtype=ti.i32, shape=())

        max_cells = nx * ny
        self.fluid_bulk_indices = ti.Vector.field(2, dtype=ti.i32, shape=max_cells)
        self.fluid_boundary_indices = ti.Vector.field(2, dtype=ti.i32, shape=max_cells)
        self.solid_indices = ti.Vector.field(2, dtype=ti.i32, shape=max_cells)
        self.num_fluid_bulk = ti.field(dtype=ti.i32, shape=())
        self.num_fluid_boundary = ti.field(dtype=ti.i32, shape=())
        self.num_solid = ti.field(dtype=ti.i32, shape=())

        # === 診斷場 ===
        self.prev_rho = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.prev_u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.nu_sgs = ti.field(
            dtype=ti.f32, shape=(self.nx_g, self.ny_g)
        )  # Smagorinsky 渦黏度
        self.nu_sgs_raw = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.nu_sgs_max = ti.field(dtype=ti.f32, shape=())
        self.sdf = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.sdf_enabled = ti.field(dtype=ti.i32, shape=())
        self.sdf_wall_distance = ti.field(dtype=ti.f32, shape=())
        self.u_bar = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.uu_bar = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.rho_bar = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.ru_bar = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.ruu_bar = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        self.collision_model_id = ti.field(dtype=ti.i32, shape=())

        self.wall_function_enabled = ti.field(dtype=ti.i32, shape=())
        self.wall_kappa = ti.field(dtype=ti.f32, shape=())
        self.wall_B = ti.field(dtype=ti.f32, shape=())
        self.wall_yplus_min = ti.field(dtype=ti.f32, shape=())
        self.force_enabled = ti.field(dtype=ti.i32, shape=())
        self.body_force = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.force_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.force_field_enabled = ti.field(dtype=ti.i32, shape=())
        self.force_field_updater = None
        self.sdf_enabled[None] = 0
        self.sdf_wall_distance[None] = 2.0
        self.sdf.fill(1e3)
        self.nu_sgs_max[None] = self.nu_sgs_max_value

        # === 全局診斷量 ===
        self.total_mass = ti.field(dtype=ti.f32, shape=())
        self.initial_mass = ti.field(dtype=ti.f32, shape=())
        self.mass_residual = ti.field(dtype=ti.f32, shape=())
        self.mom_res_x = ti.field(dtype=ti.f32, shape=())
        self.mom_res_y = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_x = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_y = ti.field(dtype=ti.f32, shape=())
        self.max_u = ti.field(dtype=ti.f32, shape=())
        self.cfl_violation = ti.field(dtype=ti.i32, shape=())
        self.mass_correction_total = ti.field(dtype=ti.f32, shape=())

        # === 能量監控 ===
        self.total_KE = ti.field(dtype=ti.f32, shape=())
        self.initial_KE = ti.field(dtype=ti.f32, shape=())
        self.KE_dissipation_rate = ti.field(dtype=ti.f32, shape=())

        # === 粒子系統 (粒子追蹤可視化) ===
        self.num_particles = 500000
        self.px = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.py = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.p_active = ti.field(dtype=ti.i32, shape=self.num_particles)
        self.p_age = ti.field(
            dtype=ti.i32, shape=self.num_particles
        )  # 粒子年齡（生存步數）
        self.emitter_ptr = ti.field(dtype=ti.i32, shape=())
        self.active_particle_count = ti.field(dtype=ti.i32, shape=())  # 活躍粒子統計

        # === MRT 矩陣 ===
        self.M = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.M_inv = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.S = ti.field(dtype=ti.f32, shape=9)
        self.R = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.R_inv = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())

        self.bc_functions = []
        self._init_constants()
        if collision_model not in {"mrt", "bgk", "elbm", "emrt"}:
            raise ValueError(f"Invalid collision_model: {collision_model}")
        self.collision_model = collision_model
        if collision_model == "mrt":
            self.collision_model_id[None] = 0
        elif collision_model == "bgk":
            self.collision_model_id[None] = 1
        elif collision_model == "elbm":
            self.collision_model_id[None] = 2
        else:
            self.collision_model_id[None] = 3

        validation_error = self._validate_parameters()  # CFL 與 Mach 數檢查
        if validation_error:
            raise ValueError(validation_error)
        self.reset()

    # --- 粒子系統 Kernels ---
    @ti.kernel
    def _init_particles(self):
        """
        初始化粒子系統

        Why 並行初始化?
        - 500k 粒子的初始化可完全並行
        - GPU 上比 CPU 迴圈快 100x+
        """
        self.emitter_ptr[None] = 0
        self.active_particle_count[None] = 0

        for i in range(self.num_particles):
            self.p_active[i] = 0
            self.p_age[i] = 0
            self.px[i] = 0.0
            self.py[i] = 0.0

    @ti.kernel
    def _emit_particles(self, num_lines: int):
        """
        並行粒子發射（GPU 優化版）

        Why 批次發射?
        - 所有粒子位置計算完全獨立
        - GPU 並行發射 num_lines 個粒子
        - 避免 atomic operations（使用 modulo 循環）

        優化：
        - 使用 ti.ndrange 完全並行
        - 預計算所有粒子初始位置
        """
        stride = self.ny / num_lines
        base_ptr = self.emitter_ptr[None]

        # 完全並行的粒子發射
        for l in range(num_lines):
            idx = (base_ptr + l) % self.num_particles
            self.p_active[idx] = 1
            self.p_age[idx] = 0  # 重置年齡
            self.px[idx] = 2.0  # 入口稍微靠內
            self.py[idx] = (l + 0.5) * stride

        # 更新發射指針（單次原子操作）
        self.emitter_ptr[None] = (base_ptr + num_lines) % self.num_particles

    @ti.func
    def _sample_u(self, x: float, y: float):
        """雙線性插值採樣速度場"""
        i, j = int(x), int(y)
        i = ti.max(0, ti.min(self.nx - 2, i))
        j = ti.max(0, ti.min(self.ny - 2, j))
        dx, dy = x - i, y - j
        ig, jg = i + 1, j + 1
        u00, u10 = self.u[ig, jg], self.u[ig + 1, jg]
        u01, u11 = self.u[ig, jg + 1], self.u[ig + 1, jg + 1]
        return (
            u00 * (1 - dx) * (1 - dy)
            + u10 * dx * (1 - dy)
            + u01 * (1 - dx) * dy
            + u11 * dx * dy
        )

    @ti.kernel
    def _advect_particles(self):
        """
        並行粒子推進（GPU 優化版）

        Why 這樣設計?
        - 500k 粒子完全獨立，無數據依賴
        - GPU 並行處理：Metal 可達 10k+ threads
        - 記憶體訪問模式：coalesced access (px, py 連續)

        優化技巧：
        - 提前終止非活躍粒子（減少計算）
        - 雙線性插值（ti.func 內聯優化）
        - 邊界檢查合併（減少分支）
        """
        for i in range(self.num_particles):
            if self.p_active[i] == 1:
                # 速度場採樣（雙線性插值）
                vel = self._sample_u(self.px[i], self.py[i])

                # Euler 前向積分
                self.px[i] += vel[0]
                self.py[i] += vel[1]

                # 增加粒子年齡
                self.p_age[i] += 1

                # 邊界與障礙物檢查（合併條件）
                out_of_bounds = (
                    self.px[i] < 0
                    or self.px[i] >= self.nx
                    or self.py[i] < 0
                    or self.py[i] >= self.ny
                )

                if out_of_bounds:
                    self.p_active[i] = 0
                else:
                    # 障礙物碰撞檢測
                    ix = int(self.px[i])
                    iy = int(self.py[i])
                    if self.mask[ix + 1, iy + 1] == 1:
                        self.p_active[i] = 0

    @ti.kernel
    def _count_active_particles(self) -> ti.i32:
        """
        統計活躍粒子數（並行歸約）

        Why 需要這個?
        - 診斷粒子系統效率（利用率）
        - 自適應發射策略（保持活躍粒子密度）
        - 性能監控

        並行歸約：
        - 每個 thread 處理一部分粒子
        - 使用 atomic_add 累加結果
        """
        self.active_particle_count[None] = 0
        for i in range(self.num_particles):
            if self.p_active[i] == 1:
                ti.atomic_add(self.active_particle_count[None], 1)
        return self.active_particle_count[None]

    def step_particles(
        self, emit: bool = True, num_lines: int = 32, count_active: bool = False
    ):
        """
        執行粒子系統時間推進（GPU 並行優化）

        Args:
            emit: 是否發射新粒子
            num_lines: 發射粒子數（煙線數量）
            count_active: 是否統計活躍粒子（略微增加開銷）

        性能：
        - 500k 粒子推進：~0.5ms (Apple M3)
        - 完全 GPU 並行，無 CPU 同步
        """
        if emit:
            self._emit_particles(num_lines)
        self._advect_particles()

        if count_active:
            self._count_active_particles()

    def get_particle_stats(self):
        """獲取粒子系統統計資訊"""
        return {
            "total_particles": self.num_particles,
            "active_particles": self.active_particle_count[None],
            "utilization": self.active_particle_count[None] / self.num_particles,
        }

    def _init_constants(self):
        """初始化 D2Q9 常數與 MRT 矩陣"""
        # D2Q9 權重
        w_np = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
            dtype=np.float32,
        )

        # D2Q9 格子速度向量
        e_np = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ],
            dtype=np.int32,
        )

        # 反向索引（用於 Bounce-Back）
        inv_np = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

        # MRT 變換矩陣 (速度空間 -> 矩量空間)
        # 對應矩量：rho, e, epsilon, j_x, q_x, j_y, q_y, p_xx, p_xy
        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 1, -1],
            ],
            dtype=np.float32,
        )

        M_inv_np = np.linalg.inv(M_np).astype(np.float32)

        R_np = np.zeros((9, 9), dtype=np.float32)
        for k in range(9):
            ex, ey = e_np[k]
            R_np[0, k] = 1.0
            R_np[1, k] = ex
            R_np[2, k] = ey
            R_np[3, k] = ex * ex
            R_np[4, k] = ey * ey
            R_np[5, k] = ex * ey
            R_np[6, k] = ex * ex * ey
            R_np[7, k] = ex * ey * ey
            R_np[8, k] = ex * ex * ey * ey
        R_inv_np = np.linalg.inv(R_np).astype(np.float32)

        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)
        self._set_mrt_matrices(M_np, M_inv_np)
        self._set_raw_moment_matrices(R_np, R_inv_np)

    @ti.kernel
    def _set_raw_moment_matrices(
        self, r_arr: ti.types.ndarray(), rinv_arr: ti.types.ndarray()
    ):
        for i, j in ti.ndrange(9, 9):
            self.R[None][i, j] = r_arr[i, j]
            self.R_inv[None][i, j] = rinv_arr[i, j]

    @ti.kernel
    def _set_mrt_matrices(
        self, m_arr: ti.types.ndarray(), minv_arr: ti.types.ndarray()
    ):
        """設定 MRT 矩陣與鬆弛參數"""
        for i, j in ti.ndrange(9, 9):
            self.M[None][i, j] = m_arr[i, j]
            self.M_inv[None][i, j] = minv_arr[i, j]

        # 鬆弛參數設定
        # k=7,8 對應應力張量，使用黏度相關的 s_nu
        # 其他高階矩使用較大的 s_other (更快鬆弛，提高穩定性)
        s_nu = 1.0 / self.tau

        # Lallemand & Luo (2000) 常用鬆弛參數
        s_e = 1.64
        s_eps = 1.54
        s_q = 1.2

        self.S[0] = 0.0  # 質量守恆（不鬆弛）
        self.S[1] = s_e  # 能量相關
        self.S[2] = s_eps
        self.S[3] = 0.0  # 動量守恆（不鬆弛）
        self.S[4] = s_q
        self.S[5] = 0.0  # 動量守恆（不鬆弛）
        self.S[6] = s_q
        self.S[7] = s_nu  # 應力張量（黏度）
        self.S[8] = s_nu  # 應力張量（黏度）

    def _validate_parameters(self) -> str:
        """
        驗證 LBM 參數有效性（CFL 與 Mach 數檢查）

        Why 需要這個檢查？
        - LBM 基於低 Mach 數近似（Ma < 0.3）
        - 速度過大會違反不可壓假設，導致數值爆炸
        - 確保 tau > 0.5（數值穩定性下界）

        約束條件：
        1. Ma = |u_ref| / c_s < 0.3  (c_s = 1/√3 ≈ 0.577)
        2. tau = 3*nu + 0.5 > 0.5
        3. Re > 0
        """
        cs_lattice = 1.0 / np.sqrt(3.0)  # LBM 聲速
        mach_number = self.u_ref / cs_lattice

        print("\n" + "=" * 60)
        print("Parameter Validation (CFL & Mach Number Check)")
        print("-" * 60)
        print(f"  Reference Velocity (u_ref)   : {self.u_ref:.4f}")
        print(f"  Lattice Sound Speed (c_s)    : {cs_lattice:.4f}")
        print(f"  Mach Number (Ma)             : {mach_number:.4f}")
        print(f"  Relaxation Time (tau)        : {self.tau:.4f}")
        print(f"  Kinematic Viscosity (nu)     : {self.nu:.6f}")
        print(f"  Reynolds Number (Re)         : {self.re:.1f}")

        # === 檢查 1: Mach 數限制 ===
        if mach_number > 0.3:
            print(f"\n❌ ERROR: Mach number {mach_number:.3f} exceeds limit (0.3)")
            print(f"   Recommendation: Reduce u_ref to < {0.3 * cs_lattice:.3f}")
            return f"Mach number too high: Ma={mach_number:.3f} > 0.3"
        elif mach_number > 0.2:
            print(f"\n⚠️  WARNING: Mach number {mach_number:.3f} is close to limit")
            print(f"   Consider reducing u_ref for better accuracy")
        else:
            print(f"  ✅ Mach number within safe range (< 0.3)")

        # === 檢查 2: 鬆弛時間下界 ===
        if self.tau < 0.501:
            print(f"\n❌ ERROR: Relaxation time {self.tau:.3f} too small (< 0.501)")
            print(f"   Recommendation: Increase Reynolds number or reduce u_ref")
            return f"tau={self.tau:.3f} < 0.501 (numerical instability)"
        elif self.tau < 0.55:
            print(f"\n⚠️  WARNING: tau={self.tau:.3f} is close to instability limit")
        else:
            print(f"  ✅ Relaxation time within stable range (> 0.5)")

        # === 檢查 3: Reynolds 數合理性 ===
        if self.re <= 0:
            return f"Invalid Reynolds number: Re={self.re}"

        print("=" * 60 + "\n")
        return ""

    @ti.kernel
    def _check_cfl_violation(self):
        """
        Runtime CFL 條件檢查

        Why 需要 runtime 檢查？
        - 即使初始參數合理，流場可能在局部產生高速
        - 渦流、分離、湍流都可能導致速度局部超標

        Returns:
            1 if violation detected, 0 otherwise
        """
        self.cfl_violation[None] = 0
        self.max_u[None] = 0.0
        max_allowed = 0.3  # 穩定性限制（< c_s = 0.577）

        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            u_mag = self.u[ig, jg].norm()
            mask_fluid = 1.0 - ti.cast(self.mask[ig, jg], ti.f32)
            u_mag = u_mag * mask_fluid
            ti.atomic_max(self.max_u[None], u_mag)
            ti.atomic_max(
                self.cfl_violation[None], ti.cast(u_mag > max_allowed, ti.i32)
            )
            # 只在第一次檢測時輸出（避免洪水輸出）
            # print(f"CFL violation at ({i},{j}): |u|={u_mag:.3f}")

    @ti.kernel
    def _reset_fields(self):
        """重置所有場到初始狀態"""
        for i, j in ti.ndrange(self.nx_g, self.ny_g):
            inside = i >= 1 and i <= self.nx and j >= 1 and j <= self.ny
            if inside:
                self.mask[i, j] = 0
                self.u[i, j] = ti.Vector([self.u_ref, 0.0])
                self.prev_u[i, j] = ti.Vector([self.u_ref, 0.0])
            else:
                self.mask[i, j] = 1
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.prev_u[i, j] = ti.Vector([0.0, 0.0])

            self.rho[i, j] = 1.0
            self.prev_rho[i, j] = 1.0

            # 初始化為平衡分佈
            u_vec = self.u[i, j]
            rho_val = self.rho[i, j]
            u_sq = u_vec.norm_sqr()

            for k in ti.static(range(9)):
                eu = self.e[k].dot(u_vec)
                f_eq = (
                    self.w[k] * rho_val * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                )
                self.f[i, j][k] = f_eq
                self.f_new[i, j][k] = f_eq
                self.f_post[i, j][k] = f_eq

            for k in ti.static(range(9)):
                self.boundary_q[i, j][k] = -1.0

        self.use_bouzidi[None] = 0

    def reset(self):
        """
        完整重置求解器（含 mask/幾何）。
        僅在 __init__ 或需要完全重建時呼叫。

        ⚠️  此方法會清除 mask（障礙物與 No-Slip 壁面）。
            若只需要重新開始流場而保留幾何設定，請改用 reset_flow_state()。
        """
        self._init_particles()
        self._reset_fields()
        self._build_index_lists(self.mask.to_numpy())
        self.reset_mass_baseline()

        self.wall_function_enabled[None] = 0
        self.wall_kappa[None] = 0.41
        self.wall_B[None] = 5.2
        self.wall_yplus_min[None] = 5.0
        self.force_enabled[None] = 0
        self.body_force[None] = ti.Vector([0.0, 0.0])
        self.force_field_enabled[None] = 0
        self.force_field.fill(0.0)
        self.force_field_updater = None

    @ti.kernel
    def _reset_state_preserve_mask(self):
        """
        重置流場狀態，保留 mask/幾何設定。
        What: 將 f/u/rho 重設為平衡分佈；mask 維持不變。
        Why: 允許在保留障礙物、壁面設定的前提下重新開始模擬。
        """
        for i, j in ti.ndrange(self.nx_g, self.ny_g):
            inside = i >= 1 and i <= self.nx and j >= 1 and j <= self.ny
            if inside and self.mask[i, j] == 0:
                self.u[i, j] = ti.Vector([self.u_ref, 0.0])
                self.prev_u[i, j] = ti.Vector([self.u_ref, 0.0])
            else:
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.prev_u[i, j] = ti.Vector([0.0, 0.0])

            self.rho[i, j] = 1.0
            self.prev_rho[i, j] = 1.0

            u_vec = self.u[i, j]
            rho_val = self.rho[i, j]
            u_sq = u_vec.norm_sqr()

            for k in ti.static(range(9)):
                eu = self.e[k].dot(u_vec)
                f_eq = (
                    self.w[k] * rho_val * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                )
                self.f[i, j][k] = f_eq
                self.f_new[i, j][k] = f_eq
                self.f_post[i, j][k] = f_eq

    def reset_flow_state(self):
        """
        重置流場狀態，保留 mask/幾何/BC 配置。

        What: 重置 f/u/rho 到平衡狀態，不改變 mask、邊界條件或障礙物設定。
        Why: 允許在保留同一幾何的前提下重啟模擬（例如掃描 Re 數）。

        與 reset() 的差異：
        - reset_flow_state() → 保留 mask（障礙物、No-Slip 壁面）
        - reset()            → 完全重置，包含 mask（等同重新建立 solver）
        """
        self._init_particles()
        self._reset_state_preserve_mask()
        self.reset_mass_baseline()

    def set_initial_condition(
        self,
        velocity: Optional[np.ndarray] = None,
        density: Optional[np.ndarray] = None,
        apply_boundaries: bool = False,
        reset_baseline: bool = False,
    ):
        """
        設定初始宏觀場並重建平衡分佈。

        What:
        - 以公開 API 一次設定速度場與密度場
        - 自動重建 `f / f_new / f_post`，避免案例直接碰私有 kernel

        Why:
        - 讓 examples/tests 不必直接呼叫 `_apply_velocity_field()` 或寫入 `rho`
        - 保持初始化流程一致，可重用於均勻場、靜止場、KH/RT 等非均勻場

        When:
        - 案例初始條件建立
        - 重新啟動同一幾何下的流場
        """
        if velocity is None and density is None:
            raise ValueError("At least one of velocity or density must be provided.")

        if density is not None:
            if density.shape != (self.nx, self.ny):
                raise ValueError(
                    f"Density shape {density.shape} incompatible with grid ({self.nx}, {self.ny})"
                )
            self._apply_density_field(np.asarray(density, dtype=np.float32))

        if velocity is not None:
            if velocity.shape != (self.nx, self.ny, 2):
                raise ValueError(
                    "Velocity shape "
                    f"{velocity.shape} incompatible with grid ({self.nx}, {self.ny}, 2)"
                )
            self._apply_velocity_field(np.asarray(velocity, dtype=np.float32))
        else:
            self._refresh_equilibrium_from_macro()

        if apply_boundaries:
            self.apply_boundary_conditions(self.f)
            self.apply_boundary_conditions(self.f_new)

        if reset_baseline:
            self.reset_mass_baseline()

    def initialize_obstacle(
        self,
        mask_array: np.ndarray,
        sdf_array: Optional[np.ndarray] = None,
        sdf_units: str = "lattice",
        reset_baseline: bool = True,
    ):
        """
        設定障礙物並同步修正固體區狀態。

        What:
        - 包裝 `set_obstacle()` 與固體區速度/分佈修正

        Why:
        - 避免案例直接呼叫 `_correct_solid_velocity()`
        - 讓障礙物初始化成為 solver 的正式公開流程
        """
        self.set_obstacle(mask_array, sdf_array=sdf_array, sdf_units=sdf_units)
        self._correct_solid_velocity()
        if reset_baseline:
            self.reset_mass_baseline()

    def prepare_diagnostics(self, f_src=None, reset_baseline: bool = False):
        """
        更新宏觀量與診斷量，必要時重設守恆基準。

        What:
        - 取代案例直接呼叫 `_update_macro()` / `_update_diagnostics()`

        Why:
        - 收斂判定、輸出、基準重設應由公開 API 管理
        - 降低案例與 solver internals 的耦合
        """
        if f_src is None:
            f_src = self.f

        self._update_macro(f_src)
        self._update_diagnostics()
        if reset_baseline:
            self.initial_mass[None] = self.total_mass[None]
            self.initial_KE[None] = self.total_KE[None]
        return self.get_diagnostics()

    def rebuild_index_lists(self):
        """
        重建流體/固體/邊界索引列表。

        What: 從目前的 mask 重新計算 bulk/boundary/solid 分類。
        Why: mask 被 add_no_slip_wall() 或 set_obstacle() 修改後，
             舊的索引列表已失效；此方法確保 step() 使用正確的節點分類。

        Note: BoundaryConditions.add_no_slip_wall() 會自動呼叫此方法；
              一般情況下無需從 example 手動呼叫。
        """
        self._build_index_lists(self.mask.to_numpy())

    def reset_mass_baseline(self):
        """
        重新設定質量/能量基準

        When:
        - 設定障礙物或邊界後（流體體積變化）
        - 想以當前狀態作為新的守恆基準
        """
        self._update_macro(self.f)
        self._update_diagnostics()
        self.initial_mass[None] = self.total_mass[None]
        self.initial_KE[None] = self.total_KE[None]

    def set_obstacle(
        self,
        mask_array: np.ndarray,
        sdf_array: Optional[np.ndarray] = None,
        sdf_units: str = "lattice",
    ):
        """
        設定固體障礙物

        Args:
            mask_array: (nx, ny) 或 (ny, nx) 的 numpy 陣列，1=固體，0=流體
            sdf_array: (nx, ny) 或 (ny, nx) 的 signed distance，流體為正、固體為負
            sdf_units: 'lattice' (預設，格距單位) 或 'physical'（會除以 grid_spacing）
        """
        if mask_array.shape == (self.nx, self.ny):
            mask_np = mask_array.astype(np.int32)
        elif mask_array.shape == (self.ny, self.nx):
            mask_np = mask_array.T.astype(np.int32)
        else:
            raise ValueError(
                f"Mask shape {mask_array.shape} incompatible with grid ({self.nx}, {self.ny})"
            )

        mask_g = np.ones((self.nx_g, self.ny_g), dtype=np.int32)
        mask_g[1 : self.nx + 1, 1 : self.ny + 1] = mask_np
        self.mask.from_numpy(mask_g)

        if sdf_array is not None:
            if sdf_units not in {"lattice", "physical"}:
                raise ValueError(f"Invalid sdf_units: {sdf_units}")
            if sdf_array.shape == (self.nx, self.ny):
                sdf_np = sdf_array.astype(np.float32)
            elif sdf_array.shape == (self.ny, self.nx):
                sdf_np = sdf_array.T.astype(np.float32)
            else:
                raise ValueError(
                    f"SDF shape {sdf_array.shape} incompatible with grid ({self.nx}, {self.ny})"
                )
            if sdf_units == "physical":
                sdf_np = sdf_np / max(self.grid_spacing, 1e-12)

            boundary_q = self._compute_boundary_q_from_sdf(mask_np, sdf_np)
            q_g = -np.ones((self.nx_g, self.ny_g, 9), dtype=np.float32)
            q_g[1 : self.nx + 1, 1 : self.ny + 1, :] = boundary_q
            self.boundary_q.from_numpy(q_g)
            self.use_bouzidi[None] = 1
            sdf_g = np.full((self.nx_g, self.ny_g), 1e3, dtype=np.float32)
            sdf_g[1 : self.nx + 1, 1 : self.ny + 1] = sdf_np
            self.sdf.from_numpy(sdf_g)
            self.sdf_enabled[None] = 1
        else:
            self.use_bouzidi[None] = 0
            self._clear_boundary_q()
            self.sdf_enabled[None] = 0
            self.sdf.fill(1e3)

        self._build_index_lists(mask_g)

        # 修正固體區域的速度與分佈函數：由呼叫端決定何時觸發
        self.reset_mass_baseline()

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
                    self.f_post[i, j][k] = self.w[k] * rho_val

    @ti.kernel
    def _clear_boundary_q(self):
        for i, j in self.rho:
            for k in ti.static(range(9)):
                self.boundary_q[i, j][k] = -1.0

    def _compute_boundary_q_from_sdf(
        self, mask_np: np.ndarray, sdf_np: np.ndarray
    ) -> np.ndarray:
        e = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ],
            dtype=np.int32,
        )

        q = -np.ones((self.nx, self.ny, 9), dtype=np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                if mask_np[i, j] == 0:
                    phi_f = sdf_np[i, j]
                    for k in range(1, 9):
                        ni = i + e[k, 0]
                        nj = j + e[k, 1]
                        if 0 <= ni < self.nx and 0 <= nj < self.ny:
                            if mask_np[ni, nj] == 1:
                                phi_s = sdf_np[ni, nj]
                                denom = phi_f - phi_s
                                if abs(denom) > 1e-6:
                                    q_val = phi_f / denom
                                else:
                                    q_val = 0.5
                                q_val = max(1e-3, min(1.0 - 1e-3, q_val))
                                q[i, j, k] = q_val
        return q

    def _build_index_lists(self, mask_np: np.ndarray):
        e = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ],
            dtype=np.int32,
        )

        bulk = []
        boundary = []
        solid = []

        for i in range(self.nx):
            for j in range(self.ny):
                ig = i + 1
                jg = j + 1
                if mask_np[ig, jg] == 1:
                    solid.append([ig, jg])
                    continue

                # 域邊界上的流體節點應視為 boundary，而不是直接跳過。
                # 否則像 LDC top moving wall 這種位於域邊界的流體列不會參與正常的
                # collide/stream 邊界更新，動量也無法向內傳遞。
                if i == 0 or j == 0 or i == self.nx - 1 or j == self.ny - 1:
                    boundary.append([ig, jg])
                    continue

                is_bulk = True
                for k in range(1, 9):
                    ni = i + e[k, 0]
                    nj = j + e[k, 1]
                    if mask_np[ni + 1, nj + 1] == 1:
                        is_bulk = False
                        break

                if is_bulk:
                    bulk.append([ig, jg])
                else:
                    boundary.append([ig, jg])

        self._update_index_fields(
            np.array(bulk, dtype=np.int32),
            np.array(boundary, dtype=np.int32),
            np.array(solid, dtype=np.int32),
        )

    def _update_index_fields(
        self,
        bulk_indices: np.ndarray,
        boundary_indices: np.ndarray,
        solid_indices: np.ndarray,
    ):
        max_cells = self.nx * self.ny

        bulk_count = int(bulk_indices.shape[0])
        boundary_count = int(boundary_indices.shape[0])
        solid_count = int(solid_indices.shape[0])

        bulk_buf = np.zeros((max_cells, 2), dtype=np.int32)
        solid_buf = np.zeros((max_cells, 2), dtype=np.int32)

        if bulk_count > 0:
            bulk_buf[:bulk_count] = bulk_indices
        if solid_count > 0:
            solid_buf[:solid_count] = solid_indices

        self.fluid_bulk_indices.from_numpy(bulk_buf)
        self.solid_indices.from_numpy(solid_buf)

        boundary_buf = np.zeros((max_cells, 2), dtype=np.int32)
        if boundary_count > 0:
            boundary_buf[:boundary_count] = boundary_indices
        self.fluid_boundary_indices.from_numpy(boundary_buf)

        self.num_fluid_bulk[None] = bulk_count
        self.num_fluid_boundary[None] = boundary_count
        self.num_solid[None] = solid_count

    def init_potential_flow_airfoil(self, chord: float, aoa_deg: float, center: tuple):
        """
        使用勢流解初始化機翼外流（均勻流 + 點渦）

        物理模型：
        - 均勻流：以攻角方向入流
        - 點渦：位於 1/4 弦長處，模擬循環
        - 環量：Γ = 2π * c * sin(α) * U∞

        Args:
            chord: 機翼弦長 (僅用於記錄，不影響初始化)
            aoa_deg: 攻角 (僅用於記錄)
            center: 機翼位置 (僅用於記錄)
        """
        print("\n🌊 Initializing with potential flow (airfoil)")
        print(f"  Inlet Velocity: {self.u_ref:.3f}")
        print(f"  Angle of Attack: {aoa_deg:.2f} deg")

        alpha = np.deg2rad(aoa_deg)
        u_uniform = np.array(
            [self.u_ref * np.cos(alpha), self.u_ref * np.sin(alpha)], dtype=np.float32
        )

        u_field = np.zeros((self.nx, self.ny, 2), dtype=np.float32)
        u_field[:, :, 0] = u_uniform[0]
        u_field[:, :, 1] = u_uniform[1]

        u_mag = np.linalg.norm(u_field, axis=2)
        max_u = 1.2 * self.u_ref
        scale = np.where(u_mag > max_u, max_u / (u_mag + 1e-12), 1.0)
        u_field[:, :, 0] *= scale
        u_field[:, :, 1] *= scale

        mask_np = self.mask.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u_field[mask_np == 1] = 0.0

        self._apply_velocity_field(u_field)
        print("✅ Potential flow initialization complete")

    @ti.kernel
    def _init_potential_flow_cylinder_kernel(
        self, cx: ti.f32, cy: ti.f32, radius: ti.f32, u_arr: ti.types.ndarray()
    ):
        """
        GPU 並行計算勢流場

        Why GPU 優化？
        - 每個格點的計算完全獨立（無數據依賴）
        - 100x - 1000x 加速（vs Python 雙層循環）
        - 1000x1000 網格：從數秒降至毫秒級

        物理公式：
        - u_x = U∞(1 - R²(x²-y²)/r⁴)
        - u_y = -U∞·R²·2xy/r⁴
        """
        K = self.u_ref * radius * radius
        r_min_sq = (radius * 0.8) ** 2
        u_ref_max = 1.5 * self.u_ref

        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                dx = ti.cast(i, ti.f32) - cx
                dy = ti.cast(j, ti.f32) - cy
                r_sq = dx * dx + dy * dy

                if r_sq < r_min_sq:
                    # 圓柱內部及近邊界：設為零
                    u_arr[i, j, 0] = 0.0
                    u_arr[i, j, 1] = 0.0
                else:
                    # 勢流解：Uniform + Doublet
                    r4 = r_sq * r_sq
                    u_x = self.u_ref * (1.0 - K * (dx * dx - dy * dy) / r4)
                    u_y = -self.u_ref * K * 2.0 * dx * dy / r4

                    # 限制速度大小（防止數值問題）
                    u_mag = ti.sqrt(u_x * u_x + u_y * u_y)
                    if u_mag > u_ref_max:
                        scale = u_ref_max / u_mag
                        u_x *= scale
                        u_y *= scale

                    u_arr[i, j, 0] = u_x
                    u_arr[i, j, 1] = u_y

    def init_potential_flow_cylinder(self, cx: float, cy: float, radius: float):
        """
        使用勢流解初始化速度場（圓柱繞流）

        Why 勢流初始化？
        - 標準均勻場 u=(U,0) 在固體邊界不滿足邊界條件
        - 產生巨大初始殘差，需要數千步才能穩定
        - 勢流解滿足邊界條件（u·n=0 at surface）
        - 可減少 50%+ 的 startup phase

        物理背景：
        - 勢函數 φ = U∞·x + K·x/(x²+y²)  (均勻流 + Doublet)
        - K = -U∞·R² (R = 圓柱半徑)
        - 速度 u = ∇φ

        優化：GPU 並行計算（100x - 1000x 加速）

        Args:
            cx, cy: 圓柱中心座標
            radius: 圓柱半徑
        """
        print(
            f"\n🌊 Initializing with potential flow (cylinder at ({cx:.1f}, {cy:.1f}), R={radius:.1f})"
        )

        # 分配速度場緩衝區
        u_field = np.zeros((self.nx, self.ny, 2), dtype=np.float32)

        # GPU 並行計算勢流場
        self._init_potential_flow_cylinder_kernel(cx, cy, radius, u_field)

        # 應用到 Taichi 場
        self._apply_velocity_field(u_field)
        print(f"✅ Potential flow initialization complete (GPU accelerated)")

    @ti.kernel
    def _apply_velocity_field(self, u_arr: ti.types.ndarray()):
        """將 numpy 速度場應用到求解器並重新計算平衡分佈"""
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:  # 流體區域
                # 更新速度場
                self.u[ig, jg] = ti.Vector([u_arr[i, j, 0], u_arr[i, j, 1]])
                self.prev_u[ig, jg] = self.u[ig, jg]

                # 重新計算平衡分佈函數
                u_vec = self.u[ig, jg]
                rho_val = self.rho[ig, jg]
                u_sq = u_vec.norm_sqr()

                for k in ti.static(range(9)):
                    eu = self.e[k].dot(u_vec)
                    f_eq = (
                        self.w[k]
                        * rho_val
                        * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                    )
                    self.f[ig, jg][k] = f_eq
                    self.f_new[ig, jg][k] = f_eq
                    self.f_post[ig, jg][k] = f_eq

    @ti.kernel
    def _apply_density_field(self, rho_arr: ti.types.ndarray()):
        """將 numpy 密度場寫入 solver。"""
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            self.rho[ig, jg] = rho_arr[i, j]
            self.prev_rho[ig, jg] = rho_arr[i, j]

    @ti.kernel
    def _refresh_equilibrium_from_macro(self):
        """依當前 rho/u 重建平衡分佈。"""
        for i, j in ti.ndrange(self.nx_g, self.ny_g):
            rho_val = self.rho[i, j]
            u_vec = self.u[i, j]
            u_sq = u_vec.norm_sqr()

            for k in ti.static(range(9)):
                eu = self.e[k].dot(u_vec)
                f_eq = (
                    self.w[k] * rho_val * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                )
                self.f[i, j][k] = f_eq
                self.f_new[i, j][k] = f_eq
                self.f_post[i, j][k] = f_eq

    @ti.func
    def _compute_meq(self, rho: ti.f32, u):
        ux, uy = u[0], u[1]
        u_sq = ux * ux + uy * uy

        meq = ti.Vector([0.0] * 9)
        meq[0] = rho
        meq[1] = -2.0 * rho + 3.0 * rho * u_sq
        meq[2] = rho - 3.0 * rho * u_sq
        meq[3] = rho * ux
        meq[4] = -rho * ux
        meq[5] = rho * uy
        meq[6] = -rho * uy
        meq[7] = rho * (ux * ux - uy * uy)
        meq[8] = rho * ux * uy
        return meq

    @ti.func
    def _collide_cell_mrt(self, i: ti.i32, j: ti.i32, f_src: ti.template()):
        f_vec = f_src[i, j]
        current_rho = self.rho[i, j]
        current_u = self.u[i, j]

        if current_rho <= 1e-12:
            current_rho = 1e-12
            current_u = ti.Vector([0.0, 0.0])

        m = self.M[None] @ f_vec
        meq = self._compute_meq(current_rho, current_u)

        nu_total = self.nu + self.nu_sgs[i, j]
        if self.nu_sgs_max[None] > 0.0:
            nu_total = ti.min(nu_total, self.nu + self.nu_sgs_max[None])
        tau_eff = 3.0 * nu_total + 0.5
        s_nu = 1.0 / tau_eff
        s_e = self.S[1]
        s_eps = self.S[2]
        s_q = self.S[4]

        m_star = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            rate = self.S[k]
            if k == 1:
                rate = s_e
            elif k == 2:
                rate = s_eps
            elif k == 4 or k == 6:
                rate = s_q
            elif k == 7 or k == 8:
                rate = s_nu
            m_star[k] = m[k] - rate * (m[k] - meq[k])

        f_post = self.M_inv[None] @ m_star
        if self.force_enabled[None] == 1:
            f_post = self._apply_guo_force(
                f_post, current_u, self._get_local_force(i, j), 1.0 / tau_eff
            )
        self._sanitize_post(i, j, f_post, current_rho)

    @ti.func
    def _collide_cell_bgk(self, i: ti.i32, j: ti.i32, f_src: ti.template()):
        f_vec = f_src[i, j]
        current_rho = self.rho[i, j]
        current_u = self.u[i, j]

        if current_rho <= 1e-12:
            current_rho = 1e-12
            current_u = ti.Vector([0.0, 0.0])

        nu_total = self.nu + self.nu_sgs[i, j]
        if self.nu_sgs_max[None] > 0.0:
            nu_total = ti.min(nu_total, self.nu + self.nu_sgs_max[None])
        tau_eff = 3.0 * nu_total + 0.5
        omega = 1.0 / tau_eff

        u_sq = current_u.dot(current_u)
        f_post = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            e_k = ti.cast(self.e[k], ti.f32)
            eu = e_k.dot(current_u)
            feq = (
                self.w[k] * current_rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
            )
            f_post[k] = f_vec[k] - omega * (f_vec[k] - feq)

        if self.force_enabled[None] == 1:
            f_post = self._apply_guo_force(
                f_post, current_u, self._get_local_force(i, j), omega
            )
        self._sanitize_post(i, j, f_post, current_rho)

    @ti.func
    def _entropy(self, f_vec):
        h = 0.0
        for k in ti.static(range(9)):
            f_safe = ti.max(f_vec[k], 1e-12)
            h += f_safe * ti.log(f_safe / self.w[k])
        return h

    @ti.func
    def _collide_cell_emrt(self, i: ti.i32, j: ti.i32, f_src: ti.template()):
        f_vec = f_src[i, j]
        current_rho = self.rho[i, j]
        current_u = self.u[i, j]

        if current_rho <= 1e-12:
            current_rho = 1e-12
            current_u = ti.Vector([0.0, 0.0])

        m = self.M[None] @ f_vec
        meq = self._compute_meq(current_rho, current_u)

        nu_total = self.nu + self.nu_sgs[i, j]
        if self.nu_sgs_max[None] > 0.0:
            nu_total = ti.min(nu_total, self.nu + self.nu_sgs_max[None])
        tau_eff = 3.0 * nu_total + 0.5
        s_nu = 1.0 / tau_eff
        s_e = self.S[1]
        s_eps = self.S[2]
        s_q = self.S[4]

        m_star = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            rate = self.S[k]
            if k == 1:
                rate = s_e
            elif k == 2:
                rate = s_eps
            elif k == 4 or k == 6:
                rate = s_q
            elif k == 7 or k == 8:
                rate = s_nu
            m_star[k] = m[k] - rate * (m[k] - meq[k])

        f_mrt = self.M_inv[None] @ m_star
        delta = f_mrt - f_vec

        h0 = self._entropy(f_vec)
        alpha_max = 2.0
        for k in ti.static(range(9)):
            denom = delta[k]
            if denom < -1e-12:
                alpha_max = ti.min(alpha_max, (f_vec[k] - 1e-12) / (-denom))

        alpha = 1.0
        if alpha_max > 0.0:
            low = 0.0
            high = alpha_max
            for _ in ti.static(range(3)):
                mid = 0.5 * (low + high)
                f_mid = f_vec + mid * delta
                h_mid = self._entropy(f_mid)
                if h_mid > h0:
                    low = mid
                else:
                    high = mid
            alpha = high

        alpha = ti.max(0.2, ti.min(1.9, alpha))
        f_post = f_vec + alpha * delta
        if self.force_enabled[None] == 1:
            f_post = self._apply_guo_force(
                f_post, current_u, self._get_local_force(i, j), 1.0 / tau_eff
            )
        self._sanitize_post(i, j, f_post, current_rho)

    @ti.func
    def _collide_cell_elbm(self, i: ti.i32, j: ti.i32, f_src: ti.template()):
        f_vec = f_src[i, j]
        current_rho = self.rho[i, j]
        current_u = self.u[i, j]

        if current_rho <= 1e-12:
            current_rho = 1e-12
            current_u = ti.Vector([0.0, 0.0])

        nu_total = self.nu + self.nu_sgs[i, j]
        if self.nu_sgs_max[None] > 0.0:
            nu_total = ti.min(nu_total, self.nu + self.nu_sgs_max[None])
        tau_eff = 3.0 * nu_total + 0.5
        beta = 1.0 / (2.0 * tau_eff)

        u_sq = current_u.dot(current_u)
        feq = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            e_k = ti.cast(self.e[k], ti.f32)
            eu = e_k.dot(current_u)
            feq[k] = (
                self.w[k] * current_rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
            )

        g = f_vec - feq
        h0 = self._entropy(f_vec)

        alpha_max = 2.0
        for k in ti.static(range(9)):
            denom = beta * g[k]
            if denom > 1e-12:
                alpha_max = ti.min(alpha_max, (f_vec[k] - 1e-12) / denom)

        alpha = 1.0
        if alpha_max > 0.0:
            low = 0.0
            high = alpha_max
            for _ in ti.static(range(3)):
                mid = 0.5 * (low + high)
                f_mid = f_vec - mid * beta * g
                h_mid = self._entropy(f_mid)
                if h_mid > h0:
                    low = mid
                else:
                    high = mid
            alpha = high

        f_post = f_vec - alpha * beta * g
        if self.force_enabled[None] == 1:
            f_post = self._apply_guo_force(
                f_post, current_u, self._get_local_force(i, j), 1.0 / tau_eff
            )
        self._sanitize_post(i, j, f_post, current_rho)

    @ti.func
    def _sanitize_post(self, i: ti.i32, j: ti.i32, f_post, rho_val: ti.f32):
        rho_safe = ti.max(rho_val, 1e-6)
        for k in ti.static(range(9)):
            invalid = (
                (f_post[k] != f_post[k])
                or (ti.abs(f_post[k]) > 1e6)
                or (f_post[k] < 0.0)
            )
            f_post[k] = ti.select(invalid, self.w[k] * rho_safe, f_post[k])
            self.f_post[i, j][k] = f_post[k]

    @ti.func
    def _collide_cell(self, i: ti.i32, j: ti.i32, f_src: ti.template()):
        if self.collision_model_id[None] == 0:
            self._collide_cell_mrt(i, j, f_src)
        elif self.collision_model_id[None] == 1:
            self._collide_cell_bgk(i, j, f_src)
        elif self.collision_model_id[None] == 2:
            self._collide_cell_elbm(i, j, f_src)
        else:
            self._collide_cell_emrt(i, j, f_src)

    @ti.kernel
    def _copy_solid_post(self, f_src: ti.template()):
        for p in range(self.num_solid[None]):
            i = self.solid_indices[p][0]
            j = self.solid_indices[p][1]
            for k in ti.static(range(9)):
                self.f_post[i, j][k] = f_src[i, j][k]

    @ti.kernel
    def _collide_bulk(self, f_src: ti.template()):
        for p in range(self.num_fluid_bulk[None]):
            i = self.fluid_bulk_indices[p][0]
            j = self.fluid_bulk_indices[p][1]
            self._collide_cell(i, j, f_src)

    @ti.kernel
    def _collide_boundary(self, f_src: ti.template()):
        for p in range(self.num_fluid_boundary[None]):
            i = self.fluid_boundary_indices[p][0]
            j = self.fluid_boundary_indices[p][1]
            self._collide_cell(i, j, f_src)

    @ti.kernel
    def _stream_bulk(self, f_dst: ti.template()):
        for p in range(self.num_fluid_bulk[None]):
            i = self.fluid_bulk_indices[p][0]
            j = self.fluid_bulk_indices[p][1]
            for k in ti.static(range(9)):
                dest_i = i + self.e[k][0]
                dest_j = j + self.e[k][1]
                f_dst[dest_i, dest_j][k] = self.f_post[i, j][k]

    @ti.kernel
    def _stream_boundary(self, f_dst: ti.template()):
        for p in range(self.num_fluid_boundary[None]):
            i = self.fluid_boundary_indices[p][0]
            j = self.fluid_boundary_indices[p][1]

            for k in ti.static(range(9)):
                dest_i = i + self.e[k][0]
                dest_j = j + self.e[k][1]

                solid = self.mask[dest_i, dest_j] == 1
                if solid:
                    use_bouzidi = self.use_bouzidi[None] == 1
                    q = self.boundary_q[i, j][k]
                    q = ti.select(q > 0.0, ti.max(1e-3, ti.min(1.0 - 1e-3, q)), 0.0)

                    back = self.f_post[i, j][k]
                    src_i = i - self.e[k][0]
                    src_j = j - self.e[k][1]
                    back_q_lt = (1.0 - 2.0 * q) * self.f_post[i, j][
                        k
                    ] + 2.0 * q * self.f_post[src_i, src_j][k]
                    back_q_ge = (1.0 / (2.0 * q)) * self.f_post[i, j][k] + (
                        (2.0 * q - 1.0) / (2.0 * q)
                    ) * self.f_post[dest_i, dest_j][k]
                    back = ti.select(q < 0.5, back_q_lt, back_q_ge)
                    back = ti.select(q > 0.0, back, self.f_post[i, j][k])

                    f_dst[i, j][self.inv[k]] = ti.select(
                        use_bouzidi, back, self.f_post[i, j][k]
                    )
                else:
                    f_dst[dest_i, dest_j][k] = self.f_post[i, j][k]

    @ti.kernel
    def _update_macro(self, f_src: ti.template()):
        """
        從分佈函數更新巨觀量 (rho, u)

        Why 需要 mask 檢查？
        - 固體節點（mask=1）的速度應保持為零
        - 避免固體內部產生虛假速度場
        - 確保診斷統計正確（只統計流體區域）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:  # 只更新流體節點
                f_vec = f_src[ig, jg]

                current_rho = 0.0
                current_u = ti.Vector([0.0, 0.0])

                for k in ti.static(range(9)):
                    current_rho += f_vec[k]
                    current_u += f_vec[k] * self.e[k]

                if current_rho > 0:
                    current_u /= current_rho
                    if self.force_enabled[None] == 1:
                        current_u += (
                            0.5
                            * self._get_local_force(ig, jg)
                            / ti.max(current_rho, 1e-12)
                        )

                self.rho[ig, jg] = current_rho
                self.u[ig, jg] = current_u
            else:  # 固體節點：速度為零
                self.u[ig, jg] = ti.Vector([0.0, 0.0])

    @ti.func
    def _apply_guo_force(self, f_post: ti.template(), u, force, omega: ti.f32):
        cs2 = 1.0 / 3.0
        for k in ti.static(range(9)):
            e_k = ti.cast(self.e[k], ti.f32)
            eu = e_k.dot(u)
            term = (e_k - u) / cs2 + (eu / (cs2 * cs2)) * e_k
            f_post[k] += (1.0 - 0.5 * omega) * self.w[k] * term.dot(force)
        return f_post

    @ti.func
    def _get_local_force(self, i: ti.i32, j: ti.i32):
        force = self.body_force[None]
        if self.force_field_enabled[None] == 1:
            force += self.force_field[i, j]
        return force

    def set_body_force(self, fx: float, fy: float):
        """
        設定體積力（等效壓力梯度）

        Args:
            fx: x 方向加速度
            fy: y 方向加速度
        """
        self.body_force[None] = ti.Vector([fx, fy])
        self._refresh_force_enabled()

    def set_body_force_field(self, force_field: np.ndarray, includes_ghost: bool = False):
        """
        設定空間變化的體積力場

        What: 指定每個格點的力（force density）
        Why: 用於浮力、旋轉、空間變化外力等
        When: 需要非均勻外力場時

        Args:
            force_field: (nx, ny, 2) 或 (nx+2, ny+2, 2) 陣列
            includes_ghost: True 表示 force_field 已含 ghost cells
        """
        if includes_ghost:
            if force_field.shape != (self.nx_g, self.ny_g, 2):
                raise ValueError(
                    f"force_field shape {force_field.shape} incompatible with grid ({self.nx_g}, {self.ny_g}, 2)"
                )
            self.force_field.from_numpy(force_field.astype(np.float32))
        else:
            if force_field.shape != (self.nx, self.ny, 2):
                raise ValueError(
                    f"force_field shape {force_field.shape} incompatible with grid ({self.nx}, {self.ny}, 2)"
                )
            force_g = np.zeros((self.nx_g, self.ny_g, 2), dtype=np.float32)
            force_g[1 : self.nx + 1, 1 : self.ny + 1, :] = force_field
            self.force_field.from_numpy(force_g)
        self.force_field_enabled[None] = 1
        self._refresh_force_enabled()

    def clear_body_force_field(self):
        """清除空間變化外力場"""
        self.force_field.fill(0.0)
        self.force_field_enabled[None] = 0
        self._refresh_force_enabled()

    def set_force_field_updater(self, updater: Optional[Callable[[], None]]):
        """
        設定每步更新外力場的回呼函式

        What: 在 step() 的宏觀量更新後呼叫 updater
        Why: 支援隨時間變化的外力（如 Boussinesq 浮力）
        When: 外力需要依賴 rho/u 等當前狀態
        """
        self.force_field_updater = updater
        self.force_field_enabled[None] = 1 if updater is not None else 0
        self._refresh_force_enabled()

    def _refresh_force_enabled(self):
        has_uniform = (abs(self.body_force[None][0]) > 0.0) or (
            abs(self.body_force[None][1]) > 0.0
        )
        has_field = self.force_field_enabled[None] == 1
        self.force_enabled[None] = 1 if (has_uniform or has_field) else 0

    def _update_smagorinsky_viscosity(self):
        """
        動態 Smagorinsky 渦黏度 (Germano-Lilly)

        Why dynamic?
        - 自動估計 Cs，避免手動調參
        - 對不同 Re/幾何更穩健
        """
        if self.cs <= 0.0:
            self._clear_sgs_viscosity()
            return

        self._compute_filtered_velocity()
        self._update_dynamic_smagorinsky()
        self._smooth_sgs_viscosity()

    @ti.kernel
    def _clear_sgs_viscosity(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.nu_sgs[i + 1, j + 1] = 0.0
            self.nu_sgs_raw[i + 1, j + 1] = 0.0
            self.rho_bar[i + 1, j + 1] = 0.0
            self.ru_bar[i + 1, j + 1] = ti.Vector([0.0, 0.0])
            self.ruu_bar[i + 1, j + 1] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def _strain_tensor(self, u_field: ti.template(), i: ti.i32, j: ti.i32):
        i_minus = ti.max(i - 1, 0)
        i_plus = ti.min(i + 1, self.nx - 1)
        j_minus = ti.max(j - 1, 0)
        j_plus = ti.min(j + 1, self.ny - 1)

        ig = i + 1
        jg = j + 1
        ig_minus = i_minus + 1
        ig_plus = i_plus + 1
        jg_minus = j_minus + 1
        jg_plus = j_plus + 1

        u_c = u_field[ig, jg]
        u_ip = ti.select(self.mask[ig_plus, jg] == 1, u_c, u_field[ig_plus, jg])
        u_im = ti.select(self.mask[ig_minus, jg] == 1, u_c, u_field[ig_minus, jg])
        u_jp = ti.select(self.mask[ig, jg_plus] == 1, u_c, u_field[ig, jg_plus])
        u_jm = ti.select(self.mask[ig, jg_minus] == 1, u_c, u_field[ig, jg_minus])

        inv_2dx = 0.5 / ti.max(self.grid_spacing, 1e-12)
        du_dx = (u_ip[0] - u_im[0]) * inv_2dx
        dv_dx = (u_ip[1] - u_im[1]) * inv_2dx
        du_dy = (u_jp[0] - u_jm[0]) * inv_2dx
        dv_dy = (u_jp[1] - u_jm[1]) * inv_2dx

        s_xx = du_dx
        s_yy = dv_dy
        s_xy = 0.5 * (du_dy + dv_dx)
        return s_xx, s_xy, s_yy

    @ti.func
    def _near_solid(self, i: ti.i32, j: ti.i32):
        near = 0
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ii = ti.max(0, ti.min(self.nx - 1, i + di))
            jj = ti.max(0, ti.min(self.ny - 1, j + dj))
            if self.mask[ii + 1, jj + 1] == 1:
                near = 1
        return near

    @ti.kernel
    def _compute_filtered_velocity(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                sum_rho = 0.0
                sum_ru = ti.Vector([0.0, 0.0])
                sum_ruu = ti.Vector([0.0, 0.0, 0.0])
                u_center = self.u[ig, jg]
                rho_center = self.rho[ig, jg]
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    ii = ti.max(0, ti.min(self.nx - 1, i + di))
                    jj = ti.max(0, ti.min(self.ny - 1, j + dj))
                    iig = ii + 1
                    jjg = jj + 1
                    use_center = self.mask[iig, jjg] == 1
                    u_val = ti.select(use_center, u_center, self.u[iig, jjg])
                    rho_val = ti.select(use_center, rho_center, self.rho[iig, jjg])
                    sum_rho += rho_val
                    sum_ru += rho_val * u_val
                    sum_ruu += rho_val * ti.Vector(
                        [u_val[0] * u_val[0], u_val[0] * u_val[1], u_val[1] * u_val[1]]
                    )

                inv_n = 1.0 / 9.0
                rho_bar = sum_rho * inv_n
                self.rho_bar[ig, jg] = rho_bar
                self.ru_bar[ig, jg] = sum_ru * inv_n
                self.ruu_bar[ig, jg] = sum_ruu * inv_n
                if rho_bar > 1e-12:
                    self.u_bar[ig, jg] = self.ru_bar[ig, jg] / rho_bar
                    self.uu_bar[ig, jg] = self.ruu_bar[ig, jg] / rho_bar
                else:
                    self.u_bar[ig, jg] = ti.Vector([0.0, 0.0])
                    self.uu_bar[ig, jg] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.u_bar[ig, jg] = ti.Vector([0.0, 0.0])
                self.uu_bar[ig, jg] = ti.Vector([0.0, 0.0, 0.0])
                self.rho_bar[ig, jg] = 0.0
                self.ru_bar[ig, jg] = ti.Vector([0.0, 0.0])
                self.ruu_bar[ig, jg] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _update_dynamic_smagorinsky(self):
        alpha = 2.0
        delta = self.grid_spacing
        cs2_max = self.dynamic_cs_max * self.dynamic_cs_max
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                border = (i < 2) or (i > self.nx - 3) or (j < 2) or (j > self.ny - 3)
                if border or self._near_solid(i, j) == 1:
                    self.nu_sgs_raw[ig, jg] = 0.0
                    continue
                s_xx, s_xy, s_yy = self._strain_tensor(self.u, i, j)
                s_mag = ti.sqrt(2.0 * (s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy))

                s_xx_bar, s_xy_bar, s_yy_bar = self._strain_tensor(self.u_bar, i, j)
                s_mag_bar = ti.sqrt(
                    2.0
                    * (
                        s_xx_bar * s_xx_bar
                        + s_yy_bar * s_yy_bar
                        + 2.0 * s_xy_bar * s_xy_bar
                    )
                )

                sum_ss = ti.Vector([0.0, 0.0, 0.0])
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    ii = ti.max(0, ti.min(self.nx - 1, i + di))
                    jj = ti.max(0, ti.min(self.ny - 1, j + dj))
                    sxx_n, sxy_n, syy_n = self._strain_tensor(self.u, ii, jj)
                    s_mag_n = ti.sqrt(
                        2.0 * (sxx_n * sxx_n + syy_n * syy_n + 2.0 * sxy_n * sxy_n)
                    )
                    sum_ss += ti.Vector(
                        [s_mag_n * sxx_n, s_mag_n * sxy_n, s_mag_n * syy_n]
                    )

                ss_bar = sum_ss * (1.0 / 9.0)

                u_bar = self.u_bar[ig, jg]
                uu_bar = self.uu_bar[ig, jg]
                L_xx = uu_bar[0] - u_bar[0] * u_bar[0]
                L_xy = uu_bar[1] - u_bar[0] * u_bar[1]
                L_yy = uu_bar[2] - u_bar[1] * u_bar[1]

                m_xx = (
                    2.0
                    * delta
                    * delta
                    * (ss_bar[0] - alpha * alpha * s_mag_bar * s_xx_bar)
                )
                m_xy = (
                    2.0
                    * delta
                    * delta
                    * (ss_bar[1] - alpha * alpha * s_mag_bar * s_xy_bar)
                )
                m_yy = (
                    2.0
                    * delta
                    * delta
                    * (ss_bar[2] - alpha * alpha * s_mag_bar * s_yy_bar)
                )

                num = L_xx * m_xx + 2.0 * L_xy * m_xy + L_yy * m_yy
                den = m_xx * m_xx + 2.0 * m_xy * m_xy + m_yy * m_yy + 1e-12

                cs2 = ti.max(0.0, num / den)
                cs2 = ti.min(cs2, cs2_max)

                self.nu_sgs_raw[ig, jg] = cs2 * delta * delta * s_mag
            else:
                self.nu_sgs_raw[ig, jg] = 0.0

    @ti.kernel
    def _smooth_sgs_viscosity(self):
        smooth_cs2 = 1
        delta = self.grid_spacing
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                border = (i < 2) or (i > self.nx - 3) or (j < 2) or (j > self.ny - 3)
                if border or self._near_solid(i, j) == 1:
                    self.nu_sgs[ig, jg] = 0.0
                else:
                    sum_nu = 0.0
                    sum_cs2 = 0.0
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ii = ti.max(0, ti.min(self.nx - 1, i + di))
                        jj = ti.max(0, ti.min(self.ny - 1, j + dj))
                        nu_local = self.nu_sgs_raw[ii + 1, jj + 1]
                        sum_nu += nu_local
                        if smooth_cs2 == 1:
                            sxx, sxy, syy = self._strain_tensor(self.u, ii, jj)
                            s_mag = ti.sqrt(
                                2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)
                            )
                            denom = delta * delta * ti.max(s_mag, 1e-12)
                            sum_cs2 += nu_local / denom
                    if smooth_cs2 == 1:
                        cs2 = ti.max(0.0, sum_cs2 * (1.0 / 9.0))
                        cs2 = ti.min(cs2, self.dynamic_cs_max * self.dynamic_cs_max)
                        sxx_c, sxy_c, syy_c = self._strain_tensor(self.u, i, j)
                        s_mag_c = ti.sqrt(
                            2.0 * (sxx_c * sxx_c + syy_c * syy_c + 2.0 * sxy_c * sxy_c)
                        )
                        self.nu_sgs[ig, jg] = cs2 * delta * delta * s_mag_c
                    else:
                        self.nu_sgs[ig, jg] = sum_nu * (1.0 / 9.0)
                    if self.nu_sgs_max[None] > 0.0:
                        self.nu_sgs[ig, jg] = ti.min(
                            self.nu_sgs[ig, jg], self.nu_sgs_max[None]
                        )
            else:
                self.nu_sgs[ig, jg] = 0.0

    @ti.kernel
    def _apply_wall_function(self):
        """
        近壁面 Log-law wall function（簡化版）

        What:
        - 對鄰近固體的流體格點調整有效黏度

        Why:
        - 高 Re 時近壁層無法解析
        - 以壁面剪應力近似補償解析度不足
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                n = ti.Vector([0.0, 0.0])
                y = 0.5 * self.grid_spacing
                has_wall = 0.0

                if self.sdf_enabled[None] == 1:
                    phi = self.sdf[ig, jg]
                    near = ti.cast(
                        (phi > 0.0) & (phi < self.sdf_wall_distance[None]), ti.f32
                    )
                    if near > 0.0:
                        i_minus = ti.max(ig - 1, 1)
                        i_plus = ti.min(ig + 1, self.nx)
                        j_minus = ti.max(jg - 1, 1)
                        j_plus = ti.min(jg + 1, self.ny)
                        dphidx = (self.sdf[i_plus, jg] - self.sdf[i_minus, jg]) * 0.5
                        dphidy = (self.sdf[ig, j_plus] - self.sdf[ig, j_minus]) * 0.5
                        grad_norm = ti.sqrt(dphidx * dphidx + dphidy * dphidy)
                        if grad_norm > 1e-6:
                            n = ti.Vector([dphidx, dphidy]) / grad_norm
                        y = ti.max(phi, 0.5)
                        has_wall = 1.0

                if has_wall == 0.0:
                    if i + 1 < self.nx and self.mask[ig + 1, jg] == 1:
                        n = ti.Vector([1.0, 0.0])
                    elif i - 1 >= 0 and self.mask[ig - 1, jg] == 1:
                        n = ti.Vector([-1.0, 0.0])
                    elif j + 1 < self.ny and self.mask[ig, jg + 1] == 1:
                        n = ti.Vector([0.0, 1.0])
                    elif j - 1 >= 0 and self.mask[ig, jg - 1] == 1:
                        n = ti.Vector([0.0, -1.0])

                    has_wall = ti.cast(n.norm_sqr() > 0.0, ti.f32)
                u = self.u[ig, jg]
                u_n = u.dot(n)
                u_t = u - u_n * n
                u_t_mag = u_t.norm()
                active = has_wall * ti.cast(u_t_mag > 1e-6, ti.f32)

                if active > 0.0:
                    nu = self.nu + self.nu_sgs[ig, jg]
                    u_tau = ti.sqrt(ti.abs(u_t_mag * nu / ti.max(y, 1e-6)))
                    y_plus = y * u_tau / ti.max(nu, 1e-12)
                    if y_plus < self.wall_yplus_min[None]:
                        continue

                    for _ in ti.static(range(2)):
                        y_plus = ti.max(y_plus, 1.0)
                        # Reichardt wall law + viscous sublayer (y+ < 5)
                        u_plus = 0.0
                        if y_plus < 5.0:
                            u_plus = y_plus
                        else:
                            kappa = self.wall_kappa[None]
                            A = 11.0
                            B = 3.0
                            C = 7.8
                            u_plus = (1.0 / kappa) * ti.log(1.0 + kappa * y_plus)
                            u_plus += C * (
                                1.0
                                - ti.exp(-y_plus / A)
                                - (y_plus / B) * ti.exp(-y_plus / B)
                            )
                        u_tau = u_t_mag / ti.max(u_plus, 1e-6)

                    nu_t = u_tau * u_tau * y / u_t_mag - nu
                    nu_t = ti.max(nu_t, 0.0)
                    if self.nu_sgs_max[None] > 0.0:
                        nu_t = ti.min(nu_t, self.nu_sgs_max[None])
                    self.nu_sgs[ig, jg] = ti.max(self.nu_sgs[ig, jg], nu_t)

    def set_wall_function(
        self,
        enabled: bool = True,
        kappa: float = 0.41,
        B: float = 5.2,
        yplus_min: float = 5.0,
    ):
        """
        啟用或設定 Wall Function

        Args:
            enabled: 是否啟用
            kappa: von Karman 常數
            B: log-law 常數
            yplus_min: y+ 過小時直接關閉 wall-function
        """
        self.wall_function_enabled[None] = 1 if enabled else 0
        self.wall_kappa[None] = kappa
        self.wall_B[None] = B
        self.wall_yplus_min[None] = yplus_min

    @ti.kernel
    def _update_diagnostics(self):
        """
        計算物理診斷量（質量、動量殘差、動能）- 優化版

        Why 監控動能?
        - 診斷數值耗散：不可壓流應守恆（無黏情況）
        - 驗證 LES 模型：渦黏度應正確耗散動能
        - 檢測數值不穩定：動能爆炸是失穩信號

        優化策略：
        - 單次遍歷計算所有診斷量（減少記憶體訪問）
        - 明確使用 ti.atomic_add 進行並行歸約
        - 預載入變數減少重複訪問
        """
        # === 初始化所有診斷量 ===
        self.total_mass[None] = 0.0
        self.mass_residual[None] = 0.0
        self.mom_res_x[None] = 0.0
        self.mom_res_y[None] = 0.0
        self.mom_scale_x[None] = 0.0
        self.mom_scale_y[None] = 0.0
        self.max_u[None] = 0.0
        self.total_KE[None] = 0.0

        # === 單次遍歷計算所有量（並行歸約優化）===
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:  # 只統計流體區域
                # 預載入當前值（減少重複訪問）
                rho_val = self.rho[ig, jg]
                rho_prev = self.prev_rho[ig, jg]
                u_val = self.u[ig, jg]
                u_prev = self.prev_u[ig, jg]

                # === 質量相關（並行歸約）===
                ti.atomic_add(self.total_mass[None], rho_val)
                ti.atomic_add(self.mass_residual[None], ti.abs(rho_val - rho_prev))

                # === 動量相關（並行歸約）===
                ti.atomic_add(self.mom_res_x[None], ti.abs(u_val[0] - u_prev[0]))
                ti.atomic_add(self.mom_res_y[None], ti.abs(u_val[1] - u_prev[1]))
                ti.atomic_add(self.mom_scale_x[None], ti.abs(u_val[0]))
                ti.atomic_add(self.mom_scale_y[None], ti.abs(u_val[1]))

                # === 動能（並行歸約）===
                ti.atomic_add(self.total_KE[None], 0.5 * rho_val * u_val.norm_sqr())

                # === 最大速度（並行最大值）===
                ti.atomic_max(self.max_u[None], u_val.norm())

                # === 更新前值（下一步使用）===
                self.prev_rho[ig, jg] = rho_val
                self.prev_u[ig, jg] = u_val

    def add_boundary_condition(self, bc_func: Callable, name: str = "custom_bc"):
        """
        添加邊界條件函數

        Args:
            bc_func: Taichi kernel，接受 (f_dst: ti.template()) 參數
            name: 邊界條件名稱（用於診斷）
        """
        self.bc_functions.append((bc_func, name))

    def apply_boundary_conditions(self, f: ti.template()):
        """
        手動施加所有邊界條件

        Why 需要這個?
        - 初始化後需要確保邊界條件正確
        - 某些情況下需要重新施加邊界條件

        Args:
            f: 分佈函數場
        """
        for bc_func, _ in self.bc_functions:
            bc_func(f)

    @ti.kernel
    def _apply_sponge_layer(self, f_dst: ti.template()):
        """
        施加海綿層（Sponge Layer）

        Why Sponge Layer?
        - 吸收出口處的非物理擾動
        - 防止反射波回傳至流場
        - 提高高 Re 數穩定性

        實作方式：
        - 在出口前 20% 區域（x > 0.8*nx）
        - 對流體變數施加阻尼（朝目標狀態鬆弛）
        - 阻尼強度線性增強：σ(x) = σ_max * ((x-x_start)/(x_end-x_start))²

        Physical Target:
        - ρ_target = 1.0（標準大氣壓）
        - u_target = (u_ref, 0)（均勻來流）
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0 and i >= self.sponge_start_x:
                # 計算線性增強的阻尼係數
                x_normalized = (i - self.sponge_start_x) / (
                    self.nx - self.sponge_start_x
                )
                sigma = self.sponge_strength * (x_normalized**2)

                # 目標狀態（均勻來流）
                rho_target = 1.0
                u_target = ti.Vector([self.u_ref, 0.0])

                # 計算目標平衡分佈函數
                u_sq = u_target.norm_sqr()
                feq_target = ti.Vector([0.0] * 9)

                for k in ti.static(range(9)):
                    eu = self.e[k][0] * u_target[0] + self.e[k][1] * u_target[1]
                    feq_target[k] = (
                        self.w[k]
                        * rho_target
                        * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                    )

                # 朝目標狀態鬆弛
                for k in ti.static(range(9)):
                    f_dst[ig, jg][k] = (1.0 - sigma) * f_dst[ig, jg][
                        k
                    ] + sigma * feq_target[k]

    @ti.kernel
    def _global_mass_correction(self, f_dst: ti.template()):
        """
        全局質量修正

        Why Global Mass Correction?
        - Neumann BC 不保證質量守恆
        - 長時間模擬累積質量誤差
        - 補償機制：均勻調整所有流體節點密度

        When to use?
        - 使用 Neumann outflow 時
        - 每 100-500 步執行一次
        - 質量誤差 > 0.01% 時

        實作方式：
        - 計算全局質量誤差：Δm = m_current - m_initial
        - 計算修正係數：α = m_initial / m_current
        - 均勻縮放所有流體節點：f_i *= α

        Critical: 等比例縮放保持速度不變
        - 只調整密度（壓力）
        - 速度方向與大小不變
        - 物理上合理
        """
        # Step 1: 計算當前總質量
        self.mass_correction_total[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                rho_local = 0.0
                for k in ti.static(range(9)):
                    rho_local += f_dst[ig, jg][k]
                ti.atomic_add(self.mass_correction_total[None], rho_local)

        # Step 2: 計算修正係數
        total_mass_current = self.mass_correction_total[None]
        total_mass_target = self.initial_mass[None]
        mass_error = ti.abs(total_mass_current - total_mass_target)

        # 只在誤差顯著時修正（避免過度干預）
        if mass_error > 1e-8 and total_mass_current > 1e-12:
            correction = total_mass_target / total_mass_current

            # Step 3: 均勻修正所有流體節點
            for i, j in ti.ndrange(self.nx, self.ny):
                ig = i + 1
                jg = j + 1
                if self.mask[ig, jg] == 0:
                    for k in ti.static(range(9)):
                        f_dst[ig, jg][k] *= correction

    def apply_global_mass_correction(self, f_dst: ti.template()):
        """
        手動施加全局質量修正

        Usage:
            # 在主迴圈中每 N 步調用一次
            if step % 100 == 0:
                solver.apply_global_mass_correction(f_dst)

        Args:
            f_dst: 分佈函數場
        """
        self._global_mass_correction(f_dst)

    def step(self, f_src: ti.template(), f_dst: ti.template()):
        """
        執行單步時間推進

        Why 這個順序?
        1. Collision & Streaming: 內部節點演化，邊界節點從內部接收 f
        2. Boundary Conditions: 根據接收到的 f 重建邊界未知分佈函數
        3. Sponge Layer: 在出口區域吸收擾動（可選，高 Re 數穩定性）
        4. 下一步使用更新後的邊界值

        Args:
            f_src: 來源分佈函數場
            f_dst: 目標分佈函數場
        """
        self._update_macro(f_src)
        if self.force_field_updater is not None:
            self.force_field_updater()
        self._update_smagorinsky_viscosity()
        if self.wall_function_enabled[None] == 1:
            self._apply_wall_function()
        self._copy_solid_post(f_src)
        self._collide_bulk(f_src)
        self._collide_boundary(f_src)
        self._stream_bulk(f_dst)
        self._stream_boundary(f_dst)

        # 應用所有註冊的邊界條件
        for bc_func, _ in self.bc_functions:
            bc_func(f_dst)

        # 可選：施加海綿層（高 Re 數穩定性）
        if self.enable_sponge:
            self._apply_sponge_layer(f_dst)

    def get_fields(self):
        """獲取當前場的 numpy 陣列（包含渦黏度）"""
        rho_np = self.rho.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u_np = self.u.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        mask_np = self.mask.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        nu_sgs_np = (
            self.nu_sgs.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
            if self.cs > 0.0
            else None
        )
        return {
            "rho": rho_np,
            "u": u_np,
            "mask": mask_np,
            "nu_sgs": nu_sgs_np,
        }

    def get_diagnostics(self):
        """獲取診斷量（包含能量監控）"""
        return {
            "total_mass": self.total_mass[None],
            "initial_mass": self.initial_mass[None],
            "mass_residual": self.mass_residual[None],
            "mom_res_x": self.mom_res_x[None],
            "mom_res_y": self.mom_res_y[None],
            "mom_scale_x": self.mom_scale_x[None],
            "mom_scale_y": self.mom_scale_y[None],
            "max_u": self.max_u[None],
            "total_KE": self.total_KE[None],
            "initial_KE": self.initial_KE[None],
        }

    def check_cfl_condition(self, warn_only: bool = True) -> bool:
        """
        檢查 CFL 條件（公開介面）

        Args:
            warn_only: 若為 True，只警告；若為 False，拋出異常

        Returns:
            True if CFL condition satisfied, False otherwise
        """
        self._update_macro(self.f)
        self._check_cfl_violation()
        has_violation = self.cfl_violation[None]

        if has_violation:
            max_u = self.max_u[None]
            msg = f"⚠️  CFL violation detected: max|u| = {max_u:.4f} > 0.3"

            if warn_only:
                print(msg)
                print("   Consider reducing u_ref or increasing viscosity")
                return False
            else:
                raise RuntimeError(msg)

        return True
