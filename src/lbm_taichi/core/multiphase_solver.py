"""
多相 LBM 求解器（Shan-Chen, 雙組分）
==================================

What:
- 提供雙組分 Shan-Chen pseudopotential 的最小可用多相 LBM 求解器
- 支援表面張力（由交互作用強度 G 控制）

Why:
- 用最少變動實作多相/界面動力學，避免破壞單相流程
- 允許 Rayleigh–Taylor 不穩定性在真實分層下演化

When:
- 需要密度分層與界面效應（表面張力）時
- 想在不引入複雜多相模型的情況下快速驗證
"""

from typing import Callable, Optional, Tuple

import numpy as np
import taichi as ti


@ti.data_oriented
class MultiphaseLBMSolver:
    """
    雙組分 Shan-Chen 多相 LBM 求解器（D2Q9, BGK/MRT）

    What:
    - 以兩套分佈函數描述兩相組分 (A/B)
    - 使用 Shan-Chen 交互作用力形成界面與表面張力

    Why:
    - 實作成本低、可形成界面、適合作為最小可用多相基礎

    When:
    - 需要 RT、密度分層、界面張力的最小可行模擬
    """

    solver_family = "lbm"
    equation_set = "multiphase"
    regime = "low_mach"

    def __init__(
        self,
        nx: int,
        ny: int,
        tau_a: float = 0.8,
        tau_b: float = 0.8,
        g_interaction: float = 3.5,
        collision_model: str = "mrt",
        gravity: Tuple[float, float] = (0.0, 0.0),
        gravity_mode: str = "buoyancy",
        rho_floor: float = 1e-6,
        u_cap: float = 0.08,
        force_cap: float = 5e-4,
        recolor_beta: float = 0.0,
        g_wall_a: float = 0.0,
        g_wall_b: float = 0.0,
        psi_wall: float = 1.0,
        mass_correction_interval: int = 0,
        mass_correction_strength: float = 0.02,
    ):
        """
        Args:
            nx: X 方向格點數
            ny: Y 方向格點數
            tau_a: 組分 A 的鬆弛時間 (必須 > 0.5)
            tau_b: 組分 B 的鬆弛時間 (必須 > 0.5)
            g_interaction: Shan-Chen 交互作用強度 (正值產生相分離與表面張力)
            collision_model: 'mrt' 或 'bgk'
            gravity: 重力加速度向量 (gx, gy)
            gravity_mode: 'absolute' 或 'buoyancy'
            rho_floor: 密度下限（數值穩定）
            u_cap: 速度上限（數值穩定）
            force_cap: 力上限（數值穩定）
            recolor_beta: recoloring 強度（0 關閉，建議 0.5~0.9）
            g_wall_a: 組分 A 與壁面交互作用強度（濕潤性）
            g_wall_b: 組分 B 與壁面交互作用強度（濕潤性）
            psi_wall: 壁面 pseudo-potential 常數
            mass_correction_interval: 質量修正間隔（0 表示關閉）
            mass_correction_strength: 質量修正強度（0~1）
        """
        if tau_a <= 0.5 or tau_b <= 0.5:
            raise ValueError("tau_a 與 tau_b 必須 > 0.5")

        self.nx = nx
        self.ny = ny
        self.nx_g = nx + 2
        self.ny_g = ny + 2

        self.tau_a = tau_a
        self.tau_b = tau_b
        self.omega_a = 1.0 / tau_a
        self.omega_b = 1.0 / tau_b

        self.G = g_interaction
        self.rho_floor = rho_floor
        self.u_cap = max(1e-4, float(u_cap))
        self.force_cap = max(1e-8, float(force_cap))
        self.recolor_beta = ti.field(dtype=ti.f32, shape=())
        self.recolor_beta[None] = max(0.0, float(recolor_beta))
        self.collision_model = collision_model.lower()
        if self.collision_model not in {"mrt", "bgk"}:
            raise ValueError(f"Invalid collision_model: {collision_model}")
        self.gravity_mode = gravity_mode.lower()
        if self.gravity_mode not in {"absolute", "buoyancy"}:
            raise ValueError(f"Invalid gravity_mode: {gravity_mode}")
        if self.G <= 0.0:
            print(
                "\n⚠️  Warning: g_interaction 建議為正值（多相相分離）。"
                " 若為負值可能造成非物理混合。"
            )

        # 邊界控制旗標
        self.periodic_x = ti.field(dtype=ti.i32, shape=())
        self.periodic_y = ti.field(dtype=ti.i32, shape=())
        self.periodic_x[None] = 0
        self.periodic_y[None] = 0

        # 障礙物遮罩（1=固體, 0=流體）
        self.mask = ti.field(dtype=ti.i32, shape=(self.nx_g, self.ny_g))
        self.mask.fill(0)

        # 壁面旗標（濕潤性用）
        self.wall_top = ti.field(dtype=ti.i32, shape=())
        self.wall_bottom = ti.field(dtype=ti.i32, shape=())
        self.wall_left = ti.field(dtype=ti.i32, shape=())
        self.wall_right = ti.field(dtype=ti.i32, shape=())
        self.wall_top[None] = 0
        self.wall_bottom[None] = 0
        self.wall_left[None] = 0
        self.wall_right[None] = 0

        # D2Q9 常數
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)

        # MRT 矩陣
        self.M = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.M_inv = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.S = ti.field(dtype=ti.f32, shape=9)

        # 分佈函數（雙組分）
        self.fA = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.fB = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.fA_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.fB_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.fA_post = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.fB_post = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        # 宏觀量
        self.rhoA = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.rhoB = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        # 交互作用力
        self.forceA = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.forceB = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        # 濕潤性參數
        self.g_wall_a = ti.field(dtype=ti.f32, shape=())
        self.g_wall_b = ti.field(dtype=ti.f32, shape=())
        self.psi_wall = ti.field(dtype=ti.f32, shape=())
        self.g_wall_a[None] = g_wall_a
        self.g_wall_b[None] = g_wall_b
        self.psi_wall[None] = psi_wall

        # 重力
        self.gravity = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.gravity[None] = ti.Vector([gravity[0], gravity[1]])
        self.rho_ref = ti.field(dtype=ti.f32, shape=())
        self.fluid_cell_count = ti.field(dtype=ti.i32, shape=())
        self._count_fluid_cells()
        self.rho_ref[None] = 1.0

        # 質量修正
        self.mass_correction_interval = ti.field(dtype=ti.i32, shape=())
        self.mass_correction_strength = ti.field(dtype=ti.f32, shape=())
        self.mass_correction_counter = ti.field(dtype=ti.i32, shape=())
        self.mass_correction_interval[None] = mass_correction_interval
        self.mass_correction_strength[None] = mass_correction_strength
        self.mass_correction_counter[None] = 0
        self.mass_correction_total_a = ti.field(dtype=ti.f32, shape=())
        self.mass_correction_total_b = ti.field(dtype=ti.f32, shape=())
        self.initial_mass_a = ti.field(dtype=ti.f32, shape=())
        self.initial_mass_b = ti.field(dtype=ti.f32, shape=())

        self.step_count = 0
        self.bc_functions = []
        self._init_constants()
        self.reset()

    def _init_constants(self):
        """
        What: 初始化 D2Q9 權重與速度集合
        Why: 統一 D2Q9 常數，確保模型一致性
        When: solver 初始化時
        """
        w_np = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
            dtype=np.float32,
        )
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
        inv_np = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

        # MRT 變換矩陣 (Lallemand & Luo, 2000)
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

        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)
        self._set_mrt_matrices(M_np, M_inv_np)

    @ti.kernel
    def _set_mrt_matrices(
        self, m_arr: ti.types.ndarray(), minv_arr: ti.types.ndarray()
    ):
        for i, j in ti.ndrange(9, 9):
            self.M[None][i, j] = m_arr[i, j]
            self.M_inv[None][i, j] = minv_arr[i, j]

        # MRT 鬆弛參數（非黏度項）
        s_e = 1.64
        s_eps = 1.54
        s_q = 1.2
        self.S[0] = 0.0
        self.S[1] = s_e
        self.S[2] = s_eps
        self.S[3] = 0.0
        self.S[4] = s_q
        self.S[5] = 0.0
        self.S[6] = s_q
        self.S[7] = 0.0
        self.S[8] = 0.0

    def reset(self):
        """
        What: 重置場變數（初始化為均勻靜止）
        Why: 確保 solver 於乾淨狀態啟動
        When: 建立 solver 或需要重置模擬時
        """
        self._reset_fields(1.0, 1.0)
        self.reset_mass_baseline()
        self.mass_correction_counter[None] = 0

    def _reset_fields(self, rho_a: float, rho_b: float):
        """
        What: 以指定密度初始化所有場
        Why: 讓初始化狀態可控
        When: reset 或初始化特定狀態
        """
        rhoA_g = np.full((self.nx_g, self.ny_g), rho_a, dtype=np.float32)
        rhoB_g = np.full((self.nx_g, self.ny_g), rho_b, dtype=np.float32)
        u_g = np.zeros((self.nx_g, self.ny_g, 2), dtype=np.float32)
        self.rhoA.from_numpy(rhoA_g)
        self.rhoB.from_numpy(rhoB_g)
        self.u.from_numpy(u_g)
        self._initialize_distributions()
        self.reset_mass_baseline()

    def set_initial_fields(
        self,
        rho_a: np.ndarray,
        rho_b: np.ndarray,
        u: Optional[np.ndarray] = None,
    ):
        """
        What: 以使用者指定的場初始化
        Why: 支援多相初始分層/擾動
        When: 建立特定案例（如 Rayleigh–Taylor）

        Args:
            rho_a: (nx, ny) 組分 A 密度
            rho_b: (nx, ny) 組分 B 密度
            u: (nx, ny, 2) 初始速度場（可選，預設為 0）
        """
        if rho_a.shape != (self.nx, self.ny) or rho_b.shape != (self.nx, self.ny):
            raise ValueError("rho_a/rho_b 形狀需為 (nx, ny)")

        rhoA_g = _pad_edge(rho_a.astype(np.float32))
        rhoB_g = _pad_edge(rho_b.astype(np.float32))
        if u is None:
            u = np.zeros((self.nx, self.ny, 2), dtype=np.float32)
        if u.shape != (self.nx, self.ny, 2):
            raise ValueError("u 形狀需為 (nx, ny, 2)")
        u_g = _pad_edge_vec(u.astype(np.float32))

        self.rhoA.from_numpy(rhoA_g)
        self.rhoB.from_numpy(rhoB_g)
        self.u.from_numpy(u_g)
        self._initialize_distributions()

    def set_interaction_strength(self, g_interaction: float):
        """
        What: 設定 Shan-Chen 交互作用強度 G
        Why: G 控制表面張力與相分離強度
        When: 需要調整界面張力或密度比時
        """
        self.G = g_interaction
        if self.G <= 0.0:
            print(
                "\n⚠️  Warning: g_interaction 建議為正值（多相相分離）。"
                " 若為負值可能造成非物理混合。"
            )

    def set_gravity(self, gx: float, gy: float):
        """
        What: 設定重力加速度向量
        Why: RT 等案例需要體積力驅動
        When: 啟動或更新重力
        """
        self.gravity[None] = ti.Vector([gx, gy])

    def set_gravity_mode(self, mode: str):
        """
        What: 設定重力模式
        Why: 'buoyancy' 可移除整體漂移，只保留密度差驅動
        When: RT 等需要浮力不穩定時
        """
        mode = mode.lower()
        if mode not in {"absolute", "buoyancy"}:
            raise ValueError(f"Invalid gravity_mode: {mode}")
        self.gravity_mode = mode

    def set_recolor_beta(self, beta: float):
        """
        What: 設定 recoloring 強度
        Why: 控制界面銳化，降低數值混相
        When: 多相案例調參
        """
        self.recolor_beta[None] = max(0.0, float(beta))

    def set_periodic(self, direction: str, enabled: bool = True):
        """
        What: 設定週期邊界旗標
        Why: 週期邊界在 streaming 與力計算中需明確
        When: 使用 periodic boundary 時
        """
        if direction == "x":
            self.periodic_x[None] = 1 if enabled else 0
        elif direction == "y":
            self.periodic_y[None] = 1 if enabled else 0
        else:
            raise ValueError("direction 必須為 'x' 或 'y'")

    def enable_wall_boundary(self, location: str, enabled: bool = True):
        """
        What: 設定壁面旗標（濕潤性用）
        Why: Shan-Chen wall force 需要知道哪些邊界是固體
        When: 使用 no-slip/moving wall 或需要 wetting
        """
        flag = 1 if enabled else 0
        if location == "top":
            self.wall_top[None] = flag
        elif location == "bottom":
            self.wall_bottom[None] = flag
        elif location == "left":
            self.wall_left[None] = flag
        elif location == "right":
            self.wall_right[None] = flag
        else:
            raise ValueError(f"Invalid location: {location}")

    def set_wetting(self, g_wall_a: float, g_wall_b: float, psi_wall: float = 1.0):
        """
        What: 設定濕潤性參數（壁面交互作用）
        Why: 控制接觸角與相偏好
        When: 需要固壁濕潤性時
        """
        self.g_wall_a[None] = g_wall_a
        self.g_wall_b[None] = g_wall_b
        self.psi_wall[None] = psi_wall

    def enable_mass_correction(self, interval: int, strength: float = 0.02):
        """
        What: 啟用質量修正（低頻）
        Why: 抑制長時間質量漂移
        When: 長時間多相模擬
        """
        self.mass_correction_interval[None] = int(interval)
        self.mass_correction_strength[None] = float(strength)
        self.mass_correction_counter[None] = 0

    def disable_mass_correction(self):
        """關閉質量修正"""
        self.mass_correction_interval[None] = 0
        self.mass_correction_counter[None] = 0

    def set_obstacle(self, mask_array: np.ndarray):
        """
        What: 設定內部固體障礙物遮罩
        Why: 支援障礙物與固體區域
        When: 需要固體邊界或障礙物時
        """
        if mask_array.shape == (self.nx, self.ny):
            mask_np = mask_array.astype(np.int32)
        elif mask_array.shape == (self.ny, self.nx):
            mask_np = mask_array.T.astype(np.int32)
        else:
            raise ValueError(
                f"Mask shape {mask_array.shape} incompatible with grid ({self.nx}, {self.ny})"
            )
        mask_g = np.zeros((self.nx_g, self.ny_g), dtype=np.int32)
        mask_g[1 : self.nx + 1, 1 : self.ny + 1] = mask_np
        self.mask.from_numpy(mask_g)
        self._count_fluid_cells()
        self._initialize_distributions()
        self.reset_mass_baseline()

    def clear_obstacles(self):
        """清除所有內部障礙物遮罩"""
        self.mask.fill(0)
        self._count_fluid_cells()
        self._initialize_distributions()
        self.reset_mass_baseline()

    def add_boundary_condition(self, bc_func: Callable, name: str = "custom_bc"):
        """
        What: 註冊邊界條件函式
        Why: 維持與單相 solver 相同的擴充模式
        When: 建立案例時配置邊界條件
        """
        self.bc_functions.append((bc_func, name))

    def apply_boundary_conditions(self):
        """
        What: 手動施加所有邊界條件
        Why: 初始化或需要強制更新邊界
        When: 初始化後、或外部想強制刷新邊界
        """
        for bc_func, _ in self.bc_functions:
            bc_func(self.fA, self.fB)

    @ti.kernel
    def _initialize_distributions(self):
        """
        What: 以目前 rho/u 初始化平衡分佈
        Why: 讓 f 與宏觀量一致
        When: 初始化或重設場
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                self.rhoA[ig, jg] = 0.0
                self.rhoB[ig, jg] = 0.0
                self.u[ig, jg] = ti.Vector([0.0, 0.0])
                for k in ti.static(range(9)):
                    self.fA[ig, jg][k] = 0.0
                    self.fB[ig, jg][k] = 0.0
                    self.fA_new[ig, jg][k] = 0.0
                    self.fB_new[ig, jg][k] = 0.0
                    self.fA_post[ig, jg][k] = 0.0
                    self.fB_post[ig, jg][k] = 0.0
            else:
                rho_a = self.rhoA[ig, jg]
                rho_b = self.rhoB[ig, jg]
                u_vec = self.u[ig, jg]
                u_sq = u_vec.dot(u_vec)
                for k in ti.static(range(9)):
                    e_k = ti.cast(self.e[k], ti.f32)
                    eu = e_k.dot(u_vec)
                    feq_a = (
                        self.w[k]
                        * rho_a
                        * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                    )
                    feq_b = (
                        self.w[k]
                        * rho_b
                        * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                    )
                    self.fA[ig, jg][k] = feq_a
                    self.fB[ig, jg][k] = feq_b
                    self.fA_new[ig, jg][k] = feq_a
                    self.fB_new[ig, jg][k] = feq_b
                    self.fA_post[ig, jg][k] = feq_a
                    self.fB_post[ig, jg][k] = feq_b

    @ti.kernel
    def _compute_density(self):
        """
        What: 計算各組分密度與總密度
        Why: Shan-Chen 力與碰撞需要 rho
        When: 每步更新前
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                self.rhoA[ig, jg] = 0.0
                self.rhoB[ig, jg] = 0.0
                self.rho[ig, jg] = 0.0
                self.u[ig, jg] = ti.Vector([0.0, 0.0])
            else:
                rho_a = 0.0
                rho_b = 0.0
                for k in ti.static(range(9)):
                    rho_a += self.fA[ig, jg][k]
                    rho_b += self.fB[ig, jg][k]
                rho_a = ti.max(rho_a, self.rho_floor)
                rho_b = ti.max(rho_b, self.rho_floor)
                self.rhoA[ig, jg] = rho_a
                self.rhoB[ig, jg] = rho_b
                self.rho[ig, jg] = rho_a + rho_b

    @ti.kernel
    def _compute_interaction_forces(self):
        """
        What: 計算 Shan-Chen 交互作用力與重力
        Why: 形成界面張力與相分離
        When: 每步碰撞前
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                self.forceA[ig, jg] = ti.Vector([0.0, 0.0])
                self.forceB[ig, jg] = ti.Vector([0.0, 0.0])
            else:
                rho_a = self.rhoA[ig, jg]
                rho_b = self.rhoB[ig, jg]
                psi_a = 1.0 - ti.exp(-rho_a)
                psi_b = 1.0 - ti.exp(-rho_b)

                sum_ab = ti.Vector([0.0, 0.0])
                sum_ba = ti.Vector([0.0, 0.0])
                sum_wall = ti.Vector([0.0, 0.0])

                for k in ti.static(range(9)):
                    ni = ig + self.e[k][0]
                    nj = jg + self.e[k][1]

                    # 週期邊界處理
                    if self.periodic_x[None] == 1:
                        if ni < 1:
                            ni = self.nx
                        elif ni > self.nx:
                            ni = 1
                    if self.periodic_y[None] == 1:
                        if nj < 1:
                            nj = self.ny
                        elif nj > self.ny:
                            nj = 1

                    e_k = ti.cast(self.e[k], ti.f32)
                    out_x = (ni < 1) | (ni > self.nx)
                    out_y = (nj < 1) | (nj > self.ny)
                    out_domain = out_x | out_y

                    if out_domain:
                        # 邊界外：視為壁面（若啟用）
                        wall_hit = 0
                        if ni < 1 and self.wall_left[None] == 1:
                            wall_hit = 1
                        if ni > self.nx and self.wall_right[None] == 1:
                            wall_hit = 1
                        if nj < 1 and self.wall_bottom[None] == 1:
                            wall_hit = 1
                        if nj > self.ny and self.wall_top[None] == 1:
                            wall_hit = 1
                        if wall_hit == 1:
                            sum_wall += self.w[k] * self.psi_wall[None] * e_k
                    else:
                        # 內部障礙物視為壁面
                        if self.mask[ni, nj] == 1:
                            sum_wall += self.w[k] * self.psi_wall[None] * e_k
                        else:
                            psi_b_n = 1.0 - ti.exp(-self.rhoB[ni, nj])
                            psi_a_n = 1.0 - ti.exp(-self.rhoA[ni, nj])
                            sum_ab += self.w[k] * psi_b_n * e_k
                            sum_ba += self.w[k] * psi_a_n * e_k

                force_a = -self.G * psi_a * sum_ab
                force_b = -self.G * psi_b * sum_ba

                if ti.abs(self.g_wall_a[None]) > 0.0:
                    force_a += -self.g_wall_a[None] * psi_a * sum_wall
                if ti.abs(self.g_wall_b[None]) > 0.0:
                    force_b += -self.g_wall_b[None] * psi_b * sum_wall

                g = self.gravity[None]
                if self.gravity_mode == "buoyancy":
                    rho_total = rho_a + rho_b
                    rho_ref = self.rho_ref[None]
                    buoy = (rho_total - rho_ref) * g
                    if rho_total > 1e-12:
                        force_a += (rho_a / rho_total) * buoy
                        force_b += (rho_b / rho_total) * buoy
                else:
                    force_a += rho_a * g
                    force_b += rho_b * g

                force_a = self._clip_vec(force_a, self.force_cap)
                force_b = self._clip_vec(force_b, self.force_cap)
                self.forceA[ig, jg] = force_a
                self.forceB[ig, jg] = force_b

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
    def _clip_vec(self, v, vmax: ti.f32):
        norm_v = v.norm()
        out = v
        if norm_v > vmax and norm_v > 1e-12:
            out = v * (vmax / norm_v)
        return out

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

    @ti.kernel
    def _collide_bgk(self):
        """
        What: BGK 碰撞（含 Guo 力項）
        Why: 將交互作用力正確耦合至動量方程
        When: 每步碰撞階段
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                self.u[ig, jg] = ti.Vector([0.0, 0.0])
                for k in ti.static(range(9)):
                    self.fA_post[ig, jg][k] = self.fA[ig, jg][k]
                    self.fB_post[ig, jg][k] = self.fB[ig, jg][k]
                continue

            rho_a = self.rhoA[ig, jg]
            rho_b = self.rhoB[ig, jg]
            rho = rho_a + rho_b
            force_a = self.forceA[ig, jg]
            force_b = self.forceB[ig, jg]
            force = force_a + force_b

            momentum = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                e_k = ti.cast(self.e[k], ti.f32)
                momentum += (self.fA[ig, jg][k] + self.fB[ig, jg][k]) * e_k

            u_vec = ti.Vector([0.0, 0.0])
            if rho > 0.0:
                u_vec = (momentum + 0.5 * force) / rho
            u_vec = self._clip_vec(u_vec, self.u_cap)
            self.u[ig, jg] = u_vec
            u_a = u_vec
            u_b = u_vec
            if rho_a > self.rho_floor:
                u_a = u_vec + self.tau_a * force_a / rho_a
            if rho_b > self.rho_floor:
                u_b = u_vec + self.tau_b * force_b / rho_b
            u_a = self._clip_vec(u_a, self.u_cap)
            u_b = self._clip_vec(u_b, self.u_cap)

            u_sq_a = u_a.dot(u_a)
            u_sq_b = u_b.dot(u_b)
            rho_a_safe = ti.max(rho_a, self.rho_floor)
            rho_b_safe = ti.max(rho_b, self.rho_floor)
            for k in ti.static(range(9)):
                e_k = ti.cast(self.e[k], ti.f32)
                eu_a = e_k.dot(u_a)
                eu_b = e_k.dot(u_b)

                feq_a = (
                    self.w[k]
                    * rho_a
                    * (1.0 + 3.0 * eu_a + 4.5 * eu_a * eu_a - 1.5 * u_sq_a)
                )
                feq_b = (
                    self.w[k]
                    * rho_b
                    * (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * u_sq_b)
                )

                f_post_a = (
                    self.fA[ig, jg][k]
                    - self.omega_a * (self.fA[ig, jg][k] - feq_a)
                )
                f_post_b = (
                    self.fB[ig, jg][k]
                    - self.omega_b * (self.fB[ig, jg][k] - feq_b)
                )

                invalid_a = (f_post_a != f_post_a) or (ti.abs(f_post_a) > 1e6)
                invalid_b = (f_post_b != f_post_b) or (ti.abs(f_post_b) > 1e6)
                f_post_a = ti.select(invalid_a, self.w[k] * rho_a_safe, f_post_a)
                f_post_b = ti.select(invalid_b, self.w[k] * rho_b_safe, f_post_b)
                self.fA_post[ig, jg][k] = f_post_a
                self.fB_post[ig, jg][k] = f_post_b

    @ti.kernel
    def _collide_mrt(self):
        """
        What: MRT 碰撞（含 Guo 力項）
        Why: 提升穩定性，分離不同矩的鬆弛尺度
        When: 需要穩定多相界面時
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                self.u[ig, jg] = ti.Vector([0.0, 0.0])
                for k in ti.static(range(9)):
                    self.fA_post[ig, jg][k] = self.fA[ig, jg][k]
                    self.fB_post[ig, jg][k] = self.fB[ig, jg][k]
                continue

            rho_a = self.rhoA[ig, jg]
            rho_b = self.rhoB[ig, jg]
            rho = rho_a + rho_b
            force_a = self.forceA[ig, jg]
            force_b = self.forceB[ig, jg]
            force = force_a + force_b

            momentum = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                e_k = ti.cast(self.e[k], ti.f32)
                momentum += (self.fA[ig, jg][k] + self.fB[ig, jg][k]) * e_k

            u_vec = ti.Vector([0.0, 0.0])
            if rho > 0.0:
                u_vec = (momentum + 0.5 * force) / rho
            u_vec = self._clip_vec(u_vec, self.u_cap)
            self.u[ig, jg] = u_vec
            u_a = u_vec
            u_b = u_vec
            if rho_a > self.rho_floor:
                u_a = u_vec + self.tau_a * force_a / rho_a
            if rho_b > self.rho_floor:
                u_b = u_vec + self.tau_b * force_b / rho_b
            u_a = self._clip_vec(u_a, self.u_cap)
            u_b = self._clip_vec(u_b, self.u_cap)

            # MRT for component A
            f_vec_a = self.fA[ig, jg]
            m_a = self.M[None] @ f_vec_a
            meq_a = self._compute_meq(rho_a, u_a)
            s_nu_a = 1.0 / self.tau_a
            m_star_a = ti.Vector([0.0] * 9)
            for k in ti.static(range(9)):
                rate = self.S[k]
                if k == 7 or k == 8:
                    rate = s_nu_a
                m_star_a[k] = m_a[k] - rate * (m_a[k] - meq_a[k])
            f_post_a = self.M_inv[None] @ m_star_a

            # MRT for component B
            f_vec_b = self.fB[ig, jg]
            m_b = self.M[None] @ f_vec_b
            meq_b = self._compute_meq(rho_b, u_b)
            s_nu_b = 1.0 / self.tau_b
            m_star_b = ti.Vector([0.0] * 9)
            for k in ti.static(range(9)):
                rate = self.S[k]
                if k == 7 or k == 8:
                    rate = s_nu_b
                m_star_b[k] = m_b[k] - rate * (m_b[k] - meq_b[k])
            f_post_b = self.M_inv[None] @ m_star_b

            rho_a_safe = ti.max(rho_a, self.rho_floor)
            rho_b_safe = ti.max(rho_b, self.rho_floor)
            for k in ti.static(range(9)):
                invalid_a = (f_post_a[k] != f_post_a[k]) or (ti.abs(f_post_a[k]) > 1e6)
                invalid_b = (f_post_b[k] != f_post_b[k]) or (ti.abs(f_post_b[k]) > 1e6)
                self.fA_post[ig, jg][k] = ti.select(
                    invalid_a, self.w[k] * rho_a_safe, f_post_a[k]
                )
                self.fB_post[ig, jg][k] = ti.select(
                    invalid_b, self.w[k] * rho_b_safe, f_post_b[k]
                )

    @ti.kernel
    def _recolor_post_collision(self):
        """
        What: 對碰撞後分佈做 recoloring（保持總分佈，重分配 A/B）
        Why: 降低數值混相，維持清晰界面
        When: 碰撞後、串流前
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                continue

            rho_a = self.rhoA[ig, jg]
            rho_b = self.rhoB[ig, jg]
            rho = rho_a + rho_b
            if rho <= 1e-12:
                continue

            phi_c = (rho_a - rho_b) / (rho + 1e-12)
            grad = ti.Vector([0.0, 0.0])
            for k in ti.static(range(1, 9)):
                ni = ig + self.e[k][0]
                nj = jg + self.e[k][1]

                if self.periodic_x[None] == 1:
                    if ni < 1:
                        ni = self.nx
                    elif ni > self.nx:
                        ni = 1
                if self.periodic_y[None] == 1:
                    if nj < 1:
                        nj = self.ny
                    elif nj > self.ny:
                        nj = 1

                phi_n = phi_c
                in_domain = (ni >= 1 and ni <= self.nx and nj >= 1 and nj <= self.ny)
                if in_domain and self.mask[ni, nj] == 0:
                    rho_a_n = self.rhoA[ni, nj]
                    rho_b_n = self.rhoB[ni, nj]
                    rho_n = rho_a_n + rho_b_n
                    if rho_n > 1e-12:
                        phi_n = (rho_a_n - rho_b_n) / rho_n

                e_k = ti.cast(self.e[k], ti.f32)
                grad += self.w[k] * phi_n * e_k

            n_hat = ti.Vector([0.0, 0.0])
            grad_norm = grad.norm()
            if grad_norm > 1e-12:
                n_hat = grad / grad_norm

            mix = (rho_a * rho_b) / (rho * rho + 1e-12)
            beta = self.recolor_beta[None]
            ratio_a = rho_a / rho

            for k in ti.static(range(9)):
                e_k = ti.cast(self.e[k], ti.f32)
                cos_theta = e_k.dot(n_hat)
                f_tot = self.fA_post[ig, jg][k] + self.fB_post[ig, jg][k]
                delta = beta * mix * self.w[k] * rho * cos_theta
                f_a = ratio_a * f_tot + delta
                f_a = ti.max(0.0, ti.min(f_tot, f_a))
                self.fA_post[ig, jg][k] = f_a
                self.fB_post[ig, jg][k] = f_tot - f_a

    @ti.kernel
    def _stream(self):
        """
        What: Streaming 步驟（支援週期邊界）
        Why: 推進分佈函數至下一步
        When: 碰撞之後
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 1:
                continue
            for k in ti.static(range(9)):
                dest_i = ig + self.e[k][0]
                dest_j = jg + self.e[k][1]

                if self.periodic_x[None] == 1:
                    if dest_i < 1:
                        dest_i = self.nx
                    elif dest_i > self.nx:
                        dest_i = 1

                if self.periodic_y[None] == 1:
                    if dest_j < 1:
                        dest_j = self.ny
                    elif dest_j > self.ny:
                        dest_j = 1

                solid = 0
                if 1 <= dest_i <= self.nx and 1 <= dest_j <= self.ny:
                    if self.mask[dest_i, dest_j] == 1:
                        solid = 1

                if solid == 1:
                    inv_k = self.inv[k]
                    self.fA_new[ig, jg][inv_k] = self.fA_post[ig, jg][k]
                    self.fB_new[ig, jg][inv_k] = self.fB_post[ig, jg][k]
                else:
                    self.fA_new[dest_i, dest_j][k] = self.fA_post[ig, jg][k]
                    self.fB_new[dest_i, dest_j][k] = self.fB_post[ig, jg][k]

    @ti.kernel
    def _compute_mass_totals(self, fA: ti.template(), fB: ti.template()):
        self.mass_correction_total_a[None] = 0.0
        self.mass_correction_total_b[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                rho_a = 0.0
                rho_b = 0.0
                for k in ti.static(range(9)):
                    rho_a += fA[ig, jg][k]
                    rho_b += fB[ig, jg][k]
                ti.atomic_add(self.mass_correction_total_a[None], rho_a)
                ti.atomic_add(self.mass_correction_total_b[None], rho_b)

    @ti.kernel
    def _count_fluid_cells(self):
        self.fluid_cell_count[None] = 0
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                self.fluid_cell_count[None] += 1

    @ti.kernel
    def _update_rho_reference(self):
        total = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                total += self.rho[ig, jg]
        count = ti.cast(self.fluid_cell_count[None], ti.f32)
        if count > 0.0:
            self.rho_ref[None] = total / count
        else:
            self.rho_ref[None] = 1.0

    @ti.kernel
    def _apply_mass_correction(
        self, fA: ti.template(), fB: ti.template(), corr_a: ti.f32, corr_b: ti.f32
    ):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            if self.mask[ig, jg] == 0:
                for k in ti.static(range(9)):
                    fA[ig, jg][k] *= corr_a
                    fB[ig, jg][k] *= corr_b

    @ti.kernel
    def _commit_new_distributions(self):
        """
        What: 將新分佈場提交為當前場
        Why: 避免 Python 交換 field 參考在 Taichi kernel 下失效
        When: 每步 streaming + BC 之後
        """
        for i, j in ti.ndrange(self.nx_g, self.ny_g):
            self.fA[i, j] = self.fA_new[i, j]
            self.fB[i, j] = self.fB_new[i, j]

    def reset_mass_baseline(self):
        """
        What: 設定質量基準
        Why: 全局質量修正與診斷需要基準值
        When: 初始化或重設場後
        """
        self._compute_mass_totals(self.fA, self.fB)
        self.initial_mass_a[None] = self.mass_correction_total_a[None]
        self.initial_mass_b[None] = self.mass_correction_total_b[None]

    def step(self):
        """
        What: 執行單步多相 LBM 推進
        Why: 統一更新密度、力、碰撞、串流與邊界
        When: 時間演化主迴圈
        """
        self._compute_density()
        if self.gravity_mode == "buoyancy":
            self._update_rho_reference()
        self._compute_interaction_forces()
        if self.collision_model == "mrt":
            self._collide_mrt()
        else:
            self._collide_bgk()
        if self.recolor_beta[None] > 1e-8:
            self._recolor_post_collision()
        self._stream()

        for bc_func, _ in self.bc_functions:
            bc_func(self.fA_new, self.fB_new)

        # 低頻質量修正
        if self.mass_correction_interval[None] > 0:
            self.mass_correction_counter[None] += 1
            if self.mass_correction_counter[None] >= self.mass_correction_interval[None]:
                self._compute_mass_totals(self.fA_new, self.fB_new)
                total_a = self.mass_correction_total_a[None]
                total_b = self.mass_correction_total_b[None]
                corr_a = 1.0
                corr_b = 1.0
                strength = self.mass_correction_strength[None]
                if total_a > 1e-12:
                    corr_a = (1.0 - strength) + strength * (
                        self.initial_mass_a[None] / total_a
                    )
                if total_b > 1e-12:
                    corr_b = (1.0 - strength) + strength * (
                        self.initial_mass_b[None] / total_b
                    )
                self._apply_mass_correction(self.fA_new, self.fB_new, corr_a, corr_b)
                self.mass_correction_counter[None] = 0

        self._commit_new_distributions()
        self.step_count += 1

    def get_fields(self):
        """
        What: 匯出目前場
        Why: 後處理/可視化
        When: 需要輸出時
        """
        rhoA_np = self.rhoA.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        rhoB_np = self.rhoB.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        rho_np = self.rho.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u_np = self.u.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        mask_np = self.mask.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        phi = (rhoA_np - rhoB_np) / (rho_np + 1e-12)
        return {
            "rhoA": rhoA_np,
            "rhoB": rhoB_np,
            "rho": rho_np,
            "u": u_np,
            "mask": mask_np,
            "phi": phi,
        }

    def get_diagnostics(self):
        """
        What: 匯出診斷量供 CaseRunner / Protocol 層使用
        Why: SolverProtocol 要求 get_diagnostics；多相流的關鍵診斷是相分率與速度尺度
        """
        fields = self.get_fields()
        phi = fields["phi"]
        u = fields["u"]
        u_max = float(np.max(np.linalg.norm(u, axis=-1)))
        return {
            "step_count": int(self.step_count),
            "phi_min": float(phi.min()),
            "phi_max": float(phi.max()),
            "u_max": u_max,
        }


def _pad_edge(arr: np.ndarray) -> np.ndarray:
    """
    What: 為 (nx, ny) 陣列建立 ghost cells（edge replicate）
    Why: 方便初始化與邊界處理
    When: set_initial_fields 時
    """
    nx, ny = arr.shape
    out = np.zeros((nx + 2, ny + 2), dtype=arr.dtype)
    out[1 : nx + 1, 1 : ny + 1] = arr
    out[0, 1 : ny + 1] = arr[0, :]
    out[nx + 1, 1 : ny + 1] = arr[-1, :]
    out[:, 0] = out[:, 1]
    out[:, ny + 1] = out[:, ny]
    return out


def _pad_edge_vec(arr: np.ndarray) -> np.ndarray:
    """
    What: 為 (nx, ny, 2) 向量場建立 ghost cells
    Why: 初始化速度場一致性
    When: set_initial_fields 時
    """
    nx, ny, _ = arr.shape
    out = np.zeros((nx + 2, ny + 2, 2), dtype=arr.dtype)
    out[1 : nx + 1, 1 : ny + 1, :] = arr
    out[0, 1 : ny + 1, :] = arr[0, :, :]
    out[nx + 1, 1 : ny + 1, :] = arr[-1, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, ny + 1, :] = out[:, ny, :]
    return out
