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

    def step(self, g_src, g_dst):
        """
        執行溫度場單步推進（BGK 碰撞 + 串流）

        Why 先 update_temperature 再 collision?
        - 從 g_src 重算 T 確保每步一致性
        - 避免上一步 BC 修改造成的 T 不一致

        【注意 self.T 時序語義】
        呼叫 step(g_src, g_dst) 後，self.T 對應 g_src 的溫度（前一時間步）。
        若需要 g_dst 對應的最新溫度，請手動呼叫：
            thermal._update_temperature(g_dst)
        例如：在可視化或 Nusselt 數計算前需更新。
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
                # TODO: 固體熱邊界由上層 ThermalBoundaryConditions 處理
                #       （熱 Bounce-Back 或 Dirichlet g_dst），此處 continue 僅防止固體
                #       向外串流，固體內 g_dst 由 BC 負責重建。（Task 4 實作）
                continue

            # TODO: 效率優化：可改用 self.T[i, j]（已由 _update_temperature 計算，
            #       節省每節點 8 次 f32 讀取）。目前保持獨立計算以利可讀性。
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

    def compute_buoyancy(self):
        """
        計算浮力並更新 solver.force_field

        What: 計算 Boussinesq 浮力場並寫入 solver.force_field
        Why:  透過 force_field 保持與 LBMSolver Guo forcing scheme 的相容性；
              此函式由 solver.force_field_updater 每步自動呼叫，
              確保浮力使用最新溫度場（T 在流場計算前更新）

        呼叫時序: solver.step() → force_field_updater() → compute_buoyancy()
                  → 更新 force_field → 碰撞串流使用最新浮力
        """
        self._compute_buoyancy_kernel()

    @ti.kernel
    def _compute_buoyancy_kernel(self):
        """
        Boussinesq 浮力: F_y = ρ · g_gravity · β · (T - T_ref)

        Why Boussinesq?
            密度變化只在浮力項計入，流場仍視為不可壓縮。
            在 |T - T_ref| << T_ref 時近似有效。
        Why ρ 乘?
            LBM 中 force_field 是體積力密度（力/體積），即加速度 × ρ
        """
        for i, j in ti.ndrange((1, self.nx + 1), (1, self.ny + 1)):
            if self.solver.mask[i, j] == 1:
                self.solver.force_field[i, j] = ti.Vector([0.0, 0.0])
            else:
                T_loc = self.T[i, j]
                rho_loc = self.solver.rho[i, j]
                F_y = rho_loc * self.g_gravity * self.beta * (T_loc - self.T_ref)
                self.solver.force_field[i, j] = ti.Vector([0.0, F_y])

    def get_nusselt(self, T_bot: float, T_top: float) -> float:
        """
        計算 Nusselt 數（使用中間截面的溫度梯度）

        What: Nu = -H * mean(∂T/∂y|_{y=H/2}) / ΔT
        Why 用中間截面?
            - 避開壁面 BC 數值影響（壁面附近的梯度計算誤差較大）
            - 穩態下各橫截面的熱通量守恆，中間截面最具代表性
        Why 中心差分?
            - 二階精度，不引入人工偏移

        純導熱穩態: ∂T/∂y = -ΔT/H → Nu = 1

        Args:
            T_bot: 底壁溫度
            T_top: 頂壁溫度

        Returns:
            float: Nusselt 數（純導熱=1.0，對流越強則越大）

        【重要】Note: 呼叫前必須確保 self.T 已更新至最新時間步
                     （呼叫 self._update_temperature(g_dst)）
                     否則計算結果對應前一時間步的溫度。
        """
        # === 防衛檢查 1: ny 必須 >= 4 避免越界 ===
        if self.ny < 4:
            raise ValueError(
                f"get_nusselt() 需要 ny >= 4 以便計算中間截面梯度。"
                f"當前 ny={self.ny}。中心差分需讀取 T[i,j_mid±1]；"
                f"當 ny < 4 時 j_mid 靠近邊界，可能讀到 ghost cell。"
            )

        # === 防衛檢查 2: 溫度差不能過小（避免除以零）===
        delta_T = T_bot - T_top
        if abs(delta_T) < 1e-10:
            raise ValueError(
                f"get_nusselt() 偵測到溫度差過小（ΔT={delta_T:.2e}），"
                f"無法計算有意義的 Nusselt 數。"
                f"請確保 T_bot ≠ T_top。"
            )

        # === 計算 Nusselt 數 ===
        self.nu_sum[None] = 0.0
        self._compute_nu_kernel()
        mean_grad = self.nu_sum[None] / self.nx
        H = self.ny
        return float(-H * mean_grad / delta_T)

    @ti.kernel
    def _compute_nu_kernel(self):
        """
        計算中間截面（j=ny//2）的平均溫度梯度

        使用中心差分：∂T/∂y|_{j} ≈ (T[i,j+1] - T[i,j-1]) / 2
        結果累加至 self.nu_sum（再除以 nx 即為平均值）
        """
        j_mid = self.ny // 2
        for i in range(1, self.nx + 1):
            dT_dy = (self.T[i, j_mid + 1] - self.T[i, j_mid - 1]) / 2.0
            ti.atomic_add(self.nu_sum[None], dT_dy)

    def register_with_solver(self):
        """
        將 compute_buoyancy 註冊為 solver 的 force_field_updater

        What: 讓 solver.step() 在每步開頭自動呼叫 compute_buoyancy()
        Why:  solver.set_force_field_updater() 同時設定 force_field_enabled=1
              和呼叫 _refresh_force_enabled()，確保浮力生效

        使用方法:
            thermal.register_with_solver()
            # 之後 solver.step() 每步自動更新浮力
        """
        self.solver.set_force_field_updater(self.compute_buoyancy)

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


@ti.data_oriented
class ThermalBoundaryConditions:
    """
    溫度場邊界條件

    What: 管理 g 分佈函數在壁面的 Dirichlet/Neumann BC
    Why:
      - Dirichlet（固定溫度）: Anti-Bounce-Back scheme
            g_ᾱ(wall) = -g_α(wall) + 2·w_α·T_wall
        物理上等效於壁面溫度固定在 T_wall
      - Neumann（絕熱, ∂T/∂n=0）: Bounce-Back
            g_ᾱ(wall) = g_α(wall)
        物理上等效於無熱通量穿越邊界

    使用：
        tbc = ThermalBoundaryConditions(thermal)
        tbc.add_hot_wall(T_hot=1.0, location='bottom')
        tbc.add_cold_wall(T_cold=0.0, location='top')
        tbc.add_adiabatic_wall('left')
        tbc.apply(g_dst)  # 每步呼叫一次
    """

    def __init__(self, thermal):
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

        # 驗證硬編碼 inv 常數與 solver.inv 一致（防止 D2Q9 定義變更時的 silent bug）
        # NOTE: 各壁面 kernel 因 Taichi 1.7.4 的 ti.static+field 索引 bug 改用硬編碼，
        #       升級 Taichi 後可改回動態索引並移除此 assert
        _expected_inv = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        for _k in range(9):
            assert thermal.inv[_k] == _expected_inv[_k], (
                f"ThermalBoundaryConditions: inv[{_k}] = {thermal.inv[_k]}, "
                f"expected {_expected_inv[_k]}. "
                f"BC kernels use hardcoded inv — update ThermalBoundaryConditions if D2Q9 inv changes."
            )

    def add_hot_wall(self, T_hot: float, location: str):
        """固定高溫壁（Dirichlet，Anti-Bounce-Back）"""
        self._set_wall(location, bc_type=1, T_val=T_hot)

    def add_cold_wall(self, T_cold: float, location: str):
        """固定低溫壁（Dirichlet，Anti-Bounce-Back）"""
        self._set_wall(location, bc_type=1, T_val=T_cold)

    def add_adiabatic_wall(self, location: str):
        """絕熱壁（Neumann，Bounce-Back，零熱通量）"""
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

    def apply(self, g):
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
        底部壁面 BC（壁在 j=0 外，流體層在 j=1）

        Dirichlet（Anti-Bounce-Back）:
            只修正從底壁進入流體的未知方向 k=2(↑), 5(↗), 6(↖)
            g_k(i,1) = -g_{inv[k]}(i,1) + 2·w_k·T_wall

            同時設定 ghost cell j=0 為壁面平衡態，確保下一步串流
            從 ghost cell 帶入的值與壁面溫度一致（不污染 j=1 的平衡狀態）。
            Ghost cell 不是流體格點，不參與碰撞，此設定只影響串流行為。

        Neumann（絕熱, Bounce-Back）:
            g_k(i,1) = g_{inv[k]}(i,1)  for k=2,5,6

        未知方向（從底壁進入流體）: k=2(↑), k=5(↗), k=6(↖)
        inv 對應: inv[2]=4, inv[5]=7, inv[6]=8
        """
        T_wall = self.T_bottom[None]
        bc     = self.bc_bottom[None]
        for i in range(1, self.nx + 1):
            if bc == 1:  # Dirichlet: Anti-Bounce-Back
                # 設定 ghost cell j=0 為壁面平衡態，確保下一步串流正確
                for q in ti.static(range(9)):
                    g[i, 0][q] = self.thermal.w[q] * T_wall
                # 修正流體層 j=1 的未知方向（ABB）
                # TODO: Taichi > 1.7.4 修復後改回動態 inv 索引；見 __init__ 中的 NOTE
                # inv[2]=4, inv[5]=7, inv[6]=8（硬編碼 inv 避免 ti.static 的索引問題）
                g[i, 1][2] = -g[i, 1][4] + 2.0 * self.thermal.w[2] * T_wall
                g[i, 1][5] = -g[i, 1][7] + 2.0 * self.thermal.w[5] * T_wall
                g[i, 1][6] = -g[i, 1][8] + 2.0 * self.thermal.w[6] * T_wall
            else:        # Neumann: Bounce-Back
                # inv[2]=4, inv[5]=7, inv[6]=8
                g[i, 1][2] = g[i, 1][4]
                g[i, 1][5] = g[i, 1][7]
                g[i, 1][6] = g[i, 1][8]

    @ti.kernel
    def _apply_top_bc(self, g: ti.template()):
        """
        頂部壁面 BC（壁在 j=ny+1 外，流體層在 j=ny）

        Dirichlet（Anti-Bounce-Back）:
            只修正從頂壁進入流體的未知方向 k=4(↓), 7(↙), 8(↘)
            g_k(i,ny) = -g_{inv[k]}(i,ny) + 2·w_k·T_wall

            同時設定 ghost cell j=ny+1 為壁面平衡態，確保下一步串流正確。

        Neumann（絕熱, Bounce-Back）:
            g_k(i,ny) = g_{inv[k]}(i,ny)  for k=4,7,8

        未知方向（從頂壁進入流體）: k=4(↓), k=7(↙), k=8(↘)
        inv 對應: inv[4]=2, inv[7]=5, inv[8]=6
        """
        T_wall = self.T_top[None]
        bc     = self.bc_top[None]
        ny     = self.ny
        for i in range(1, self.nx + 1):
            if bc == 1:  # Dirichlet: Anti-Bounce-Back
                # 設定 ghost cell j=ny+1 為壁面平衡態
                for q in ti.static(range(9)):
                    g[i, ny + 1][q] = self.thermal.w[q] * T_wall
                # 修正流體層 j=ny 的未知方向（ABB）
                # TODO: 同底壁，Taichi bug workaround
                # inv[4]=2, inv[7]=5, inv[8]=6（硬編碼 inv 避免 ti.static 的索引問題）
                g[i, ny][4] = -g[i, ny][2] + 2.0 * self.thermal.w[4] * T_wall
                g[i, ny][7] = -g[i, ny][5] + 2.0 * self.thermal.w[7] * T_wall
                g[i, ny][8] = -g[i, ny][6] + 2.0 * self.thermal.w[8] * T_wall
            else:        # Neumann: Bounce-Back
                # inv[4]=2, inv[7]=5, inv[8]=6
                g[i, ny][4] = g[i, ny][2]
                g[i, ny][7] = g[i, ny][5]
                g[i, ny][8] = g[i, ny][6]

    @ti.kernel
    def _apply_left_bc(self, g: ti.template()):
        """
        左壁 BC（壁在 i=0 外，流體層在 i=1）

        Dirichlet（Anti-Bounce-Back）:
            只修正從左壁進入流體的未知方向 k=1(→), 5(↗), 8(↘)
            g_k(1,j) = -g_{inv[k]}(1,j) + 2·w_k·T_wall

            同時設定 ghost cell i=0 為壁面平衡態，確保下一步串流正確。

        Neumann（絕熱, Bounce-Back）:
            g_k(1,j) = g_{inv[k]}(1,j)  for k=1,5,8

        未知方向（從左壁進入流體）: k=1(→), k=5(↗), k=8(↘)
        inv 對應: inv[1]=3, inv[5]=7, inv[8]=6
        """
        T_wall = self.T_left[None]
        bc     = self.bc_left[None]
        for j in range(1, self.ny + 1):
            if bc == 1:  # Dirichlet: Anti-Bounce-Back
                # 設定 ghost cell i=0 為壁面平衡態
                for q in ti.static(range(9)):
                    g[0, j][q] = self.thermal.w[q] * T_wall
                # 修正流體層 i=1 的未知方向（ABB）
                # TODO: 同底壁，Taichi bug workaround
                # inv[1]=3, inv[5]=7, inv[8]=6（硬編碼 inv 避免 ti.static 的索引問題）
                g[1, j][1] = -g[1, j][3] + 2.0 * self.thermal.w[1] * T_wall
                g[1, j][5] = -g[1, j][7] + 2.0 * self.thermal.w[5] * T_wall
                g[1, j][8] = -g[1, j][6] + 2.0 * self.thermal.w[8] * T_wall
            else:        # Neumann: Bounce-Back
                # inv[1]=3, inv[5]=7, inv[8]=6
                g[1, j][1] = g[1, j][3]
                g[1, j][5] = g[1, j][7]
                g[1, j][8] = g[1, j][6]

    @ti.kernel
    def _apply_right_bc(self, g: ti.template()):
        """
        右壁 BC（壁在 i=nx+1 外，流體層在 i=nx）

        Dirichlet（Anti-Bounce-Back）:
            只修正從右壁進入流體的未知方向 k=3(←), 6(↖), 7(↙)
            g_k(nx,j) = -g_{inv[k]}(nx,j) + 2·w_k·T_wall

            同時設定 ghost cell i=nx+1 為壁面平衡態，確保下一步串流正確。

        Neumann（絕熱, Bounce-Back）:
            g_k(nx,j) = g_{inv[k]}(nx,j)  for k=3,6,7

        未知方向（從右壁進入流體）: k=3(←), k=6(↖), k=7(↙)
        inv 對應: inv[3]=1, inv[6]=8, inv[7]=5
        """
        T_wall = self.T_right[None]
        bc     = self.bc_right[None]
        nx     = self.nx
        for j in range(1, self.ny + 1):
            if bc == 1:  # Dirichlet: Anti-Bounce-Back
                # 設定 ghost cell i=nx+1 為壁面平衡態
                for q in ti.static(range(9)):
                    g[nx + 1, j][q] = self.thermal.w[q] * T_wall
                # 修正流體層 i=nx 的未知方向（ABB）
                # TODO: 同底壁，Taichi bug workaround
                # inv[3]=1, inv[6]=8, inv[7]=5（硬編碼 inv 避免 ti.static 的索引問題）
                g[nx, j][3] = -g[nx, j][1] + 2.0 * self.thermal.w[3] * T_wall
                g[nx, j][6] = -g[nx, j][8] + 2.0 * self.thermal.w[6] * T_wall
                g[nx, j][7] = -g[nx, j][5] + 2.0 * self.thermal.w[7] * T_wall
            else:        # Neumann: Bounce-Back
                # inv[3]=1, inv[6]=8, inv[7]=5
                g[nx, j][3] = g[nx, j][1]
                g[nx, j][6] = g[nx, j][8]
                g[nx, j][7] = g[nx, j][5]
