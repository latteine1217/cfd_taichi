"""
NavierStokesSolver — 2D 可壓縮層流 Navier-Stokes FV 求解器
==========================================================

What: 在既有 EulerSolver 的對流通量基礎上，加入黏性應力與熱傳導，
      提供 structured grid 上的第一版 laminar
      compressible Navier-Stokes solver
Why:  從 Euler 升級到有限 Reynolds 數流動，最先需要的是
      - 黏性通量
      - no-slip wall
      - 可解釋的 Re / Pr 參數
      先把這條路徑在可驗證的 structured grid 上做穩，再擴展到 O-grid 與機翼
When: 適用於 laminar、顯式時間推進的驗證案例：
      Cartesian / sheared channel 的 Couette、Poiseuille、低 Re 通道流

限制：
- laminar / constant-eddy-viscosity / experimental Spalart-Allmaras closure
- no-slip wall 目前只支援 structured grid 的 j_min / j_max 邊界
- explicit SSP-RK3（高 Re 或高解析邊界層時 dt 會很小）
"""

import numpy as np
import taichi as ti

from .euler_solver import EulerSolver


@ti.data_oriented
class NavierStokesSolver(EulerSolver):
    """
    2D 可壓縮層流 Navier-Stokes solver（structured-grid 第一版）

    What: 延用 EulerSolver 的 MUSCL + HLLC 對流項，再加上
      中心差分黏性/熱傳通量
    Why:  這樣可以在不破壞現有 Euler 路徑的前提下，
          逐步導入 finite-Re 物理與 turbulence closure 骨架
    When: 目前建議先用 Cartesian / affine curvilinear benchmark 驗證，
          黏性翼型的高 Re 路徑目前只提供 experimental constant eddy viscosity
    """

    solver_family = "fvm"
    equation_set = "navier_stokes"
    regime = "compressible"

    def __init__(
            self,
            ni: int,
            nj: int,
            gamma: float = 1.4,
            cfl: float = 0.2,
            re: float = 100.0,
            pr: float = 0.72,
            u_ref: float = 1.0,
            length_scale: float = 1.0,
            rho_ref: float = 1.0,
    ):
        """
        Args:
            ni, nj:       interior cell counts
            gamma:        比熱比
            cfl:          顯式 CFL，NS 建議比 Euler 更保守
            re:           Reynolds number = rho_ref * u_ref * L / mu
            pr:           Prandtl number = mu * cp / kappa
            u_ref:        參考速度
            length_scale: 特徵長度
            rho_ref:      參考密度
        """
        if re <= 0.0:
            raise ValueError(f"Reynolds number must be positive, got {re}")
        if pr <= 0.0:
            raise ValueError(f"Prandtl number must be positive, got {pr}")
        if u_ref <= 0.0:
            raise ValueError(f"u_ref must be positive, got {u_ref}")
        if length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        if rho_ref <= 0.0:
            raise ValueError(f"rho_ref must be positive, got {rho_ref}")

        super().__init__(ni=ni, nj=nj, gamma=gamma, cfl=cfl)

        cp = gamma / (gamma - 1.0)
        mu = rho_ref * u_ref * length_scale / re
        kappa = mu * cp / pr

        self.re = float(re)
        self.pr = float(pr)
        self.u_ref = float(u_ref)
        self.length_scale = float(length_scale)
        self.rho_ref = float(rho_ref)

        self.mu_field = ti.field(ti.f32, shape=())
        self.kappa_field = ti.field(ti.f32, shape=())
        self.mu_t_field = ti.field(ti.f32, shape=())
        self.eff_mu_field = ti.field(ti.f32, shape=())
        self.eff_kappa_field = ti.field(ti.f32, shape=())
        self.pr_t_field = ti.field(ti.f32, shape=())
        self.dx_field = ti.field(ti.f32, shape=())
        self.dy_field = ti.field(ti.f32, shape=())
        self.body_force = ti.Vector.field(2, ti.f32, shape=())
        self.mu_t_cell = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.mu_eff_cell = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.kappa_eff_cell = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.wall_dist = ti.field(ti.f32, shape=(self.NI, self.NJ))

        self.mu_field[None] = float(mu)
        self.kappa_field[None] = float(kappa)
        self.mu_t_field[None] = 0.0
        self.eff_mu_field[None] = float(mu)
        self.eff_kappa_field[None] = float(kappa)
        self.pr_t_field[None] = 0.9
        self.body_force[None] = ti.Vector([0.0, 0.0])

        self._cp = float(cp)
        self._turbulence_model = "laminar"
        self._eddy_viscosity_ratio = 0.0
        self._sa_nu_tilde_inf_ratio = 3.0
        self._wall_distance_ready = False

        self.nu_tilde = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.nu_tilde0 = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.nu_tilde1 = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.nu_tilde2 = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.R_nu = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.R_nu_tmp = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.grad_nu_tilde = ti.Vector.field(2, ti.f32, shape=(self.NI, self.NJ))
        self.NuFlux_i = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.NuFlux_j = ti.field(ti.f32, shape=(self.NI, self.NJ))
        self.sa_nu_inf_field = ti.field(ti.f32, shape=())
        self.sa_sigma_field = ti.field(ti.f32, shape=())
        self.sa_cb1_field = ti.field(ti.f32, shape=())
        self.sa_cb2_field = ti.field(ti.f32, shape=())
        self.sa_kappa_field = ti.field(ti.f32, shape=())
        self.sa_cw1_field = ti.field(ti.f32, shape=())
        self.sa_cw2_field = ti.field(ti.f32, shape=())
        self.sa_cw3_field = ti.field(ti.f32, shape=())
        self.sa_cv1_field = ti.field(ti.f32, shape=())
        self.sa_ct3_field = ti.field(ti.f32, shape=())
        self.sa_ct4_field = ti.field(ti.f32, shape=())

        sa_sigma = 2.0 / 3.0
        sa_cb1 = 0.1355
        sa_cb2 = 0.622
        sa_kappa = 0.41
        sa_cw2 = 0.3
        sa_cw3 = 2.0
        sa_cv1 = 7.1
        sa_ct3 = 1.2
        sa_ct4 = 0.5
        sa_cw1 = sa_cb1 / (sa_kappa * sa_kappa) + (1.0 + sa_cb2) / sa_sigma
        self.sa_nu_inf_field[None] = 3.0 * mu / rho_ref
        self.sa_sigma_field[None] = sa_sigma
        self.sa_cb1_field[None] = sa_cb1
        self.sa_cb2_field[None] = sa_cb2
        self.sa_kappa_field[None] = sa_kappa
        self.sa_cw1_field[None] = sa_cw1
        self.sa_cw2_field[None] = sa_cw2
        self.sa_cw3_field[None] = sa_cw3
        self.sa_cv1_field[None] = sa_cv1
        self.sa_ct3_field[None] = sa_ct3
        self.sa_ct4_field[None] = sa_ct4

        self.VFlux_i = ti.Vector.field(4, ti.f32, shape=(self.NI, self.NJ))
        self.VFlux_j = ti.Vector.field(4, ti.f32, shape=(self.NI, self.NJ))
        self.grad_u = ti.Vector.field(2, ti.f32, shape=(self.NI, self.NJ))
        self.grad_v = ti.Vector.field(2, ti.f32, shape=(self.NI, self.NJ))
        self.grad_T = ti.Vector.field(2, ti.f32, shape=(self.NI, self.NJ))

        self._cartesian_ready = False
        self._metric_mode = "none"

        self._no_slip_j_min = False
        self._no_slip_j_max = False
        self._j_min_adiabatic = True
        self._j_max_adiabatic = True
        self._j_min_u_wall = 0.0
        self._j_min_v_wall = 0.0
        self._j_max_u_wall = 0.0
        self._j_max_v_wall = 0.0
        self._j_min_t_wall = 1.0
        self._j_max_t_wall = 1.0

        self._fill_uniform_transport_fields(float(self.mu_field[None]), float(self.kappa_field[None]), 0.0)
        self._fill_nu_tilde(float(self.sa_nu_inf_field[None]))

    def set_cartesian_grid(self, dx: float, dy: float):
        """
        設定直角網格，並記錄 NS 黏性項所需的 dx / dy
        """
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")
        self.dx_field[None] = float(dx)
        self.dy_field[None] = float(dy)
        self._cartesian_ready = True
        self._metric_mode = "cartesian"
        super().set_cartesian_grid(dx=dx, dy=dy)
        self._update_wall_distance_if_ready()

    def set_curvilinear_grid(self, x_node: np.ndarray, y_node: np.ndarray):
        """
        設定 curvilinear structured grid

        What: 使用 EulerSolver 的幾何建構，並啟用 Green-Gauss curvilinear
              viscous flux 路徑
        Why:  黏性應力需要 physical-space gradient，不能把 Cartesian 差分
              直接硬套到 O-grid / skewed grid
        """
        self._cartesian_ready = False
        self._metric_mode = "curvilinear"
        super().set_curvilinear_grid(x_node, y_node)
        self._update_wall_distance_if_ready()

    def set_body_force(self, fx: float = 0.0, fy: float = 0.0):
        """
        設定均勻體積力

        What: 對所有 cell 加入固定 body force source term
        Why:  Poiseuille / channel flow 需要壓力梯度或等效體積力驅動
        When: 初始化後、模擬前呼叫
        """
        self.body_force[None] = ti.Vector([float(fx), float(fy)])

    def set_no_slip_wall(
            self,
            location: str,
            u_wall: float = 0.0,
            v_wall: float = 0.0,
            temperature: float | None = None):
        """
        設定 structured-grid j_min / j_max no-slip wall

        What: 牆面速度以 ghost mirror impose，溫度可選 adiabatic 或 isothermal
        Why:  finite-Re 流動的邊界層與壁面剪應力都依賴 no-slip wall
        When: Cartesian channel、sheared channel、body-fitted curvilinear j-boundaries

        Args:
            location: 'bottom'/'top' 或 'j_min'/'j_max'
            u_wall, v_wall: 牆面速度
            temperature: None -> adiabatic；否則視為 wall temperature
        """
        aliases = {
            "bottom": "j_min",
            "top": "j_max",
            "j_min": "j_min",
            "j_max": "j_max",
            "airfoil": "j_min",
            "outer": "j_max",
        }
        if location not in aliases:
            raise NotImplementedError(
                "NavierStokesSolver implements no-slip walls on "
                "'bottom'/'top' or 'j_min'/'j_max' only."
            )
        edge = aliases[location]

        adiabatic = temperature is None
        t_wall = 1.0 if temperature is None else float(temperature)

        if edge == "j_min":
            self._no_slip_j_min = True
            self._wall_j_min = False
            self._j_min_u_wall = float(u_wall)
            self._j_min_v_wall = float(v_wall)
            self._j_min_adiabatic = adiabatic
            self._j_min_t_wall = t_wall
        else:
            self._no_slip_j_max = True
            self._wall_j_max = False
            self._far_field_j_max = False
            self._j_max_u_wall = float(u_wall)
            self._j_max_v_wall = float(v_wall)
            self._j_max_adiabatic = adiabatic
            self._j_max_t_wall = t_wall
        self._update_wall_distance_if_ready()

    @ti.kernel
    def _fill_uniform_transport_fields(
            self,
            mu_eff: ti.f32,
            kappa_eff: ti.f32,
            mu_t: ti.f32):
        for i, j in ti.ndrange(self.NI, self.NJ):
            self.mu_t_cell[i, j] = mu_t
            self.mu_eff_cell[i, j] = mu_eff
            self.kappa_eff_cell[i, j] = kappa_eff

    @ti.kernel
    def _fill_nu_tilde(self, value: ti.f32):
        for i, j in ti.ndrange(self.NI, self.NJ):
            self.nu_tilde[i, j] = value

    @ti.kernel
    def _copy_scalar_field(self, dst: ti.template(), src: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            dst[i, j] = src[i, j]

    @ti.kernel
    def _apply_scalar_periodic_i(self, field: ti.template()):
        for g, j in ti.ndrange(self.NG, self.NJ):
            field[g, j] = field[self.ni + g, j]
            field[self.NG + self.ni + g, j] = field[self.NG + g, j]

    @ti.kernel
    def _apply_scalar_neumann_i(self, field: ti.template()):
        ri = self.NG + self.ni - 1
        for g, j in ti.ndrange(self.NG, self.NJ):
            field[g, j] = field[self.NG, j]
            field[self.NG + self.ni + g, j] = field[ri, j]

    @ti.kernel
    def _apply_scalar_periodic_j_min(self, field: ti.template()):
        for i, g in ti.ndrange(self.NI, self.NG):
            field[i, g] = field[i, self.nj + g]

    @ti.kernel
    def _apply_scalar_periodic_j_max(self, field: ti.template()):
        for i, g in ti.ndrange(self.NI, self.NG):
            field[i, self.NG + self.nj + g] = field[i, self.NG + g]

    @ti.kernel
    def _apply_scalar_neumann_j_min(self, field: ti.template()):
        for i, g in ti.ndrange(self.NI, self.NG):
            field[i, g] = field[i, self.NG]

    @ti.kernel
    def _apply_scalar_neumann_j_max(self, field: ti.template()):
        tj = self.NG + self.nj - 1
        for i, g in ti.ndrange(self.NI, self.NG):
            field[i, self.NG + self.nj + g] = field[i, tj]

    @ti.kernel
    def _apply_scalar_dirichlet_j_min_zero(self, field: ti.template()):
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int = self.NG + g
            j_ghost = self.NG - 1 - g
            field[i, j_ghost] = -field[i, j_int]

    @ti.kernel
    def _apply_scalar_dirichlet_j_max_zero(self, field: ti.template()):
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int = self.NG + self.nj - 1 - g
            j_ghost = self.NG + self.nj + g
            field[i, j_ghost] = -field[i, j_int]

    @ti.kernel
    def _apply_scalar_far_field_j_max(self, field: ti.template(), value: ti.f32):
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int = self.NG + self.nj - 1 - g
            j_ghost = self.NG + self.nj + g
            field[i, j_ghost] = 2.0 * value - field[i, j_int]

    @ti.kernel
    def _update_transport_from_sa(self):
        mu_lam = self.mu_field[None]
        kappa_lam = self.kappa_field[None]
        pr_t = self.pr_t_field[None]
        cv1 = self.sa_cv1_field[None]

        for i, j in ti.ndrange(self.NI, self.NJ):
            rho = ti.max(self.W[i, j][0], 1e-8)
            nu_lam = mu_lam / rho
            nu_tilde = ti.max(self.nu_tilde[i, j], 0.0)
            chi = nu_tilde / ti.max(nu_lam, 1e-12)
            chi3 = chi * chi * chi
            fv1 = chi3 / (chi3 + cv1 * cv1 * cv1 + 1e-12)
            nu_t = nu_tilde * fv1
            mu_t = rho * nu_t
            self.mu_t_cell[i, j] = mu_t
            self.mu_eff_cell[i, j] = mu_lam + mu_t
            self.kappa_eff_cell[i, j] = kappa_lam + mu_t * self._cp / ti.max(pr_t, 1e-12)

    def _refresh_transport_coefficients(self):
        """
        同步 laminar / turbulent / effective transport coefficients

        What: 根據目前 closure 設定刷新有效黏度與熱傳導係數
        Why:  黏性通量與 diffusion CFL 都必須使用同一組 effective transport，
              否則 turbulence closure 只會改一半
        """
        mu_lam = float(self.mu_field[None])
        mu_t = float(self.mu_t_field[None])
        kappa_lam = float(self.kappa_field[None])
        pr_t = float(self.pr_t_field[None])

        kappa_t = mu_t * self._cp / max(pr_t, 1e-12)
        self.eff_mu_field[None] = mu_lam + mu_t
        self.eff_kappa_field[None] = kappa_lam + kappa_t
        self._fill_uniform_transport_fields(
            float(self.eff_mu_field[None]),
            float(self.eff_kappa_field[None]),
            float(mu_t),
        )

    def _update_wall_distance_if_ready(self):
        """
        依結構網格與 no-slip 邊界更新 wall distance

        What: 以 j 向 cell thickness 累積近似最近壁面距離
        Why:  SA source term 需要壁距 d；在 body-fitted structured grid 上，
              沿 j 線的累積距離是務實且可驗證的第一版近似
        """
        if not self._grid_ready:
            return

        NG = self.NG
        vol = self.vol.to_numpy()
        s_j = self.S_j.to_numpy()
        wall_dist = np.full((self.NI, self.NJ), 1.0e6, dtype=np.float32)

        vol_int = vol[NG:NG + self.ni, NG:NG + self.nj]

        if self._no_slip_j_min:
            dist_bottom = np.zeros((self.ni, self.nj), dtype=np.float32)
            dn_prev = None
            for jj in range(self.nj):
                j = NG + jj
                ds_b = np.linalg.norm(s_j[NG:NG + self.ni, j - 1, :], axis=1)
                ds_t = np.linalg.norm(s_j[NG:NG + self.ni, j, :], axis=1)
                dn = vol_int[:, jj] / np.maximum(0.5 * (ds_b + ds_t), 1e-12)
                if jj == 0:
                    dist_bottom[:, jj] = 0.5 * dn
                else:
                    dist_bottom[:, jj] = dist_bottom[:, jj - 1] + 0.5 * (dn_prev + dn)
                dn_prev = dn
            wall_dist[NG:NG + self.ni, NG:NG + self.nj] = np.minimum(
                wall_dist[NG:NG + self.ni, NG:NG + self.nj], dist_bottom)

        if self._no_slip_j_max:
            dist_top = np.zeros((self.ni, self.nj), dtype=np.float32)
            dn_prev = None
            for offset, jj in enumerate(range(self.nj - 1, -1, -1)):
                j = NG + jj
                ds_b = np.linalg.norm(s_j[NG:NG + self.ni, j - 1, :], axis=1)
                ds_t = np.linalg.norm(s_j[NG:NG + self.ni, j, :], axis=1)
                dn = vol_int[:, jj] / np.maximum(0.5 * (ds_b + ds_t), 1e-12)
                if offset == 0:
                    dist_top[:, jj] = 0.5 * dn
                else:
                    dist_top[:, jj] = dist_top[:, jj + 1] + 0.5 * (dn_prev + dn)
                dn_prev = dn
            wall_dist[NG:NG + self.ni, NG:NG + self.nj] = np.minimum(
                wall_dist[NG:NG + self.ni, NG:NG + self.nj], dist_top)

        if not (self._no_slip_j_min or self._no_slip_j_max):
            wall_dist[NG:NG + self.ni, NG:NG + self.nj] = 1.0e6

        wall_dist[:NG, :] = wall_dist[NG:NG + 1, :]
        wall_dist[NG + self.ni:, :] = wall_dist[NG + self.ni - 1:NG + self.ni, :]
        wall_dist[:, :NG] = wall_dist[:, NG:NG + 1]
        wall_dist[:, NG + self.nj:] = wall_dist[:, NG + self.nj - 1:NG + self.nj]
        self.wall_dist.from_numpy(wall_dist)
        self._wall_distance_ready = True

    def _set_sa_summary_transport(self):
        """
        用 freestream SA 狀態更新 summary transport
        """
        nu_lam = float(self.mu_field[None]) / max(self.rho_ref, 1e-12)
        nu_tilde_inf = float(self.sa_nu_inf_field[None])
        cv1 = float(self.sa_cv1_field[None])
        chi = nu_tilde_inf / max(nu_lam, 1e-12)
        chi3 = chi ** 3
        fv1 = chi3 / max(chi3 + cv1 ** 3, 1e-12)
        mu_t_inf = self.rho_ref * nu_tilde_inf * fv1
        self.mu_t_field[None] = float(mu_t_inf)
        self.eff_mu_field[None] = float(self.mu_field[None] + mu_t_inf)
        self.eff_kappa_field[None] = float(
            self.kappa_field[None] + mu_t_inf * self._cp / max(float(self.pr_t_field[None]), 1e-12)
        )

    def set_turbulence_model(
            self,
            model: str = "laminar",
            eddy_viscosity_ratio: float = 0.0,
            turbulent_prandtl: float = 0.9,
            sa_nu_tilde_inf_ratio: float = 3.0):
        """
        設定 turbulence closure

        What: 提供 laminar 與 constant-eddy-viscosity 兩種 transport closure
        Why:  高 Re 路徑至少需要可插拔的 turbulence contract，之後才能安全擴展到
              SA / k-omega 等一方程或兩方程模型
        When: solver 初始化後、時間推進前呼叫
        """
        if turbulent_prandtl <= 0.0:
            raise ValueError(f"turbulent_prandtl must be positive, got {turbulent_prandtl}")
        if eddy_viscosity_ratio < 0.0:
            raise ValueError(f"eddy_viscosity_ratio must be >= 0, got {eddy_viscosity_ratio}")
        if sa_nu_tilde_inf_ratio <= 0.0:
            raise ValueError(
                f"sa_nu_tilde_inf_ratio must be positive, got {sa_nu_tilde_inf_ratio}")

        if model == "laminar":
            self._turbulence_model = "laminar"
            self._eddy_viscosity_ratio = 0.0
            self.pr_t_field[None] = float(turbulent_prandtl)
            self.mu_t_field[None] = 0.0
            self.sa_nu_inf_field[None] = float(sa_nu_tilde_inf_ratio) * (
                float(self.mu_field[None]) / max(self.rho_ref, 1e-12))
        elif model == "constant_eddy_viscosity":
            mu_lam = float(self.mu_field[None])
            self._turbulence_model = model
            self._eddy_viscosity_ratio = float(eddy_viscosity_ratio)
            self.pr_t_field[None] = float(turbulent_prandtl)
            self.mu_t_field[None] = float(eddy_viscosity_ratio) * mu_lam
            self.sa_nu_inf_field[None] = float(sa_nu_tilde_inf_ratio) * (
                float(self.mu_field[None]) / max(self.rho_ref, 1e-12))
        elif model == "spalart_allmaras":
            self._turbulence_model = model
            self._eddy_viscosity_ratio = 0.0
            self.pr_t_field[None] = float(turbulent_prandtl)
            self.mu_t_field[None] = 0.0
            self._sa_nu_tilde_inf_ratio = float(sa_nu_tilde_inf_ratio)
            self.sa_nu_inf_field[None] = float(sa_nu_tilde_inf_ratio) * (
                float(self.mu_field[None]) / max(self.rho_ref, 1e-12))
            self._fill_nu_tilde(float(self.sa_nu_inf_field[None]))
        else:
            raise ValueError(
                "turbulence model must be 'laminar', 'constant_eddy_viscosity', or "
                "'spalart_allmaras', "
                f"got {model}"
            )

        if model == "spalart_allmaras":
            self._update_wall_distance_if_ready()
            self._set_sa_summary_transport()
            self._update_turbulence_state()
        else:
            self._refresh_transport_coefficients()

    def get_transport_coefficients(self) -> dict:
        """
        回傳目前使用的 transport coefficients
        """
        return {
            "mu": float(self.mu_field[None]),
            "kappa": float(self.kappa_field[None]),
            "mu_t": float(self.mu_t_field[None]),
            "mu_eff": float(self.eff_mu_field[None]),
            "kappa_eff": float(self.eff_kappa_field[None]),
            "re": self.re,
            "pr": self.pr,
            "pr_t": float(self.pr_t_field[None]),
            "u_ref": self.u_ref,
            "length_scale": self.length_scale,
            "turbulence_model": self._turbulence_model,
            "eddy_viscosity_ratio": self._eddy_viscosity_ratio,
            "sa_nu_tilde_inf_ratio": self._sa_nu_tilde_inf_ratio,
        }

    def set_sa_nu_tilde_from_numpy(self, nu_tilde_np: np.ndarray):
        """
        從 numpy 陣列設定 SA working variable

        What: 將 interior `nu_tilde` 直接上傳到 solver，並同步 ghost 與有效傳輸係數
        Why:  高 Re case 常需要比 uniform freestream 更平順的 SA 初始場，
              例如 wall-ramp 或 boundary-layer-like initialization
        When: `set_turbulence_model(model="spalart_allmaras")` 後、進入時間推進前
        """
        if self._turbulence_model != "spalart_allmaras":
            raise RuntimeError(
                "set_sa_nu_tilde_from_numpy() requires turbulence_model='spalart_allmaras'"
            )
        if nu_tilde_np.shape != (self.ni, self.nj):
            raise ValueError(
                f"Expected nu_tilde shape ({self.ni}, {self.nj}), got {nu_tilde_np.shape}"
            )

        NG = self.NG
        nu_full = np.zeros((self.NI, self.NJ), dtype=np.float32)
        nu_full[NG:NG + self.ni, NG:NG + self.nj] = np.maximum(nu_tilde_np, 0.0).astype(np.float32)
        self.nu_tilde.from_numpy(nu_full)
        self._update_turbulence_state()

    def update_viscous_fluxes(self):
        """
        以目前狀態更新黏性通量場

        What: 根據目前的 W、幾何與 transport coefficients 重算 face viscous flux
        Why:  後處理壁面剪應力、黏性阻力與熱通量時需要最新 face traction，
              不應要求案例直接手動調 private kernels
        When: 一個 step 完成後、計算 wall force / heat flux 前呼叫
        """
        if self._metric_mode == "cartesian":
            self._compute_viscous_fluxes_cartesian()
        elif self._metric_mode == "curvilinear":
            self._compute_cell_gradients_green_gauss()
            self._compute_viscous_fluxes_curvilinear()
        else:
            raise RuntimeError(
                "NavierStokesSolver requires set_cartesian_grid() or "
                "set_curvilinear_grid() before updating viscous fluxes."
            )

    def _update_turbulence_state(self):
        """
        依目前 closure 刷新 turbulence auxiliary fields

        What: 更新 scalar ghost、wall distance 與 cell-wise effective transport
        Why:  high-Re closure 若不跟著 stage state 同步，黏性通量與 source term 會失配
        """
        if self._turbulence_model == "laminar":
            self._refresh_transport_coefficients()
            return

        if self._turbulence_model == "constant_eddy_viscosity":
            self._refresh_transport_coefficients()
            return

        if self._turbulence_model != "spalart_allmaras":
            raise RuntimeError(f"Unknown turbulence model: {self._turbulence_model}")

        if not self._wall_distance_ready:
            self._update_wall_distance_if_ready()
        if not self._wall_distance_ready:
            raise RuntimeError("Spalart-Allmaras requires wall distance; call set_*_grid() and set_no_slip_wall() first.")

        if self._periodic_i:
            self._apply_scalar_periodic_i(self.nu_tilde)
        else:
            self._apply_scalar_neumann_i(self.nu_tilde)

        if self._no_slip_j_min:
            self._apply_scalar_dirichlet_j_min_zero(self.nu_tilde)
        elif self._periodic_j:
            self._apply_scalar_periodic_j_min(self.nu_tilde)
        else:
            self._apply_scalar_neumann_j_min(self.nu_tilde)

        if self._no_slip_j_max:
            self._apply_scalar_dirichlet_j_max_zero(self.nu_tilde)
        elif self._far_field_j_max:
            self._apply_scalar_far_field_j_max(self.nu_tilde, self.sa_nu_inf_field[None])
        elif self._periodic_j:
            self._apply_scalar_periodic_j_max(self.nu_tilde)
        else:
            self._apply_scalar_neumann_j_max(self.nu_tilde)

        self._update_transport_from_sa()

    def refresh_ghost_cells(self):
        """
        依目前邊界條件重建 ghost cells

        What: 重新施加目前設定的 boundary conditions 到 ghost layers
        Why:  初始化後若要立刻做力係數、wall traction 或 state 輸出，
              需要先讓 ghost state 與 wall/far-field 契約一致
        When: init_uniform / init_from_primitive_numpy 後、進入時間推進前
        """
        self._update_ghost()

    def step(self):
        """
        執行一個 Navier-Stokes RK3 時間步

        Why: 黏性項目前只在 Cartesian metric 下實作，因此先在入口處 fail fast
        """
        if self._metric_mode == "none":
            raise RuntimeError(
                "NavierStokesSolver requires set_cartesian_grid() or "
                "set_curvilinear_grid() before stepping."
            )
        if self._turbulence_model != "spalart_allmaras":
            return super().step()

        self._prepare_local_pseudo_controls_for_step()
        self._compute_dt()
        ti.sync()
        dt = float(self.dt_field[None])

        self._save_U0()
        self._copy_scalar_field(self.nu_tilde0, self.nu_tilde)

        self._eval_residual()
        self._eval_sa_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage1_local()
            self._rk_stage1_scalar_local(self.nu_tilde0, self.R_nu, self.nu_tilde1)
        else:
            self._rk_stage1(dt)
            self._rk_stage1_scalar(dt, self.nu_tilde0, self.R_nu, self.nu_tilde1)
        self._assert_positive_state(self.U1, "RK stage 1")
        self._set_state(self.U1)
        self._set_scalar_state(self.nu_tilde1)

        self._eval_residual()
        self._eval_sa_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage2_local()
            self._rk_stage2_scalar_local(self.nu_tilde0, self.nu_tilde1, self.R_nu, self.nu_tilde2)
        else:
            self._rk_stage2(dt)
            self._rk_stage2_scalar(dt, self.nu_tilde0, self.nu_tilde1, self.R_nu, self.nu_tilde2)
        self._assert_positive_state(self.U2, "RK stage 2")
        self._set_state(self.U2)
        self._set_scalar_state(self.nu_tilde2)

        self._eval_residual()
        self._eval_sa_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage3_local()
            self._rk_stage3_scalar_local(self.nu_tilde0, self.nu_tilde2, self.R_nu, self.nu_tilde)
        else:
            self._rk_stage3(dt)
            self._rk_stage3_scalar(dt, self.nu_tilde0, self.nu_tilde2, self.R_nu, self.nu_tilde)
        self._assert_positive_state(self.U, "RK stage 3")
        self._update_primitive()
        self._update_ghost()
        if self._time_marching_mode == "local_pseudo":
            residual = float(self._compute_residual_norm())
            self._record_local_pseudo_residual(residual)
            self.pseudo_iter_field[None] += 1
        return dt

    def _eval_residual(self):
        """
        對流 + 黏性 + body force 的總殘差
        """
        self._reconstruct_faces()
        self._compute_fluxes()
        if self._metric_mode == "cartesian":
            self._compute_viscous_fluxes_cartesian()
        elif self._metric_mode == "curvilinear":
            self._compute_cell_gradients_green_gauss()
            self._compute_viscous_fluxes_curvilinear()
        else:
            raise RuntimeError(f"Unknown metric mode: {self._metric_mode}")
        self._accumulate_total_residual()
        if self._time_marching_mode == "local_pseudo":
            self._smooth_conserved_residual()

    def _update_ghost(self):
        """
        NS 版 ghost dispatcher

        Why: Euler 的 slip wall 不足以描述 finite-Re 壁面；需要 no-slip + thermal BC。
              同時保留 nozzle i_min/i_max characteristic BC，讓內流案例可用 p0/pb 控制。
        """
        if self._periodic_i:
            self._apply_periodic_i_bc()
        else:
            if self._nozzle_inlet_i_min:
                self._apply_nozzle_inlet_i_min(
                    float(self._nozzle_p0),
                    float(self._nozzle_t0),
                    float(self._nozzle_mach_in),
                    float(self._nozzle_inlet_relax),
                )
            else:
                self._apply_neumann_i_min_bc()

            if self._nozzle_outlet_i_max:
                self._apply_nozzle_outlet_i_max(
                    float(self._nozzle_back_pressure),
                    float(self._nozzle_outlet_relax),
                )
            else:
                self._apply_neumann_i_max_bc()

        if self._no_slip_j_min:
            self._apply_no_slip_j_min(
                float(self._j_min_u_wall),
                float(self._j_min_v_wall),
                float(self._j_min_t_wall),
                1 if self._j_min_adiabatic else 0,
            )
        elif self._wall_j_min:
            self._apply_slip_wall_j_min()
        elif self._periodic_j:
            self._apply_periodic_j_min_bc()
        else:
            self._apply_neumann_j_min_bc()

        if self._no_slip_j_max:
            self._apply_no_slip_j_max(
                float(self._j_max_u_wall),
                float(self._j_max_v_wall),
                float(self._j_max_t_wall),
                1 if self._j_max_adiabatic else 0,
            )
        elif self._wall_j_max:
            self._apply_slip_wall_j_max()
        elif self._far_field_j_max:
            self._apply_far_field_j_max(
                float(self._rho_inf), float(self._u_inf),
                float(self._v_inf), float(self._p_inf))
        elif self._periodic_j:
            self._apply_periodic_j_max_bc()
        else:
            self._apply_neumann_j_max_bc()

        self._update_turbulence_state()

    def _set_scalar_state(self, src):
        """
        更新 turbulence scalar state 並同步 ghost / effective transport
        """
        self._copy_scalar_field(self.nu_tilde, src)
        self._update_turbulence_state()

    def _eval_sa_residual(self):
        """
        計算 Spalart-Allmaras scalar residual

        What: 使用一方程 advection-diffusion-source 更新 nu_tilde
        Why:  這是 high-Re path 的最小可插拔 turbulence closure
        """
        if self._turbulence_model != "spalart_allmaras":
            return

        if self._metric_mode == "cartesian":
            self._compute_nu_tilde_gradients_cartesian()
            self._compute_sa_fluxes_cartesian()
        elif self._metric_mode == "curvilinear":
            self._compute_nu_tilde_gradients_curvilinear()
            self._compute_sa_fluxes_curvilinear()
        else:
            raise RuntimeError(f"Unknown metric mode: {self._metric_mode}")
        self._accumulate_sa_residual()
        if self._time_marching_mode == "local_pseudo":
            self._smooth_sa_residual()

    def _smooth_sa_residual(self):
        """
        對 SA scalar residual 做顯式平滑

        What: 與主方程相同的 5-point Jacobi-like filter
        Why:  若只平滑守恆量 residual，closure residual 仍可能維持高頻 stiffness
        """
        passes = int(self.residual_smoothing_passes_field[None])
        eps = float(self.residual_smoothing_eps_field[None])
        if passes <= 0 or eps <= 0.0:
            return
        for _ in range(passes):
            self._sa_residual_smoothing_pass(eps)
            self._copy_smoothed_sa_residual()

    @ti.kernel
    def _apply_no_slip_j_min(
            self,
            u_wall: ti.f32,
            v_wall: ti.f32,
            t_wall: ti.f32,
            adiabatic: ti.i32):
        """
        bottom no-slip wall
        """
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int = self.NG + g
            j_ghost = self.NG - 1 - g

            rho = self.W[i, j_int][0]
            u_int = self.W[i, j_int][1]
            v_int = self.W[i, j_int][2]
            p_int = self.W[i, j_int][3]
            t_int = p_int / ti.max(rho, 1e-8)

            u_g = 2.0 * u_wall - u_int
            v_g = 2.0 * v_wall - v_int
            t_g = ti.select(adiabatic == 1, t_int, 2.0 * t_wall - t_int)
            t_g = ti.max(t_g, 1e-6)
            p_g = ti.max(rho * t_g, 1e-6)

            self.W[i, j_ghost] = ti.Vector([rho, u_g, v_g, p_g])
            self.U[i, j_ghost] = self._prim2cons(self.W[i, j_ghost])

    @ti.kernel
    def _apply_no_slip_j_max(
            self,
            u_wall: ti.f32,
            v_wall: ti.f32,
            t_wall: ti.f32,
            adiabatic: ti.i32):
        """
        top no-slip wall
        """
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int = self.NG + self.nj - 1 - g
            j_ghost = self.NG + self.nj + g

            rho = self.W[i, j_int][0]
            u_int = self.W[i, j_int][1]
            v_int = self.W[i, j_int][2]
            p_int = self.W[i, j_int][3]
            t_int = p_int / ti.max(rho, 1e-8)

            u_g = 2.0 * u_wall - u_int
            v_g = 2.0 * v_wall - v_int
            t_g = ti.select(adiabatic == 1, t_int, 2.0 * t_wall - t_int)
            t_g = ti.max(t_g, 1e-6)
            p_g = ti.max(rho * t_g, 1e-6)

            self.W[i, j_ghost] = ti.Vector([rho, u_g, v_g, p_g])
            self.U[i, j_ghost] = self._prim2cons(self.W[i, j_ghost])

    @ti.kernel
    def _compute_nu_tilde_gradients_cartesian(self):
        dx = self.dx_field[None]
        dy = self.dy_field[None]

        for i, j in ti.ndrange((1, self.NI - 1), (1, self.NJ - 1)):
            dnu_dx = (self.nu_tilde[i + 1, j] - self.nu_tilde[i - 1, j]) / (2.0 * dx)
            dnu_dy = (self.nu_tilde[i, j + 1] - self.nu_tilde[i, j - 1]) / (2.0 * dy)
            self.grad_nu_tilde[i, j] = ti.Vector([dnu_dx, dnu_dy])

    @ti.kernel
    def _compute_nu_tilde_gradients_curvilinear(self):
        for i, j in ti.ndrange((1, self.NI - 1), (1, self.NJ - 1)):
            nu_c = self.nu_tilde[i, j]
            nu_e = 0.5 * (nu_c + self.nu_tilde[i + 1, j])
            nu_w = 0.5 * (self.nu_tilde[i - 1, j] + nu_c)
            nu_n = 0.5 * (nu_c + self.nu_tilde[i, j + 1])
            nu_s = 0.5 * (self.nu_tilde[i, j - 1] + nu_c)

            grad_num = (
                nu_e * self.S_i[i, j]
                - nu_w * self.S_i[i - 1, j]
                + nu_n * self.S_j[i, j]
                - nu_s * self.S_j[i, j - 1]
            )
            self.grad_nu_tilde[i, j] = grad_num / ti.max(self.vol[i, j], 1e-12)

    @ti.kernel
    def _compute_sa_fluxes_cartesian(self):
        sigma = self.sa_sigma_field[None]
        dx = self.dx_field[None]
        dy = self.dy_field[None]
        mu_lam = self.mu_field[None]

        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho_l = ti.max(self.W[i, j][0], 1e-8)
            rho_r = ti.max(self.W[i + 1, j][0], 1e-8)
            nu_lam_l = mu_lam / rho_l
            nu_lam_r = mu_lam / rho_r
            phi_l = ti.max(self.nu_tilde[i, j], 0.0)
            phi_r = ti.max(self.nu_tilde[i + 1, j], 0.0)
            u_face = 0.5 * (self.W[i, j][1] + self.W[i + 1, j][1])
            adv_speed = u_face
            phi_up = ti.select(adv_speed >= 0.0, phi_l, phi_r)
            gamma_face = 0.5 * ((nu_lam_l + phi_l) + (nu_lam_r + phi_r)) / sigma
            dphi_dx = (phi_r - phi_l) / dx
            self.NuFlux_i[i, j] = phi_up * adv_speed * dy - gamma_face * dphi_dx * dy

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            rho_b = ti.max(self.W[i, j][0], 1e-8)
            rho_t = ti.max(self.W[i, j + 1][0], 1e-8)
            nu_lam_b = mu_lam / rho_b
            nu_lam_t = mu_lam / rho_t
            phi_b = ti.max(self.nu_tilde[i, j], 0.0)
            phi_t = ti.max(self.nu_tilde[i, j + 1], 0.0)
            v_face = 0.5 * (self.W[i, j][2] + self.W[i, j + 1][2])
            adv_speed = v_face
            phi_up = ti.select(adv_speed >= 0.0, phi_b, phi_t)
            gamma_face = 0.5 * ((nu_lam_b + phi_b) + (nu_lam_t + phi_t)) / sigma
            dphi_dy = (phi_t - phi_b) / dy
            self.NuFlux_j[i, j] = phi_up * adv_speed * dx - gamma_face * dphi_dy * dx

    @ti.kernel
    def _compute_sa_fluxes_curvilinear(self):
        sigma = self.sa_sigma_field[None]
        mu_lam = self.mu_field[None]

        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho_l = ti.max(self.W[i, j][0], 1e-8)
            rho_r = ti.max(self.W[i + 1, j][0], 1e-8)
            nu_lam_l = mu_lam / rho_l
            nu_lam_r = mu_lam / rho_r
            phi_l = ti.max(self.nu_tilde[i, j], 0.0)
            phi_r = ti.max(self.nu_tilde[i + 1, j], 0.0)
            u_face = 0.5 * (self.W[i, j][1] + self.W[i + 1, j][1])
            v_face = 0.5 * (self.W[i, j][2] + self.W[i + 1, j][2])
            s = self.S_i[i, j]
            adv_speed = u_face * s[0] + v_face * s[1]
            phi_up = ti.select(adv_speed >= 0.0, phi_l, phi_r)
            grad_face = 0.5 * (self.grad_nu_tilde[i, j] + self.grad_nu_tilde[i + 1, j])
            gamma_face = 0.5 * ((nu_lam_l + phi_l) + (nu_lam_r + phi_r)) / sigma
            diff = gamma_face * (grad_face[0] * s[0] + grad_face[1] * s[1])
            self.NuFlux_i[i, j] = phi_up * adv_speed - diff

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            rho_b = ti.max(self.W[i, j][0], 1e-8)
            rho_t = ti.max(self.W[i, j + 1][0], 1e-8)
            nu_lam_b = mu_lam / rho_b
            nu_lam_t = mu_lam / rho_t
            phi_b = ti.max(self.nu_tilde[i, j], 0.0)
            phi_t = ti.max(self.nu_tilde[i, j + 1], 0.0)
            u_face = 0.5 * (self.W[i, j][1] + self.W[i, j + 1][1])
            v_face = 0.5 * (self.W[i, j][2] + self.W[i, j + 1][2])
            s = self.S_j[i, j]
            adv_speed = u_face * s[0] + v_face * s[1]
            phi_up = ti.select(adv_speed >= 0.0, phi_b, phi_t)
            grad_face = 0.5 * (self.grad_nu_tilde[i, j] + self.grad_nu_tilde[i, j + 1])
            gamma_face = 0.5 * ((nu_lam_b + phi_b) + (nu_lam_t + phi_t)) / sigma
            diff = gamma_face * (grad_face[0] * s[0] + grad_face[1] * s[1])
            self.NuFlux_j[i, j] = phi_up * adv_speed - diff

    @ti.kernel
    def _accumulate_sa_residual(self):
        cb1 = self.sa_cb1_field[None]
        cb2 = self.sa_cb2_field[None]
        kappa = self.sa_kappa_field[None]
        cw1 = self.sa_cw1_field[None]
        cw2 = self.sa_cw2_field[None]
        cw3 = self.sa_cw3_field[None]
        cv1 = self.sa_cv1_field[None]
        ct3 = self.sa_ct3_field[None]
        ct4 = self.sa_ct4_field[None]
        mu_lam = self.mu_field[None]
        sigma = self.sa_sigma_field[None]

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho = ti.max(self.W[i, j][0], 1e-8)
            nu_lam = mu_lam / rho
            nu_tilde = ti.max(self.nu_tilde[i, j], 0.0)
            d = ti.max(self.wall_dist[i, j], 1e-6)

            chi = nu_tilde / ti.max(nu_lam, 1e-12)
            chi3 = chi * chi * chi
            fv1 = chi3 / (chi3 + cv1 * cv1 * cv1 + 1e-12)
            fv2 = 1.0 - chi / ti.max(1.0 + chi * fv1, 1e-12)
            ft2 = ct3 * ti.exp(-ct4 * chi * chi)

            omega = ti.abs(self.grad_v[i, j][0] - self.grad_u[i, j][1])
            s_bar = nu_tilde * fv2 / ti.max(kappa * kappa * d * d, 1e-12)
            s_tilde = ti.max(omega + s_bar, 0.3 * omega)

            r = ti.min(
                nu_tilde / ti.max(s_tilde * kappa * kappa * d * d, 1e-12),
                10.0,
            )
            g = r + cw2 * (ti.pow(r, 6.0) - r)
            fw = g * ti.pow(
                (1.0 + ti.pow(cw3, 6.0)) / ti.max(ti.pow(g, 6.0) + ti.pow(cw3, 6.0), 1e-12),
                1.0 / 6.0,
            )

            grad_nu = self.grad_nu_tilde[i, j]
            grad_sq = grad_nu.dot(grad_nu)
            production = cb1 * (1.0 - ft2) * s_tilde * nu_tilde
            destruction = cw1 * fw * ti.pow(nu_tilde / d, 2.0)
            cross_diff = cb2 * grad_sq / sigma
            source = production - destruction + cross_diff

            self.R_nu[i, j] = (
                (self.NuFlux_i[i, j] - self.NuFlux_i[i - 1, j]
                 + self.NuFlux_j[i, j] - self.NuFlux_j[i, j - 1])
                / ti.max(self.vol[i, j], 1e-12)
                - source
            )

    @ti.kernel
    def _rk_stage1_scalar(
            self,
            dt: ti.f32,
            q0: ti.template(),
            rq: ti.template(),
            q1: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q1[i, j] = ti.max(q0[i, j] - dt * rq[i, j], 0.0)

    @ti.kernel
    def _rk_stage1_scalar_local(
            self,
            q0: ti.template(),
            rq: ti.template(),
            q1: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q1[i, j] = ti.max(q0[i, j] - self.dt_local[i, j] * rq[i, j], 0.0)

    @ti.kernel
    def _rk_stage2_scalar(
            self,
            dt: ti.f32,
            q0: ti.template(),
            q1_src: ti.template(),
            rq: ti.template(),
            q2: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q2[i, j] = ti.max(0.75 * q0[i, j] + 0.25 * q1_src[i, j] - 0.25 * dt * rq[i, j], 0.0)

    @ti.kernel
    def _rk_stage2_scalar_local(
            self,
            q0: ti.template(),
            q1_src: ti.template(),
            rq: ti.template(),
            q2: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q2[i, j] = ti.max(
                0.75 * q0[i, j] + 0.25 * q1_src[i, j] - 0.25 * self.dt_local[i, j] * rq[i, j],
                0.0,
            )

    @ti.kernel
    def _rk_stage3_scalar(
            self,
            dt: ti.f32,
            q0: ti.template(),
            q2_src: ti.template(),
            rq: ti.template(),
            q_out: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q_out[i, j] = ti.max(
                (1.0 / 3.0) * q0[i, j] + (2.0 / 3.0) * q2_src[i, j] - (2.0 / 3.0) * dt * rq[i, j],
                0.0,
            )

    @ti.kernel
    def _rk_stage3_scalar_local(
            self,
            q0: ti.template(),
            q2_src: ti.template(),
            rq: ti.template(),
            q_out: ti.template()):
        for i, j in ti.ndrange(self.NI, self.NJ):
            q_out[i, j] = ti.max(
                (1.0 / 3.0) * q0[i, j]
                + (2.0 / 3.0) * q2_src[i, j]
                - (2.0 / 3.0) * self.dt_local[i, j] * rq[i, j],
                0.0,
            )

    @ti.kernel
    def _sa_residual_smoothing_pass(self, eps: ti.f32):
        """
        What: R_nu_tmp = (R_nu + eps * Σ_neighbors R_nu) / (1 + 4 eps)
        Why:  Jacobi 形式可完全平行，適合作為 pseudo-time 第一版平滑
        """
        denom = 1.0 + 4.0 * eps
        i_min = self.NG
        i_max = self.NG + self.ni - 1
        j_min = self.NG
        j_max = self.NG + self.nj - 1
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            im = ti.max(i - 1, i_min)
            ip = ti.min(i + 1, i_max)
            jm = ti.max(j - 1, j_min)
            jp = ti.min(j + 1, j_max)
            self.R_nu_tmp[i, j] = (
                self.R_nu[i, j]
                + eps * (
                    self.R_nu[im, j]
                    + self.R_nu[ip, j]
                    + self.R_nu[i, jm]
                    + self.R_nu[i, jp]
                )
            ) / denom

    @ti.kernel
    def _copy_smoothed_sa_residual(self):
        """把 smoothing 暫存結果覆寫回 SA residual"""
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            self.R_nu[i, j] = self.R_nu_tmp[i, j]

    @ti.kernel
    def _compute_viscous_fluxes_cartesian(self):
        """
        Cartesian face viscous fluxes

        What: 以 cell-centered primitive variables 的中心差分近似 face gradient，
              組成 Newtonian stress + Fourier heat flux
        Why:  第一版先用最穩妥的二階中心差分，把 laminar transport 建起來
        """
        dx = self.dx_field[None]
        dy = self.dy_field[None]
        # x-faces
        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho_l = self.W[i, j][0]
            rho_r = self.W[i + 1, j][0]
            u_l = self.W[i, j][1]
            u_r = self.W[i + 1, j][1]
            v_l = self.W[i, j][2]
            v_r = self.W[i + 1, j][2]
            p_l = self.W[i, j][3]
            p_r = self.W[i + 1, j][3]

            t_l = p_l / ti.max(rho_l, 1e-8)
            t_r = p_r / ti.max(rho_r, 1e-8)
            u_face = 0.5 * (u_l + u_r)
            v_face = 0.5 * (v_l + v_r)

            du_dx = (u_r - u_l) / dx
            dv_dx = (v_r - v_l) / dx
            dT_dx = (t_r - t_l) / dx

            du_dy_l = (self.W[i, j + 1][1] - self.W[i, j - 1][1]) / (2.0 * dy)
            du_dy_r = (self.W[i + 1, j + 1][1] - self.W[i + 1, j - 1][1]) / (2.0 * dy)
            dv_dy_l = (self.W[i, j + 1][2] - self.W[i, j - 1][2]) / (2.0 * dy)
            dv_dy_r = (self.W[i + 1, j + 1][2] - self.W[i + 1, j - 1][2]) / (2.0 * dy)

            du_dy = 0.5 * (du_dy_l + du_dy_r)
            dv_dy = 0.5 * (dv_dy_l + dv_dy_r)
            mu = 0.5 * (self.mu_eff_cell[i, j] + self.mu_eff_cell[i + 1, j])
            kappa = 0.5 * (self.kappa_eff_cell[i, j] + self.kappa_eff_cell[i + 1, j])

            div_u = du_dx + dv_dy
            tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * div_u
            tau_xy = mu * (du_dy + dv_dx)
            q_x = -kappa * dT_dx

            self.VFlux_i[i, j] = ti.Vector([
                0.0,
                tau_xx * dy,
                tau_xy * dy,
                (u_face * tau_xx + v_face * tau_xy - q_x) * dy,
            ])

        # y-faces
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            rho_b = self.W[i, j][0]
            rho_t = self.W[i, j + 1][0]
            u_b = self.W[i, j][1]
            u_t = self.W[i, j + 1][1]
            v_b = self.W[i, j][2]
            v_t = self.W[i, j + 1][2]
            p_b = self.W[i, j][3]
            p_t = self.W[i, j + 1][3]

            t_b = p_b / ti.max(rho_b, 1e-8)
            t_t = p_t / ti.max(rho_t, 1e-8)
            u_face = 0.5 * (u_b + u_t)
            v_face = 0.5 * (v_b + v_t)

            du_dy = (u_t - u_b) / dy
            dv_dy = (v_t - v_b) / dy
            dT_dy = (t_t - t_b) / dy

            du_dx_b = (self.W[i + 1, j][1] - self.W[i - 1, j][1]) / (2.0 * dx)
            du_dx_t = (self.W[i + 1, j + 1][1] - self.W[i - 1, j + 1][1]) / (2.0 * dx)
            dv_dx_b = (self.W[i + 1, j][2] - self.W[i - 1, j][2]) / (2.0 * dx)
            dv_dx_t = (self.W[i + 1, j + 1][2] - self.W[i - 1, j + 1][2]) / (2.0 * dx)

            du_dx = 0.5 * (du_dx_b + du_dx_t)
            dv_dx = 0.5 * (dv_dx_b + dv_dx_t)
            mu = 0.5 * (self.mu_eff_cell[i, j] + self.mu_eff_cell[i, j + 1])
            kappa = 0.5 * (self.kappa_eff_cell[i, j] + self.kappa_eff_cell[i, j + 1])

            div_u = du_dx + dv_dy
            tau_xy = mu * (du_dy + dv_dx)
            tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * div_u
            q_y = -kappa * dT_dy

            self.VFlux_j[i, j] = ti.Vector([
                0.0,
                tau_xy * dx,
                tau_yy * dx,
                (u_face * tau_xy + v_face * tau_yy - q_y) * dx,
            ])

    @ti.kernel
    def _compute_cell_gradients_green_gauss(self):
        """
        Green-Gauss cell gradients on structured curvilinear grid

        What: 對 u, v, T 三個 scalar fields 分別套用
              grad(phi) * vol = sum(phi_face * S_face_outward)
        Why:  structured curvilinear viscous flux 需要 physical-space gradient；
              Green-Gauss 是第一版最務實且可驗證的做法
        """
        for i, j in ti.ndrange((1, self.NI - 1), (1, self.NJ - 1)):
            u_c = self.W[i, j][1]
            v_c = self.W[i, j][2]
            T_c = self.W[i, j][3] / ti.max(self.W[i, j][0], 1e-8)

            u_e = 0.5 * (u_c + self.W[i + 1, j][1])
            u_w = 0.5 * (self.W[i - 1, j][1] + u_c)
            u_n = 0.5 * (u_c + self.W[i, j + 1][1])
            u_s = 0.5 * (self.W[i, j - 1][1] + u_c)

            v_e = 0.5 * (v_c + self.W[i + 1, j][2])
            v_w = 0.5 * (self.W[i - 1, j][2] + v_c)
            v_n = 0.5 * (v_c + self.W[i, j + 1][2])
            v_s = 0.5 * (self.W[i, j - 1][2] + v_c)

            T_e = 0.5 * (T_c + self.W[i + 1, j][3] / ti.max(self.W[i + 1, j][0], 1e-8))
            T_w = 0.5 * (self.W[i - 1, j][3] / ti.max(self.W[i - 1, j][0], 1e-8) + T_c)
            T_n = 0.5 * (T_c + self.W[i, j + 1][3] / ti.max(self.W[i, j + 1][0], 1e-8))
            T_s = 0.5 * (self.W[i, j - 1][3] / ti.max(self.W[i, j - 1][0], 1e-8) + T_c)

            grad_u_num = (
                u_e * self.S_i[i, j]
                - u_w * self.S_i[i - 1, j]
                + u_n * self.S_j[i, j]
                - u_s * self.S_j[i, j - 1]
            )
            grad_v_num = (
                v_e * self.S_i[i, j]
                - v_w * self.S_i[i - 1, j]
                + v_n * self.S_j[i, j]
                - v_s * self.S_j[i, j - 1]
            )
            grad_T_num = (
                T_e * self.S_i[i, j]
                - T_w * self.S_i[i - 1, j]
                + T_n * self.S_j[i, j]
                - T_s * self.S_j[i, j - 1]
            )

            inv_vol = 1.0 / ti.max(self.vol[i, j], 1e-12)
            self.grad_u[i, j] = grad_u_num * inv_vol
            self.grad_v[i, j] = grad_v_num * inv_vol
            self.grad_T[i, j] = grad_T_num * inv_vol

    @ti.kernel
    def _compute_viscous_fluxes_curvilinear(self):
        """
        Curvilinear face viscous fluxes from cell gradients
        """
        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            grad_u_face = 0.5 * (self.grad_u[i, j] + self.grad_u[i + 1, j])
            grad_v_face = 0.5 * (self.grad_v[i, j] + self.grad_v[i + 1, j])
            grad_T_face = 0.5 * (self.grad_T[i, j] + self.grad_T[i + 1, j])

            u_face = 0.5 * (self.W[i, j][1] + self.W[i + 1, j][1])
            v_face = 0.5 * (self.W[i, j][2] + self.W[i + 1, j][2])
            s = self.S_i[i, j]

            du_dx = grad_u_face[0]
            du_dy = grad_u_face[1]
            dv_dx = grad_v_face[0]
            dv_dy = grad_v_face[1]
            dT_dx = grad_T_face[0]
            dT_dy = grad_T_face[1]
            mu = 0.5 * (self.mu_eff_cell[i, j] + self.mu_eff_cell[i + 1, j])
            kappa = 0.5 * (self.kappa_eff_cell[i, j] + self.kappa_eff_cell[i + 1, j])

            div_u = du_dx + dv_dy
            tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * div_u
            tau_xy = mu * (du_dy + dv_dx)
            tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * div_u
            q_x = -kappa * dT_dx
            q_y = -kappa * dT_dy

            self.VFlux_i[i, j] = ti.Vector([
                0.0,
                tau_xx * s[0] + tau_xy * s[1],
                tau_xy * s[0] + tau_yy * s[1],
                (u_face * tau_xx + v_face * tau_xy - q_x) * s[0]
                + (u_face * tau_xy + v_face * tau_yy - q_y) * s[1],
            ])

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            grad_u_face = 0.5 * (self.grad_u[i, j] + self.grad_u[i, j + 1])
            grad_v_face = 0.5 * (self.grad_v[i, j] + self.grad_v[i, j + 1])
            grad_T_face = 0.5 * (self.grad_T[i, j] + self.grad_T[i, j + 1])

            u_face = 0.5 * (self.W[i, j][1] + self.W[i, j + 1][1])
            v_face = 0.5 * (self.W[i, j][2] + self.W[i, j + 1][2])
            s = self.S_j[i, j]

            du_dx = grad_u_face[0]
            du_dy = grad_u_face[1]
            dv_dx = grad_v_face[0]
            dv_dy = grad_v_face[1]
            dT_dx = grad_T_face[0]
            dT_dy = grad_T_face[1]
            mu = 0.5 * (self.mu_eff_cell[i, j] + self.mu_eff_cell[i, j + 1])
            kappa = 0.5 * (self.kappa_eff_cell[i, j] + self.kappa_eff_cell[i, j + 1])

            div_u = du_dx + dv_dy
            tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * div_u
            tau_xy = mu * (du_dy + dv_dx)
            tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * div_u
            q_x = -kappa * dT_dx
            q_y = -kappa * dT_dy

            self.VFlux_j[i, j] = ti.Vector([
                0.0,
                tau_xx * s[0] + tau_xy * s[1],
                tau_xy * s[0] + tau_yy * s[1],
                (u_face * tau_xx + v_face * tau_xy - q_x) * s[0]
                + (u_face * tau_xy + v_face * tau_yy - q_y) * s[1],
            ])

    @ti.kernel
    def _accumulate_total_residual(self):
        """
        總殘差：convective - viscous - source
        """
        fx = self.body_force[None][0]
        fy = self.body_force[None][1]

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            convective = (
                self.Flux_i[i, j] - self.Flux_i[i - 1, j]
                + self.Flux_j[i, j] - self.Flux_j[i, j - 1]
            )
            viscous = (
                self.VFlux_i[i, j] - self.VFlux_i[i - 1, j]
                + self.VFlux_j[i, j] - self.VFlux_j[i, j - 1]
            )

            rho = self.W[i, j][0]
            u = self.W[i, j][1]
            v = self.W[i, j][2]
            source = ti.Vector([
                0.0,
                rho * fx,
                rho * fy,
                rho * (u * fx + v * fy),
            ])

            self.R[i, j] = (convective - viscous) / self.vol[i, j] - source

    @ti.kernel
    def _compute_dt(self):
        """
        同時考慮對流 CFL 與黏性 diffusion CFL
        """
        cp = self.gamma / (self.gamma - 1.0)

        self.dt_field[None] = 1.0e10
        self.dt_min_field[None] = 1.0e10
        self.dt_max_field[None] = 0.0
        self.dt_sum_field[None] = 0.0
        if self.time_marching_mode_field[None] == 1:
            self.pseudo_precond_scale_min_field[None] = 1.0e10
            self.pseudo_precond_scale_max_field[None] = 0.0
            self.pseudo_precond_scale_sum_field[None] = 0.0
            if self.pseudo_adaptive_enabled_field[None] == 1:
                self.pseudo_cfl_eff_field[None] = self.pseudo_cfl_target_field[None]
            else:
                cfl_start = self.pseudo_cfl_start_field[None]
                ramp_steps = self.pseudo_cfl_ramp_steps_field[None]
                iter_count = self.pseudo_iter_field[None]
                if ramp_steps > 0:
                    ramp_ratio = ti.min(1.0, ti.cast(iter_count, ti.f32) / ti.cast(ramp_steps, ti.f32))
                    self.pseudo_cfl_eff_field[None] = cfl_start + ramp_ratio * (self.cfl - cfl_start)
                else:
                    self.pseudo_cfl_eff_field[None] = self.cfl
        else:
            self.pseudo_cfl_eff_field[None] = self.cfl
            self.pseudo_precond_scale_min_field[None] = 1.0
            self.pseudo_precond_scale_max_field[None] = 1.0
            self.pseudo_precond_scale_sum_field[None] = 0.0
        for i, j in self.dt_local:
            self.dt_local[i, j] = 0.0

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho = self.W[i, j][0]
            u = self.W[i, j][1]
            v = self.W[i, j][2]
            p = self.W[i, j][3]

            rho_safe = ti.max(rho, 1e-6)
            p_safe = ti.max(p, 1e-6)
            c = ti.sqrt(self.gamma * p_safe / rho_safe)
            c_eff = c
            if self.time_marching_mode_field[None] == 1:
                vel_mag = ti.sqrt(u * u + v * v)
                mach_local = vel_mag / ti.max(c, 1e-12)
                ref_mach = ti.max(self.pseudo_precond_ref_mach_field[None], 1e-6)
                min_scale = self.pseudo_precond_min_scale_field[None]
                scale = ti.max(min_scale, ti.min(1.0, mach_local / ref_mach))
                c_eff = c * scale
                ti.atomic_min(self.pseudo_precond_scale_min_field[None], scale)
                ti.atomic_max(self.pseudo_precond_scale_max_field[None], scale)
                ti.atomic_add(self.pseudo_precond_scale_sum_field[None], scale)

            lam_conv = 0.0
            len_i = 1.0e10
            len_j = 1.0e10

            for di in ti.static(range(2)):
                si = self.S_i[i - 1 + di, j]
                ds = ti.math.length(si)
                nx = si[0] / ti.max(ds, 1e-12)
                ny = si[1] / ti.max(ds, 1e-12)
                un = ti.abs(u * nx + v * ny)
                lam_conv += (un + c_eff) * ds
                len_i = ti.min(len_i, self.vol[i, j] / ti.max(ds, 1e-12))

            for dj in ti.static(range(2)):
                sj = self.S_j[i, j - 1 + dj]
                ds = ti.math.length(sj)
                nx = sj[0] / ti.max(ds, 1e-12)
                ny = sj[1] / ti.max(ds, 1e-12)
                un = ti.abs(u * nx + v * ny)
                lam_conv += (un + c_eff) * ds
                len_j = ti.min(len_j, self.vol[i, j] / ti.max(ds, 1e-12))

            dt_conv = self.pseudo_cfl_eff_field[None] * self.vol[i, j] / ti.max(lam_conv, 1e-12)

            mu_eff = self.mu_eff_cell[i, j]
            kappa_eff = self.kappa_eff_cell[i, j]
            nu = mu_eff / rho_safe
            alpha = kappa_eff / (rho_safe * cp)
            diff = ti.max(nu, alpha)
            inv_sum = 1.0 / ti.max(len_i * len_i, 1e-12) + 1.0 / ti.max(len_j * len_j, 1e-12)
            dt_diff = 0.5 * self.pseudo_cfl_eff_field[None] / ti.max(diff * inv_sum, 1e-12)

            dt_local = ti.min(dt_conv, dt_diff)
            self.dt_local[i, j] = dt_local
            ti.atomic_min(self.dt_field[None], dt_local)
            ti.atomic_min(self.dt_min_field[None], dt_local)
            ti.atomic_max(self.dt_max_field[None], dt_local)
            ti.atomic_add(self.dt_sum_field[None], dt_local)
