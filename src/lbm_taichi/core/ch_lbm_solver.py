"""
Cahn-Hilliard + Incompressible LBM Solver
=========================================

What:
- 以 Cahn-Hilliard 相場方程 + D2Q9 LBM 速度場，建立獨立多相求解器

Why:
- 相場模型可更自然地描述界面演化與拓撲變化
- 避免沿用既有單相/舊多相的假設耦合

When:
- 需要 Rayleigh-Taylor 等界面不穩定性的物理保真模擬
"""

from typing import Tuple

import numpy as np
import taichi as ti


@ti.data_oriented
class CHLBMSolver:
    """
    Cahn-Hilliard + Incompressible LBM 求解器（D2Q9）
    """

    solver_family = "lbm"
    equation_set = "cahn_hilliard"
    regime = "low_mach"

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float = 0.8,
        mobility: float = 0.002,
        a: float = 0.04,
        kappa: float = 0.04,
        rho_heavy: float = 1.2,
        rho_light: float = 0.2,
        gravity: Tuple[float, float] = (0.0, -1e-5),
        u_cap: float = 0.08,
        force_cap: float = 5e-4,
        boundary_y: str = "noslip",
        low_mach_corr: float = 0.2,
        compressibility: float = 0.2,
    ):
        if tau <= 0.5:
            raise ValueError("tau 必須 > 0.5")
        if mobility <= 0.0:
            raise ValueError("mobility 必須 > 0")

        self.nx = nx
        self.ny = ny
        self.nx_g = nx + 2
        self.ny_g = ny + 2

        self.tau = tau
        self.omega = 1.0 / tau
        self.mobility = mobility
        self.a = a
        self.kappa = kappa
        self.rho_heavy = rho_heavy
        self.rho_light = rho_light
        self.rho_ref = 0.5 * (rho_heavy + rho_light)
        self.rho_floor = 1e-6
        self.u_cap = max(1e-4, float(u_cap))
        self.force_cap = max(1e-8, float(force_cap))
        self.low_mach_corr = max(0.0, float(low_mach_corr))
        self.compressibility = min(1.0, max(0.0, float(compressibility)))
        by = boundary_y.lower()
        if by not in {"noslip", "freeslip"}:
            raise ValueError("boundary_y 必須為 'noslip' 或 'freeslip'")
        self.boundary_y_mode = 0 if by == "noslip" else 1

        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)

        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        self.rho = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.phi = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.phi_new = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.mu = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.force = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx_g, self.ny_g))
        self.div_u = ti.field(dtype=ti.f32, shape=(self.nx_g, self.ny_g))

        self.gravity = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.gravity[None] = ti.Vector([gravity[0], gravity[1]])

        self.periodic_x = ti.field(dtype=ti.i32, shape=())
        self.periodic_x[None] = 1
        self.step_count = 0

        self._init_constants()
        self._initialize_fields()

    def _init_constants(self):
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
        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)

    @ti.func
    def _clip_vec(self, v, vmax: ti.f32):
        n = v.norm()
        out = v
        if n > vmax and n > 1e-12:
            out = v * (vmax / n)
        return out

    @ti.func
    def _x_wrap(self, i: ti.i32) -> ti.i32:
        out = i
        if self.periodic_x[None] == 1:
            if out < 1:
                out = self.nx
            elif out > self.nx:
                out = 1
        return out

    @ti.func
    def _y_clamp(self, j: ti.i32) -> ti.i32:
        return ti.min(self.ny, ti.max(1, j))

    @ti.kernel
    def _initialize_fields(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            self.rho[ig, jg] = self.rho_ref
            self.u[ig, jg] = ti.Vector([0.0, 0.0])
            self.phi[ig, jg] = 0.0
            self.phi_new[ig, jg] = 0.0
            self.mu[ig, jg] = 0.0
            self.force[ig, jg] = ti.Vector([0.0, 0.0])
            self.div_u[ig, jg] = 0.0
            for k in ti.static(range(9)):
                self.f[ig, jg][k] = self.w[k] * self.rho_ref
                self.f_new[ig, jg][k] = self.w[k] * self.rho_ref

    def set_initial_phi(self, phi: np.ndarray):
        if phi.shape != (self.nx, self.ny):
            raise ValueError("phi shape 必須為 (nx, ny)")
        phi_g = np.zeros((self.nx_g, self.ny_g), dtype=np.float32)
        phi_g[1 : self.nx + 1, 1 : self.ny + 1] = np.clip(phi, -1.0, 1.0)
        phi_g[0, 1 : self.ny + 1] = phi_g[self.nx, 1 : self.ny + 1]
        phi_g[self.nx + 1, 1 : self.ny + 1] = phi_g[1, 1 : self.ny + 1]
        phi_g[:, 0] = phi_g[:, 1]
        phi_g[:, self.ny + 1] = phi_g[:, self.ny]
        self.phi.from_numpy(phi_g)
        self._initialize_from_phi()

    @ti.kernel
    def _initialize_from_phi(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            phi = self.phi[ig, jg]
            rho = 0.5 * (self.rho_heavy + self.rho_light) + 0.5 * (
                self.rho_heavy - self.rho_light
            ) * phi
            rho = ti.max(rho, self.rho_floor)
            self.rho[ig, jg] = rho
            self.u[ig, jg] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.f[ig, jg][k] = self.w[k] * rho
                self.f_new[ig, jg][k] = self.w[k] * rho

    @ti.kernel
    def _compute_macro(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            rho = 0.0
            mom = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                f_k = self.f[ig, jg][k]
                rho += f_k
                mom += f_k * ti.cast(self.e[k], ti.f32)
            rho = ti.max(rho, self.rho_floor)
            u = (mom + 0.5 * self.force[ig, jg]) / rho
            u = self._clip_vec(u, self.u_cap)
            self.rho[ig, jg] = rho
            self.u[ig, jg] = u

    @ti.kernel
    def _apply_noflux_bc(self):
        """
        What: 對 phi / mu 施加 no-flux 邊界（Neumann: 法向梯度 0）
        Why: Cahn-Hilliard 需要化學位與相場的無通量條件以維持物理一致性
        When: 每步在更新 mu 與 phi 前後
        """
        for i in ti.ndrange(self.nx + 2):
            self.phi[i, 0] = self.phi[i, 1]
            self.phi[i, self.ny + 1] = self.phi[i, self.ny]
            self.mu[i, 0] = self.mu[i, 1]
            self.mu[i, self.ny + 1] = self.mu[i, self.ny]

        if self.periodic_x[None] == 1:
            for j in ti.ndrange(self.ny + 2):
                self.phi[0, j] = self.phi[self.nx, j]
                self.phi[self.nx + 1, j] = self.phi[1, j]
                self.mu[0, j] = self.mu[self.nx, j]
                self.mu[self.nx + 1, j] = self.mu[1, j]

    @ti.kernel
    def _compute_divergence(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            il = self._x_wrap(ig - 1)
            ir = self._x_wrap(ig + 1)
            jb = self._y_clamp(jg - 1)
            jt = self._y_clamp(jg + 1)
            div = 0.5 * (self.u[ir, jg][0] - self.u[il, jg][0]) + 0.5 * (
                self.u[ig, jt][1] - self.u[ig, jb][1]
            )
            self.div_u[ig, jg] = div

    @ti.kernel
    def _apply_low_mach_correction(self):
        beta = self.low_mach_corr
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            il = self._x_wrap(ig - 1)
            ir = self._x_wrap(ig + 1)
            jb = self._y_clamp(jg - 1)
            jt = self._y_clamp(jg + 1)
            grad_div = ti.Vector(
                [
                    0.5 * (self.div_u[ir, jg] - self.div_u[il, jg]),
                    0.5 * (self.div_u[ig, jt] - self.div_u[ig, jb]),
                ]
            )
            corr = -beta * grad_div
            self.force[ig, jg] = self._clip_vec(
                self.force[ig, jg] + corr, self.force_cap
            )

    @ti.kernel
    def _compute_mu_and_force(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1

            il = self._x_wrap(ig - 1)
            ir = self._x_wrap(ig + 1)
            jb = self._y_clamp(jg - 1)
            jt = self._y_clamp(jg + 1)

            phi_c = self.phi[ig, jg]
            phi_l = self.phi[il, jg]
            phi_r = self.phi[ir, jg]
            phi_b = self.phi[ig, jb]
            phi_t = self.phi[ig, jt]

            lap_phi = phi_l + phi_r + phi_b + phi_t - 4.0 * phi_c
            mu = self.a * (phi_c * phi_c * phi_c - phi_c) - self.kappa * lap_phi
            self.mu[ig, jg] = mu

        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            il = self._x_wrap(ig - 1)
            ir = self._x_wrap(ig + 1)
            jb = self._y_clamp(jg - 1)
            jt = self._y_clamp(jg + 1)

            grad_mu = ti.Vector(
                [
                    0.5 * (self.mu[ir, jg] - self.mu[il, jg]),
                    0.5 * (self.mu[ig, jt] - self.mu[ig, jb]),
                ]
            )
            phi_c = self.phi[ig, jg]
            rho_mix = 0.5 * (self.rho_heavy + self.rho_light) + 0.5 * (
                self.rho_heavy - self.rho_light
            ) * phi_c
            buoy = (rho_mix - self.rho_ref) * self.gravity[None]
            f = -phi_c * grad_mu + buoy
            self.force[ig, jg] = self._clip_vec(f, self.force_cap)

    @ti.kernel
    def _collide_stream(self):
        cs2 = 1.0 / 3.0
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            rho = self.rho[ig, jg]
            u = self.u[ig, jg]
            u2 = u.dot(u)
            force = self.force[ig, jg]
            rho_eff = self.rho_ref + self.compressibility * (rho - self.rho_ref)
            rho_eff = ti.max(rho_eff, self.rho_floor)

            for k in ti.static(range(9)):
                e_k = ti.cast(self.e[k], ti.f32)
                eu = e_k.dot(u)
                feq = self.w[k] * rho_eff * (
                    1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
                )
                term = (e_k - u) / cs2 + (eu / (cs2 * cs2)) * e_k
                forcing = (1.0 - 0.5 * self.omega) * self.w[k] * term.dot(force)
                f_post = (
                    self.f[ig, jg][k]
                    - self.omega * (self.f[ig, jg][k] - feq)
                    + forcing
                )

                ni = ig + self.e[k][0]
                nj = jg + self.e[k][1]
                ni = self._x_wrap(ni)
                nj_clamped = self._y_clamp(nj)

                wall_hit = nj != nj_clamped
                if wall_hit:
                    if self.boundary_y_mode == 0:
                        inv_k = self.inv[k]
                        self.f_new[ig, jg][inv_k] = f_post
                    else:
                        # free-slip: 反射法向分量，保留切向
                        map_k = k
                        if k == 2:
                            map_k = 4
                        elif k == 4:
                            map_k = 2
                        elif k == 5:
                            map_k = 8
                        elif k == 8:
                            map_k = 5
                        elif k == 6:
                            map_k = 7
                        elif k == 7:
                            map_k = 6
                        self.f_new[ig, jg][map_k] = f_post
                else:
                    self.f_new[ni, nj_clamped][k] = f_post

    @ti.kernel
    def _update_phi(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            il = self._x_wrap(ig - 1)
            ir = self._x_wrap(ig + 1)
            jb = self._y_clamp(jg - 1)
            jt = self._y_clamp(jg + 1)

            phi_c = self.phi[ig, jg]
            u = self.u[ig, jg]
            grad_phi = ti.Vector(
                [
                    0.5 * (self.phi[ir, jg] - self.phi[il, jg]),
                    0.5 * (self.phi[ig, jt] - self.phi[ig, jb]),
                ]
            )
            adv = u.dot(grad_phi)

            lap_mu = (
                self.mu[il, jg]
                + self.mu[ir, jg]
                + self.mu[ig, jb]
                + self.mu[ig, jt]
                - 4.0 * self.mu[ig, jg]
            )

            phi_new = phi_c + (-adv + self.mobility * lap_mu)
            self.phi_new[ig, jg] = ti.min(1.0, ti.max(-1.0, phi_new))

        for i, j in ti.ndrange(self.nx, self.ny):
            ig = i + 1
            jg = j + 1
            self.phi[ig, jg] = self.phi_new[ig, jg]

    @ti.kernel
    def _commit_stream(self):
        for i, j in ti.ndrange(self.nx_g, self.ny_g):
            self.f[i, j] = self.f_new[i, j]

    def step(self):
        self._apply_noflux_bc()
        self._compute_mu_and_force()
        self._compute_macro()
        self._compute_divergence()
        self._apply_low_mach_correction()
        self._collide_stream()
        self._commit_stream()
        self._compute_macro()
        self._update_phi()
        self._apply_noflux_bc()
        self.step_count += 1

    def get_fields(self):
        rho = self.rho.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u = self.u.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        phi = self.phi.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        return {
            "rho": rho,
            "u": u,
            "phi": phi,
            "step": self.step_count,
        }

    def get_diagnostics(self):
        """
        Why: SolverProtocol 要求 get_diagnostics；直接讀 phi/u field 確保 step() 前後語意正確
        """
        phi_np = self.phi.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u_np = self.u.to_numpy()[1 : self.nx + 1, 1 : self.ny + 1]
        u_max = float(np.max(np.linalg.norm(u_np, axis=-1)))
        return {
            "step_count": int(self.step_count),
            "phi_min": float(phi_np.min()),
            "phi_max": float(phi_np.max()),
            "u_max": u_max,
        }
