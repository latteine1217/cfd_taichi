"""
IncompressibleNavierStokesSolver — 2D 不可壓縮 Navier-Stokes 骨架
=================================================================

What:
- 提供不可壓縮 FVM Navier-Stokes 的正式 solver 類別骨架
- 定義與現有 FVM solver 對齊的 grid / BC / initialization / diagnostics 介面

Why:
- `fvm` 與 `compressible` 不應被綁成同一件事
- 真正的不可壓縮 solver 需要獨立的狀態變數、壓力投影與邊界條件語義，
  不能再從 compressible Euler state 硬裁切

When:
- 作為 projection / pressure-Poisson 路線的正式落點
- 讓 Couette / Poiseuille / cavity 類 benchmark 之後可以轉移到真正的 incompressible 路徑

目前狀態：
- 可建立 solver、設定網格與邊界、初始化場、輸出診斷量
- `step()` 已接上 Cartesian uniform-grid 的第一版 pressure projection scaffold
- curvilinear 幾何與完整對流項仍待後續補齊
"""

from typing import Any

import numpy as np
import taichi as ti

from .poisson_solver import JacobiPressurePoissonSolver


@ti.data_oriented
class IncompressibleNavierStokesSolver:
    """
    2D 不可壓縮 Navier-Stokes solver 骨架（structured-grid）

    What:
    - 使用 cell-centered 速度/壓力場，保留未來 projection method 所需的工作陣列

    Why:
    - incompressible solver 的主要未知量是 `(u, v, p)`，不是 compressible 的
      conserved variables；因此需要獨立類別維持正確的數學語義

    When:
    - 目前可用於 API 對齊、benchmark 腳手架與後續 pressure-Poisson 開發
    """

    solver_family = "fvm"
    equation_set = "navier_stokes"
    regime = "incompressible"

    NG = 2

    def __init__(
        self,
        ni: int,
        nj: int,
        cfl: float = 0.5,
        re: float | None = 100.0,
        nu: float | None = None,
        rho_ref: float = 1.0,
        u_ref: float = 1.0,
        length_scale: float = 1.0,
    ):
        """
        Args:
            ni, nj: interior cell counts
            cfl: predictor step 用的對流 CFL 目標
            re: Reynolds number；若 `nu` 未指定，則由 `nu = u_ref * L / Re` 取得
            nu: kinematic viscosity，直接指定時優先於 `re`
            rho_ref: 參考密度（pressure gradient / force scaling 用）
            u_ref: 參考速度
            length_scale: 特徵長度
        """
        if ni <= 0 or nj <= 0:
            raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
        if cfl <= 0.0:
            raise ValueError(f"cfl must be positive, got {cfl}")
        if rho_ref <= 0.0:
            raise ValueError(f"rho_ref must be positive, got {rho_ref}")
        if u_ref <= 0.0:
            raise ValueError(f"u_ref must be positive, got {u_ref}")
        if length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")

        if nu is None:
            if re is None or re <= 0.0:
                raise ValueError("Either nu or a positive re must be provided.")
            nu = u_ref * length_scale / re
        elif nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")

        self.ni = int(ni)
        self.nj = int(nj)
        self.cfl = float(cfl)
        self.re = None if re is None else float(re)
        self.nu = float(nu)
        self.rho_ref = float(rho_ref)
        self.u_ref = float(u_ref)
        self.length_scale = float(length_scale)
        self.mu = self.rho_ref * self.nu

        self.NI = self.ni + 2 * self.NG
        self.NJ = self.nj + 2 * self.NG

        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.NI, self.NJ))
        self.u_star = ti.Vector.field(2, dtype=ti.f32, shape=(self.NI, self.NJ))
        self.u_prev = ti.Vector.field(2, dtype=ti.f32, shape=(self.NI, self.NJ))
        self.p = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.p_corr = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.div = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.rhs = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.cell_volume = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.cell_size = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.xc = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))
        self.yc = ti.field(dtype=ti.f32, shape=(self.NI, self.NJ))

        self.dt_field = ti.field(dtype=ti.f32, shape=())
        self.body_force = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.body_force[None] = ti.Vector([0.0, 0.0])
        self.dt_field[None] = 0.0

        self._grid_ready = False
        self._state_ready = False
        self._metric_mode = "none"
        self._periodic_i = False
        self._periodic_j = False

        self._wall_bc: dict[str, dict[str, Any]] = {}
        self._node_shape = (self.NI + 1, self.NJ + 1)
        self._projection_max_iters = 200
        self._projection_tol = 1e-6
        self._projection_last_iters = 0
        self._projection_last_residual = np.nan
        self._poisson_solver = JacobiPressurePoissonSolver(
            ng=self.NG,
            ni=self.ni,
            nj=self.nj,
            max_iters=self._projection_max_iters,
            tol=self._projection_tol,
        )

    @ti.kernel
    def _fill_cartesian_geometry_kernel(self, dx: ti.f32, dy: ti.f32):
        for i, j in self.cell_volume:
            self.cell_volume[i, j] = dx * dy
            self.cell_size[i, j] = ti.min(dx, dy)
            self.xc[i, j] = (i - self.NG + 0.5) * dx
            self.yc[i, j] = (j - self.NG + 0.5) * dy

    @ti.kernel
    def _init_uniform_kernel(self, u0: ti.f32, v0: ti.f32, p0: ti.f32):
        for i, j in self.p:
            vel = ti.Vector([u0, v0])
            self.u[i, j] = vel
            self.u_star[i, j] = vel
            self.u_prev[i, j] = vel
            self.p[i, j] = p0
            self.p_corr[i, j] = 0.0
            self.div[i, j] = 0.0
            self.rhs[i, j] = 0.0

    @ti.kernel
    def _load_state_from_numpy(
        self,
        uv_np: ti.types.ndarray(),
        p_np: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(self.ni, self.nj):
            ig = i + self.NG
            jg = j + self.NG
            vel = ti.Vector([uv_np[i, j, 0], uv_np[i, j, 1]])
            self.u[ig, jg] = vel
            self.u_star[ig, jg] = vel
            self.u_prev[ig, jg] = vel
            self.p[ig, jg] = p_np[i, j]
            self.p_corr[ig, jg] = 0.0
            self.div[ig, jg] = 0.0
            self.rhs[ig, jg] = 0.0

    @ti.kernel
    def _copy_interior_to_ghost(self):
        ng = ti.static(self.NG)
        ni = ti.static(self.ni)
        nj = ti.static(self.nj)

        for g, j in ti.ndrange(ng, nj):
            jg = j + ng
            left_src = ng
            right_src = ng + ni - 1
            self.u[g, jg] = self.u[left_src, jg]
            self.u_star[g, jg] = self.u_star[left_src, jg]
            self.u_prev[g, jg] = self.u_prev[left_src, jg]
            self.p[g, jg] = self.p[left_src, jg]
            self.p_corr[g, jg] = self.p_corr[left_src, jg]

            ig_r = ng + ni + g
            self.u[ig_r, jg] = self.u[right_src, jg]
            self.u_star[ig_r, jg] = self.u_star[right_src, jg]
            self.u_prev[ig_r, jg] = self.u_prev[right_src, jg]
            self.p[ig_r, jg] = self.p[right_src, jg]
            self.p_corr[ig_r, jg] = self.p_corr[right_src, jg]

        for i, g in ti.ndrange(self.NI, ng):
            bottom_src = ng
            top_src = ng + nj - 1
            self.u[i, g] = self.u[i, bottom_src]
            self.u_star[i, g] = self.u_star[i, bottom_src]
            self.u_prev[i, g] = self.u_prev[i, bottom_src]
            self.p[i, g] = self.p[i, bottom_src]
            self.p_corr[i, g] = self.p_corr[i, bottom_src]

            jg_t = ng + nj + g
            self.u[i, jg_t] = self.u[i, top_src]
            self.u_star[i, jg_t] = self.u_star[i, top_src]
            self.u_prev[i, jg_t] = self.u_prev[i, top_src]
            self.p[i, jg_t] = self.p[i, top_src]
            self.p_corr[i, jg_t] = self.p_corr[i, top_src]

    @ti.kernel
    def _compute_divergence_cartesian(self, dx: ti.f32, dy: ti.f32):
        for i, j in ti.ndrange((self.NG, self.NG + self.ni), (self.NG, self.NG + self.nj)):
            du_dx = 0.5 * (self.u[i + 1, j][0] - self.u[i - 1, j][0]) / dx
            dv_dy = 0.5 * (self.u[i, j + 1][1] - self.u[i, j - 1][1]) / dy
            self.div[i, j] = du_dx + dv_dy

    def set_cartesian_grid(self, dx: float, dy: float):
        """
        設定均勻直角網格幾何。

        What:
        - 記錄 cell volume / characteristic size / cell center

        Why:
        - projection method 的 predictor CFL、Poisson scaling 與 diagnostics
          都需要統一的幾何資訊
        """
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")
        self._fill_cartesian_geometry_kernel(float(dx), float(dy))
        self._grid_ready = True
        self._metric_mode = "cartesian"
        self._dx = float(dx)
        self._dy = float(dy)

    def set_curvilinear_grid(self, x_node: np.ndarray, y_node: np.ndarray):
        """
        設定曲線 structured grid。

        What:
        - 接受與 compressible FVM 相同的 node layout：shape = (NI+1, NJ+1)

        Why:
        - 讓同一套 O-grid / sheared-grid 生成器能被 incompressible solver 重用
        - 即使 step 尚未實作，幾何介面必須先固定下來
        """
        expected = self._node_shape
        if x_node.shape != expected or y_node.shape != expected:
            raise ValueError(
                f"Expected node shape {expected}, got x={x_node.shape}, y={y_node.shape}"
            )

        xn = np.asarray(x_node, dtype=np.float32)
        yn = np.asarray(y_node, dtype=np.float32)
        xc = 0.25 * (xn[:-1, :-1] + xn[1:, :-1] + xn[:-1, 1:] + xn[1:, 1:])
        yc = 0.25 * (yn[:-1, :-1] + yn[1:, :-1] + yn[:-1, 1:] + yn[1:, 1:])
        area = 0.5 * np.abs(
            (xn[1:, :-1] - xn[:-1, :-1]) * (yn[:-1, 1:] - yn[:-1, :-1])
            - (yn[1:, :-1] - yn[:-1, :-1]) * (xn[:-1, 1:] - xn[:-1, :-1])
        )
        area += 0.5 * np.abs(
            (xn[1:, 1:] - xn[1:, :-1]) * (yn[:-1, 1:] - yn[1:, :-1])
            - (yn[1:, 1:] - yn[1:, :-1]) * (xn[:-1, 1:] - xn[1:, :-1])
        )
        size = np.sqrt(np.maximum(area, 1e-12))

        self.xc.from_numpy(np.ascontiguousarray(xc))
        self.yc.from_numpy(np.ascontiguousarray(yc))
        self.cell_volume.from_numpy(np.ascontiguousarray(area.astype(np.float32)))
        self.cell_size.from_numpy(np.ascontiguousarray(size.astype(np.float32)))

        self._grid_ready = True
        self._metric_mode = "curvilinear"
        self._x_node = xn
        self._y_node = yn

    def set_periodic_bc(self, i_dir: bool = False, j_dir: bool = False):
        """
        設定週期邊界旗標。

        Why:
        - Taylor-Green / channel 類 benchmark 需要與 compressible solver 對齊的公開 API
        """
        self._periodic_i = bool(i_dir)
        self._periodic_j = bool(j_dir)

    def set_body_force(self, fx: float = 0.0, fy: float = 0.0):
        """
        設定均勻體積力。

        Why:
        - incompressible channel / Poiseuille 通常以 body-force 取代壓力入口出口
        """
        self.body_force[None] = ti.Vector([float(fx), float(fy)])

    def set_projection_controls(self, max_iters: int = 200, tol: float = 1e-6):
        """
        設定 pressure Poisson 的 Jacobi 迭代控制。

        What:
        - 調整最大迭代次數與收斂容差

        Why:
        - pressure projection 的成本與收斂程度需要可觀測、可調整
        """
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")
        self._projection_max_iters = int(max_iters)
        self._projection_tol = float(tol)
        self._poisson_solver.set_controls(max_iters=max_iters, tol=tol)

    def set_no_slip_wall(
        self,
        location: str,
        u_wall: float = 0.0,
        v_wall: float = 0.0,
    ):
        """
        註冊 no-slip wall 邊界。

        What:
        - 先固定牆面語義與公開 API，供未來 projection BC 使用

        Why:
        - incompressible 壁面速度與 pressure-Neumann 會一起影響 Poisson 問題，
          因此需要獨立於 compressible ghost mirror 的邊界描述
        """
        aliases = {
            "left": "i_min",
            "right": "i_max",
            "bottom": "j_min",
            "top": "j_max",
            "i_min": "i_min",
            "i_max": "i_max",
            "j_min": "j_min",
            "j_max": "j_max",
        }
        if location not in aliases:
            raise NotImplementedError(
                "IncompressibleNavierStokesSolver implements no-slip walls on "
                "'left'/'right'/'bottom'/'top' only."
            )
        edge = aliases[location]
        self._wall_bc[edge] = {
            "type": "no_slip",
            "u_wall": float(u_wall),
            "v_wall": float(v_wall),
        }

    def _apply_pressure_bc_numpy(self, p: np.ndarray) -> None:
        ng = self.NG
        ni = self.ni
        nj = self.nj

        if self._periodic_i:
            for g in range(ng):
                p[g, :] = p[ni + g, :]
                p[ng + ni + g, :] = p[ng + g, :]
        else:
            for g in range(ng):
                p[g, :] = p[ng, :]
                p[ng + ni + g, :] = p[ng + ni - 1, :]

        if self._periodic_j:
            for g in range(ng):
                p[:, g] = p[:, nj + g]
                p[:, ng + nj + g] = p[:, ng + g]
        else:
            for g in range(ng):
                p[:, g] = p[:, ng]
                p[:, ng + nj + g] = p[:, ng + nj - 1]

    def _apply_velocity_bc_numpy(self, uv: np.ndarray) -> None:
        ng = self.NG
        ni = self.ni
        nj = self.nj

        if self._periodic_i:
            for g in range(ng):
                uv[g, :, :] = uv[ni + g, :, :]
                uv[ng + ni + g, :, :] = uv[ng + g, :, :]
        else:
            for g in range(ng):
                uv[g, :, :] = uv[ng, :, :]
                uv[ng + ni + g, :, :] = uv[ng + ni - 1, :, :]

        if self._periodic_j:
            for g in range(ng):
                uv[:, g, :] = uv[:, nj + g, :]
                uv[:, ng + nj + g, :] = uv[:, ng + g, :]
        else:
            for g in range(ng):
                uv[:, g, :] = uv[:, ng, :]
                uv[:, ng + nj + g, :] = uv[:, ng + nj - 1, :]

        bc = self._wall_bc.get("i_min")
        if bc is not None:
            u_wall = bc["u_wall"]
            v_wall = bc["v_wall"]
            for g in range(ng):
                src = ng + g
                dst = ng - 1 - g
                uv[dst, :, 0] = 2.0 * u_wall - uv[src, :, 0]
                uv[dst, :, 1] = 2.0 * v_wall - uv[src, :, 1]

        bc = self._wall_bc.get("i_max")
        if bc is not None:
            u_wall = bc["u_wall"]
            v_wall = bc["v_wall"]
            for g in range(ng):
                src = ng + ni - 1 - g
                dst = ng + ni + g
                uv[dst, :, 0] = 2.0 * u_wall - uv[src, :, 0]
                uv[dst, :, 1] = 2.0 * v_wall - uv[src, :, 1]

        bc = self._wall_bc.get("j_min")
        if bc is not None:
            u_wall = bc["u_wall"]
            v_wall = bc["v_wall"]
            for g in range(ng):
                src = ng + g
                dst = ng - 1 - g
                uv[:, dst, 0] = 2.0 * u_wall - uv[:, src, 0]
                uv[:, dst, 1] = 2.0 * v_wall - uv[:, src, 1]

        bc = self._wall_bc.get("j_max")
        if bc is not None:
            u_wall = bc["u_wall"]
            v_wall = bc["v_wall"]
            for g in range(ng):
                src = ng + nj - 1 - g
                dst = ng + nj + g
                uv[:, dst, 0] = 2.0 * u_wall - uv[:, src, 0]
                uv[:, dst, 1] = 2.0 * v_wall - uv[:, src, 1]

    def _apply_state_bc(self):
        uv = np.asarray(self.u.to_numpy(), dtype=np.float32)
        p = np.asarray(self.p.to_numpy(), dtype=np.float32)
        self._apply_velocity_bc_numpy(uv)
        self._apply_pressure_bc_numpy(p)
        self.u.from_numpy(np.ascontiguousarray(uv))
        self.p.from_numpy(np.ascontiguousarray(p))

    def _compute_dt_numpy(self, uv: np.ndarray) -> float:
        ng = self.NG
        speed = np.linalg.norm(uv[ng:ng + self.ni, ng:ng + self.nj], axis=2)
        u_max = float(np.max(speed))
        h_min = float(
            np.min(
                self.cell_size.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
            )
        )
        adv_scale = max(u_max, self.u_ref, 1e-6)
        dt_adv = self.cfl * h_min / adv_scale
        dt_diff = 0.25 * h_min * h_min / max(self.nu, 1e-12)
        return float(min(dt_adv, dt_diff))

    def _compute_divergence_cartesian_numpy(self, uv: np.ndarray) -> np.ndarray:
        div = np.zeros((self.NI, self.NJ), dtype=np.float32)
        ng = self.NG
        div[ng:ng + self.ni, ng:ng + self.nj] = (
            0.5 * (uv[ng + 1:ng + self.ni + 1, ng:ng + self.nj, 0] - uv[ng - 1:ng + self.ni - 1, ng:ng + self.nj, 0]) / self._dx
            + 0.5 * (uv[ng:ng + self.ni, ng + 1:ng + self.nj + 1, 1] - uv[ng:ng + self.ni, ng - 1:ng + self.nj - 1, 1]) / self._dy
        )
        return div

    def _project_cartesian_step(self) -> float:
        u0 = np.asarray(self.u.to_numpy(), dtype=np.float32)
        p0 = np.asarray(self.p.to_numpy(), dtype=np.float32)

        self._apply_velocity_bc_numpy(u0)
        self._apply_pressure_bc_numpy(p0)

        dt = self._compute_dt_numpy(u0)
        dx2 = self._dx * self._dx
        dy2 = self._dy * self._dy
        fx = float(self.body_force[None][0])
        fy = float(self.body_force[None][1])

        ng = self.NG
        interior = (slice(ng, ng + self.ni), slice(ng, ng + self.nj))

        u_star = u0.copy()

        lap_u = (
            (u0[ng + 1:ng + self.ni + 1, ng:ng + self.nj, 0] - 2.0 * u0[interior + (0,)] + u0[ng - 1:ng + self.ni - 1, ng:ng + self.nj, 0]) / dx2
            + (u0[ng:ng + self.ni, ng + 1:ng + self.nj + 1, 0] - 2.0 * u0[interior + (0,)] + u0[ng:ng + self.ni, ng - 1:ng + self.nj - 1, 0]) / dy2
        )
        lap_v = (
            (u0[ng + 1:ng + self.ni + 1, ng:ng + self.nj, 1] - 2.0 * u0[interior + (1,)] + u0[ng - 1:ng + self.ni - 1, ng:ng + self.nj, 1]) / dx2
            + (u0[ng:ng + self.ni, ng + 1:ng + self.nj + 1, 1] - 2.0 * u0[interior + (1,)] + u0[ng:ng + self.ni, ng - 1:ng + self.nj - 1, 1]) / dy2
        )

        # TODO: 補上對流 predictor；目前先以 Stokes + body-force projection scaffold 驗證壓力校正鏈。
        u_star[interior + (0,)] = u0[interior + (0,)] + dt * (self.nu * lap_u + fx)
        u_star[interior + (1,)] = u0[interior + (1,)] + dt * (self.nu * lap_v + fy)
        self._apply_velocity_bc_numpy(u_star)

        rhs = np.zeros((self.NI, self.NJ), dtype=np.float32)
        div_star = self._compute_divergence_cartesian_numpy(u_star)
        rhs[interior] = self.rho_ref / max(dt, 1e-12) * div_star[interior]

        p_corr0 = np.zeros((self.NI, self.NJ), dtype=np.float32)
        p_corr, iters, residual = self._poisson_solver.solve(
            rhs=rhs,
            p_seed=p_corr0,
            dx=self._dx,
            dy=self._dy,
            apply_pressure_bc=self._apply_pressure_bc_numpy,
        )

        u_new = u_star.copy()
        grad_p_x = 0.5 * (
            p_corr[ng + 1:ng + self.ni + 1, ng:ng + self.nj]
            - p_corr[ng - 1:ng + self.ni - 1, ng:ng + self.nj]
        ) / self._dx
        grad_p_y = 0.5 * (
            p_corr[ng:ng + self.ni, ng + 1:ng + self.nj + 1]
            - p_corr[ng:ng + self.ni, ng - 1:ng + self.nj - 1]
        ) / self._dy
        u_new[interior + (0,)] = u_star[interior + (0,)] - dt / self.rho_ref * grad_p_x
        u_new[interior + (1,)] = u_star[interior + (1,)] - dt / self.rho_ref * grad_p_y
        self._apply_velocity_bc_numpy(u_new)

        p_new = p0 + p_corr
        self._apply_pressure_bc_numpy(p_new)

        div_new = self._compute_divergence_cartesian_numpy(u_new)

        self.u_prev.from_numpy(np.ascontiguousarray(u0))
        self.u_star.from_numpy(np.ascontiguousarray(u_star))
        self.u.from_numpy(np.ascontiguousarray(u_new))
        self.p_corr.from_numpy(np.ascontiguousarray(p_corr))
        self.p.from_numpy(np.ascontiguousarray(p_new))
        self.rhs.from_numpy(np.ascontiguousarray(rhs))
        self.div.from_numpy(np.ascontiguousarray(div_new))

        self.dt_field[None] = float(dt)
        self._projection_last_iters = int(iters)
        self._projection_last_residual = float(residual)
        return float(dt)

    def init_uniform(self, u: float = 0.0, v: float = 0.0, p: float = 0.0):
        """
        以均勻速度/壓力初始化整個計算域。

        Why:
        - skeleton 階段先固定不可壓縮 state 定義：`u=(u,v)`, `p`
        """
        self._init_uniform_kernel(float(u), float(v), float(p))
        self._apply_state_bc()
        self._state_ready = True

    def init_from_primitive_numpy(self, W_np: np.ndarray):
        """
        由 numpy 場初始化 interior cells。

        What:
        - 接受 shape `(ni, nj, 3)`，最後一維為 `[u, v, p]`

        Why:
        - 與既有 compressible solver 的 `init_from_primitive_numpy()` 命名對齊，
          但內容改成真正的 incompressible primitive variables
        """
        if W_np.shape != (self.ni, self.nj, 3):
            raise ValueError(
                f"Expected shape ({self.ni}, {self.nj}, 3), got {W_np.shape}"
            )
        W = np.asarray(W_np, dtype=np.float32)
        self._load_state_from_numpy(
            np.ascontiguousarray(W[:, :, :2]),
            np.ascontiguousarray(W[:, :, 2]),
        )
        self._apply_state_bc()
        self._state_ready = True

    def get_primitive(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        回傳 interior cell 的 `(u, v, p)`。
        """
        ng = self.NG
        u_np = self.u.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
        p_np = self.p.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
        return u_np[:, :, 0], u_np[:, :, 1], p_np

    def get_fields(self) -> dict[str, np.ndarray]:
        """
        回傳標準化場資料。

        Why:
        - 與 LBM / compressible FVM 一樣，後續 diagnostics / visualization
          需要統一的資料出口
        """
        ng = self.NG
        u_np = self.u.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
        p_np = self.p.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
        div_np = self.div.to_numpy()[ng:ng + self.ni, ng:ng + self.nj]
        return {
            "u": u_np,
            "p": p_np,
            "divergence": div_np,
        }

    def get_transport_coefficients(self) -> dict[str, float]:
        """
        回傳不可壓縮 transport 參數。
        """
        return {
            "rho_ref": self.rho_ref,
            "nu": self.nu,
            "mu": self.mu,
            "re": np.nan if self.re is None else self.re,
        }

    def get_dt_diagnostics(self) -> dict[str, float | str]:
        """
        回傳 predictor CFL 診斷。

        Why:
        - 即使 step 尚未實作，案例腳手架仍可先看 grid / velocity 對 dt 的約束
        """
        if not self._grid_ready:
            raise RuntimeError("Grid is not ready. Call set_cartesian_grid() or set_curvilinear_grid() first.")
        if not self._state_ready:
            raise RuntimeError("State is not initialized. Call init_uniform() or init_from_primitive_numpy() first.")

        u_np = self.u.to_numpy()[self.NG:self.NG + self.ni, self.NG:self.NG + self.nj]
        speed = np.linalg.norm(u_np, axis=2)
        u_max = float(np.max(speed))
        h_min = float(np.min(self.cell_size.to_numpy()[self.NG:self.NG + self.ni, self.NG:self.NG + self.nj]))
        vel_scale = max(u_max, self.u_ref, 1e-6)
        dt = self.cfl * h_min / vel_scale
        self.dt_field[None] = dt

        if self._metric_mode == "cartesian":
            self._compute_divergence_cartesian(float(self._dx), float(self._dy))
            div_np = self.div.to_numpy()[self.NG:self.NG + self.ni, self.NG:self.NG + self.nj]
            div_linf = float(np.max(np.abs(div_np)))
        else:
            div_linf = np.nan

        return {
            "mode": "incompressible_skeleton",
            "metric_mode": self._metric_mode,
            "dt": float(self.dt_field[None]),
            "u_max": u_max,
            "h_min": h_min,
            "div_linf": div_linf,
            "projection_iters": int(self._projection_last_iters),
            "projection_residual": float(self._projection_last_residual),
        }

    def get_diagnostics(self) -> dict[str, float | str]:
        """
        回傳結構化診斷量。
        """
        fields = self.get_fields()
        dt_diag = self.get_dt_diagnostics()
        return {
            **dt_diag,
            "p_min": float(np.min(fields["p"])),
            "p_max": float(np.max(fields["p"])),
            "u_max": float(np.max(np.linalg.norm(fields["u"], axis=2))),
        }

    def step(self):
        """
        執行一個不可壓縮時間步。

        Why:
        - 第一版先建立 predictor + pressure Poisson + velocity correction 的完整鏈
        - 目前只支援 Cartesian uniform grid；curvilinear 與完整對流項後續再補
        """
        if not self._grid_ready:
            raise RuntimeError("Grid is not ready. Call set_cartesian_grid() or set_curvilinear_grid() first.")
        if not self._state_ready:
            raise RuntimeError("State is not initialized. Call init_uniform() or init_from_primitive_numpy() first.")
        if self._metric_mode != "cartesian":
            raise NotImplementedError(
                "IncompressibleNavierStokesSolver step() currently supports Cartesian grids only."
            )
        return self._project_cartesian_step()
