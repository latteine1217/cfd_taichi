"""
不可壓縮壓力 Poisson 求解器
===========================

What:
- 提供 projection method 所需的獨立 Poisson 線性求解子模組

Why:
- 壓力修正方程不應內嵌在 Navier-Stokes solver 本體
- 未來若要切換 Jacobi / SOR / multigrid，應只替換此子模組

When:
- incompressible Navier-Stokes 的 pressure projection
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class JacobiPressurePoissonSolver:
    """
    2D Cartesian Poisson 的 Jacobi 求解器。

    What:
    - 解 `∇²p = rhs`

    Why:
    - Jacobi 完全平行且結構簡單，適合作為第一版可驗證基線
    """

    def __init__(
        self,
        *,
        ng: int,
        ni: int,
        nj: int,
        max_iters: int = 200,
        tol: float = 1e-6,
    ):
        if ng <= 0 or ni <= 0 or nj <= 0:
            raise ValueError(f"Invalid grid shape ng={ng}, ni={ni}, nj={nj}")
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")

        self.ng = int(ng)
        self.ni = int(ni)
        self.nj = int(nj)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def set_controls(self, *, max_iters: int | None = None, tol: float | None = None):
        if max_iters is not None:
            if max_iters <= 0:
                raise ValueError(f"max_iters must be positive, got {max_iters}")
            self.max_iters = int(max_iters)
        if tol is not None:
            if tol <= 0.0:
                raise ValueError(f"tol must be positive, got {tol}")
            self.tol = float(tol)

    def solve(
        self,
        *,
        rhs: np.ndarray,
        p_seed: np.ndarray,
        dx: float,
        dy: float,
        apply_pressure_bc: Callable[[np.ndarray], None],
    ) -> tuple[np.ndarray, int, float]:
        """
        求解 pressure correction。

        What:
        - 使用 Jacobi 迭代更新 interior，再由 callback 套用 BC

        Why:
        - BC 語義屬於 flow solver；線性求解器只負責 interior stencil
        """
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")

        rhs_arr = np.asarray(rhs, dtype=np.float32)
        p_old = np.asarray(p_seed, dtype=np.float32).copy()
        if rhs_arr.shape != p_old.shape:
            raise ValueError(
                f"rhs and p_seed must share shape, got {rhs_arr.shape} and {p_old.shape}"
            )

        p_new = p_old.copy()
        dx2 = float(dx * dx)
        dy2 = float(dy * dy)
        coef = 1.0 / (2.0 * (dx2 + dy2))
        ng = self.ng
        interior = (slice(ng, ng + self.ni), slice(ng, ng + self.nj))
        residual = np.inf

        for it in range(1, self.max_iters + 1):
            p_new[interior] = coef * (
                (
                    p_old[ng + 1:ng + self.ni + 1, ng:ng + self.nj]
                    + p_old[ng - 1:ng + self.ni - 1, ng:ng + self.nj]
                )
                * dy2
                + (
                    p_old[ng:ng + self.ni, ng + 1:ng + self.nj + 1]
                    + p_old[ng:ng + self.ni, ng - 1:ng + self.nj - 1]
                )
                * dx2
                - rhs_arr[interior] * dx2 * dy2
            )
            apply_pressure_bc(p_new)
            residual = float(np.max(np.abs(p_new - p_old)))
            p_old, p_new = p_new, p_old
            if residual < self.tol:
                return p_old, it, residual

        return p_old, self.max_iters, residual
