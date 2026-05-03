"""
fvm_taichi.compressible
=======================

What:
- 匯出目前已實作的可壓縮 FVM solver

Why:
- 將「FVM」與「compressible / incompressible」兩個維度拆開
- 讓可壓縮性成為顯式選項，而不是 `fvm_taichi` 的唯一預設語義
"""

from ..euler_solver import EulerSolver as CompressibleEulerSolver
from ..navier_stokes_solver import (
    NavierStokesSolver as CompressibleNavierStokesSolver,
)

__all__ = [
    "CompressibleEulerSolver",
    "CompressibleNavierStokesSolver",
]
