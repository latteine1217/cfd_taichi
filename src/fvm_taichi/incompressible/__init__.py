"""
fvm_taichi.incompressible
=========================

What:
- 提供不可壓縮 FVM solver 的命名空間

Why:
- 專案對外語義需區分 solver family 與流動體制
- 不可壓縮 FVM 已有 Navier-Stokes solver 骨架，後續可在同一命名空間擴充
"""

from .navier_stokes_solver import IncompressibleNavierStokesSolver
from .poisson_solver import JacobiPressurePoissonSolver


__all__ = ["IncompressibleNavierStokesSolver", "JacobiPressurePoissonSolver"]
