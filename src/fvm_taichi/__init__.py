"""
fvm_taichi — 有限體積求解器工具集
==================================

與 lbm_taichi 並列的獨立模組，使用 FVM 方法求解 Euler / Navier-Stokes 類方程。

目前已實作：
- CompressibleEulerSolver         : 2D 可壓縮 Euler FV 求解器（HLLC + MUSCL + SSP-RK3）
- CompressibleNavierStokesSolver  : 2D 可壓縮層流 NS 求解器（Cartesian v1）

相容 alias：
- EulerSolver                     : = CompressibleEulerSolver
- NavierStokesSolver              : = CompressibleNavierStokesSolver

說明：
- `compressible` 是 FVM 內的體制選項，不再是整個 `fvm_taichi` 的唯一語義
- incompressible FVM 已有 `IncompressibleNavierStokesSolver` 骨架
- 目前 `step()` 支援 Cartesian uniform-grid 的第一版 pressure projection scaffold

其他元件：
- generate_o_grid_naca0012 : NACA0012 代數 O-grid 生成器
- generate_sheared_channel_grid : 剪切通道網格（curvilinear benchmark）
- generate_bump_channel_grid : transonic bump channel 網格
- generate_cd_nozzle_grid : converging-diverging nozzle 網格
- generate_reference_nozzle_grid : 參考示意圖風格 nozzle 網格
- plot_cp                  : 翼型表面 Cp 分佈圖
- plot_flow_field           : 流場等高線圖（壓力/密度/馬赫數）
- plot_cell_scalar_field   : 任意 cell scalar 等高線圖
- plot_cl_aoa              : CL vs AOA 曲線圖
- plot_coeff_vs_aoa        : 黏性案例係數-AOA 曲線
- plot_coeff_vs_re         : 黏性案例係數-Re 曲線
- plot_force_polar         : 黏性案例升阻極線圖
"""

from .euler_solver import EulerSolver as CompressibleEulerSolver
from .navier_stokes_solver import (
    NavierStokesSolver as CompressibleNavierStokesSolver,
)
from .incompressible import (
    IncompressibleNavierStokesSolver,
    JacobiPressurePoissonSolver,
)
from .grid_gen import (
    generate_o_grid_naca0012,
    generate_sheared_channel_grid,
    generate_bump_channel_grid,
    generate_cd_nozzle_grid,
    generate_reference_nozzle_grid,
)
from .plot import (
    plot_cp,
    plot_surface_cp_cf,
    plot_wall_shear_distribution,
    plot_wall_unit_distribution,
    plot_flow_field,
    plot_cell_scalar_field,
    plot_cl_aoa,
    plot_coeff_vs_aoa,
    plot_coeff_vs_re,
    plot_force_polar,
    make_flow_gif,
)

EulerSolver = CompressibleEulerSolver
NavierStokesSolver = CompressibleNavierStokesSolver

__all__ = [
    "CompressibleEulerSolver",
    "CompressibleNavierStokesSolver",
    "IncompressibleNavierStokesSolver",
    "JacobiPressurePoissonSolver",
    "EulerSolver",
    "NavierStokesSolver",
    "generate_o_grid_naca0012",
    "generate_sheared_channel_grid",
    "generate_bump_channel_grid",
    "generate_cd_nozzle_grid",
    "generate_reference_nozzle_grid",
    "plot_cp",
    "plot_surface_cp_cf",
    "plot_wall_shear_distribution",
    "plot_wall_unit_distribution",
    "plot_flow_field",
    "plot_cell_scalar_field",
    "plot_cl_aoa",
    "plot_coeff_vs_aoa",
    "plot_coeff_vs_re",
    "plot_force_polar",
    "make_flow_gif",
]
