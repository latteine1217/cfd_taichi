"""
Solver Factory
==============

What:
- 提供專案層級的 solver 建立入口，顯式區分 family / equation / regime

Why:
- `lbm` 與 `fvm` 是第一層分類；`compressible` 只應是 FVM 內的體制選項
- 既有相容 alias 可以保留，但新入口必須避免把「FVM = 可壓縮」寫死
"""

from typing import Any

from fvm_taichi import (
    CompressibleEulerSolver,
    IncompressibleNavierStokesSolver,
    CompressibleNavierStokesSolver,
)
from lbm_taichi import CHLBMSolver, LBMSolver, MultiphaseLBMSolver


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def create_solver(
    method: str,
    equation: str | None = None,
    regime: str | None = None,
    **kwargs: Any,
):
    """
    建立 solver 實例。

    What:
    - 依照 `method` / `equation` / `regime` 回傳對應求解器

    Why:
    - 對新 API 強制要求 FVM 明確指定流動體制，避免再把 compressible 當成隱性預設
    - 保留舊 alias 與既有案例相容，但引導新程式碼走語義清楚的入口

    When:
    - 新案例
    - 交互式 benchmark / notebook
    - 之後擴充 incompressible FVM 時的統一入口
    """
    method_key = _normalize(method)
    equation_key = _normalize(equation)
    regime_key = _normalize(regime)

    if method_key == "lbm":
        if equation_key in (
            None,
            "single_phase",
            "single-phase",
            "navier_stokes",
            "thermal_boussinesq",
            "thermal-boussinesq",
        ):
            return LBMSolver(**kwargs)
        if equation_key in ("multiphase", "multi_phase", "two_phase", "two-phase"):
            return MultiphaseLBMSolver(**kwargs)
        if equation_key in ("cahn_hilliard", "cahn-hilliard", "ch"):
            return CHLBMSolver(**kwargs)
        raise ValueError(
            "Unsupported LBM equation. Choose from: single_phase, multiphase, cahn_hilliard."
        )

    if method_key != "fvm":
        raise ValueError("Unsupported method. Choose 'lbm' or 'fvm'.")

    if equation_key is None:
        raise ValueError(
            "FVM solver selection requires explicit equation='euler' or 'navier_stokes'."
        )
    if regime_key is None:
        raise ValueError(
            "FVM solver selection requires explicit regime='compressible' or 'incompressible'."
        )

    if regime_key == "compressible":
        if equation_key == "euler":
            return CompressibleEulerSolver(**kwargs)
        if equation_key in ("navier_stokes", "navier-stokes", "ns"):
            return CompressibleNavierStokesSolver(**kwargs)
        raise ValueError(
            "Unsupported compressible FVM equation. Choose 'euler' or 'navier_stokes'."
        )

    if regime_key == "incompressible":
        if equation_key == "euler":
            raise ValueError("Incompressible Euler is not exposed in this toolset.")
        if equation_key in ("navier_stokes", "navier-stokes", "ns"):
            return IncompressibleNavierStokesSolver(**kwargs)
        raise ValueError(
            "Unsupported incompressible FVM equation. Choose 'navier_stokes'."
        )

    raise ValueError("Unsupported regime. Choose 'compressible' or 'incompressible'.")
