"""
Solver Descriptors
==================

What:
- 定義 Grid、BoundaryCondition 與 SolverControl 的描述子資料結構

Why:
- 把「如何配置 solver」從 case script 與 solver instance method 解耦
- 讓同一份設定可以被不同 solver family 的 adapter 解讀
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridDescriptor:
    """
    網格描述子。
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def cartesian(cls, dx: float, dy: float) -> "GridDescriptor":
        return cls(kind="cartesian", params={"dx": float(dx), "dy": float(dy)})

    @classmethod
    def curvilinear(cls, x_node: np.ndarray, y_node: np.ndarray) -> "GridDescriptor":
        return cls(
            kind="curvilinear",
            params={
                "x_node": np.asarray(x_node, dtype=np.float32),
                "y_node": np.asarray(y_node, dtype=np.float32),
            },
        )

    @classmethod
    def lattice(cls, nx: int, ny: int) -> "GridDescriptor":
        return cls(kind="lattice", params={"nx": int(nx), "ny": int(ny)})


@dataclass(frozen=True)
class BoundaryConditionDescriptor:
    """
    邊界條件描述子。
    """

    kind: str
    location: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def periodic(cls, direction: str) -> "BoundaryConditionDescriptor":
        return cls(kind="periodic", location=direction)

    @classmethod
    def no_slip(
        cls,
        location: str,
        u_wall: float = 0.0,
        v_wall: float = 0.0,
        temperature: float | None = None,
        exclude_corners: bool = False,
    ) -> "BoundaryConditionDescriptor":
        return cls(
            kind="no_slip",
            location=location,
            params={
                "u_wall": float(u_wall),
                "v_wall": float(v_wall),
                "temperature": temperature,
                "exclude_corners": bool(exclude_corners),
            },
        )

    @classmethod
    def free_slip(cls, location: str) -> "BoundaryConditionDescriptor":
        return cls(kind="free_slip", location=location)

    @classmethod
    def moving_wall(
        cls,
        location: str,
        velocity_profile: np.ndarray,
        handle_corners: bool = True,
        corner_mode: str = "generic",
    ) -> "BoundaryConditionDescriptor":
        return cls(
            kind="moving_wall",
            location=location,
            params={
                "velocity_profile": np.asarray(velocity_profile, dtype=np.float32),
                "handle_corners": bool(handle_corners),
                "corner_mode": corner_mode,
            },
        )

    @classmethod
    def velocity_inlet(
        cls,
        location: str,
        u_in: float,
        epsilon: float = 0.02,
        omega: float | None = None,
        strouhal: float = 0.2,
        asymmetry: float = 0.0,
        method: str = "neq",
    ) -> "BoundaryConditionDescriptor":
        return cls(
            kind="velocity_inlet",
            location=location,
            params={
                "u_in": float(u_in),
                "epsilon": float(epsilon),
                "omega": omega,
                "strouhal": float(strouhal),
                "asymmetry": float(asymmetry),
                "method": method,
            },
        )

    @classmethod
    def stable_outlet(
        cls,
        location: str,
        rho_out: float = 1.0,
        relaxation: float = 0.02,
    ) -> "BoundaryConditionDescriptor":
        return cls(
            kind="stable_outlet",
            location=location,
            params={"rho_out": float(rho_out), "relaxation": float(relaxation)},
        )

    @classmethod
    def orlanski_outflow(
        cls,
        location: str,
        rho_target: float = 1.0,
        relaxation: float = 0.02,
    ) -> "BoundaryConditionDescriptor":
        return cls(
            kind="orlanski_outflow",
            location=location,
            params={"rho_target": float(rho_target), "relaxation": float(relaxation)},
        )

    @classmethod
    def body_force(cls, fx: float = 0.0, fy: float = 0.0) -> "BoundaryConditionDescriptor":
        return cls(kind="body_force", params={"fx": float(fx), "fy": float(fy)})


@dataclass(frozen=True)
class SolverControlDescriptor:
    """
    solver 控制描述子。

    What:
    - 表達與邊界條件不同的 solver 行為控制，例如重建階數、pseudo-time、nozzle BC

    Why:
    - 這些控制原本散落在案例內直接呼叫 solver method，會讓 toolkit workflow 只剩外殼
    - 應提升為正式配置語言，讓 CaseRunner / registry 能完整表達 compressible workflow
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def spatial_reconstruction(cls, order: str = "second") -> "SolverControlDescriptor":
        return cls(kind="spatial_reconstruction", params={"order": str(order)})

    @classmethod
    def time_marching(cls, mode: str = "global") -> "SolverControlDescriptor":
        return cls(kind="time_marching", params={"mode": str(mode)})

    @classmethod
    def pseudo_time_controls(
        cls,
        *,
        cfl_start: float | None = None,
        ramp_steps: int = 0,
        precond_ref_mach: float = 0.3,
        precond_min_scale: float = 0.25,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="pseudo_time_controls",
            params={
                "cfl_start": cfl_start,
                "ramp_steps": int(ramp_steps),
                "precond_ref_mach": float(precond_ref_mach),
                "precond_min_scale": float(precond_min_scale),
            },
        )

    @classmethod
    def residual_smoothing(
        cls,
        *,
        epsilon: float = 0.0,
        passes: int = 0,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="residual_smoothing",
            params={"epsilon": float(epsilon), "passes": int(passes)},
        )

    @classmethod
    def adaptive_pseudo_strategy(
        cls,
        *,
        enabled: bool = False,
        cfl_min: float | None = None,
        cfl_max: float | None = None,
        growth: float = 1.03,
        shrink: float = 0.70,
        target_ratio: float = 0.995,
        fail_ratio: float = 1.010,
        interval: int = 20,
        smoothing_max_eps: float | None = None,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="adaptive_pseudo_strategy",
            params={
                "enabled": bool(enabled),
                "cfl_min": None if cfl_min is None else float(cfl_min),
                "cfl_max": None if cfl_max is None else float(cfl_max),
                "growth": float(growth),
                "shrink": float(shrink),
                "target_ratio": float(target_ratio),
                "fail_ratio": float(fail_ratio),
                "interval": int(interval),
                "smoothing_max_eps": None if smoothing_max_eps is None else float(smoothing_max_eps),
            },
        )

    @classmethod
    def turbulence_model(
        cls,
        *,
        model: str = "laminar",
        eddy_viscosity_ratio: float = 0.0,
        turbulent_prandtl: float = 0.9,
        sa_nu_tilde_inf_ratio: float = 3.0,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="turbulence_model",
            params={
                "model": str(model),
                "eddy_viscosity_ratio": float(eddy_viscosity_ratio),
                "turbulent_prandtl": float(turbulent_prandtl),
                "sa_nu_tilde_inf_ratio": float(sa_nu_tilde_inf_ratio),
            },
        )

    @classmethod
    def nozzle_inlet(
        cls,
        *,
        p0: float,
        t0: float,
        mach_in: float,
        relax: float = 0.35,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="nozzle_inlet",
            params={
                "p0": float(p0),
                "t0": float(t0),
                "mach_in": float(mach_in),
                "relax": float(relax),
            },
        )

    @classmethod
    def nozzle_outlet(
        cls,
        *,
        back_pressure: float,
        relax: float = 0.35,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="nozzle_outlet",
            params={"back_pressure": float(back_pressure), "relax": float(relax)},
        )

    @classmethod
    def far_field(
        cls,
        *,
        location: str,
        rho: float,
        u: float,
        v: float,
        p: float,
    ) -> "SolverControlDescriptor":
        return cls(
            kind="far_field",
            params={
                "location": str(location),
                "rho": float(rho),
                "u": float(u),
                "v": float(v),
                "p": float(p),
            },
        )
