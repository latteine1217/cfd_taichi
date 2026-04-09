"""
Solver Configuration Adapters
=============================

What:
- 將 Grid / BoundaryCondition / SolverControl descriptor 套用到既有 solver instance

Why:
- 現有 solver API 仍保留 family-specific method；adapter 讓 case layer 先脫離這些細節
- 先建立配置面，再逐步收斂 solver class 本體
"""

from collections.abc import Sequence
from typing import Any

from lbm_taichi import BoundaryConditions

from .descriptors import (
    BoundaryConditionDescriptor,
    GridDescriptor,
    SolverControlDescriptor,
)


def _solver_family(solver: Any) -> str:
    family = getattr(solver, "solver_family", None)
    if family is None:
        raise TypeError(f"Solver {type(solver).__name__} has no solver_family metadata.")
    return str(family)


def apply_grid_descriptor(solver: Any, descriptor: Any) -> Any:
    """
    將 grid descriptor 套用到 solver。
    """
    if hasattr(descriptor, "to_descriptor"):
        descriptor = descriptor.to_descriptor()
    if not isinstance(descriptor, GridDescriptor):
        raise TypeError(
            "descriptor must be a GridDescriptor or Grid2D-like object with to_descriptor()."
        )
    kind = descriptor.kind

    if kind == "lattice":
        nx = descriptor.params["nx"]
        ny = descriptor.params["ny"]
        if getattr(solver, "nx", None) != nx or getattr(solver, "ny", None) != ny:
            raise ValueError(
                f"Lattice descriptor ({nx}, {ny}) incompatible with solver "
                f"({getattr(solver, 'nx', None)}, {getattr(solver, 'ny', None)})"
            )
        return solver

    if kind == "cartesian":
        if not hasattr(solver, "set_cartesian_grid"):
            raise NotImplementedError(f"{type(solver).__name__} does not support cartesian grids.")
        solver.set_cartesian_grid(
            dx=descriptor.params["dx"],
            dy=descriptor.params["dy"],
        )
        return solver

    if kind == "curvilinear":
        if not hasattr(solver, "set_curvilinear_grid"):
            raise NotImplementedError(f"{type(solver).__name__} does not support curvilinear grids.")
        solver.set_curvilinear_grid(
            descriptor.params["x_node"],
            descriptor.params["y_node"],
        )
        return solver

    raise ValueError(f"Unsupported grid descriptor kind: {kind}")


def apply_boundary_descriptors(
    solver: Any,
    descriptors: Sequence[BoundaryConditionDescriptor],
):
    """
    將 boundary descriptors 套用到 solver。

    回傳：
    - LBM: `BoundaryConditions` manager
    - FVM: solver 自身
    """
    family = _solver_family(solver)

    if family == "lbm":
        bc = BoundaryConditions(solver)
        for descriptor in descriptors:
            kind = descriptor.kind
            loc = descriptor.location
            params = descriptor.params

            if kind == "no_slip":
                bc.add_no_slip_wall(
                    loc,
                    exclude_corners=bool(params.get("exclude_corners", False)),
                )
            elif kind == "free_slip":
                bc.add_free_slip_wall(loc)
            elif kind == "moving_wall":
                bc.add_moving_wall(
                    params["velocity_profile"],
                    location=loc if loc is not None else "top",
                    handle_corners=bool(params.get("handle_corners", True)),
                    corner_mode=params.get("corner_mode", "generic"),
                )
            elif kind == "velocity_inlet":
                bc.add_velocity_inlet(
                    u_in=params["u_in"],
                    location=loc if loc is not None else "left",
                    epsilon=params.get("epsilon", 0.02),
                    omega=params.get("omega"),
                    strouhal=params.get("strouhal", 0.2),
                    asymmetry=params.get("asymmetry", 0.0),
                    method=params.get("method", "neq"),
                )
            elif kind == "stable_outlet":
                bc.add_stable_outlet(
                    rho_out=params.get("rho_out", 1.0),
                    location=loc if loc is not None else "right",
                    relaxation=params.get("relaxation", 0.02),
                )
            elif kind == "orlanski_outflow":
                bc.add_orlanski_outflow(
                    location=loc if loc is not None else "right",
                    rho_target=params.get("rho_target", 1.0),
                    relaxation=params.get("relaxation", 0.02),
                )
            elif kind == "periodic":
                bc.add_periodic_boundary(loc if loc is not None else "x")
            else:
                raise NotImplementedError(f"LBM boundary kind '{kind}' is not supported by adapter.")
        return bc

    if family == "fvm":
        for descriptor in descriptors:
            kind = descriptor.kind
            loc = descriptor.location
            params = descriptor.params

            if kind == "periodic":
                current_i = bool(getattr(solver, "_periodic_i", False))
                current_j = bool(getattr(solver, "_periodic_j", False))
                direction = loc if loc is not None else "x"
                if direction == "x":
                    solver.set_periodic_bc(i_dir=True, j_dir=current_j)
                elif direction == "y":
                    solver.set_periodic_bc(i_dir=current_i, j_dir=True)
                else:
                    raise ValueError(f"Invalid periodic direction: {direction}")
            elif kind == "body_force":
                if not hasattr(solver, "set_body_force"):
                    raise NotImplementedError(f"{type(solver).__name__} does not support body_force.")
                solver.set_body_force(
                    fx=params.get("fx", 0.0),
                    fy=params.get("fy", 0.0),
                )
            elif kind == "free_slip":
                location = loc if loc is not None else "bottom"
                if location in {"bottom", "j_min"}:
                    if not hasattr(solver, "set_slip_wall_j_min"):
                        raise NotImplementedError(f"{type(solver).__name__} does not support j_min free-slip walls.")
                    solver.set_slip_wall_j_min()
                elif location in {"top", "j_max"}:
                    if not hasattr(solver, "set_slip_wall_j_max"):
                        raise NotImplementedError(f"{type(solver).__name__} does not support j_max free-slip walls.")
                    solver.set_slip_wall_j_max()
                else:
                    raise NotImplementedError(
                        "FVM free-slip adapter currently supports 'bottom'/'j_min' and 'top'/'j_max' only."
                    )
            elif kind == "no_slip":
                if getattr(solver, "regime", None) == "compressible":
                    solver.set_no_slip_wall(
                        loc if loc is not None else "bottom",
                        u_wall=params.get("u_wall", 0.0),
                        v_wall=params.get("v_wall", 0.0),
                        temperature=params.get("temperature"),
                    )
                else:
                    solver.set_no_slip_wall(
                        loc if loc is not None else "bottom",
                        u_wall=params.get("u_wall", 0.0),
                        v_wall=params.get("v_wall", 0.0),
                    )
            else:
                raise NotImplementedError(f"FVM boundary kind '{kind}' is not supported by adapter.")
        return solver

    raise NotImplementedError(f"Unsupported solver family for boundary adapter: {family}")


def apply_solver_control_descriptors(
    solver: Any,
    descriptors: Sequence[SolverControlDescriptor],
):
    """
    將 solver control descriptors 套用到 solver。

    What:
    - 把時間推進、重建階數與 nozzle characteristic BC 等控制集中到正式 adapter

    Why:
    - compressible benchmark 若仍需在 initializer 內直接呼叫 solver method，就還不算 toolkit 化
    """
    family = _solver_family(solver)
    if family != "fvm":
        raise NotImplementedError(
            f"Solver controls are currently only supported for FVM solvers, got {family!r}."
        )

    for descriptor in descriptors:
        kind = descriptor.kind
        params = descriptor.params

        if kind == "spatial_reconstruction":
            if not hasattr(solver, "set_spatial_reconstruction"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support spatial reconstruction controls."
                )
            solver.set_spatial_reconstruction(params["order"])
        elif kind == "time_marching":
            if not hasattr(solver, "set_time_marching_mode"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support time-marching controls."
                )
            solver.set_time_marching_mode(params["mode"])
        elif kind == "pseudo_time_controls":
            if not hasattr(solver, "set_pseudo_time_controls"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support pseudo-time controls."
                )
            solver.set_pseudo_time_controls(
                cfl_start=params.get("cfl_start"),
                ramp_steps=params.get("ramp_steps", 0),
                precond_ref_mach=params.get("precond_ref_mach", 0.3),
                precond_min_scale=params.get("precond_min_scale", 0.25),
            )
        elif kind == "residual_smoothing":
            if not hasattr(solver, "set_residual_smoothing"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support residual smoothing controls."
                )
            solver.set_residual_smoothing(
                epsilon=params.get("epsilon", 0.0),
                passes=params.get("passes", 0),
            )
        elif kind == "adaptive_pseudo_strategy":
            if not hasattr(solver, "set_adaptive_pseudo_strategy"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support adaptive pseudo controls."
                )
            solver.set_adaptive_pseudo_strategy(
                enabled=params.get("enabled", False),
                cfl_min=params["cfl_min"],
                cfl_max=params["cfl_max"],
                growth=params.get("growth", 1.03),
                shrink=params.get("shrink", 0.70),
                target_ratio=params.get("target_ratio", 0.995),
                fail_ratio=params.get("fail_ratio", 1.010),
                interval=params.get("interval", 20),
                smoothing_max_eps=params.get("smoothing_max_eps", 0.24),
            )
        elif kind == "nozzle_inlet":
            if not hasattr(solver, "set_nozzle_inlet_i_min"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support nozzle inlet controls."
                )
            solver.set_nozzle_inlet_i_min(
                p0=params["p0"],
                t0=params["t0"],
                mach_in=params["mach_in"],
                relax=params.get("relax", 0.35),
            )
        elif kind == "nozzle_outlet":
            if not hasattr(solver, "set_nozzle_outlet_i_max"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support nozzle outlet controls."
                )
            solver.set_nozzle_outlet_i_max(
                back_pressure=params["back_pressure"],
                relax=params.get("relax", 0.35),
            )
        elif kind == "far_field":
            if not hasattr(solver, "set_far_field_j_max"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support far-field controls."
                )
            location = params.get("location", "j_max")
            if location not in {"top", "j_max"}:
                raise NotImplementedError(
                    "FVM far-field adapter currently supports 'top'/'j_max' only."
                )
            solver.set_far_field_j_max(
                params["rho"],
                params["u"],
                params["v"],
                params["p"],
            )
        elif kind == "turbulence_model":
            if not hasattr(solver, "set_turbulence_model"):
                raise NotImplementedError(
                    f"{type(solver).__name__} does not support turbulence-model controls."
                )
            solver.set_turbulence_model(
                model=params.get("model", "laminar"),
                eddy_viscosity_ratio=params.get("eddy_viscosity_ratio", 0.0),
                turbulent_prandtl=params.get("turbulent_prandtl", 0.9),
                sa_nu_tilde_inf_ratio=params.get("sa_nu_tilde_inf_ratio", 3.0),
            )
        else:
            raise NotImplementedError(f"Solver control kind '{kind}' is not supported by adapter.")
    return solver
