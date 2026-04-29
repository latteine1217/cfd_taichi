"""
共同 solver protocol 與 descriptor adapter 測試
==============================================

What:
- 驗證 `SolverProtocol`、`GridDescriptor`、`BoundaryConditionDescriptor`
  與 adapter 層可套用到現有 LBM / FVM solver

Why:
- 這些結構層是工具化架構的共同語言；若不能驅動現有 solver，就只是空殼
"""

import numpy as np
import taichi as ti

from cfd_taichi import (
    BoundaryConditionDescriptor,
    GridDescriptor,
    SolverProtocol,
    apply_boundary_descriptors,
    apply_grid_descriptor,
    apply_solver_control_descriptors,
    SolverControlDescriptor,
)


def test_compressible_fvm_matches_solver_protocol():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import CompressibleNavierStokesSolver

    solver = CompressibleNavierStokesSolver(
        ni=8,
        nj=6,
        re=100.0,
        pr=0.72,
        u_ref=0.1,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8, dy=1.0 / 6)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.1, temperature=1.0)
    solver.init_uniform(rho=1.0, u=0.0, v=0.0, p=1.0)

    assert isinstance(solver, SolverProtocol)
    fields = solver.get_fields()
    diag = solver.get_diagnostics()

    assert fields["u"].shape == (8, 6, 2)
    assert fields["rho"].shape == (8, 6)
    assert "mach" in fields
    assert diag["u_max"] >= 0.0


def test_incompressible_fvm_descriptor_adapter_configures_solver():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import IncompressibleNavierStokesSolver

    solver = IncompressibleNavierStokesSolver(
        ni=8,
        nj=6,
        re=100.0,
        u_ref=0.1,
        length_scale=1.0,
    )
    apply_grid_descriptor(solver, GridDescriptor.cartesian(dx=1.0 / 8, dy=1.0 / 6))
    apply_boundary_descriptors(
        solver,
        [
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.no_slip("top", u_wall=0.1),
            BoundaryConditionDescriptor.body_force(fx=1e-3, fy=0.0),
        ],
    )
    solver.init_uniform(u=0.0, v=0.0, p=0.0)
    solver.step()

    assert isinstance(solver, SolverProtocol)
    diag = solver.get_diagnostics()
    fields = solver.get_fields()

    assert diag["metric_mode"] == "cartesian"
    assert diag["projection_iters"] > 0
    assert float(fields["u"][:, :, 0].mean()) > 0.0


def test_lbm_descriptor_adapter_builds_boundary_manager():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import BoundaryConditions, LBMSolver

    solver = LBMSolver(
        nx=8,
        ny=8,
        re=100.0,
        u_ref=0.05,
        length_scale=8.0,
        cs=-1.0,
    )
    apply_grid_descriptor(solver, GridDescriptor.lattice(nx=8, ny=8))
    bc = apply_boundary_descriptors(
        solver,
        [
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
    )

    u0 = np.zeros((8, 8, 2), dtype=np.float32)
    solver.set_initial_condition(velocity=u0, apply_boundaries=True, reset_baseline=True)
    solver.prepare_diagnostics(reset_baseline=True)

    assert isinstance(solver, SolverProtocol)
    assert isinstance(bc, BoundaryConditions)
    assert np.count_nonzero(solver.mask.to_numpy()) > 0
    assert solver.get_fields()["u"].shape == (8, 8, 2)


def test_compressible_fvm_solver_control_adapter_configures_nozzle_controls():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import CompressibleEulerSolver, generate_reference_nozzle_grid

    solver = CompressibleEulerSolver(ni=20, nj=8, gamma=1.4, cfl=0.2)
    x_node, y_node = generate_reference_nozzle_grid(
        ni=20,
        nj=8,
        length=3.0,
        h_inlet=1.0,
        h_throat=0.7,
        h_exit=1.0,
        throat_x=None,
        conv_power=2.4,
        div_power=3.0,
        throat_blend=0.35,
        NG=solver.NG,
    )
    apply_grid_descriptor(solver, GridDescriptor.curvilinear(x_node=x_node, y_node=y_node))
    apply_boundary_descriptors(
        solver,
        [
            BoundaryConditionDescriptor.free_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
    )
    apply_solver_control_descriptors(
        solver,
        [
            SolverControlDescriptor.nozzle_inlet(p0=0.75, t0=0.72, mach_in=0.12, relax=0.35),
            SolverControlDescriptor.nozzle_outlet(back_pressure=0.73, relax=0.35),
            SolverControlDescriptor.spatial_reconstruction("first"),
            SolverControlDescriptor.time_marching("local_pseudo"),
            SolverControlDescriptor.pseudo_time_controls(
                cfl_start=0.05,
                ramp_steps=10,
                precond_ref_mach=0.2,
                precond_min_scale=0.2,
            ),
            SolverControlDescriptor.residual_smoothing(epsilon=0.05, passes=1),
            SolverControlDescriptor.adaptive_pseudo_strategy(
                enabled=False,
                cfl_min=0.05,
                cfl_max=0.2,
                growth=1.03,
                shrink=0.7,
                target_ratio=0.995,
                fail_ratio=1.01,
                interval=20,
                smoothing_max_eps=0.24,
            ),
        ],
    )
    solver.init_uniform(rho=1.0, u=0.12, v=0.0, p=1.0 / 1.4)
    solver._update_ghost()
    solver.step()

    diag = solver.get_diagnostics()

    assert diag["mode"] == "local_pseudo"
    assert diag["pseudo_cfl"] > 0.0
    assert diag["u_max"] > 0.0


def test_multiphase_lbm_matches_solver_protocol():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import MultiphaseLBMSolver

    solver = MultiphaseLBMSolver(
        nx=16,
        ny=16,
        tau_a=0.8,
        tau_b=0.8,
        g_interaction=3.5,
    )
    rho_a = np.ones((16, 16), dtype=np.float32)
    rho_b = np.full((16, 16), 0.1, dtype=np.float32)
    solver.set_initial_fields(rho_a, rho_b)

    assert isinstance(solver, SolverProtocol)

    fields = solver.get_fields()
    assert "rhoA" in fields
    assert "u" in fields

    diag = solver.get_diagnostics()
    assert "step_count" in diag
    assert "u_max" in diag
    assert diag["u_max"] >= 0.0
