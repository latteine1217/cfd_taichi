"""
Solver factory 與命名空間測試
============================

What:
- 驗證 `cfd_taichi.create_solver()` 與 FVM regime 命名空間的最小可用性

Why:
- 新 API 必須明確區分 solver family 與 compressibility
- 舊 alias 仍需保持相容，避免既有案例被破壞
"""

import pytest
import taichi as ti
import numpy as np

from cfd_taichi import create_solver


def test_fvm_factory_requires_explicit_regime():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="explicit regime"):
        create_solver(method="fvm", equation="navier_stokes")


def test_fvm_factory_returns_compressible_ns_solver():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import CompressibleNavierStokesSolver

    solver = create_solver(
        method="fvm",
        equation="navier_stokes",
        regime="compressible",
        ni=8,
        nj=6,
        re=100.0,
        pr=0.72,
        u_ref=0.1,
        length_scale=1.0,
    )

    assert isinstance(solver, CompressibleNavierStokesSolver)


def test_fvm_factory_returns_incompressible_ns_solver():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import IncompressibleNavierStokesSolver

    solver = create_solver(
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        ni=8,
        nj=6,
        re=100.0,
        u_ref=0.1,
        length_scale=1.0,
    )

    assert isinstance(solver, IncompressibleNavierStokesSolver)


def test_lbm_factory_returns_single_phase_solver():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import LBMSolver

    solver = create_solver(
        method="lbm",
        equation="single_phase",
        nx=8,
        ny=8,
        re=100.0,
        u_ref=0.05,
        length_scale=8.0,
        cs=-1.0,
    )

    assert isinstance(solver, LBMSolver)


def test_fvm_legacy_aliases_still_point_to_compressible_classes():
    from fvm_taichi import (
        CompressibleEulerSolver,
        CompressibleNavierStokesSolver,
        EulerSolver,
        NavierStokesSolver,
    )

    assert EulerSolver is CompressibleEulerSolver
    assert NavierStokesSolver is CompressibleNavierStokesSolver


def test_incompressible_ns_skeleton_supports_cartesian_projection_step():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import IncompressibleNavierStokesSolver

    solver = IncompressibleNavierStokesSolver(
        ni=8,
        nj=6,
        re=100.0,
        u_ref=0.1,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8, dy=1.0 / 6)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_body_force(fx=1e-3, fy=0.0)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0)
    solver.set_no_slip_wall("top", u_wall=0.1, v_wall=0.0)
    solver.init_uniform(u=0.0, v=0.0, p=0.0)

    diag0 = solver.get_dt_diagnostics()
    dt = solver.step()
    diag1 = solver.get_dt_diagnostics()
    u, v, p = solver.get_primitive()

    assert diag0["mode"] == "incompressible_skeleton"
    assert diag1["metric_mode"] == "cartesian"
    assert dt > 0.0
    assert diag1["dt"] > 0.0
    assert diag1["projection_iters"] > 0
    assert diag1["projection_residual"] >= 0.0
    assert u.shape == (8, 6)
    assert v.shape == (8, 6)
    assert p.shape == (8, 6)
    assert abs(diag1["div_linf"]) <= abs(diag0["div_linf"]) + 1e-5
    assert float(u.mean()) > 0.0
    assert np.isfinite(u).all()
    assert np.isfinite(v).all()
    assert np.isfinite(p).all()


def test_incompressible_ns_rest_state_stays_quiet_after_one_step():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from fvm_taichi import IncompressibleNavierStokesSolver

    solver = IncompressibleNavierStokesSolver(
        ni=8,
        nj=8,
        re=100.0,
        u_ref=0.1,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8, dy=1.0 / 8)
    solver.set_periodic_bc(i_dir=True, j_dir=True)
    solver.init_uniform(u=0.0, v=0.0, p=0.0)

    dt = solver.step()
    diag = solver.get_dt_diagnostics()
    u, v, p = solver.get_primitive()

    assert dt > 0.0
    assert np.max(np.abs(u)) < 1e-6
    assert np.max(np.abs(v)) < 1e-6
    assert np.max(np.abs(p)) < 1e-4
    assert abs(diag["div_linf"]) < 1e-6
