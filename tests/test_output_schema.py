"""
統一輸出 schema 與 Poisson 子模組測試
====================================

What:
- 驗證 state/history schema 已由 diagnostics 實際輸出
- 驗證 incompressible Poisson solver 已從主 solver 拆成獨立子模組

Why:
- 這兩層若沒有測試保護，很容易再次退回案例私有格式或 solver 內嵌線性代數
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from cfd_taichi.output_schema import build_state_payload


def test_lbm_diagnostics_save_data_writes_standard_state_payload(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import Diagnostics, LBMSolver

    solver = LBMSolver(
        nx=8,
        ny=8,
        re=100.0,
        u_ref=0.05,
        length_scale=8.0,
        cs=-1.0,
    )
    u0 = np.zeros((8, 8, 2), dtype=np.float32)
    solver.set_initial_condition(velocity=u0, apply_boundaries=True, reset_baseline=True)
    solver.prepare_diagnostics(reset_baseline=True)

    diag = Diagnostics(solver, output_dir=str(tmp_path))
    diag.save_data(3, additional_data={"time": 1.25})

    payload = np.load(tmp_path / "state_000003.npy", allow_pickle=True).item()

    assert payload["solver_family"] == "lbm"
    assert payload["equation_set"] == "navier_stokes"
    assert payload["regime"] == "low_mach"
    assert payload["step"] == 3
    assert payload["time"] == 1.25
    assert payload["u"].shape == (8, 8, 2)
    assert payload["rho"].shape == (8, 8)
    assert payload["mask"].shape == (8, 8)
    assert payload["vorticity"].shape == (8, 8)


def test_lbm_diagnostics_save_history_writes_standard_history_payload(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import Diagnostics, LBMSolver

    solver = LBMSolver(
        nx=8,
        ny=8,
        re=100.0,
        u_ref=0.05,
        length_scale=8.0,
        cs=-1.0,
    )
    u0 = np.zeros((8, 8, 2), dtype=np.float32)
    solver.set_initial_condition(velocity=u0, apply_boundaries=True, reset_baseline=True)
    solver.prepare_diagnostics(reset_baseline=True)

    diag = Diagnostics(solver, output_dir=str(tmp_path))
    headers = diag.print_header(include_forces=False)
    row = diag.print_step_info(10, speed=1.0, eta_seconds=0.0, f_field=solver.f)
    diag.history.append(row)
    history_path = diag.save_history(params={"case": "smoke"})

    payload = np.load(history_path, allow_pickle=True).item()

    assert payload["solver_family"] == "lbm"
    assert payload["steps"].tolist() == [10]
    assert payload["mass_error"].shape == (1,)
    assert payload["mom_res_x"].shape == (1,)
    assert payload["mom_res_y"].shape == (1,)
    assert payload["u_max"].shape == (1,)
    assert payload["cfl"].shape == (1,)
    assert payload["headers"] == headers
    assert payload["data"] == diag.history
    assert payload["params"]["case"] == "smoke"


def test_build_state_payload_fills_mask_and_vorticity():
    class DummySolver:
        solver_family = "fvm"
        equation_set = "navier_stokes"
        regime = "incompressible"

    u = np.zeros((4, 3, 2), dtype=np.float32)
    u[:, :, 0] = 1.0
    payload = build_state_payload(
        solver=DummySolver(),
        fields={"u": u, "p": np.zeros((4, 3), dtype=np.float32)},
        step=2,
        time_value=0.5,
    )

    assert payload["mask"].shape == (4, 3)
    assert payload["vorticity"].shape == (4, 3)
    assert np.allclose(payload["vorticity"], 0.0)


def test_jacobi_pressure_poisson_solver_solves_trivial_rhs():
    from fvm_taichi import JacobiPressurePoissonSolver

    solver = JacobiPressurePoissonSolver(ng=2, ni=6, nj=4, max_iters=50, tol=1e-8)
    rhs = np.zeros((10, 8), dtype=np.float32)
    p0 = np.zeros_like(rhs)

    p, iters, residual = solver.solve(
        rhs=rhs,
        p_seed=p0,
        dx=1.0,
        dy=1.0,
        apply_pressure_bc=lambda arr: None,
    )

    assert p.shape == rhs.shape
    assert iters == 1
    assert residual == 0.0
    assert np.allclose(p, 0.0)
