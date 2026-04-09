"""
NavierStokesSolver 基本測試
==========================

What: 驗證第一版 laminar NS solver 的最小可用性
Why:  新 solver 至少要守住兩件事：
      - Cartesian / curvilinear 路徑都可用
      - Couette steady profile 不會被數值方法破壞
"""

import numpy as np
import pytest
import taichi as ti

from examples.naca0012_ns import run_naca0012_ns
from examples.naca0012_ns_sweep import run_sweep as run_naca0012_ns_sweep


def test_couette_sheared_grid_profile_is_steady():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver, generate_sheared_channel_grid

    ni, nj = 32, 16
    u_top = 0.1

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=u_top,
        length_scale=1.0,
    )
    x_node, y_node = generate_sheared_channel_grid(
        ni=ni, nj=nj, length=1.0, height=1.0, shear=0.25, NG=solver.NG)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=u_top, v_wall=0.0, temperature=1.0)

    y = (np.arange(nj, dtype=np.float32) + 0.5) / nj
    u_exact = u_top * y

    W = np.zeros((ni, nj, 4), dtype=np.float32)
    W[:, :, 0] = 1.0
    W[:, :, 1] = u_exact[None, :]
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    for _ in range(20):
        solver.step()

    rho_n, u_n, v_n, p_n = solver.get_primitive()
    u_profile = np.mean(u_n, axis=0)

    assert np.sqrt(np.mean((u_profile - u_exact) ** 2)) < 2e-3
    assert np.max(np.abs(v_n)) < 2e-4
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0


def test_couette_linear_profile_is_steady():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    ni, nj = 32, 16
    u_top = 0.1

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=u_top,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / ni, dy=1.0 / nj)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=u_top, v_wall=0.0, temperature=1.0)

    y = (np.arange(nj, dtype=np.float32) + 0.5) / nj
    u_exact = u_top * y

    W = np.zeros((ni, nj, 4), dtype=np.float32)
    W[:, :, 0] = 1.0
    W[:, :, 1] = u_exact[None, :]
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    for _ in range(20):
        solver.step()

    rho_n, u_n, v_n, p_n = solver.get_primitive()
    u_profile = np.mean(u_n, axis=0)

    assert np.sqrt(np.mean((u_profile - u_exact) ** 2)) < 5e-4
    assert np.max(np.abs(v_n)) < 1e-5
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0


def test_couette_local_pseudo_profile_is_steady():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    ni, nj = 32, 16
    u_top = 0.1

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=u_top,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / ni, dy=1.0 / nj)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=u_top, v_wall=0.0, temperature=1.0)
    solver.set_time_marching_mode("local_pseudo")

    y = (np.arange(nj, dtype=np.float32) + 0.5) / nj
    u_exact = u_top * y

    W = np.zeros((ni, nj, 4), dtype=np.float32)
    W[:, :, 0] = 1.0
    W[:, :, 1] = u_exact[None, :]
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    for _ in range(20):
        solver.step()

    rho_n, u_n, v_n, p_n = solver.get_primitive()
    u_profile = np.mean(u_n, axis=0)
    dt_diag = solver.get_dt_diagnostics()

    assert np.sqrt(np.mean((u_profile - u_exact) ** 2)) < 5e-4
    assert np.max(np.abs(v_n)) < 1e-5
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0
    assert dt_diag["mode"] == "local_pseudo"
    assert dt_diag["dt_min"] > 0.0
    assert dt_diag["dt_max"] >= dt_diag["dt_min"]


def test_local_pseudo_controls_report_ramp_and_preconditioning():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    solver = NavierStokesSolver(
        ni=8,
        nj=6,
        gamma=1.4,
        cfl=0.02,
        re=200.0,
        pr=0.72,
        u_ref=0.2,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8, dy=1.0 / 6)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.2, v_wall=0.0, temperature=1.0)
    solver.set_time_marching_mode("local_pseudo")
    solver.set_pseudo_time_controls(
        cfl_start=0.005,
        ramp_steps=10,
        precond_ref_mach=0.3,
        precond_min_scale=0.2,
    )
    solver.set_residual_smoothing(epsilon=0.25, passes=2)

    W = np.zeros((8, 6, 4), dtype=np.float32)
    W[:, :, 0] = 1.0
    W[:, :, 1] = 0.02
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    solver.step()
    dt_diag = solver.get_dt_diagnostics()

    assert dt_diag["mode"] == "local_pseudo"
    assert dt_diag["pseudo_iter"] == 1
    assert 0.005 - 1e-6 <= dt_diag["pseudo_cfl"] <= 0.02 + 1e-6
    assert 0.2 <= dt_diag["pseudo_precond_scale_min"] <= 1.0
    assert dt_diag["pseudo_precond_scale_max"] <= 1.0
    assert abs(dt_diag["residual_smoothing_eps"] - 0.25) < 1e-6
    assert dt_diag["residual_smoothing_passes"] == 2


def test_adaptive_pseudo_strategy_updates_targets_within_bounds():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    solver = NavierStokesSolver(
        ni=8,
        nj=6,
        gamma=1.4,
        cfl=0.02,
        re=200.0,
        pr=0.72,
        u_ref=0.2,
        length_scale=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8, dy=1.0 / 6)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.2, v_wall=0.0, temperature=1.0)
    solver.set_time_marching_mode("local_pseudo")
    solver.set_pseudo_time_controls(
        cfl_start=0.005,
        ramp_steps=0,
        precond_ref_mach=0.3,
        precond_min_scale=0.2,
    )
    solver.set_residual_smoothing(epsilon=0.1, passes=2)
    solver.set_adaptive_pseudo_strategy(
        enabled=True,
        cfl_min=0.005,
        cfl_max=0.02,
        growth=1.05,
        shrink=0.7,
        target_ratio=0.999,
        fail_ratio=1.001,
        interval=1,
        smoothing_max_eps=0.3,
    )

    W = np.zeros((8, 6, 4), dtype=np.float32)
    W[:, :, 0] = 1.0
    W[:, :, 1] = 0.02
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    solver.step()
    solver.step()
    dt_diag = solver.get_dt_diagnostics()

    assert dt_diag["pseudo_adaptive_enabled"] is True
    assert 0.005 - 1e-6 <= dt_diag["pseudo_cfl_target"] <= 0.02 + 1e-6
    assert dt_diag["pseudo_residual_ratio"] > 0.0
    assert 0.1 - 1e-6 <= dt_diag["residual_smoothing_eps"] <= 0.3 + 1e-6


def test_constant_eddy_viscosity_updates_effective_transport():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    solver = NavierStokesSolver(
        ni=8,
        nj=4,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=0.1,
        length_scale=1.0,
        rho_ref=1.0,
    )
    transport_lam = solver.get_transport_coefficients()
    solver.set_turbulence_model(
        model="constant_eddy_viscosity",
        eddy_viscosity_ratio=3.0,
        turbulent_prandtl=0.9,
    )
    transport_turb = solver.get_transport_coefficients()

    assert transport_lam["turbulence_model"] == "laminar"
    assert transport_turb["turbulence_model"] == "constant_eddy_viscosity"
    assert np.isclose(transport_turb["mu_t"], 3.0 * transport_lam["mu"])
    assert np.isclose(transport_turb["mu_eff"], transport_lam["mu"] + transport_turb["mu_t"])
    assert transport_turb["kappa_eff"] > transport_lam["kappa"]


def test_spalart_allmaras_transport_initializes_positive_nu_t():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    solver = NavierStokesSolver(
        ni=8,
        nj=4,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=0.1,
        length_scale=1.0,
        rho_ref=1.0,
    )
    solver.set_cartesian_grid(dx=1.0 / 8.0, dy=1.0 / 4.0)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.init_uniform(rho=1.0, u=0.1, v=0.0, p=1.0)
    solver.set_turbulence_model(
        model="spalart_allmaras",
        turbulent_prandtl=0.9,
        sa_nu_tilde_inf_ratio=3.0,
    )
    solver.refresh_ghost_cells()

    transport = solver.get_transport_coefficients()
    assert transport["turbulence_model"] == "spalart_allmaras"
    assert transport["mu_eff"] >= transport["mu"]


def test_poiseuille_parabolic_profile_is_steady():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver

    ni, nj = 32, 16
    u_max = 0.05
    rho0 = 1.0
    height = 1.0

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=20.0,
        pr=0.72,
        u_ref=u_max,
        length_scale=height,
        rho_ref=rho0,
    )
    solver.set_cartesian_grid(dx=1.0 / ni, dy=1.0 / nj)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.0, v_wall=0.0, temperature=1.0)

    transport = solver.get_transport_coefficients()
    nu = transport["mu"] / rho0
    body_force_x = 8.0 * nu * u_max / (height * height)
    solver.set_body_force(fx=body_force_x, fy=0.0)

    y = (np.arange(nj, dtype=np.float32) + 0.5) / nj
    u_exact = body_force_x * y * (height - y) / (2.0 * nu)

    W = np.zeros((ni, nj, 4), dtype=np.float32)
    W[:, :, 0] = rho0
    W[:, :, 1] = u_exact[None, :]
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    for _ in range(20):
        solver.step()

    rho_n, u_n, v_n, p_n = solver.get_primitive()
    u_profile = np.mean(u_n, axis=0)

    assert np.sqrt(np.mean((u_profile - u_exact) ** 2)) < 5e-4
    assert np.max(np.abs(v_n)) < 1e-5
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0


def test_poiseuille_sheared_grid_profile_is_steady():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver, generate_sheared_channel_grid

    ni, nj = 32, 16
    u_max = 0.05
    rho0 = 1.0
    height = 1.0

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=20.0,
        pr=0.72,
        u_ref=u_max,
        length_scale=height,
        rho_ref=rho0,
    )
    x_node, y_node = generate_sheared_channel_grid(
        ni=ni, nj=nj, length=1.0, height=1.0, shear=0.25, NG=solver.NG)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("bottom", u_wall=0.0, v_wall=0.0, temperature=1.0)
    solver.set_no_slip_wall("top", u_wall=0.0, v_wall=0.0, temperature=1.0)

    transport = solver.get_transport_coefficients()
    nu = transport["mu"] / rho0
    body_force_x = 8.0 * nu * u_max / (height * height)
    solver.set_body_force(fx=body_force_x, fy=0.0)

    y = (np.arange(nj, dtype=np.float32) + 0.5) / nj
    u_exact = body_force_x * y * (height - y) / (2.0 * nu)

    W = np.zeros((ni, nj, 4), dtype=np.float32)
    W[:, :, 0] = rho0
    W[:, :, 1] = u_exact[None, :]
    W[:, :, 2] = 0.0
    W[:, :, 3] = 1.0
    solver.init_from_primitive_numpy(W)

    for _ in range(20):
        solver.step()

    rho_n, u_n, v_n, p_n = solver.get_primitive()
    u_profile = np.mean(u_n, axis=0)

    assert np.sqrt(np.mean((u_profile - u_exact) ** 2)) < 2e-3
    assert np.max(np.abs(v_n)) < 2e-4
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0


def test_curved_wall_no_slip_imposes_wall_state_on_j_min_face():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver, generate_o_grid_naca0012

    ni, nj = 40, 16
    u_wall = 0.03
    v_wall = -0.02
    t_wall = 1.15

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=1.4,
        cfl=0.2,
        re=100.0,
        pr=0.72,
        u_ref=0.1,
        length_scale=1.0,
        rho_ref=1.0,
    )
    x_node, y_node = generate_o_grid_naca0012(
        ni=ni, nj=nj, R_far=8.0, NG=solver.NG)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("j_min", u_wall=u_wall, v_wall=v_wall, temperature=t_wall)

    solver.init_uniform(rho=1.0, u=0.12, v=0.05, p=1.0)
    solver._update_ghost()

    W = solver.W.to_numpy()
    NG = solver.NG

    wall_face_u = 0.5 * (W[NG:NG + ni, NG - 1, 1] + W[NG:NG + ni, NG, 1])
    wall_face_v = 0.5 * (W[NG:NG + ni, NG - 1, 2] + W[NG:NG + ni, NG, 2])

    T_int = W[NG:NG + ni, NG, 3] / W[NG:NG + ni, NG, 0]
    T_ghost = W[NG:NG + ni, NG - 1, 3] / W[NG:NG + ni, NG - 1, 0]
    wall_face_T = 0.5 * (T_int + T_ghost)

    assert np.max(np.abs(wall_face_u - u_wall)) < 1e-6
    assert np.max(np.abs(wall_face_v - v_wall)) < 1e-6
    assert np.max(np.abs(wall_face_T - t_wall)) < 1e-6


def test_o_grid_wall_spacing_ratio_clusters_near_wall():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import generate_o_grid_naca0012, NavierStokesSolver

    ni, nj = 32, 16
    NG = NavierStokesSolver.NG
    x_lin, y_lin = generate_o_grid_naca0012(ni=ni, nj=nj, R_far=8.0, NG=NG, wall_spacing_ratio=1.0)
    x_cls, y_cls = generate_o_grid_naca0012(ni=ni, nj=nj, R_far=8.0, NG=NG, wall_spacing_ratio=0.01)

    xcc_lin = 0.25 * (
        x_lin[NG:NG + ni, NG] + x_lin[NG + 1:NG + ni + 1, NG]
        + x_lin[NG:NG + ni, NG + 1] + x_lin[NG + 1:NG + ni + 1, NG + 1]
    )
    ycc_lin = 0.25 * (
        y_lin[NG:NG + ni, NG] + y_lin[NG + 1:NG + ni + 1, NG]
        + y_lin[NG:NG + ni, NG + 1] + y_lin[NG + 1:NG + ni + 1, NG + 1]
    )
    xcc_cls = 0.25 * (
        x_cls[NG:NG + ni, NG] + x_cls[NG + 1:NG + ni + 1, NG]
        + x_cls[NG:NG + ni, NG + 1] + x_cls[NG + 1:NG + ni + 1, NG + 1]
    )
    ycc_cls = 0.25 * (
        y_cls[NG:NG + ni, NG] + y_cls[NG + 1:NG + ni + 1, NG]
        + y_cls[NG:NG + ni, NG + 1] + y_cls[NG + 1:NG + ni + 1, NG + 1]
    )

    wall_x = x_lin[NG:NG + ni, NG]
    wall_y = y_lin[NG:NG + ni, NG]
    d_lin = np.sqrt((xcc_lin - wall_x) ** 2 + (ycc_lin - wall_y) ** 2)
    d_cls = np.sqrt((xcc_cls - wall_x) ** 2 + (ycc_cls - wall_y) ** 2)

    assert np.max(d_cls) < np.min(d_lin)


def test_naca_o_grid_navier_stokes_step_stays_finite():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from fvm_taichi import NavierStokesSolver, generate_o_grid_naca0012

    ni, nj = 24, 10
    gamma = 1.4
    ma = 0.12
    aoa = np.deg2rad(3.0)
    rho_inf = 1.0
    p_inf = rho_inf / gamma
    u_inf = ma * np.cos(aoa)
    v_inf = ma * np.sin(aoa)

    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=gamma,
        cfl=0.08,
        re=300.0,
        pr=0.72,
        u_ref=ma,
        length_scale=1.0,
        rho_ref=rho_inf,
    )
    x_node, y_node = generate_o_grid_naca0012(
        ni=ni, nj=nj, R_far=8.0, NG=solver.NG)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("airfoil", u_wall=0.0, v_wall=0.0, temperature=None)
    solver.set_far_field_j_max(rho_inf, u_inf, v_inf, p_inf)
    solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)

    solver.step()
    solver.update_viscous_fluxes()

    rho_n, u_n, v_n, p_n = solver.get_primitive()

    assert np.isfinite(rho_n).all()
    assert np.isfinite(u_n).all()
    assert np.isfinite(v_n).all()
    assert np.isfinite(p_n).all()
    assert np.min(rho_n) > 0.0
    assert np.min(p_n) > 0.0


def test_naca_ns_aoa_sweep_writes_polar_outputs(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_naca0012_ns_sweep(
        sweep="aoa",
        values=[0.0, 2.0],
        ni=24,
        nj=10,
        ma=0.08,
        re=200.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.03,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=str(tmp_path),
        save_cases=False,
    )

    assert len(results) == 2
    assert (tmp_path / "sweep_results.npy").exists()
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "matrix_summary.npy").exists()
    assert (tmp_path / "cl_aoa_ma008_re00200.png").exists()
    assert (tmp_path / "cd_aoa_ma008_re00200.png").exists()
    assert (tmp_path / "polar_cl_cd_ma008_re00200.png").exists()


def test_naca_ns_transonic_requires_explicit_opt_in():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="transonic regime"):
        run_naca0012_ns(
            ni=24,
            nj=10,
            ma=0.8,
            re=300.0,
            aoa=2.0,
            steps=1,
            r_far=8.0,
            cfl=0.03,
            report=1,
            tol=1e-8,
            save_interval=1,
            output_dir=None,
            allow_transonic=False,
        )


def test_naca_ns_transonic_returns_shock_metrics():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_naca0012_ns(
        ni=24,
        nj=10,
        ma=0.8,
        re=300.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.02,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=None,
        allow_transonic=True,
    )

    assert result["transonic_mode"] is True
    assert result["mach_max"] > 0.0
    assert result["shock_sensor_max"] >= 0.0
    assert result["entropy_rise_max"] >= 0.0
    assert np.isfinite(result["shock_peak_x"])
    assert np.isfinite(result["shock_peak_y"])
    assert np.isfinite(result["y_plus_max"])
    assert result["y_plus_max"] >= 0.0


def test_naca_ns_constant_eddy_viscosity_runs():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_naca0012_ns(
        ni=24,
        nj=10,
        ma=0.3,
        re=50000.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.02,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=None,
        turbulence_model="constant_eddy_viscosity",
        eddy_viscosity_ratio=8.0,
        turbulent_prandtl=0.9,
    )

    assert result["turbulence_model"] == "constant_eddy_viscosity"
    assert result["mu_t"] > 0.0
    assert result["mu_eff"] > result["mu"]
    assert np.isfinite(result["CD"])


def test_naca_ns_spalart_allmaras_runs():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_naca0012_ns(
        ni=24,
        nj=10,
        ma=0.3,
        re=50000.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.01,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=None,
        turbulence_model="spalart_allmaras",
        turbulent_prandtl=0.9,
        sa_nu_tilde_inf_ratio=3.0,
    )

    assert result["turbulence_model"] == "spalart_allmaras"


def test_naca_ns_local_pseudo_runs():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_naca0012_ns(
        ni=24,
        nj=10,
        ma=0.3,
        re=50000.0,
        aoa=2.0,
        steps=2,
        r_far=8.0,
        cfl=0.02,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=None,
        turbulence_model="spalart_allmaras",
        time_marching="local_pseudo",
    )

    assert result["time_marching"] == "local_pseudo"
    assert result["sim_time"] > 0.0
    assert result["residual"] >= 0.0


def test_naca_ns_local_pseudo_rejects_target_time():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="target_sim_time is only valid"):
        run_naca0012_ns(
            ni=24,
            nj=10,
            ma=0.3,
            re=50000.0,
            aoa=2.0,
            steps=1,
            r_far=8.0,
            cfl=0.02,
            report=1,
            tol=1e-8,
            save_interval=1,
            output_dir=None,
            turbulence_model="spalart_allmaras",
            time_marching="local_pseudo",
            target_sim_time=1e-3,
        )


def test_naca_ns_global_rejects_pseudo_controls():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="pseudo-time ramp / preconditioning controls require"):
        run_naca0012_ns(
            ni=24,
            nj=10,
            ma=0.3,
            re=50000.0,
            aoa=2.0,
            steps=1,
            r_far=8.0,
            cfl=0.02,
            report=1,
            tol=1e-8,
            save_interval=1,
            output_dir=None,
            turbulence_model="spalart_allmaras",
            time_marching="global",
            pseudo_ramp_steps=20,
        )


def test_naca_ns_global_rejects_residual_smoothing_controls():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="pseudo-time ramp / preconditioning controls require"):
        run_naca0012_ns(
            ni=24,
            nj=10,
            ma=0.3,
            re=50000.0,
            aoa=2.0,
            steps=1,
            r_far=8.0,
            cfl=0.02,
            report=1,
            tol=1e-8,
            save_interval=1,
            output_dir=None,
            turbulence_model="spalart_allmaras",
            time_marching="global",
            residual_smoothing_eps=0.25,
            residual_smoothing_passes=2,
        )


def test_naca_ns_global_rejects_adaptive_pseudo_controls():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    with pytest.raises(ValueError, match="pseudo-time ramp / preconditioning controls require"):
        run_naca0012_ns(
            ni=24,
            nj=10,
            ma=0.3,
            re=50000.0,
            aoa=2.0,
            steps=1,
            r_far=8.0,
            cfl=0.02,
            report=1,
            tol=1e-8,
            save_interval=1,
            output_dir=None,
            turbulence_model="spalart_allmaras",
            time_marching="global",
            adaptive_pseudo=True,
        )


def test_naca_ns_output_writes_wall_yplus_plot(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_naca0012_ns(
        ni=24,
        nj=10,
        ma=0.12,
        re=300.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.03,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=str(tmp_path),
    )

    assert result["output_dir"] == str(tmp_path)
    assert (tmp_path / "surface_final.npy").exists()
    assert (tmp_path / "summary.npy").exists()
    assert (tmp_path / "history.npy").exists()
    assert (tmp_path / "wall_yplus_ma012_re00300_aoa002.png").exists()


def test_naca_ns_transonic_ma_sweep_writes_shock_outputs(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_naca0012_ns_sweep(
        sweep="ma",
        values=[0.75, 0.8],
        ni=24,
        nj=10,
        ma=0.8,
        re=200.0,
        aoa=2.0,
        steps=1,
        r_far=8.0,
        cfl=0.03,
        report=1,
        tol=1e-8,
        save_interval=1,
        output_dir=str(tmp_path),
        save_cases=False,
        allow_transonic=True,
        plateau_window=3,
        shock_tol=0.1,
        mach_tol=0.1,
    )

    assert len(results) == 2
    assert (tmp_path / "sweep_results.npy").exists()
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "matrix_summary.npy").exists()
    assert (tmp_path / "shock_peak_vs_ma_re00200_aoa002.png").exists()
    assert (tmp_path / "mach_max_vs_ma_re00200_aoa002.png").exists()
    assert (tmp_path / "supersonic_fraction_vs_ma_re00200_aoa002.png").exists()
