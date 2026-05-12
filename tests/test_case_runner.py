"""
CaseRunner 與 Grid2D 測試
=========================

What:
- 驗證 Grid2D 物件能提供尺寸與 descriptor
- 驗證 CaseRunner 能驅動 FVM / LBM 的 build-config-init-step 流程

Why:
- 這是 solver toolkit workflow 層的第一版骨架，必須有回歸保護
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from cfd_taichi import (
    BenchmarkMatrixEntry,
    BoundaryConditionDescriptor,
    build_naca0012_ns_sweep_preset,
    build_transonic_bump_sweep_preset,
    build_benchmark_matrix_payload,
    create_benchmark_runner,
    CartesianGrid2D,
    CaseRunner,
    CurvilinearGrid2D,
    get_benchmark_spec,
    GridDescriptor,
    LatticeGrid2D,
    list_benchmarks,
    run_benchmark_matrix,
    save_benchmark_matrix_payload,
)


def test_grid2d_objects_expose_size_and_descriptor():
    cart = CartesianGrid2D(ni=8, nj=6, dx=0.125, dy=0.25)
    lattice = LatticeGrid2D(nx=16, ny=12)
    x_node = np.zeros((13, 11), dtype=np.float32)
    y_node = np.zeros((13, 11), dtype=np.float32)
    curv = CurvilinearGrid2D(x_node=x_node, y_node=y_node, ng=2)

    assert cart.to_descriptor() == GridDescriptor.cartesian(dx=0.125, dy=0.25)
    assert lattice.to_descriptor() == GridDescriptor.lattice(nx=16, ny=12)
    assert curv.to_descriptor().kind == "curvilinear"
    assert cart.solver_size_kwargs("fvm") == {"ni": 8, "nj": 6}
    assert cart.solver_size_kwargs("lbm") == {"nx": 8, "ny": 6}
    assert lattice.solver_size_kwargs("lbm") == {"nx": 16, "ny": 12}
    assert curv.ni == 8
    assert curv.nj == 6


def test_case_runner_runs_incompressible_fvm_and_builds_history(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    def _initializer(runner, solver, _bc):
        solver.init_uniform(u=0.0, v=0.0, p=0.0)

    runner = CaseRunner(
        name="inc_ns_smoke",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=8, nj=6, dx=1.0 / 8, dy=1.0 / 6),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.1,
            "length_scale": 1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.no_slip("top", u_wall=0.1),
            BoundaryConditionDescriptor.body_force(fx=1e-3, fy=0.0),
        ],
        initializer=_initializer,
    )

    solver = runner.build_solver()
    runner.configure()
    runner.initialize()
    dt = runner.step_once()
    payload = runner.save_state(additional_data={"case": "smoke"})
    sample = runner.record_history_sample(extra={"dt": float(dt)})
    history = runner.build_history_payload(params={"name": "inc_ns_smoke"})

    assert solver is runner.solver
    assert dt > 0.0
    assert payload["solver_family"] == "fvm"
    assert payload["regime"] == "incompressible"
    assert payload["u"].shape == (8, 6, 2)
    assert payload["p"].shape == (8, 6)
    assert payload["vorticity"].shape == (8, 6)
    assert payload["case"] == "smoke"
    assert sample["step"] == 1
    assert history["steps"].tolist() == [1]
    assert history["u_max"].shape == (1,)
    assert history["dt"].shape == (1,)
    assert history["params"]["name"] == "inc_ns_smoke"

    history_path = runner.save_history_file(
        tmp_path / "inc_history.npy",
        params={"name": "inc_ns_smoke"},
    )
    reloaded = CaseRunner(
        name="inc_ns_loaded",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=8, nj=6, dx=1.0 / 8, dy=1.0 / 6),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.1,
            "length_scale": 1.0,
        },
    )
    loaded_payload = reloaded.load_history(history_path)
    loaded_history = reloaded.build_history_payload(params={"name": "loaded"})

    assert history_path.exists()
    assert loaded_payload["steps"].tolist() == [1]
    assert reloaded.current_step == 1
    assert loaded_history["dt"].shape == (1,)


def test_case_runner_configures_lbm_and_uses_boundary_manager():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    from lbm_taichi import BoundaryConditions

    def _initializer(runner, solver, _bc):
        u0 = np.zeros((8, 8, 2), dtype=np.float32)
        solver.set_initial_condition(
            velocity=u0,
            apply_boundaries=True,
            reset_baseline=True,
        )
        runner.prepare_observables(reset_baseline=True)

    runner = CaseRunner(
        name="lbm_smoke",
        method="lbm",
        equation="single_phase",
        grid=LatticeGrid2D(nx=8, ny=8),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.05,
            "length_scale": 8.0,
            "cs": -1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
        initializer=_initializer,
    )

    runner.build_solver()
    runner.configure()
    runner.initialize()
    runner.step_once()
    runner.prepare_observables(reset_baseline=False)
    payload = runner.save_state(step=runner.current_step)

    assert isinstance(runner.boundary_handle, BoundaryConditions)
    assert runner.current_step == 1
    assert payload["solver_family"] == "lbm"
    assert payload["u"].shape == (8, 8, 2)
    assert payload["rho"].shape == (8, 8)


def test_case_runner_can_restart_incompressible_fvm_from_state_payload(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    def _initializer(_runner, solver, _bc):
        solver.init_uniform(u=0.0, v=0.0, p=0.0)

    runner = CaseRunner(
        name="inc_restart",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=8, nj=6, dx=1.0 / 8, dy=1.0 / 6),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.1,
            "length_scale": 1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.no_slip("top", u_wall=0.1),
        ],
        initializer=_initializer,
    )
    runner.build_solver()
    runner.configure()
    runner.initialize()
    runner.step_once()
    runner.step_once()

    state_path = runner.save_state_file(tmp_path / "inc_state.npy", additional_data={"tag": "restart"})
    ref_u, ref_v, ref_p = runner.solver.get_primitive()

    restarted = CaseRunner(
        name="inc_restart_copy",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=8, nj=6, dx=1.0 / 8, dy=1.0 / 6),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.1,
            "length_scale": 1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.no_slip("top", u_wall=0.1),
        ],
    )
    payload = restarted.load_state(state_path)
    u2, v2, p2 = restarted.solver.get_primitive()
    dt = restarted.step_once()

    assert payload["tag"] == "restart"
    assert restarted.current_step == 3
    assert dt > 0.0
    assert np.allclose(u2, ref_u)
    assert np.allclose(v2, ref_v)
    assert np.allclose(p2, ref_p)


def test_case_runner_can_restart_lbm_from_state_payload():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    def _initializer(runner, solver, _bc):
        u0 = np.zeros((8, 8, 2), dtype=np.float32)
        solver.set_initial_condition(
            velocity=u0,
            apply_boundaries=True,
            reset_baseline=True,
        )
        runner.prepare_observables(reset_baseline=True)

    runner = CaseRunner(
        name="lbm_restart",
        method="lbm",
        equation="single_phase",
        grid=LatticeGrid2D(nx=8, ny=8),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.05,
            "length_scale": 8.0,
            "cs": -1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
        initializer=_initializer,
    )
    runner.build_solver()
    runner.configure()
    runner.initialize()
    runner.step_once()
    runner.prepare_observables(reset_baseline=False)
    payload = runner.save_state(additional_data={"tag": "lbm-restart"})
    fields_ref = runner.solver.get_fields()

    restarted = CaseRunner(
        name="lbm_restart_copy",
        method="lbm",
        equation="single_phase",
        grid=LatticeGrid2D(nx=8, ny=8),
        solver_kwargs={
            "re": 100.0,
            "u_ref": 0.05,
            "length_scale": 8.0,
            "cs": -1.0,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.no_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
    )
    restarted.load_state(payload)
    fields_new = restarted.solver.get_fields()
    restarted.step_once()

    assert restarted.current_step == 2
    assert payload["tag"] == "lbm-restart"
    assert np.allclose(fields_new["u"], fields_ref["u"])
    assert np.allclose(fields_new["rho"], fields_ref["rho"])


def test_benchmark_registry_lists_and_builds_lid_runner():
    specs = list_benchmarks()
    names = [spec.name for spec in specs]

    assert "lid_driven_cavity" in names

    spec = get_benchmark_spec("lid_driven_cavity")
    runner = create_benchmark_runner("lid_driven_cavity", res=16, re=100.0, lid_vel=0.05)

    assert spec.name == "lid_driven_cavity"
    assert "verification" in spec.tags
    assert runner.name == "lid_driven_cavity"
    assert runner.method == "lbm"
    assert runner.grid.solver_size_kwargs("lbm") == {"nx": 16, "ny": 16}


def test_benchmark_registry_builds_transonic_bump_runner():
    spec = get_benchmark_spec("transonic_bump_euler")
    runner = create_benchmark_runner("transonic_bump_euler", ni=40, nj=12, ma=0.7)

    assert spec.name == "transonic_bump_euler"
    assert "compressible" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "transonic_bump_euler"
    assert runner.method == "fvm"
    assert runner.regime == "compressible"
    assert runner.grid.kind == "curvilinear"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 40, "nj": 12}


def test_benchmark_registry_builds_cd_nozzle_runner():
    spec = get_benchmark_spec("cd_nozzle")
    runner = create_benchmark_runner("cd_nozzle", ni=40, nj=12, ma_init=0.12)

    assert spec.name == "cd_nozzle_euler"
    assert "nozzle" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "cd_nozzle_euler"
    assert runner.method == "fvm"
    assert runner.regime == "compressible"
    assert runner.grid.kind == "curvilinear"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 40, "nj": 12}


def test_benchmark_registry_builds_flow_over_cylinder_runner():
    spec = get_benchmark_spec("cylinder")
    runner = create_benchmark_runner("cylinder", res_y=24, re=100.0, u_in=0.05, cs=-1.0)

    assert spec.name == "flow_over_cylinder"
    assert "cylinder" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "flow_over_cylinder"
    assert runner.method == "lbm"
    assert runner.equation == "single_phase"
    assert runner.regime == "low_mach"
    assert runner.grid.solver_size_kwargs("lbm") == {"nx": 60, "ny": 24}


def test_benchmark_registry_builds_rayleigh_benard_runner():
    spec = get_benchmark_spec("rayleigh_benard")
    runner = create_benchmark_runner(
        "rb",
        ny=12,
        Ra=1e4,
        Pr=0.71,
        u_ref=0.05,
        aspect=1.0,
    )

    assert spec.name == "rayleigh_benard"
    assert "thermal" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "rayleigh_benard"
    assert runner.method == "lbm"
    assert runner.equation == "thermal_boussinesq"
    assert runner.regime == "low_mach"
    assert runner.grid.solver_size_kwargs("lbm") == {"nx": 12, "ny": 12}


def test_benchmark_registry_builds_naca0012_euler_runner():
    spec = get_benchmark_spec("naca_euler")
    runner = create_benchmark_runner("naca_euler", ni=40, nj=16, ma=0.3, aoa=5.0)

    assert spec.name == "naca0012_euler"
    assert "airfoil" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "naca0012_euler"
    assert runner.method == "fvm"
    assert runner.regime == "compressible"
    assert runner.grid.kind == "curvilinear"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 40, "nj": 16}


def test_benchmark_registry_builds_naca0012_ns_runner():
    spec = get_benchmark_spec("naca_ns")
    runner = create_benchmark_runner(
        "naca_ns",
        ni=24,
        nj=10,
        ma=0.12,
        re=300.0,
        aoa=2.0,
        r_far=8.0,
        cfl=0.03,
    )

    assert spec.name == "naca0012_ns"
    assert "airfoil" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "naca0012_ns"
    assert runner.method == "fvm"
    assert runner.equation == "navier_stokes"
    assert runner.regime == "compressible"
    assert runner.grid.kind == "curvilinear"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 24, "nj": 10}


def test_naca0012_ns_sweep_preset_builds_matrix_entries(tmp_path):
    entries = build_naca0012_ns_sweep_preset(
        sweep="aoa",
        values=[0.0, 2.0],
        ni=24,
        nj=10,
        ma=0.12,
        re=300.0,
        aoa=2.0,
        steps=3,
        report=1,
        output_dir=tmp_path,
        save_cases=True,
    )

    assert len(entries) == 2
    assert all(entry.benchmark == "naca0012_ns" for entry in entries)
    assert [entry.overrides["aoa"] for entry in entries] == [0.0, 2.0]
    assert all(entry.sample_interval == 1 for entry in entries)
    assert all(entry.save_state for entry in entries)
    assert all(entry.save_history for entry in entries)
    assert all(entry.output_dir is not None for entry in entries)


def test_transonic_bump_sweep_preset_builds_matrix_entries(tmp_path):
    entries = build_transonic_bump_sweep_preset(
        sweep="ma",
        values=[0.62, 0.66],
        ni=40,
        nj=12,
        ma=0.66,
        steps=4,
        report=2,
        output_dir=tmp_path,
        save_cases=True,
    )

    assert len(entries) == 2
    assert all(entry.benchmark == "transonic_bump_euler" for entry in entries)
    assert [entry.overrides["ma"] for entry in entries] == [0.62, 0.66]
    assert all(entry.sample_interval == 2 for entry in entries)
    assert all(entry.save_state for entry in entries)
    assert all(entry.save_history for entry in entries)
    assert all(entry.output_dir is not None for entry in entries)


def test_benchmark_registry_builds_poiseuille_runner():
    spec = get_benchmark_spec("poiseuille")
    runner = create_benchmark_runner("poiseuille", ni=24, nj=12, re=20.0, u_max=0.05)

    assert spec.name == "poiseuille_flow_ns"
    assert "incompressible" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "poiseuille_flow_ns"
    assert runner.method == "fvm"
    assert runner.regime == "incompressible"
    assert runner.grid.kind == "cartesian"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 24, "nj": 12}


def test_benchmark_registry_builds_couette_runner():
    spec = get_benchmark_spec("couette")
    runner = create_benchmark_runner("couette", ni=24, nj=12, re=100.0, u_top=0.1)

    assert spec.name == "couette_flow_ns"
    assert "incompressible" in spec.tags
    assert len(spec.acceptance_criteria) >= 1
    assert runner.name == "couette_flow_ns"
    assert runner.method == "fvm"
    assert runner.regime == "incompressible"
    assert runner.grid.kind == "cartesian"
    assert runner.grid.solver_size_kwargs("fvm") == {"ni": 24, "nj": 12}


def test_benchmark_matrix_runs_lbm_and_fvm_entries(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_benchmark_matrix(
        [
            BenchmarkMatrixEntry(
                benchmark="lid_driven_cavity",
                steps=2,
                sample_interval=1,
                record_initial=True,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"res": 12, "re": 100.0, "lid_vel": 0.05},
                history_params={"case": "ldc-matrix"},
                additional_state_data={"case": "ldc-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="transonic_bump_euler",
                steps=1,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"ni": 40, "nj": 12, "ma": 0.70},
                history_params={"case": "bump-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="cd_nozzle_euler",
                steps=1,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"ni": 40, "nj": 12, "ma_init": 0.12},
                history_params={"case": "nozzle-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="flow_over_cylinder",
                steps=2,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"res_y": 24, "re": 100.0, "u_in": 0.05, "cs": -1.0},
                history_params={"case": "cylinder-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="naca0012_euler",
                steps=1,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"ni": 40, "nj": 16, "ma": 0.3, "aoa": 5.0},
                history_params={"case": "naca-euler-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="naca0012_ns",
                steps=1,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={
                    "ni": 24,
                    "nj": 10,
                    "ma": 0.12,
                    "re": 300.0,
                    "aoa": 2.0,
                    "r_far": 8.0,
                    "cfl": 0.03,
                },
                history_params={"case": "naca-ns-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="poiseuille_flow_ns",
                steps=2,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"ni": 24, "nj": 12, "re": 20.0, "u_max": 0.05},
                history_params={"case": "poiseuille-matrix"},
            ),
            BenchmarkMatrixEntry(
                benchmark="couette_flow_ns",
                steps=2,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={"ni": 24, "nj": 12, "re": 100.0, "u_top": 0.1},
                history_params={"case": "couette-matrix"},
            ),
        ]
    )

    summary = build_benchmark_matrix_payload(results)
    summary_path = save_benchmark_matrix_payload(tmp_path / "matrix_summary.npy", results)

    assert len(results) == 8
    assert results[0].benchmark == "lid_driven_cavity"
    assert results[0].solver_family == "lbm"
    assert results[0].history_samples >= 2
    assert results[0].acceptance_passed is True
    assert results[0].state_path is not None
    assert results[0].history_path is not None
    assert (tmp_path / "lid_driven_cavity" / "state_final.npy").exists()
    assert (tmp_path / "lid_driven_cavity" / "history.npy").exists()

    assert results[1].benchmark == "transonic_bump_euler"
    assert results[1].solver_family == "fvm"
    assert results[1].equation_set == "euler"
    assert results[1].acceptance_passed is True
    assert (tmp_path / "transonic_bump_euler" / "state_final.npy").exists()
    assert (tmp_path / "transonic_bump_euler" / "history.npy").exists()

    assert results[2].benchmark == "cd_nozzle_euler"
    assert results[2].solver_family == "fvm"
    assert results[2].equation_set == "euler"
    assert results[2].regime == "compressible"
    assert results[2].acceptance_passed is True
    assert results[2].diagnostics["nozzle_acceleration_ratio"] >= 1.0
    assert (tmp_path / "cd_nozzle_euler" / "state_final.npy").exists()
    assert (tmp_path / "cd_nozzle_euler" / "history.npy").exists()

    assert results[3].benchmark == "flow_over_cylinder"
    assert results[3].solver_family == "lbm"
    assert results[3].equation_set == "navier_stokes"
    assert results[3].regime == "low_mach"
    assert results[3].acceptance_passed is True
    assert results[3].diagnostics["obstacle_cells"] >= 1.0
    assert np.isfinite(results[3].diagnostics["drag_coefficient"])
    assert (tmp_path / "flow_over_cylinder" / "state_final.npy").exists()
    assert (tmp_path / "flow_over_cylinder" / "history.npy").exists()

    assert results[4].benchmark == "naca0012_euler"
    assert results[4].solver_family == "fvm"
    assert results[4].equation_set == "euler"
    assert results[4].regime == "compressible"
    assert results[4].acceptance_passed is True
    assert np.isfinite(results[4].diagnostics["lift_coefficient"])
    assert np.isfinite(results[4].diagnostics["drag_coefficient"])
    assert (tmp_path / "naca0012_euler" / "state_final.npy").exists()
    assert (tmp_path / "naca0012_euler" / "history.npy").exists()

    assert results[5].benchmark == "naca0012_ns"
    assert results[5].solver_family == "fvm"
    assert results[5].equation_set == "navier_stokes"
    assert results[5].regime == "compressible"
    assert results[5].acceptance_passed is True
    assert np.isfinite(results[5].diagnostics["lift_coefficient"])
    assert np.isfinite(results[5].diagnostics["drag_coefficient"])
    assert results[5].diagnostics["y_plus_max"] >= 0.0
    assert (tmp_path / "naca0012_ns" / "state_final.npy").exists()
    assert (tmp_path / "naca0012_ns" / "history.npy").exists()

    assert results[6].benchmark == "poiseuille_flow_ns"
    assert results[6].solver_family == "fvm"
    assert results[6].equation_set == "navier_stokes"
    assert results[6].regime == "incompressible"
    assert results[6].acceptance_passed is True
    assert results[6].diagnostics["l2_profile_error"] < 1e-3
    assert (tmp_path / "poiseuille_flow_ns" / "state_final.npy").exists()
    assert (tmp_path / "poiseuille_flow_ns" / "history.npy").exists()

    assert results[7].benchmark == "couette_flow_ns"
    assert results[7].solver_family == "fvm"
    assert results[7].equation_set == "navier_stokes"
    assert results[7].regime == "incompressible"
    assert results[7].acceptance_passed is True
    assert results[7].diagnostics["l2_profile_error"] < 1e-5
    assert (tmp_path / "couette_flow_ns" / "state_final.npy").exists()
    assert (tmp_path / "couette_flow_ns" / "history.npy").exists()

    assert summary["benchmarks"].tolist() == [
        "lid_driven_cavity",
        "transonic_bump_euler",
        "cd_nozzle_euler",
        "flow_over_cylinder",
        "naca0012_euler",
        "naca0012_ns",
        "poiseuille_flow_ns",
        "couette_flow_ns",
    ]
    assert summary["steps"].tolist() == [2, 1, 1, 2, 1, 1, 2, 2]
    assert summary["acceptance_passed"].tolist() == [True, True, True, True, True, True, True, True]
    assert summary["acceptance_failed_count"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]
    assert summary["history_samples"].tolist()[0] >= 2
    assert summary["nozzle_acceleration_ratio"][2] >= 1.0
    assert summary["mdot_balance"][2] <= 2e-1
    assert summary["obstacle_cells"][3] >= 1.0
    assert np.isfinite(summary["drag_coefficient"][3])
    assert summary["mach_max"][4] > 0.0
    assert np.isfinite(summary["mach_max"][4])
    assert np.isfinite(summary["drag_coefficient"][5])
    assert summary["drag_coefficient_abs"][5] >= 0.0
    assert summary["l2_profile_error"][6] < 1e-3
    assert summary["div_linf"][6] <= 1e-5
    assert summary["l2_profile_error"][7] < 1e-5
    assert summary["div_linf"][7] <= 1e-5
    assert summary_path.exists()


def test_benchmark_matrix_runs_rayleigh_benard_entry(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_benchmark_matrix(
        [
            BenchmarkMatrixEntry(
                benchmark="rayleigh_benard",
                steps=1,
                sample_interval=1,
                output_dir=tmp_path,
                save_state=True,
                save_history=True,
                overrides={
                    "ny": 8,
                    "Ra": 1e4,
                    "Pr": 0.71,
                    "u_ref": 0.05,
                    "aspect": 1.0,
                },
                history_params={"case": "rb-matrix"},
            ),
        ]
    )

    result = results[0]
    state = np.load(tmp_path / "rayleigh_benard" / "state_final.npy", allow_pickle=True).item()
    history = np.load(tmp_path / "rayleigh_benard" / "history.npy", allow_pickle=True).item()

    assert result.benchmark == "rayleigh_benard"
    assert result.solver_family == "lbm"
    assert result.equation_set == "navier_stokes"
    assert result.acceptance_passed is True
    assert result.diagnostics["nusselt"] >= 0.0
    assert np.isfinite(result.diagnostics["temperature_mid"])
    assert state["temperature"].shape == (8, 8)
    assert state["Ra"] == 1e4
    assert history["nusselt"].shape == (1,)


def test_rayleigh_benard_stepper_uses_active_temperature_for_buoyancy():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from examples.rayleigh_benard import build_rayleigh_benard_runner

    runner, _setup = build_rayleigh_benard_runner(
        ny=8,
        Ra=1e4,
        Pr=0.71,
        u_ref=0.05,
        aspect=1.0,
        perturbation=0.0,
    )
    solver = runner.build_solver()
    runner.configure()
    runner.initialize()

    thermal = runner.thermal
    thermal._fill_equilibrium(0.5)
    g_np = thermal.g.to_numpy()
    w_np = thermal.w.to_numpy()
    for k in range(9):
        g_np[:, :, k] = w_np[k] * 1.0
    thermal.g.from_numpy(g_np)

    runner.step_once()

    force_y = solver.force_field.to_numpy()[1:solver.nx + 1, 1:solver.ny + 1, 1]
    assert float(np.min(force_y)) > 0.0


def test_benchmark_matrix_can_continue_after_failure():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_benchmark_matrix(
        [
            BenchmarkMatrixEntry(benchmark="missing_case", steps=1),
            BenchmarkMatrixEntry(
                benchmark="lid_driven_cavity",
                steps=1,
                sample_interval=1,
                overrides={"res": 8, "re": 100.0, "lid_vel": 0.05},
            ),
        ],
        stop_on_error=False,
    )

    assert len(results) == 2
    assert results[0].error is not None
    assert results[1].benchmark == "lid_driven_cavity"
    assert results[1].error is None


def test_benchmark_matrix_reports_acceptance_failure():
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_benchmark_matrix(
        [
            BenchmarkMatrixEntry(
                benchmark="lid_driven_cavity",
                steps=0,
                sample_interval=0,
                record_initial=False,
                record_final=False,
                overrides={"res": 8, "re": 100.0, "lid_vel": 0.05},
            )
        ]
    )
    summary = build_benchmark_matrix_payload(results)

    assert len(results) == 1
    assert results[0].acceptance_passed is False
    assert any(not detail["passed"] for detail in results[0].acceptance_details)
    assert summary["acceptance_passed"].tolist() == [False]
    assert summary["acceptance_failed_count"].tolist() == [1]
