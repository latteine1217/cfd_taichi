"""
CFD Taichi 套件入口
===================

What:
- 提供專案層級的公開 API 匯出

Why:
- `cfd_taichi.output_schema` 需要能被 diagnostics 直接引用
- package `__init__` 若過度 eager import，會在 `lbm_taichi <-> cfd_taichi`
  之間形成初始化循環；改用 lazy export 可保持 surface 穩定且避免循環
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "__version__",
    "SolverProtocol",
    "Grid2D",
    "CartesianGrid2D",
    "CurvilinearGrid2D",
    "LatticeGrid2D",
    "GridDescriptor",
    "BoundaryConditionDescriptor",
    "SolverControlDescriptor",
    "CaseRunner",
    "apply_grid_descriptor",
    "apply_boundary_descriptors",
    "apply_solver_control_descriptors",
    "build_state_payload",
    "build_history_payload",
    "compute_vorticity_2d",
    "BenchmarkSpec",
    "BenchmarkAcceptanceCriterion",
    "list_benchmarks",
    "get_benchmark_spec",
    "create_benchmark_runner",
    "BenchmarkMatrixEntry",
    "BenchmarkMatrixResult",
    "run_benchmark_matrix",
    "build_benchmark_matrix_payload",
    "save_benchmark_matrix_payload",
    "build_naca0012_ns_sweep_preset",
    "build_transonic_bump_sweep_preset",
    "create_solver",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    if name in {
        "apply_boundary_descriptors",
        "apply_grid_descriptor",
        "apply_solver_control_descriptors",
    }:
        from .configuration import (
            apply_boundary_descriptors,
            apply_grid_descriptor,
            apply_solver_control_descriptors,
        )

        return {
            "apply_boundary_descriptors": apply_boundary_descriptors,
            "apply_grid_descriptor": apply_grid_descriptor,
            "apply_solver_control_descriptors": apply_solver_control_descriptors,
        }[name]

    if name in {"BoundaryConditionDescriptor", "GridDescriptor", "SolverControlDescriptor"}:
        from .descriptors import (
            BoundaryConditionDescriptor,
            GridDescriptor,
            SolverControlDescriptor,
        )

        return {
            "BoundaryConditionDescriptor": BoundaryConditionDescriptor,
            "GridDescriptor": GridDescriptor,
            "SolverControlDescriptor": SolverControlDescriptor,
        }[name]

    if name in {"Grid2D", "CartesianGrid2D", "CurvilinearGrid2D", "LatticeGrid2D"}:
        from .grid2d import Grid2D, CartesianGrid2D, CurvilinearGrid2D, LatticeGrid2D

        return {
            "Grid2D": Grid2D,
            "CartesianGrid2D": CartesianGrid2D,
            "CurvilinearGrid2D": CurvilinearGrid2D,
            "LatticeGrid2D": LatticeGrid2D,
        }[name]

    if name in {"build_state_payload", "build_history_payload", "compute_vorticity_2d"}:
        from .output_schema import (
            build_history_payload,
            build_state_payload,
            compute_vorticity_2d,
        )

        return {
            "build_state_payload": build_state_payload,
            "build_history_payload": build_history_payload,
            "compute_vorticity_2d": compute_vorticity_2d,
        }[name]

    if name == "SolverProtocol":
        from .protocols import SolverProtocol

        return SolverProtocol

    if name == "create_solver":
        from .solver_factory import create_solver

        return create_solver

    if name == "CaseRunner":
        from .case_runner import CaseRunner

        return CaseRunner

    if name in {
        "BenchmarkSpec",
        "BenchmarkAcceptanceCriterion",
        "list_benchmarks",
        "get_benchmark_spec",
        "create_benchmark_runner",
    }:
        from .benchmark_registry import (
            BenchmarkAcceptanceCriterion,
            BenchmarkSpec,
            create_benchmark_runner,
            get_benchmark_spec,
            list_benchmarks,
        )

        return {
            "BenchmarkSpec": BenchmarkSpec,
            "BenchmarkAcceptanceCriterion": BenchmarkAcceptanceCriterion,
            "list_benchmarks": list_benchmarks,
            "get_benchmark_spec": get_benchmark_spec,
            "create_benchmark_runner": create_benchmark_runner,
        }[name]

    if name in {
        "BenchmarkMatrixEntry",
        "BenchmarkMatrixResult",
        "run_benchmark_matrix",
        "build_benchmark_matrix_payload",
        "save_benchmark_matrix_payload",
    }:
        from .benchmark_matrix import (
            BenchmarkMatrixEntry,
            BenchmarkMatrixResult,
            build_benchmark_matrix_payload,
            run_benchmark_matrix,
            save_benchmark_matrix_payload,
        )

        return {
            "BenchmarkMatrixEntry": BenchmarkMatrixEntry,
            "BenchmarkMatrixResult": BenchmarkMatrixResult,
            "run_benchmark_matrix": run_benchmark_matrix,
            "build_benchmark_matrix_payload": build_benchmark_matrix_payload,
            "save_benchmark_matrix_payload": save_benchmark_matrix_payload,
        }[name]

    if name in {"build_naca0012_ns_sweep_preset", "build_transonic_bump_sweep_preset"}:
        from .benchmark_presets import (
            build_naca0012_ns_sweep_preset,
            build_transonic_bump_sweep_preset,
        )

        return {
            "build_naca0012_ns_sweep_preset": build_naca0012_ns_sweep_preset,
            "build_transonic_bump_sweep_preset": build_transonic_bump_sweep_preset,
        }[name]

    raise AttributeError(f"module 'cfd_taichi' has no attribute {name!r}")
