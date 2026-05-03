"""
Benchmark Registry
==================

What:
- 提供 benchmark 的註冊、查詢與 CaseRunner 建立入口

Why:
- solver toolkit 需要正式的 benchmark registry，才能支撐 batch / regression workflow
- benchmark 的 metadata、別名與預設參數不應散落在 CLI 或個別腳本中

When:
- 列舉可用 benchmark
- 依 benchmark 名稱建立 CaseRunner
- 建立回歸矩陣與批次掃描入口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from examples.cd_nozzle_euler import build_cd_nozzle_runner
from examples.couette_flow_ns import build_couette_runner
from examples.lid_driven_cavity import build_lid_driven_cavity_runner
from examples.naca0012_euler import build_naca0012_euler_runner
from examples.naca0012_ns import build_naca0012_ns_runner
from examples.poiseuille_flow_ns import build_poiseuille_runner
from examples.transonic_bump_euler import build_transonic_bump_runner

from .case_runner import CaseRunner


@dataclass(frozen=True)
class BenchmarkAcceptanceCriterion:
    """
    單一 benchmark acceptance criterion。

    What:
    - 定義 matrix / regression 執行後應檢查的單一指標門檻

    Why:
    - benchmark 的通過條件應屬於 registry metadata，而不是散落在測試或 notebook
    """

    name: str
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class BenchmarkSpec:
    """
    單一 benchmark 規格。
    """

    name: str
    builder: Callable[..., Any]
    description: str
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    default_params: dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: tuple[BenchmarkAcceptanceCriterion, ...] = ()

    def build_runner(self, **overrides: Any) -> CaseRunner:
        """
        依 spec 建立對應的 CaseRunner。
        """
        params = dict(self.default_params)
        params.update(overrides)
        built = self.builder(**params)
        if isinstance(built, tuple):
            runner = built[0]
        else:
            runner = built
        if not isinstance(runner, CaseRunner):
            raise TypeError(
                f"Benchmark builder '{self.builder.__name__}' did not return CaseRunner."
            )
        return runner


_BENCHMARKS: dict[str, BenchmarkSpec] = {}
_ALIASES: dict[str, str] = {}


def register_benchmark(spec: BenchmarkSpec) -> BenchmarkSpec:
    """
    註冊 benchmark spec。
    """
    key = spec.name.strip().lower()
    if key in _BENCHMARKS:
        raise ValueError(f"Benchmark '{spec.name}' is already registered.")
    _BENCHMARKS[key] = spec
    _ALIASES[key] = key
    for alias in spec.aliases:
        alias_key = alias.strip().lower()
        if alias_key in _ALIASES:
            raise ValueError(f"Benchmark alias '{alias}' is already registered.")
        _ALIASES[alias_key] = key
    return spec


def _resolve_benchmark_key(name: str) -> str:
    key = name.strip().lower()
    resolved = _ALIASES.get(key)
    if resolved is None:
        raise KeyError(f"Unknown benchmark: {name}")
    return resolved


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    """
    取得 benchmark spec。
    """
    return _BENCHMARKS[_resolve_benchmark_key(name)]


def list_benchmarks() -> list[BenchmarkSpec]:
    """
    列出所有已註冊 benchmark。
    """
    return [spec for _, spec in sorted(_BENCHMARKS.items(), key=lambda item: item[0])]


def create_benchmark_runner(name: str, **overrides: Any) -> CaseRunner:
    """
    依名稱建立 benchmark runner。
    """
    spec = get_benchmark_spec(name)
    return spec.build_runner(**overrides)


def _register_builtin_benchmarks():
    register_benchmark(
        BenchmarkSpec(
            name="lid_driven_cavity",
            builder=build_lid_driven_cavity_runner,
            description="Square cavity with moving lid for LBM verification.",
            tags=("lbm", "single_phase", "verification", "steady"),
            aliases=(),
            default_params={
                "res": 128,
                "re": 100.0,
                "lid_vel": 0.1,
                "cs": 0.0,
                "collision_model": "mrt",
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="LDC regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="moving-lid-active",
                    metric="max_u",
                    min_value=1e-6,
                    description="上蓋驅動至少應產生非零速度場。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="transonic_bump_euler",
            builder=build_transonic_bump_runner,
            description="Curvilinear Euler transonic bump benchmark with shock diagnostics.",
            tags=("fvm", "euler", "compressible", "transonic", "verification"),
            aliases=("transonic_bump",),
            default_params={
                "ni": 240,
                "nj": 80,
                "ma": 0.675,
                "length": 3.0,
                "height": 1.0,
                "bump_center": 1.5,
                "bump_width": 1.0,
                "bump_height": 0.12,
                "cfl": 0.35,
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="Transonic bump regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="cfl-bound",
                    metric="pseudo_cfl",
                    max_value=0.35,
                    description="Transonic bump 的 pseudo CFL 不應超過設定上限。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="flow-evolved",
                    metric="u_max",
                    min_value=1e-6,
                    description="自由流初始化後應維持非零速度尺度。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="cd_nozzle_euler",
            builder=build_cd_nozzle_runner,
            description="Curvilinear converging-diverging nozzle Euler benchmark with throat and mass-flow diagnostics.",
            tags=("fvm", "euler", "compressible", "nozzle", "verification"),
            aliases=("cd_nozzle",),
            default_params={
                "ni": 80,
                "nj": 32,
                "ma_init": 0.12,
                "h_throat": 0.70,
                "pb_ratio": 0.985,
                "cfl": 0.20,
                "time_marching": "local_pseudo",
                "initial_condition": "quasi1d",
                "reconstruction_order": "first",
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="CD nozzle regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="positive-density",
                    metric="rho_min",
                    min_value=1e-6,
                    description="Nozzle 內部密度必須維持正值。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="accelerating-through-throat",
                    metric="nozzle_acceleration_ratio",
                    min_value=1.0,
                    description="喉部 Mach 應至少不小於入口 Mach。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="mass-balance-bounded",
                    metric="mdot_balance",
                    max_value=2e-1,
                    description="入口/喉部/出口質量流率差異應維持在可接受範圍。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="naca0012_euler",
            builder=build_naca0012_euler_runner,
            description="NACA0012 Euler O-grid benchmark with characteristic far-field control and lift/drag diagnostics.",
            tags=("fvm", "euler", "compressible", "airfoil", "verification"),
            aliases=("naca_euler",),
            default_params={
                "ni": 120,
                "nj": 40,
                "ma": 0.3,
                "aoa": 5.0,
                "r_far": 12.0,
                "cfl": 0.45,
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="NACA0012 Euler regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="positive-density",
                    metric="rho_min",
                    min_value=1e-6,
                    description="翼型流場密度必須維持正值。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="positive-pressure",
                    metric="p_min",
                    min_value=1e-6,
                    description="翼型流場壓力必須維持正值。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="flow-evolved",
                    metric="mach_max",
                    min_value=1e-6,
                    description="自由流初始化後應維持非零 Mach 尺度。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="naca0012_ns",
            builder=build_naca0012_ns_runner,
            description="NACA0012 viscous Navier-Stokes O-grid benchmark with no-slip wall, far-field control, and aero diagnostics.",
            tags=("fvm", "navier_stokes", "compressible", "airfoil", "viscous", "verification"),
            aliases=("naca_ns",),
            default_params={
                "ni": 120,
                "nj": 40,
                "ma": 0.15,
                "re": 500.0,
                "aoa": 4.0,
                "r_far": 12.0,
                "cfl": 0.08,
                "time_marching": "global",
                "turbulence_model": "laminar",
                "initial_condition": "auto",
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="NACA0012 NS regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="positive-density",
                    metric="rho_min",
                    min_value=1e-6,
                    description="黏性翼型流場密度必須維持正值。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="positive-pressure",
                    metric="p_min",
                    min_value=1e-6,
                    description="黏性翼型流場壓力必須維持正值。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="flow-evolved",
                    metric="mach_max",
                    min_value=1e-6,
                    description="有限 Re 翼型案例應維持非零 Mach 尺度。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="drag-diagnostic-available",
                    metric="drag_coefficient_abs",
                    min_value=0.0,
                    description="應能輸出有限 Re 翼型的阻力診斷。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="poiseuille_flow_ns",
            builder=build_poiseuille_runner,
            description="Incompressible Navier-Stokes Poiseuille benchmark with exact profile diagnostics.",
            tags=("fvm", "navier_stokes", "incompressible", "laminar", "verification"),
            aliases=("poiseuille",),
            default_params={
                "ni": 128,
                "nj": 64,
                "re": 20.0,
                "u_max": 0.05,
                "init_mode": "parabolic",
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="Poiseuille regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="projection-active",
                    metric="projection_iters",
                    min_value=1.0,
                    description="Poisson projection 必須至少執行 1 次迭代。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="divergence-bounded",
                    metric="div_linf",
                    max_value=1e-5,
                    description="不可壓縮通道的散度殘留應維持在小量級。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="profile-error-bounded",
                    metric="l2_profile_error",
                    max_value=1e-3,
                    description="Poiseuille 解析剖面 L2 誤差應維持在 1e-3 內。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="flow-evolved",
                    metric="u_max",
                    min_value=1e-6,
                    description="Poiseuille 通道內必須維持非零流速。",
                ),
            ),
        )
    )
    register_benchmark(
        BenchmarkSpec(
            name="couette_flow_ns",
            builder=build_couette_runner,
            description="Incompressible Navier-Stokes Couette benchmark with exact linear profile diagnostics.",
            tags=("fvm", "navier_stokes", "incompressible", "laminar", "verification"),
            aliases=("couette",),
            default_params={
                "ni": 128,
                "nj": 64,
                "re": 100.0,
                "u_top": 0.1,
                "init_mode": "linear",
            },
            acceptance_criteria=(
                BenchmarkAcceptanceCriterion(
                    name="history-sampled",
                    metric="history_samples",
                    min_value=1.0,
                    description="Couette regression 至少應留下 1 筆 history sample。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="projection-active",
                    metric="projection_iters",
                    min_value=1.0,
                    description="Couette projection 步必須至少執行 1 次迭代。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="divergence-bounded",
                    metric="div_linf",
                    max_value=1e-5,
                    description="Couette 通道的散度殘留應維持在小量級。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="profile-error-bounded",
                    metric="l2_profile_error",
                    max_value=1e-5,
                    description="Couette 線性剖面 L2 誤差應維持在 1e-5 內。",
                ),
                BenchmarkAcceptanceCriterion(
                    name="wall-driven-flow-active",
                    metric="u_max",
                    min_value=1e-6,
                    description="Moving-wall case 必須維持非零流速。",
                ),
            ),
        )
    )


_register_builtin_benchmarks()
