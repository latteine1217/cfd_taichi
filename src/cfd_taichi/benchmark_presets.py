"""
Benchmark Presets
=================

What:
- 提供 benchmark matrix 的可重用 preset builder

Why:
- sweep / regression 不應各自手寫 entry 組裝邏輯
- preset 讓高階 workflow 只描述「掃哪些點」，而不是重造 matrix execution path
"""

from __future__ import annotations

from pathlib import Path

from .benchmark_matrix import BenchmarkMatrixEntry


def build_naca0012_ns_sweep_preset(
    *,
    sweep: str,
    values: list[float] | tuple[float, ...],
    ni: int = 120,
    nj: int = 40,
    ma: float = 0.12,
    re: float = 300.0,
    aoa: float = 3.0,
    steps: int = 6000,
    r_far: float = 15.0,
    cfl: float = 0.08,
    report: int = 200,
    tol: float = 5e-5,
    save_interval: int = 1000,
    wall_spacing_ratio: float = 1.0,
    output_dir: str | Path | None = None,
    save_cases: bool = False,
    allow_transonic: bool = False,
    turbulence_model: str = "laminar",
    eddy_viscosity_ratio: float = 0.0,
    turbulent_prandtl: float = 0.9,
    sa_nu_tilde_inf_ratio: float = 3.0,
    plateau_window: int = 5,
    shock_tol: float = 0.03,
    mach_tol: float = 0.05,
) -> list[BenchmarkMatrixEntry]:
    """
    建立 NACA0012 NS sweep 的 matrix preset。

    What:
    - 把 sweep 點轉成 `BenchmarkMatrixEntry` 清單

    Why:
    - 掃描案例應重用 matrix workflow，而不是維護另一套逐點執行器
    """
    if sweep not in {"aoa", "re", "ma"}:
        raise ValueError(f"sweep must be 'aoa', 're', or 'ma', got {sweep}")
    if not values:
        raise ValueError("values must contain at least one sweep point")

    sweep_values = [float(value) for value in values]
    if sweep == "re" and any(value <= 0.0 for value in sweep_values):
        raise ValueError("Re sweep values must all be positive")
    sweep_values = sorted(sweep_values)

    root_output = None if output_dir is None else Path(output_dir)
    entries: list[BenchmarkMatrixEntry] = []
    for value in sweep_values:
        current_aoa = value if sweep == "aoa" else aoa
        current_re = value if sweep == "re" else re
        current_ma = value if sweep == "ma" else ma

        case_output = None
        if save_cases and root_output is not None:
            case_tag = (
                f"ma{int(round(current_ma * 100)):03d}_"
                f"re{int(round(current_re)):05d}_"
                f"aoa{int(round(current_aoa)):03d}"
            )
            case_output = root_output / "cases" / case_tag

        overrides = {
            "ni": ni,
            "nj": nj,
            "ma": current_ma,
            "re": current_re,
            "aoa": current_aoa,
            "r_far": r_far,
            "cfl": cfl,
            "allow_transonic": allow_transonic,
            "turbulence_model": turbulence_model,
            "eddy_viscosity_ratio": eddy_viscosity_ratio,
            "turbulent_prandtl": turbulent_prandtl,
            "sa_nu_tilde_inf_ratio": sa_nu_tilde_inf_ratio,
            "wall_spacing_ratio": wall_spacing_ratio,
        }
        entries.append(
            BenchmarkMatrixEntry(
                benchmark="naca0012_ns",
                steps=steps,
                sample_interval=max(int(report), 1),
                output_dir=case_output,
                save_state=save_cases,
                save_history=save_cases,
                overrides=overrides,
                history_params={
                    "benchmark": "naca0012_ns",
                    "sweep": sweep,
                    "sweep_value": value,
                    "report": report,
                    "tol": tol,
                    "save_interval": save_interval,
                    "plateau_window": plateau_window,
                    "shock_tol": shock_tol,
                    "mach_tol": mach_tol,
                    **overrides,
                },
            )
        )
    return entries


def build_transonic_bump_sweep_preset(
    *,
    sweep: str,
    values: list[float] | tuple[float, ...],
    ni: int = 180,
    nj: int = 60,
    ma: float = 0.675,
    steps: int = 2000,
    length: float = 3.0,
    height: float = 1.0,
    bump_center: float = 1.5,
    bump_width: float = 1.0,
    bump_height: float = 0.12,
    cfl: float = 0.35,
    report: int = 200,
    tol: float = 1e-5,
    plateau_window: int = 5,
    shock_tol: float = 0.02,
    mach_tol: float = 0.03,
    output_dir: str | Path | None = None,
    save_cases: bool = False,
) -> list[BenchmarkMatrixEntry]:
    """
    建立 transonic bump sweep 的 matrix preset。
    """
    if sweep not in {"ma", "bump_height"}:
        raise ValueError(f"sweep must be 'ma' or 'bump_height', got {sweep}")
    if not values:
        raise ValueError("values must contain at least one sweep point")

    sweep_values = sorted(float(value) for value in values)
    root_output = None if output_dir is None else Path(output_dir)
    entries: list[BenchmarkMatrixEntry] = []
    for value in sweep_values:
        case_ma = value if sweep == "ma" else ma
        case_bump_height = value if sweep == "bump_height" else bump_height

        case_output = None
        if save_cases and root_output is not None:
            case_output = root_output / f"{sweep}_{value:.4f}".replace(".", "p")

        overrides = {
            "ni": ni,
            "nj": nj,
            "ma": case_ma,
            "length": length,
            "height": height,
            "bump_center": bump_center,
            "bump_width": bump_width,
            "bump_height": case_bump_height,
            "cfl": cfl,
        }
        entries.append(
            BenchmarkMatrixEntry(
                benchmark="transonic_bump_euler",
                steps=steps,
                sample_interval=max(int(report), 1),
                output_dir=case_output,
                save_state=save_cases,
                save_history=save_cases,
                overrides=overrides,
                history_params={
                    "benchmark": "transonic_bump_euler",
                    "sweep": sweep,
                    "sweep_value": value,
                    "report": report,
                    "tol": tol,
                    "plateau_window": plateau_window,
                    "shock_tol": shock_tol,
                    "mach_tol": mach_tol,
                    **overrides,
                },
            )
        )
    return entries
