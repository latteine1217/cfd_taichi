"""
NACA0012 Laminar Navier-Stokes Sweep
===================================

What: 對 NACA0012 黏性翼型案例做 AOA / Reynolds / Mach 掃描，並自動整理極線與 shock 輸出
Why:  單一工作點只能看局部結果；參數掃描才能建立極線、辨識黏性與壓力分量趨勢
      掃描執行現在應重用 benchmark matrix preset，而不是維持獨立逐點 workflow
When: 適用於 baseline laminar 與 experimental transonic/high-Re 探索，不建議直接作工程預測
"""

import os

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from cfd_taichi import (
    build_naca0012_ns_sweep_preset,
    run_benchmark_matrix,
    save_benchmark_matrix_payload,
)


def parse_sweep_values(values_text: str | None, start: float, stop: float, step: float) -> list[float]:
    """
    解析 sweep 取樣值

    What: 支援明確值列表，或由 start/stop/step 生成含終點的等間距序列
    Why:  參數掃描通常需要可重現的固定樣本點，不能依賴模糊的互動輸入
    """
    if values_text:
        values = [float(token.strip()) for token in values_text.split(",") if token.strip()]
        if not values:
            raise ValueError("values must contain at least one numeric entry")
        return values

    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"stop must be >= start, got start={start}, stop={stop}")

    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = [start + k * step for k in range(count)]
    if values[-1] < stop - 1e-10:
        values.append(stop)
    return [float(value) for value in values]


def format_case_tag(ma: float, re: float, aoa: float) -> str:
    """
    生成單一工作點標籤
    """
    return f"ma{int(round(ma * 100)):03d}_re{int(round(re)):05d}_aoa{int(round(aoa)):03d}"


def save_sweep_outputs(output_dir: str, sweep: str, results: list[dict], params: dict):
    """
    保存 sweep 摘要檔

    What: 同時保存 CSV 與 NPY，讓人類與程式都能直接消費 sweep 結果
    Why:  極線研究需要後續比對與再繪圖，單靠 console output 不可重現
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "sweep_results.npy"), {
        "mode": sweep,
        "params": params,
        "results": results,
    })

    fieldnames = [
        "aoa", "re", "ma",
        "CL", "CD", "CL_p", "CD_p", "CL_v", "CD_v",
        "converged", "final_step", "residual",
        "mass_error", "mach_max", "supersonic_fraction",
        "entropy_rise_max", "shock_sensor_max",
        "shock_peak_x", "shock_peak_y", "shock_plateau",
        "shock_peak_span", "mach_peak_span",
        "cfl_est", "runtime_seconds", "transonic_mode",
        "turbulence_model", "eddy_viscosity_ratio", "turbulent_prandtl",
        "sa_nu_tilde_inf_ratio",
        "mu", "mu_t", "mu_eff",
        "output_dir",
    ]
    with open(os.path.join(output_dir, "sweep_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_series(
        x_values: np.ndarray,
        y_values: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: str):
    """
    繪製單一 sweep 指標曲線
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, y_values, "ko-", lw=1.6, ms=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def write_sweep_figures(output_dir: str, sweep: str, results: list[dict], ma: float, re: float, aoa: float):
    """
    生成 sweep 對應的極線與係數圖
    """
    from fvm_taichi import plot_coeff_vs_aoa, plot_coeff_vs_re, plot_force_polar

    if sweep == "aoa":
        plot_coeff_vs_aoa(
            results, coeff="CL", ma=ma, re=re,
            save_path=os.path.join(output_dir, f"cl_aoa_ma{int(round(ma * 100)):03d}_re{int(round(re)):05d}.png"),
        )
        plot_coeff_vs_aoa(
            results, coeff="CD", ma=ma, re=re,
            save_path=os.path.join(output_dir, f"cd_aoa_ma{int(round(ma * 100)):03d}_re{int(round(re)):05d}.png"),
        )
        plot_force_polar(
            results,
            sweep_key="aoa",
            sweep_label="AOA (deg)",
            ma=ma,
            fixed_label=f"Re={re:.0f}",
            save_path=os.path.join(output_dir, f"polar_cl_cd_ma{int(round(ma * 100)):03d}_re{int(round(re)):05d}.png"),
        )
        return

    if sweep == "ma":
        results_sorted = sorted(results, key=lambda row: row["ma"])
        ma_values = np.array([row["ma"] for row in results_sorted], dtype=float)
        plot_series(
            ma_values,
            np.array([row["shock_peak_x"] for row in results_sorted], dtype=float),
            xlabel="Freestream Mach  Ma",
            ylabel="Shock Peak x",
            title=f"Shock Position vs Ma — NACA0012, Re={re:.0f}, AOA={aoa:.1f} deg",
            save_path=os.path.join(output_dir, f"shock_peak_vs_ma_re{int(round(re)):05d}_aoa{int(round(aoa)):03d}.png"),
        )
        plot_series(
            ma_values,
            np.array([row["mach_max"] for row in results_sorted], dtype=float),
            xlabel="Freestream Mach  Ma",
            ylabel="Max Mach",
            title=f"Max Mach vs Ma — NACA0012, Re={re:.0f}, AOA={aoa:.1f} deg",
            save_path=os.path.join(output_dir, f"mach_max_vs_ma_re{int(round(re)):05d}_aoa{int(round(aoa)):03d}.png"),
        )
        plot_series(
            ma_values,
            100.0 * np.array([row["supersonic_fraction"] for row in results_sorted], dtype=float),
            xlabel="Freestream Mach  Ma",
            ylabel="Supersonic Area (%)",
            title=f"Supersonic Fraction vs Ma — NACA0012, Re={re:.0f}, AOA={aoa:.1f} deg",
            save_path=os.path.join(output_dir, f"supersonic_fraction_vs_ma_re{int(round(re)):05d}_aoa{int(round(aoa)):03d}.png"),
        )
        return

    plot_coeff_vs_re(
        results, coeff="CL", aoa_deg=aoa, ma=ma,
        save_path=os.path.join(output_dir, f"cl_re_ma{int(round(ma * 100)):03d}_aoa{int(round(aoa)):03d}.png"),
    )
    plot_coeff_vs_re(
        results, coeff="CD", aoa_deg=aoa, ma=ma,
        save_path=os.path.join(output_dir, f"cd_re_ma{int(round(ma * 100)):03d}_aoa{int(round(aoa)):03d}.png"),
    )
    plot_force_polar(
        results,
        sweep_key="re",
        sweep_label="Re",
        ma=ma,
        fixed_label=f"AOA={aoa:.1f} deg",
        save_path=os.path.join(output_dir, f"polar_cl_cd_ma{int(round(ma * 100)):03d}_aoa{int(round(aoa)):03d}.png"),
    )


def run_sweep(
        sweep: str = "aoa",
        values: list[float] | None = None,
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
        output_dir: str | None = None,
        save_cases: bool = False,
        allow_transonic: bool = False,
        turbulence_model: str = "laminar",
        eddy_viscosity_ratio: float = 0.0,
        turbulent_prandtl: float = 0.9,
        sa_nu_tilde_inf_ratio: float = 3.0,
        plateau_window: int = 5,
        shock_tol: float = 0.03,
        mach_tol: float = 0.05,
):
    """
    執行 AOA 或 Re sweep

    What: 逐一呼叫單 case NS solver，收集 CL/CD 與診斷量
    Why:  單一入口維持 case 與 sweep 的物理契約一致，避免兩套 driver 漂移
    """
    if sweep not in {"aoa", "re", "ma"}:
        raise ValueError(f"sweep must be 'aoa', 're', or 'ma', got {sweep}")
    if values is None or len(values) == 0:
        raise ValueError("values must contain at least one sweep point")

    values = [float(value) for value in values]
    if sweep in {"aoa", "ma"}:
        values = sorted(values)
    else:
        values = sorted(values)
        if any(value <= 0.0 for value in values):
            raise ValueError("Re sweep values must all be positive")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 92)
    print("  NACA0012 Navier-Stokes Sweep")
    print("=" * 92)
    print(f"  Sweep mode: {sweep}")
    print(f"  Sweep values: {values}")
    print(f"  Grid: ni={ni} x nj={nj}, R_far={r_far}, steps={steps}, CFL={cfl}, tol={tol:.1e}")
    print(f"  Fixed params: Ma={ma:.3f}, Re={re:.1f}, AOA={aoa:.2f} deg")
    print(f"  Save cases: {save_cases}, Output: {output_dir or '(disabled)'}")
    if (ma >= 0.7) or (sweep == "ma" and max(values) >= 0.7):
        print(f"  Transonic: experimental={allow_transonic}")
    print("-" * 92)
    print(f"  {'value':>10}  {'step':>7}  {'conv':>5}  {'res':>10}  "
          f"{'CL':>8}  {'CD':>9}  {'CD_p':>9}  {'CD_v':>9}  {'mass_err':>10}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*10}  "
          f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")

    sweep_start = time.time()
    entries = build_naca0012_ns_sweep_preset(
        sweep=sweep,
        values=values,
        ni=ni,
        nj=nj,
        ma=ma,
        re=re,
        aoa=aoa,
        steps=steps,
        r_far=r_far,
        cfl=cfl,
        report=report,
        tol=tol,
        save_interval=save_interval,
        wall_spacing_ratio=wall_spacing_ratio,
        output_dir=output_dir,
        save_cases=save_cases,
        allow_transonic=allow_transonic,
        turbulence_model=turbulence_model,
        eddy_viscosity_ratio=eddy_viscosity_ratio,
        turbulent_prandtl=turbulent_prandtl,
        sa_nu_tilde_inf_ratio=sa_nu_tilde_inf_ratio,
        plateau_window=plateau_window,
        shock_tol=shock_tol,
        mach_tol=mach_tol,
    )
    matrix_results = run_benchmark_matrix(entries)
    results = []

    for value, entry, matrix_result in zip(values, entries, matrix_results, strict=True):
        if matrix_result.error is not None:
            raise RuntimeError(f"Case failed during {sweep} sweep at value={value}: {matrix_result.error}")

        current_aoa = float(entry.overrides["aoa"])
        current_re = float(entry.overrides["re"])
        current_ma = float(entry.overrides["ma"])
        diagnostics = matrix_result.diagnostics
        case_output_dir = None
        if matrix_result.state_path is not None:
            case_output_dir = os.path.dirname(matrix_result.state_path)
        elif entry.output_dir is not None:
            case_output_dir = str(entry.output_dir)

        residual = float(diagnostics.get("residual", np.nan))
        row = {
            "aoa": current_aoa,
            "re": current_re,
            "ma": current_ma,
            "CL": float(diagnostics.get("lift_coefficient", np.nan)),
            "CD": float(diagnostics.get("drag_coefficient", np.nan)),
            "CL_p": float("nan"),
            "CD_p": float(diagnostics.get("pressure_drag_coefficient", np.nan)),
            "CL_v": float("nan"),
            "CD_v": float(diagnostics.get("viscous_drag_coefficient", np.nan)),
            "converged": bool(np.isfinite(residual) and residual <= tol),
            "final_step": int(matrix_result.step),
            "residual": residual,
            "mass_error": float(diagnostics.get("mass_error", np.nan)),
            "mach_max": float(diagnostics.get("mach_max", np.nan)),
            "supersonic_fraction": float(diagnostics.get("supersonic_fraction", np.nan)),
            "entropy_rise_max": float(diagnostics.get("entropy_rise_max", np.nan)),
            "shock_sensor_max": float(diagnostics.get("shock_sensor_max", np.nan)),
            "shock_peak_x": float(diagnostics.get("shock_peak_x", np.nan)),
            "shock_peak_y": float(diagnostics.get("shock_peak_y", np.nan)),
            "shock_plateau": False,
            "shock_peak_span": float("nan"),
            "mach_peak_span": float("nan"),
            "cfl_est": float(diagnostics.get("cfl_est", np.nan)),
            "runtime_seconds": float("nan"),
            "turbulence_model": str(diagnostics.get("turbulence_model", turbulence_model)),
            "eddy_viscosity_ratio": float(entry.overrides["eddy_viscosity_ratio"]),
            "turbulent_prandtl": float(entry.overrides["turbulent_prandtl"]),
            "sa_nu_tilde_inf_ratio": float(entry.overrides["sa_nu_tilde_inf_ratio"]),
            "mu": float(diagnostics.get("mu", np.nan)),
            "mu_t": float(diagnostics.get("mu_t", np.nan)),
            "mu_eff": float(diagnostics.get("mu_eff", np.nan)),
            "output_dir": case_output_dir,
            "walltime_seconds": float("nan"),
            "transonic_mode": bool(current_ma >= 0.7),
        }
        results.append(row)

        if sweep == "re":
            value_text = f"{value:10.1f}"
        else:
            value_text = f"{value:10.3f}"
        conv_text = "Yes" if row["converged"] else "No"
        print(f"  {value_text}  {row['final_step']:>7}  {conv_text:>5}  {row['residual']:>10.3e}  "
              f"{row['CL']:>8.4f}  {row['CD']:>9.5f}  {row['CD_p']:>9.5f}  {row['CD_v']:>9.5f}  "
              f"{row['mass_error']:>10.3e}")

    total_elapsed = time.time() - sweep_start
    print("-" * 92)
    print(f"  Completed {len(results)} cases in {total_elapsed:.2f} s")
    print("=" * 92)

    if output_dir:
        params = {
            "sweep": sweep,
            "values": values,
            "ni": ni,
            "nj": nj,
            "ma": ma,
            "re": re,
            "aoa": aoa,
            "steps": steps,
            "r_far": r_far,
            "cfl": cfl,
            "report": report,
            "tol": tol,
            "save_interval": save_interval,
            "wall_spacing_ratio": wall_spacing_ratio,
            "save_cases": save_cases,
            "allow_transonic": allow_transonic,
            "turbulence_model": turbulence_model,
            "eddy_viscosity_ratio": eddy_viscosity_ratio,
            "turbulent_prandtl": turbulent_prandtl,
            "sa_nu_tilde_inf_ratio": sa_nu_tilde_inf_ratio,
            "plateau_window": plateau_window,
            "shock_tol": shock_tol,
            "mach_tol": mach_tol,
            "total_runtime_seconds": total_elapsed,
        }
        save_sweep_outputs(output_dir, sweep, results, params)
        save_benchmark_matrix_payload(
            os.path.join(output_dir, "matrix_summary.npy"),
            matrix_results,
            additional_payload={
                "preset": "naca0012_ns_sweep",
                "sweep": sweep,
                "values": np.asarray(values, dtype=np.float64),
            },
        )
        write_sweep_figures(output_dir, sweep, results, ma=ma, re=re, aoa=aoa)

    return results


def main():
    parser = argparse.ArgumentParser(description="NACA0012 Navier-Stokes sweep")
    parser.add_argument("--sweep", choices=["aoa", "re", "ma"], default="aoa", help="Sweep variable")
    parser.add_argument("--values", type=str, default="", help="Comma-separated sweep values; overrides start/stop/step")
    parser.add_argument("--start", type=float, default=0.0, help="Sweep start value")
    parser.add_argument("--stop", type=float, default=6.0, help="Sweep stop value")
    parser.add_argument("--step", type=float, default=2.0, help="Sweep increment")
    parser.add_argument("--ni", type=int, default=120, help="O-grid circumferential cells")
    parser.add_argument("--nj", type=int, default=40, help="O-grid radial cells")
    parser.add_argument("--ma", type=float, default=0.12, help="Freestream Mach number")
    parser.add_argument("--re", type=float, default=300.0, help="Fixed Reynolds number for AOA sweep")
    parser.add_argument("--aoa", type=float, default=3.0, help="Fixed AOA for Re sweep")
    parser.add_argument("--steps", type=int, default=6000, help="Maximum steps per case")
    parser.add_argument("--rfar", type=float, default=15.0, help="Far-field radius in chords")
    parser.add_argument("--cfl", type=float, default=0.08, help="Explicit CFL number")
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=5e-5, help="Residual RMS tolerance")
    parser.add_argument("--save_interval", type=int, default=1000, help="State save interval for per-case outputs")
    parser.add_argument(
        "--wall_spacing_ratio",
        type=float,
        default=1.0,
        help="First radial spacing relative to linear average 1/nj; <1 clusters near wall",
    )
    parser.add_argument("--save_cases", action="store_true", help="Save full outputs for each work point")
    parser.add_argument("--transonic", action="store_true", help="Enable experimental transonic path for Ma >= 0.7")
    parser.add_argument(
        "--turbulence",
        choices=["laminar", "constant_eddy_viscosity", "spalart_allmaras"],
        default="laminar",
        help="Transport closure model",
    )
    parser.add_argument(
        "--eddy_ratio",
        type=float,
        default=0.0,
        help="Eddy viscosity ratio mu_t / mu for constant_eddy_viscosity",
    )
    parser.add_argument(
        "--pr_t",
        type=float,
        default=0.9,
        help="Turbulent Prandtl number for constant_eddy_viscosity",
    )
    parser.add_argument(
        "--sa_nu_ratio",
        type=float,
        default=3.0,
        help="Freestream nu_tilde / nu ratio for Spalart-Allmaras",
    )
    parser.add_argument("--plateau_window", type=int, default=5, help="History window for shock plateau check")
    parser.add_argument("--shock_tol", type=float, default=0.03, help="Shock-position plateau tolerance")
    parser.add_argument("--mach_tol", type=float, default=0.05, help="Max-Mach plateau tolerance")
    parser.add_argument(
        "--output",
        type=str,
        default="output_naca0012_ns_sweep",
        help="Sweep output directory (empty string disables file output)",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        values = parse_sweep_values(
            values_text=args.values if args.values else None,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        run_sweep(
            sweep=args.sweep,
            values=values,
            ni=args.ni,
            nj=args.nj,
            ma=args.ma,
            re=args.re,
            aoa=args.aoa,
            steps=args.steps,
            r_far=args.rfar,
            cfl=args.cfl,
            report=args.report,
            tol=args.tol,
            save_interval=args.save_interval,
            wall_spacing_ratio=args.wall_spacing_ratio,
            output_dir=args.output if args.output else None,
            save_cases=args.save_cases,
            allow_transonic=args.transonic,
            turbulence_model=args.turbulence,
            eddy_viscosity_ratio=args.eddy_ratio,
            turbulent_prandtl=args.pr_t,
            sa_nu_tilde_inf_ratio=args.sa_nu_ratio,
            plateau_window=args.plateau_window,
            shock_tol=args.shock_tol,
            mach_tol=args.mach_tol,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ NACA0012 NS sweep failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
