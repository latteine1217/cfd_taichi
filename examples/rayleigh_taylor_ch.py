"""
Rayleigh-Taylor Instability (Cahn-Hilliard + Incompressible LBM)
=================================================================
"""

import argparse
import os
import time

import numpy as np
import taichi as ti

from lbm_taichi.core import CHDiagnostics, CHLBMSolver


def _build_initial_phi(
    nx: int,
    ny: int,
    interface_ratio: float,
    thickness: float,
    amp: float,
    mode: int,
):
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    y0 = interface_ratio * (ny - 1)
    if amp > 0 and mode > 0:
        y0 = y0 + amp * np.sin(2.0 * np.pi * mode * X / max(1.0, nx))
    return np.tanh((Y - y0) / max(1.0, thickness)).astype(np.float32)


def run_rayleigh_taylor_ch(
    res_y: int = 256,
    aspect_ratio: float = 0.5,
    tau: float = 0.85,
    mobility: float = 0.0015,
    a: float = 0.04,
    kappa: float = 0.02,
    gravity: float = 3.0e-5,
    rho_heavy: float = 1.2,
    rho_light: float = 0.2,
    interface_ratio: float = 0.5,
    thickness_ratio: float = 0.012,
    perturb_amp_ratio: float = 0.08,
    perturb_mode: int = 1,
    u_cap: float = 0.2,
    force_cap: float = 2e-3,
    boundary_y: str = "freeslip",
    low_mach_corr: float = 0.0,
    compressibility: float = 0.0,
    steps: int = 50000,
    interval: int = 1000,
    output_dir: str = "output_rayleigh_taylor_ch",
):
    nx = int(aspect_ratio * res_y)
    ny = res_y
    thickness = max(1.0, thickness_ratio * ny)
    amp = max(0.0, perturb_amp_ratio * ny)
    g = -abs(gravity)

    print("=" * 70)
    print(" " * 10 + "RAYLEIGH-TAYLOR (CAHN-HILLIARD + INC-LBM)")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}")
    print(f"Interface: y/ny = {interface_ratio:.2f}")
    print(f"tau={tau:.3f}, M={mobility:.3e}, a={a:.3e}, kappa={kappa:.3e}")
    print(f"Gravity: g={g:.2e}")
    print(f"Density: rho_heavy={rho_heavy:.3f}, rho_light={rho_light:.3f}")
    print(
        f"Perturbation: amp={amp:.2f} cells, mode={perturb_mode}, thickness={thickness:.2f}"
    )
    print(
        f"Stabilizers: u_cap={u_cap:.3f}, force_cap={force_cap:.2e}, "
        f"low_mach_corr={low_mach_corr:.2f}, compressibility={compressibility:.2f}"
    )

    solver = CHLBMSolver(
        nx=nx,
        ny=ny,
        tau=tau,
        mobility=mobility,
        a=a,
        kappa=kappa,
        rho_heavy=rho_heavy,
        rho_light=rho_light,
        gravity=(0.0, g),
        u_cap=u_cap,
        force_cap=force_cap,
        boundary_y=boundary_y,
        low_mach_corr=low_mach_corr,
        compressibility=compressibility,
    )

    phi0 = _build_initial_phi(
        nx=nx,
        ny=ny,
        interface_ratio=interface_ratio,
        thickness=thickness,
        amp=amp,
        mode=max(1, int(perturb_mode)),
    )
    solver.set_initial_phi(phi0)

    diag = CHDiagnostics(solver, output_dir=output_dir)
    headers = diag.print_header()
    diag.set_initial_baseline()
    diag.save_data(0, 0.0)

    print("\n=== Boundary Conditions ===")
    print("X-direction : Periodic")
    print(f"Y-direction : {boundary_y} (with no-flux phi/mu)")
    print("\n=== Simulation Start ===")

    start = time.time()
    run_start = start
    for step in range(1, steps + 1):
        solver.step()
        if step % 100 == 0:
            elapsed = time.time() - run_start
            speed = step / elapsed if elapsed > 1e-12 else 0.0
            eta = (steps - step) / speed if speed > 1e-12 else 0.0
            row = diag.print_step_info(step, speed, eta)
            if step % 500 == 0:
                diag.history.append(row)
            if step % interval == 0:
                diag.save_data(step, float(step))
        elif step % interval == 0:
            diag.save_data(step, float(step))

    total = time.time() - start
    print(f"\n--- Simulation completed in {total:.2f} seconds ---")

    history_file = diag.save_history(
        params={
            "res_y": res_y,
            "aspect_ratio": aspect_ratio,
            "tau": tau,
            "mobility": mobility,
            "a": a,
            "kappa": kappa,
            "gravity": gravity,
            "rho_heavy": rho_heavy,
            "rho_light": rho_light,
            "interface_ratio": interface_ratio,
            "thickness_ratio": thickness_ratio,
            "perturb_amp_ratio": perturb_amp_ratio,
            "perturb_mode": perturb_mode,
            "u_cap": u_cap,
            "force_cap": force_cap,
            "boundary_y": boundary_y,
            "low_mach_corr": low_mach_corr,
            "compressibility": compressibility,
        }
    )
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Rayleigh-Taylor with Cahn-Hilliard + Incompressible LBM"
    )
    parser.add_argument("--res", type=int, default=256, help="Y resolution")
    parser.add_argument("--aspect", type=float, default=0.5, help="nx = aspect * res")
    parser.add_argument("--tau", type=float, default=0.85, help="LBM relaxation time")
    parser.add_argument("--mobility", type=float, default=0.0015, help="CH mobility")
    parser.add_argument("--a", type=float, default=0.04, help="Bulk energy coefficient")
    parser.add_argument("--kappa", type=float, default=0.02, help="Gradient energy")
    parser.add_argument("--g", type=float, default=3e-5, help="Gravity magnitude")
    parser.add_argument("--rho_heavy", type=float, default=1.2, help="Top heavy density")
    parser.add_argument("--rho_light", type=float, default=0.2, help="Bottom light density")
    parser.add_argument("--interface", type=float, default=0.5, help="Interface ratio")
    parser.add_argument("--thickness", type=float, default=0.012, help="Thickness ratio")
    parser.add_argument("--perturb_amp", type=float, default=0.08, help="Amplitude ratio")
    parser.add_argument("--perturb_mode", type=int, default=1, help="X perturb mode")
    parser.add_argument("--u_cap", type=float, default=0.2, help="Velocity cap")
    parser.add_argument("--force_cap", type=float, default=2e-3, help="Force cap")
    parser.add_argument(
        "--boundary_y",
        type=str,
        default="freeslip",
        choices=["noslip", "freeslip"],
        help="Y boundary mode for flow",
    )
    parser.add_argument(
        "--low_mach_corr",
        type=float,
        default=0.0,
        help="Low-Mach divergence correction strength",
    )
    parser.add_argument(
        "--compressibility",
        type=float,
        default=0.0,
        help="Compressibility factor in feq (0=incompressible)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=1000, help="Save interval")
    parser.add_argument(
        "--output",
        type=str,
        default="output_rayleigh_taylor_ch",
        help="Output directory",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_rayleigh_taylor_ch(
        res_y=args.res,
        aspect_ratio=args.aspect,
        tau=args.tau,
        mobility=args.mobility,
        a=args.a,
        kappa=args.kappa,
        gravity=args.g,
        rho_heavy=args.rho_heavy,
        rho_light=args.rho_light,
        interface_ratio=args.interface,
        thickness_ratio=args.thickness,
        perturb_amp_ratio=args.perturb_amp,
        perturb_mode=args.perturb_mode,
        u_cap=args.u_cap,
        force_cap=args.force_cap,
        boundary_y=args.boundary_y,
        low_mach_corr=args.low_mach_corr,
        compressibility=args.compressibility,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
