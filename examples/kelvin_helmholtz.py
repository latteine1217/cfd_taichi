"""
Kelvin-Helmholtz Instability
============================

上下兩層不同密度流體的剪切不穩定性。

Why 這個 case?
- 驗證剪切層在 LBM 中的非定常演化
- 測試密度分層對速度剪切的不穩定影響
- 作為高 Re 非定常案例的基準

物理現象：
- 上下兩層流體密度不同
- 速度方向相反，剪切層產生 Kelvin-Helmholtz 渦卷
- 小擾動在界面處成長並形成渦列

⚠️ 限制：本求解器為單一不可壓流體模型
密度分層僅作為初始條件，沒有多相張力或重力耦合。
"""

import argparse
import os
import sys
import time

import numpy as np
import taichi as ti

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from lbm_taichi.core import BoundaryConditions, Diagnostics, LBMSolver


def _build_kelvin_helmholtz_fields(
    nx: int,
    ny: int,
    u0: float,
    rho_top: float,
    rho_bottom: float,
    shear_thickness: float,
    perturb_amp: float,
    perturb_mode: int,
    perturb_sigma: float,
):
    """
    生成 KH 不穩定性的初始場

    What: 建立密度分層 + 剪切速度 + 界面擾動
    Why: 使用平滑 tanh 剪切層避免數值振盪
    When: 初始化 Kelvin-Helmholtz case
    """
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    y0 = 0.5 * (ny - 1)
    delta = max(1.0, shear_thickness)

    tanh_arg = (Y - y0) / delta
    profile = np.tanh(tanh_arg)

    rho = 0.5 * (rho_top + rho_bottom) + 0.5 * (rho_top - rho_bottom) * profile
    u_x = u0 * profile

    u_y = np.zeros((nx, ny), dtype=np.float32)

    u = np.zeros((nx, ny, 2), dtype=np.float32)
    u[:, :, 0] = u_x
    u[:, :, 1] = u_y

    return rho.astype(np.float32), u


def run_kelvin_helmholtz(
    res_y: int = 128,
    aspect_ratio: float = 4.0,
    re: float = 1000.0,
    u0: float = 0.08,
    rho_top: float = 2.0,
    rho_bottom: float = 1.0,
    shear_thickness: float = 6.0,
    perturb_amp: float = 0.01,
    perturb_mode: int = 2,
    perturb_sigma: float = 12.0,
    cs: float = 0.16,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    output_dir: str = "output_kelvin_helmholtz",
    vtk_output: bool = False,
    collision_model: str = "mrt",
):
    """
    執行 Kelvin-Helmholtz 不穩定性模擬

    Args:
        res_y: Y 方向解析度
        aspect_ratio: 計算域長寬比 (nx = aspect_ratio * res_y)
        re: Reynolds 數 (以剪切厚度為特徵長度)
        u0: 上下層速度大小 (上層 +u0, 下層 -u0)
        rho_top: 上層密度
        rho_bottom: 下層密度
        shear_thickness: 剪切層厚度 (lattice units)
        perturb_amp: 擾動速度幅值
        perturb_mode: x 方向擾動模態數
        perturb_sigma: 擾動垂向衰減尺度
        cs: LES 啟用旗標 (<=0 表示不使用 LES；動態 Smagorinsky 自動估計)
        steps: 總步數
        interval: 儲存間隔
        tol: 收斂容差
        output_dir: 輸出目錄
    """
    nx = int(aspect_ratio * res_y)
    ny = res_y

    print("=" * 70)
    print(" " * 18 + "KELVIN-HELMHOLTZ INSTABILITY")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}")
    print(f"Shear thickness: {shear_thickness:.2f}")

    u_ref = max(abs(u0), 1e-6)
    length_scale = max(2.0, shear_thickness)
    tau_est = 0.5 + 3.0 * u_ref * length_scale / re
    min_tau = 0.505
    if tau_est < min_tau:
        re_limit = 3.0 * u_ref * length_scale / (min_tau - 0.5)
        print(
            "\n⚠️  Reynolds number too high for stability. "
            f"Adjusting Re: {re:.1f} → {re_limit:.1f} (tau≈{min_tau:.3f})"
        )
        re = re_limit

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_ref,
        length_scale=length_scale,
        cs=cs,
        collision_model=collision_model,
    )

    bc = BoundaryConditions(solver)
    bc.add_periodic_boundary("x")
    bc.add_neumann_outflow("top")
    bc.add_neumann_outflow("bottom")

    print("\nBoundary Conditions:")
    print("  X-direction : Periodic")
    print("  Y-direction : Neumann Outflow")

    print("\nInitializing Kelvin-Helmholtz fields...")
    rho_field, u_field = _build_kelvin_helmholtz_fields(
        nx=nx,
        ny=ny,
        u0=u0,
        rho_top=rho_top,
        rho_bottom=rho_bottom,
        shear_thickness=shear_thickness,
        perturb_amp=perturb_amp,
        perturb_mode=perturb_mode,
        perturb_sigma=perturb_sigma,
    )

    rho_g = np.ones((nx + 2, ny + 2), dtype=rho_field.dtype)
    rho_g[1 : nx + 1, 1 : ny + 1] = rho_field
    solver.rho.from_numpy(rho_g)
    solver.prev_rho.from_numpy(rho_g)
    solver._apply_velocity_field(u_field)

    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    diag = Diagnostics(solver, output_dir=output_dir, output_vtk=vtk_output)

    print(f"Reynolds Number: {re}")
    print(f"Reference Velocity: {u_ref}")
    print(f"Viscosity: {solver.nu:.6f}")
    print(f"Density Ratio (top/bottom): {rho_top}/{rho_bottom}")
    print("\nStarting simulation...")

    headers = diag.print_header(include_forces=False)

    history_steps = []
    history_mass = []
    history_mom_x = []
    history_mom_y = []
    history_umax = []
    history_cfl = []

    global_start = time.time()

    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]
    solver.initial_KE[None] = solver.total_KE[None]
    diag.save_data(0, additional_data={"time": 0.0})

    sim_start = time.time()

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % 100 == 0:
            solver._update_macro(f_dst)
            solver._update_diagnostics()
            ti.sync()

            total_elapsed = time.time() - sim_start
            speed = step / total_elapsed if total_elapsed > 1e-6 else 0.0
            remaining_steps = steps - step
            eta_seconds = remaining_steps / speed if speed > 0 else 0.0

            row = diag.print_step_info(
                step, speed, eta_seconds, include_forces=False, f_field=f_dst
            )

            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                diag.save_data(step, additional_data={"time": float(step)})

            if step % 1000 == 0:
                solver.check_cfl_condition(warn_only=True)

            res = diag.get_residuals()
            diag_vals = solver.get_diagnostics()
            history_steps.append(step)
            history_mass.append(
                abs(diag_vals["total_mass"] - diag_vals["initial_mass"])
                / (diag_vals["initial_mass"] + 1e-12)
            )
            history_mom_x.append(res["R_u"])
            history_mom_y.append(res["R_v"])
            history_umax.append(diag_vals["max_u"])
            history_cfl.append(diag_vals["max_u"])

            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                if step % interval != 0:
                    diag.save_data(step, additional_data={"time": float(step)})
                break

        elif step % interval == 0:
            solver._update_macro(f_dst)
            diag.save_data(step, additional_data={"time": float(step)})

    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    history_file = os.path.join(output_dir, "history.npy")
    history_payload = np.array(
        {
            "steps": history_steps,
            "mass_error": history_mass,
            "mom_res_x": history_mom_x,
            "mom_res_y": history_mom_y,
            "u_max": history_umax,
            "cfl": history_cfl,
            "params": {
                "res_y": res_y,
                "aspect_ratio": aspect_ratio,
                "re": re,
                "u0": u0,
                "rho_top": rho_top,
                "rho_bottom": rho_bottom,
                "shear_thickness": shear_thickness,
                "perturb_amp": perturb_amp,
                "perturb_mode": perturb_mode,
                "perturb_sigma": perturb_sigma,
                "cs": cs,
            },
        },
        dtype=object,
    )
    np.save(history_file, history_payload, allow_pickle=True)
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Kelvin-Helmholtz Instability")
    parser.add_argument("--res", type=int, default=128, help="Y resolution")
    parser.add_argument(
        "--aspect",
        type=float,
        default=4.0,
        help="Domain aspect ratio (nx = aspect * res)",
    )
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument(
        "--u0", type=float, default=0.08, help="Shear velocity magnitude"
    )
    parser.add_argument("--rho_top", type=float, default=2.0, help="Top layer density")
    parser.add_argument(
        "--rho_bottom", type=float, default=1.0, help="Bottom layer density"
    )
    parser.add_argument(
        "--delta", type=float, default=6.0, help="Shear layer thickness (lattice units)"
    )
    parser.add_argument(
        "--perturb_amp", type=float, default=0.01, help="Initial perturbation amplitude"
    )
    parser.add_argument(
        "--perturb_mode", type=int, default=2, help="Perturbation mode in x"
    )
    parser.add_argument(
        "--perturb_sigma",
        type=float,
        default=12.0,
        help="Perturbation vertical decay scale",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=0.16,
        help="LES enable flag (<=0 disables dynamic Smagorinsky)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument(
        "--output",
        type=str,
        default="output_kelvin_helmholtz",
        help="Output directory",
    )
    parser.add_argument(
        "--vtk",
        action="store_true",
        help="Export VTK (.vti) alongside npy outputs",
    )
    parser.add_argument(
        "--collision",
        type=str,
        default="mrt",
        choices=["mrt", "bgk", "elbm", "emrt"],
        help="Collision model: mrt, bgk, elbm, or emrt",
    )

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_kelvin_helmholtz(
        res_y=args.res,
        aspect_ratio=args.aspect,
        re=args.re,
        u0=args.u0,
        rho_top=args.rho_top,
        rho_bottom=args.rho_bottom,
        shear_thickness=args.delta,
        perturb_amp=args.perturb_amp,
        perturb_mode=args.perturb_mode,
        perturb_sigma=args.perturb_sigma,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        output_dir=args.output,
        vtk_output=args.vtk,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
