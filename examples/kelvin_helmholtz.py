"""
Kelvin-Helmholtz Instability (Single Shear Layer)
=================================================

上下兩層剪切界面（雙剪切層）的 Kelvin-Helmholtz 不穩定性。

Why 這個 case?
- 驗證剪切層在 LBM 中的非定常演化
- 測試速度剪切對渦卷形成的影響
- 作為高 Re 非定常案例的基準

物理現象：
- 上下兩層流體速度方向相反（均一密度）
- 剪切層產生 Kelvin-Helmholtz 渦卷
- 小擾動在界面處成長並形成渦列

經典設置（雙剪切層）：
- X 方向：週期邊界（無限延伸剪切層）
- Y 方向：自由滑移（模擬無限域）
- 擾動：多模態（允許自然選擇最不穩定波長）
- Re=400：層流解析，關閉 LES

⚠️ 限制：本求解器為單一不可壓流體模型
不可壓假設下密度必須均一，密度分層需多相 LBM 求解器。
"""

import argparse
import os
import time

import numpy as np
import taichi as ti

from lbm_taichi.core import BoundaryConditions, Diagnostics, LBMSolver


def _build_kelvin_helmholtz_fields(
    nx: int,
    ny: int,
    u0: float,
    rho_top: float,
    rho_bottom: float,
    shear_thickness: float,
    shear_center_ratio: float,
    perturb_amp: float,
    perturb_mode: int,
    perturb_sigma: float,
    perturb_type: str = "single",
):
    """
    生成 KH 不穩定性的初始場

    What: 建立均一密度 + 剪切速度 + 界面擾動
    Why: 使用平滑 tanh 剪切層避免數值振盪
    When: 初始化 Kelvin-Helmholtz case

    Note: 不可壓 LBM 要求 rho_top == rho_bottom，密度分層需多相求解器

    Args:
        perturb_type: 'single' 單一模態, 'multi' 多模態, 'white' 白噪聲
    """
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    delta = max(1.0, shear_thickness)
    y1 = float(shear_center_ratio * (ny - 1))
    profile = np.tanh((Y - y1) / delta)

    if abs(rho_top - rho_bottom) < 1e-6:
        rho = np.ones((nx, ny), dtype=np.float32) * rho_top
    else:
        rho = 0.5 * (rho_top + rho_bottom) + 0.5 * (rho_top - rho_bottom) * profile
    u_x = u0 * profile

    u_y = np.zeros((nx, ny), dtype=np.float32)

    if perturb_amp > 0.0:
        if perturb_sigma > 1e-6:
            envelope = np.exp(-0.5 * ((Y - y1) / perturb_sigma) ** 2)
        else:
            envelope = 1.0

        if perturb_type == "single" and perturb_mode > 0:
            phase = 2.0 * np.pi * perturb_mode * X / max(1.0, nx)
            u_y = perturb_amp * np.sin(phase) * envelope
        elif perturb_type == "multi":
            mode1 = (
                perturb_mode
                if perturb_mode > 0
                else max(2, int(nx / (8 * shear_thickness)))
            )
            mode2 = 2 * mode1
            phase1 = 2.0 * np.pi * mode1 * X / max(1.0, nx)
            phase2 = 2.0 * np.pi * mode2 * X / max(1.0, nx)
            u_y = perturb_amp * (np.sin(phase1) + 0.5 * np.sin(phase2)) * envelope
        elif perturb_type == "white":
            np.random.seed(42)
            noise = np.random.randn(nx, ny).astype(np.float32)
            noise = noise - np.mean(noise)
            noise = noise / np.std(noise) if np.std(noise) > 1e-12 else noise
            u_y = perturb_amp * noise * envelope

    u = np.zeros((nx, ny, 2), dtype=np.float32)
    u[:, :, 0] = u_x
    u[:, :, 1] = u_y

    return rho.astype(np.float32), u


def run_kelvin_helmholtz(
    res_y: int = 128,
    aspect_ratio: float = 4.0,
    re: float = 400.0,
    u0: float = 0.06,
    rho_top: float = 1.0,
    rho_bottom: float = 1.0,
    shear_center_ratio: float = 0.5,
    perturb_amp: float = 0.02,
    perturb_mode: int = 0,
    perturb_sigma: float = 0.0,
    perturb_type: str = "multi",
    cs: float = 0.0,
    steps: int = 50000,
    interval: int = 1000,
    output_dir: str = "output_kelvin_helmholtz",
    vtk_output: bool = False,
    collision_model: str = "mrt",
):
    """
    執行 Kelvin-Helmholtz 不穩定性模擬（單剪切層）

    Args:
        res_y: Y 方向解析度
        aspect_ratio: 計算域長寬比 (nx = aspect_ratio * res_y)
        re: Reynolds 數 (以剪切厚度為特徵長度)
        u0: 上下層速度大小 (上層 +u0, 下層 -u0)
        rho_top: 上層密度 (不可壓模型必須 = rho_bottom)
        rho_bottom: 下層密度 (不可壓模型必須 = rho_top)
        shear_center_ratio: 剪切層中心位置比例（0.5 表示中央單剪切層）
        perturb_amp: 擾動速度幅值 (絕對速度)
        perturb_mode: x 方向擾動模態數，預設為 int(nx/(8*shear_thickness))
        perturb_sigma: 擾動垂向衰減尺度
        perturb_type: 'single' 單一模態, 'multi' 多模態 (預設), 'white' 白噪聲
        cs: LES Smagorinsky 常數 (0.0 表示關閉 LES，Re=400 層流解析足夠)
        steps: 總步數
        interval: 儲存間隔
        output_dir: 輸出目錄
    """
    nx = int(aspect_ratio * res_y)
    ny = res_y

    shear_thickness = 0.05 * ny
    shear_center_ratio = float(np.clip(shear_center_ratio, 0.1, 0.9))

    print("=" * 70)
    print(" " * 18 + "KELVIN-HELMHOLTZ INSTABILITY")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}")
    print(f"Shear thickness: {shear_thickness:.2f}")
    print(
        f"Perturbation: type={perturb_type}, amp={perturb_amp:.4f}, "
        f"mode={perturb_mode}, sigma={perturb_sigma:.2f}"
    )
    print(f"Shear center: y/ny = {shear_center_ratio:.2f}")

    u_ref = max(abs(u0), 1e-6)
    length_scale = max(2.0, shear_thickness)
    tau_est = 0.5 + 3.0 * u_ref * length_scale / re
    if collision_model in ("bgk", "elbm"):
        min_tau = 0.53
        warn_tau = 0.535
    else:
        min_tau = 0.5005
        warn_tau = 0.505
    if tau_est < min_tau:
        re_limit = 3.0 * u_ref * length_scale / (min_tau - 0.5)
        print(
            f"\n⚠️  Warning: tau={tau_est:.4f} < {min_tau:.4f}. "
            f"Adjusting Re: {re:.1f} → {re_limit:.1f} for stability"
        )
        re = re_limit
    elif tau_est < warn_tau:
        print(
            f"\n⚠️  Warning: tau={tau_est:.4f} is near stability limit "
            f"({min_tau:.4f})."
        )

    auto_les = re > 2000.0
    cs_eff = cs if cs > 0.0 else (0.16 if auto_les else cs)

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_ref,
        length_scale=length_scale,
        cs=cs_eff,
        collision_model=collision_model,
    )
    if re > 1000.0:
        solver.set_wall_function(enabled=True, yplus_min=5.0)
        print("Wall Function: Enabled (Re > 1000, yplus_min=5.0)")
    else:
        solver.set_wall_function(enabled=False)
        print("Wall Function: Disabled (Re <= 1000)")

    bc = BoundaryConditions(solver)
    bc.add_periodic_boundary("x")
    bc.add_orlanski_outflow(location="top", relaxation=0.02)
    bc.add_orlanski_outflow(location="bottom", relaxation=0.02)

    print("\nBoundary Conditions:")
    print("  X-direction : Periodic")
    print("  Y-direction : Orlanski Outflow (top/bottom)")
    print("\nNote: Outflow Y boundaries approximate an unbounded domain")

    print("\nInitializing Kelvin-Helmholtz fields...")
    if perturb_mode == 0:
        perturb_mode = max(2, int(nx / (8 * shear_thickness)))
        if perturb_type == "multi":
            print(
                f"  Auto-calculated perturbation mode: {perturb_mode} "
                f"(wavelength ≈ 8 * shear_thickness, multi-mode perturbation)"
            )
        else:
            print(
                f"  Auto-calculated perturbation mode: {perturb_mode} "
                f"(wavelength ≈ 8 * shear_thickness)"
            )
    rho_field, u_field = _build_kelvin_helmholtz_fields(
        nx=nx,
        ny=ny,
        u0=u0,
        rho_top=rho_top,
        rho_bottom=rho_bottom,
        shear_thickness=shear_thickness,
        shear_center_ratio=shear_center_ratio,
        perturb_amp=perturb_amp,
        perturb_mode=perturb_mode,
        perturb_sigma=perturb_sigma,
        perturb_type=perturb_type,
    )

    solver.set_initial_condition(
        velocity=u_field,
        density=rho_field,
        apply_boundaries=True,
    )

    diag = Diagnostics(solver, output_dir=output_dir, output_vtk=vtk_output)

    print(f"Reynolds Number: {re} (based on shear thickness)")
    re_height = re * (ny / shear_thickness)
    print(f"Reynolds Number: {re_height:.1f} (based on domain height)")
    print(f"Reference Velocity: {u_ref}")
    print(f"Viscosity: {solver.nu:.6f}")
    print(f"Density (uniform): {rho_top:.1f}")
    if cs_eff > 0.0:
        print("LES Model: Dynamic Smagorinsky (auto Cs)")
    else:
        print("LES Model: Disabled")
    print("\nStarting simulation...")

    headers = diag.print_header(include_forces=False)

    global_start = time.time()

    solver.prepare_diagnostics(reset_baseline=True)
    diag.save_data(0, additional_data={"time": 0.0})

    sim_start = time.time()

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % 100 == 0:
            solver.prepare_diagnostics(f_dst)
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

        elif step % interval == 0:
            solver.prepare_diagnostics(f_dst)
            diag.save_data(step, additional_data={"time": float(step)})

    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    history_file = diag.save_history(
        params={
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
            "perturb_type": perturb_type,
            "cs": cs,
        }
    )
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
    parser.add_argument("--re", type=float, default=400.0, help="Reynolds number")
    parser.add_argument(
        "--u0", type=float, default=0.06, help="Shear velocity magnitude"
    )
    parser.add_argument(
        "--rho_top",
        type=float,
        default=1.0,
        help="Top layer density (must equal rho_bottom)",
    )
    parser.add_argument(
        "--rho_bottom",
        type=float,
        default=1.0,
        help="Bottom layer density (must equal rho_top)",
    )
    # shear_thickness 固定為 0.05*ny（不對外開放）
    parser.add_argument(
        "--shear_center",
        type=float,
        default=0.5,
        help="Shear layer center ratio (single layer at y/ny=r)",
    )
    parser.add_argument(
        "--perturb_amp",
        type=float,
        default=0.02,
        help="Initial perturbation amplitude (absolute velocity)",
    )
    parser.add_argument(
        "--perturb_mode",
        type=int,
        default=0,
        help="Perturbation mode in x (0 = auto-calculate = nx/(8*shear_thickness))",
    )
    parser.add_argument(
        "--perturb_sigma",
        type=float,
        default=0.0,
        help="Perturbation vertical decay scale (0.0 = global perturbation)",
    )
    parser.add_argument(
        "--perturb_type",
        type=str,
        default="multi",
        choices=["single", "multi", "white"],
        help="Perturbation type: single mode, multi-mode, or white noise",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=0.0,
        help="LES Smagorinsky constant (0.0 disables LES, recommended for Re=400)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=1000, help="Save interval")
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
        shear_center_ratio=args.shear_center,
        perturb_amp=args.perturb_amp,
        perturb_mode=args.perturb_mode,
        perturb_sigma=args.perturb_sigma,
        perturb_type=args.perturb_type,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
        vtk_output=args.vtk,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
