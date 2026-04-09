"""
Rayleigh-Taylor Instability (Multiphase Shan-Chen)
==================================================

多相雙組分 Rayleigh-Taylor 不穩定性。

Why 這個 case?
- 驗證多相 LBM 的界面與表面張力效應
- 檢驗上重下輕的真實分層演化
- 作為多相 RT 的最小可用基準

物理現象：
- 上層組分 A 為重相、下層組分 B 為輕相
- 在重力下形成指狀下沉與氣泡上升
- 界面由 Shan-Chen 表面張力維持
"""

import argparse
import os
import time

import numpy as np
import taichi as ti

from lbm_taichi.core import (
    MultiphaseBoundaryConditions,
    MultiphaseDiagnostics,
    MultiphaseLBMSolver,
)


def _build_rt_multiphase_fields(
    nx: int,
    ny: int,
    rho_heavy: float,
    rho_light: float,
    rho_eps: float,
    interface_ratio: float,
    thickness: float,
    perturb_amp: float,
    perturb_mode: int,
):
    """
    生成多相 Rayleigh-Taylor 初始場

    What: 兩組分密度分層 + 介面擾動
    Why: 平滑介面避免初期數值振盪
    When: 初始化 RT 多相案例
    """
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    y0 = float(interface_ratio * (ny - 1))
    if perturb_amp > 0.0 and perturb_mode > 0:
        phase = 2.0 * np.pi * perturb_mode * X / max(1.0, nx)
        y0 = y0 + perturb_amp * np.sin(phase)

    delta = max(1.0, thickness)
    profile = np.tanh((Y - y0) / delta)

    rho_a_top = rho_heavy
    rho_a_bottom = rho_eps
    rho_b_top = rho_eps
    rho_b_bottom = rho_light

    rho_a = 0.5 * (rho_a_top + rho_a_bottom) + 0.5 * (
        rho_a_top - rho_a_bottom
    ) * profile
    rho_b = 0.5 * (rho_b_top + rho_b_bottom) + 0.5 * (
        rho_b_top - rho_b_bottom
    ) * profile

    return rho_a.astype(np.float32), rho_b.astype(np.float32)


def run_rayleigh_taylor_multiphase(
    res_y: int = 128,
    aspect_ratio: float = 0.5,
    tau_a: float = 0.8,
    tau_b: float = 0.8,
    g_interaction: float = 3.5,
    collision_model: str = "mrt",
    gravity: float = 5.0e-6,
    gravity_mode: str = "buoyancy",
    u_cap: float = 0.05,
    force_cap: float = 1.0e-4,
    recolor_beta: float = 0.0,
    rho_heavy: float = 1.2,
    rho_light: float = 0.2,
    rho_eps: float = 0.02,
    interface_ratio: float = 0.5,
    thickness_ratio: float = 0.03,
    perturb_amp_ratio: float = 0.02,
    perturb_mode: int = 2,
    mass_correction_interval: int = 0,
    mass_correction_strength: float = 0.02,
    steps: int = 50000,
    interval: int = 1000,
    output_dir: str = "output_rayleigh_taylor_multiphase",
):
    """
    執行多相 Rayleigh-Taylor 不穩定性

    Args:
        res_y: Y 方向解析度
        aspect_ratio: 計算域長寬比 (nx = aspect_ratio * res_y)
        tau_a: 組分 A 鬆弛時間 (>0.5)
        tau_b: 組分 B 鬆弛時間 (>0.5)
        g_interaction: Shan-Chen 交互作用強度 (正值產生相分離)
        collision_model: 'mrt' 或 'bgk'
        gravity: 重力加速度大小（正值，方向向下）
        gravity_mode: 'buoyancy' 或 'absolute'
        u_cap: 速度上限（穩定性）
        force_cap: 力上限（穩定性）
        recolor_beta: recoloring 強度（介面銳化）
        rho_heavy: 重流體密度（上層）
        rho_light: 輕流體密度（下層）
        rho_eps: 另一組分的背景密度（避免 0）
        interface_ratio: 介面位置比例（0.5 = 中央）
        thickness_ratio: 介面厚度比例（相對 ny）
        perturb_amp_ratio: 介面擾動振幅比例（相對 ny）
        perturb_mode: x 方向擾動模態數
        mass_correction_interval: 質量修正間隔（0=關閉）
        mass_correction_strength: 質量修正強度
        steps: 總步數
        interval: 輸出間隔
        output_dir: 輸出目錄
    """
    nx = int(aspect_ratio * res_y)
    ny = res_y

    interface_ratio = float(np.clip(interface_ratio, 0.1, 0.9))
    thickness = max(1.0, thickness_ratio * ny)
    perturb_amp = max(0.0, perturb_amp_ratio * ny)
    perturb_mode = max(1, int(perturb_mode))

    g = -abs(gravity)

    print("=" * 70)
    print(" " * 14 + "RAYLEIGH-TAYLOR (MULTIPHASE)")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}")
    print(f"Interface: y/ny = {interface_ratio:.2f}")
    print(f"Shan-Chen G: {g_interaction:.3f} (surface tension control)")
    print(f"Collision Model: {collision_model}")
    print(
        f"Stabilizers: u_cap={u_cap:.3f}, force_cap={force_cap:.2e}, "
        f"recolor_beta={recolor_beta:.2f}"
    )
    print(f"Gravity: g={g:.2e} (downward, mode={gravity_mode})")
    print(
        f"Density levels: rho_heavy={rho_heavy:.3f}, "
        f"rho_light={rho_light:.3f}, rho_eps={rho_eps:.3f}"
    )
    print(
        f"Perturbation: amp={perturb_amp:.2f} cells, mode={perturb_mode}, "
        f"thickness={thickness:.2f} cells"
    )

    solver = MultiphaseLBMSolver(
        nx=nx,
        ny=ny,
        tau_a=tau_a,
        tau_b=tau_b,
        g_interaction=g_interaction,
        collision_model=collision_model,
        gravity=(0.0, g),
        gravity_mode=gravity_mode,
        u_cap=u_cap,
        force_cap=force_cap,
        recolor_beta=recolor_beta,
        mass_correction_interval=mass_correction_interval,
        mass_correction_strength=mass_correction_strength,
    )

    bc = MultiphaseBoundaryConditions(solver)
    bc.add_periodic_boundary("x")
    bc.add_free_slip_wall("top")
    bc.add_free_slip_wall("bottom")

    rho_a, rho_b = _build_rt_multiphase_fields(
        nx=nx,
        ny=ny,
        rho_heavy=rho_heavy,
        rho_light=rho_light,
        rho_eps=rho_eps,
        interface_ratio=interface_ratio,
        thickness=thickness,
        perturb_amp=perturb_amp,
        perturb_mode=perturb_mode,
    )
    solver.set_initial_fields(rho_a, rho_b)
    solver.apply_boundary_conditions()

    diag = MultiphaseDiagnostics(solver, output_dir=output_dir)

    print("\n=== Boundary Conditions ===")
    print("X-direction : Periodic")
    print("Y-direction : Free-Slip (top/bottom)")

    print("\n=== Simulation Start ===")
    headers = diag.print_header()

    global_start = time.time()
    diag.set_initial_baseline()
    diag.save_data(0, time_value=0.0)

    sim_start = time.time()

    for step in range(1, steps + 1):
        solver.step()

        if step % 100 == 0:
            total_elapsed = time.time() - sim_start
            speed = step / total_elapsed if total_elapsed > 1e-6 else 0.0
            remaining_steps = steps - step
            eta_seconds = remaining_steps / speed if speed > 0 else 0.0

            row = diag.print_step_info(step, speed, eta_seconds)
            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                diag.save_data(step, time_value=float(step))

        elif step % interval == 0:
            diag.save_data(step, time_value=float(step))

    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    history_file = diag.save_history(
        params={
            "res_y": res_y,
            "aspect_ratio": aspect_ratio,
            "tau_a": tau_a,
            "tau_b": tau_b,
            "g_interaction": g_interaction,
            "gravity": gravity,
            "u_cap": u_cap,
            "force_cap": force_cap,
            "recolor_beta": recolor_beta,
            "rho_heavy": rho_heavy,
            "rho_light": rho_light,
            "rho_eps": rho_eps,
            "interface_ratio": interface_ratio,
            "thickness_ratio": thickness_ratio,
            "perturb_amp_ratio": perturb_amp_ratio,
            "perturb_mode": perturb_mode,
        }
    )
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Multiphase Rayleigh-Taylor Instability")
    parser.add_argument("--res", type=int, default=128, help="Y resolution")
    parser.add_argument(
        "--aspect",
        type=float,
        default=0.5,
        help="Domain aspect ratio (nx = aspect * res)",
    )
    parser.add_argument("--tau_a", type=float, default=0.8, help="Relaxation time A")
    parser.add_argument("--tau_b", type=float, default=0.8, help="Relaxation time B")
    parser.add_argument(
        "--G",
        type=float,
        default=3.5,
        help="Shan-Chen interaction strength (positive for phase separation)",
    )
    parser.add_argument(
        "--collision",
        type=str,
        default="mrt",
        choices=["mrt", "bgk"],
        help="Collision model: mrt or bgk",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=5.0e-6,
        help="Gravity magnitude (positive, downward)",
    )
    parser.add_argument(
        "--gravity_mode",
        type=str,
        default="buoyancy",
        choices=["buoyancy", "absolute"],
        help="Gravity mode: buoyancy removes mean acceleration",
    )
    parser.add_argument("--u_cap", type=float, default=0.05, help="Velocity cap")
    parser.add_argument(
        "--force_cap", type=float, default=1.0e-4, help="Force cap"
    )
    parser.add_argument(
        "--recolor_beta",
        type=float,
        default=0.0,
        help="Recoloring strength (0 disables)",
    )
    parser.add_argument(
        "--rho_heavy",
        type=float,
        default=1.2,
        help="Heavy fluid density (top layer)",
    )
    parser.add_argument(
        "--rho_light",
        type=float,
        default=0.2,
        help="Light fluid density (bottom layer)",
    )
    parser.add_argument(
        "--rho_eps",
        type=float,
        default=0.02,
        help="Background density for the other component",
    )
    parser.add_argument(
        "--interface",
        type=float,
        default=0.5,
        help="Interface location ratio (0.5 = center)",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.03,
        help="Interface thickness ratio (relative to ny)",
    )
    parser.add_argument(
        "--perturb_amp",
        type=float,
        default=0.02,
        help="Perturbation amplitude ratio (relative to ny)",
    )
    parser.add_argument(
        "--perturb_mode",
        type=int,
        default=2,
        help="Perturbation mode in x-direction",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=1000, help="Save interval")
    parser.add_argument(
        "--mass_corr_interval",
        type=int,
        default=0,
        help="Mass correction interval (0 disables)",
    )
    parser.add_argument(
        "--mass_corr_strength",
        type=float,
        default=0.02,
        help="Mass correction strength",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_rayleigh_taylor_multiphase",
        help="Output directory",
    )

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_rayleigh_taylor_multiphase(
        res_y=args.res,
        aspect_ratio=args.aspect,
        tau_a=args.tau_a,
        tau_b=args.tau_b,
        g_interaction=args.G,
        collision_model=args.collision,
        gravity=args.g,
        gravity_mode=args.gravity_mode,
        u_cap=args.u_cap,
        force_cap=args.force_cap,
        recolor_beta=args.recolor_beta,
        rho_heavy=args.rho_heavy,
        rho_light=args.rho_light,
        rho_eps=args.rho_eps,
        interface_ratio=args.interface,
        thickness_ratio=args.thickness,
        perturb_amp_ratio=args.perturb_amp,
        perturb_mode=args.perturb_mode,
        mass_correction_interval=args.mass_corr_interval,
        mass_correction_strength=args.mass_corr_strength,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
