"""
Backward-Facing Step
====================

Why this case?
- 經典分離/再附著驗證
- 同時測試入口、壁面、出口邊界條件穩定性
- LES/渦黏度模型在回流區的行為
"""

import argparse
import os
import time

import numpy as np
import taichi as ti

from lbm_taichi.core import BoundaryConditions, Diagnostics, LBMSolver


def create_step_mask(nx: int, ny: int, step_len: int, step_height: int) -> np.ndarray:
    """
    建立 backward-facing step mask

    流體域：
    - x < step_len: 有台階阻塞（y < step_height 為固體）
    - x >= step_len: 底部壁面（y = 0）為固體
    """
    mask = np.zeros((nx, ny), dtype=np.int32)
    # 底部壁面（整段）
    mask[:, 0] = 1
    # 台階區塊
    if step_len > 0 and step_height > 1:
        mask[:step_len, :step_height] = 1
    return mask


def run_backward_facing_step(
    res_y: int = 128,
    re: float = 1000.0,
    u_in: float = 0.1,
    cs: float = 0.16,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    step_height_ratio: float = 0.5,
    step_length_ratio: float = 0.2,
    output_dir: str = "output_bfs",
):
    """
    執行 backward-facing step 模擬
    """
    nx = int(4.0 * res_y)
    ny = res_y

    step_height = max(2, int(step_height_ratio * ny))
    step_len = max(2, int(step_length_ratio * nx))

    print("=" * 70)
    print(" " * 18 + "BACKWARD-FACING STEP")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}")
    print(f"Step height: {step_height} (ratio {step_height_ratio:.2f})")
    print(f"Step length: {step_len} (ratio {step_length_ratio:.2f})")

    auto_les = re > 2000.0
    cs_eff = cs if cs > 0.0 else (0.16 if auto_les else cs)

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_in,
        length_scale=ny,
        cs=cs_eff,
    )
    if re > 1000.0:
        solver.set_wall_function(enabled=True, yplus_min=5.0)
        print("Wall Function: Enabled (Re > 1000, yplus_min=5.0)")
    else:
        solver.set_wall_function(enabled=False)
        print("Wall Function: Disabled (Re <= 1000)")

    # === Geometry ===
    mask = create_step_mask(nx, ny, step_len, step_height)
    solver.initialize_obstacle(mask)

    # === Boundary Conditions ===
    bc = BoundaryConditions(solver)
    bc.add_velocity_inlet(
        u_in,
        location="left",
        method="neq",
        epsilon=0.0,
        asymmetry=0.0,
        omega=0.0,
    )
    bc.add_stable_outlet(rho_out=1.0, location="right", relaxation=0.02)
    bc.add_free_slip_wall("top")

    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    diag = Diagnostics(solver, output_dir=output_dir)

    print(f"Reynolds Number: {re}")
    print(f"Inlet Velocity: {u_in}")
    print(f"Viscosity: {solver.nu:.6f}")
    if cs_eff > 0.0:
        print("LES Model: Dynamic Smagorinsky (auto Cs)")
    else:
        print("LES Model: Disabled")
    print("\nStarting simulation...")

    headers = diag.print_header(include_forces=False)

    global_start = time.time()

    solver.prepare_diagnostics(reset_baseline=True)
    diag.save_data(0)

    sim_start = time.time()

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % 100 == 0:
            solver.prepare_diagnostics(f_dst)
            ti.sync()

            elapsed = time.time() - sim_start
            speed = step / elapsed if elapsed > 1e-6 else 0.0
            remaining = steps - step
            eta_seconds = remaining / speed if speed > 0 else 0.0

            row = diag.print_step_info(step, speed, eta_seconds, include_forces=False)

            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                diag.save_data(step)

            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                if step % interval != 0:
                    diag.save_data(step)
                break

        elif step % interval == 0:
            solver.prepare_diagnostics(f_dst)
            diag.save_data(step)

    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    history_file = diag.save_history(
        params={
            "res_y": res_y,
            "re": re,
            "u_in": u_in,
            "cs": cs,
            "step_height_ratio": step_height_ratio,
            "step_length_ratio": step_length_ratio,
        }
    )
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Backward-Facing Step Simulation")
    parser.add_argument("--res", type=int, default=128, help="Y resolution")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--u_in", type=float, default=0.1, help="Inlet velocity")
    parser.add_argument("--cs", type=float, default=0.16, help="Smagorinsky constant")
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument(
        "--step_h",
        type=float,
        default=0.5,
        help="Step height ratio (relative to ny)",
    )
    parser.add_argument(
        "--step_len",
        type=float,
        default=0.2,
        help="Step length ratio (relative to nx)",
    )
    parser.add_argument(
        "--output", type=str, default="output_bfs", help="Output directory"
    )

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_backward_facing_step(
        res_y=args.res,
        re=args.re,
        u_in=args.u_in,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        step_height_ratio=args.step_h,
        step_length_ratio=args.step_len,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
