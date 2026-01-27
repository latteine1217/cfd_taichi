"""
Taylor-Green Vortex
===================

標準 CFD 基準測試：週期邊界條件驗證

Why Taylor-Green Vortex?
- 理論解析解存在
- 驗證週期邊界條件正確性
- 測試數值耗散
- 基準測試

Physical Setup:
- 雙週期域（X 和 Y 方向都週期）
- 初始條件：
  u = -U₀ cos(kx) sin(ky)
  v = U₀ sin(kx) cos(ky)
- 理論解：能量指數衰減 E(t) ~ exp(-2νk²t)

驗證指標：
- 總動能 vs 時間
- 能量衰減率（與理論對比）
- 渦度場演化
"""

import taichi as ti
import numpy as np
import argparse
import time
import os
from core import LBMSolver, BoundaryConditions, Diagnostics


def run_taylor_green(
    res: int = 128,
    re: float = 100.0,
    u0: float = 0.1,
    steps: int = 10000,
    interval: int = 100,
    output_dir: str = "output_taylor_green",
    collision_model: str = "mrt",
):
    """
    運行 Taylor-Green Vortex 模擬

    Args:
        res: 解析度（nx = ny = res）
        re: Reynolds 數
        u0: 初始最大速度
        steps: 總步數
        interval: 保存間隔
        output_dir: 輸出目錄
    """
    nx, ny = res, res
    print(f"\n{'Taylor-Green Vortex Simulation':<30}")
    print(f"  Resolution    : {nx} × {ny}")
    print(f"  Reynolds No.  : {re}")
    print(f"  Max Velocity  : {u0}")

    # === 創建 Solver ===
    solver = LBMSolver(nx=nx, ny=ny, re=re, u_ref=u0, collision_model=collision_model)

    # === 週期邊界條件 ===
    bc = BoundaryConditions(solver)
    bc.add_periodic_boundary("x")
    bc.add_periodic_boundary("y")
    print(f"\nBoundary Conditions:")
    print(f"  X-direction   : Periodic")
    print(f"  Y-direction   : Periodic")

    # === Taylor-Green Vortex 初始條件 ===
    print(f"\nInitializing Taylor-Green Vortex...")
    k = 2.0 * np.pi / nx  # 波數

    for i in range(nx):
        for j in range(ny):
            x = i
            y = j
            u_x = -u0 * np.cos(k * x) * np.sin(k * y)
            u_y = u0 * np.sin(k * x) * np.cos(k * y)
            solver.u[i + 1, j + 1] = [u_x, u_y]
            solver.rho[i + 1, j + 1] = 1.0

    # 從速度場重建分佈函數
    solver._init_from_macro()
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # === 理論衰減率 ===
    nu = solver.nu
    decay_rate_theory = 2.0 * nu * k**2
    print(f"\nTheoretical Decay Rate: γ = {decay_rate_theory:.6f}")
    print(f"Wave Number: k = {k:.6f}")
    print(f"Viscosity: ν = {nu:.6f}")

    # === 診斷系統 ===
    os.makedirs(output_dir, exist_ok=True)
    diag = Diagnostics(solver, output_dir=output_dir)

    # === 初始動能 ===
    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_KE[None] = solver.total_KE[None]
    initial_KE = solver.total_KE[None]
    print(f"Initial Kinetic Energy: {initial_KE:.6f}")

    # 保存初始場
    diag.save_data(0, additional_data={"decay_rate_theory": decay_rate_theory})

    # === 主迴圈 ===
    print(f"\n{'=' * 60}")
    print(f"{'Step':<8} {'KE':<12} {'Decay Rate':<12} {'Speed':<12} {'ETA':<12}")
    print(f"{'=' * 60}")

    KE_history = [initial_KE]
    time_history = [0]

    sim_start = time.time()

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % 100 == 0:
            solver._update_macro(f_dst)
            solver._update_diagnostics()
            ti.sync()

            current_KE = solver.total_KE[None]
            KE_history.append(current_KE)
            time_history.append(step)

            # 計算實際衰減率（對數擬合最近 10 個點）
            if len(KE_history) > 10:
                log_KE = np.log(np.array(KE_history[-10:]) + 1e-12)
                times = np.array(time_history[-10:])
                # 線性擬合：log(KE) = log(KE0) - γt
                decay_rate_measured = -np.polyfit(times, log_KE, 1)[0]
                error = (
                    abs(decay_rate_measured - decay_rate_theory)
                    / decay_rate_theory
                    * 100
                )
            else:
                decay_rate_measured = 0.0
                error = 0.0

            # 效能統計
            total_elapsed = time.time() - sim_start
            speed = step / total_elapsed if total_elapsed > 1e-6 else 0.0
            remaining_steps = steps - step
            eta_seconds = remaining_steps / speed if speed > 0 else 0.0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            if step % 500 == 0:
                print(
                    f"{step:<8} {current_KE:<12.6f} {decay_rate_measured:<12.6f} "
                    f"{speed:<12.1f} {eta_str:<12}"
                )

            # 保存數據
            if step % interval == 0:
                diag.save_data(
                    step,
                    additional_data={
                        "KE": current_KE,
                        "decay_rate_measured": decay_rate_measured,
                        "decay_rate_error": error,
                    },
                )

    # === 總結 ===
    total_time = time.time() - sim_start
    print(f"{'=' * 60}")
    print(f"\nSimulation completed in {total_time:.2f} seconds")

    # 最終分析
    final_KE = KE_history[-1]
    energy_decay = (initial_KE - final_KE) / initial_KE

    # 計算整體衰減率（全部數據點）
    log_KE_all = np.log(np.array(KE_history) + 1e-12)
    times_all = np.array(time_history)
    decay_rate_overall = -np.polyfit(times_all, log_KE_all, 1)[0]
    error_overall = (
        abs(decay_rate_overall - decay_rate_theory) / decay_rate_theory * 100
    )

    print(f"\n{'Final Statistics':<30}")
    print(f"  Initial KE         : {initial_KE:.6f}")
    print(f"  Final KE           : {final_KE:.6f}")
    print(f"  Energy Decay       : {energy_decay * 100:.2f}%")
    print(f"\n{'Decay Rate Analysis':<30}")
    print(f"  Theoretical γ      : {decay_rate_theory:.6f}")
    print(f"  Measured γ         : {decay_rate_overall:.6f}")
    print(f"  Error              : {error_overall:.2f}%")

    # 驗證結果
    print(f"\n{'Validation':<30}")
    if error_overall < 5.0:
        print(f"  ✅ PASS: Decay rate error < 5%")
    elif error_overall < 10.0:
        print(f"  ⚠️  ACCEPTABLE: Decay rate error < 10%")
    else:
        print(f"  ❌ FAIL: Decay rate error > 10%")

    # 保存歷史數據
    history_file = os.path.join(output_dir, "history.npy")
    np.save(
        history_file,
        {
            "KE_history": KE_history,
            "time_history": time_history,
            "decay_rate_theory": decay_rate_theory,
            "decay_rate_measured": decay_rate_overall,
            "error": error_overall,
            "params": {
                "res": res,
                "re": re,
                "u0": u0,
                "nx": nx,
                "ny": ny,
                "nu": nu,
                "k": k,
            },
        },
    )
    print(f"\n📊 History saved to {history_file}")

    # === 物理解釋 ===
    print(f"\n{'Physical Interpretation':<30}")
    print(f"  - 動能指數衰減驗證週期邊界條件正確")
    print(f"  - 衰減率 γ = 2νk² 反映數值耗散")
    print(f"  - 誤差 < 5% 表示 LBM 格式精度良好")
    print(f"  - 週期性消除邊界影響，純粹測試內部格式")


def main():
    parser = argparse.ArgumentParser(description="Taylor-Green Vortex Simulation")
    parser.add_argument("--res", type=int, default=128, help="Resolution (nx=ny=res)")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--u0", type=float, default=0.1, help="Initial max velocity")
    parser.add_argument("--steps", type=int, default=10000, help="Total steps")
    parser.add_argument("--interval", type=int, default=100, help="Save interval")
    parser.add_argument(
        "--output", type=str, default="output_taylor_green", help="Output directory"
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

    run_taylor_green(
        res=args.res,
        re=args.re,
        u0=args.u0,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
