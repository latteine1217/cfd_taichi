"""
Lid-Driven Cavity Flow
======================

經典 CFD 驗證案例：方腔內上蓋驅動流

Why 這個 case?
- CFD 標準驗證案例
- 豐富的流動結構（主渦、二次渦）
- 廣泛的基準數據可供比較

物理現象：
- 上蓋以固定速度移動
- 四周壁面 No-Slip
- 形成再迴圈流動結構
- Re 增加 -> 渦結構變複雜

參考：
- Ghia et al. (1982), Re=100, 400, 1000, 3200
"""

import os
import taichi as ti
import numpy as np
import argparse
import time

from cfd_taichi import (
    BoundaryConditionDescriptor,
    CaseRunner,
    LatticeGrid2D,
)
from lbm_taichi.core import LBMSolver, Diagnostics
from lbm_taichi.utils.geometry import create_lid_velocity_profile


def initialize_ldc_rest_state(solver: LBMSolver):
    """LDC 專用初始化：腔內流體初始靜止，不沿用外流 case 的均勻 u_ref 初始場。"""
    u0 = np.zeros((solver.nx, solver.ny, 2), dtype=np.float32)
    solver.set_initial_condition(velocity=u0)


def _initialize_ldc_case(runner: CaseRunner, solver: LBMSolver, _bc_handle):
    """
    LDC CaseRunner 初始化流程。

    What:
    - 使用靜止初始場並立即施加邊界
    - 以 quiescent state 重設質量/能量基準

    Why:
    - LDC 不應沿用 solver.reset() 的均勻 `u_ref` 初始場
    - CaseRunner 應接管正式初始化流程，而不是讓案例手動散落在主函式中
    """
    initialize_ldc_rest_state(solver)
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)
    runner.prepare_observables(reset_baseline=True)


def build_lid_driven_cavity_runner(
    *,
    res: int = 256,
    re: float = 1000.0,
    lid_vel: float = 0.1,
    cs: float = 0.0,
    collision_model: str = "mrt",
) -> tuple[CaseRunner, float]:
    """
    建立 Lid-Driven Cavity 的 CaseRunner。

    What:
    - 將 solver/grid/BC/initializer 的組裝集中成正式 builder

    Why:
    - benchmark registry 與 batch workflow 需要直接取得 runner，而不是只能呼叫整個案例腳本
    """
    auto_les = re > 2000.0
    if cs > 0.0:
        cs_eff = cs
    elif auto_les:
        cs_eff = 0.16
    else:
        cs_eff = -1.0

    runner = CaseRunner(
        name="lid_driven_cavity",
        method="lbm",
        equation="single_phase",
        grid=LatticeGrid2D(nx=res, ny=res),
        solver_kwargs={
            "re": re,
            "u_ref": lid_vel,
            "length_scale": res,
            "cs": cs_eff,
            "collision_model": collision_model,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.no_slip("bottom", exclude_corners=True),
            BoundaryConditionDescriptor.no_slip("left", exclude_corners=True),
            BoundaryConditionDescriptor.no_slip("right", exclude_corners=True),
            BoundaryConditionDescriptor.moving_wall(
                "top",
                velocity_profile=create_lid_velocity_profile(res, lid_vel),
                corner_mode="ldc",
            ),
        ],
        initializer=_initialize_ldc_case,
    )
    return runner, cs_eff


def run_lid_driven_cavity(
    res: int = 256,
    re: float = 1000.0,
    lid_vel: float = 0.1,
    cs: float = 0.0,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    output_dir: str = "output_ldc",
    collision_model: str = "mrt",
):
    """
    執行 Lid-Driven Cavity 模擬（嚴格驗證模式）

    Args:
        res: 解析度（方形網格）
        re: Reynolds 數
        lid_vel: 上蓋速度
        cs: LES Smagorinsky 常數（預設 0.0 = 關閉，與 Ghia 基準一致）
            設為 >0 可手動啟用 LES；Re > 2000 時自動啟用。
        steps: 總步數
        interval: 儲存間隔
        tol: 收斂容差
        output_dir: 輸出目錄
    """
    print("=" * 70)
    print(" " * 20 + "LID-DRIVEN CAVITY FLOW")
    print("=" * 70)

    runner, cs_eff = build_lid_driven_cavity_runner(
        res=res,
        re=re,
        lid_vel=lid_vel,
        cs=cs,
        collision_model=collision_model,
    )
    solver = runner.build_solver()
    runner.configure()
    runner.initialize()

    # 壁面函數預設關閉。
    # Why: LDC 是層流/轉捩驗證案例；壁面函數為湍流邊界層設計，
    #      在 Re <= 3200 的 LDC 中引入非必要的不確定性。
    solver.set_wall_function(enabled=False)
    print("Wall Function: Disabled (LDC verification mode)")

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir)

    # === 主迴圈 ===
    print(f"\nGrid: {res}x{res}")
    print(f"Reynolds Number: {re}")
    print(f"Lid Velocity: {lid_vel}")
    print(f"Viscosity: {solver.nu:.6f}")
    if cs_eff > 0.0:
        print("LES Model: Dynamic Smagorinsky (auto Cs)")
    else:
        print("LES Model: Disabled")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=False)

    global_start = time.time()

    # 以 LDC 專用初始化後的狀態重設守恆/能量基準
    # 否則 initial_KE 會殘留 solver.reset() 時的均勻 u_ref 初始場，
    # 使得 LDC 的能量診斷與 mass/residual 基準失真。
    runner.prepare_observables(reset_baseline=True)
    diag.save_data(0)

    # 真正的模擬開始時間（排除初始化與第一次 Kernel 編譯）
    sim_start = time.time()

    for step in range(1, steps + 1):
        runner.step_once()
        f_dst = runner.active_distribution_field()

        # 診斷與輸出
        if step % 100 == 0:
            runner.prepare_observables(reset_baseline=False)
            ti.sync()

            # 使用從模擬開始以來的平均速度計算 ETA (更穩定)
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
                diag.save_data(step)

            # 收斂檢查（LDC 早期容易被弱殘差指標誤判；至少跑完整個指定步數再由使用者判讀）
            # if diag.check_convergence(tol):
            #     print(f"\n✅ Converged at step {step}")
            #     diag.history.append(row)
            #     if step % interval != 0:
            #         diag.save_data(step)
            #     break

        elif step % interval == 0:
            runner.prepare_observables(reset_baseline=False)
            diag.save_data(step)

    # === 總結 ===
    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    # 存儲歷史數據
    history_file = diag.save_history(
        params={"res": res, "re": re, "lid_vel": lid_vel, "cs": cs}
    )
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity Flow Simulation")
    parser.add_argument("--res", type=int, default=256, help="Resolution (NxN grid)")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--lid_vel", type=float, default=0.1, help="Lid velocity")
    parser.add_argument(
        "--cs",
        type=float,
        default=0.0,
        help="Smagorinsky constant (<=0 = disabled; Re>2000 auto-enables at 0.16)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=100, help="Save interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument(
        "--output", type=str, default="output_ldc", help="Output directory"
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

    run_lid_driven_cavity(
        res=args.res,
        re=args.re,
        lid_vel=args.lid_vel,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        output_dir=args.output,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
