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

import taichi as ti
import argparse
import time
import os
from core import LBMSolver, BoundaryConditions, Diagnostics
from utils.geometry import create_lid_velocity_profile


def run_lid_driven_cavity(
    res: int = 256,
    re: float = 1000.0,
    lid_vel: float = 0.1,
    cs: float = 0.16,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    output_dir: str = "output_ldc"
):
    """
    執行 Lid-Driven Cavity 模擬

    Args:
        res: 解析度（方形網格）
        re: Reynolds 數
        lid_vel: 上蓋速度
        cs: Smagorinsky 常數
        steps: 總步數
        interval: 儲存間隔
        tol: 收斂容差
        output_dir: 輸出目錄
    """
    print("="*70)
    print(" "*20 + "LID-DRIVEN CAVITY FLOW")
    print("="*70)

    # === 初始化求解器 ===
    solver = LBMSolver(
        nx=res,
        ny=res,
        re=re,
        u_ref=lid_vel,
        length_scale=res,  # 特徵長度 = 腔體尺寸
        cs=cs
    )

    # === 設定邊界條件 ===
    bc = BoundaryConditions(solver)

    # 底部、左、右：No-Slip（通過 mask 或 Bounce-Back）
    # 這裡我們不需要額外設定，因為預設就是固體邊界

    # 上蓋：運動壁面（平滑速度 profile）
    u_wall_profile = create_lid_velocity_profile(res, lid_vel)
    bc.add_moving_wall(u_wall_profile, location='top')

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir)

    # === 主迴圈 ===
    print(f"\nGrid: {res}x{res}")
    print(f"Reynolds Number: {re}")
    print(f"Lid Velocity: {lid_vel}")
    print(f"Viscosity: {solver.nu:.6f}")
    print(f"Smagorinsky Cs: {cs}")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=False)

    global_start = time.time()
    
    # 記錄初始質量
    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]
    diag.save_data(0)

    # 真正的模擬開始時間（排除初始化與第一次 Kernel 編譯）
    sim_start = time.time()

    for step in range(1, steps + 1):
        # 雙緩衝交替
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        # 時間推進
        solver.step(f_src, f_dst)

        # 診斷與輸出
        if step % 100 == 0:
            solver._update_macro(f_dst)
            solver._update_diagnostics()
            ti.sync()

            # 使用從模擬開始以來的平均速度計算 ETA (更穩定)
            total_elapsed = time.time() - sim_start
            speed = step / total_elapsed if total_elapsed > 1e-6 else 0.0
            remaining_steps = steps - step
            eta_seconds = remaining_steps / speed if speed > 0 else 0.0

            row = diag.print_step_info(step, speed, eta_seconds, include_forces=False)

            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                diag.save_data(step)

            # 收斂檢查
            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                if step % interval != 0:
                    diag.save_data(step)
                break

        elif step % interval == 0:
            solver._update_macro(f_dst)
            diag.save_data(step)

    # === 總結 ===
    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)
    
    # 存儲歷史數據
    history_file = os.path.join(output_dir, "history.npy")
    np.save(history_file, {
        'headers': headers,
        'data': diag.history,
        'params': {
            'res': res, 're': re, 'lid_vel': lid_vel, 'cs': cs
        }
    })
    print(f"📊 History saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity Flow Simulation")
    parser.add_argument('--res', type=int, default=256, help='Resolution (NxN grid)')
    parser.add_argument('--re', type=float, default=1000.0, help='Reynolds number')
    parser.add_argument('--lid_vel', type=float, default=0.1, help='Lid velocity')
    parser.add_argument('--cs', type=float, default=0.16, help='Smagorinsky constant')
    parser.add_argument('--steps', type=int, default=50000, help='Total steps')
    parser.add_argument('--interval', type=int, default=100, help='Save interval')
    parser.add_argument('--tol', type=float, default=1e-5, help='Convergence tolerance')
    parser.add_argument('--output', type=str, default='output_ldc', help='Output directory')

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
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
