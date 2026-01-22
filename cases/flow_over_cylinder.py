"""
Flow Over Cylinder
==================

圓柱繞流：經典鈍體空氣動力學問題

Why 這個 case?
- 簡單幾何，複雜物理
- 卡門渦街（Kármán vortex street）
- 阻力係數驗證
- Re 範圍廣：從層流到湍流

物理現象：
- Re < 5: 附著流
- 5 < Re < 40: 定常分離
- 40 < Re < 150: 週期性卡門渦街
- 150 < Re < 300: 三維轉變
- Re > 300: 湍流尾流

參考數據：
- Re=100: Cd ≈ 1.4, St ≈ 0.16
- Re=150: Cd ≈ 1.3, St ≈ 0.18
"""

import taichi as ti
import argparse
import time
import os
from core import LBMSolver, BoundaryConditions, Diagnostics
from utils.geometry import create_circle_mask


def run_flow_over_cylinder(
    res_y: int = 128,
    re: float = 150.0,
    u_in: float = 0.1,
    diameter: float = None,
    cs: float = 0.16,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    output_dir: str = "output_cylinder"
):
    """
    執行圓柱繞流模擬

    Args:
        res_y: Y 方向解析度
        re: Reynolds 數
        u_in: 入口速度
        diameter: 圓柱直徑（若為 None 則使用 res_y/9）
        cs: Smagorinsky 常數
        steps: 總步數
        interval: 儲存間隔
        tol: 收斂容差
        output_dir: 輸出目錄
    """
    print("="*70)
    print(" "*20 + "FLOW OVER CYLINDER")
    print("="*70)

    # === 計算網格與幾何參數 ===
    nx = int(3.5 * res_y)  # 長寬比 3.5:1
    ny = res_y

    if diameter is None:
        diameter = res_y / 9.0  # 標準直徑

    radius = diameter / 2.0

    # 圓柱位置：X 方向 1/4，Y 方向居中
    cx = nx / 4.0
    cy = ny / 2.0

    print(f"\nGrid: {nx}x{ny}")
    print(f"Cylinder: center=({cx:.1f}, {cy:.1f}), diameter={diameter:.1f}")

    # === 初始化求解器 ===
    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_in,
        length_scale=diameter,
        cs=cs
    )

    # === 設定障礙物 ===
    mask = create_circle_mask(nx, ny, center=(cx, cy), radius=radius)
    solver.set_obstacle(mask)

    # === 設定邊界條件 ===
    bc = BoundaryConditions(solver)
    bc.add_zou_he_velocity_inlet(u_in, location='left')
    bc.add_zou_he_pressure_outlet(rho_out=1.0, location='right')
    bc.add_free_slip_wall('top')
    bc.add_free_slip_wall('bottom')

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir)

    # === 主迴圈 ===
    print(f"Reynolds Number: {re}")
    print(f"Inlet Velocity: {u_in}")
    print(f"Viscosity: {solver.nu:.6f}")
    print(f"Smagorinsky Cs: {cs}")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=True)

    global_start = time.time()
    
    # 記錄初始質量
    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]
    diag.save_data(0, additional_data={'cd': 0.0, 'cl': 0.0})

    # 模擬開始時間
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

            row = diag.print_step_info(step, speed, eta_seconds, include_forces=True)

            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                cd, cl = diag.get_force_coefficients()
                diag.save_data(step, additional_data={'cd': cd, 'cl': cl})

            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                if step % interval != 0:
                    cd, cl = diag.get_force_coefficients()
                    diag.save_data(step, additional_data={'cd': cd, 'cl': cl})
                break

        elif step % interval == 0:
            solver._update_macro(f_dst)
            cd, cl = diag.get_force_coefficients()
            diag.save_data(step, additional_data={'cd': cd, 'cl': cl})

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
            'res_y': res_y, 're': re, 'u_in': u_in, 'diameter': diameter, 'cs': cs
        }
    })
    print(f"📊 History saved to {history_file}")

    # === 物理結果 ===
    cd, cl = diag.get_force_coefficients()
    print(f"\n{'Physical Results':<30}")
    print(f"  Final Drag Coefficient (Cd)  : {cd:.4f}")
    print(f"  Final Lift Coefficient (Cl)  : {cl:.4f}")

    # 參考值
    if 90 < re < 110:
        print(f"\n  Reference (Re=100):")
        print(f"    Cd ≈ 1.4 (literature)")
        print(f"    St ≈ 0.16 (Strouhal number)")
    elif 140 < re < 160:
        print(f"\n  Reference (Re=150):")
        print(f"    Cd ≈ 1.3 (literature)")
        print(f"    St ≈ 0.18 (Strouhal number)")


def main():
    parser = argparse.ArgumentParser(description="Flow Over Cylinder Simulation")
    parser.add_argument('--res', type=int, default=128, help='Y resolution')
    parser.add_argument('--re', type=float, default=150.0, help='Reynolds number')
    parser.add_argument('--u_in', type=float, default=0.1, help='Inlet velocity')
    parser.add_argument('--diameter', type=float, default=None, help='Cylinder diameter')
    parser.add_argument('--cs', type=float, default=0.16, help='Smagorinsky constant')
    parser.add_argument('--steps', type=int, default=50000, help='Total steps')
    parser.add_argument('--interval', type=int, default=100, help='Save interval')
    parser.add_argument('--tol', type=float, default=1e-5, help='Convergence tolerance')
    parser.add_argument('--output', type=str, default='output_cylinder', help='Output directory')

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_flow_over_cylinder(
        res_y=args.res,
        re=args.re,
        u_in=args.u_in,
        diameter=args.diameter,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
