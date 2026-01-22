"""
High-Lift Airfoil System
========================

高升力機翼系統：縫翼 + 主翼 + 襟翼

Why 這個配置?
- 商用飛機起降配置
- 複雜幾何測試 LBM 能力
- 升阻比優化問題
- 多體干涉效應

設計參數（基於 Boeing/Airbus 標準）：
- 縫翼：15% 主翼弦長，偏轉 20°
- 主翼：NACA 2412（2% 彎度）
- 襟翼：30% 主翼弦長，偏轉 30°

物理目標：
- 最大升力係數 Cl_max
- 升阻比 L/D
- 失速特性
"""

import taichi as ti
import numpy as np
import argparse
import time
import os
from core import LBMSolver, BoundaryConditions, Diagnostics
from utils.geometry import create_airfoil_system


def run_airfoil(
    res_y: int = 256,
    re: float = 1000.0,
    u_in: float = 0.1,
    main_naca: str = "2412",
    main_chord: float = None,
    aoa: float = 10.0,
    slat_angle: float = 20.0,
    flap_angle: float = 30.0,
    cs: float = 0.16,
    steps: int = 50000,
    interval: int = 1000,
    tol: float = 1e-5,
    output_dir: str = "output_airfoil"
):
    # === 計算網格與弦長比例 ===
    nx = int(3.5 * res_y)  # 調整計算域為 3.5 倍高度
    ny = res_y
    
    if main_chord is None:
        main_chord = res_y * 0.6  # 自動設定弦長為高度的 60%

    print(f"\nConfiguration:")
    print(f"  Main Airfoil   : NACA {main_naca}")
    print(f"  Chord Length   : {main_chord:.1f} lu")
    print(f"  Angle of Attack: {aoa:.1f}°")
    print(f"  Slat Angle     : {slat_angle:.1f}°")
    print(f"  Flap Angle     : {flap_angle:.1f}°")
    print(f"\nGrid: {nx}x{ny}")

    # === 初始化求解器 ===
    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_in,
        length_scale=main_chord,
        cs=cs
    )

    # === 生成機翼幾何 ===
    print("\nGenerating airfoil geometry...")
    mask = create_airfoil_system(
        nx=nx,
        ny=ny,
        main_naca=main_naca,
        slat_naca="0012",
        flap_naca="4415",
        main_chord=main_chord,
        aoa=aoa,
        slat_angle=slat_angle,
        flap_angle=flap_angle,
        center_position=(nx / 3, ny / 2)
    )

    solver.set_obstacle(mask)
    print(f"Solid cells: {mask.sum()} / {nx*ny} ({100*mask.sum()/(nx*ny):.2f}%)")

    # === 設定邊界條件 ===
    bc = BoundaryConditions(solver)
    bc.add_zou_he_velocity_inlet(u_in, location='left')
    bc.add_zou_he_pressure_outlet(rho_out=1.0, location='right')
    bc.add_free_slip_wall('top')
    bc.add_free_slip_wall('bottom')

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir)

    # === 主迴圈 ===
    print(f"\nReynolds Number: {re}")
    print(f"Inlet Velocity: {u_in}")
    print(f"Viscosity: {solver.nu:.6f}")
    print(f"Smagorinsky Cs: {cs}")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=True)

    global_start = time.time()
    
    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]
    diag.save_data(0, additional_data={'cd': 0.0, 'cl': 0.0, 'mask': mask})

    # 模擬開始時間
    sim_start = time.time()

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        # 時間推進
        solver.step(f_src, f_dst)
        
        # 粒子推進
        solver.step_particles(emit=(step % 5 == 0), num_lines=32)

        # 診斷與輸出
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
            'res_y': res_y, 're': re, 'u_in': u_in, 'aoa': aoa,
            'slat': slat_angle, 'flap': flap_angle, 'chord': main_chord
        }
    })
    print(f"📊 History saved to {history_file}")

    # === 空氣動力學結果 ===
    cd, cl = diag.get_force_coefficients()
    ld_ratio = cl / cd if abs(cd) > 1e-9 else 0.0

    print(f"\n{'Aerodynamic Performance':<30}")
    print(f"  Lift Coefficient (Cl)        : {cl:.4f}")
    print(f"  Drag Coefficient (Cd)        : {cd:.4f}")
    print(f"  Lift-to-Drag Ratio (L/D)     : {ld_ratio:.2f}")

    print(f"\n{'Configuration Analysis':<30}")
    print(f"  High-Lift System Benefit:")
    print(f"    - Slat: Delays leading edge separation")
    print(f"    - Flap: Increases camber & lift")
    print(f"    - Expected Cl increase: 50-100% vs clean wing")

    if cl > 1.5:
        print(f"\n  ✅ High-Lift System Effective (Cl > 1.5)")
    elif cl > 1.0:
        print(f"\n  ⚠️  Moderate Lift (Cl > 1.0)")
    else:
        print(f"\n  ❌ Low Lift (Cl < 1.0) - Check configuration")


def main():
    parser = argparse.ArgumentParser(description="High-Lift Airfoil System Simulation")
    parser.add_argument('--res', type=int, default=256, help='Y resolution')
    parser.add_argument('--re', type=float, default=1000.0, help='Reynolds number')
    parser.add_argument('--u_in', type=float, default=0.1, help='Inlet velocity')
    parser.add_argument('--naca', type=str, default='2412', help='Main airfoil NACA code')
    parser.add_argument('--chord', type=float, default=None, help='Main airfoil chord (default: 60% of res_y)')
    parser.add_argument('--aoa', type=float, default=10.0, help='Angle of attack (deg)')
    parser.add_argument('--slat', type=float, default=20.0, help='Slat angle (deg)')
    parser.add_argument('--flap', type=float, default=30.0, help='Flap angle (deg)')
    parser.add_argument('--cs', type=float, default=0.16, help='Smagorinsky constant')
    parser.add_argument('--steps', type=int, default=50000, help='Total steps')
    parser.add_argument('--interval', type=int, default=100, help='Save interval')
    parser.add_argument('--tol', type=float, default=1e-5, help='Convergence tolerance')
    parser.add_argument('--output', type=str, default='output_airfoil', help='Output directory')

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_airfoil(
        res_y=args.res,
        re=args.re,
        u_in=args.u_in,
        main_naca=args.naca,
        main_chord=args.chord,
        aoa=args.aoa,
        slat_angle=args.slat,
        flap_angle=args.flap,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
