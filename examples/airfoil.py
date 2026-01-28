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

import os
import sys
import taichi as ti
import numpy as np
import argparse
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lbm_taichi.core import LBMSolver, BoundaryConditions, Diagnostics
from lbm_taichi.utils.geometry import create_airfoil_system


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
    output_dir: str = "output_airfoil",
    sidewall: str = "outflow",
    outflow_type: str = "orlanski",
    outlet_relaxation: float = 0.02,
    vtk_output: bool = False,
    collision_model: str = "mrt",
):
    # === 計算網格與弦長比例 ===
    nx = int(2.5 * res_y)  # 調整計算域為 3.5 倍高度
    ny = res_y

    if main_chord is None:
        main_chord = res_y * 0.5  # 自動設定弦長為高度的 60%

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
        cs=cs,
        collision_model=collision_model,
    )

    # === 生成機翼幾何 ===
    print("\nGenerating airfoil geometry...")
    mask, sdf = create_airfoil_system(
        nx=nx,
        ny=ny,
        main_naca=main_naca,
        main_chord=main_chord,
        aoa=aoa,
        slat_angle=slat_angle,
        flap_angle=flap_angle,
        center_position=(nx / 5, ny / 2),
        return_sdf=True,
    )

    solver.set_obstacle(mask, sdf_array=sdf)
    solver._correct_solid_velocity()
    print(
        f"Solid cells: {mask.sum()} / {nx * ny} ({100 * mask.sum() / (nx * ny):.2f}%)"
    )

    # === 勢流初始化（考慮攻角與環量）===
    solver.init_potential_flow_airfoil(
        chord=main_chord, aoa_deg=aoa, center=(nx / 4, ny / 2)
    )

    # === 設定邊界條件 ===
    bc = BoundaryConditions(solver)

    # 入口：固定速度
    bc.add_velocity_inlet(u_in, location="left")

    # 出口：根據 outflow_type 選擇
    if outflow_type == "orlanski":
        bc.add_stable_outlet(
            rho_out=1.0,
            location="right",
            relaxation=outlet_relaxation,
        )
        print(f"  Outflow Type: Orlanski Outflow (relaxation={outlet_relaxation})")
    elif outflow_type == "neumann":
        bc.add_neumann_outflow(location="right")
        print(f"  Outflow Type: Neumann (zero-gradient)")
    else:
        raise ValueError(f"Unknown outflow_type: {outflow_type}")

    # 上下邊界：根據 sidewall 選擇
    if sidewall == "outflow":
        bc.add_neumann_outflow("top")
        bc.add_neumann_outflow("bottom")
        print(f"  Sidewall Type: Neumann Outflow (open sky)")
    elif sidewall == "freeslip":
        bc.add_free_slip_wall("top", mode="symmetric")
        bc.add_free_slip_wall("bottom", mode="symmetric")
        print(f"  Sidewall Type: Free-Slip (wind tunnel)")
    else:
        raise ValueError(f"Unknown sidewall type: {sidewall}")

    # 施加初始邊界條件
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir, output_vtk=vtk_output)

    # === 主迴圈 ===
    print(f"\nReynolds Number: {re}")
    print(f"Inlet Velocity: {u_in}")
    print(f"Viscosity: {solver.nu:.6f}")
    if cs > 0.0:
        print("LES Model: Dynamic Smagorinsky (auto Cs)")
    else:
        print("LES Model: Disabled")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=True)

    global_start = time.time()

    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]
    solver.initial_KE[None] = solver.total_KE[None]
    diag.save_data(0, additional_data={"cd": 0.0, "cl": 0.0, "mask": mask})

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

            row = diag.print_step_info(
                step, speed, eta_seconds, include_forces=True, f_field=f_dst
            )

            if step % 500 == 0:
                diag.history.append(row)

            if step % interval == 0:
                cd, cl = diag.get_force_coefficients()
                diag.save_data(step, additional_data={"cd": cd, "cl": cl})

            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                if step % interval != 0:
                    cd, cl = diag.get_force_coefficients()
                    diag.save_data(step, additional_data={"cd": cd, "cl": cl})
                break

        elif step % interval == 0:
            solver._update_macro(f_dst)
            cd, cl = diag.get_force_coefficients()
            diag.save_data(step, additional_data={"cd": cd, "cl": cl})

    # === 總結 ===
    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    # 存儲歷史數據
    history_file = os.path.join(output_dir, "history.npy")
    np.save(
        history_file,
        {
            "headers": headers,
            "data": diag.history,
            "params": {
                "res_y": res_y,
                "re": re,
                "u_in": u_in,
                "aoa": aoa,
                "slat": slat_angle,
                "flap": flap_angle,
                "chord": main_chord,
            },
        },
    )
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
    parser.add_argument("--res", type=int, default=256, help="Y resolution")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--u_in", type=float, default=0.1, help="Inlet velocity")
    parser.add_argument(
        "--naca", type=str, default="2412", help="Main airfoil NACA code"
    )
    parser.add_argument(
        "--chord",
        type=float,
        default=None,
        help="Main airfoil chord (default: 60% of res_y)",
    )
    parser.add_argument("--aoa", type=float, default=10.0, help="Angle of attack (deg)")
    parser.add_argument("--slat", type=float, default=20.0, help="Slat angle (deg)")
    parser.add_argument("--flap", type=float, default=30.0, help="Flap angle (deg)")
    parser.add_argument(
        "--cs",
        type=float,
        default=0.16,
        help="LES enable flag (<=0 disables dynamic Smagorinsky)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total steps")
    parser.add_argument("--interval", type=int, default=100, help="Save interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument(
        "--output", type=str, default="output_airfoil", help="Output directory"
    )
    parser.add_argument(
        "--sidewall",
        type=str,
        default="outflow",
        choices=["outflow", "freeslip"],
        help="Top/bottom boundary: outflow (open sky) or freeslip (wind tunnel)",
    )
    parser.add_argument(
        "--outflow",
        type=str,
        default="orlanski",
        choices=["orlanski", "neumann"],
        help="Right outlet type: orlanski or neumann",
    )
    parser.add_argument(
        "--outlet_relax",
        type=float,
        default=0.02,
        help="Outlet relaxation factor (stable outlet)",
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
        output_dir=args.output,
        sidewall=args.sidewall,
        outflow_type=args.outflow,
        outlet_relaxation=args.outlet_relax,
        vtk_output=args.vtk,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
