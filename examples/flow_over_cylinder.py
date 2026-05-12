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
from lbm_taichi.core import LBMSolver, BoundaryConditions, Diagnostics
from lbm_taichi.utils.geometry import create_circle_mask_and_sdf


def _cylinder_force_coefficients_numpy(
    runner: CaseRunner,
    solver: LBMSolver,
    *,
    diameter: float,
) -> tuple[float, float]:
    """
    計算圓柱 benchmark 的動量交換力係數。

    What:
    - 以目前 active LBM 分佈函數與固體 mask 估算 Cd / Cl

    Why:
    - registry diagnostics hook 不應建立完整 `Diagnostics` 輸出管理器
    - benchmark acceptance 需要力係數可觀測量，而不是只看速度場是否非零
    """
    f_field = runner.active_distribution_field()
    if f_field is None:
        f_field = solver.f
    f_np = f_field.to_numpy()
    mask = solver.mask.to_numpy()
    e = solver.e.to_numpy()
    inv = solver.inv.to_numpy()

    force_x = 0.0
    force_y = 0.0
    for i in range(solver.nx):
        for j in range(solver.ny):
            ig = i + 1
            jg = j + 1
            if mask[ig, jg] != 0:
                continue
            for k in range(9):
                ni = i + int(e[k, 0])
                nj = j + int(e[k, 1])
                if 0 <= ni < solver.nx and 0 <= nj < solver.ny:
                    if mask[ni + 1, nj + 1] == 1:
                        bounced = float(f_np[ig, jg, int(inv[k])])
                        force_x += 2.0 * bounced * float(e[k, 0])
                        force_y += 2.0 * bounced * float(e[k, 1])

    denom = 0.5 * solver.u_ref**2 * diameter
    return force_x / (denom + 1e-12), force_y / (denom + 1e-12)


def _initialize_cylinder_case(
    runner: CaseRunner,
    solver: LBMSolver,
    _bc_handle,
    *,
    cx: float,
    cy: float,
    radius: float,
    mask: np.ndarray,
    sdf: np.ndarray,
    re: float,
):
    """
    圓柱繞流 CaseRunner 初始化流程。

    What:
    - 設定圓柱障礙物、勢流初始場與初始邊界條件

    Why:
    - registry 建出的 runner 必須能自洽完成幾何與流場初始化
    - standalone driver 與 benchmark workflow 應共用同一套物理初始化假設
    """
    solver.initialize_obstacle(mask, sdf_array=sdf)
    solver.init_potential_flow_cylinder(cx=cx, cy=cy, radius=radius)
    solver.set_wall_function(enabled=re > 1000.0, yplus_min=5.0)
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)
    runner.prepare_observables(reset_baseline=True)


def _cylinder_diagnostics_hook(
    runner: CaseRunner,
    solver: LBMSolver,
    diagnostics: dict,
    *,
    diameter: float,
    obstacle_area: int,
    sidewall: str,
    outflow_type: str,
) -> dict:
    """
    派生 cylinder benchmark 專屬 diagnostics。

    Why:
    - 圓柱繞流的可觀測量至少應包含 obstacle 幾何、Cd/Cl 與尾流活性
    """
    cd, cl = _cylinder_force_coefficients_numpy(runner, solver, diameter=diameter)
    u_max = float(diagnostics.get("u_max", diagnostics.get("max_u", np.nan)))
    return {
        "u_max": u_max,
        "drag_coefficient": float(cd),
        "drag_coefficient_abs": float(abs(cd)),
        "lift_coefficient": float(cl),
        "lift_coefficient_abs": float(abs(cl)),
        "obstacle_cells": float(obstacle_area),
        "diameter": float(diameter),
        "sidewall": sidewall,
        "outflow_type": outflow_type,
    }


def build_flow_over_cylinder_runner(
    *,
    res_y: int = 128,
    re: float = 150.0,
    u_in: float = 0.1,
    diameter: float | None = None,
    cs: float = 0.16,
    sidewall: str = "orlanski",
    outflow_type: str = "orlanski",
    outlet_relaxation: float = 0.02,
    collision_model: str = "mrt",
) -> tuple[CaseRunner, dict]:
    """
    建立圓柱繞流的 CaseRunner。

    What:
    - 將 LBM solver、圓柱幾何、入口/出口/側壁 BC 與 diagnostics hook 集中成正式 builder

    Why:
    - Flow-over-cylinder 是 LBM bluff-body 主線，應進入 registry / matrix workflow
    """
    if res_y <= 0:
        raise ValueError(f"res_y must be positive, got {res_y}")
    if re <= 0.0:
        raise ValueError(f"re must be positive, got {re}")
    if u_in <= 0.0:
        raise ValueError(f"u_in must be positive, got {u_in}")
    if sidewall not in {"orlanski", "freeslip"}:
        raise ValueError(f"sidewall must be 'orlanski' or 'freeslip', got {sidewall}")
    if outflow_type != "orlanski":
        raise ValueError("Only Orlanski outflow is supported in the toolkit runner.")

    nx = int(2.5 * res_y)
    ny = int(res_y)
    diameter_eff = float(res_y / 9.0 if diameter is None else diameter)
    radius = diameter_eff / 2.0
    cx = nx / 4.0
    cy = ny / 2.0
    auto_les = re > 2000.0
    cs_eff = cs if cs > 0.0 else (0.16 if auto_les else cs)
    mask, sdf = create_circle_mask_and_sdf(nx, ny, center=(cx, cy), radius=radius)

    boundary_conditions = [
        BoundaryConditionDescriptor.velocity_inlet(
            "left",
            u_in=u_in,
            epsilon=0.02,
            strouhal=0.2,
            asymmetry=0.03,
        ),
        BoundaryConditionDescriptor.stable_outlet(
            "right",
            rho_out=1.0,
            relaxation=outlet_relaxation,
        ),
    ]
    if sidewall == "orlanski":
        boundary_conditions.extend(
            [
                BoundaryConditionDescriptor.orlanski_outflow(
                    "top",
                    relaxation=outlet_relaxation,
                ),
                BoundaryConditionDescriptor.orlanski_outflow(
                    "bottom",
                    relaxation=outlet_relaxation,
                ),
            ]
        )
    else:
        boundary_conditions.extend(
            [
                BoundaryConditionDescriptor.free_slip("top"),
                BoundaryConditionDescriptor.free_slip("bottom"),
            ]
        )

    runner = CaseRunner(
        name="flow_over_cylinder",
        method="lbm",
        equation="single_phase",
        regime="low_mach",
        grid=LatticeGrid2D(nx=nx, ny=ny),
        solver_kwargs={
            "re": re,
            "u_ref": u_in,
            "length_scale": diameter_eff,
            "cs": cs_eff,
            "collision_model": collision_model,
        },
        boundary_conditions=boundary_conditions,
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_cylinder_case(
            runner_obj,
            solver_obj,
            bc_handle,
            cx=cx,
            cy=cy,
            radius=radius,
            mask=mask,
            sdf=sdf,
            re=re,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _cylinder_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            diameter=diameter_eff,
            obstacle_area=int(np.count_nonzero(mask)),
            sidewall=sidewall,
            outflow_type=outflow_type,
        ),
    )
    setup = {
        "nx": nx,
        "ny": ny,
        "diameter": diameter_eff,
        "radius": radius,
        "cx": cx,
        "cy": cy,
        "cs_eff": cs_eff,
        "obstacle_cells": int(np.count_nonzero(mask)),
    }
    return runner, setup


def run_flow_over_cylinder(
    res_y: int = 256,
    re: float = 150.0,
    u_in: float = 0.1,
    diameter: float = None,
    cs: float = 0.16,
    steps: int = 10000,
    interval: int = 100,
    tol: float = 1e-5,
    output_dir: str = "output_cylinder",
    sidewall: str = "orlanski",
    outflow_type: str = "orlanski",
    outlet_relaxation: float = 0.02,
    vtk_output: bool = False,
    collision_model: str = "mrt",
):
    """
    執行圓柱繞流模擬

    Args:
        res_y: Y 方向解析度
        re: Reynolds 數
        u_in: 入口速度
        diameter: 圓柱直徑（若為 None 則使用 res_y/9）
        cs: LES 啟用旗標 (<=0 表示不使用 LES；動態 Smagorinsky 自動估計)
        steps: 總步數
        interval: 儲存間隔
        tol: 收斂容差
        output_dir: 輸出目錄
        sidewall: 上下邊界類型
            - 'orlanski': 開放邊界（Orlanski 非反射）
            - 'freeslip': Free-Slip 壁面（模擬風洞側壁）
        outflow_type: 出口邊界類型
            - 'orlanski': 非反射出口（推薦）
    """
    print("=" * 70)
    print(" " * 20 + "FLOW OVER CYLINDER")
    print("=" * 70)

    # === 計算網格與幾何參數 ===
    nx = int(2.5 * res_y)  # 長寬比 2.5:1
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
    auto_les = re > 2000.0
    cs_eff = cs if cs > 0.0 else (0.16 if auto_les else cs)

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        re=re,
        u_ref=u_in,
        length_scale=diameter,
        cs=cs_eff,
        collision_model=collision_model,
    )
    if re > 1000.0:
        solver.set_wall_function(enabled=True, yplus_min=5.0)
        print("Wall Function: Enabled (Re > 1000, yplus_min=5.0)")
    else:
        solver.set_wall_function(enabled=False)
        print("Wall Function: Disabled (Re <= 1000)")

    # === 設定障礙物 ===
    mask, sdf = create_circle_mask_and_sdf(nx, ny, center=(cx, cy), radius=radius)
    solver.initialize_obstacle(mask, sdf_array=sdf)

    # === 勢流初始化（減少初始擾動）===
    solver.init_potential_flow_cylinder(cx=cx, cy=cy, radius=radius)

    # === 設定邊界條件 ===
    bc = BoundaryConditions(solver)

    # 入口：固定速度
    bc.add_velocity_inlet(
        u_in,
        location="left",
        epsilon=0.02,
        strouhal=0.2,
        asymmetry=0.03,
    )
    inlet_eps = float(bc.u_inlet_perturb[None])
    inlet_omega = float(bc.u_inlet_omega[None])
    inlet_asym = float(bc.u_inlet_asym[None])
    if inlet_omega > 0.0:
        inlet_period = 2.0 * np.pi / inlet_omega
        print(
            f"  Inlet Perturbation: epsilon={inlet_eps:.3f}, "
            f"omega={inlet_omega:.4e}, period={inlet_period:.1f} steps, "
            f"asymmetry={inlet_asym:.3f}"
        )

    # 出口：根據 outflow_type 選擇
    if outflow_type == "orlanski":
        bc.add_stable_outlet(
            rho_out=1.0,
            location="right",
            relaxation=outlet_relaxation,
        )
        print(f"  Outflow Type: Orlanski Outflow (relaxation={outlet_relaxation})")
    elif outflow_type == "neumann":
        raise ValueError("Neumann outflow removed; use --outflow orlanski")
    else:
        raise ValueError(f"Unknown outflow_type: {outflow_type}")

    # 上下邊界：根據 sidewall 選擇
    if sidewall == "orlanski":
        bc.add_orlanski_outflow(location="top", relaxation=outlet_relaxation)
        bc.add_orlanski_outflow(location="bottom", relaxation=outlet_relaxation)
        print(
            f"  Sidewall Type: Orlanski Outflow (open sky, relaxation={outlet_relaxation})"
        )
    elif sidewall == "freeslip":
        bc.add_free_slip_wall("top")
        bc.add_free_slip_wall("bottom")
        print("  Sidewall Type: Free-Slip (wind tunnel)")
    else:
        raise ValueError(f"Unknown sidewall type: {sidewall}")

    # 施加初始邊界條件
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # === 診斷系統 ===
    diag = Diagnostics(solver, output_dir=output_dir, output_vtk=vtk_output)

    # === 主迴圈 ===
    print(f"Reynolds Number: {re}")
    print(f"Inlet Velocity: {u_in}")
    print(f"Viscosity: {solver.nu:.6f}")
    if cs_eff > 0.0:
        print("LES Model: Dynamic Smagorinsky (auto Cs)")
    else:
        print("LES Model: Disabled")
    print(f"\nStarting simulation...")

    headers = diag.print_header(include_forces=True)

    global_start = time.time()

    # 記錄初始質量與能量
    solver.prepare_diagnostics(reset_baseline=True)
    diag.save_data(0, additional_data={"cd": 0.0, "cl": 0.0})

    # 模擬開始時間
    sim_start = time.time()

    last_step = steps
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
                step, speed, eta_seconds, include_forces=True, f_field=f_dst
            )
            cd, cl = diag.get_force_coefficients()
            diag.record_forces(step, cd, cl)

            if step % 500 == 0:
                diag.history.append(row)

            # CFL 條件檢查（每 1000 步）
            if step % 1000 == 0:
                solver.check_cfl_condition(warn_only=True)

            if step % interval == 0:
                diag.save_data(step, additional_data={"cd": cd, "cl": cl})

            if diag.check_convergence(tol):
                print(f"\n✅ Converged at step {step}")
                diag.history.append(row)
                last_step = step
                if step % interval != 0:
                    diag.save_data(step, additional_data={"cd": cd, "cl": cl})
                break

        elif step % interval == 0:
            solver.prepare_diagnostics(f_dst)
            cd, cl = diag.get_force_coefficients()
            diag.record_forces(step, cd, cl)
            diag.save_data(step, additional_data={"cd": cd, "cl": cl})

    # === 總結 ===
    total_time = time.time() - global_start
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")

    diag.print_summary(headers)

    spectrum_start = int(0.2 * last_step)
    diag.print_force_spectrum(
        diameter=diameter,
        u_ref=u_in,
        min_step=spectrum_start,
        max_peaks=3,
    )

    min_step = int(0.5 * last_step)
    stats = diag.compute_force_stats(min_step=min_step)
    st = diag.compute_strouhal(diameter=diameter, u_ref=u_in, min_step=min_step)
    if stats["samples"] > 0:
        print("\n--- DFG Benchmark Metrics ---")
        print(f"Cd_mean   : {stats['cd_mean']:.4f}")
        print(f"Cl_rms    : {stats['cl_rms']:.4f}")
        print(f"Strouhal  : {st:.4f}")

    # 存儲歷史數據
    history_file = diag.save_history(
        params={
            "res_y": res_y,
            "re": re,
            "u_in": u_in,
            "diameter": diameter,
            "cs": cs,
        },
        additional_payload={
            "dfg": {
                "cd_mean": stats.get("cd_mean", 0.0),
                "cl_rms": stats.get("cl_rms", 0.0),
                "strouhal": st,
            }
        },
    )
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
    parser.add_argument("--res", type=int, default=128, help="Y resolution")
    parser.add_argument("--re", type=float, default=150.0, help="Reynolds number")
    parser.add_argument("--u_in", type=float, default=0.1, help="Inlet velocity")
    parser.add_argument(
        "--diameter", type=float, default=None, help="Cylinder diameter"
    )
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
        "--output", type=str, default="output_cylinder", help="Output directory"
    )
    parser.add_argument(
        "--sidewall",
        type=str,
        default="orlanski",
        choices=["orlanski", "freeslip"],
        help="Top/bottom boundary: orlanski (open sky) or freeslip (wind tunnel)",
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

    run_flow_over_cylinder(
        res_y=args.res,
        re=args.re,
        u_in=args.u_in,
        diameter=args.diameter,
        cs=args.cs,
        steps=args.steps,
        interval=args.interval,
        tol=args.tol,
        output_dir=args.output,
        sidewall=args.sidewall,
        outflow_type="orlanski",
        outlet_relaxation=args.outlet_relax,
        vtk_output=args.vtk,
        collision_model=args.collision,
    )


if __name__ == "__main__":
    main()
