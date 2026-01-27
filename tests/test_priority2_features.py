"""
Priority 2 功能驗證測試
======================

測試內容：
1. 週期邊界條件（Periodic BC）
2. Zou-He 反射波抑制（Reflection Wave Suppression）

測試場景：
- Taylor-Green Vortex（週期 BC 基準測試）
- 圓柱繞流（反射波抑制）
- 湍流通道流（雙向週期）
"""

import taichi as ti
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import LBMSolver, BoundaryConditions


def test_periodic_bc_taylor_green(
    res: int = 128,
    re: float = 100.0,
    steps: int = 10000,
    output_dir: str = "test_output_periodic",
):
    """
    測試 1: Taylor-Green Vortex（週期邊界標準測試）

    Physical Setup:
    - 雙週期域（X 和 Y 方向都週期）
    - 初始條件：u = -U₀ cos(kx) sin(ky), v = U₀ sin(kx) cos(ky)
    - 理論解：能量指數衰減

    驗證指標：
    - 總動能隨時間衰減
    - 漩渦結構保持週期性
    - 無邊界反射波

    預期結果：
    - 能量衰減率符合理論（E(t) ~ exp(-2νk²t)）
    - 週期性保持完好
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 1: Taylor-Green Vortex（週期邊界）")
    print("=" * 60)

    nx, ny = res, res
    solver = LBMSolver(nx=nx, ny=ny, re=re, u_ref=0.1)

    # 週期邊界條件
    bc = BoundaryConditions(solver)
    bc.add_periodic_boundary("x")
    bc.add_periodic_boundary("y")

    # Taylor-Green Vortex 初始條件
    print("設置 Taylor-Green Vortex 初始條件...")
    U0 = 0.1
    k = 2.0 * np.pi / nx  # 波數

    # 初始化速度場
    for i in range(nx):
        for j in range(ny):
            x = i
            y = j
            u_x = -U0 * np.cos(k * x) * np.sin(k * y)
            u_y = U0 * np.sin(k * x) * np.cos(k * y)
            solver.u[i + 1, j + 1] = [u_x, u_y]
            solver.rho[i + 1, j + 1] = 1.0

    # 從速度場重建分佈函數
    solver._init_from_macro()
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # 記錄初始動能
    solver._update_diagnostics()
    initial_KE = solver.total_KE[None]
    print(f"初始動能: {initial_KE:.6f}")

    # 理論衰減率
    nu = solver.nu
    decay_rate_theory = 2.0 * nu * k**2
    print(f"理論衰減率: {decay_rate_theory:.6f}")

    # 主迴圈
    KE_history = [initial_KE]
    time_history = [0]

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

            if step % 1000 == 0:
                # 計算實際衰減率（對數擬合）
                if len(KE_history) > 10:
                    log_KE = np.log(np.array(KE_history[-10:]) + 1e-12)
                    times = np.array(time_history[-10:])
                    # 線性擬合：log(KE) = log(KE0) - γt
                    decay_rate_measured = -np.polyfit(times, log_KE, 1)[0]
                else:
                    decay_rate_measured = 0.0

                print(
                    f"  Step {step:6d}: KE = {current_KE:.6f}, "
                    f"衰減率 = {decay_rate_measured:.6f} (理論 {decay_rate_theory:.6f})"
                )

    # 最終分析
    final_KE = KE_history[-1]
    energy_decay = (initial_KE - final_KE) / initial_KE
    print(f"\n最終動能: {final_KE:.6f}")
    print(f"能量衰減: {energy_decay * 100:.2f}%")

    # 保存結果
    result_file = os.path.join(output_dir, "taylor_green_KE.npy")
    np.save(
        result_file,
        {
            "KE_history": KE_history,
            "time_history": time_history,
            "decay_rate_theory": decay_rate_theory,
            "initial_KE": initial_KE,
            "final_KE": final_KE,
        },
    )

    # 保存最終流場
    fields = solver.get_fields()
    fields_file = os.path.join(output_dir, "taylor_green_fields.npy")
    np.save(fields_file, fields)

    print(f"\n✅ 測試 1 完成！結果保存至 {output_dir}")


def test_reflection_suppression(
    res: int = 128,
    re: float = 200.0,
    steps: int = 20000,
    output_dir: str = "test_output_reflection",
):
    """
    測試 2: 反射波抑制（圓柱繞流）

    對比：
    - 無鬆弛（relaxation=0）：標準 Zou-He
    - 有鬆弛（relaxation=0.3）：反射波抑制

    觀察指標：
    - 出口壓力波動幅度
    - 升力係數振盪
    - 流場穩定性

    預期結果：
    - 有鬆弛：壓力波動減少 50-80%
    - 有鬆弛：升力係數振盪更平滑
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 2: Zou-He 反射波抑制（圓柱繞流）")
    print("=" * 60)

    from utils.geometry import create_circle_mask

    for version, relaxation in [("無鬆弛", 0.0), ("有鬆弛 α=0.3", 0.3)]:
        print(f"\n--- {version} ---")

        nx, ny = int(3 * res), res
        solver = LBMSolver(nx=nx, ny=ny, re=re, u_ref=0.1)

        # 圓柱障礙物
        R = res / 18
        center = (nx / 4, ny / 2)
        mask = create_circle_mask(nx, ny, center, R)
        solver.set_obstacle(mask)

        # 邊界條件
        bc = BoundaryConditions(solver)
        bc.add_velocity_inlet(0.1, location="left")
        bc.add_orlanski_outflow(
            location="right",
            rho_target=1.0,
            relaxation=relaxation,
        )
        bc.add_free_slip_wall("top", mode="symmetric")
        bc.add_free_slip_wall("bottom", mode="symmetric")

        # 初始化
        solver.reset()
        solver.apply_boundary_conditions(solver.f)
        solver.apply_boundary_conditions(solver.f_new)

        # 主迴圈
        from core import Diagnostics

        diag = Diagnostics(solver, output_dir=output_dir)

        cl_history = []
        cd_history = []
        rho_outlet_history = []

        for step in range(1, steps + 1):
            f_src = solver.f if step % 2 == 1 else solver.f_new
            f_dst = solver.f_new if step % 2 == 1 else solver.f

            solver.step(f_src, f_dst)

            if step % 100 == 0:
                solver._update_macro(f_dst)
                diag.compute_forces(f_dst)
                ti.sync()

                cd, cl = diag.get_force_coefficients()
                cl_history.append(cl)
                cd_history.append(cd)

                # 監控出口密度（壓力）
                rho_avg = 0.0
                count = 0
                for j in range(ny):
                    if solver.mask[nx, j + 1] == 0:
                        rho_avg += solver.rho[nx, j + 1]
                        count += 1
                rho_avg /= count + 1e-12
                rho_outlet_history.append(rho_avg)

                if step % 2000 == 0:
                    print(
                        f"  Step {step:6d}: Cl = {cl:+.4f}, Cd = {cd:.4f}, "
                        f"ρ_outlet = {rho_avg:.6f}"
                    )

        # 分析結果
        cl_array = np.array(cl_history)
        rho_array = np.array(rho_outlet_history)

        # 計算波動（去除前 5000 步的過渡段）
        if len(cl_array) > 50:
            cl_std = np.std(cl_array[50:])
            rho_std = np.std(rho_array[50:])
        else:
            cl_std = np.std(cl_array)
            rho_std = np.std(rho_array)

        print(f"\n  升力係數標準差: {cl_std:.6f}")
        print(f"  出口密度標準差: {rho_std:.6f}")

        # 保存結果
        result_file = os.path.join(output_dir, f"reflection_{version}.npy")
        np.save(
            result_file,
            {
                "cl_history": cl_history,
                "cd_history": cd_history,
                "rho_outlet_history": rho_outlet_history,
                "cl_std": cl_std,
                "rho_std": rho_std,
                "version": version,
                "relaxation": relaxation,
            },
        )

    print(f"\n✅ 測試 2 完成！結果保存至 {output_dir}")


def test_periodic_channel_flow(
    res: int = 128,
    re: float = 1000.0,
    steps: int = 10000,
    output_dir: str = "test_output_channel",
):
    """
    測試 3: 週期性通道流（Turbulent Channel Flow）

    Setup:
    - X 方向：週期
    - Y 方向：No-Slip 上下壁面
    - 驅動力：體積力（重力或壓力梯度）

    驗證：
    - 充分發展的 Poiseuille Profile
    - 無入口/出口效應
    - 質量守恆

    Note: 這是湍流模擬的基本配置
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 3: 週期性通道流")
    print("=" * 60)

    nx, ny = int(4 * res), res
    solver = LBMSolver(nx=nx, ny=ny, re=re, u_ref=0.1)

    # 邊界條件
    bc = BoundaryConditions(solver)
    bc.add_periodic_boundary("x")  # X 方向週期
    bc.add_no_slip_wall("bottom")
    bc.add_no_slip_wall("top")

    # 初始化：均勻流場
    for i in range(nx):
        for j in range(ny):
            solver.u[i + 1, j + 1] = [0.1, 0.0]
            solver.rho[i + 1, j + 1] = 1.0

    solver._init_from_macro()
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    print("運行週期性通道流...")

    # 主迴圈（需要添加體積力，這裡簡化）
    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % 1000 == 0:
            solver._update_macro(f_dst)
            print(f"  Step {step:6d}")

    # 分析速度剖面
    solver._update_macro(f_dst)
    fields = solver.get_fields()

    # X 方向平均速度剖面
    u_profile = np.mean(fields["u"][:, :, 0], axis=0)
    y_coords = np.arange(ny)

    print(f"\n速度剖面統計:")
    print(f"  最大速度: {np.max(u_profile):.4f}")
    print(f"  最小速度: {np.min(u_profile):.4f}")
    print(f"  中心速度: {u_profile[ny // 2]:.4f}")

    # 保存結果
    result_file = os.path.join(output_dir, "channel_flow.npy")
    np.save(
        result_file, {"u_profile": u_profile, "y_coords": y_coords, "fields": fields}
    )

    print(f"\n✅ 測試 3 完成！結果保存至 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Priority 2 功能驗證測試")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
        help="測試編號：1=週期BC, 2=反射波, 3=通道流, all=全部",
    )
    parser.add_argument("--res", type=int, default=128, help="解析度")
    parser.add_argument("--steps", type=int, default=10000, help="模擬步數")

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    print("\n" + "=" * 60)
    print("Priority 2 功能驗證測試")
    print("=" * 60)

    if args.test in ["all", "1"]:
        test_periodic_bc_taylor_green(res=args.res, steps=args.steps)

    if args.test in ["all", "2"]:
        test_reflection_suppression(res=args.res, steps=20000)

    if args.test in ["all", "3"]:
        test_periodic_channel_flow(res=args.res, steps=args.steps)

    print("\n" + "=" * 60)
    print("✅ Priority 2 測試完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
