"""
邊界條件改進驗證測試
===================

測試 Critical Issues 修正：
1. Neumann Outflow 質量修正
2. 角點外推處理
3. Sponge Layer 穩定性
4. 全局質量修正

測試方法：
- 長時間模擬（50,000 步）
- 監控質量守恆誤差
- 對比修正前後差異
"""

import taichi as ti
import numpy as np
import argparse
import os
import sys

# 添加父目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import LBMSolver, BoundaryConditions


def test_neumann_mass_conservation(
    steps: int = 50000, res: int = 128, output_dir: str = "test_output_neumann"
):
    """
    測試 1: Neumann BC 質量守恆

    對比：
    - 舊版：純零梯度（mass_corrected=False）
    - 新版：零梯度 + 質量修正（mass_corrected=True）

    預期結果：
    - 舊版：質量誤差 ±0.5-1%
    - 新版：質量誤差 < 0.01%
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 1: Neumann Outflow 質量守恆")
    print("=" * 60)

    nx, ny = int(3 * res), res
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)

    # 設置邊界條件
    bc = BoundaryConditions(solver)
    bc.add_velocity_inlet(0.1, location="left")
    bc.add_neumann_outflow(location="right")
    bc.add_neumann_outflow(location="top")
    bc.add_neumann_outflow(location="bottom")

    # 初始化
    solver.reset()
    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # 記錄初始質量
    solver._update_macro(solver.f)
    solver._update_diagnostics()
    solver.initial_mass[None] = solver.total_mass[None]

    # 主迴圈
    mass_errors = []
    check_interval = 1000

    for step in range(1, steps + 1):
        f_src = solver.f if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new if step % 2 == 1 else solver.f

        solver.step(f_src, f_dst)

        if step % check_interval == 0:
            solver._update_macro(f_dst)
            solver._update_diagnostics()
            ti.sync()

            diag = solver.get_diagnostics()
            mass_error = (
                abs(diag["total_mass"] - diag["initial_mass"]) / diag["initial_mass"]
            )
            mass_errors.append(mass_error)

            print(
                f"  Step {step:6d}: 質量誤差 = {mass_error:.6f} ({mass_error * 100:.4f}%)"
            )

    # 最終統計
    final_error = mass_errors[-1]
    max_error = max(mass_errors)
    print(f"\n  最終質量誤差: {final_error:.6f} ({final_error * 100:.4f}%)")
    print(f"  最大質量誤差: {max_error:.6f} ({max_error * 100:.4f}%)")

    # 保存結果
    result_file = os.path.join(output_dir, "mass_error.npy")
    np.save(
        result_file,
        {
            "mass_errors": mass_errors,
            "final_error": final_error,
            "max_error": max_error,
        },
    )

    print("\n✅ 測試 1 完成！結果保存至", output_dir)


def test_corner_handling(
    res: int = 64, steps: int = 10000, output_dir: str = "test_output_corner"
):
    """
    測試 2: 角點處理

    對比：
    - 方法 A：固體角點（set_corners_solid）
    - 方法 B：外推角點（handle_corners_extrapolation）

    場景：Lid-Driven Cavity（小計算域）

    預期結果：
    - 方法 A：中心渦流位置可能偏移（角點阻礙流動）
    - 方法 B：中心渦流位置更準確
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 2: 角點處理（Lid-Driven Cavity）")
    print("=" * 60)

    from utils.geometry import create_lid_velocity_profile

    for method, use_extrapolation in [("固體角點", False), ("外推角點", True)]:
        print(f"\n--- {method} ---")

        solver = LBMSolver(nx=res, ny=res, re=100.0, u_ref=0.1)
        bc = BoundaryConditions(solver)

        # 設置邊界條件
        bc.add_no_slip_wall("bottom", exclude_corners=True)
        bc.add_no_slip_wall("left", exclude_corners=True)
        bc.add_no_slip_wall("right", exclude_corners=True)

        u_wall_profile = create_lid_velocity_profile(res, 0.1)
        bc.add_moving_wall(u_wall_profile, location="top")

        if use_extrapolation:
            bc.handle_corners_extrapolation()
        else:
            bc.set_corners_solid()

        # 初始化
        solver.reset()
        solver.apply_boundary_conditions(solver.f)
        solver.apply_boundary_conditions(solver.f_new)

        # 運行模擬
        for step in range(1, steps + 1):
            f_src = solver.f if step % 2 == 1 else solver.f_new
            f_dst = solver.f_new if step % 2 == 1 else solver.f

            solver.step(f_src, f_dst)

            if step % 1000 == 0:
                print(f"  Step {step:6d}")

        # 分析流場
        solver._update_macro(f_dst)
        fields = solver.get_fields()
        u_mag = np.linalg.norm(fields["u"], axis=2)

        # 找到中心渦流位置（最大速度位置）
        max_idx = np.unravel_index(np.argmax(u_mag), u_mag.shape)
        center_x, center_y = max_idx[0] / res, max_idx[1] / res

        print(f"  中心渦流位置: ({center_x:.3f}, {center_y:.3f})")
        print(f"  最大速度: {np.max(u_mag):.4f}")

        # 保存流場
        result_file = os.path.join(output_dir, f"cavity_{method}.npy")
        np.save(result_file, fields)

    print("\n✅ 測試 2 完成！結果保存至", output_dir)


def test_sponge_layer(
    res: int = 128,
    re: float = 5000,
    steps: int = 20000,
    output_dir: str = "test_output_sponge",
):
    """
    測試 3: Sponge Layer 穩定性

    對比：
    - 無海綿層：enable_sponge=False
    - 有海綿層：enable_sponge=True

    場景：高 Re 數圓柱繞流

    預期結果：
    - 無海綿層：可能出現出口振盪/發散
    - 有海綿層：穩定收斂
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 3: Sponge Layer 穩定性（Re=5000）")
    print("=" * 60)

    from utils.geometry import create_circle_mask

    for version, enable_sponge in [("無海綿層", False), ("有海綿層", True)]:
        print(f"\n--- {version} ---")

        nx, ny = int(3 * res), res
        solver = LBMSolver(
            nx=nx,
            ny=ny,
            re=re,
            u_ref=0.08,
            enable_sponge=enable_sponge,
            sponge_strength=0.5,
        )

        # 圓柱障礙物
        R = res / 18
        center = (nx / 4, ny / 2)
        mask = create_circle_mask(nx, ny, center, R)
        solver.set_obstacle(mask)

        # 邊界條件
        bc = BoundaryConditions(solver)
        bc.add_velocity_inlet(0.08, location="left")
        bc.add_orlanski_outflow(location="right", rho_target=1.0)
        bc.add_free_slip_wall("top", mode="symmetric")
        bc.add_free_slip_wall("bottom", mode="symmetric")

        # 初始化
        solver.reset()
        solver.apply_boundary_conditions(solver.f)
        solver.apply_boundary_conditions(solver.f_new)

        # 主迴圈
        max_u_history = []
        diverged = False

        for step in range(1, steps + 1):
            f_src = solver.f if step % 2 == 1 else solver.f_new
            f_dst = solver.f_new if step % 2 == 1 else solver.f

            solver.step(f_src, f_dst)

            if step % 100 == 0:
                solver._update_macro(f_dst)
                solver._update_diagnostics()
                ti.sync()

                max_u = solver.max_u[None]
                max_u_history.append(max_u)

                if step % 1000 == 0:
                    print(f"  Step {step:6d}: max|u| = {max_u:.4f}")

                # 檢查發散
                if max_u > 0.5:  # 超過合理範圍
                    print(f"  ⚠️ 發散！max|u| = {max_u:.4f} > 0.5")
                    diverged = True
                    break

        if not diverged:
            print(f"  ✅ 穩定收斂！最終 max|u| = {max_u_history[-1]:.4f}")
        else:
            print(f"  ❌ 數值發散於步數 {step}")

        # 保存結果
        result_file = os.path.join(output_dir, f"max_u_{version}.npy")
        np.save(
            result_file,
            {"max_u_history": max_u_history, "diverged": diverged, "version": version},
        )

    print("\n✅ 測試 3 完成！結果保存至", output_dir)


def test_global_mass_correction(
    steps: int = 50000, res: int = 128, output_dir: str = "test_output_global_mass"
):
    """
    測試 4: 全局質量修正

    對比：
    - 無全局修正
    - 有全局修正（每 100 步）

    場景：Neumann BC（無局部修正）

    預期結果：
    - 無全局修正：質量誤差累積
    - 有全局修正：質量誤差 < 0.01%
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("測試 4: 全局質量修正")
    print("=" * 60)

    for version, apply_global_correction in [
        ("無全局修正", False),
        ("有全局修正", True),
    ]:
        print(f"\n--- {version} ---")

        nx, ny = int(3 * res), res
        solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)

        # 使用無局部修正的 Neumann BC
        bc = BoundaryConditions(solver)
        bc.add_velocity_inlet(0.1, location="left")
        bc.add_neumann_outflow(location="right")
        bc.add_neumann_outflow(location="top")
        bc.add_neumann_outflow(location="bottom")

        if not apply_global_correction:
            bc.mass_correction_strength[None] = 0.0

        # 初始化
        solver.reset()
        solver.apply_boundary_conditions(solver.f)
        solver.apply_boundary_conditions(solver.f_new)

        solver._update_macro(solver.f)
        solver._update_diagnostics()
        solver.initial_mass[None] = solver.total_mass[None]

        # 主迴圈
        mass_errors = []
        check_interval = 1000

        for step in range(1, steps + 1):
            f_src = solver.f if step % 2 == 1 else solver.f_new
            f_dst = solver.f_new if step % 2 == 1 else solver.f

            solver.step(f_src, f_dst)

            # 全局質量修正（每 100 步）
            if apply_global_correction and step % 100 == 0:
                solver.apply_global_mass_correction(f_dst)

            if step % check_interval == 0:
                solver._update_macro(f_dst)
                solver._update_diagnostics()
                ti.sync()

                diag = solver.get_diagnostics()
                mass_error = (
                    abs(diag["total_mass"] - diag["initial_mass"])
                    / diag["initial_mass"]
                )
                mass_errors.append(mass_error)

                print(
                    f"  Step {step:6d}: 質量誤差 = {mass_error:.6f} ({mass_error * 100:.4f}%)"
                )

        # 最終統計
        final_error = mass_errors[-1]
        max_error = max(mass_errors)
        print(f"\n  最終質量誤差: {final_error:.6f} ({final_error * 100:.4f}%)")
        print(f"  最大質量誤差: {max_error:.6f} ({max_error * 100:.4f}%)")

    print("\n✅ 測試 4 完成！結果保存至", output_dir)


def main():
    parser = argparse.ArgumentParser(description="邊界條件改進驗證測試")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4"],
        help="測試編號：1=Neumann質量, 2=角點, 3=Sponge, 4=全局質量, all=全部",
    )
    parser.add_argument("--steps", type=int, default=50000, help="模擬步數")
    parser.add_argument("--res", type=int, default=128, help="解析度")

    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    print("\n" + "=" * 60)
    print("邊界條件改進驗證測試")
    print("=" * 60)

    if args.test in ["all", "1"]:
        test_neumann_mass_conservation(steps=args.steps, res=args.res)

    if args.test in ["all", "2"]:
        test_corner_handling(res=64, steps=10000)

    if args.test in ["all", "3"]:
        test_sponge_layer(res=args.res, re=5000, steps=20000)

    if args.test in ["all", "4"]:
        test_global_mass_correction(steps=args.steps, res=args.res)

    print("\n" + "=" * 60)
    print("✅ 所有測試完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
