"""
MVU-5: NACA0012 AOA 掃描 + 穩態 CL 驗證
=========================================

What: 在 Ma=0.3 下對 AOA=0,2,4,6,8° 各跑至 CL 收斂，
      與 Prandtl-Glauert 薄翼理論比較

Why:  驗證特徵遠場 BC 在多種攻角下均能產生物理正確的升力曲線；
      同時確認 Euler 流（無黏）在穩態時 CD 趨近於 0（D'Alembert 悖論）

驗收標準：
  - CL(AOA=0°) < 0.05  （對稱翼型在 0° 攻角無升力）
  - CL 對 AOA 單調遞增
  - CL / CL_PG ∈ [0.60, 1.20]  （Prandtl-Glauert 理論的 ±40%，粗網格容許）

執行：
    uv run python tests/validate_euler_naca_sweep.py [--ni 120] [--nj 40]
"""

import taichi as ti
import numpy as np
import argparse
import time


# ── CL/CD 壓力積分（精簡版，只需 CL/CD 不需 Cp）────────────────────────────
def get_cl_cd(solver, alpha_rad: float, rho_inf: float, v_mag: float):
    """
    壓力積分計算 CL, CD

    F_airfoil = -∮ p · S_j_wall    (S_j 指向流體 +j 方向)
    L = F_y cosα - F_x sinα,  D = F_x cosα + F_y sinα
    CL = L/q∞,  CD = D/q∞,  q∞ = ½ρ∞V∞²
    """
    from fvm_taichi import EulerSolver
    NG = EulerSolver.NG
    ni = solver.ni

    _, _, _, p_n = solver.get_primitive()   # (ni, nj)
    p_wall = p_n[:, 0]                       # j=0 interior → first cell layer

    S_j_np   = solver.S_j.to_numpy()                    # (NI, NJ, 2)
    S_j_wall = S_j_np[NG:NG + ni, NG - 1, :]            # (ni, 2)

    F_x = float(-np.sum(p_wall * S_j_wall[:, 0]))
    F_y = float(-np.sum(p_wall * S_j_wall[:, 1]))

    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    L =  F_y * cos_a - F_x * sin_a
    D =  F_x * cos_a + F_y * sin_a

    q_inf = 0.5 * rho_inf * v_mag**2
    return L / q_inf, D / q_inf


# ── 單一攻角穩態模擬（跑至 CL 收斂）────────────────────────────────────────
def run_to_convergence(
        ni: int, nj: int, ma: float, aoa: float,
        max_steps: int = 15000,
        check_every: int = 500,
        cl_tol: float = 1e-3,
        r_far: float = 15.0,
):
    """
    NACA0012 Euler 模擬，跑至 CL 收斂或達到最大步數

    收斂準則：|CL_new - CL_prev| < cl_tol（每 check_every 步檢查一次）

    Returns:
        (CL, CD, steps, converged, res_norm_final)
    """
    from fvm_taichi import EulerSolver, generate_o_grid_naca0012

    gamma     = 1.4
    alpha_rad = np.deg2rad(aoa)
    rho_inf   = 1.0
    u_inf     = ma * np.cos(alpha_rad)
    v_inf     = ma * np.sin(alpha_rad)
    p_inf     = rho_inf / gamma

    x_node, y_node = generate_o_grid_naca0012(
        ni=ni, nj=nj, R_far=r_far, NG=EulerSolver.NG)

    solver = EulerSolver(ni=ni, nj=nj, gamma=gamma, cfl=0.45)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_slip_wall_j_min()
    solver.set_far_field_j_max(rho_inf, u_inf, v_inf, p_inf)
    solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)

    CL_prev   = 0.0
    converged = False
    res_final = float('nan')

    for step in range(1, max_steps + 1):
        solver.step()

        if step % check_every == 0:
            ti.sync()
            CL, CD = get_cl_cd(solver, alpha_rad, rho_inf, ma)
            res_final = solver.get_residual_norm()

            # 收斂判斷：CL 絕對變化量 < tol（對 AOA=0° 時 CL≈0 也適用）
            if step > check_every and abs(CL - CL_prev) < cl_tol:
                converged = True
                break

            CL_prev = CL

    ti.sync()
    CL, CD = get_cl_cd(solver, alpha_rad, rho_inf, ma)
    res_final = solver.get_residual_norm()
    return CL, CD, step, converged, res_final


# ── AOA 掃描主程序 ────────────────────────────────────────────────────────
def run_sweep(ni: int = 120, nj: int = 40, ma: float = 0.3,
              aoa_list=None, max_steps: int = 15000,
              output_dir: str = None):
    """
    多攻角掃描，每個 AOA 各跑至 CL 收斂

    驗收：
      - CL(0°) < 0.05
      - CL 單調遞增
      - CL / CL_PG ∈ [0.60, 1.20] for AOA ≥ 2°
    """
    if aoa_list is None:
        aoa_list = [0.0, 2.0, 4.0, 6.0, 8.0]

    gamma  = 1.4
    beta   = np.sqrt(max(1.0 - ma**2, 0.01))   # Prandtl-Glauert 修正因子

    print("=" * 76)
    print("  MVU-5: NACA0012 AOA 掃描 + 穩態 CL 驗證")
    print(f"  Grid: ni={ni} × nj={nj},  Ma={ma:.2f},  max_steps={max_steps}")
    print("=" * 76)
    print(f"  {'AOA':>5}  {'Steps':>7}  {'Conv':>5}  "
          f"{'CL_calc':>9}  {'CL_PG':>8}  {'CL/PG':>7}  "
          f"{'CD_calc':>9}  {'res':>10}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*5}  "
          f"{'-'*9}  {'-'*8}  {'-'*7}  "
          f"{'-'*9}  {'-'*10}")

    results = []
    t_start = time.time()

    for aoa in aoa_list:
        alpha = np.deg2rad(aoa)
        CL_pg = (2.0 * np.pi * np.sin(alpha)) / beta   # Prandtl-Glauert

        t0 = time.time()
        CL, CD, steps, conv, res = run_to_convergence(
            ni=ni, nj=nj, ma=ma, aoa=aoa, max_steps=max_steps)
        elapsed = time.time() - t0

        ratio_str = (f"{CL / CL_pg:7.3f}" if abs(CL_pg) > 1e-4
                     else "    ---")
        conv_str  = "Yes" if conv else "No "

        print(f"  {aoa:>5.1f}  {steps:>7}  {conv_str:>5}  "
              f"{CL:>9.4f}  {CL_pg:>8.4f}  {ratio_str:>7}  "
              f"{CD:>9.5f}  {res:>10.3e}  ({elapsed:.1f}s)")

        results.append({
            'aoa': aoa, 'CL': CL, 'CD': CD,
            'CL_pg': CL_pg, 'steps': steps, 'converged': conv,
        })

    total = time.time() - t_start
    print(f"\n  總耗時: {total:.1f}s")
    print()

    # ── 驗收邏輯 ─────────────────────────────────────────────────────
    ok = True
    failures = []

    # 1. CL(0°) 接近 0
    cl_zero = results[0]['CL']
    if abs(cl_zero) >= 0.05:
        ok = False
        failures.append(f"CL(0°) = {cl_zero:.4f} >= 0.05")

    # 2. CL 單調遞增（對有限差分容許 5% 翻轉）
    for k in range(1, len(results)):
        if (results[k]['aoa'] > 0.5 and
                results[k]['CL'] < results[k - 1]['CL'] - 0.02):
            ok = False
            failures.append(
                f"CL not monotone: AOA={results[k]['aoa']}° "
                f"CL={results[k]['CL']:.4f} < {results[k-1]['CL']:.4f}")

    # 3. CL / CL_PG ∈ [0.60, 1.20] for AOA ≥ 2°
    for r in results:
        if r['aoa'] >= 2.0 and abs(r['CL_pg']) > 1e-4:
            ratio = r['CL'] / r['CL_pg']
            if not (0.60 <= ratio <= 1.20):
                ok = False
                failures.append(
                    f"CL/CL_PG = {ratio:.3f} 超出 [0.60, 1.20] "
                    f"at AOA={r['aoa']}°")

    # ── 結論 ─────────────────────────────────────────────────────────
    print("=" * 76)
    if ok:
        print("  ✅ MVU-5 PASS")
        print("    - CL(0°) ≈ 0  ✓")
        print("    - CL 單調遞增  ✓")
        print("    - CL/CL_PG ∈ [0.60, 1.20]  ✓")
    else:
        print("  ❌ MVU-5 FAIL")
        for f in failures:
            print(f"    - {f}")
    print("=" * 76)

    # ── CL vs AOA 圖 ──────────────────────────────────────────────
    if output_dir is not None:
        import os
        from fvm_taichi.plot import plot_cl_aoa
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"cl_aoa_ma{int(ma*100):03d}.png")
        plot_cl_aoa(results, ma, save_path=path)

    return ok, results


def main():
    parser = argparse.ArgumentParser(description="MVU-5: NACA0012 AOA Sweep")
    parser.add_argument('--ni',     type=int,   default=120)
    parser.add_argument('--nj',     type=int,   default=40)
    parser.add_argument('--ma',     type=float, default=0.3)
    parser.add_argument('--steps',  type=int,   default=15000)
    parser.add_argument('--output', type=str,   default='output_naca0012',
                        help='輸出圖片目錄（空字串停用）')
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    ok, _ = run_sweep(
        ni=args.ni, nj=args.nj, ma=args.ma, max_steps=args.steps,
        output_dir=args.output if args.output else None)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
