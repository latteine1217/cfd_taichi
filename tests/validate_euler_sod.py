"""
驗證：EulerSolver (2D nj=1) vs Sod 精確解
==========================================

What: 用 2D EulerSolver 以 nj=1 模擬 1D Sod 激波管，
      對比 MVU-1 (sod_shock_tube.py) 的結果與精確解
Why:  確保 2D FV 核心（HLLC + MUSCL + SSP-RK3）
      在退化為 1D 時與已驗證的 1D kernel 等價

執行：
    uv run python tests/validate_euler_sod.py
"""

import taichi as ti
import numpy as np
import time

# ── 精確解（從 MVU-1 複製，不依賴 examples 路徑）────────────────
def sod_exact(x_arr, t):
    from scipy.optimize import brentq
    g = 1.4
    rho1, u1, p1 = 1.0,   0.0, 1.0
    rho4, u4, p4 = 0.125, 0.0, 0.1
    c1 = np.sqrt(g * p1 / rho1);  c4 = np.sqrt(g * p4 / rho4)

    def f_eq(ps):
        uL = u1 - 2*c1/(g-1) * ((ps/p1)**((g-1)/(2*g)) - 1)
        A = 2/((g+1)*rho4);  B = (g-1)/(g+1)*p4
        uR = u4 + (ps - p4)*np.sqrt(A/(ps + B))
        return uL - uR

    p_star  = brentq(f_eq, 1e-10, 100.0, xtol=1e-12)
    A = 2/((g+1)*rho4);  B = (g-1)/(g+1)*p4
    u_star  = u4 + (p_star - p4)*np.sqrt(A/(p_star + B))
    cL_star = c1*(p_star/p1)**((g-1)/(2*g))
    rho_sL  = rho1*(p_star/p1)**(1.0/g)
    rho_sR  = rho4*(p_star + (g-1)/(g+1)*p4) / ((g-1)/(g+1)*p_star + p4)
    v_sh    = u4 + c4*np.sqrt((g+1)/(2*g)*p_star/p4 + (g-1)/(2*g))
    v_tail  = u1 - c1;  x0 = 0.5

    rho_e = np.empty_like(x_arr);  u_e = np.empty_like(x_arr);  p_e = np.empty_like(x_arr)
    for k, x in enumerate(x_arr):
        xi = (x - x0) / t
        if   xi < v_tail:
            rho_e[k], u_e[k], p_e[k] = rho1, u1, p1
        elif xi < u_star - cL_star:
            u_loc = 2/(g+1)*(c1 + (g-1)/2*u1 + xi)
            c_loc = c1 + (g-1)/2*(u1 - u_loc)
            rho_e[k] = rho1*(c_loc/c1)**(2/(g-1))
            u_e[k]   = u_loc
            p_e[k]   = p1*(c_loc/c1)**(2*g/(g-1))
        elif xi < u_star:
            rho_e[k], u_e[k], p_e[k] = rho_sL, u_star, p_star
        elif xi < v_sh:
            rho_e[k], u_e[k], p_e[k] = rho_sR, u_star, p_star
        else:
            rho_e[k], u_e[k], p_e[k] = rho4, u4, p4
    return rho_e, u_e, p_e


def run_validation(ni=400, t_end=0.2, cfl=0.45):
    from fvm_taichi import EulerSolver

    dx = 1.0 / ni
    dy = dx          # nj=1：dy=dx，j 方向不貢獻物理
    NG = EulerSolver.NG

    solver = EulerSolver(ni=ni, nj=1, gamma=1.4, cfl=cfl)
    solver.set_cartesian_grid(dx=dx, dy=dy)

    # ── Sod 初始條件（自訂 kernel，因為 init_uniform 只能單一狀態）
    @ti.kernel
    def init_sod(s: ti.template()):
        for i, j in s.U:
            x = (i - NG) * dx + 0.5 * dx
            rho = 0.0;  u = 0.0;  p = 0.0
            if x < 0.5:
                rho, u, p = 1.0,   0.0, 1.0
            else:
                rho, u, p = 0.125, 0.0, 0.1
            E = p / (s.gamma - 1.0) + 0.5 * rho * u * u
            s.W[i, j] = ti.Vector([rho, u, 0.0, p])
            s.U[i, j] = ti.Vector([rho, rho*u, 0.0, E])

    init_sod(solver)
    solver._update_ghost()

    # ── 主迴圈
    t = 0.0;  step = 0;  t0 = time.time()
    while t < t_end:
        dt = solver.step()
        t += dt;  step += 1

    elapsed = time.time() - t0
    ti.sync()

    # ── 取結果
    rho_n, u_n, _, p_n = solver.get_primitive()
    rho_n = rho_n[:, 0]   # nj=1 → 取第 0 列
    u_n   = u_n  [:, 0]
    p_n   = p_n  [:, 0]

    x_int = np.array([(i + 0.5) * dx for i in range(ni)])
    rho_ex, u_ex, p_ex = sod_exact(x_int, t_end)

    l1_rho = float(np.mean(np.abs(rho_n - rho_ex)))
    l1_u   = float(np.mean(np.abs(u_n   - u_ex)))
    l1_p   = float(np.mean(np.abs(p_n   - p_ex)))

    # ── 結果報告
    print("=" * 58)
    print("  EulerSolver 1D-Sod 驗證（nj=1 退化測試）")
    print("=" * 58)
    print(f"  ni={ni}  t_end={t_end}  CFL={cfl}")
    print(f"  steps={step}  elapsed={elapsed:.2f}s")
    print(f"  ρ_min={rho_n.min():.5f}  p_min={p_n.min():.5f}")
    print(f"\n  L1 error vs exact:")
    print(f"    ρ = {l1_rho:.4e}")
    print(f"    u = {l1_u:.4e}")
    print(f"    p = {l1_p:.4e}")

    # MVU-1 參考值（同 nx=400, CFL=0.45）
    ref = dict(rho=2.4e-3, u=4.0e-3, p=1.6e-3)
    tol = 1.1   # 允許 10% 偏差（float32 vs float64 精確解）

    ok = (l1_rho < ref['rho'] * tol and
          l1_u   < ref['u']   * tol and
          l1_p   < ref['p']   * tol)

    print(f"\n  參考（MVU-1 nx=400）: ρ≈{ref['rho']:.1e}  u≈{ref['u']:.1e}  p≈{ref['p']:.1e}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  (容許 10% 偏差)")
    print("=" * 58)
    return ok, l1_rho, l1_u, l1_p


if __name__ == "__main__":
    ti.init(arch=ti.metal, default_fp=ti.f32)
    ok, *_ = run_validation(ni=400, t_end=0.2, cfl=0.45)
    sys.exit(0 if ok else 1)
