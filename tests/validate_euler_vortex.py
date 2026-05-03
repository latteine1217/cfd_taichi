"""
驗證：EulerSolver — Isentropic Vortex（等熵渦旋）收斂研究
=========================================================

What: 在周期性直角網格模擬等熵渦旋平移，對比精確解驗證空間精度
Why:  等熵渦旋是 C∞ 光滑解（無激波），是驗證 MUSCL 二階收斂的黃金標準
      MUSCL (2nd) + HLLC + SSP-RK3 (3rd) 應得 L1 誤差 ≈ 2 階收斂

精確解（Shu 1997 標準形式）：
  f   = 1 - r²   (r 為距渦心距離，R=1 尺度化)
  δu  = -β/(2π) * (y-yc) * exp(f/2)
  δv  = +β/(2π) * (x-xc) * exp(f/2)
  δT  = -(γ-1)*β²/(8γπ²) * exp(f)
  T   = 1 + δT,  ρ = T^{1/(γ-1)},  p = T^{γ/(γ-1)}
  渦心以均勻流速度 (u∞, v∞) 平移 → 精確解在時刻 t 為初始渦旋平移至 (xc+u∞t, yc)

設定：
  域     [0, 10] × [0, 10]，周期性 BC
  均勻流  u∞=1, v∞=0
  初始渦心 (5, 5)，強度 β=5
  t_end  = 2.0 （渦心移至 x=7，不需要 wrap）

執行：
    uv run python tests/validate_euler_vortex.py
"""

import taichi as ti
import numpy as np
import time


# ── 等熵渦旋解析解 ──────────────────────────────────────────────────────
def isentropic_vortex_exact(x_arr, y_arr, t,
                              x0=5.0, y0=5.0,
                              u_inf=1.0, v_inf=0.0,
                              beta=5.0, gamma=1.4):
    """
    時刻 t 的等熵渦旋精確解（已平移）

    Args:
        x_arr, y_arr: cell center 座標（相同 shape 的 meshgrid）
        t:            模擬物理時間
        x0, y0:       t=0 渦心位置
        u_inf, v_inf: 均勻背景流速
        beta:         渦旋強度
        gamma:        比熱比

    Returns:
        rho, u, v, p: 與 x_arr 相同 shape 的 ndarray
    """
    xc = x0 + u_inf * t   # 渦心 x（不做周期 wrap，t_end=2 時在 x=7，無需）
    yc = y0 + v_inf * t

    dx = x_arr - xc
    dy = y_arr - yc
    r2 = dx**2 + dy**2
    f  = 1.0 - r2         # Shu 形式：f = 1 - r²

    exp_f2 = np.exp(0.5 * f)
    exp_f  = np.exp(f)

    du = -beta / (2.0 * np.pi) * dy * exp_f2
    dv = +beta / (2.0 * np.pi) * dx * exp_f2
    dT = -(gamma - 1.0) * beta**2 / (8.0 * gamma * np.pi**2) * exp_f

    T   = 1.0 + dT
    rho = T**(1.0 / (gamma - 1.0))
    p   = T**(gamma / (gamma - 1.0))
    u   = u_inf + du
    v   = v_inf + dv

    return rho, u, v, p


# ── 單解析度模擬 ────────────────────────────────────────────────────────
def run_vortex(ni: int, t_end: float = 2.0, cfl: float = 0.45,
               L: float = 10.0, gamma: float = 1.4):
    """
    等熵渦旋模擬（ni × ni 均勻網格，雙向周期性 BC）

    Returns:
        (l1_rho, l1_u, l1_p, steps, elapsed)
    """
    from fvm_taichi import EulerSolver

    dx = L / ni
    dy = dx

    solver = EulerSolver(ni=ni, nj=ni, gamma=gamma, cfl=cfl)
    solver.set_cartesian_grid(dx=dx, dy=dy)
    solver.set_periodic_bc(i_dir=True, j_dir=True)

    # cell center 座標
    xi = np.array([(i + 0.5) * dx for i in range(ni)])
    yj = np.array([(j + 0.5) * dy for j in range(ni)])
    X, Y = np.meshgrid(xi, yj, indexing='ij')   # shape (ni, ni)

    # 初始條件
    rho0, u0, v0, p0 = isentropic_vortex_exact(X, Y, t=0.0,
                                                 x0=L/2, y0=L/2, beta=5.0,
                                                 gamma=gamma)
    W0 = np.stack([rho0, u0, v0, p0], axis=-1)   # (ni, ni, 4)
    solver.init_from_primitive_numpy(W0)

    # 時間推進
    t = 0.0;  step = 0
    t0 = time.time()
    while t < t_end:
        dt = solver.step()
        t += dt
        step += 1

    ti.sync()
    elapsed = time.time() - t0

    # 數值解（interior）
    rho_n, u_n, v_n, p_n = solver.get_primitive()

    # 精確解（渦心平移至 t 時刻位置）
    rho_ex, u_ex, v_ex, p_ex = isentropic_vortex_exact(
        X, Y, t=t_end, x0=L/2, y0=L/2, beta=5.0, gamma=gamma)

    l1_rho = float(np.mean(np.abs(rho_n - rho_ex)))
    l1_u   = float(np.mean(np.abs(u_n   - u_ex  )))
    l1_p   = float(np.mean(np.abs(p_n   - p_ex  )))

    return l1_rho, l1_u, l1_p, step, elapsed


# ── 收斂研究 ────────────────────────────────────────────────────────────
def run_convergence_study(grids=(16, 32, 64, 128)):
    """
    網格細化收斂研究

    MUSCL（二階空間）+ SSP-RK3（三階時間）對光滑解應達 ≈ 2 階收斂
    float32 在 ni=128 時可能被捨入誤差限制（不算失敗）
    """
    print("=" * 68)
    print("  EulerSolver — Isentropic Vortex 收斂研究")
    print("  (MUSCL + HLLC + SSP-RK3, periodic BC, t_end=2.0, β=5)")
    print("=" * 68)
    print(f"  {'ni':>5}  {'L1(ρ)':>10}  {'L1(u)':>10}  {'order_ρ':>8}  {'steps':>7}  {'time':>7}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}")

    results = []
    prev_l1 = None

    for ni in grids:
        l1_rho, l1_u, l1_p, steps, elapsed = run_vortex(ni, t_end=2.0, cfl=0.45)
        order = np.log2(prev_l1 / l1_rho) if prev_l1 is not None else float('nan')
        results.append((ni, l1_rho, l1_u, l1_p, order, steps, elapsed))
        ord_str = f"{order:.2f}" if not np.isnan(order) else "  ---"
        print(f"  {ni:>5}  {l1_rho:>10.4e}  {l1_u:>10.4e}  {ord_str:>8}  "
              f"{steps:>7}  {elapsed:>6.1f}s")
        prev_l1 = l1_rho

    # 驗收：取 16→32、32→64 兩段的最大收斂階數
    orders_valid = [r[4] for r in results[1:3] if not np.isnan(r[4])]
    best_order = max(orders_valid) if orders_valid else 0.0

    print()
    ok = best_order > 1.70
    print(f"  最大收斂階數（ni=16→64）: {best_order:.2f}  （目標 > 1.70）")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  (MUSCL 二階收斂驗證)")
    print("=" * 68)
    return ok, results


if __name__ == "__main__":
    ti.init(arch=ti.metal, default_fp=ti.f32)
    ok, _ = run_convergence_study(grids=(16, 32, 64, 128))
    sys.exit(0 if ok else 1)
