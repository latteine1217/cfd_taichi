"""
NACA0012 Euler Flow — O-grid + 特徵遠場 BC + CL/CD
===================================================

What: NACA0012 翼型無黏（Euler）流動，代數 O-grid，支援任意攻角
Why:  - 特徵遠場 BC：減少遠場反射，允許 AOA≠0 的非對稱來流
     - CL/CD 計算：壓力積分（D'Alembert 悖論：Euler 流無波阻時 CD≈0）
     - 薄翼理論驗算：CL ≈ 2π sinα（Prandtl-Glauert 壓縮修正）

物理設定：
  Ma = 0.3（亞音速無黏）
  AOA = 0° / 5° / 10°（依參數）
  γ = 1.4
  ρ∞=1, c∞=1（→ p∞=1/γ, u∞=Ma·cosα, v∞=Ma·sinα）

BC：
  j_min（翼型）：slip wall
  j_max（遠場）：特徵 Riemann invariant BC
  i（O-grid）  ：periodic

執行：
    uv run python examples/naca0012_euler.py [--ni 180] [--nj 60]
                                              [--ma 0.3] [--aoa 5]
                                              [--steps 10000]
"""

import os
import sys

import taichi as ti
import numpy as np
import argparse
import time

from cfd_taichi import (
    BoundaryConditionDescriptor,
    CaseRunner,
    CurvilinearGrid2D,
    SolverControlDescriptor,
)


def compute_reference_scales(
        ma: float,
        alpha_rad: float,
        rho_inf: float,
        gamma: float,
        chord: float = 1.0) -> dict:
    """
    回報此 Euler case 的參考尺度與無黏 Reynolds 數

    What: 依自由流設定整理 V∞、a∞、q∞、弦長 c 與 Re
    Why:  這個案例是無黏 Euler 方程，黏性係數 μ = 0，
          因此 Reynolds 數不是使用者輸入，而是理論上趨近無限大；
          明確列印可避免把 Ma-only 設定誤解成缺漏參數
    When: 啟動案例時輸出模擬參考參數

    Args:
        ma:        自由流 Mach number
        alpha_rad: 攻角（弧度）
        rho_inf:   自由流密度
        gamma:     比熱比
        chord:     特徵長度；NACA0012 幾何已做 chord=1 無量綱化

    Returns:
        dict: 結構化參考尺度資訊
    """
    u_inf = ma * np.cos(alpha_rad)
    v_inf = ma * np.sin(alpha_rad)
    v_mag = float(np.sqrt(u_inf**2 + v_inf**2))
    a_inf = float(np.sqrt(gamma * (rho_inf / gamma) / rho_inf))
    q_inf = 0.5 * rho_inf * v_mag**2

    return {
        'u_inf': float(u_inf),
        'v_inf': float(v_inf),
        'v_mag': v_mag,
        'a_inf': a_inf,
        'q_inf': q_inf,
        'chord': float(chord),
        'mu_inf': 0.0,
        're': float('inf'),
    }


def compute_ramped_ma(
        step: int,
        ma_target: float,
        ma_start: float | None,
        ma_ramp_steps: int) -> float:
    """
    計算當前步數應使用的自由流 Mach（線性 ramp）

    What: 在前 ma_ramp_steps 內把 Ma 從 ma_start 線性拉到 ma_target
    Why:  transonic case 直接高 Ma 啟動容易引入過大初始暫態；
          逐步升高遠場來流可降低非物理解風險
    """
    if ma_start is None or ma_ramp_steps <= 0:
        return float(ma_target)
    ratio = min(max(step, 0), ma_ramp_steps) / float(ma_ramp_steps)
    return float(ma_start + ratio * (ma_target - ma_start))


# ── CL/CD 壓力積分 ───────────────────────────────────────────────────────
def compute_cl_cd(solver, alpha_rad: float,
                  rho_inf: float, v_mag: float) -> tuple:
    """
    從壓力場計算 CL, CD（無黏，壓力積分）

    F_airfoil = -∮ p · S_j_wall   (S_j 指向 +j = 從翼型表面指向流體)
    L = F · (-sinα, cosα)          (升力：垂直於來流)
    D = F · ( cosα, sinα)          (阻力：平行於來流)
    CL = L / q∞,  CD = D / q∞,  q∞ = ½ρ∞V∞²

    Args:
        solver:     EulerSolver instance
        alpha_rad:  攻角（弧度）
        rho_inf:    自由流密度
        v_mag:      自由流速度大小

    Returns:
        (CL, CD, Cp_array, x_cc, y_cc)
    """
    from fvm_taichi import EulerSolver
    NG = EulerSolver.NG
    ni = solver.ni

    rho_n, u_n, v_n, p_n = solver.get_primitive()   # (ni, nj)

    # 翼面壓力（取第一層 interior cell，j=0 of interior = index NG in full array）
    p_wall = p_n[:, 0]   # shape (ni,)

    # 翼面 j-face 向量 S_j[i, NG-1] 指向流體（+j 方向）
    S_j_np   = solver.S_j.to_numpy()              # (NI, NJ, 2)
    S_j_wall = S_j_np[NG:NG + ni, NG - 1, :]     # (ni, 2)

    # F_airfoil：壓力由流體推向翼型，方向 = -S_j（法向內指）
    F_x = float(-np.sum(p_wall * S_j_wall[:, 0]))
    F_y = float(-np.sum(p_wall * S_j_wall[:, 1]))

    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    L =  F_y * cos_a - F_x * sin_a   # 升力
    D =  F_x * cos_a + F_y * sin_a   # 阻力

    q_inf = 0.5 * rho_inf * v_mag**2
    CL = L / q_inf
    CD = D / q_inf

    # Cp 分佈（用於後處理）
    p_inf = rho_inf / solver.gamma  # c∞=1 → p∞ = ρ/γ
    Cp = (p_wall - p_inf) / q_inf

    # Cell center 座標（翼面上）
    x_node_np = solver.S_i.to_numpy()  # 借用讀取，但改用另外傳入
    # 直接從 S_j_wall 的 y 分量判斷上下表面（快速近似）

    return CL, CD, Cp, F_x, F_y


def get_airfoil_cc(x_node, y_node, ni, NG=2):
    """取翼面 cell center 座標（用於 Cp 圖）"""
    # cell center i = NG..NG+ni-1 對應 x_node NG..NG+ni 取平均
    x_cc = 0.5 * (x_node[NG:NG + ni, NG] + x_node[NG + 1:NG + ni + 1, NG])
    y_cc = 0.5 * (y_node[NG:NG + ni, NG] + y_node[NG + 1:NG + ni + 1, NG])
    return x_cc, y_cc


def _initialize_naca0012_case(
        _runner: CaseRunner,
        solver,
        _bc_handle,
        *,
        rho_inf: float,
        u_inf: float,
        v_inf: float,
        p_inf: float):
    """
    NACA0012 Euler 的正式初始化流程。

    What:
    - 以均勻自由流建立 O-grid 初始場

    Why:
    - characteristic far-field、periodic 與 slip wall 應由 descriptor 控制；
      initializer 只保留物理初始狀態
    """
    solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)


def _build_naca0012_solver_controls(
        *,
        rho_inf: float,
        u_inf: float,
        v_inf: float,
        p_inf: float) -> list[SolverControlDescriptor]:
    """
    建立 NACA0012 Euler 的正式 solver control descriptors。
    """
    return [
        SolverControlDescriptor.far_field(
            location="j_max",
            rho=rho_inf,
            u=u_inf,
            v=v_inf,
            p=p_inf,
        )
    ]


def _naca0012_diagnostics_hook(
        runner: CaseRunner,
        solver,
        diagnostics: dict,
        *,
        alpha_rad: float,
        rho_inf: float,
        v_mag: float) -> dict:
    """
    派生 NACA0012 Euler 專屬 diagnostics。
    """
    cl, cd, _cp, _fx, _fy = compute_cl_cd(solver, alpha_rad, rho_inf, v_mag)
    return {
        "lift_coefficient": float(cl),
        "drag_coefficient": float(cd),
        "drag_coefficient_abs": float(abs(cd)),
    }


def build_naca0012_euler_runner(
        *,
        ni: int = 120,
        nj: int = 40,
        ma: float = 0.3,
        aoa: float = 5.0,
        r_far: float = 12.0,
        cfl: float = 0.45) -> tuple[CaseRunner, dict]:
    """
    建立 NACA0012 Euler 的 CaseRunner。

    What:
    - 組裝 O-grid、periodic/slip wall BC 與 far-field characteristic control

    Why:
    - 外流翼型案例能否接入 registry，是 solver toolkit 是否能支撐 external aero workflow 的關鍵驗證
    """
    from fvm_taichi import EulerSolver, generate_o_grid_naca0012

    if not (0.0 < ma < 1.0):
        raise ValueError(
            f"This builder uses a subsonic characteristic far-field BC; got Ma={ma}. "
            "Please use 0 < Ma < 1."
        )
    gamma = 1.4
    alpha_rad = float(np.deg2rad(aoa))
    rho_inf = 1.0
    u_inf = float(ma * np.cos(alpha_rad))
    v_inf = float(ma * np.sin(alpha_rad))
    p_inf = float(rho_inf / gamma)
    ref = compute_reference_scales(ma, alpha_rad, rho_inf, gamma)

    x_node, y_node = generate_o_grid_naca0012(
        ni=ni,
        nj=nj,
        R_far=r_far,
        NG=EulerSolver.NG,
    )

    runner = CaseRunner(
        name="naca0012_euler",
        method="fvm",
        equation="euler",
        regime="compressible",
        grid=CurvilinearGrid2D(x_node=x_node, y_node=y_node, ng=EulerSolver.NG),
        solver_kwargs={
            "gamma": gamma,
            "cfl": cfl,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.free_slip("bottom"),
        ],
        solver_controls=_build_naca0012_solver_controls(
            rho_inf=rho_inf,
            u_inf=u_inf,
            v_inf=v_inf,
            p_inf=p_inf,
        ),
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_naca0012_case(
            runner_obj,
            solver_obj,
            bc_handle,
            rho_inf=rho_inf,
            u_inf=u_inf,
            v_inf=v_inf,
            p_inf=p_inf,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _naca0012_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            alpha_rad=alpha_rad,
            rho_inf=rho_inf,
            v_mag=ref["v_mag"],
        ),
    )
    return runner, {
        "alpha_rad": alpha_rad,
        "rho_inf": rho_inf,
        "u_inf": u_inf,
        "v_inf": v_inf,
        "p_inf": p_inf,
        "ref": ref,
    }


# ── 主模擬函數 ────────────────────────────────────────────────────────────
def run_naca0012(
        ni:         int   = 180,
        nj:         int   = 60,
        ma:         float = 0.3,
        aoa:        float = 0.0,   # 攻角（度）
        ma_start:   float | None = None,
        ma_ramp_steps: int = 0,
        steps:      int   = 8000,
        R_far:      float = 15.0,
        cfl:        float = 0.45,
        report:     int   = 1000,
        tol:        float = 1e-5,
        output_dir: str   = None,  # 非 None 時自動輸出圖片
        gif_every:  int   = 0,     # >0 時每隔此步數存一幀，最後合成 GIF
        gif_field:  str   = 'velocity',
):
    """
    NACA0012 Euler 模擬（可壓縮無黏，特徵遠場 BC）

    Returns:
        (CL, CD) 或 None（若發生非物理解）
    """
    from cfd_taichi.configuration import apply_solver_control_descriptors
    from fvm_taichi import EulerSolver, generate_o_grid_naca0012

    gamma     = 1.4
    alpha_rad = np.deg2rad(aoa)

    if not (0.0 < ma < 1.0):
        raise ValueError(
            f"This case uses a subsonic characteristic far-field BC; got Ma={ma}. "
            "Please use 0 < Ma < 1."
        )
    if ma_start is not None:
        if not (0.0 < ma_start < 1.0):
            raise ValueError(f"ma_start must satisfy 0 < ma_start < 1, got {ma_start}")
        if ma_start > ma:
            raise ValueError(
                f"ma_start={ma_start} must be <= target ma={ma} for monotonic ramp-up."
            )
    if ma_ramp_steps < 0:
        raise ValueError(f"ma_ramp_steps must be >= 0, got {ma_ramp_steps}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    # 無量綱化：ρ∞=1, c∞=1 → p∞=ρ/γ=1/γ, Ma=u∞/c∞=u∞
    rho_inf = 1.0
    ma_init = compute_ramped_ma(0, ma, ma_start, ma_ramp_steps)
    u_inf   = ma_init * np.cos(alpha_rad)
    v_inf   = ma_init * np.sin(alpha_rad)
    p_inf   = rho_inf / gamma          # c² = γp/ρ = 1
    ref     = compute_reference_scales(ma_init, alpha_rad, rho_inf, gamma)

    print("=" * 64)
    print("  NACA0012 Euler Flow (O-grid + Characteristic Far-Field BC)")
    print("=" * 64)
    print(f"  Grid:   ni={ni} × nj={nj},  R_far={R_far}")
    if ma_start is None or ma_ramp_steps <= 0:
        print(f"  Ma:     {ma:.3f},  AOA: {aoa:.1f}°,  p∞: {p_inf:.5f}")
    else:
        print(f"  Ma:     {ma_start:.3f} -> {ma:.3f} (ramp {ma_ramp_steps} steps),  AOA: {aoa:.1f}°,  p∞: {p_inf:.5f}")
    print(f"  V∞:     ({ref['u_inf']:.5f}, {ref['v_inf']:.5f}), |V∞|={ref['v_mag']:.5f}")
    print(f"  a∞:     {ref['a_inf']:.5f},  q∞: {ref['q_inf']:.5f},  chord={ref['chord']:.3f}")
    print("  Re:     ∞ (Euler inviscid limit, μ=0; chord-based Reynolds number)")
    print(f"  Steps:  {steps},  CFL: {cfl},  Residual tol: {tol:.1e}")

    # ── 網格生成 ─────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1] Generating O-grid...", end=" ", flush=True)
    x_node, y_node = generate_o_grid_naca0012(
        ni=ni, nj=nj, R_far=R_far, NG=EulerSolver.NG)
    print(f"done  ({time.time()-t0:.2f}s)")

    # ── 求解器初始化 ──────────────────────────────────────────────
    print("[2] Initializing solver...", end=" ", flush=True)
    solver = EulerSolver(ni=ni, nj=nj, gamma=gamma, cfl=cfl)
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_slip_wall_j_min()
    apply_solver_control_descriptors(
        solver,
        _build_naca0012_solver_controls(
            rho_inf=rho_inf,
            u_inf=u_inf,
            v_inf=v_inf,
            p_inf=p_inf,
        ),
    )

    # 均勻來流初始化
    solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)
    print(f"done  ({time.time()-t0:.2f}s)")

    # ── 主迴圈 ────────────────────────────────────────────────────
    print(f"\n[3] Running {steps} steps...")
    print(f"  {'step':>6}  {'Ma':>7}  {'dt':>10}  {'res':>10}  {'CL':>8}  {'CD':>8}  "
          f"{'p_min':>8}  {'p_max':>8}  {'t(s)':>7}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}")

    sim_start  = time.time()
    CL_history = []
    CD_history = []
    converged = False
    final_step = steps
    final_residual = np.nan

    for step in range(1, steps + 1):
        ma_step = compute_ramped_ma(step - 1, ma, ma_start, ma_ramp_steps)
        ref_step = compute_reference_scales(ma_step, alpha_rad, rho_inf, gamma)
        apply_solver_control_descriptors(
            solver,
            _build_naca0012_solver_controls(
                rho_inf=rho_inf,
                u_inf=ref_step['u_inf'],
                v_inf=ref_step['v_inf'],
                p_inf=p_inf,
            ),
        )

        try:
            dt = solver.step()
        except RuntimeError as exc:
            print(f"\n  ❌ Solver fail-fast at step {step}: {exc}")
            return None

        if step % report == 0 or step == 1:
            ti.sync()
            rho_n, _, _, p_n = solver.get_primitive()
            residual = solver.get_residual_norm()
            CL, CD, Cp, Fx, Fy = compute_cl_cd(
                solver, alpha_rad, rho_inf, ref_step['v_mag'])
            elapsed = time.time() - sim_start

            print(f"  {step:>6}  {ma_step:>7.4f}  {dt:>10.4e}  {residual:>10.3e}  "
                  f"{CL:>8.4f}  {CD:>8.5f}  "
                  f"{p_n.min():>8.5f}  {p_n.max():>8.5f}  {elapsed:>7.2f}s")

            CL_history.append(CL)
            CD_history.append(CD)
            final_residual = residual
            final_step = step

            if not np.isfinite(p_n).all() or p_n.min() < 0.0:
                print(f"\n  ❌ 非物理解（step {step}），中止")
                return None

            if residual < tol:
                converged = True
                print(f"\n  ✅ Residual converged at step {step} "
                      f"(RMS={residual:.3e})")
                break

    total_time = time.time() - t0

    # ── 最終結果 ──────────────────────────────────────────────────
    ti.sync()
    ma_final = compute_ramped_ma(final_step, ma, ma_start, ma_ramp_steps)
    ref_final = compute_reference_scales(ma_final, alpha_rad, rho_inf, gamma)
    CL_final, CD_final, Cp, Fx, Fy = compute_cl_cd(
        solver, alpha_rad, rho_inf, ref_final['v_mag'])

    print(f"\n  模擬完成，總耗時 {total_time:.2f}s")
    print(f"\n{'='*64}")
    print(f"  Last reported step: {final_step}")
    print(f"  Final Ma:           {ma_final:.4f}")
    print(f"  Residual RMS:       {final_residual:.3e}")
    if converged:
        print(f"  收斂結果：CL = {CL_final:.4f},  CD = {CD_final:.5f}")
    else:
        print("  ⚠️  尚未收斂，下列 CL/CD 只能視為最後一步估計值")
        print(f"  最後一步估計：CL = {CL_final:.4f},  CD = {CD_final:.5f}")

    # ── 薄翼理論對比 ──────────────────────────────────────────────
    CL_thin  = 2.0 * np.pi * np.sin(alpha_rad)
    beta_pg  = np.sqrt(max(1.0 - ma_final**2, 0.01))
    CL_pg    = CL_thin / beta_pg   # Prandtl-Glauert 壓縮修正

    print(f"\n  理論值：")
    print(f"    薄翼（不可壓縮）: CL = {CL_thin:.4f}")
    print(f"    Prandtl-Glauert:  CL = {CL_pg:.4f}")
    print(f"    計算值:           CL = {CL_final:.4f}")

    if aoa > 0.1:
        ratio = CL_final / CL_pg
        within = 0.7 < ratio < 1.3
        print(f"    CL / CL_PG = {ratio:.3f}  "
              f"{'(合理範圍 0.7–1.3)' if within else '(偏差過大，可能需要更多步數)'}")

    # ── Cp 對稱性（AOA=0°）──────────────────────────────────────
    x_cc, y_cc = get_airfoil_cc(x_node, y_node, ni)
    if abs(aoa) < 0.1:
        upper = y_cc >= 0.0
        lower = y_cc <  0.0
        if upper.sum() > 1 and lower.sum() > 1:
            x_common = np.linspace(0.05, 0.95, 40)
            Cp_u = np.interp(x_common, x_cc[upper][::-1], Cp[upper][::-1])
            Cp_l = np.interp(x_common, x_cc[lower],       Cp[lower])
            asymm = float(np.mean(np.abs(Cp_u - Cp_l)))
            ok_sym = asymm < 0.05
            print(f"\n  Cp 上下對稱誤差（AOA=0°）= {asymm:.4e}  "
                  f"{'✅' if ok_sym else '❌'}")

    print("=" * 64)

    # ── 可視化輸出 ────────────────────────────────────────────────
    if output_dir is not None:
        import os
        from fvm_taichi.plot import plot_cp, plot_flow_field
        os.makedirs(output_dir, exist_ok=True)

        aoa_tag = f"aoa{int(aoa):+03d}" if aoa != 0 else "aoa000"
        tag = f"ma{int(ma*100):03d}_{aoa_tag}"

        print(f"\n[4] 輸出圖片至 {output_dir}/")

        plot_cp(solver, alpha_rad, rho_inf, ref['v_mag'],
                x_node, y_node,
                save_path=os.path.join(output_dir, f"cp_{tag}.png"))

        for field in ('velocity', 'pressure', 'density', 'mach'):
            plot_flow_field(solver, x_node, y_node,
                            field=field,
                            save_path=os.path.join(
                                output_dir, f"{field}_{tag}.png"))

    return CL_final, CD_final


def main():
    parser = argparse.ArgumentParser(description="NACA0012 Euler O-grid")
    parser.add_argument('--ni',     type=int,   default=180)
    parser.add_argument('--nj',     type=int,   default=60)
    parser.add_argument('--ma',     type=float, default=0.3)
    parser.add_argument('--ma_start', type=float, default=None,
                        help='Ramp start Mach number (default: disabled)')
    parser.add_argument('--ma_ramp_steps', type=int, default=0,
                        help='Number of steps for linear Mach ramp to target --ma')
    parser.add_argument('--aoa',    type=float, default=0.0)
    parser.add_argument('--steps',  type=int,   default=8000)
    parser.add_argument('--rfar',   type=float, default=15.0)
    parser.add_argument('--cfl',    type=float, default=0.45)
    parser.add_argument('--report', type=int,   default=1000)
    parser.add_argument('--tol',    type=float, default=1e-5,
                        help='Residual RMS convergence tolerance')
    parser.add_argument('--output', type=str,   default='output_naca0012',
                        help='輸出圖片目錄（設為空字串停用）')
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        result = run_naca0012(
            ni=args.ni, nj=args.nj, ma=args.ma, aoa=args.aoa,
            ma_start=args.ma_start, ma_ramp_steps=args.ma_ramp_steps,
            steps=args.steps, R_far=args.rfar, cfl=args.cfl,
            report=args.report, tol=args.tol,
            output_dir=args.output if args.output else None)
    except ValueError as exc:
        print(f"\n  ❌ Invalid input: {exc}")
        sys.exit(1)

    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
