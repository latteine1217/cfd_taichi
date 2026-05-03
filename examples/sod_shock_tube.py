"""
Sod Shock Tube — MVU-1 (Minimum Viable Unit)
=============================================

What: 1D Sod 激波管，驗證 HLLC + MUSCL(minmod) + SSP-RK3 數值核心
Why:  在引入 2D 幾何與 C-grid 前，確保可壓縮 FV 求解器邏輯正確
When: 可壓縮求解器開發的第一步驗證

索引規格（2 層 ghost）：
  NG=2, N=nx+4
  Interior : [NG, NG+nx)  = [2, nx+2)
  Ghost-L  : [0, 1]
  Ghost-R  : [nx+2, nx+3]
  Face i   : 位於 cell[i] 與 cell[i+1] 之間
  Int-faces: [NG-1, NG+nx) = [1, nx+2)   ← 共 nx+1 個

執行：
  uv run python examples/sod_shock_tube.py
  uv run python examples/sod_shock_tube.py --nx 800 --cfl 0.45

預期 L1 誤差（nx=400, t=0.2）：
  ρ ≈ 1–2e-3,  u ≈ 1e-3,  p ≈ 5e-4
"""

import taichi as ti
import numpy as np
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 1.4  # 理想氣體比熱比（空氣）


# ──────────────────────────────────────────────────────────────
# 精確解（module level，不依賴 Taichi）
# ──────────────────────────────────────────────────────────────
def sod_exact(x_arr: np.ndarray, t: float) -> tuple:
    """
    Sod 問題精確 Riemann 解
    左: ρ=1.0, u=0.0, p=1.0  /  右: ρ=0.125, u=0.0, p=0.1
    """
    from scipy.optimize import brentq

    g = GAMMA
    rho1, u1, p1 = 1.0,   0.0, 1.0
    rho4, u4, p4 = 0.125, 0.0, 0.1
    c1 = np.sqrt(g * p1 / rho1)
    c4 = np.sqrt(g * p4 / rho4)

    # 左膨脹 / 右激波 pressure equation
    def f_eq(ps):
        ustar_L = u1 - 2*c1/(g-1) * ((ps/p1)**((g-1)/(2*g)) - 1)
        A = 2 / ((g+1)*rho4);  B = (g-1)/(g+1)*p4
        ustar_R = u4 + (ps - p4) * np.sqrt(A/(ps + B))
        return ustar_L - ustar_R

    p_star = brentq(f_eq, 1e-10, 100.0, xtol=1e-12)

    A = 2/((g+1)*rho4);  B = (g-1)/(g+1)*p4
    u_star    = u4 + (p_star - p4)*np.sqrt(A/(p_star + B))
    c_star_L  = c1*(p_star/p1)**((g-1)/(2*g))
    rho_starL = rho1*(p_star/p1)**(1.0/g)
    rho_starR = rho4*(p_star + (g-1)/(g+1)*p4) / ((g-1)/(g+1)*p_star + p4)
    v_sh      = u4 + c4*np.sqrt((g+1)/(2*g)*p_star/p4 + (g-1)/(2*g))
    v_tail    = u1 - c1          # left rarefaction head

    x0 = 0.5
    rho_e = np.empty_like(x_arr)
    u_e   = np.empty_like(x_arr)
    p_e   = np.empty_like(x_arr)

    for k, x in enumerate(x_arr):
        xi = (x - x0) / t
        if xi < v_tail:
            rho_e[k], u_e[k], p_e[k] = rho1, u1, p1
        elif xi < u_star - c_star_L:
            u_loc  = 2/(g+1)*(c1 + (g-1)/2*u1 + xi)
            c_loc  = c1 + (g-1)/2*(u1 - u_loc)
            rho_e[k] = rho1*(c_loc/c1)**(2/(g-1))
            u_e[k]   = u_loc
            p_e[k]   = p1*(c_loc/c1)**(2*g/(g-1))
        elif xi < u_star:
            rho_e[k], u_e[k], p_e[k] = rho_starL, u_star, p_star
        elif xi < v_sh:
            rho_e[k], u_e[k], p_e[k] = rho_starR, u_star, p_star
        else:
            rho_e[k], u_e[k], p_e[k] = rho4, u4, p4

    return rho_e, u_e, p_e


# ──────────────────────────────────────────────────────────────
# 主求解器
# ──────────────────────────────────────────────────────────────
def run_sod(nx: int = 400, t_end: float = 0.2,
            cfl: float = 0.45, output_dir: str = "output_sod"):
    """
    執行 1D Sod 激波管。

    Args:
        nx:         內部格點數
        t_end:      結束時間（Sod 標準 t=0.2）
        cfl:        CFL 數（建議 0.4–0.5）
        output_dir: 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)

    NG = 2
    N  = nx + 2 * NG   # = nx + 4
    dx = 1.0 / nx

    print("=" * 64)
    print("  SOD SHOCK TUBE  ·  MVU-1  (Euler + HLLC + MUSCL + SSP-RK3)")
    print("=" * 64)
    print(f"  nx={nx}  N={N}  dx={dx:.4e}  t_end={t_end}  CFL={cfl}")
    print(f"  Interior [{NG}, {NG+nx-1}]   Faces [{NG-1}, {NG+nx-1}]")
    print("=" * 64)

    # ── Fields ──────────────────────────────────────────────────
    # U = [ρ, ρu, E]
    U  = ti.Vector.field(3, ti.f32, shape=N)   # 當前步
    U0 = ti.Vector.field(3, ti.f32, shape=N)   # RK3 步起點快照
    U1 = ti.Vector.field(3, ti.f32, shape=N)   # RK3 中間步 1
    U2 = ti.Vector.field(3, ti.f32, shape=N)   # RK3 中間步 2

    # W = [ρ, u, p]  原始量（limiter 在原始量空間操作，避免非物理狀態）
    W = ti.Vector.field(3, ti.f32, shape=N)

    # Face 重建結果
    WL_face = ti.Vector.field(3, ti.f32, shape=N)
    WR_face = ti.Vector.field(3, ti.f32, shape=N)
    Flux    = ti.Vector.field(3, ti.f32, shape=N)

    # 殘差
    R = ti.Vector.field(3, ti.f32, shape=N)

    dt_global = ti.field(ti.f32, shape=())

    # ── @ti.func：在 kernel 內呼叫 ───────────────────────────────
    @ti.func
    def prim2cons(rho: ti.f32, u: ti.f32, p: ti.f32):
        E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
        return ti.Vector([rho, rho * u, E])

    @ti.func
    def cons2prim(rho: ti.f32, rhou: ti.f32, E: ti.f32):
        u = rhou / rho
        p = (GAMMA - 1.0) * (E - 0.5 * rho * u * u)
        return ti.Vector([rho, u, p])

    @ti.func
    def euler_flux(rho: ti.f32, u: ti.f32, p: ti.f32):
        """1D Euler 通量 F = [ρu, ρu²+p, (E+p)u]"""
        E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
        return ti.Vector([rho*u,  rho*u*u + p,  (E+p)*u])

    @ti.func
    def minmod(a: ti.f32, b: ti.f32) -> ti.f32:
        r = 0.0
        if a * b > 0.0:
            r = ti.math.sign(a) * ti.min(ti.abs(a), ti.abs(b))
        return r

    @ti.func
    def hllc_1d(wL: ti.types.vector(3, ti.f32),
                wR: ti.types.vector(3, ti.f32)):
        """
        1D HLLC Riemann Solver（Toro 1994）
        What: 三波估計（S_L, S_M, S_R）組合界面通量
        Why:  精確捕捉激波（非擴散）、接觸面（密度跳）、膨脹波
        """
        rhoL, uL, pL = wL[0], wL[1], wL[2]
        rhoR, uR, pR = wR[0], wR[1], wR[2]
        EL = pL/(GAMMA-1.0) + 0.5*rhoL*uL*uL
        ER = pR/(GAMMA-1.0) + 0.5*rhoR*uR*uR

        cL = ti.sqrt(GAMMA * pL / rhoL)
        cR = ti.sqrt(GAMMA * pR / rhoR)

        # 波速估計（Davis / Einfeldt）
        SL = ti.min(uL - cL, uR - cR)
        SR = ti.max(uL + cL, uR + cR)

        # 接觸面速度（Rankine-Hugoniot）
        denom = rhoL*(SL - uL) - rhoR*(SR - uR)
        SM = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / denom

        UL_vec = ti.Vector([rhoL,       rhoL*uL,       EL])
        UR_vec = ti.Vector([rhoR,       rhoR*uR,       ER])
        FL     = euler_flux(rhoL, uL, pL)
        FR     = euler_flux(rhoR, uR, pR)

        # Star states（Toro 1994 eq. 10.38）
        coefL = rhoL * (SL - uL) / (SL - SM)
        coefR = rhoR * (SR - uR) / (SR - SM)
        UstarL = ti.Vector([
            coefL,
            coefL * SM,
            coefL * (EL/rhoL + (SM - uL)*(SM + pL/(rhoL*(SL - uL)))),
        ])
        UstarR = ti.Vector([
            coefR,
            coefR * SM,
            coefR * (ER/rhoR + (SM - uR)*(SM + pR/(rhoR*(SR - uR)))),
        ])

        result = FR
        if SL >= 0.0:
            result = FL
        elif SM >= 0.0:
            result = FL + SL * (UstarL - UL_vec)
        elif SR >= 0.0:
            result = FR + SR * (UstarR - UR_vec)
        return result

    # ── Kernels ──────────────────────────────────────────────────
    @ti.kernel
    def init_sod():
        """Sod 初始條件：x < 0.5 左狀態，x ≥ 0.5 右狀態"""
        for i in range(N):
            x = (i - NG) * dx + 0.5 * dx   # cell 中心座標
            rho = 0.0;  u = 0.0;  p = 0.0
            if x < 0.5:
                rho, u, p = 1.0,   0.0, 1.0
            else:
                rho, u, p = 0.125, 0.0, 0.1
            W[i] = ti.Vector([rho, u, p])
            U[i] = prim2cons(rho, u, p)

    @ti.kernel
    def update_ghost():
        """Zero-gradient outflow BC：ghost ← 最近 interior cell"""
        for g in range(NG):
            W[NG - 1 - g]   = W[NG]
            U[NG - 1 - g]   = U[NG]
            W[NG + nx + g]  = W[NG + nx - 1]
            U[NG + nx + g]  = U[NG + nx - 1]

    @ti.kernel
    def update_primitive():
        """Interior cells：U → W（cons → prim）"""
        for i in range(NG, NG + nx):
            W[i] = cons2prim(U[i][0], U[i][1], U[i][2])

    @ti.kernel
    def reconstruct():
        """
        MUSCL face 重建（原始量空間）
        Face i 在 cell[i] 與 cell[i+1] 之間。
          左狀態來自 cell[i]  ：slope(i-1, i, i+1)
          右狀態來自 cell[i+1]：slope(i, i+1, i+2)
        Loop: [NG-1, NG+nx) → 訪問 W[i-1..i+2]，2 層 ghost 完全安全
        正性保護：ρ 或 p ≤ 0 → 降回一階（piecewise constant）
        """
        for i in range(NG - 1, NG + nx):
            # Left state（來自 cell i）
            dWm = W[i]     - W[i - 1]
            dWp = W[i + 1] - W[i]
            sL  = ti.Vector([minmod(dWm[0], dWp[0]),
                              minmod(dWm[1], dWp[1]),
                              minmod(dWm[2], dWp[2])])
            wL_cand = W[i] + 0.5 * sL

            # Right state（來自 cell i+1）
            dWm2 = W[i + 1] - W[i]
            dWp2 = W[i + 2] - W[i + 1]
            sR   = ti.Vector([minmod(dWm2[0], dWp2[0]),
                               minmod(dWm2[1], dWp2[1]),
                               minmod(dWm2[2], dWp2[2])])
            wR_cand = W[i + 1] - 0.5 * sR

            # 正性回退：任一側 ρ/p ≤ 0 → 一階
            if wL_cand[0] <= 0.0 or wL_cand[2] <= 0.0 or \
               wR_cand[0] <= 0.0 or wR_cand[2] <= 0.0:
                WL_face[i] = W[i]
                WR_face[i] = W[i + 1]
            else:
                WL_face[i] = wL_cand
                WR_face[i] = wR_cand

    @ti.kernel
    def compute_fluxes():
        """Interior faces [NG-1, NG+nx)：呼叫 HLLC"""
        for i in range(NG - 1, NG + nx):
            Flux[i] = hllc_1d(WL_face[i], WR_face[i])

    @ti.kernel
    def accumulate_residual():
        """
        R[i] = (Flux[i] - Flux[i-1]) / dx
        cell[i] 的右面 = Flux[i]，左面 = Flux[i-1]
        """
        for i in range(NG, NG + nx):
            R[i] = (Flux[i] - Flux[i - 1]) / dx

    @ti.kernel
    def compute_dt():
        """全域最小時間步（顯式 Euler CFL）"""
        dt_global[None] = 1.0e10
        for i in range(NG, NG + nx):
            rho, u, p = W[i][0], W[i][1], W[i][2]
            lam = ti.abs(u) + ti.sqrt(GAMMA * p / rho)
            ti.atomic_min(dt_global[None], cfl * dx / lam)

    @ti.kernel
    def save_U0():
        for i in range(N):
            U0[i] = U[i]

    @ti.kernel
    def rk_stage1(dt: ti.f32):
        """U1 = U0 - dt·R(U^n)"""
        for i in range(NG, NG + nx):
            U1[i] = U0[i] - dt * R[i]

    @ti.kernel
    def rk_stage2(dt: ti.f32):
        """U2 = ¾ U0 + ¼ U1 - ¼ dt·R(U1)"""
        for i in range(NG, NG + nx):
            U2[i] = 0.75*U0[i] + 0.25*U1[i] - 0.25*dt*R[i]

    @ti.kernel
    def rk_stage3(dt: ti.f32):
        """U = ⅓ U0 + ⅔ U2 - ⅔ dt·R(U2)  →  U^{n+1}"""
        for i in range(NG, NG + nx):
            U[i] = (1.0/3.0)*U0[i] + (2.0/3.0)*U2[i] - (2.0/3.0)*dt*R[i]

    @ti.kernel
    def copy_to_U(src: ti.template()):
        """把 src field 複製至 U（RK3 stage 切換用）"""
        for i in range(N):
            U[i] = src[i]

    # ── 殘差評估流程（封裝重複呼叫）────────────────────────────────
    def eval_residual():
        """reconstruct → flux → residual（用當前 U/W）"""
        reconstruct()
        compute_fluxes()
        accumulate_residual()

    def set_state(src):
        """切換求解狀態：copy_to_U(src) → update_primitive → ghost"""
        copy_to_U(src)
        update_primitive()
        update_ghost()

    # ── 初始化 ──────────────────────────────────────────────────
    init_sod()
    update_ghost()

    t     = 0.0
    step  = 0
    t0    = time.time()

    print(f"\n{'step':>7}  {'t':>8}  {'dt':>10}  "
          f"{'ρ_min':>8}  {'p_min':>8}  {'sps':>6}")
    print("─" * 58)

    # ── 主迴圈 ──────────────────────────────────────────────────
    while t < t_end:
        compute_dt()
        ti.sync()
        dt_val = min(float(dt_global[None]), t_end - t)

        # SSP-RK3（Shu–Osher 三步）
        save_U0()

        # Stage 1：R(U^n), → U1
        eval_residual()
        rk_stage1(dt_val)
        set_state(U1)

        # Stage 2：R(U1), → U2
        eval_residual()
        rk_stage2(dt_val)
        set_state(U2)

        # Stage 3：R(U2), → U^{n+1}（寫入 U）
        eval_residual()
        rk_stage3(dt_val)
        update_primitive()
        update_ghost()

        t    += dt_val
        step += 1

        if step % 500 == 0 or abs(t - t_end) < 1e-12:
            ti.sync()
            W_np = W.to_numpy()
            rho_min = W_np[NG:NG+nx, 0].min()
            p_min   = W_np[NG:NG+nx, 2].min()
            sps     = step / (time.time() - t0)
            print(f"{step:>7}  {t:>8.4f}  {dt_val:>10.3e}  "
                  f"{rho_min:>8.5f}  {p_min:>8.5f}  {sps:>6.1f}")

    ti.sync()
    elapsed = time.time() - t0
    print(f"\n  Completed: t={t:.4f},  steps={step},  elapsed={elapsed:.2f}s")

    # ── 驗證：L1 vs 精確解 ────────────────────────────────────────
    W_np  = W.to_numpy()
    x_int = np.array([(i - NG + 0.5)*dx for i in range(NG, NG+nx)])
    rho_n = W_np[NG:NG+nx, 0]
    u_n   = W_np[NG:NG+nx, 1]
    p_n   = W_np[NG:NG+nx, 2]

    try:
        rho_ex, u_ex, p_ex = sod_exact(x_int, t_end)
        l1_rho = float(np.mean(np.abs(rho_n - rho_ex)))
        l1_u   = float(np.mean(np.abs(u_n   - u_ex)))
        l1_p   = float(np.mean(np.abs(p_n   - p_ex)))
        print(f"\n  L1 error:  ρ={l1_rho:.4e}  u={l1_u:.4e}  p={l1_p:.4e}")
        has_exact = True
    except ImportError:
        print("\n  [WARN] scipy not available; skipping L1 comparison")
        has_exact = False
        rho_ex = u_ex = p_ex = None
        l1_rho = l1_u = l1_p = float("nan")

    # ── 繪圖 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        (rho_n, rho_ex, "Density ρ",   [0.0, 1.1]),
        (u_n,   u_ex,   "Velocity u",  [-0.1, 1.0]),
        (p_n,   p_ex,   "Pressure p",  [0.0, 1.1]),
    ]
    for ax, (num, ex, name, ylim) in zip(axes, pairs):
        if has_exact:
            ax.plot(x_int, ex, "k-",  lw=2.0, label="Exact",        zorder=3)
        ax.plot(x_int, num, "r--", lw=1.5,
                label=f"HLLC+MUSCL  nx={nx}", zorder=2)
        ax.set_xlabel("x");  ax.set_ylabel(name);  ax.set_title(name)
        ax.set_ylim(ylim);   ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sod Shock Tube   t={t_end:.2f}   nx={nx}   "
        f"L1(ρ)={l1_rho:.2e}", fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(output_dir, f"sod_nx{nx}_t{t_end:.2f}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_png}\n")

    return l1_rho, l1_u, l1_p


def main():
    parser = argparse.ArgumentParser(description="Sod Shock Tube (MVU-1)")
    parser.add_argument("--nx",     type=int,   default=400)
    parser.add_argument("--t_end",  type=float, default=0.2)
    parser.add_argument("--cfl",    type=float, default=0.45)
    parser.add_argument("--output", type=str,   default="output_sod")
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_sod(nx=args.nx, t_end=args.t_end,
            cfl=args.cfl, output_dir=args.output)


if __name__ == "__main__":
    main()
