"""
fvm_taichi.plot — FVM Euler 求解器可視化工具
=============================================

What: 三類圖表工具，用於後處理與驗證
Why:  純數字輸出難以直觀判斷流場物理正確性；
      Cp 分佈、流場等高線、CL-AOA 曲線是翼型 CFD 的標準驗證圖

使用方式：
    from fvm_taichi.plot import plot_cp, plot_flow_field, plot_cl_aoa

提供：
    plot_cp(...)              翼型表面壓力係數分佈（上/下表面）
    plot_surface_cp_cf(...)   翼面 Cp / Cf 專用圖
    plot_wall_shear_distribution(...)  壁面剪應力分佈圖
    plot_wall_unit_distribution(...)   翼面 y+ 分佈圖
    plot_flow_field(...)      流場等高線（壓力 / 密度 / 馬赫數）
    plot_cell_scalar_field(...) 任意 cell scalar field 等高線
    plot_cl_aoa(...)          CL vs AOA 曲線（計算值 vs 理論對比）
    plot_coeff_vs_aoa(...)    黏性案例的 CL/CD vs AOA 曲線
    plot_coeff_vs_re(...)     黏性案例的 CL/CD vs Re 曲線
    plot_force_polar(...)     CL-CD 極線圖
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 無顯示器環境下安全；show() 前可更改 backend
import matplotlib.pyplot as plt
from typing import Optional


def _split_airfoil_surfaces(x: np.ndarray, y: np.ndarray, values: np.ndarray):
    """
    依 y 座標將封閉翼型表面資料分為上下表面，並按 x/c 排序
    """
    upper = y >= 0.0
    lower = y < 0.0
    iu = np.argsort(x[upper])
    il = np.argsort(x[lower])
    return (
        x[upper][iu], values[upper][iu],
        x[lower][il], values[lower][il],
    )


def _plot_force_components(ax, x_values: np.ndarray, results: list, coeff: str):
    """
    在同一張圖上畫總力係數與 pressure / viscous 分量
    """
    coeff = coeff.upper()
    component_p = f"{coeff}_p"
    component_v = f"{coeff}_v"
    total = np.array([row[coeff] for row in results], dtype=float)
    pressure = np.array([row.get(component_p, np.nan) for row in results], dtype=float)
    viscous = np.array([row.get(component_v, np.nan) for row in results], dtype=float)

    ax.plot(x_values, total, "ko-", ms=6, lw=1.8, label=f"Total {coeff}")
    if np.isfinite(pressure).any():
        ax.plot(x_values, pressure, "b--o", ms=4, lw=1.2, label=f"Pressure {coeff}")
    if np.isfinite(viscous).any():
        ax.plot(x_values, viscous, "r--o", ms=4, lw=1.2, label=f"Viscous {coeff}")


# ── Cp 分佈圖 ────────────────────────────────────────────────────────────────
def plot_cp(solver, alpha_rad: float, rho_inf: float, v_mag: float,
            x_node: np.ndarray, y_node: np.ndarray,
            save_path: Optional[str] = None,
            show: bool = False):
    """
    繪製翼型表面 Cp 分佈

    What: 從第一層 interior cells (j=0) 取壓力，計算 Cp = (p-p∞)/q∞，
          分上（y≥0）/下（y<0）表面按 x/c 排序後分色繪製
    Why:  Cp 分佈是翼型氣動力的主要驗證圖：
          - AOA=0° 時上下 Cp 應對稱
          - 跨音速時可見 Cp 跳躍（激波）
          - CL ∝ 上下 Cp 面積差

    Args:
        solver:         EulerSolver instance（模擬已完成）
        alpha_rad:      攻角（弧度）
        rho_inf:        自由流密度
        v_mag:          自由流速度大小（Ma，因 c∞=1）
        x_node, y_node: 節點座標，shape (NI+1, NJ+1)
        save_path:      儲存路徑；None = 不存檔
        show:           True = 顯示互動視窗
    """
    from fvm_taichi import EulerSolver
    NG    = EulerSolver.NG
    ni    = solver.ni
    gamma = solver.gamma

    _, _, _, p_n = solver.get_primitive()   # (ni, nj)
    p_wall = p_n[:, 0]

    # Cell center = 4 角節點平均（j=0 interior layer）
    x_cc = 0.25 * (x_node[NG:NG+ni,   NG  ] + x_node[NG+1:NG+ni+1, NG  ]
                 + x_node[NG:NG+ni,   NG+1] + x_node[NG+1:NG+ni+1, NG+1])
    y_cc = 0.25 * (y_node[NG:NG+ni,   NG  ] + y_node[NG+1:NG+ni+1, NG  ]
                 + y_node[NG:NG+ni,   NG+1] + y_node[NG+1:NG+ni+1, NG+1])

    p_inf = rho_inf / gamma
    q_inf = 0.5 * rho_inf * v_mag**2
    Cp    = (p_wall - p_inf) / q_inf

    upper = y_cc >= 0.0
    lower = y_cc <  0.0
    iu    = np.argsort(x_cc[upper])
    il    = np.argsort(x_cc[lower])

    aoa_deg = float(np.rad2deg(alpha_rad))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_cc[upper][iu], Cp[upper][iu],
            'b-o', ms=3, lw=1.5, label='Upper surface')
    ax.plot(x_cc[lower][il], Cp[lower][il],
            'r-o', ms=3, lw=1.5, label='Lower surface')

    ax.invert_yaxis()                        # 標準航空慣例：-Cp 向上
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel(r'$C_p$', fontsize=12)
    ax.set_title(
        f'Cp Distribution — NACA0012,  Ma={v_mag:.2f},  AOA={aoa_deg:.1f}°',
        fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_surface_cp_cf(
        x_surface: np.ndarray,
        y_surface: np.ndarray,
        cp: np.ndarray,
        cf: np.ndarray,
        aoa_deg: float,
        re: float,
        ma: float,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製翼面 Cp / Cf 專用圖

    What: 上圖畫 Cp，下圖畫 Cf，皆按上/下表面分色
    Why:  壓力與摩擦貢獻需要一起看，才能判斷 drag 來源與邊界層狀態
    """
    xu, cpu, xl, cpl = _split_airfoil_surfaces(x_surface, y_surface, cp)
    xu_cf, cfu, xl_cf, cfl = _split_airfoil_surfaces(x_surface, y_surface, cf)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(xu, cpu, 'b-o', ms=3, lw=1.4, label='Upper surface')
    axes[0].plot(xl, cpl, 'r-o', ms=3, lw=1.4, label='Lower surface')
    axes[0].invert_yaxis()
    axes[0].axhline(0.0, color='k', lw=0.5, ls='--')
    axes[0].set_ylabel(r'$C_p$', fontsize=12)
    axes[0].set_title(
        f'Surface Coefficients — NACA0012, Re={re:.0f}, Ma={ma:.2f}, AOA={aoa_deg:.1f}°',
        fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(xu_cf, cfu, 'b-o', ms=3, lw=1.4, label='Upper surface')
    axes[1].plot(xl_cf, cfl, 'r-o', ms=3, lw=1.4, label='Lower surface')
    axes[1].axhline(0.0, color='k', lw=0.5, ls='--')
    axes[1].set_xlabel('x/c', fontsize=12)
    axes[1].set_ylabel(r'$C_f$', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[0].set_xlim(-0.02, 1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_wall_shear_distribution(
        x_surface: np.ndarray,
        y_surface: np.ndarray,
        tau_t: np.ndarray,
        aoa_deg: float,
        re: float,
        ma: float,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製壁面切向剪應力分佈圖

    What: 分上/下表面畫 tau_t(x)
    Why:  分離、再附著與黏性 drag 來源都會直接反映在 wall shear sign / magnitude
    """
    xu, tau_u, xl, tau_l = _split_airfoil_surfaces(x_surface, y_surface, tau_t)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xu, tau_u, 'b-o', ms=3, lw=1.5, label='Upper surface')
    ax.plot(xl, tau_l, 'r-o', ms=3, lw=1.5, label='Lower surface')
    ax.axhline(0.0, color='k', lw=0.6, ls='--')
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel(r'$\tau_t$', fontsize=12)
    ax.set_title(
        f'Wall Shear Distribution — NACA0012, Re={re:.0f}, Ma={ma:.2f}, AOA={aoa_deg:.1f}°',
        fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_wall_unit_distribution(
        x_surface: np.ndarray,
        y_surface: np.ndarray,
        y_plus: np.ndarray,
        aoa_deg: float,
        re: float,
        ma: float,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製壁面 y+ 分佈圖

    What: 分上/下表面畫 y+(x)
    Why:  高 Re 與 RANS 近壁解析度是否合理，第一個要看的就是 y+ 分佈
    """
    xu, ypu, xl, ypl = _split_airfoil_surfaces(x_surface, y_surface, y_plus)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xu, ypu, 'b-o', ms=3, lw=1.5, label='Upper surface')
    ax.plot(xl, ypl, 'r-o', ms=3, lw=1.5, label='Lower surface')
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel(r'$y^+$', fontsize=12)
    ax.set_title(
        f'Wall Unit Distribution — NACA0012, Re={re:.0f}, Ma={ma:.2f}, AOA={aoa_deg:.1f}°',
        fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── 流場等高線圖 ──────────────────────────────────────────────────────────────
def plot_flow_field(solver, x_node: np.ndarray, y_node: np.ndarray,
                    field: str = 'pressure',
                    xlim: tuple = (-0.6, 1.6),
                    ylim: tuple = (-1.0, 1.0),
                    n_levels: int = 40,
                    resolution: int = 300,
                    save_path: Optional[str] = None,
                    show: bool = False):
    """
    繪製 O-grid 流場等高線（scipy 插值 → 規則格點 → contourf）

    What: 把 interior cells 的標量場（壓力/密度/馬赫數）從曲線 O-grid
          插值到笛卡兒規則格點，再用 contourf 繪製平滑等高線；
          翼型內部以 matplotlib.path.Path 遮罩
    Why:  直接在 O-grid 上使用 pcolormesh 會產生明顯的扭曲色塊與
          週期性接縫，插值到規則格點後可得到平滑、發表品質的等高線圖

    Args:
        solver:         EulerSolver instance
        x_node, y_node: 節點座標，shape (NI+1, NJ+1)
        field:          'pressure' | 'density' | 'mach'
        xlim, ylim:     顯示範圍（預設：翼型附近 ±1c）
        n_levels:       等高線層數
        resolution:     笛卡兒規則格點解析度（預設 300×300）
        save_path, show: 同 plot_cp
    """
    from fvm_taichi import EulerSolver
    from scipy.interpolate import griddata
    from matplotlib.path import Path

    NG    = EulerSolver.NG
    ni    = solver.ni
    nj    = solver.nj
    gamma = solver.gamma

    rho_n, u_n, v_n, p_n = solver.get_primitive()   # (ni, nj)

    if field == 'pressure':
        F, cmap, label = p_n,  'RdBu_r',  'Pressure  p'
    elif field == 'density':
        F, cmap, label = rho_n, 'viridis', 'Density  ρ'
    elif field == 'mach':
        c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, 1e-6))
        F   = np.sqrt(u_n**2 + v_n**2) / c_n
        cmap, label = 'hot_r', 'Mach Number'
    elif field == 'velocity':
        F   = np.sqrt(u_n**2 + v_n**2)
        cmap, label = 'turbo', 'Velocity  |u|'
    else:
        raise ValueError(f"Unknown field '{field}'; choose 'pressure', 'density', 'mach', 'velocity'")

    # 所有 interior cell centers，shape (ni, nj)
    xc = 0.25 * (x_node[NG:NG+ni,   NG:NG+nj]   + x_node[NG+1:NG+ni+1, NG:NG+nj]
               + x_node[NG:NG+ni,   NG+1:NG+nj+1] + x_node[NG+1:NG+ni+1, NG+1:NG+nj+1])
    yc = 0.25 * (y_node[NG:NG+ni,   NG:NG+nj]   + y_node[NG+1:NG+ni+1, NG:NG+nj]
               + y_node[NG:NG+ni,   NG+1:NG+nj+1] + y_node[NG+1:NG+ni+1, NG+1:NG+nj+1])

    # 使用全部 O-grid 點（R_far=15 涵蓋顯示框四角）以消除 convex hull 缺口
    pts  = np.column_stack([xc.ravel(), yc.ravel()])
    vals = F.ravel()

    # 規則笛卡兒格點
    xi = np.linspace(xlim[0], xlim[1], resolution)
    yi = np.linspace(ylim[0], ylim[1], resolution)
    XI, YI = np.meshgrid(xi, yi)

    FI = griddata(pts, vals, (XI, YI), method='linear')

    # 翼型表面節點（閉合多邊形）
    x_foil = x_node[NG:NG+ni+1, NG]
    y_foil = y_node[NG:NG+ni+1, NG]

    # 遮罩翼型內部
    foil_path = Path(np.column_stack([x_foil, y_foil]))
    inside = foil_path.contains_points(
        np.column_stack([XI.ravel(), YI.ravel()])
    ).reshape(XI.shape)
    FI[inside] = np.nan

    # 用原始 O-grid 場計算 percentile（近壁面格點密集，能反映真實物理範圍）
    # 避免規則格點因遠場均勻流佔多數而夾死色階
    vmin = float(np.percentile(F, 2))
    vmax = float(np.percentile(F, 98))
    if vmin >= vmax:
        # 如果場幾乎是常數，則人為擴展範圍以避免 contourf 報錯
        eps = max(1e-6, 0.05 * abs(vmin))
        vmin -= eps
        vmax += eps

    fig, ax = plt.subplots(figsize=(11, 6))

    cf = ax.contourf(XI, YI, FI,
                     levels=np.linspace(vmin, vmax, n_levels),
                     cmap=cmap, extend='both')
    cb = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label, fontsize=11)

    # 翼型輪廓（填充 + 邊線）
    ax.fill(x_foil, y_foil, color='0.75', zorder=5, linewidth=0)
    ax.plot(x_foil, y_foil, 'k-', lw=0.8, zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    ax.set_title(f'Flow Field ({label.split()[0]}) — NACA0012', fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cell_scalar_field(
        x_node: np.ndarray,
        y_node: np.ndarray,
        field_values: np.ndarray,
        field_label: str,
        title: str,
        xlim: tuple = (-0.6, 1.6),
        ylim: tuple = (-1.0, 1.0),
        n_levels: int = 40,
        resolution: int = 300,
        cmap: str = "viridis",
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製任意 cell scalar field 的 O-grid 等高線

    What: 將已知 cell-centered 標量場插值到規則格點，再繪製 contourf
    Why:  transonic 診斷量如 entropy rise / shock sensor 不在 solver 內建欄位，
          但仍需要與標準流場圖一致的後處理品質
    """
    from matplotlib.path import Path
    from scipy.interpolate import griddata

    ni, nj = field_values.shape
    NG = (x_node.shape[0] - 1 - ni) // 2

    xc = 0.25 * (x_node[NG:NG+ni,   NG:NG+nj] + x_node[NG+1:NG+ni+1, NG:NG+nj]
               + x_node[NG:NG+ni,   NG+1:NG+nj+1] + x_node[NG+1:NG+ni+1, NG+1:NG+nj+1])
    yc = 0.25 * (y_node[NG:NG+ni,   NG:NG+nj] + y_node[NG+1:NG+ni+1, NG:NG+nj]
               + y_node[NG:NG+ni,   NG+1:NG+nj+1] + y_node[NG+1:NG+ni+1, NG+1:NG+nj+1])

    xg = np.linspace(xlim[0], xlim[1], resolution)
    yg = np.linspace(ylim[0], ylim[1], resolution)
    Xg, Yg = np.meshgrid(xg, yg)

    points = np.column_stack([xc.ravel(), yc.ravel()])
    values = field_values.ravel()
    Fg = griddata(points, values, (Xg, Yg), method="linear")
    Fg_near = griddata(points, values, (Xg, Yg), method="nearest")
    Fg = np.where(np.isnan(Fg), Fg_near, Fg)

    x_foil = x_node[NG:NG+ni+1, NG]
    y_foil = y_node[NG:NG+ni+1, NG]
    foil_path = Path(np.column_stack([x_foil, y_foil]))
    inside_foil = foil_path.contains_points(np.column_stack([Xg.ravel(), Yg.ravel()]))
    Fg = np.ma.array(Fg, mask=inside_foil.reshape(Xg.shape))

    fig, ax = plt.subplots(figsize=(8, 4))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if abs(vmax - vmin) < 1e-12:
        pad = 1e-6 if abs(vmax) < 1e-6 else 1e-3 * abs(vmax)
        vmin -= pad
        vmax += pad
    levels = np.linspace(vmin, vmax, n_levels)
    cf = ax.contourf(Xg, Yg, Fg, levels=levels, cmap=cmap, extend="both")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(field_label, fontsize=11)

    ax.fill(x_foil, y_foil, color="0.75", zorder=5, linewidth=0)
    ax.plot(x_foil, y_foil, "k-", lw=0.8, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", fontsize=12)
    ax.set_ylabel("y/c", fontsize=12)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CL vs AOA 曲線 ────────────────────────────────────────────────────────────
def plot_cl_aoa(results: list, ma: float,
                save_path: Optional[str] = None,
                show: bool = False):
    """
    繪製 CL vs AOA 曲線（計算值 vs 薄翼理論）

    What: 從 validate_euler_naca_sweep 的結果列表（含 aoa, CL, CL_pg）
          繪製三條線：計算值（點+線）、PG 理論、不可壓理論
    Why:  CL 線性度是亞音速 Euler 求解器的核心驗證：
          - 斜率 ≈ 2π/β（Prandtl-Glauert）
          - 計算值偏低 ~25% 為粗網格厚翼效應，非程式錯誤

    Args:
        results: list of dict，每個元素須含 'aoa', 'CL', 'CL_pg'
        ma:      自由流馬赫數（標題顯示）
        save_path, show: 同 plot_cp
    """
    beta   = np.sqrt(max(1.0 - ma**2, 0.01))
    aoas    = np.array([r['aoa']   for r in results])
    cl_calc = np.array([r['CL']    for r in results])
    cl_pg   = np.array([r['CL_pg'] for r in results])

    # 理論曲線（密集點）
    a_fine     = np.linspace(0, max(aoas) + 1, 300)
    cl_thin_f  = 2.0 * np.pi * np.sin(np.deg2rad(a_fine))
    cl_pg_fine = cl_thin_f / beta

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a_fine, cl_pg_fine, 'k--', lw=1.5, alpha=0.8,
            label=f'Prandtl-Glauert  (Ma={ma:.2f})')
    ax.plot(a_fine, cl_thin_f,  'k:',  lw=1.2, alpha=0.6,
            label='Thin airfoil (incompressible)')
    ax.plot(aoas, cl_calc, 'bo-', ms=7, lw=2.0,
            label='EulerSolver (computed)')

    ax.set_xlabel('Angle of Attack  α (°)', fontsize=12)
    ax.set_ylabel(r'Lift Coefficient  $C_L$', fontsize=12)
    ax.set_title(f'CL vs AOA — NACA0012,  Ma={ma:.2f}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(aoas) + 1)
    ax.set_ylim(-0.05, np.max(cl_pg_fine) * 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_coeff_vs_aoa(
        results: list,
        coeff: str,
        ma: float,
        re: float,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製黏性翼型案例的係數-攻角曲線

    What: 將 sweep 結果中的 CL 或 CD 對 AOA 繪出，並附 pressure / viscous 分量
    Why:  有限 Re 案例需要同時觀察總力與分量，才能區分壓力與摩擦主導區間
    """
    coeff = coeff.upper()
    if coeff not in {"CL", "CD"}:
        raise ValueError(f"coeff must be 'CL' or 'CD', got {coeff}")

    results_sorted = sorted(results, key=lambda row: row["aoa"])
    aoa_values = np.array([row["aoa"] for row in results_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_force_components(ax, aoa_values, results_sorted, coeff)
    ax.set_xlabel("Angle of Attack  alpha (deg)", fontsize=12)
    ax.set_ylabel(f"{coeff}", fontsize=12)
    ax.set_title(f"{coeff} vs AOA — NACA0012, Re={re:.0f}, Ma={ma:.2f}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_coeff_vs_re(
        results: list,
        coeff: str,
        aoa_deg: float,
        ma: float,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製黏性翼型案例的係數-Re 曲線

    What: 將 sweep 結果中的 CL 或 CD 對 Reynolds 數繪出，x 軸採對數刻度
    Why:  黏性效應常以 Re 的 log 尺度變化，線性軸會壓縮低 Re 資訊
    """
    coeff = coeff.upper()
    if coeff not in {"CL", "CD"}:
        raise ValueError(f"coeff must be 'CL' or 'CD', got {coeff}")

    results_sorted = sorted(results, key=lambda row: row["re"])
    re_values = np.array([row["re"] for row in results_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_force_components(ax, re_values, results_sorted, coeff)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds Number  Re", fontsize=12)
    ax.set_ylabel(f"{coeff}", fontsize=12)
    ax.set_title(f"{coeff} vs Re — NACA0012, AOA={aoa_deg:.1f} deg, Ma={ma:.2f}", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_force_polar(
        results: list,
        sweep_key: str,
        sweep_label: str,
        ma: float,
        fixed_label: str,
        save_path: Optional[str] = None,
        show: bool = False):
    """
    繪製 CL-CD 極線圖

    What: 以 CD 為 x 軸、CL 為 y 軸，並在各點標註 sweep 參數值
    Why:  極線是翼型參數掃描的核心輸出，可直接讀取升阻性能變化
    """
    results_sorted = sorted(results, key=lambda row: row[sweep_key])
    cd_values = np.array([row["CD"] for row in results_sorted], dtype=float)
    cl_values = np.array([row["CL"] for row in results_sorted], dtype=float)
    sweep_values = np.array([row[sweep_key] for row in results_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cd_values, cl_values, "k-", lw=1.2, alpha=0.8)
    ax.scatter(cd_values, cl_values, c="tab:blue", s=42, zorder=3)

    for cd, cl, value in zip(cd_values, cl_values, sweep_values):
        if sweep_key == "re":
            label = f"{value:.0f}"
        else:
            label = f"{value:.1f}"
        ax.annotate(label, (cd, cl), textcoords="offset points", xytext=(5, 4), fontsize=9)

    ax.set_xlabel("Drag Coefficient  CD", fontsize=12)
    ax.set_ylabel("Lift Coefficient  CL", fontsize=12)
    ax.set_title(
        f"Lift-Drag Polar — NACA0012, Ma={ma:.2f}, {fixed_label}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.02,
        f"Annotated by {sweep_label}",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── GIF 動畫合成 ──────────────────────────────────────────────────────────────
def make_flow_gif(
        frame_dir: str,
        field: str = 'velocity',
        save_path: Optional[str] = None,
        fps: int = 10,
        duration_ms: Optional[int] = None,
):
    """
    將 frame_dir 中同一 field 的 PNG 幀合成為 GIF 動畫

    What: 搜集 {frame_dir}/{field}_step*.png，依步數排序後合成 GIF
    Why:  靜態圖難以觀察流場收斂過程；GIF 直觀展示壓力波、
          停滯點形成、尾流發展等暫態特徵

    Args:
        frame_dir:   存放幀圖的目錄
        field:       'velocity' | 'pressure' | 'density' | 'mach'
        save_path:   輸出 GIF 路徑；None → {frame_dir}/{field}.gif
        fps:         幀率（幀/秒）
        duration_ms: 每幀持續時間（毫秒）；指定後覆蓋 fps
    """
    import glob
    import os
    from PIL import Image

    pattern = os.path.join(frame_dir, f"{field}_step*.png")
    paths   = sorted(glob.glob(pattern),
                     key=lambda p: int(
                         os.path.splitext(os.path.basename(p))[0]
                         .split('step')[-1]))

    if not paths:
        print(f"  ⚠️  No frames found: {pattern}")
        return

    if save_path is None:
        save_path = os.path.join(frame_dir, f"{field}.gif")

    ms = duration_ms if duration_ms is not None else max(1, 1000 // fps)

    frames = [Image.open(p) for p in paths]
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=ms,
        loop=0,
    )
    print(f"  Saved GIF ({len(frames)} frames, {ms}ms/frame): {save_path}")
