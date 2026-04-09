"""
fvm_taichi.grid_gen — 結構化網格生成器
========================================

What: 為 EulerSolver 生成節點座標 (x_node, y_node)
Why:  翼型模擬需要曲線結構網格；此模組封裝幾何計算，
      與求解器邏輯分離

目前提供：
- generate_o_grid_naca0012 : NACA0012 代數 O-grid（可選近壁 clustering）
- generate_sheared_channel_grid : 仿射剪切通道網格（curvilinear benchmark）
- generate_bump_channel_grid : 下壁 bump channel 網格（transonic benchmark）
- generate_cd_nozzle_grid : 對稱 converging-diverging nozzle 網格
- generate_reference_nozzle_grid : 參考示意圖風格的內流 nozzle 網格
"""

import numpy as np


# ── NACA 4 位數翼型 ─────────────────────────────────────────────────────
def naca4_thickness(x: np.ndarray, max_t: float = 0.12) -> np.ndarray:
    """
    NACA 4 位數對稱翼型的厚度分佈（上半面 y/c）

    Args:
        x:     chord 座標，[0, 1]
        max_t: 最大厚度比（NACA0012 → 0.12）
    """
    x = np.maximum(x, 0.0)   # 避免 sqrt 負數
    return 5.0 * max_t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4   # 開放 TE；如需閉合 TE 改為 -0.1036
    )


def naca0012_surface(ni: int, max_t: float = 0.12):
    """
    生成 NACA 4 位數翼型表面節點（ni+1 個，逆時針繞行）

    參數化：θ ∈ [0, 2π]
      x(θ) = 0.5*(1 + cos θ) → θ=0,2π: x=1 (TE), θ=π: x=0 (LE)
      上表面 (0 < θ < π): y = +thickness(x)
      下表面 (π < θ < 2π): y = -thickness(x)
      LE/TE (θ=0,π,2π): y = 0

    Returns:
        x_s, y_s: shape (ni+1,) — 節點座標，x_s[0] = x_s[ni] = TE
    """
    theta = np.linspace(0, 2 * np.pi, ni + 1)
    x_s   = 0.5 * (1.0 + np.cos(theta))
    y_mag = naca4_thickness(x_s, max_t)

    # 以 sin(θ) 的符號決定上/下表面；TE/LE 強制 y=0
    y_s = np.where(
        (theta > 0.0) & (theta < np.pi),   y_mag,
        np.where(
        (theta > np.pi) & (theta < 2*np.pi), -y_mag,
        0.0))

    return x_s, y_s


def radial_geometric_distribution(nj: int, wall_spacing_ratio: float = 1.0) -> np.ndarray:
    """
    生成 0..1 的徑向節點分佈

    What: 以 cumulative geometric spacing 生成 j 方向參數 t_j
    Why:  高 Re / RANS 的第一層 wall distance 需要遠小於線性 O-grid；
          用單一 wall_spacing_ratio 就能可重現地控制近壁 clustering

    Args:
        nj: interior radial cells
        wall_spacing_ratio:
            第一層厚度相對於線性平均厚度 1/nj 的比例
            1.0 -> 線性分佈
            <1  -> 近壁加密

    Returns:
        t_j: shape (nj+1,), 從 0 到 1 單調遞增
    """
    if nj <= 0:
        raise ValueError(f"nj must be positive, got {nj}")
    if wall_spacing_ratio <= 0.0:
        raise ValueError(f"wall_spacing_ratio must be positive, got {wall_spacing_ratio}")
    if abs(wall_spacing_ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, nj + 1, dtype=np.float64)

    dt0 = wall_spacing_ratio / nj

    def series_sum(r: float) -> float:
        if abs(r - 1.0) < 1e-14:
            return dt0 * nj
        return dt0 * (r**nj - 1.0) / (r - 1.0)

    if wall_spacing_ratio < 1.0:
        lo, hi = 1.0, 2.0
        while series_sum(hi) < 1.0:
            hi *= 2.0
            if hi > 1.0e6:
                raise RuntimeError(
                    "Failed to construct geometric radial distribution: "
                    f"wall_spacing_ratio={wall_spacing_ratio} is too small for nj={nj}"
                )
    else:
        lo, hi = 1.0e-8, 1.0
        while series_sum(lo) > 1.0:
            lo *= 0.5
            if lo < 1.0e-16:
                raise RuntimeError(
                    "Failed to construct geometric radial distribution: "
                    f"wall_spacing_ratio={wall_spacing_ratio} is too large for nj={nj}"
                )

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        smid = series_sum(mid)
        if wall_spacing_ratio < 1.0:
            if smid < 1.0:
                lo = mid
            else:
                hi = mid
        else:
            if smid > 1.0:
                lo = mid
            else:
                hi = mid

    r = 0.5 * (lo + hi)
    dt = dt0 * r ** np.arange(nj, dtype=np.float64)
    t_j = np.concatenate([[0.0], np.cumsum(dt)])
    t_j[-1] = 1.0
    return t_j


# ── O-grid 生成器 ────────────────────────────────────────────────────────
def generate_o_grid_naca0012(
        ni: int,
        nj: int,
        R_far: float = 15.0,
        max_t: float = 0.12,
        NG:    int   = 2,
        wall_spacing_ratio: float = 1.0,
) -> tuple:
    """
    NACA0012 代數 O-grid（可選近壁 clustering）

    What: 從翼型表面到遠場圓弧，以代數插值連接，形成 O-grid 拓撲
    Why:  O-grid 在 i 方向封閉（周期性），j_min 為翼型表面（滑動壁），
          j_max 為遠場（Neumann），無需處理 C-grid 尾緣切割
    When: 搭配 EulerSolver.set_curvilinear_grid + set_slip_wall_j_min
          + set_periodic_bc(i_dir=True, j_dir=False)

    座標系：
      翼型弦長 = 1，前緣 x=0，尾緣 x=1，遠場圓心在 (0.5, 0)
      逆時針繞行：i=0 從 TE 上表面出發

    Args:
        ni:    i 方向 interior cell 數（繞翼型一圈的分割數，建議偶數）
        nj:    j 方向 interior cell 數（從翼型到遠場的層數）
        R_far: 遠場圓弧半徑（弦長倍數，建議 ≥ 10）
        max_t: 翼型最大厚度比（NACA0012 → 0.12）
        NG:    ghost 層數（必須與 EulerSolver.NG 一致，預設 2）
        wall_spacing_ratio:
            第一層徑向厚度相對於線性平均厚度 1/nj 的比例
            1.0 -> 線性分佈；<1 -> 近壁加密

    Returns:
        x_node, y_node: shape (NI+1, NJ+1) 節點座標，float32
                        NI = ni + 2*NG, NJ = nj + 2*NG
    """
    NI = ni + 2 * NG
    NJ = nj + 2 * NG

    # ── 1. 翼型表面節點（ni+1 個逆時針節點）──────────────────────
    x_air, y_air = naca0012_surface(ni, max_t)

    # ── 2. 遠場圓弧節點（以翼型法向角對齊）──────────────────────
    xc, yc = 0.5, 0.0   # 遠場圓心（mid-chord）
    ang = np.arctan2(y_air - yc, x_air - xc)
    x_out = xc + R_far * np.cos(ang)
    y_out = yc + R_far * np.sin(ang)

    # ── 3. j 方向代數插值（ni+1 條射線，nj+1 層）─────────────────
    t_j = radial_geometric_distribution(nj, wall_spacing_ratio=wall_spacing_ratio)
    # x_phy[k, jl] = x_air[k] + t_j[jl] * (x_out[k] - x_air[k])
    x_phy = x_air[:, None] + t_j[None, :] * (x_out - x_air)[:, None]
    y_phy = y_air[:, None] + t_j[None, :] * (y_out - y_air)[:, None]
    # x_phy: shape (ni+1, nj+1) — interior node positions

    # ── 4. 全域節點陣列（含 ghost 節點）───────────────────────────
    x_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)
    y_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)

    # 4a. 放入 interior 節點 [NG:NG+ni+1, NG:NG+nj+1]
    x_node[NG:NG + ni + 1, NG:NG + nj + 1] = x_phy
    y_node[NG:NG + ni + 1, NG:NG + nj + 1] = y_phy

    # 4b. j_min ghost（往翼型內線性外插）
    #     node[NG-1-g] = 2*node[NG+g] - node[NG+g+1]
    for g in range(NG):
        jg  = NG - 1 - g
        j1  = NG + g
        j2  = NG + g + 1
        x_node[:, jg] = 2.0 * x_node[:, j1] - x_node[:, j2]
        y_node[:, jg] = 2.0 * y_node[:, j1] - y_node[:, j2]

    # 4c. j_max ghost（往遠場線性外插）
    for g in range(NG):
        jg = NG + nj + 1 + g
        j1 = NG + nj - g
        j2 = NG + nj - g - 1
        x_node[:, jg] = 2.0 * x_node[:, j1] - x_node[:, j2]
        y_node[:, jg] = 2.0 * y_node[:, j1] - y_node[:, j2]

    # 4d. i ghost（O-grid 周期繞行）
    #     ghost_left[g]     ← interior[ni+g]     (g=0: ni, g=1: ni+1)
    #     ghost_right[NG+ni+1+g] ← interior[NG+1+g] (g=0: NG+1, g=1: NG+2)
    for g in range(NG):
        x_node[g, :]              = x_node[ni + g, :]
        y_node[g, :]              = y_node[ni + g, :]
        x_node[NG + ni + 1 + g, :] = x_node[NG + 1 + g, :]
        y_node[NG + ni + 1 + g, :] = y_node[NG + 1 + g, :]

    return x_node.astype(np.float32), y_node.astype(np.float32)


def generate_sheared_channel_grid(
        ni: int,
        nj: int,
        length: float = 1.0,
        height: float = 1.0,
        shear: float = 0.2,
        NG: int = 2,
) -> tuple:
    """
    生成仿射剪切通道網格（適合 curvilinear NS benchmark）

    What: 由直角 computational grid 經 affine shear
          x = ξ + shear * η, y = η 映射到 physical plane
    Why:  它保留解析 Couette / Poiseuille 解的簡潔性，但幾何已非 Cartesian，
          很適合驗證 curvilinear viscous flux 是否正確
    When: 用於 NavierStokesSolver 的 skewed-channel 測試與範例

    Args:
        ni, nj: interior cells
        length, height: 通道長與高
        shear: x 方向剪切量；0 -> Cartesian
        NG: ghost layers

    Returns:
        x_node, y_node: shape (NI+1, NJ+1) 節點座標，float32
    """
    NI = ni + 2 * NG
    NJ = nj + 2 * NG

    xi = (np.arange(NI + 1, dtype=np.float64) - NG) * (length / ni)
    eta = (np.arange(NJ + 1, dtype=np.float64) - NG) * (height / nj)
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")

    x_node = XI + shear * (ETA / max(height, 1e-12))
    y_node = ETA

    return x_node.astype(np.float32), y_node.astype(np.float32)


def generate_bump_channel_grid(
        ni: int,
        nj: int,
        length: float = 3.0,
        height: float = 1.0,
        bump_center: float = 1.5,
        bump_width: float = 1.0,
        bump_height: float = 0.08,
        NG: int = 2,
) -> tuple:
    """
    生成下壁含 bump 的通道曲線網格

    What: 通道上壁保持平直，下壁以 cosine bump 抬升，再沿 j 方向做代數插值
    Why:  transonic bump channel 是經典的 shock-capturing 驗證幾何；
          與 wedge 相比，不需要額外的 supersonic inflow/outflow BC 就能先觀察 shock-like 壓縮結構
    When: 搭配 EulerSolver.set_curvilinear_grid + 上下 slip wall，用於 transonic benchmark

    Args:
        ni, nj: interior cells
        length, height: 通道長與高
        bump_center: bump 中心 x 位置
        bump_width: bump 支撐寬度
        bump_height: bump 最大高度
        NG: ghost layers

    Returns:
        x_node, y_node: shape (NI+1, NJ+1) 節點座標，float32
    """
    if length <= 0.0 or height <= 0.0:
        raise ValueError(f"length and height must be positive, got {length}, {height}")
    if bump_width <= 0.0:
        raise ValueError(f"bump_width must be positive, got {bump_width}")
    if bump_height < 0.0:
        raise ValueError(f"bump_height must be non-negative, got {bump_height}")
    if bump_height >= 0.95 * height:
        raise ValueError(
            f"bump_height={bump_height} is too large for channel height={height}; "
            "require bump_height < 0.95 * height"
        )

    NI = ni + 2 * NG
    NJ = nj + 2 * NG

    xi = (np.arange(NI + 1, dtype=np.float64) - NG) * (length / ni)
    eta = (np.arange(NJ + 1, dtype=np.float64) - NG) / max(nj, 1)

    half_width = 0.5 * bump_width
    x_local = (xi - bump_center) / max(half_width, 1e-12)
    bump_mask = np.abs(x_local) <= 1.0
    y_bottom = np.zeros_like(xi)
    y_bottom[bump_mask] = 0.5 * bump_height * (1.0 + np.cos(np.pi * x_local[bump_mask]))
    y_top = np.full_like(xi, height)

    x_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)
    y_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)

    for i in range(NI + 1):
        channel_height = y_top[i] - y_bottom[i]
        x_node[i, :] = xi[i]
        y_node[i, :] = y_bottom[i] + eta * channel_height

    return x_node.astype(np.float32), y_node.astype(np.float32)


def generate_cd_nozzle_grid(
        ni: int,
        nj: int,
        length: float = 3.0,
        h_inlet: float = 1.0,
        h_throat: float = 0.40,
        h_exit: float | None = None,
        throat_x: float | None = None,
        conv_power: float = 2.4,
        div_power: float = 3.0,
        NG: int = 2,
) -> tuple:
    """
    生成對稱 converging-diverging nozzle 曲線網格

    What: 上下壁以同一條面積函數對稱收縮/擴張，j 方向做代數插值
    Why:  nozzle 是可壓縮流經典 benchmark，適合檢查亞音速與超音速路徑
    When: 搭配 EulerSolver.set_curvilinear_grid + 上下 slip wall

    Geometry:
      x ∈ [0, L]，喉部在 x = x_t
      x <= x_t:
          h(x) = h_t + (h_in - h_t) * ((x_t - x)/x_t)^n_conv
      x > x_t:
          h(x) = h_t + (h_out - h_t) * ((x - x_t)/(L - x_t))^n_div
      y_bottom = -0.5 h(x), y_top = +0.5 h(x)

    Args:
        ni, nj: interior cells
        length: 噴嘴長度 L
        h_inlet: 入口/出口通道高度（對稱設計）
        h_throat: 喉部最小高度
        h_exit: 出口高度；None -> h_inlet
        throat_x: 喉部位置；None -> length * 0.5
        conv_power: 收縮段曲率次方（越大代表前段收縮更快）
        div_power: 擴張段曲率次方（>1 代表喉後擴張更慢）
        NG: ghost layers

    Returns:
        x_node, y_node: shape (NI+1, NJ+1) 節點座標，float32
    """
    if ni <= 0 or nj <= 0:
        raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if h_inlet <= 0.0 or h_throat <= 0.0:
        raise ValueError(f"h_inlet and h_throat must be positive, got {h_inlet}, {h_throat}")
    if h_throat >= h_inlet:
        raise ValueError(
            f"h_throat must be smaller than h_inlet for converging-diverging nozzle, got "
            f"h_throat={h_throat}, h_inlet={h_inlet}"
        )
    if h_exit is None:
        h_exit = h_inlet
    if h_exit <= 0.0:
        raise ValueError(f"h_exit must be positive, got {h_exit}")
    if conv_power < 1.0 or div_power < 1.0:
        raise ValueError(
            f"conv_power and div_power must be >= 1.0, got conv_power={conv_power}, div_power={div_power}"
        )
    if h_throat >= h_exit:
        raise ValueError(
            f"h_throat must be smaller than h_exit for diverging section, got "
            f"h_throat={h_throat}, h_exit={h_exit}"
        )

    NI = ni + 2 * NG
    NJ = nj + 2 * NG

    if throat_x is None:
        throat_x = 0.5 * length
    if not (0.0 < throat_x < length):
        raise ValueError(f"throat_x must satisfy 0 < throat_x < length, got {throat_x}")

    xi = (np.arange(NI + 1, dtype=np.float64) - NG) * (length / ni)
    eta = (np.arange(NJ + 1, dtype=np.float64) - NG) / max(nj, 1)

    x_clip = np.clip(xi, 0.0, length)
    h_x = np.zeros_like(x_clip)

    mask_conv = x_clip <= throat_x
    x_conv = x_clip[mask_conv]
    ratio_conv = ((throat_x - x_conv) / max(throat_x, 1e-12)) ** conv_power
    h_x[mask_conv] = h_throat + (h_inlet - h_throat) * ratio_conv

    mask_div = ~mask_conv
    x_div = x_clip[mask_div]
    ratio_div = ((x_div - throat_x) / max(length - throat_x, 1e-12)) ** div_power
    h_x[mask_div] = h_throat + (h_exit - h_throat) * ratio_div

    y_bottom = -0.5 * h_x
    y_top = 0.5 * h_x

    x_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)
    y_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)

    for i in range(NI + 1):
        channel_height = y_top[i] - y_bottom[i]
        x_node[i, :] = xi[i]
        y_node[i, :] = y_bottom[i] + eta * channel_height

    return x_node.astype(np.float32), y_node.astype(np.float32)


def generate_reference_nozzle_grid(
        ni: int,
        nj: int,
        length: float = 3.0,
        h_inlet: float = 1.0,
        h_throat: float = 0.30,
        h_exit: float | None = None,
        throat_x: float | None = None,
        conv_power: float = 3.0,
        div_power: float = 3.8,
        throat_blend: float = 0.35,
        NG: int = 2,
) -> tuple:
    """
    生成「示意圖風格」converging-diverging nozzle 內部網格

    What: 以分段解析式定義通道高度 h(x)，前段快速收縮、後段慢速擴張
    Why:  標準對稱 power profile 常會呈現過度平滑或喉部過寬，不符合教科書示意
    When: 需要更接近 reference sketch 的幾何外形時

    Profile:
      x <= x_t:
          s = x/x_t
          f0 = 1 - (1-s)^a_c
          f  = β s + (1-β) f0    (β > 0 可避免喉部過平，縮短 throat)
          h = h_in - (h_in-h_t) f
      x > x_t:
          s = (x-x_t)/(L-x_t)
          f0 = s^a_d
          f  = β s + (1-β) f0    (保留慢擴張同時減少喉部拖長)
          h = h_t + (h_out-h_t) f
    """
    if ni <= 0 or nj <= 0:
        raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if h_inlet <= 0.0 or h_throat <= 0.0:
        raise ValueError(f"h_inlet and h_throat must be positive, got {h_inlet}, {h_throat}")
    if h_exit is None:
        h_exit = h_inlet
    if h_exit <= 0.0:
        raise ValueError(f"h_exit must be positive, got {h_exit}")
    if h_throat >= min(h_inlet, h_exit):
        raise ValueError(
            f"h_throat must be smaller than inlet/exit heights, got "
            f"h_throat={h_throat}, h_inlet={h_inlet}, h_exit={h_exit}"
        )
    if conv_power <= 1.0 or div_power <= 1.0:
        raise ValueError(
            f"conv_power and div_power must be > 1.0, got conv_power={conv_power}, div_power={div_power}"
        )
    if not (0.0 <= throat_blend <= 1.0):
        raise ValueError(f"throat_blend must satisfy 0 <= throat_blend <= 1, got {throat_blend}")

    NI = ni + 2 * NG
    NJ = nj + 2 * NG

    if throat_x is None:
        throat_x = 0.35 * length
    if not (0.0 < throat_x < length):
        raise ValueError(f"throat_x must satisfy 0 < throat_x < length, got {throat_x}")

    xi = (np.arange(NI + 1, dtype=np.float64) - NG) * (length / ni)
    eta = (np.arange(NJ + 1, dtype=np.float64) - NG) / max(nj, 1)
    x_clip = np.clip(xi, 0.0, length)
    h_x = np.zeros_like(x_clip)

    mask_conv = x_clip <= throat_x
    x_conv = x_clip[mask_conv]
    s_conv = x_conv / max(throat_x, 1e-12)
    f_conv_base = 1.0 - np.power(1.0 - s_conv, conv_power)
    f_conv = throat_blend * s_conv + (1.0 - throat_blend) * f_conv_base
    h_x[mask_conv] = h_inlet - (h_inlet - h_throat) * f_conv

    mask_div = ~mask_conv
    x_div = x_clip[mask_div]
    s_div = (x_div - throat_x) / max(length - throat_x, 1e-12)
    f_div_base = np.power(s_div, div_power)
    f_div = throat_blend * s_div + (1.0 - throat_blend) * f_div_base
    h_x[mask_div] = h_throat + (h_exit - h_throat) * f_div

    y_bottom = -0.5 * h_x
    y_top = 0.5 * h_x

    x_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)
    y_node = np.zeros((NI + 1, NJ + 1), dtype=np.float64)

    for i in range(NI + 1):
        channel_height = y_top[i] - y_bottom[i]
        x_node[i, :] = xi[i]
        y_node[i, :] = y_bottom[i] + eta * channel_height

    return x_node.astype(np.float32), y_node.astype(np.float32)
