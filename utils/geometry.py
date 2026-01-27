"""
幾何生成工具
============

提供高品質、具備美感的機翼系統幾何。
修正：採用穩健的參數化包絡法生成 Nike Swoosh 縫翼，並修正縮放與姿態。
"""

import numpy as np
from typing import Tuple, Optional, List


def generate_naca(naca_code: str, num_points: int = 500) -> np.ndarray:
    """生成 NACA 4 位數翼型座標"""
    m, p, t = (
        int(naca_code[0]) / 100.0,
        int(naca_code[1]) / 10.0,
        int(naca_code[2:]) / 100.0,
    )
    beta = np.linspace(0, np.pi, num_points // 2)
    x = (1 - np.cos(beta)) / 2
    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    if m == 0 or p == 0:
        yc, dyc_dx = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        dyc_dx = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc_dx)
    xu, yu, xl, yl = (
        x - yt * np.sin(theta),
        yc + yt * np.cos(theta),
        x + yt * np.sin(theta),
        yc - yt * np.cos(theta),
    )
    return np.concatenate(
        [np.stack([xu[::-1], yu[::-1]], axis=1), np.stack([xl[1:], yl[1:]], axis=1)]
    )


def generate_nike_slat(num_points: int = 200) -> np.ndarray:
    """
    生成 Nike Swoosh 造型縫翼（強化曲率版）

    設計邏輯：圓頭在前上方，尾部向下彎曲包覆主翼前緣。
    根據真實高升力系統照片調整曲率與厚度分布。
    """
    t = np.linspace(0, 1, num_points // 2)

    # 1. 定義超強弧線彎曲
    y = -t  # 從 0 到 -1 (由上而下)

    # 極限增強 x 方向彎曲：產生更明顯的深 C 型勾
    # 使用三次項增強尾部彎曲
    x_camber = 1.6 * y * (1 + y) - 0.9 * y**2 - 0.6 * y**3

    # 2. 改進厚度分布：頭部更圓潤，尾部更尖
    thick = 0.35 * np.sqrt(-y + 0.01) * (1 + y) ** 1.5

    # 3. 生成內外邊界
    xi = x_camber + thick / 2  # 內側（靠近主翼）
    xo = x_camber - thick / 2  # 外側

    # 4. 組合為閉合路徑
    top_to_bottom = np.stack([xo, y], axis=1)
    bottom_to_top = np.stack([xi[::-1], y[::-1]], axis=1)
    coords = np.concatenate([top_to_bottom, bottom_to_top])

    # 5. 歸一化
    coords[:, 0] -= np.min(coords[:, 0])
    coords[:, 1] -= np.max(coords[:, 1])
    dx = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    dy = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    max_dim = dx if dx >= dy else dy
    coords /= max_dim

    return coords


def rotate_coords(
    coords: np.ndarray, angle_deg: float, pivot: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    return (coords - pivot) @ rot.T + pivot


def create_airfoil_system(
    nx: int,
    ny: int,
    main_naca: str = "2412",
    main_chord: float = 100.0,
    aoa: float = 0.0,
    slat_angle: float = 15.0,
    flap_angle: float = 30.0,
    center_position: Optional[Tuple[float, float] | np.ndarray] = None,
    return_sdf: bool = False,
) -> np.ndarray:
    """
    生成機翼系統幾何 - 採用兩階段旋轉策略

    階段1：整機攻角 aoa（所有部件以主翼中心一起旋轉）
    階段2：高升力裝置偏轉（各部件繞自身鉸鏈點旋轉）
        - 縫翼：繞主翼前緣旋轉 slat_angle（向上）
        - 襟翼：繞主翼後緣旋轉 flap_angle（向下）
    """
    if center_position is None:
        center = np.array([nx / 4, ny / 2])
    else:
        center = np.array(center_position)

    # === 階段 0：生成局部坐標系中的幾何（水平狀態）===
    main_coords = generate_naca(main_naca) * main_chord
    main_center_local = np.array([0.5 * main_chord, 0.0])
    leading_edge_local = np.array([0.0, 0.0])
    trailing_edge_local = np.array([main_chord, 0.0])

    # 縫翼（預旋轉調整 Nike Swoosh 姿態）
    slat_c = 0.14 * main_chord
    slat_raw = generate_nike_slat() * slat_c
    slat_shaped = rotate_coords(slat_raw, -10, pivot=np.array([0, 0]))
    slat_offset = np.array([-0.07 * main_chord, 0.08 * main_chord])
    slat_local = slat_shaped + slat_offset

    # 襟翼
    flap_c = 0.35 * main_chord
    flap_raw = generate_naca("4415") * flap_c
    flap_offset = np.array([0.96 * main_chord, -0.06 * main_chord])
    flap_local = flap_raw + flap_offset

    # === 階段 1：整機攻角（所有部件以主翼中心旋轉 aoa）===
    main_rotated = rotate_coords(main_coords, -aoa, pivot=main_center_local)
    slat_rotated = rotate_coords(slat_local, -aoa, pivot=main_center_local)
    flap_rotated = rotate_coords(flap_local, -aoa, pivot=main_center_local)

    # 計算旋轉後的鉸鏈點位置
    leading_edge_rotated = rotate_coords(
        np.array([leading_edge_local]), -aoa, pivot=main_center_local
    )[0]
    trailing_edge_rotated = rotate_coords(
        np.array([trailing_edge_local]), -aoa, pivot=main_center_local
    )[0]

    # === 階段 2：高升力裝置偏轉（繞鉸鏈點旋轉）===
    # 主翼：無額外旋轉
    main_final = main_rotated + center

    # 縫翼：繞前緣向上偏轉（圖像坐標系中，向上打開 = 逆時針 = 正角度）
    slat_final = (
        rotate_coords(slat_rotated, +slat_angle, pivot=leading_edge_rotated) + center
    )

    # 襟翼：繞後緣向下偏轉（圖像坐標系中，向下打開 = 順時針 = 正角度）
    flap_final = (
        rotate_coords(flap_rotated, +flap_angle, pivot=trailing_edge_rotated) + center
    )

    # 5. 生成 Mask
    from matplotlib.path import Path

    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    points = np.stack([x.ravel(), y.ravel()], axis=1)

    mask = np.zeros((nx, ny), dtype=np.int32)
    parts = [main_final, slat_final, flap_final]
    for coords in parts:
        p = Path(coords)
        mask = np.maximum(
            mask, p.contains_points(points).reshape(nx, ny).astype(np.int32)
        )

    if return_sdf:
        sdf = _signed_distance_from_polygons(mask, parts, nx, ny)
        return mask, sdf

    return mask


def apply_leading_edge_droop(
    coords: np.ndarray, droop_fraction: float, droop_amount: float
) -> np.ndarray:
    """
    對前緣做下垂變形（Leading-edge droop）

    What:
    - 針對 x <= droop_fraction 的區段做平滑下壓

    Why:
    - 增加前緣曲率與有效迎角
    - 延後前緣分離，提高失速迎角
    """
    adjusted = coords.copy()
    x = adjusted[:, 0]
    mask = x <= droop_fraction
    if np.any(mask):
        denom = max(float(droop_fraction), 1e-6)
        s = 1.0 - x[mask] / denom
        adjusted[mask, 1] -= droop_amount * (s**2)
    return adjusted


def apply_leading_edge_wave(
    coords: np.ndarray, wave_fraction: float, amplitude: float
) -> np.ndarray:
    """
    前緣波紋擾動（仿生結節）

    What:
    - 針對前緣區域加入正弦擾動

    Why:
    - 促進局部渦結構，延緩分離
    - 提升高迎角下的升力穩定性
    """
    adjusted = coords.copy()
    x = adjusted[:, 0]
    y = adjusted[:, 1]
    mask = x <= wave_fraction
    if np.any(mask):
        denom = max(float(wave_fraction), 1e-6)
        phase = np.pi * x[mask] / denom
        sign = np.sign(y[mask])
        sign[sign == 0] = 1.0
        adjusted[mask, 1] = y[mask] + sign * amplitude * np.sin(phase)
    return adjusted


def create_coanda_flap(
    trailing_edge: np.ndarray,
    radius: float,
    arc_angle_deg: float = 120.0,
    num_points: int = 120,
    tangent_angle_deg: float = 0.0,
) -> np.ndarray:
    """
    生成 Coanda 型曲面襟翼

    What:
    - 在後緣下方建立圓弧曲面

    Why:
    - 曲面延長附著流，提供循環增益
    """
    arc = np.radians(arc_angle_deg)
    tangent_angle = np.radians(tangent_angle_deg)
    theta_start = tangent_angle - np.pi / 2.0
    center = trailing_edge - radius * np.array(
        [np.cos(theta_start), np.sin(theta_start)]
    )
    theta_end = theta_start - arc
    theta = np.linspace(theta_start, theta_end, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.stack([x, y], axis=1)


def create_vortex_generators(
    base_points: np.ndarray, height: float, width: float
) -> List[np.ndarray]:
    """
    生成小型三角渦產生器

    What:
    - 在指定位置建立小型三角片

    Why:
    - 提升邊界層能量，降低局部分離
    """
    generators = []
    for base in base_points:
        x0, y0 = base
        generators.append(
            np.array(
                [
                    [x0 - 0.5 * width, y0],
                    [x0 + 0.5 * width, y0],
                    [x0, y0 + height],
                ]
            )
        )
    return generators


def merge_tail_surface(
    flap_coords: np.ndarray, coanda_coords: np.ndarray
) -> np.ndarray:
    """
    合併襟翼與 Coanda 曲面為單一連續曲面

    What:
    - 以襟翼前半段為主體，尾緣接上 Coanda 曲面

    Why:
    - 消除幾何斷裂，形成單一連續曲率尾緣
    """
    max_x = np.max(flap_coords[:, 0])
    te_indices = np.where(np.isclose(flap_coords[:, 0], max_x, atol=1e-6))[0]
    if te_indices.size < 2:
        upper_te = int(np.argmax(flap_coords[:, 0]))
        lower_te = upper_te
    else:
        upper_te = int(te_indices[0])
        lower_te = int(te_indices[-1])

    upper_surface = flap_coords[: upper_te + 1]
    lower_surface = flap_coords[lower_te:]

    return np.concatenate([upper_surface, coanda_coords, lower_surface])


def create_stall_resistant_airfoil_system(
    nx: int,
    ny: int,
    main_naca: str = "4415",
    main_chord: float = 100.0,
    aoa: float = 0.0,
    slat_angle: float = 18.0,
    flap_angle: float = 32.0,
    flap2_angle: float = 12.0,
    droop_fraction: float = 0.18,
    droop_amount_ratio: float = 0.04,
    center_position: Optional[Tuple[float, float] | np.ndarray] = None,
) -> np.ndarray:
    """
    生成失速風險抑制型機翼系統（可變弧度 + 雙段襟翼）

    What:
    - 前緣下垂主翼 + 縫翼 + 雙段襟翼 + Gurney tab

    Why:
    - 前緣下垂可推遲前緣分離
    - 雙段襟翼增加高升力配置的可控性
    - Gurney tab 強化循環增益，提升低速升力
    """
    if center_position is None:
        center = np.array([nx / 4, ny / 2])
    else:
        center = np.array(center_position)

    droop_amount = droop_amount_ratio * main_chord

    # === 階段 0：生成局部坐標系中的幾何（水平狀態）===
    main_coords = generate_naca(main_naca) * main_chord
    main_coords = apply_leading_edge_droop(
        main_coords, droop_fraction * main_chord, droop_amount
    )

    main_center_local = np.array([0.5 * main_chord, 0.0])
    leading_edge_local = np.array([0.0, 0.0])
    trailing_edge_local = np.array([main_chord, 0.0])

    # 縫翼：縮小弦長 + 提高前緣拱度
    slat_c = 0.12 * main_chord
    slat_raw = generate_nike_slat() * slat_c
    slat_shaped = rotate_coords(slat_raw, -12, pivot=np.array([0.0, 0.0]))
    slat_offset = np.array([-0.08 * main_chord, 0.1 * main_chord])
    slat_local = slat_shaped + slat_offset

    # 襟翼 1
    flap1_c = 0.28 * main_chord
    flap1_raw = generate_naca("4415") * flap1_c
    flap1_offset = np.array([0.92 * main_chord, -0.04 * main_chord])
    flap1_local = flap1_raw + flap1_offset

    # 襟翼 2（雙段襟翼）
    flap2_c = 0.18 * main_chord
    flap2_raw = generate_naca("4412") * flap2_c
    flap1_te_local = flap1_offset + np.array([flap1_c, 0.0])
    flap2_offset = flap1_te_local + np.array([0.07 * main_chord, -0.06 * main_chord])
    flap2_local = flap2_raw + flap2_offset

    # Gurney tab
    gurney_length = 0.03 * main_chord
    gurney_height = 0.015 * main_chord
    flap2_te_local = flap2_offset + np.array([flap2_c, 0.0])
    gurney_local = np.array(
        [
            [flap2_te_local[0], flap2_te_local[1]],
            [flap2_te_local[0] + gurney_length, flap2_te_local[1]],
            [flap2_te_local[0] + gurney_length, flap2_te_local[1] - gurney_height],
            [flap2_te_local[0], flap2_te_local[1] - gurney_height],
        ]
    )

    # === 階段 1：整機攻角 ===
    main_rotated = rotate_coords(main_coords, -aoa, pivot=main_center_local)
    slat_rotated = rotate_coords(slat_local, -aoa, pivot=main_center_local)
    flap1_rotated = rotate_coords(flap1_local, -aoa, pivot=main_center_local)
    flap2_rotated = rotate_coords(flap2_local, -aoa, pivot=main_center_local)
    gurney_rotated = rotate_coords(gurney_local, -aoa, pivot=main_center_local)

    leading_edge_rotated = rotate_coords(
        np.array([leading_edge_local]), -aoa, pivot=main_center_local
    )[0]
    trailing_edge_rotated = rotate_coords(
        np.array([trailing_edge_local]), -aoa, pivot=main_center_local
    )[0]

    flap1_te_rotated = rotate_coords(
        np.array([flap1_te_local]), -aoa, pivot=main_center_local
    )[0]

    # === 階段 2：高升力裝置偏轉 ===
    main_final = main_rotated + center

    slat_final = (
        rotate_coords(slat_rotated, +slat_angle, pivot=leading_edge_rotated) + center
    )

    flap1_final = (
        rotate_coords(flap1_rotated, +flap_angle, pivot=trailing_edge_rotated) + center
    )

    flap2_intermediate = rotate_coords(
        flap2_rotated, +flap_angle, pivot=trailing_edge_rotated
    )

    gurney_intermediate = rotate_coords(
        gurney_rotated, +flap_angle, pivot=trailing_edge_rotated
    )

    flap1_te_deflected = rotate_coords(
        np.array([flap1_te_rotated]), +flap_angle, pivot=trailing_edge_rotated
    )[0]

    flap2_final = (
        rotate_coords(flap2_intermediate, +flap2_angle, pivot=flap1_te_deflected)
        + center
    )

    gurney_final = (
        rotate_coords(gurney_intermediate, +flap2_angle, pivot=flap1_te_deflected)
        + center
    )

    # === 生成 Mask ===
    from matplotlib.path import Path

    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    points = np.stack([x.ravel(), y.ravel()], axis=1)

    mask = np.zeros((nx, ny), dtype=np.int32)
    for coords in [main_final, slat_final, flap1_final, flap2_final, gurney_final]:
        p = Path(coords)
        mask = np.maximum(
            mask, p.contains_points(points).reshape(nx, ny).astype(np.int32)
        )

    return mask


def create_bioinspired_airfoil_system(
    nx: int,
    ny: int,
    main_naca: str = "5418",
    main_chord: float = 100.0,
    aoa: float = 0.0,
    slat_angle: float = 20.0,
    flap_angle: float = 28.0,
    flap2_angle: float = 11.0,
    droop_fraction: float = 0.24,
    droop_amount_ratio: float = 0.06,
    wave_fraction: float = 0.16,
    wave_amplitude_ratio: float = 0.014,
    coanda_radius_ratio: float = 0.06,
    center_position: Optional[Tuple[float, float] | np.ndarray] = None,
) -> np.ndarray:
    """
    生成仿生失速抑制機翼（多尺度流動控制）

    What:
    - 前緣下垂 + 波紋擾動 + 縫翼 + 雙段襟翼 + Gurney + Coanda 翼型

    Why:
    - 多尺度擾動與循環控制同時降低失速風險並提升升力
    """
    if center_position is None:
        center = np.array([nx / 4, ny / 2])
    else:
        center = np.array(center_position)

    droop_amount = droop_amount_ratio * main_chord
    wave_amplitude = wave_amplitude_ratio * main_chord
    coanda_radius = coanda_radius_ratio * main_chord

    main_coords = generate_naca(main_naca) * main_chord
    main_coords = apply_leading_edge_droop(
        main_coords, droop_fraction * main_chord, droop_amount
    )
    main_coords = apply_leading_edge_wave(
        main_coords, wave_fraction * main_chord, wave_amplitude
    )

    main_center_local = np.array([0.5 * main_chord, 0.0])
    leading_edge_local = np.array([0.0, 0.0])
    trailing_edge_local = np.array([main_chord, 0.0])

    slat_c = 0.14 * main_chord
    slat_raw = generate_nike_slat() * slat_c
    slat_shaped = rotate_coords(slat_raw, -12, pivot=np.array([0.0, 0.0]))
    slat_offset = np.array([-0.03 * main_chord, 0.06 * main_chord])
    slat_local = slat_shaped + slat_offset

    flap1_c = 0.31 * main_chord
    flap1_raw = generate_naca("4415") * flap1_c
    flap1_offset = np.array([0.915 * main_chord, -0.015 * main_chord])
    flap1_local = flap1_raw + flap1_offset

    flap2_c = 0.21 * main_chord
    flap2_raw = generate_naca("4412") * flap2_c
    flap1_te_local = flap1_offset + np.array([flap1_c, 0.0])
    flap2_offset = flap1_te_local + np.array([0.005 * main_chord, -0.015 * main_chord])
    flap2_local = flap2_raw + flap2_offset

    gurney_length = 0.028 * main_chord
    gurney_height = 0.012 * main_chord
    flap2_te_local = flap2_offset + np.array([flap2_c, 0.0])
    gurney_local = np.array(
        [
            [flap2_te_local[0], flap2_te_local[1]],
            [flap2_te_local[0] + gurney_length, flap2_te_local[1]],
            [flap2_te_local[0] + gurney_length, flap2_te_local[1] - gurney_height],
            [flap2_te_local[0], flap2_te_local[1] - gurney_height],
        ]
    )

    coanda_anchor = flap2_te_local + np.array(
        [-0.004 * main_chord, -0.002 * main_chord]
    )
    coanda_local = create_coanda_flap(
        coanda_anchor,
        coanda_radius * 0.85,
        arc_angle_deg=120.0,
        tangent_angle_deg=-8.0,
    )

    vg_height = 0.028 * main_chord
    vg_width = 0.035 * main_chord
    vg_bases = np.array(
        [
            [0.26 * main_chord, 0.03 * main_chord],
            [0.36 * main_chord, 0.035 * main_chord],
            [0.46 * main_chord, 0.04 * main_chord],
        ]
    )
    vg_local_list = create_vortex_generators(vg_bases, vg_height, vg_width)

    main_rotated = rotate_coords(main_coords, -aoa, pivot=main_center_local)
    slat_rotated = rotate_coords(slat_local, -aoa, pivot=main_center_local)
    flap1_rotated = rotate_coords(flap1_local, -aoa, pivot=main_center_local)
    flap2_rotated = rotate_coords(flap2_local, -aoa, pivot=main_center_local)
    gurney_rotated = rotate_coords(gurney_local, -aoa, pivot=main_center_local)
    coanda_rotated = rotate_coords(coanda_local, -aoa, pivot=main_center_local)
    vg_rotated = [
        rotate_coords(vg, -aoa, pivot=main_center_local) for vg in vg_local_list
    ]

    leading_edge_rotated = rotate_coords(
        np.array([leading_edge_local]), -aoa, pivot=main_center_local
    )[0]
    trailing_edge_rotated = rotate_coords(
        np.array([trailing_edge_local]), -aoa, pivot=main_center_local
    )[0]

    flap1_te_rotated = rotate_coords(
        np.array([flap1_te_local]), -aoa, pivot=main_center_local
    )[0]

    main_final = main_rotated + center
    slat_final = (
        rotate_coords(slat_rotated, +slat_angle, pivot=leading_edge_rotated) + center
    )
    flap1_final = (
        rotate_coords(flap1_rotated, +flap_angle, pivot=trailing_edge_rotated) + center
    )

    flap2_intermediate = rotate_coords(
        flap2_rotated, +flap_angle, pivot=trailing_edge_rotated
    )
    gurney_intermediate = rotate_coords(
        gurney_rotated, +flap_angle, pivot=trailing_edge_rotated
    )
    coanda_intermediate = rotate_coords(
        coanda_rotated, +flap_angle, pivot=trailing_edge_rotated
    )

    flap1_te_deflected = rotate_coords(
        np.array([flap1_te_rotated]), +flap_angle, pivot=trailing_edge_rotated
    )[0]

    flap2_final = (
        rotate_coords(flap2_intermediate, +flap2_angle, pivot=flap1_te_deflected)
        + center
    )
    gurney_final = (
        rotate_coords(gurney_intermediate, +flap2_angle, pivot=flap1_te_deflected)
        + center
    )
    coanda_final = (
        rotate_coords(coanda_intermediate, +flap2_angle, pivot=flap1_te_deflected)
        + center
    )

    tail_surface_final = merge_tail_surface(flap2_final, coanda_final)

    vg_final = [vg + center for vg in vg_rotated]

    from matplotlib.path import Path

    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    points = np.stack([x.ravel(), y.ravel()], axis=1)

    mask = np.zeros((nx, ny), dtype=np.int32)
    parts = [
        main_final,
        slat_final,
        flap1_final,
        flap2_final,
        gurney_final,
        tail_surface_final,
    ] + vg_final

    for coords in parts:
        p = Path(coords)
        mask = np.maximum(
            mask, p.contains_points(points).reshape(nx, ny).astype(np.int32)
        )

    return mask


def create_circle_mask(nx, ny, center, radius):
    cx, cy = center
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.int32)


def create_circle_mask_and_sdf(nx, ny, center, radius):
    cx, cy = center
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx * dx + dy * dy)
    sdf = dist - radius
    mask = (sdf <= 0.0).astype(np.int32)
    return mask, sdf.astype(np.float32)


def _signed_distance_from_polygons(
    mask: np.ndarray, polygons: List[np.ndarray], nx: int, ny: int
) -> np.ndarray:
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    points = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)

    min_dist_sq = np.full(points.shape[0], np.inf, dtype=np.float32)

    for poly in polygons:
        coords = poly.astype(np.float32)
        n = coords.shape[0]
        p0 = coords
        p1 = np.roll(coords, -1, axis=0)

        for idx in range(n):
            a = p0[idx]
            b = p1[idx]
            ab = b - a
            ab_len_sq = float(ab[0] * ab[0] + ab[1] * ab[1]) + 1e-12
            ap = points - a
            t = (ap[:, 0] * ab[0] + ap[:, 1] * ab[1]) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = a + t[:, None] * ab
            diff = points - closest
            dist_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    dist = np.sqrt(min_dist_sq).reshape(nx, ny)
    sign = np.where(mask == 1, -1.0, 1.0)
    return (dist * sign).astype(np.float32)


def create_lid_velocity_profile(nx: int, lid_vel: float) -> np.ndarray:
    x_norm = np.linspace(0, 1, nx)
    arg = 10.0 * (x_norm - 0.5)
    ch1, ch2 = 0.5 * (np.exp(arg) + np.exp(-arg)), 0.5 * (np.exp(5.0) + np.exp(-5.0))
    return (lid_vel * (1.0 - ch1 / ch2)).astype(np.float32)
