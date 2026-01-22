"""
幾何生成工具
============

提供各種幾何形狀的生成函數。
修正機翼系統：確保縫翼與襟翼隨主翼弦長正確縮放與定位。
"""

import numpy as np
from typing import Tuple, Optional


def generate_naca(naca_code: str, num_points: int = 200) -> np.ndarray:
    """生成 NACA 4 位數翼型座標"""
    if len(naca_code) != 4:
        raise ValueError(f"NACA code must be 4 digits, got {naca_code}")

    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0

    beta = np.linspace(0, np.pi, num_points // 2)
    x = (1 - np.cos(beta)) / 2

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
        dyc_dx = np.where(x < p, 2*m/p**2 * (p - x), 2*m/(1-p)**2 * (p - x))

    theta = np.arctan(dyc_dx)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.stack([x_coords, y_coords], axis=1)


def create_circle_mask(nx, ny, center, radius):
    cx, cy = center
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    return ((x - cx)**2 + (y - cy)**2 <= radius**2).astype(np.int32)


def rotate_coords(coords: np.ndarray, angle_deg: float, pivot: np.ndarray = np.array([0, 0])) -> np.ndarray:
    """繞支點旋轉座標"""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    return (coords - pivot) @ rot.T + pivot


def create_airfoil_system(
    nx: int,
    ny: int,
    main_naca: str = "2412",
    slat_naca: str = "0012",
    flap_naca: str = "4415",
    main_chord: float = 100.0,
    aoa: float = 0.0,
    slat_angle: float = 20.0,
    flap_angle: float = 30.0,
    center_position: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    創建高升力機翼系統（修正縮放與安裝位置）
    """
    if center_position is None:
        center_position = np.array([nx / 4, ny / 2])
    else:
        center_position = np.array(center_position)

    # 1. 生成原始座標
    main_raw = generate_naca(main_naca)
    slat_raw = generate_naca(slat_naca)
    flap_raw = generate_naca(flap_naca)

    # 2. 定義安裝位置 (相對於主翼 LE 的 % 弦長)
    # 縫翼：尺寸 18%，安裝在主翼前緣前方與下方
    slat_c = 0.18 * main_chord
    slat_offset = np.array([-0.16 * main_chord, -0.04 * main_chord])
    
    # 襟翼：尺寸 35%，安裝在主翼後緣附近 (LE 位於 95% 處)
    flap_c = 0.35 * main_chord
    flap_offset = np.array([0.95 * main_chord, -0.08 * main_chord])

    # 3. 本地變換 (縮放 + 偏轉)
    # 主翼
    main_coords = main_raw * main_chord
    
    # 縫翼：縮放 -> 繞自己前緣轉 -> 移動到安裝位置
    slat_coords = rotate_coords(slat_raw * slat_c, -slat_angle) + slat_offset
    
    # 襟翼：縮放 -> 繞自己前緣轉 -> 移動到安裝位置
    flap_coords = rotate_coords(flap_raw * flap_c, -flap_angle) + flap_offset

    # 4. 全局攻角旋轉 (繞主翼前緣旋轉)
    main_final = rotate_coords(main_coords, -aoa) + center_position
    slat_final = rotate_coords(slat_coords, -aoa) + center_position
    flap_final = rotate_coords(flap_coords, -aoa) + center_position

    # 5. 生成 Mask
    from matplotlib.path import Path
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    
    mask = np.zeros((nx, ny), dtype=np.int32)
    for coords in [main_final, slat_final, flap_final]:
        p = Path(coords)
        m = p.contains_points(points).reshape(nx, ny)
        mask = np.maximum(mask, m.astype(np.int32))

    return mask


def create_lid_velocity_profile(nx: int, lid_vel: float) -> np.ndarray:
    x_norm = np.linspace(0, 1, nx)
    arg = 10.0 * (x_norm - 0.5)
    ch1, ch2 = 0.5 * (np.exp(arg) + np.exp(-arg)), 0.5 * (np.exp(5.0) + np.exp(-5.0))
    return (lid_vel * (1.0 - ch1 / ch2)).astype(np.float32)
