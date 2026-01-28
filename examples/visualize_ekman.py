"""
Ekman 螺旋可視化工具
====================

功能：
1. Hodograph（速度矢量圖）
2. 深度剖面對比（數值 vs 解析解）
3. 時間演化動畫
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.dirname(__file__))
)

try:
    from ekman_spiral import ekman_analytical_solution
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from examples.ekman_spiral import ekman_analytical_solution


def plot_hodograph(u_profile, v_profile, z_depths, output_path):
    """
    繪製 Hodograph（速度矢量圖）

    Args:
        u_profile: 東向速度 (m/s)
        v_profile: 北向速度 (m/s)
        z_depths: 深度 (m)
        output_path: 輸出檔案路徑
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 繪製數值解
    ax.plot(u_profile, v_profile, 'o-', linewidth=2, markersize=6,
            label='Numerical', color='blue')

    # 標記表面與底部
    ax.plot(u_profile[0], v_profile[0], 'ro', markersize=10, label='Surface')
    ax.plot(u_profile[-1], v_profile[-1], 'ks', markersize=10, label='Bottom')

    # 添加深度標記
    for i in [0, len(u_profile)//4, len(u_profile)//2, 3*len(u_profile)//4, -1]:
        ax.annotate(f'{z_depths[i]:.0f}m',
                   xy=(u_profile[i], v_profile[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)

    ax.set_xlabel('Eastward Velocity u (m/s)', fontsize=12)
    ax.set_ylabel('Northward Velocity v (m/s)', fontsize=12)
    ax.set_title('Ekman Spiral Hodograph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Hodograph 已儲存至 {output_path}")
    plt.close()


def plot_depth_profiles(u_num, v_num, u_ana, v_ana, z_depths, output_path):
    """
    繪製深度剖面對比

    Args:
        u_num, v_num: 數值解
        u_ana, v_ana: 解析解
        z_depths: 深度 (m)
        output_path: 輸出檔案路徑
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # 速度大小
    vel_num = np.sqrt(u_num**2 + v_num**2)
    vel_ana = np.sqrt(u_ana**2 + v_ana**2)

    axes[0].plot(vel_num, -z_depths, 'o-', label='Numerical', linewidth=2)
    axes[0].plot(vel_ana, -z_depths, '--', label='Analytical', linewidth=2)
    axes[0].set_xlabel('Velocity Magnitude (m/s)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title('Velocity Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 速度方向
    angle_num = np.degrees(np.arctan2(v_num, u_num))
    angle_ana = np.degrees(np.arctan2(v_ana, u_ana))

    axes[1].plot(angle_num, -z_depths, 'o-', label='Numerical', linewidth=2)
    axes[1].plot(angle_ana, -z_depths, '--', label='Analytical', linewidth=2)
    axes[1].set_xlabel('Flow Direction (°)')
    axes[1].set_ylabel('Depth (m)')
    axes[1].set_title('Flow Direction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 相對誤差
    u_err = np.abs(u_num - u_ana) / (np.max(np.abs(u_ana)) + 1e-10) * 100
    v_err = np.abs(v_num - v_ana) / (np.max(np.abs(v_ana)) + 1e-10) * 100

    axes[2].plot(u_err, -z_depths, 'o-', label='u error', linewidth=2)
    axes[2].plot(v_err, -z_depths, 's-', label='v error', linewidth=2)
    axes[2].set_xlabel('Relative Error (%)')
    axes[2].set_ylabel('Depth (m)')
    axes[2].set_title('Relative Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"深度剖面已儲存至 {output_path}")
    plt.close()


def visualize_ekman_results(output_dir: str):
    """
    可視化 Ekman 模擬結果

    Args:
        output_dir: 模擬輸出目錄
    """
    # 讀取最終狀態
    state_files = sorted([f for f in os.listdir(output_dir) if f.startswith('state_')])
    if not state_files:
        print("錯誤：找不到狀態檔案")
        return

    final_state = np.load(os.path.join(output_dir, state_files[-1]), allow_pickle=True).item()

    u_num = final_state['u_profile']
    v_num = final_state['v_profile']
    n_layers = final_state['n_layers']
    dz = final_state['dz']

    z_depths = np.arange(n_layers) * dz

    # 讀取對比數據
    comparison_file = os.path.join(output_dir, "comparison.npy")
    if os.path.exists(comparison_file):
        comp = np.load(comparison_file, allow_pickle=True).item()
        u_ana = comp['u_analytical']
        v_ana = comp['v_analytical']
    else:
        print("警告：找不到解析解對比數據")
        u_ana = v_ana = None

    # 繪製 Hodograph
    hodograph_path = os.path.join(output_dir, "hodograph.png")
    plot_hodograph(u_num, v_num, z_depths, hodograph_path)

    # 繪製深度剖面
    if u_ana is not None:
        profiles_path = os.path.join(output_dir, "depth_profiles.png")
        plot_depth_profiles(u_num, v_num, u_ana, v_ana, z_depths, profiles_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize Ekman Spiral Results")
    parser.add_argument('--output_dir', type=str, default='output_ekman',
                       help='模擬輸出目錄')

    args = parser.parse_args()

    visualize_ekman_results(args.output_dir)


if __name__ == "__main__":
    main()
