"""
CFD Taichi - 輔助工具模組
===========================

提供幾何生成、可視化等通用功能。

模組：
- geometry: 幾何形狀生成（NACA 翼型、圓柱等）
- visualization: 統一的可視化介面
"""

from .geometry import generate_naca, create_circle_mask, create_airfoil_system
from .visualization import Visualizer

__all__ = [
    'generate_naca',
    'create_circle_mask',
    'create_airfoil_system',
    'Visualizer',
]
