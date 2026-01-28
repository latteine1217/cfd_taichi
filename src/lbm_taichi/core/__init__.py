"""
CFD Taichi - 統一 LBM 核心模組
===================================

這個模組提供統一的 Lattice Boltzmann Method (LBM) 求解器實作，
支援多種 CFD case 的模擬需求。

主要組件：
- LBMSolver: 核心 D2Q9 MRT-LBM 求解器
- BoundaryConditions: 各種邊界條件實作（Zou-He, Bounce-Back 等）
- CollisionModels: 碰撞模型（MRT, Smagorinsky LES）
- Diagnostics: 物理檢查與診斷輸出

設計原則：
- 簡潔性：消除重複程式碼，單一職責
- 模組化：邊界條件、碰撞模型可獨立配置
- 可驗證：質量守恆、動量殘差等物理檢查
- 高效能：Taichi GPU 加速，Metal 後端優化
"""

from .lbm_solver import LBMSolver
from .boundary_conditions import BoundaryConditions
from .collision_models import CollisionModels
from .diagnostics import Diagnostics

__all__ = [
    'LBMSolver',
    'BoundaryConditions',
    'CollisionModels',
    'Diagnostics',
]

__version__ = '2.0.0'