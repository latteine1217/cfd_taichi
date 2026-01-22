"""
碰撞模型模組
============

提供不同的 LBM 碰撞算子實作：
- BGK (Single Relaxation Time)
- MRT (Multiple Relaxation Time)
- LES (Large Eddy Simulation) 湍流模型

Why 需要這個模組?
- 解耦碰撞邏輯與核心 solver
- 方便擴展新的物理模型（如非牛頓流體、多相流）
- 統一不同模型所需的參數接口
"""

import taichi as ti


@ti.data_oriented
class CollisionModels:
    """
    碰撞模型模組（預留擴展）

    目前 MRT + Smagorinsky 已內建在 LBMSolver
    未來可擴展：
    - BGK (single relaxation time)
    - Regularized LBM
    - Cumulant LBM
    - 其他 LES 模型（WALE, Dynamic Smagorinsky）
    """

    pass  # TODO: 未來擴展