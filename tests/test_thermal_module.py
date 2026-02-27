# tests/test_thermal_module.py
"""
ThermalModule 單元測試
======================

驗證目標：
1. ThermalModule 初始化：場大小、物理參數計算正確
2. 線性溫度初始化：底部/頂部/中間值符合預期
"""

import taichi as ti
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_thermal_module_init():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=32, ny=32, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.0, T_ref=0.5)

    assert thermal.g.shape == (34, 34)  # nx_g=34, ny_g=34
    assert thermal.T.shape == (34, 34)
    assert abs(thermal.kappa - solver.nu / 0.71) < 1e-6
    assert thermal.tau_g > 0.5


def test_init_temperature_linear():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=32, ny=32, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal.init_temperature(T_bot=1.0, T_top=0.0)

    T_np = thermal.T.to_numpy()
    assert abs(T_np[16, 1] - 1.0) < 0.05
    assert abs(T_np[16, 32] - 0.0) < 0.05
    assert abs(T_np[16, 16] - 0.5) < 0.1


def test_thermal_step_preserves_uniform_temperature():
    """均勻溫度場在靜止流場下，經過多步後應保持不變"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=100.0, u_ref=0.05)
    solver.u.fill(0.0)  # 靜止流場

    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.7)  # 均勻溫度 0.7

    for step in range(100):
        g_src = thermal.g if step % 2 == 0 else thermal.g_new
        g_dst = thermal.g_new if step % 2 == 0 else thermal.g
        thermal.step(g_src, g_dst)

    thermal._update_temperature(g_dst)
    T_np = thermal.T.to_numpy()
    T_inner = T_np[1:17, 1:17]
    assert np.max(np.abs(T_inner - 0.7)) < 0.01


def test_thermal_diffusion_reduces_gradient():
    """
    非均勻溫度場在靜止流場下，梯度應隨時間擴散（全局溫度方差應降低）

    驗證目標：
    - 非均勻初始條件下（x 方向階梯函數），擴散使溫度場均勻化
    - 串流方向正確：方差隨步數單調遞減，否則代表串流寫反或 eu 符號錯誤
    - 場不發散：溫度值始終在合理範圍

    Why 用方差而非最大梯度?
    - 線性初始場已接近擴散穩態，最大梯度幾乎不變；
    - 非線性（階梯）初始場的方差在擴散過程中明顯下降，
      且對串流方向錯誤具有較強的偵測力。
    """
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule
    import numpy as np

    nx, ny = 32, 32
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.05)
    solver.u.fill(0.0)  # 靜止流場

    thermal = ThermalModule(solver, Pr=0.71)

    # 用 x 方向階梯函數初始化：左半 T=1.0，右半 T=0.0
    # 這樣在 x=16 附近有尖銳梯度，是典型的非穩態擴散場
    g_np = thermal.g.to_numpy()
    w_np = thermal.w.to_numpy()
    for i in range(1, nx + 1):
        T_init = 1.0 if i <= nx // 2 else 0.0
        for j in range(1, ny + 1):
            for k in range(9):
                g_np[i, j, k] = w_np[k] * T_init
    thermal.g.from_numpy(g_np)
    thermal.g_new.from_numpy(g_np)  # 雙緩衝同步

    # 記錄初始溫度場方差（衡量非均勻程度）
    thermal._update_temperature(thermal.g)
    T_before = thermal.T.to_numpy()[1:nx + 1, 1:ny + 1]
    var_before = float(np.var(T_before))

    # 執行 500 步擴散
    g_dst = thermal.g  # 初始化，避免未綁定
    for step_i in range(500):
        g_src = thermal.g if step_i % 2 == 0 else thermal.g_new
        g_dst = thermal.g_new if step_i % 2 == 0 else thermal.g
        thermal.step(g_src, g_dst)

    # 讀取最新溫度（需手動呼叫 _update_temperature(g_dst)）
    thermal._update_temperature(g_dst)
    T_after = thermal.T.to_numpy()
    var_after = float(np.var(T_after[1:nx + 1, 1:ny + 1]))

    # 擴散後方差應顯著降低（至少降低 10%），且場不發散
    assert var_after < var_before * 0.9, (
        f"溫度場方差應因擴散而降低（至少 10%），"
        f"但 before={var_before:.6f}, after={var_after:.6f}"
    )
    assert np.all(T_after[1:nx + 1, 1:ny + 1] >= -0.1), "溫度不應低於 -0.1（發散）"
    assert np.all(T_after[1:nx + 1, 1:ny + 1] <= 1.1), "溫度不應高於 1.1（發散）"


def test_buoyancy_zero_at_T_ref():
    """T = T_ref 時浮力應為零"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.001, T_ref=0.5)
    thermal._fill_equilibrium(0.5)        # T = T_ref 均勻
    thermal._update_temperature(thermal.g)

    thermal.compute_buoyancy()
    F_np = solver.force_field.to_numpy()
    assert np.max(np.abs(F_np[1:17, 1:17, 1])) < 1e-6  # F_y ≈ 0


def test_buoyancy_positive_above_T_ref():
    """T > T_ref 時 F_y 應為正（向上浮力）"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule

    solver = LBMSolver(nx=16, ny=16, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71, beta=1.0, g_gravity=0.001, T_ref=0.5)
    thermal._fill_equilibrium(1.0)        # T = 1.0 > T_ref = 0.5
    thermal._update_temperature(thermal.g)

    thermal.compute_buoyancy()
    F_np = solver.force_field.to_numpy()
    assert np.min(F_np[1:17, 1:17, 1]) > 0  # 全場 F_y > 0


def test_hot_wall_bottom_sets_temperature():
    """底部熱壁應讓 j=1 的溫度趨近 T_hot"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions

    solver = LBMSolver(nx=16, ny=16, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.5)

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location='bottom')

    # 施加 BC 多次讓溫度收斂
    for _ in range(50):
        tbc.apply(thermal.g)
        thermal._update_temperature(thermal.g)

    T_np = thermal.T.to_numpy()
    # j=1 的平均溫度應接近 1.0
    assert np.mean(T_np[1:17, 1]) > 0.9


def test_adiabatic_wall_left():
    """絕熱左壁：施加 BC 後溫度梯度不應從邊界引入熱量"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    from lbm_taichi.core import LBMSolver
    from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions

    solver = LBMSolver(nx=16, ny=16, re=100.0, u_ref=0.05)
    thermal = ThermalModule(solver, Pr=0.71)
    thermal._fill_equilibrium(0.5)

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_adiabatic_wall('left')
    tbc.apply(thermal.g)

    # 施加後分佈函數應保持合理（不爆炸）
    g_np = thermal.g.to_numpy()
    assert not np.any(np.isnan(g_np))
    assert not np.any(np.isinf(g_np))
