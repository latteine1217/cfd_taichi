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
