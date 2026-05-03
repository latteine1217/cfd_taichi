"""
Lid-Driven Cavity CaseRunner smoke test
======================================

What:
- 驗證 `examples/lid_driven_cavity.py` 已可由 CaseRunner 驅動並完成最小輸出

Why:
- 這個案例是第一個真正遷移到 workflow 層的 example，需要回歸保護
"""

from __future__ import annotations

import taichi as ti

from examples.lid_driven_cavity import run_lid_driven_cavity


def test_lid_driven_cavity_case_runs_with_case_runner(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    run_lid_driven_cavity(
        res=16,
        re=100.0,
        lid_vel=0.05,
        cs=0.0,
        steps=20,
        interval=10,
        tol=1e-5,
        output_dir=str(tmp_path),
        collision_model="mrt",
    )

    assert (tmp_path / "history.npy").exists()
    assert (tmp_path / "state_000000.npy").exists()
