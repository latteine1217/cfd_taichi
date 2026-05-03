"""
Transonic bump Euler 驗證案例 smoke test
========================================

What: 驗證 bump channel 網格與案例 driver 的最小可用性
Why:  transonic 驗證路徑至少要能穩定跑完、輸出摘要、且保持有限值
"""

import numpy as np
import taichi as ti

from examples.transonic_bump_euler import run_transonic_bump
from examples.transonic_bump_sweep import run_sweep


def test_transonic_bump_case_runs_and_writes_outputs(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    result = run_transonic_bump(
        ni=60,
        nj=20,
        ma=0.7,
        steps=10,
        length=3.0,
        height=1.0,
        bump_center=1.5,
        bump_width=1.0,
        bump_height=0.12,
        cfl=0.25,
        report=5,
        tol=1e-10,
        output_dir=str(tmp_path),
    )

    assert np.isfinite(result["residual"])
    assert result["mach_max"] > 0.0
    assert result["shock_sensor_max"] >= 0.0
    assert (tmp_path / "summary.npy").exists()
    assert (tmp_path / "history.npy").exists()
    assert (tmp_path / "state_final.npy").exists()


def test_transonic_bump_sweep_writes_outputs(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    results = run_sweep(
        sweep="ma",
        values=[0.62, 0.66],
        ni=60,
        nj=20,
        ma=0.66,
        steps=10,
        length=3.0,
        height=1.0,
        bump_center=1.5,
        bump_width=1.0,
        bump_height=0.12,
        cfl=0.25,
        report=5,
        tol=1e-10,
        plateau_window=3,
        shock_tol=0.05,
        mach_tol=0.1,
        output_dir=str(tmp_path),
        save_cases=False,
    )

    assert len(results) == 2
    assert (tmp_path / "sweep_results.npy").exists()
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "matrix_summary.npy").exists()
    assert (tmp_path / "shock_peak_vs_ma.png").exists()
    assert (tmp_path / "mach_max_vs_ma.png").exists()
    assert (tmp_path / "supersonic_fraction_vs_ma.png").exists()
