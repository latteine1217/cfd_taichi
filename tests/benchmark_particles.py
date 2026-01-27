"""
粒子系統性能基準測試
====================

測試粒子系統各個階段的 GPU 並行性能。
"""

import taichi as ti
import time
import numpy as np


def benchmark_particle_system():
    """執行粒子系統性能測試"""
    from core import LBMSolver

    print("="*70)
    print(" "*20 + "粒子系統性能基準測試")
    print("="*70)

    # 初始化
    ti.init(arch=ti.metal, default_fp=ti.f32, kernel_profiler=True)

    nx, ny = 896, 256
    solver = LBMSolver(nx=nx, ny=ny, re=1000, u_ref=0.1)

    # === 測試 1: 粒子初始化 ===
    print("\n[測試 1] 粒子初始化")
    ti.sync()
    t0 = time.perf_counter()
    solver._init_particles()
    ti.sync()
    t1 = time.perf_counter()
    init_time = (t1 - t0) * 1000
    print(f"  時間: {init_time:.3f} ms")
    print(f"  粒子數: {solver.num_particles:,}")
    print(f"  吞吐量: {solver.num_particles / init_time:.1f} 粒子/ms")

    # === 測試 2: 粒子發射 ===
    print("\n[測試 2] 粒子發射 (32 條煙線)")
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(100):
        solver._emit_particles(32)
    ti.sync()
    t1 = time.perf_counter()
    emit_time = (t1 - t0) * 10  # 平均每次
    print(f"  單次時間: {emit_time:.4f} ms")
    print(f"  發射粒子數: 32")
    print(f"  吞吐量: {32 / emit_time:.1f} 粒子/ms")

    # === 測試 3: 粒子推進 ===
    print("\n[測試 3] 粒子推進")
    # 先發射一些粒子
    for _ in range(100):
        solver._emit_particles(32)

    ti.sync()
    t0 = time.perf_counter()
    for _ in range(100):
        solver._advect_particles()
    ti.sync()
    t1 = time.perf_counter()
    advect_time = (t1 - t0) * 10
    print(f"  單次時間: {advect_time:.3f} ms")
    print(f"  處理粒子數: ~3,200 (活躍)")
    print(f"  吞吐量: {3200 / advect_time:.1f} 粒子/ms")

    # === 測試 4: 粒子統計 ===
    print("\n[測試 4] 粒子統計 (並行歸約)")
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(100):
        count = solver._count_active_particles()
    ti.sync()
    t1 = time.perf_counter()
    count_time = (t1 - t0) * 10
    print(f"  單次時間: {count_time:.3f} ms")
    print(f"  掃描粒子數: {solver.num_particles:,}")
    print(f"  活躍粒子數: {count}")
    print(f"  吞吐量: {solver.num_particles / count_time:.1f} 粒子/ms")

    # === 測試 5: 完整時間步 ===
    print("\n[測試 5] 完整粒子系統時間步 (發射+推進)")
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(100):
        solver.step_particles(emit=True, num_lines=32, count_active=False)
    ti.sync()
    t1 = time.perf_counter()
    step_time = (t1 - t0) * 10
    print(f"  單次時間: {step_time:.3f} ms")
    print(f"  等效 FPS: {1000 / step_time:.0f}")

    # === 測試 6: 數據傳輸（緊湊格式）===
    print("\n[測試 6] 粒子數據導出 (GPU→CPU)")
    # 確保有活躍粒子
    for _ in range(50):
        solver.step_particles(emit=True, num_lines=32)

    ti.sync()
    t0 = time.perf_counter()
    active = solver.p_active.to_numpy() == 1
    px = solver.px.to_numpy()
    py = solver.py.to_numpy()
    particles = np.stack([px[active], py[active]], axis=1)
    t1 = time.perf_counter()
    transfer_time = (t1 - t0) * 1000
    print(f"  時間: {transfer_time:.3f} ms")
    print(f"  活躍粒子數: {particles.shape[0]:,}")
    print(f"  數據大小: {particles.nbytes / 1024:.1f} KB")
    print(f"  傳輸速度: {particles.nbytes / 1024 / transfer_time:.1f} KB/ms")

    # === 總結 ===
    print("\n" + "="*70)
    print(" "*25 + "性能總結")
    print("-"*70)
    print(f"{'操作':<20} {'時間 (ms)':<15} {'吞吐量':<25}")
    print("-"*70)
    print(f"{'粒子初始化':<20} {init_time:<15.3f} {solver.num_particles / init_time:>20.0f} 粒子/ms")
    print(f"{'粒子發射':<20} {emit_time:<15.4f} {32 / emit_time:>20.0f} 粒子/ms")
    print(f"{'粒子推進':<20} {advect_time:<15.3f} {3200 / advect_time:>20.0f} 粒子/ms")
    print(f"{'粒子統計':<20} {count_time:<15.3f} {solver.num_particles / count_time:>20.0f} 粒子/ms")
    print(f"{'完整時間步':<20} {step_time:<15.3f} {1000 / step_time:>20.0f} FPS")
    print(f"{'數據導出':<20} {transfer_time:<15.3f} {particles.nbytes / 1024 / transfer_time:>20.1f} KB/ms")
    print("="*70)

    # === Kernel Profiler ===
    print("\n[Taichi Kernel Profiler]")
    ti.profiler.print_kernel_profiler_info()

    print("\n✅ 基準測試完成！")


if __name__ == "__main__":
    benchmark_particle_system()
