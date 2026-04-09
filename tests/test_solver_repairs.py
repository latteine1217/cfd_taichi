"""
Solver Repair 驗證測試
======================

測試三個關鍵修復點：
1. BC 所有權：add_no_slip_wall() 後索引列表自動正確（無需呼叫端手動 rebuild）
2. Reset 語意：reset_flow_state() 保留 mask/幾何設定
3. LDC 設定：不依賴私有方法 _build_index_lists()

使用 ti.cpu backend 以確保 CI / 非 Metal 環境可執行。
"""

import numpy as np
import pytest
import taichi as ti

from lbm_taichi.core import BoundaryConditions, LBMSolver

# ---------------------------------------------------------------------------
# 共用 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def taichi_init():
    """初始化 Taichi（CPU，避免 Metal GPU 需求）"""
    ti.init(arch=ti.cpu, default_fp=ti.f32)


# ---------------------------------------------------------------------------
# Test 1 — BC 所有權：add_no_slip_wall 後 boundary/solid 計數正確
# ---------------------------------------------------------------------------

def test_index_rebuild_after_no_slip_wall():
    """
    add_no_slip_wall() 後，solid / boundary 計數應立即反映壁面設定，
    不需要呼叫端手動呼叫 solver._build_index_lists()。

    Why this test:
    - 修復前：add_no_slip_wall 改了 mask，但索引列表未更新 → solid_count = 0
    - 修復後：add_no_slip_wall 自動 rebuild → solid_count > 0
    """
    nx, ny = 32, 32
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)

    # reset() 後，整個域為流體 → solid_count 應為 0
    solid_before = int(solver.num_solid[None])
    assert solid_before == 0, f"Initial solid count should be 0, got {solid_before}"

    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall("bottom")

    # add_no_slip_wall 應自動 rebuild → solid_count = nx（底壁節點數）
    solid_after = int(solver.num_solid[None])
    assert solid_after == nx, (
        f"After add_no_slip_wall('bottom'), solid count should be {nx}, got {solid_after}. "
        "Possible cause: rebuild_index_lists() not called automatically."
    )

    # boundary / bulk 分類也應有效（boundary_count > 0）
    boundary_after = int(solver.num_fluid_boundary[None])
    assert boundary_after > 0, (
        f"Expected boundary_count > 0 after adding no-slip wall, got {boundary_after}"
    )


def test_three_walls_index_counts():
    """
    LDC 典型設定：底/左/右三個 No-Slip 壁 + 排除角點。
    solid_count 應等於 3*ny - 4（角點各壁排除 2 個角落節點）。

    Why: 確保多次呼叫 add_no_slip_wall() 時，每次 rebuild 都正確累積。
    """
    nx, ny = 32, 32
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)
    bc = BoundaryConditions(solver)

    bc.add_no_slip_wall("bottom", exclude_corners=True)
    bc.add_no_slip_wall("left", exclude_corners=True)
    bc.add_no_slip_wall("right", exclude_corners=True)

    solid_count = int(solver.num_solid[None])
    # bottom: nx-2，left: ny-2，right: ny-2 → 總計 (nx-2) + 2*(ny-2)
    expected = (nx - 2) + 2 * (ny - 2)
    assert solid_count == expected, (
        f"Expected solid_count={expected} (3 walls, exclude_corners), got {solid_count}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Reset 語意：reset_flow_state() 保留 mask
# ---------------------------------------------------------------------------

def test_reset_flow_state_preserves_mask():
    """
    reset_flow_state() 必須保留 mask（障礙物 / No-Slip 壁面設定）。

    Why: 若 reset_flow_state() 清除 mask，幾何設定會靜默消失，
         使後續模擬在不含壁面的空域上運行。
    """
    nx, ny = 32, 32
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)
    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall("bottom")

    mask_before = solver.mask.to_numpy().copy()
    solid_before = int(solver.num_solid[None])

    # 重置流場狀態（應保留 mask）
    solver.reset_flow_state()

    mask_after = solver.mask.to_numpy()
    solid_after = int(solver.num_solid[None])

    assert np.array_equal(mask_before, mask_after), (
        "reset_flow_state() must NOT modify the mask. "
        "Geometry/wall setup should be preserved."
    )
    assert solid_after == solid_before, (
        f"solid_count changed: {solid_before} -> {solid_after} after reset_flow_state()"
    )


def test_reset_clears_mask():
    """
    reset()（完整重置）必須清除 mask，回到全流體狀態。

    Why: reset() 是「重新建立等效 solver」的快捷方式，
         應確保幾何完全重置以便切換案例。
    """
    nx, ny = 32, 32
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=0.1)
    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall("bottom")
    assert int(solver.num_solid[None]) > 0  # precondition

    solver.reset()

    # reset() 後，內部節點的 mask 應全為 0（流體）
    mask_np = solver.mask.to_numpy()
    interior_solid = int(mask_np[1:nx+1, 1:ny+1].sum())
    assert interior_solid == 0, (
        f"After reset(), interior solid count should be 0, got {interior_solid}"
    )


def test_reset_flow_state_velocity_field():
    """
    reset_flow_state() 後，流體節點速度應重置為 (u_ref, 0)。

    Why: 確保流場狀態確實被重置（不只是 mask 保留）。
    """
    nx, ny = 16, 16
    u_ref = 0.1
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=u_ref)

    # 手動干擾速度場
    solver.u.fill(0.99)

    solver.reset_flow_state()
    solver._update_macro(solver.f)

    u_np = solver.u.to_numpy()  # shape (nx_g, ny_g, 2)
    # 內部流體節點的 ux 應近似 u_ref
    interior_ux = u_np[1:nx+1, 1:ny+1, 0]
    assert np.allclose(interior_ux, u_ref, atol=1e-5), (
        f"After reset_flow_state(), interior ux should ≈ {u_ref}; "
        f"got mean={interior_ux.mean():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — LDC 不需要私有 _build_index_lists 呼叫
# ---------------------------------------------------------------------------

def test_ldc_setup_no_private_rebuild_needed():
    """
    LDC 典型設定後，索引列表應已正確，無需手動呼叫 solver._build_index_lists()。

    Why: 修復前，lid_driven_cavity.py 必須在 BC 設定後手動呼叫
         solver._build_index_lists(solver.mask.to_numpy())。
         修復後，add_no_slip_wall() 自動處理，呼叫端不需知道此細節。
    """
    from lbm_taichi.utils.geometry import create_lid_velocity_profile

    nx, ny = 32, 32
    u_lid = 0.1
    solver = LBMSolver(nx=nx, ny=ny, re=100.0, u_ref=u_lid, length_scale=nx)

    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall("bottom", exclude_corners=True)
    bc.add_no_slip_wall("left", exclude_corners=True)
    bc.add_no_slip_wall("right", exclude_corners=True)
    u_profile = create_lid_velocity_profile(nx, u_lid)
    bc.add_moving_wall(u_profile, location="top")

    # 不呼叫 solver._build_index_lists() — 應已由 add_no_slip_wall 自動完成

    solid_count = int(solver.num_solid[None])
    boundary_count = int(solver.num_fluid_boundary[None])

    expected_solid = (nx - 2) + 2 * (ny - 2)
    assert solid_count == expected_solid, (
        f"LDC solid_count should be {expected_solid}, got {solid_count}. "
        "If 0, add_no_slip_wall() is not triggering rebuild_index_lists()."
    )
    assert boundary_count > 0, (
        "LDC boundary_count should be > 0 after BC setup."
    )


# ---------------------------------------------------------------------------
# Entry point（手動執行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("=== Test 1a: index rebuild after no_slip_wall ===")
    test_index_rebuild_after_no_slip_wall()
    print("PASS")
    print("=== Test 1b: three walls index counts ===")
    test_three_walls_index_counts()
    print("PASS")
    print("=== Test 2a: reset_flow_state preserves mask ===")
    test_reset_flow_state_preserves_mask()
    print("PASS")
    print("=== Test 2b: reset() clears mask ===")
    test_reset_clears_mask()
    print("PASS")
    print("=== Test 2c: reset_flow_state velocity field ===")
    test_reset_flow_state_velocity_field()
    print("PASS")
    print("=== Test 3: LDC setup without private rebuild ===")
    test_ldc_setup_no_private_rebuild_needed()
    print("PASS")
    print("\nAll tests passed.")
