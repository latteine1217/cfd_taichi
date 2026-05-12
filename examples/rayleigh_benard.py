"""
Rayleigh-Bénard 對流模擬（封閉腔體）
======================================

What: 底部加熱、頂部冷卻的四壁封閉腔體對流
Why:  驗證 DDF+Boussinesq LBM 的 Nu-Ra 關係（封閉腔體設定）

物理設定:
    Ra = g·β·ΔT·H³ / (ν·κ)，β=1，ΔT=1（無因次化）
    Pr = ν / κ
    上下壁：no-slip，Dirichlet 溫度（T_bottom=1, T_top=0）
    左右壁：no-slip，絕熱（Neumann ∂T/∂x=0）

驗收標準（穩態 Nu，封閉腔體 all-no-slip walls）:
    1:1 正方形腔體（aspect=1.0）:
        Ra = 1e5 → Nu ≈ 1.7–2.0  [本代碼: 1.74, Pallares & Herrero 2005]

    2:1 矩形腔體（aspect=2.0，預設）:
        Ra = 1e5 → Nu ≈ 2.0–2.3  [本代碼: 2.10]

注意 — 與其他 benchmark 的區別:
    de Vahl Davis (1983): 側壁差溫加熱（左熱右冷），Nu≈4.52 (Ra=1e5) → 不同問題
    無限水平層（週期 BC）:  Nu≈4–5 (Ra=1e5, Pr=0.71) → 無側壁效應
    本設定的 Nu 較低，因封閉側壁（no-slip）強烈抑制對流胞的速度。
"""

import os
import taichi as ti
import numpy as np
import argparse
import time

from cfd_taichi import CaseRunner, LatticeGrid2D
from lbm_taichi.core import LBMSolver, BoundaryConditions
from lbm_taichi.core.thermal_module import ThermalModule, ThermalBoundaryConditions


def compute_lbm_params(Ra: float, Pr: float, ny: int, u_ref: float):
    """
    由 Ra、Pr 推導格子 LBM 參數

    What: 將無因次物理參數轉換為 LBM lattice units
    Why:
        自然對流的特徵速度為 U ~ sqrt(g·β·ΔT·H)。
        在 Boussinesq 近似下 β·ΔT=1，以 u_ref 作為速度量化單位：
            Re_eff = sqrt(Ra/Pr)   ← 速度-擴散型 Reynolds 數
            nu = u_ref * H / Re_eff
            kappa = nu / Pr
            g_lbm = Ra * nu * kappa / H³  ← 由 Ra 定義反推，確保 Ra 一致

    Returns:
        nu:     運動黏度（lattice units）
        kappa:  熱擴散率（lattice units）
        g_lbm:  重力加速度（lattice units，向上為正）
        Re_eff: 特徵 Reynolds 數
        tau_f:  流場鬆弛時間
        tau_g:  溫度場鬆弛時間
    """
    H = float(ny)
    Re_eff = (Ra / Pr) ** 0.5
    nu = u_ref * H / Re_eff
    kappa = nu / Pr
    tau_f = 0.5 + 3.0 * nu
    tau_g = 0.5 + 3.0 * kappa
    # 由 Ra = g·β·ΔT·H³/(nu·kappa) 反推，β·ΔT=1
    g_lbm = Ra * nu * kappa / (H ** 3)
    return nu, kappa, g_lbm, Re_eff, tau_f, tau_g


def _validate_rayleigh_benard_params(
    *,
    ny: int,
    Ra: float,
    Pr: float,
    u_ref: float,
    aspect: float,
):
    """
    驗證 Rayleigh-Bénard toolkit runner 的物理與數值前提。

    What: 檢查網格、Ra/Pr、Mach 與 LBM 鬆弛時間。
    Why:  thermal benchmark 若參數落在不穩定區，應 fail fast，而不是產生看似可用的場。
    """
    if ny < 4:
        raise ValueError(f"ny must be >= 4 for wall-gradient Nusselt diagnostics, got {ny}")
    if Ra <= 0.0:
        raise ValueError(f"Ra must be positive, got {Ra}")
    if Pr <= 0.0:
        raise ValueError(f"Pr must be positive, got {Pr}")
    if u_ref <= 0.0:
        raise ValueError(f"u_ref must be positive, got {u_ref}")
    if aspect <= 0.0:
        raise ValueError(f"aspect must be positive, got {aspect}")

    _nu, _kappa, _g_lbm, _re_eff, tau_f, tau_g = compute_lbm_params(Ra, Pr, ny, u_ref)
    if tau_f < 0.505:
        raise ValueError(
            f"tau_f={tau_f:.4f} 過小（< 0.505）。請降低 u_ref 或增大 ny。"
        )
    if tau_g < 0.505:
        raise ValueError(
            f"tau_g={tau_g:.4f} 過小（< 0.505）。請降低 u_ref 或提高 Pr。"
        )
    if u_ref * 1.732 > 0.3:
        raise ValueError(
            f"Ma={u_ref * 1.732:.3f} 超過 low-Mach Boussinesq 建議上限 0.3。"
        )


def _reset_solver_to_rest(solver: LBMSolver):
    """
    將 LBM 主分佈重設為 rho=1、u=0 的平衡態。

    Why: LBMSolver 預設較偏向入口流案例；Rayleigh-Bénard 應由靜止導熱態觸發。
    """
    weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                       dtype=np.float32)
    f0 = np.zeros((solver.nx_g, solver.ny_g, 9), dtype=np.float32)
    for k in range(9):
        f0[:, :, k] = weights[k]
    solver.f.from_numpy(f0)
    solver.f_new.from_numpy(f0)
    solver.u.fill(0.0)


def _add_rayleigh_benard_temperature_perturbation(
    thermal: ThermalModule,
    *,
    amplitude: float,
):
    """
    加入符合左右絕熱壁的 cos-sin 溫度擾動。

    Why: 線性導熱態在 x 方向均勻，需要可控擾動觸發自然對流模態。
    """
    if amplitude == 0.0:
        return

    nx = thermal.nx
    ny = thermal.ny
    g_np = thermal.g.to_numpy()
    w_np = thermal.w.to_numpy()
    x = np.arange(1, nx + 1, dtype=np.float32)
    y = np.arange(1, ny + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    perturbation = amplitude * np.cos(2.0 * np.pi * xx / nx) * np.sin(np.pi * yy / ny)
    for k in range(9):
        g_np[1:nx + 1, 1:ny + 1, k] += (w_np[k] * perturbation).astype(np.float32)
    thermal.g.from_numpy(g_np)
    thermal.g_new.from_numpy(g_np)


def _initialize_rayleigh_benard_case(
    runner: CaseRunner,
    solver: LBMSolver,
    _bc_handle,
    *,
    Ra: float,
    Pr: float,
    g_lbm: float,
    perturbation: float,
):
    """
    Rayleigh-Bénard CaseRunner 初始化流程。

    What: 建立靜止流場、DDF 熱場、Boussinesq 浮力與熱邊界條件。
    Why: toolkit runner 必須能自洽重建 standalone 腳本的物理設定。
    """
    _reset_solver_to_rest(solver)

    thermal = ThermalModule(solver, Pr=Pr, beta=1.0, g_gravity=g_lbm, T_ref=0.5)
    thermal.init_temperature(T_bot=1.0, T_top=0.0)
    _add_rayleigh_benard_temperature_perturbation(thermal, amplitude=perturbation)
    thermal.register_with_solver()

    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location="bottom")
    tbc.add_cold_wall(T_cold=0.0, location="top")
    tbc.add_adiabatic_wall(location="left")
    tbc.add_adiabatic_wall(location="right")
    tbc.apply(thermal.g)
    tbc.apply(thermal.g_new)

    runner.thermal = thermal
    runner.thermal_bc = tbc
    runner.rb_params = {"Ra": float(Ra), "Pr": float(Pr)}

    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)
    runner.prepare_observables(reset_baseline=True)


def _rayleigh_benard_stepper(runner: CaseRunner, next_step: int):
    """
    推進一個 coupled LBM thermal step。

    Why: CaseRunner 的預設 LBM step 只處理 f 分佈；Rayleigh-Bénard 需要同步推進 g 分佈。
    """
    solver = runner.solver
    thermal = runner.thermal
    tbc = runner.thermal_bc

    f_src = solver.f if next_step % 2 == 1 else solver.f_new
    f_dst = solver.f_new if next_step % 2 == 1 else solver.f
    g_src = thermal.g if next_step % 2 == 1 else thermal.g_new
    g_dst = thermal.g_new if next_step % 2 == 1 else thermal.g

    thermal._update_temperature(g_src)
    solver.step(f_src, f_dst)
    thermal.step(g_src, g_dst)
    tbc.apply(g_dst)
    return None


def _rayleigh_benard_diagnostics_hook(
    runner: CaseRunner,
    solver: LBMSolver,
    diagnostics: dict,
) -> dict:
    """
    補充 thermal benchmark 專屬診斷量。
    """
    thermal = getattr(runner, "thermal", None)
    if thermal is None:
        return {}
    g_active = thermal.g_new if runner.current_step % 2 == 1 else thermal.g
    thermal._update_temperature(g_active)
    T_np = thermal.T.to_numpy()[1:thermal.nx + 1, 1:thermal.ny + 1]
    nusselt = thermal.get_nusselt(T_bot=1.0, T_top=0.0)
    u_max = float(diagnostics.get("u_max", diagnostics.get("max_u", np.nan)))
    return {
        "u_max": u_max,
        "cfl": u_max,
        "nusselt": float(nusselt),
        "temperature_mid": float(T_np[thermal.nx // 2, thermal.ny // 2]),
        "temperature_min": float(np.min(T_np)),
        "temperature_max": float(np.max(T_np)),
        "Ra": float(runner.rb_params["Ra"]),
        "Pr": float(runner.rb_params["Pr"]),
    }


def _rayleigh_benard_state_hook(
    runner: CaseRunner,
    _solver: LBMSolver,
    _payload: dict,
) -> dict:
    """
    將溫度場與 RB 參數加入標準 state payload。
    """
    thermal = runner.thermal
    g_active = thermal.g_new if runner.current_step % 2 == 1 else thermal.g
    thermal._update_temperature(g_active)
    return {
        "temperature": thermal.T.to_numpy()[1:thermal.nx + 1, 1:thermal.ny + 1],
        "Ra": float(runner.rb_params["Ra"]),
        "Pr": float(runner.rb_params["Pr"]),
    }


def build_rayleigh_benard_runner(
    *,
    ny: int = 64,
    Ra: float = 1e5,
    Pr: float = 0.71,
    u_ref: float = 0.1,
    aspect: float = 2.0,
    perturbation: float = 0.01,
    collision_model: str = "mrt",
) -> tuple[CaseRunner, dict]:
    """
    建立 Rayleigh-Bénard 對流的 CaseRunner。

    What: 將封閉腔體 RB 的 LBM solver、熱場、熱邊界與 Nu 診斷接入 toolkit workflow。
    Why: thermal route 需要 registry/matrix 可重跑能力，而不是只依賴 standalone script。
    """
    _validate_rayleigh_benard_params(
        ny=ny,
        Ra=Ra,
        Pr=Pr,
        u_ref=u_ref,
        aspect=aspect,
    )
    nx = int(aspect * ny)
    if nx < 4:
        raise ValueError(f"aspect * ny must produce nx >= 4, got nx={nx}")

    _nu, kappa, g_lbm, Re_eff, tau_f, tau_g = compute_lbm_params(Ra, Pr, ny, u_ref)
    runner = CaseRunner(
        name="rayleigh_benard",
        method="lbm",
        equation="thermal_boussinesq",
        regime="low_mach",
        grid=LatticeGrid2D(nx=nx, ny=ny),
        solver_kwargs={
            "re": Re_eff,
            "u_ref": u_ref,
            "length_scale": float(ny),
            "cs": 0.0,
            "collision_model": collision_model,
        },
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_rayleigh_benard_case(
            runner_obj,
            solver_obj,
            bc_handle,
            Ra=Ra,
            Pr=Pr,
            g_lbm=g_lbm,
            perturbation=perturbation,
        ),
        stepper=_rayleigh_benard_stepper,
        diagnostics_hook=_rayleigh_benard_diagnostics_hook,
        state_hook=_rayleigh_benard_state_hook,
    )
    setup = {
        "nx": nx,
        "ny": int(ny),
        "Ra": float(Ra),
        "Pr": float(Pr),
        "Re_eff": float(Re_eff),
        "kappa": float(kappa),
        "g_lbm": float(g_lbm),
        "tau_f": float(tau_f),
        "tau_g": float(tau_g),
        "perturbation": float(perturbation),
    }
    return runner, setup


def run_rayleigh_benard(
    ny: int = 64,
    Ra: float = 1e5,
    Pr: float = 0.71,
    u_ref: float = 0.1,
    aspect: float = 2.0,
    steps: int = 100000,
    interval: int = 2000,
    output_dir: str = "output_rb",
):
    """
    執行 Rayleigh-Bénard 對流模擬

    Args:
        ny:         Y 方向格點數（腔體高度 H）
        Ra:         Rayleigh 數
        Pr:         Prandtl 數（0.71=空氣）
        u_ref:      LBM 參考速度（控制 Ma 數 = u_ref*sqrt(3)）
        aspect:     長寬比，nx = aspect * ny
        steps:      最大模擬步數
        interval:   儲存狀態場的間隔
        output_dir: 輸出目錄
    """
    nx = int(aspect * ny)
    H = ny

    nu, kappa, g_lbm, Re_eff, tau_f, tau_g = compute_lbm_params(Ra, Pr, ny, u_ref)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("       RAYLEIGH-BENARD CONVECTION SIMULATION")
    print("=" * 65)
    print(f"  Grid:        {nx} x {ny}")
    print(f"  Ra:          {Ra:.2e}   Pr: {Pr:.2f}")
    print(f"  Re_eff:      {Re_eff:.1f}")
    print(f"  nu:          {nu:.6f}   tau_f = {tau_f:.4f}")
    print(f"  kappa:       {kappa:.6f}   tau_g = {tau_g:.4f}")
    print(f"  g_lbm:       {g_lbm:.2e}")
    print(f"  u_ref:       {u_ref:.3f}   Ma = {u_ref * 1.732:.3f}")
    print("=" * 65)

    # 參數合理性檢查：穩定性要求 tau > 0.5（嚴格 > 0.505 預留裕度）
    if tau_f < 0.505:
        raise ValueError(
            f"tau_f={tau_f:.4f} 過小（< 0.505）。"
            f"請降低 u_ref 或增大 ny。"
        )
    if tau_g < 0.505:
        raise ValueError(
            f"tau_g={tau_g:.4f} 過小（< 0.505）。"
            f"請降低 u_ref 或提高 Pr。"
        )
    if u_ref * 1.732 > 0.3:
        print(f"  [WARNING] Ma={u_ref * 1.732:.3f} > 0.3，可壓縮效應不可忽略。")

    # === 流場求解器 ===
    # cs=0.0 禁用 Smagorinsky LES（自然對流通常 Ra < 1e7 不需要）
    solver = LBMSolver(
        nx=nx, ny=ny,
        re=Re_eff, u_ref=u_ref,
        length_scale=float(H),
        cs=0.0,
    )

    # === 重設 f 為靜止平衡態 ===
    # Why: LBMSolver 預設以 u=(u_ref, 0) 初始化（管道流設計），
    #      RB 對流必須從靜止 u=0 開始，否則水平流壓制對流胞形成。
    _w_rb = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                     dtype=np.float32)
    _f0 = np.zeros((solver.nx_g, solver.ny_g, 9), dtype=np.float32)
    for _k in range(9):
        _f0[:, :, _k] = _w_rb[_k]   # rho=1, u=(0,0)
    solver.f.from_numpy(_f0)
    solver.f_new.from_numpy(_f0)
    solver.u.fill(0.0)

    # === 熱場模組 ===
    # g_gravity > 0 表示浮力向上（熱流體上升），符合底部加熱的物理
    thermal = ThermalModule(
        solver, Pr=Pr, beta=1.0, g_gravity=g_lbm, T_ref=0.5
    )
    thermal.init_temperature(T_bot=1.0, T_top=0.0)

    # === 溫度擾動（觸發 RB 不穩定性）===
    # Why: 線性溫度場在 x 方向均勻，無法自發激發對流；
    #      加入擾動以激發 n=2（雙滾流）模態。
    #
    # 絕熱側壁條件（Neumann ∂T/∂x=0）對應 cos 模態：
    #   T' ~ cos(2πx/nx) * sin(πy/ny)
    #
    #   cos(2πx/nx)：
    #   - 兩壁面（x≈0, x≈nx）處值 ≈ +1（熱異常，驅動壁面上升流）
    #   - 中心（x=nx/2）處值 = -1（冷異常，驅動中心下降流）
    #   - ∂/∂x = 0 在 x=0 和 x=nx ← 滿足 Neumann 絕熱 BC
    #
    # 這在兩個壁面分別激發上升流，在中心激發下降流，形成兩個對稱反轉的方形滾流。
    _g_np = thermal.g.to_numpy()
    _x = np.arange(1, nx + 1, dtype=np.float32)
    _y = np.arange(1, ny + 1, dtype=np.float32)
    _xx, _yy = np.meshgrid(_x, _y, indexing='ij')
    _pert = 0.01 * np.cos(2.0 * np.pi * _xx / nx) * np.sin(np.pi * _yy / ny)
    for _k in range(9):
        _g_np[1:nx+1, 1:ny+1, _k] += (_w_rb[_k] * _pert).astype(np.float32)
    thermal.g.from_numpy(_g_np)
    thermal.g_new.from_numpy(_g_np)

    thermal.register_with_solver()

    # === 流場邊界條件 ===
    # 四壁封閉（de Vahl Davis benchmark 標準設定）
    bc = BoundaryConditions(solver)
    bc.add_no_slip_wall('top')
    bc.add_no_slip_wall('bottom')
    bc.add_no_slip_wall('left')
    bc.add_no_slip_wall('right')
    bc.set_corners_solid()

    solver.apply_boundary_conditions(solver.f)
    solver.apply_boundary_conditions(solver.f_new)

    # === 溫度邊界條件 ===
    tbc = ThermalBoundaryConditions(thermal)
    tbc.add_hot_wall(T_hot=1.0, location='bottom')
    tbc.add_cold_wall(T_cold=0.0, location='top')
    # 左右壁：絕熱（Neumann ∂T/∂x=0，Bounce-Back）
    tbc.add_adiabatic_wall(location='left')
    tbc.add_adiabatic_wall(location='right')

    tbc.apply(thermal.g)
    tbc.apply(thermal.g_new)

    # 記錄初始質量基準
    solver.prepare_diagnostics(reset_baseline=True)
    ti.sync()

    # === 主迴圈 ===
    print(f"\n{'step':>8} | {'mom_res':>10} | {'mass_err':>10} | "
          f"{'T_mid':>7} | {'Nu':>6} | {'step/s':>8}")
    print("-" * 65)

    global_start = time.time()
    nu_history = []

    for step in range(1, steps + 1):
        # 雙緩衝交替：奇數步 f→f_new，偶數步 f_new→f
        f_src = solver.f      if step % 2 == 1 else solver.f_new
        f_dst = solver.f_new  if step % 2 == 1 else solver.f
        g_src = thermal.g     if step % 2 == 1 else thermal.g_new
        g_dst = thermal.g_new if step % 2 == 1 else thermal.g

        # ① 流場推進（force_field_updater 自動呼叫 compute_buoyancy）
        solver.step(f_src, f_dst)

        # ② 溫度場推進（BGK 碰撞 + 串流）
        thermal.step(g_src, g_dst)

        # ③ 施加溫度邊界條件（每步更新 g_dst 的壁面值）
        # 封閉腔體：apply 同時處理上下 Dirichlet + 左右 Bounce-Back 絕熱 BC
        tbc.apply(g_dst)

        if step % 200 == 0:
            solver.prepare_diagnostics(f_dst)
            thermal._update_temperature(g_dst)
            ti.sync()

            elapsed = time.time() - global_start
            speed = step / elapsed if elapsed > 1e-6 else 0.0

            mom_res  = max(solver.mom_res_x[None], solver.mom_res_y[None])
            mass_err = abs(solver.mass_residual[None])
            T_mid    = float(thermal.T.to_numpy()[nx // 2, ny // 2])
            nu_val   = thermal.get_nusselt(T_bot=1.0, T_top=0.0)

            print(f"{step:>8} | {mom_res:>10.3e} | {mass_err:>10.3e} | "
                  f"{T_mid:>7.4f} | {nu_val:>6.2f} | {speed:>8.1f}")

            if step % 2000 == 0:
                nu_history.append((step, nu_val))

        if step % interval == 0:
            # 確保溫度已更新至最新步
            thermal._update_temperature(g_dst)
            T_np   = thermal.T.to_numpy()[1:nx+1, 1:ny+1]
            u_np   = solver.u.to_numpy()[1:nx+1, 1:ny+1]
            rho_np = solver.rho.to_numpy()[1:nx+1, 1:ny+1]
            state  = {
                'T':   T_np,
                'u':   u_np,
                'rho': rho_np,
                'step': step,
                'Ra':  Ra,
                'Pr':  Pr,
            }
            np.save(os.path.join(output_dir, f"state_{step:07d}.npy"), state)
            print(f"  [save] step {step} -> {output_dir}/state_{step:07d}.npy")

    total_time = time.time() - global_start
    print(f"\n--- Completed {steps} steps in {total_time:.1f}s ---")

    # 最終診斷
    solver.prepare_diagnostics(f_dst)
    thermal._update_temperature(g_dst)
    ti.sync()
    final_nu = thermal.get_nusselt(T_bot=1.0, T_top=0.0)
    print(f"\nFinal Nu  = {final_nu:.3f}  (Ra={Ra:.1e}, Pr={Pr:.2f}, aspect={aspect:.1f})")
    if aspect <= 1.1:
        print(f"Reference (1:1 closed cavity): Nu ~ 1.7-2.0 (Ra=1e5)  [Pallares & Herrero 2005]")
    else:
        print(f"Reference (2:1 closed cavity): Nu ~ 2.0-2.3 (Ra=1e5)  [this simulation]")

    # 儲存 Nu 歷史
    np.save(os.path.join(output_dir, "nu_history.npy"), nu_history)
    print(f"Nu history saved to {output_dir}/nu_history.npy")

    return final_nu


def main():
    parser = argparse.ArgumentParser(
        description="Rayleigh-Benard Convection (DDF + Boussinesq)"
    )
    parser.add_argument('--ny',       type=int,   default=64,
                        help='Y resolution (腔體高度 H)')
    parser.add_argument('--Ra',       type=float, default=1e5,
                        help='Rayleigh number')
    parser.add_argument('--Pr',       type=float, default=0.71,
                        help='Prandtl number (air=0.71)')
    parser.add_argument('--u_ref',    type=float, default=0.1,
                        help='LBM reference velocity (controls Ma number)')
    parser.add_argument('--aspect',   type=float, default=2.0,
                        help='Aspect ratio nx/ny')
    parser.add_argument('--steps',    type=int,   default=100000,
                        help='Max simulation steps')
    parser.add_argument('--interval', type=int,   default=2000,
                        help='State save interval')
    parser.add_argument('--output',   type=str,   default='output_rb',
                        help='Output directory')
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    run_rayleigh_benard(
        ny=args.ny,
        Ra=args.Ra,
        Pr=args.Pr,
        u_ref=args.u_ref,
        aspect=args.aspect,
        steps=args.steps,
        interval=args.interval,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
