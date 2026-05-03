"""
Transonic Bump Channel — Euler Validation Case
==============================================

What: 在下壁含 bump 的 2D channel 上求解 transonic Euler 流，觀察局部加速、
      shock-like 壓縮區與 wall pressure 分佈
Why:  在真正做 transonic airfoil 之前，先用幾何與流動機制更單純的 bump
      benchmark 檢查 shock-capturing 路徑是否合理
When: 適用於 EulerSolver 的 curvilinear transonic 驗證，不含黏性與湍流模型
"""

import os

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from cfd_taichi import (
    BoundaryConditionDescriptor,
    CaseRunner,
    CurvilinearGrid2D,
    build_history_payload,
    build_state_payload,
)


def compute_cell_centers(x_node: np.ndarray, y_node: np.ndarray, ni: int, nj: int, NG: int) -> tuple[np.ndarray, np.ndarray]:
    """
    計算 interior cell centers
    """
    xc = 0.25 * (
        x_node[NG:NG+ni, NG:NG+nj] +
        x_node[NG+1:NG+ni+1, NG:NG+nj] +
        x_node[NG:NG+ni, NG+1:NG+nj+1] +
        x_node[NG+1:NG+ni+1, NG+1:NG+nj+1]
    )
    yc = 0.25 * (
        y_node[NG:NG+ni, NG:NG+nj] +
        y_node[NG+1:NG+ni+1, NG:NG+nj] +
        y_node[NG:NG+ni, NG+1:NG+nj+1] +
        y_node[NG+1:NG+ni+1, NG+1:NG+nj+1]
    )
    return xc, yc


def compute_index_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    用 computational index 估計梯度
    """
    grad_i = np.zeros_like(field)
    grad_j = np.zeros_like(field)
    grad_i[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
    grad_i[0, :] = field[1, :] - field[0, :]
    grad_i[-1, :] = field[-1, :] - field[-2, :]
    grad_j[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
    grad_j[:, 0] = field[:, 1] - field[:, 0]
    grad_j[:, -1] = field[:, -1] - field[:, -2]
    return grad_i, grad_j


def compute_bump_diagnostics(
        solver,
        x_node: np.ndarray,
        y_node: np.ndarray,
        length: float,
        bump_center: float,
        ma_inf: float,
        rho_inf: float,
        p_inf: float) -> dict:
    """
    計算 transonic bump 診斷量

    What: 提供 Mach、shock sensor、entropy rise、wall Cp 與 shock 峰值位置
    Why:  bump 驗證的重點不是升阻力，而是是否出現合理的加速區與壓縮峰
    """
    rho_n, u_n, v_n, p_n = solver.get_primitive()
    gamma = solver.gamma
    vel_mag = np.sqrt(u_n**2 + v_n**2)
    c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, 1e-12))
    mach_n = vel_mag / c_n
    entropy = np.log(np.maximum(p_n, 1e-12) / np.maximum(rho_n, 1e-12) ** gamma)
    entropy_inf = np.log(p_inf / rho_inf**gamma)
    entropy_rise = entropy - entropy_inf

    drho_di, drho_dj = compute_index_gradient(rho_n)
    dp_di, dp_dj = compute_index_gradient(p_n)
    shock_sensor = np.maximum(
        np.sqrt(drho_di**2 + drho_dj**2) / max(rho_inf, 1e-12),
        np.sqrt(dp_di**2 + dp_dj**2) / max(p_inf, 1e-12),
    )

    NG = solver.NG
    ni = solver.ni
    nj = solver.nj
    xc, yc = compute_cell_centers(x_node, y_node, ni, nj, NG)
    x_lower = 0.5 * (x_node[NG:NG+ni, NG] + x_node[NG+1:NG+ni+1, NG])
    p_lower = p_n[:, 0]
    q_inf = 0.5 * rho_inf * ma_inf**2
    cp_lower = (p_lower - p_inf) / q_inf

    center_j = int(np.argmin(np.abs(np.mean(yc, axis=0) - 0.5 * np.max(y_node))))
    centerline_x = xc[:, center_j]
    centerline_mach = mach_n[:, center_j]

    shock_mask = (xc > bump_center) & (xc < length - 0.1)
    if np.any(shock_mask):
        masked_sensor = np.where(shock_mask, shock_sensor, -1.0)
        peak_flat = int(np.argmax(masked_sensor))
    else:
        peak_flat = int(np.argmax(shock_sensor))
    peak_idx = np.unravel_index(peak_flat, shock_sensor.shape)

    return {
        "rho": rho_n,
        "u": u_n,
        "v": v_n,
        "p": p_n,
        "mach": mach_n,
        "entropy_rise": entropy_rise,
        "shock_sensor": shock_sensor,
        "mach_max": float(np.max(mach_n)),
        "supersonic_fraction": float(np.mean(mach_n > 1.0)),
        "shock_sensor_max": float(np.max(shock_sensor)),
        "entropy_rise_max": float(np.max(entropy_rise)),
        "shock_peak_x": float(xc[peak_idx]),
        "shock_peak_y": float(yc[peak_idx]),
        "x_lower": x_lower,
        "cp_lower": cp_lower,
        "centerline_x": centerline_x,
        "centerline_mach": centerline_mach,
        "xc": xc,
        "yc": yc,
    }


def plot_wall_cp(x_lower: np.ndarray, cp_lower: np.ndarray, ma: float, save_path: str):
    """
    繪製下壁 Cp 分佈
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_lower, cp_lower, "k-", lw=1.8)
    ax.invert_yaxis()
    ax.axhline(0.0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel(r"$C_p$", fontsize=12)
    ax.set_title(f"Lower-Wall Pressure Coefficient — Transonic Bump, Ma={ma:.3f}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_centerline_mach(x_center: np.ndarray, mach_center: np.ndarray, ma: float, save_path: str):
    """
    繪製 centerline Mach 分佈
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_center, mach_center, "b-", lw=1.8)
    ax.axhline(1.0, color="r", lw=0.8, ls="--", label="Mach 1")
    ax.axhline(ma, color="0.4", lw=0.8, ls=":", label="Freestream Mach")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Mach Number", fontsize=12)
    ax.set_title(f"Centerline Mach — Transonic Bump, Ma={ma:.3f}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_history_series(
        steps: np.ndarray,
        values: np.ndarray,
        ylabel: str,
        title: str,
        save_path: str,
        logy: bool = False):
    """
    繪製歷史量隨步數變化

    What: 將 monitor history 以單一折線圖輸出
    Why:  transonic 驗證不能只看 final snapshot；shock 位置與強度是否平台化同樣重要
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, "k-", lw=1.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def evaluate_plateau(history: list[dict], window: int, shock_tol: float, mach_tol: float) -> dict:
    """
    檢查 shock 位置與 Mach 峰值是否進入平台

    What: 比較最近 window 個報告點的 shock_peak_x 與 mach_max 變動範圍
    Why:  對 transonic bump 而言，shock 位置平台化比單看 residual 更接近真正物理穩態
    """
    if len(history) < max(window, 2):
        return {
            "window": int(window),
            "ready": False,
            "plateau": False,
            "shock_span": np.nan,
            "mach_span": np.nan,
        }

    recent = history[-window:]
    shock_values = np.array([row["shock_peak_x"] for row in recent], dtype=float)
    mach_values = np.array([row["mach_max"] for row in recent], dtype=float)
    shock_span = float(np.max(shock_values) - np.min(shock_values))
    mach_span = float(np.max(mach_values) - np.min(mach_values))
    plateau = (shock_span <= shock_tol) and (mach_span <= mach_tol)
    return {
        "window": int(window),
        "ready": True,
        "plateau": bool(plateau),
        "shock_span": shock_span,
        "mach_span": mach_span,
    }


def _initialize_transonic_bump_case(
    _runner: CaseRunner,
    solver,
    _bc_handle,
    *,
    rho_inf: float,
    u_inf: float,
    v_inf: float,
    p_inf: float,
):
    """
    Transonic bump 的正式初始化流程。

    What:
    - 以均勻自由流初始化 curvilinear Euler 狀態

    Why:
    - benchmark registry 建出的 runner 應該能直接完成初始化，不依賴案例主函式手動補洞
    """
    solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)


def _transonic_bump_diagnostics_hook(
    _runner: CaseRunner,
    solver,
    diagnostics: dict,
    *,
    x_node: np.ndarray,
    y_node: np.ndarray,
    length: float,
    bump_center: float,
    ma_inf: float,
    rho_inf: float,
    p_inf: float,
) -> dict:
    """
    派生 transonic bump 專屬 shock diagnostics。

    Why:
    - bump sweep 與 regression 需要的是 shock observability，而不只是基本流場上下界
    """
    diagnostics = compute_bump_diagnostics(
        solver,
        x_node,
        y_node,
        length=length,
        bump_center=bump_center,
        ma_inf=ma_inf,
        rho_inf=rho_inf,
        p_inf=p_inf,
    )
    diagnostics["residual_rms"] = float(solver.get_residual_norm())
    return diagnostics


def build_transonic_bump_runner(
    *,
    ni: int = 240,
    nj: int = 80,
    ma: float = 0.675,
    length: float = 3.0,
    height: float = 1.0,
    bump_center: float = 1.5,
    bump_width: float = 1.0,
    bump_height: float = 0.12,
    cfl: float = 0.35,
):
    """
    建立 transonic bump 的 CaseRunner。

    What:
    - 組裝 curvilinear grid、Euler solver 與上下 free-slip wall

    Why:
    - 讓 FVM 代表 benchmark 能正式進入 registry / batch workflow
    """
    from fvm_taichi import EulerSolver, generate_bump_channel_grid

    gamma = 1.4
    rho_inf = 1.0
    u_inf = float(ma)
    v_inf = 0.0
    p_inf = rho_inf / gamma

    x_node, y_node = generate_bump_channel_grid(
        ni=ni,
        nj=nj,
        length=length,
        height=height,
        bump_center=bump_center,
        bump_width=bump_width,
        bump_height=bump_height,
        NG=EulerSolver.NG,
    )

    runner = CaseRunner(
        name="transonic_bump_euler",
        method="fvm",
        equation="euler",
        regime="compressible",
        grid=CurvilinearGrid2D(x_node=x_node, y_node=y_node, ng=EulerSolver.NG),
        solver_kwargs={
            "gamma": gamma,
            "cfl": cfl,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.free_slip("bottom"),
            BoundaryConditionDescriptor.free_slip("top"),
        ],
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_transonic_bump_case(
            runner_obj,
            solver_obj,
            bc_handle,
            rho_inf=rho_inf,
            u_inf=u_inf,
            v_inf=v_inf,
            p_inf=p_inf,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _transonic_bump_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            x_node=x_node,
            y_node=y_node,
            length=length,
            bump_center=bump_center,
            ma_inf=ma,
            rho_inf=rho_inf,
            p_inf=p_inf,
        ),
    )
    return runner


def run_transonic_bump(
        ni: int = 240,
        nj: int = 80,
        ma: float = 0.675,
        steps: int = 4000,
        length: float = 3.0,
        height: float = 1.0,
        bump_center: float = 1.5,
        bump_width: float = 1.0,
        bump_height: float = 0.12,
        cfl: float = 0.35,
        report: int = 200,
        tol: float = 1e-5,
        output_dir: str | None = None,
        plateau_window: int = 5,
        shock_tol: float = 0.02,
        mach_tol: float = 0.03):
    """
    執行 transonic bump Euler 驗證
    """
    from fvm_taichi import plot_cell_scalar_field

    if not (0.0 < ma < 1.0):
        raise ValueError(f"Ma must satisfy 0 < Ma < 1, got {ma}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if plateau_window <= 1:
        raise ValueError(f"plateau_window must be > 1, got {plateau_window}")
    if shock_tol <= 0.0 or mach_tol <= 0.0:
        raise ValueError(f"shock_tol and mach_tol must be positive, got {shock_tol}, {mach_tol}")

    rho_inf = 1.0
    u_inf = ma
    v_inf = 0.0
    p_inf = rho_inf / 1.4

    print("=" * 84)
    print("  Transonic Bump Channel — Euler Validation")
    print("=" * 84)
    print(f"  Grid: ni={ni} x nj={nj}, length={length}, height={height}")
    print(f"  Flow: Ma={ma:.3f}, rho_inf={rho_inf:.3f}, p_inf={p_inf:.5f}, CFL={cfl:.2f}")
    print(f"  Bump: center={bump_center:.3f}, width={bump_width:.3f}, height={bump_height:.3f}")
    print(f"  Steps: {steps}, report={report}, tol={tol:.1e}")

    t0 = time.time()
    runner = build_transonic_bump_runner(
        ni=ni,
        nj=nj,
        ma=ma,
        length=length,
        height=height,
        bump_center=bump_center,
        bump_width=bump_width,
        bump_height=bump_height,
        cfl=cfl,
    )
    solver = runner.build_solver()
    runner.configure()
    runner.initialize()
    x_node = runner.grid.x_node
    y_node = runner.grid.y_node

    history = []
    converged = False
    plateau = False
    final_residual = np.nan
    final_step = 0
    plateau_status = {
        "window": int(plateau_window),
        "ready": False,
        "plateau": False,
        "shock_span": np.nan,
        "mach_span": np.nan,
    }

    print(f"\n{'step':>8}  {'res':>10}  {'M_max':>8}  {'M>1%':>8}  {'shock':>9}  {'x_shock':>9}")
    print(f"{'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")

    diag = compute_bump_diagnostics(
        solver, x_node, y_node, length=length, bump_center=bump_center,
        ma_inf=ma, rho_inf=rho_inf, p_inf=p_inf)

    for step in range(1, steps + 1):
        runner.step_once()

        if step % report == 0 or step == 1:
            ti.sync()
            residual = solver.get_residual_norm()
            diag = compute_bump_diagnostics(
                solver, x_node, y_node, length=length, bump_center=bump_center,
                ma_inf=ma, rho_inf=rho_inf, p_inf=p_inf)
            final_residual = residual
            final_step = step
            history.append({
                "step": step,
                "residual": residual,
                "mach_max": diag["mach_max"],
                "supersonic_fraction": diag["supersonic_fraction"],
                "shock_sensor_max": diag["shock_sensor_max"],
                "entropy_rise_max": diag["entropy_rise_max"],
                "shock_peak_x": diag["shock_peak_x"],
                "shock_peak_y": diag["shock_peak_y"],
            })
            print(f"{step:8d}  {residual:10.3e}  {diag['mach_max']:8.4f}  "
                  f"{100.0 * diag['supersonic_fraction']:7.3f}%  {diag['shock_sensor_max']:9.3e}  "
                  f"{diag['shock_peak_x']:9.4f}")

            plateau_status = evaluate_plateau(
                history, window=plateau_window, shock_tol=shock_tol, mach_tol=mach_tol)
            if plateau_status["ready"] and plateau_status["plateau"] and not plateau:
                plateau = True
                print(
                    f"  -> Shock plateau detected: Δx_shock={plateau_status['shock_span']:.4e}, "
                    f"ΔM_max={plateau_status['mach_span']:.4e} over last {plateau_window} reports"
                )

            if residual < tol:
                converged = True
                print(f"\n  ✅ Residual converged at step {step}")
                break

    total_time = time.time() - t0
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")
    print(f"  Final step:          {final_step}")
    print(f"  Residual RMS:        {final_residual:.3e}")
    print(f"  Max Mach:            {diag['mach_max']:.5f}")
    print(f"  Supersonic fraction: {100.0 * diag['supersonic_fraction']:.3f}%")
    print(f"  Shock sensor max:    {diag['shock_sensor_max']:.3e}")
    print(f"  Entropy rise max:    {diag['entropy_rise_max']:.3e}")
    print(f"  Shock peak estimate: x={diag['shock_peak_x']:.4f}, y={diag['shock_peak_y']:.4f}")
    print(f"  Converged:           {converged}")
    if plateau_status["ready"]:
        print(f"  Shock plateau:       {plateau_status['plateau']}")
        print(f"  Plateau spans:       Δx_shock={plateau_status['shock_span']:.4e}, ΔM_max={plateau_status['mach_span']:.4e}")
    else:
        print(f"  Shock plateau:       not enough history (need {plateau_window} reports)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tag = f"ma{int(round(ma * 1000)):04d}_b{int(round(1000 * bump_height)):03d}"

        plot_cell_scalar_field(
            x_node, y_node, diag["mach"],
            field_label="Mach Number",
            title=f"Mach Field — Transonic Bump, Ma={ma:.3f}",
            xlim=(-0.1, length + 0.1), ylim=(0.0, height),
            cmap="hot_r",
            save_path=os.path.join(output_dir, f"mach_{tag}.png"))
        plot_cell_scalar_field(
            x_node, y_node, diag["p"],
            field_label="Pressure",
            title=f"Pressure Field — Transonic Bump, Ma={ma:.3f}",
            xlim=(-0.1, length + 0.1), ylim=(0.0, height),
            cmap="RdBu_r",
            save_path=os.path.join(output_dir, f"pressure_{tag}.png"))
        plot_cell_scalar_field(
            x_node, y_node, diag["shock_sensor"],
            field_label="Shock sensor",
            title=f"Shock Sensor — Transonic Bump, Ma={ma:.3f}",
            xlim=(-0.1, length + 0.1), ylim=(0.0, height),
            cmap="magma",
            save_path=os.path.join(output_dir, f"shock_sensor_{tag}.png"))
        plot_cell_scalar_field(
            x_node, y_node, diag["entropy_rise"],
            field_label="Entropy rise",
            title=f"Entropy Rise — Transonic Bump, Ma={ma:.3f}",
            xlim=(-0.1, length + 0.1), ylim=(0.0, height),
            cmap="plasma",
            save_path=os.path.join(output_dir, f"entropy_rise_{tag}.png"))
        plot_wall_cp(
            diag["x_lower"], diag["cp_lower"], ma=ma,
            save_path=os.path.join(output_dir, f"wall_cp_{tag}.png"))
        plot_centerline_mach(
            diag["centerline_x"], diag["centerline_mach"], ma=ma,
            save_path=os.path.join(output_dir, f"centerline_mach_{tag}.png"))
        history_steps = np.array([row["step"] for row in history], dtype=np.int32)
        if history_steps.size > 0:
            plot_history_series(
                history_steps,
                np.array([row["residual"] for row in history], dtype=np.float64),
                ylabel="Residual RMS",
                title=f"Residual History — Transonic Bump, Ma={ma:.3f}",
                save_path=os.path.join(output_dir, f"residual_history_{tag}.png"),
                logy=True,
            )
            plot_history_series(
                history_steps,
                np.array([row["shock_peak_x"] for row in history], dtype=np.float64),
                ylabel="Shock Peak x",
                title=f"Shock Position History — Transonic Bump, Ma={ma:.3f}",
                save_path=os.path.join(output_dir, f"shock_peak_history_{tag}.png"),
            )
            plot_history_series(
                history_steps,
                np.array([row["mach_max"] for row in history], dtype=np.float64),
                ylabel="Max Mach",
                title=f"Max Mach History — Transonic Bump, Ma={ma:.3f}",
                save_path=os.path.join(output_dir, f"mach_max_history_{tag}.png"),
            )

        np.save(os.path.join(output_dir, "summary.npy"), {
            "ma": ma,
            "ni": ni,
            "nj": nj,
            "length": length,
            "height": height,
            "bump_center": bump_center,
            "bump_width": bump_width,
            "bump_height": bump_height,
            "final_step": final_step,
            "residual": final_residual,
            "mach_max": diag["mach_max"],
            "supersonic_fraction": diag["supersonic_fraction"],
            "shock_sensor_max": diag["shock_sensor_max"],
            "entropy_rise_max": diag["entropy_rise_max"],
            "shock_peak_x": diag["shock_peak_x"],
            "shock_peak_y": diag["shock_peak_y"],
            "converged": converged,
            "shock_plateau": plateau_status["plateau"],
            "shock_plateau_ready": plateau_status["ready"],
            "shock_peak_span": plateau_status["shock_span"],
            "mach_peak_span": plateau_status["mach_span"],
            "plateau_window": plateau_window,
            "shock_tol": shock_tol,
            "mach_tol": mach_tol,
            "runtime_seconds": total_time,
        })
        history_payload = build_history_payload(
            solver=solver,
            steps=[row["step"] for row in history],
            params={
                "ma": ma,
                "ni": ni,
                "nj": nj,
                "length": length,
                "height": height,
                "bump_center": bump_center,
                "bump_width": bump_width,
                "bump_height": bump_height,
            },
            extra_series={
                "residual": [row["residual"] for row in history],
                "mach_max": [row["mach_max"] for row in history],
                "supersonic_fraction": [row["supersonic_fraction"] for row in history],
                "shock_sensor_max": [row["shock_sensor_max"] for row in history],
                "entropy_rise_max": [row["entropy_rise_max"] for row in history],
                "shock_peak_x": [row["shock_peak_x"] for row in history],
                "shock_peak_y": [row["shock_peak_y"] for row in history],
            },
        )
        np.save(os.path.join(output_dir, "history.npy"), history_payload, allow_pickle=True)

        state_payload = build_state_payload(
            solver=solver,
            fields={
                "u": np.stack([diag["u"], diag["v"]], axis=-1),
                "rho": diag["rho"],
                "p": diag["p"],
                "mach": diag["mach"],
            },
            step=final_step,
            time_value=float(final_step),
            additional_data={
                "x_node": x_node,
                "y_node": y_node,
                "shock_sensor": diag["shock_sensor"],
                "entropy_rise": diag["entropy_rise"],
                "x_lower": diag["x_lower"],
                "cp_lower": diag["cp_lower"],
                "centerline_x": diag["centerline_x"],
                "centerline_mach": diag["centerline_mach"],
            },
        )
        np.save(os.path.join(output_dir, "state_final.npy"), state_payload, allow_pickle=True)

    return {
        "converged": converged,
        "final_step": final_step,
        "residual": final_residual,
        "mach_max": diag["mach_max"],
        "supersonic_fraction": diag["supersonic_fraction"],
        "shock_sensor_max": diag["shock_sensor_max"],
        "entropy_rise_max": diag["entropy_rise_max"],
        "shock_peak_x": diag["shock_peak_x"],
        "shock_peak_y": diag["shock_peak_y"],
        "shock_plateau": plateau_status["plateau"],
        "shock_peak_span": plateau_status["shock_span"],
        "mach_peak_span": plateau_status["mach_span"],
        "runtime_seconds": total_time,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Transonic bump channel Euler validation")
    parser.add_argument("--ni", type=int, default=240, help="Streamwise interior cells")
    parser.add_argument("--nj", type=int, default=80, help="Wall-normal interior cells")
    parser.add_argument("--ma", type=float, default=0.675, help="Freestream Mach number")
    parser.add_argument("--steps", type=int, default=4000, help="Maximum time steps")
    parser.add_argument("--length", type=float, default=3.0, help="Channel length")
    parser.add_argument("--height", type=float, default=1.0, help="Channel height")
    parser.add_argument("--bump_center", type=float, default=1.5, help="Bump center x")
    parser.add_argument("--bump_width", type=float, default=1.0, help="Bump support width")
    parser.add_argument("--bump_height", type=float, default=0.12, help="Bump height")
    parser.add_argument("--cfl", type=float, default=0.35, help="Explicit CFL")
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Residual tolerance")
    parser.add_argument("--plateau_window", type=int, default=5, help="History window for shock plateau check")
    parser.add_argument("--shock_tol", type=float, default=0.02, help="Shock-position plateau tolerance")
    parser.add_argument("--mach_tol", type=float, default=0.03, help="Max-Mach plateau tolerance")
    parser.add_argument(
        "--output",
        type=str,
        default="output_transonic_bump",
        help="Output directory (empty string disables)",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        run_transonic_bump(
            ni=args.ni,
            nj=args.nj,
            ma=args.ma,
            steps=args.steps,
            length=args.length,
            height=args.height,
            bump_center=args.bump_center,
            bump_width=args.bump_width,
            bump_height=args.bump_height,
            cfl=args.cfl,
            report=args.report,
            tol=args.tol,
            output_dir=args.output if args.output else None,
            plateau_window=args.plateau_window,
            shock_tol=args.shock_tol,
            mach_tol=args.mach_tol,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ Transonic bump case failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
