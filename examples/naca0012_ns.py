"""
NACA0012 Laminar Navier-Stokes — O-grid + Curved No-Slip Wall
==============================================================

What: NACA0012 翼型有限 Reynolds 數模擬，使用 body-fitted O-grid、
      curved-wall no-slip BC 與亞音速 characteristic far-field
Why:  這是把現有 Euler 翼型案例推進到真正 finite-Re 黏性案例的第一步：
      - 翼面無滑移
      - 壁面剪應力參與阻力
      - 可分解 pressure / viscous drag
      - 可切換 experimental constant-eddy-viscosity closure
When: 適用於 subsonic baseline 與 experimental transonic/high-Re 探索；
      目前 turbulence path 仍不是已驗證工程工具
"""

import os

import argparse
import time

import numpy as np
import taichi as ti

from cfd_taichi import (
    BoundaryConditionDescriptor,
    CaseRunner,
    CurvilinearGrid2D,
    SolverControlDescriptor,
    build_history_payload,
    build_state_payload,
)


def compute_reference_scales(
        ma: float,
        alpha_rad: float,
        gamma: float,
        re: float,
        chord: float = 1.0) -> dict:
    """
    整理自由流與 transport reference scales
    """
    rho_inf = 1.0
    u_inf = ma * np.cos(alpha_rad)
    v_inf = ma * np.sin(alpha_rad)
    v_mag = float(np.sqrt(u_inf**2 + v_inf**2))
    p_inf = rho_inf / gamma
    a_inf = float(np.sqrt(gamma * p_inf / rho_inf))
    q_inf = 0.5 * rho_inf * v_mag**2
    mu_inf = rho_inf * v_mag * chord / re

    return {
        "rho_inf": rho_inf,
        "u_inf": float(u_inf),
        "v_inf": float(v_inf),
        "v_mag": v_mag,
        "p_inf": float(p_inf),
        "a_inf": a_inf,
        "q_inf": q_inf,
        "mu_inf": mu_inf,
        "re": float(re),
        "chord": float(chord),
    }


def validate_turbulence_inputs(
        turbulence_model: str,
        eddy_viscosity_ratio: float,
        turbulent_prandtl: float,
        sa_nu_tilde_inf_ratio: float):
    """
    驗證 turbulence closure 參數
    """
    if turbulence_model not in {"laminar", "constant_eddy_viscosity", "spalart_allmaras"}:
        raise ValueError(
            "turbulence_model must be 'laminar', 'constant_eddy_viscosity', or "
            "'spalart_allmaras', "
            f"got {turbulence_model}"
        )
    if eddy_viscosity_ratio < 0.0:
        raise ValueError(f"eddy_viscosity_ratio must be >= 0, got {eddy_viscosity_ratio}")
    if turbulent_prandtl <= 0.0:
        raise ValueError(f"turbulent_prandtl must be positive, got {turbulent_prandtl}")
    if sa_nu_tilde_inf_ratio <= 0.0:
        raise ValueError(
            f"sa_nu_tilde_inf_ratio must be positive, got {sa_nu_tilde_inf_ratio}")
    if turbulence_model == "laminar" and eddy_viscosity_ratio > 0.0:
        raise ValueError(
            "eddy_viscosity_ratio must be 0 when turbulence_model='laminar'; "
            "use turbulence_model='constant_eddy_viscosity' to enable eddy viscosity."
        )


def validate_time_marching_inputs(
        time_marching: str,
        target_sim_time: float | None):
    """
    驗證時間推進模式輸入
    """
    if time_marching not in {"global", "local_pseudo"}:
        raise ValueError(
            "time_marching must be 'global' or 'local_pseudo', "
            f"got {time_marching}"
        )
    if target_sim_time is not None and target_sim_time <= 0.0:
        raise ValueError(
            f"target_sim_time must be positive when provided, got {target_sim_time}"
        )
    if time_marching == "local_pseudo" and target_sim_time is not None:
        raise ValueError(
            "target_sim_time is only valid for global physical-time marching; "
            "local_pseudo uses residual/step convergence instead."
        )


def validate_pseudo_time_controls(
        time_marching: str,
        cfl: float,
        pseudo_cfl_start: float | None,
        pseudo_ramp_steps: int,
        pseudo_precond_ref_mach: float,
        pseudo_precond_min_scale: float,
        residual_smoothing_eps: float,
        residual_smoothing_passes: int,
        adaptive_pseudo: bool,
        adaptive_interval: int,
        adaptive_cfl_growth: float,
        adaptive_cfl_shrink: float,
        adaptive_target_ratio: float,
        adaptive_fail_ratio: float,
        adaptive_smoothing_max_eps: float | None):
    """
    驗證 pseudo-time ramp / preconditioning 輸入
    """
    if pseudo_cfl_start is not None and (pseudo_cfl_start <= 0.0 or pseudo_cfl_start > cfl):
        raise ValueError(
            f"pseudo_cfl_start must satisfy 0 < pseudo_cfl_start <= cfl ({cfl}), "
            f"got {pseudo_cfl_start}"
        )
    if pseudo_ramp_steps < 0:
        raise ValueError(f"pseudo_ramp_steps must be >= 0, got {pseudo_ramp_steps}")
    if pseudo_precond_ref_mach <= 0.0:
        raise ValueError(
            f"pseudo_precond_ref_mach must be positive, got {pseudo_precond_ref_mach}"
        )
    if not (0.0 < pseudo_precond_min_scale <= 1.0):
        raise ValueError(
            "pseudo_precond_min_scale must satisfy 0 < pseudo_precond_min_scale <= 1, "
            f"got {pseudo_precond_min_scale}"
        )
    if residual_smoothing_eps < 0.0:
        raise ValueError(
            f"residual_smoothing_eps must be >= 0, got {residual_smoothing_eps}"
        )
    if residual_smoothing_passes < 0:
        raise ValueError(
            f"residual_smoothing_passes must be >= 0, got {residual_smoothing_passes}"
        )
    if adaptive_interval <= 0:
        raise ValueError(f"adaptive_interval must be positive, got {adaptive_interval}")
    if adaptive_cfl_growth < 1.0:
        raise ValueError(f"adaptive_cfl_growth must be >= 1, got {adaptive_cfl_growth}")
    if not (0.0 < adaptive_cfl_shrink <= 1.0):
        raise ValueError(
            "adaptive_cfl_shrink must satisfy 0 < adaptive_cfl_shrink <= 1, "
            f"got {adaptive_cfl_shrink}"
        )
    if not (0.0 < adaptive_target_ratio <= adaptive_fail_ratio):
        raise ValueError(
            "adaptive_target_ratio and adaptive_fail_ratio must satisfy "
            f"0 < target <= fail, got {adaptive_target_ratio}, {adaptive_fail_ratio}"
        )
    if adaptive_smoothing_max_eps is not None and adaptive_smoothing_max_eps < residual_smoothing_eps:
        raise ValueError(
            "adaptive_smoothing_max_eps must be >= residual_smoothing_eps, "
            f"got {adaptive_smoothing_max_eps} < {residual_smoothing_eps}"
        )
    controls_are_active = (
        pseudo_cfl_start is not None
        or pseudo_ramp_steps > 0
        or pseudo_precond_ref_mach != 1.0
        or pseudo_precond_min_scale != 1.0
        or residual_smoothing_eps > 0.0
        or residual_smoothing_passes > 0
        or adaptive_pseudo
    )
    if time_marching == "global" and controls_are_active:
        raise ValueError(
            "pseudo-time ramp / preconditioning controls require "
            "time_marching='local_pseudo'."
        )


def resolve_initial_condition_mode(
        initial_condition: str,
        turbulence_model: str,
        re: float) -> str:
    """
    決定實際採用的初始場模式

    What: 將 CLI/函式輸入的 `auto|uniform|wall_ramp` 解析成具體模式
    Why:  高 Re 與 turbulence case 若仍用 uniform freestream 初始化，
          會在 no-slip wall 旁觸發很長的假暫態；auto 應偏向更平順的 wall-aware 初始場
    """
    if initial_condition not in {"auto", "uniform", "wall_ramp"}:
        raise ValueError(
            "initial_condition must be 'auto', 'uniform', or 'wall_ramp', "
            f"got {initial_condition}"
        )
    if initial_condition != "auto":
        return initial_condition
    if turbulence_model != "laminar" or re >= 1.0e4:
        return "wall_ramp"
    return "uniform"


def build_initial_primitive_state(
        solver,
        ref: dict,
        init_mode: str,
        init_bl_thickness: float) -> tuple[np.ndarray, np.ndarray | None]:
    """
    建立 wall-aware primitive 初始場

    What: 產生 `(ni, nj, 4)` primitive field，並在 SA 模式下提供對應 `nu_tilde`
    Why:  對 finite-Re no-slip airfoil，uniform freestream 會在第一步強行生成
          極大的壁面速度跳變；以 wall distance 做平滑 ramp 能減少長暫態
    """
    if init_bl_thickness <= 0.0:
        raise ValueError(
            f"init_bl_thickness must be positive, got {init_bl_thickness}"
        )

    ni = solver.ni
    nj = solver.nj
    rho_inf = float(ref["rho_inf"])
    u_inf = float(ref["u_inf"])
    v_inf = float(ref["v_inf"])
    p_inf = float(ref["p_inf"])

    w_np = np.zeros((ni, nj, 4), dtype=np.float32)
    w_np[:, :, 0] = rho_inf
    w_np[:, :, 3] = p_inf

    if init_mode == "uniform":
        w_np[:, :, 1] = u_inf
        w_np[:, :, 2] = v_inf
        return w_np, None

    NG = solver.NG
    wall_dist = solver.wall_dist.to_numpy()[NG:NG + ni, NG:NG + nj]
    ramp = 1.0 - np.exp(-wall_dist / init_bl_thickness)
    ramp = np.clip(ramp, 0.0, 1.0).astype(np.float32)

    w_np[:, :, 1] = u_inf * ramp
    w_np[:, :, 2] = v_inf * ramp

    nu_tilde_np = None
    transport = solver.get_transport_coefficients()
    if transport["turbulence_model"] == "spalart_allmaras":
        nu_inf = transport["sa_nu_tilde_inf_ratio"] * transport["mu"] / max(rho_inf, 1e-12)
        nu_tilde_np = (nu_inf * ramp).astype(np.float32)
    return w_np, nu_tilde_np


def decompose_lift_drag(fx: float, fy: float, alpha_rad: float) -> tuple:
    """
    將笛卡兒力分解到升力 / 阻力方向
    """
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    lift = fy * cos_a - fx * sin_a
    drag = fx * cos_a + fy * sin_a
    return float(lift), float(drag)


def compute_aero_forces(solver, alpha_rad: float, ref: dict) -> dict:
    """
    由壓力與黏性 traction 計算氣動力

    What:
    - pressure force = -p S_wall
    - viscous force  = tau · S_wall
    Why: finite-Re 案例的 drag 不能再只看 pressure contribution
    """
    solver.update_viscous_fluxes()
    ti.sync()

    NG = solver.NG
    ni = solver.ni
    rho_n, u_n, v_n, p_n = solver.get_primitive()
    p_wall = p_n[:, 0]

    s_wall = solver.S_j.to_numpy()[NG:NG + ni, NG - 1, :]
    visc_wall = solver.VFlux_j.to_numpy()[NG:NG + ni, NG - 1, :]
    ds_wall = np.maximum(np.linalg.norm(s_wall, axis=1), 1e-12)
    t_wall = np.column_stack([s_wall[:, 1] / ds_wall, -s_wall[:, 0] / ds_wall])

    fx_p = float(-np.sum(p_wall * s_wall[:, 0]))
    fy_p = float(-np.sum(p_wall * s_wall[:, 1]))
    fx_v = float(np.sum(visc_wall[:, 1]))
    fy_v = float(np.sum(visc_wall[:, 2]))

    fx = fx_p + fx_v
    fy = fy_p + fy_v

    lift_p, drag_p = decompose_lift_drag(fx_p, fy_p, alpha_rad)
    lift_v, drag_v = decompose_lift_drag(fx_v, fy_v, alpha_rad)
    lift, drag = decompose_lift_drag(fx, fy, alpha_rad)

    denom = ref["q_inf"] * ref["chord"]
    cp = (p_wall - ref["p_inf"]) / ref["q_inf"]
    wall_traction = np.column_stack([visc_wall[:, 1], visc_wall[:, 2]]) / ds_wall[:, None]
    tau_t = np.sum(wall_traction * t_wall, axis=1)
    cf = tau_t / ref["q_inf"]
    transport = solver.get_transport_coefficients()
    mu_wall = max(float(transport["mu"]), 1e-12)
    wall_dist = solver.wall_dist.to_numpy()[NG:NG + ni, NG]
    vel_adj = np.column_stack([u_n[:, 0], v_n[:, 0]])
    u_t_adj = np.sum(vel_adj * t_wall, axis=1)
    u_tau = np.sqrt(np.abs(tau_t) / np.maximum(rho_n[:, 0], 1e-12))
    y_plus = rho_n[:, 0] * u_tau * wall_dist / mu_wall
    u_plus = np.abs(u_t_adj) / np.maximum(u_tau, 1e-12)

    return {
        "CL": lift / denom,
        "CD": drag / denom,
        "CL_p": lift_p / denom,
        "CD_p": drag_p / denom,
        "CL_v": lift_v / denom,
        "CD_v": drag_v / denom,
        "Fx": fx,
        "Fy": fy,
        "Cp": cp,
        "Cf": cf,
        "tau_t": tau_t,
        "u_tau": u_tau,
        "y_plus": y_plus,
        "u_plus": u_plus,
        "wall_dist": wall_dist,
        "y_plus_min": float(np.min(y_plus)),
        "y_plus_mean": float(np.mean(y_plus)),
        "y_plus_max": float(np.max(y_plus)),
    }


def get_airfoil_cc(x_node, y_node, ni, NG=2):
    """
    翼面鄰近 cell center 座標
    """
    x_cc = 0.5 * (x_node[NG:NG + ni, NG] + x_node[NG + 1:NG + ni + 1, NG])
    y_cc = 0.5 * (y_node[NG:NG + ni, NG] + y_node[NG + 1:NG + ni + 1, NG])
    return x_cc, y_cc


def compute_geometry_monitor(solver) -> dict:
    """
    預計算監控所需的幾何尺度

    What: 從 face 長度與 cell 面積估計最小特徵尺度 h_min
    Why:  O-grid 下不能再用單一 dx/dy；監控 CFL 與輸出 state 需要一個
          與幾何相容的 cell size estimate
    """
    NG = solver.NG
    vol = solver.vol.to_numpy()[NG:NG + solver.ni, NG:NG + solver.nj]
    s_i = solver.S_i.to_numpy()
    s_j = solver.S_j.to_numpy()

    h_min = np.inf
    for ii in range(solver.ni):
        i = NG + ii
        for jj in range(solver.nj):
            j = NG + jj
            ds_i = max(
                np.linalg.norm(s_i[i - 1, j]),
                np.linalg.norm(s_i[i, j]),
                1e-12,
            )
            ds_j = max(
                np.linalg.norm(s_j[i, j - 1]),
                np.linalg.norm(s_j[i, j]),
                1e-12,
            )
            li = vol[ii, jj] / ds_i
            lj = vol[ii, jj] / ds_j
            h_min = min(h_min, li, lj)

    return {
        "vol": vol,
        "h_min": float(h_min),
    }


def compute_index_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    以 computational index 計算近似梯度

    What: 使用中心差分與單邊差分估計 i/j 方向梯度
    Why:  這裡的目標是 shock observability，而不是高精度物理梯度重建；
          在 transonic diagnostics 中，穩定且便宜的 sensor 比高階公式更重要
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


def compute_flow_diagnostics(
        solver,
        ref: dict,
        dt: float,
        sim_time: float,
        initial_mass: float,
        geom: dict,
        x_node: np.ndarray,
        y_node: np.ndarray) -> dict:
    """
    計算全域流場監控量

    What: 提供 residual 之外的密度/壓力/速度/馬赫數/質量守恆監控
    Why:  黏性翼型案例若只看 CL/CD 很容易誤判；需要基礎守恆量與場範圍
    """
    rho_n, u_n, v_n, p_n = solver.get_primitive()
    vel_mag = np.sqrt(u_n**2 + v_n**2)
    c_n = np.sqrt(np.maximum(solver.gamma * p_n / rho_n, 1e-12))
    mach_n = vel_mag / c_n
    total_mass = float(np.sum(rho_n * geom["vol"]))
    mass_error = abs(total_mass - initial_mass) / max(abs(initial_mass), 1e-12)
    wave_speed = vel_mag + c_n
    cfl_est = float(dt * np.max(wave_speed) / max(geom["h_min"], 1e-12))
    dt_diag = solver.get_dt_diagnostics()

    entropy = np.log(np.maximum(p_n, 1e-12) / np.maximum(rho_n, 1e-12) ** solver.gamma)
    entropy_inf = np.log(ref["p_inf"] / ref["rho_inf"] ** solver.gamma)
    entropy_rise = entropy - entropy_inf

    drho_di, drho_dj = compute_index_gradient(rho_n)
    dp_di, dp_dj = compute_index_gradient(p_n)
    schlieren = np.sqrt(drho_di**2 + drho_dj**2) / max(ref["rho_inf"], 1e-12)
    pressure_grad = np.sqrt(dp_di**2 + dp_dj**2) / max(ref["p_inf"], 1e-12)
    shock_sensor = np.maximum(schlieren, pressure_grad)
    supersonic_fraction = float(np.mean(mach_n > 1.0))
    NG = solver.NG
    x_cc = 0.25 * (
        x_node[NG:NG + solver.ni, NG:NG + solver.nj]
        + x_node[NG + 1:NG + solver.ni + 1, NG:NG + solver.nj]
        + x_node[NG:NG + solver.ni, NG + 1:NG + solver.nj + 1]
        + x_node[NG + 1:NG + solver.ni + 1, NG + 1:NG + solver.nj + 1]
    )
    y_cc = 0.25 * (
        y_node[NG:NG + solver.ni, NG:NG + solver.nj]
        + y_node[NG + 1:NG + solver.ni + 1, NG:NG + solver.nj]
        + y_node[NG:NG + solver.ni, NG + 1:NG + solver.nj + 1]
        + y_node[NG + 1:NG + solver.ni + 1, NG + 1:NG + solver.nj + 1]
    )
    shock_mask = (
        (x_cc > -0.1) & (x_cc < 1.5)
        & (np.abs(y_cc) < 0.6)
    )
    if np.any(shock_mask):
        masked_sensor = np.where(shock_mask, shock_sensor, -1.0)
        shock_peak_flat = int(np.argmax(masked_sensor))
    else:
        shock_peak_flat = int(np.argmax(shock_sensor))
    shock_peak_idx = np.unravel_index(shock_peak_flat, shock_sensor.shape)

    return {
        "time": float(sim_time),
        "dt": float(dt),
        "rho_min": float(np.min(rho_n)),
        "rho_max": float(np.max(rho_n)),
        "p_min": float(np.min(p_n)),
        "p_max": float(np.max(p_n)),
        "u_max": float(np.max(vel_mag)),
        "mach_max": float(np.max(mach_n)),
        "supersonic_fraction": supersonic_fraction,
        "entropy_rise_max": float(np.max(entropy_rise)),
        "entropy_rise_mean": float(np.mean(np.maximum(entropy_rise, 0.0))),
        "shock_sensor_max": float(np.max(shock_sensor)),
        "shock_sensor_p99": float(np.percentile(shock_sensor, 99.0)),
        "shock_peak_x": float(x_cc[shock_peak_idx]),
        "shock_peak_y": float(y_cc[shock_peak_idx]),
        "mass": total_mass,
        "mass_error": mass_error,
        "cfl_est": cfl_est,
        "dt_min": float(dt_diag["dt_min"]),
        "dt_mean": float(dt_diag["dt_mean"]),
        "dt_max": float(dt_diag["dt_max"]),
        "pseudo_cfl": float(dt_diag["pseudo_cfl"]),
        "pseudo_cfl_target": float(dt_diag["pseudo_cfl_target"]),
        "pseudo_precond_scale_min": float(dt_diag["pseudo_precond_scale_min"]),
        "pseudo_precond_scale_mean": float(dt_diag["pseudo_precond_scale_mean"]),
        "pseudo_precond_scale_max": float(dt_diag["pseudo_precond_scale_max"]),
        "residual_smoothing_eps": float(dt_diag["residual_smoothing_eps"]),
        "residual_smoothing_base_eps": float(dt_diag["residual_smoothing_base_eps"]),
        "residual_smoothing_max_eps": float(dt_diag["residual_smoothing_max_eps"]),
        "residual_smoothing_passes": int(dt_diag["residual_smoothing_passes"]),
        "pseudo_adaptive_enabled": bool(dt_diag["pseudo_adaptive_enabled"]),
        "pseudo_residual_ratio": float(dt_diag["pseudo_residual_ratio"]),
        "time_marching_mode": dt_diag["mode"],
        "rho": rho_n,
        "u": u_n,
        "v": v_n,
        "p": p_n,
        "mach": mach_n,
        "entropy_rise": entropy_rise,
        "shock_sensor": shock_sensor,
    }


def save_state_file(
        output_dir: str,
        step: int,
        sim_time: float,
        solver,
        flow: dict,
        aero: dict,
        x_node: np.ndarray,
        y_node: np.ndarray,
        x_cc: np.ndarray,
        y_cc: np.ndarray):
    """
    保存單步 state 檔案

    What: 以 `.npy` 保存流場、幾何與 surface diagnostics
    Why:  單步 state 是後處理、debug、回歸比較的最小可重現單位
    """
    state_dir = os.path.join(output_dir, "states")
    os.makedirs(state_dir, exist_ok=True)

    state = build_state_payload(
        solver=solver,
        fields={
            "u": np.stack([flow["u"], flow["v"]], axis=-1),
            "rho": flow["rho"],
            "p": flow["p"],
            "mach": flow["mach"],
        },
        step=step,
        time_value=sim_time,
        additional_data={
            "entropy_rise": flow["entropy_rise"],
            "shock_sensor": flow["shock_sensor"],
            "x_node": x_node,
            "y_node": y_node,
            "surface_x": x_cc,
            "surface_y": y_cc,
            "cp": aero["Cp"],
            "cf": aero["Cf"],
            "tau_t": aero["tau_t"],
            "u_tau": aero["u_tau"],
            "y_plus": aero["y_plus"],
            "u_plus": aero["u_plus"],
            "wall_dist": aero["wall_dist"],
            "cl": float(aero["CL"]),
            "cd": float(aero["CD"]),
            "cd_p": float(aero["CD_p"]),
            "cd_v": float(aero["CD_v"]),
            "supersonic_fraction": float(flow["supersonic_fraction"]),
            "entropy_rise_max": float(flow["entropy_rise_max"]),
            "shock_sensor_max": float(flow["shock_sensor_max"]),
            "shock_peak_x": float(flow["shock_peak_x"]),
            "shock_peak_y": float(flow["shock_peak_y"]),
            "y_plus_min": float(aero["y_plus_min"]),
            "y_plus_mean": float(aero["y_plus_mean"]),
            "y_plus_max": float(aero["y_plus_max"]),
        },
    )
    np.save(os.path.join(state_dir, f"state_{step:07d}.npy"), state)


def save_history_file(output_dir: str, history: list, params: dict):
    """
    保存 history 檔案
    """
    history_payload = build_history_payload(
        solver=type(
            "_NacaHistorySolver",
            (),
            {
                "solver_family": "fvm",
                "equation_set": "navier_stokes",
                "regime": "compressible",
            },
        )(),
        steps=[row["step"] for row in history],
        mass_error=[row["mass_error"] for row in history],
        u_max=[row["u_max"] for row in history],
        cfl=[row["cfl_est"] for row in history],
        params=params,
        extra_series={
            "time": [row["time"] for row in history],
            "dt": [row["dt"] for row in history],
            "dt_min": [row["dt_min"] for row in history],
            "dt_mean": [row["dt_mean"] for row in history],
            "dt_max": [row["dt_max"] for row in history],
            "pseudo_cfl": [row["pseudo_cfl"] for row in history],
            "pseudo_cfl_target": [row["pseudo_cfl_target"] for row in history],
            "pseudo_precond_scale_min": [row["pseudo_precond_scale_min"] for row in history],
            "pseudo_precond_scale_mean": [row["pseudo_precond_scale_mean"] for row in history],
            "pseudo_precond_scale_max": [row["pseudo_precond_scale_max"] for row in history],
            "residual_smoothing_eps": [row["residual_smoothing_eps"] for row in history],
            "residual_smoothing_base_eps": [row["residual_smoothing_base_eps"] for row in history],
            "residual_smoothing_max_eps": [row["residual_smoothing_max_eps"] for row in history],
            "residual_smoothing_passes": [row["residual_smoothing_passes"] for row in history],
            "pseudo_adaptive_enabled": [row["pseudo_adaptive_enabled"] for row in history],
            "pseudo_residual_ratio": [row["pseudo_residual_ratio"] for row in history],
            "residual": [row["residual"] for row in history],
            "cl": [row["CL"] for row in history],
            "cd": [row["CD"] for row in history],
            "cl_p": [row["CL_p"] for row in history],
            "cd_p": [row["CD_p"] for row in history],
            "cl_v": [row["CL_v"] for row in history],
            "cd_v": [row["CD_v"] for row in history],
            "rho_min": [row["rho_min"] for row in history],
            "rho_max": [row["rho_max"] for row in history],
            "p_min": [row["p_min"] for row in history],
            "p_max": [row["p_max"] for row in history],
            "mach_max": [row["mach_max"] for row in history],
            "supersonic_fraction": [row["supersonic_fraction"] for row in history],
            "entropy_rise_max": [row["entropy_rise_max"] for row in history],
            "shock_sensor_max": [row["shock_sensor_max"] for row in history],
            "shock_peak_x": [row["shock_peak_x"] for row in history],
            "shock_peak_y": [row["shock_peak_y"] for row in history],
            "y_plus_mean": [row["y_plus_mean"] for row in history],
            "y_plus_max": [row["y_plus_max"] for row in history],
        },
    )
    np.save(os.path.join(output_dir, "history.npy"), history_payload, allow_pickle=True)


def save_summary_file(output_dir: str, summary: dict):
    """
    保存 final summary 檔案
    """
    np.save(os.path.join(output_dir, "summary.npy"), summary)


def save_surface_file(output_dir: str, x_cc: np.ndarray, y_cc: np.ndarray, aero: dict):
    """
    保存 final surface diagnostics
    """
    np.save(os.path.join(output_dir, "surface_final.npy"), {
        "x": x_cc,
        "y": y_cc,
        "cp": aero["Cp"],
        "cf": aero["Cf"],
        "tau_t": aero["tau_t"],
        "u_tau": aero["u_tau"],
        "y_plus": aero["y_plus"],
        "u_plus": aero["u_plus"],
        "wall_dist": aero["wall_dist"],
    })


def plot_history_series(
        steps: np.ndarray,
        values: np.ndarray,
        ylabel: str,
        title: str,
        save_path: str,
        logy: bool = False):
    """
    繪製歷史量隨步數變化

    What: 將 monitor history 輸出為單一折線圖
    Why:  transonic NACA 要判斷 shock 與 force 是否穩定，不能只看 final snapshot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def describe_series_trend(values: np.ndarray, rel_tol: float = 1e-3) -> dict:
    """
    估計時間序列是單調趨穩還是帶有震盪

    What: 用相鄰差分的號誌翻轉次數與主方向估計趨勢
    Why:  solver history 不能只看最後一個值；需要區分 monotonic relaxation
          與 oscillatory / stalled transient
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return {
            "sign_changes": 0,
            "active_count": 0,
            "direction": "insufficient",
            "monotonic_like": True,
        }

    diffs = np.diff(values)
    scale = max(np.max(np.abs(values)), 1e-12)
    active = np.abs(diffs) > rel_tol * scale
    active_diffs = diffs[active]
    if active_diffs.size == 0:
        return {
            "sign_changes": 0,
            "active_count": 0,
            "direction": "flat",
            "monotonic_like": True,
        }

    signs = np.sign(active_diffs)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0.0))
    direction = "increasing" if np.mean(active_diffs) > 0.0 else "decreasing"
    return {
        "sign_changes": sign_changes,
        "active_count": int(active_diffs.size),
        "direction": direction,
        "monotonic_like": sign_changes == 0,
    }


def plot_force_history_diagnostics(history_dict: dict, save_path: str):
    """
    繪製 force / residual 時間歷程判讀圖

    What: 同圖輸出 CL/CD、pressure/viscous drag 與 residual history
    Why:  高 Re baseline 需要先判斷是 monotonic relaxation 還是 oscillatory drift，
          才知道下一步該延長物理時間還是回頭改模型
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray(history_dict["steps"], dtype=float)
    time = np.asarray(history_dict["time"], dtype=float)
    cl = np.asarray(history_dict["cl"], dtype=float)
    cd = np.asarray(history_dict["cd"], dtype=float)
    cd_p = np.asarray(history_dict["cd_p"], dtype=float)
    cd_v = np.asarray(history_dict["cd_v"], dtype=float)
    residual = np.asarray(history_dict["residual"], dtype=float)

    trend_cl = describe_series_trend(cl)
    trend_cd = describe_series_trend(cd)
    trend_res = describe_series_trend(residual)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(time, cl, "b-o", ms=3, lw=1.5, label="CL")
    axes[0].plot(time, cd, "r-o", ms=3, lw=1.5, label="CD")
    axes[0].set_ylabel("Force coeff.", fontsize=12)
    axes[0].set_title("Force History Diagnostics", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(time, cd_p, "k-o", ms=3, lw=1.4, label="CD_p")
    axes[1].plot(time, cd_v, color="tab:orange", marker="o", ms=3, lw=1.4, label="CD_v")
    axes[1].set_ylabel("Drag split", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    axes[2].plot(time, residual, "g-o", ms=3, lw=1.4)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Physical time", fontsize=12)
    axes[2].set_ylabel("Residual RMS", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    summary = (
        f"CL: {trend_cl['direction']}, sign flips={trend_cl['sign_changes']}\n"
        f"CD: {trend_cd['direction']}, sign flips={trend_cd['sign_changes']}\n"
        f"Residual: {trend_res['direction']}, sign flips={trend_res['sign_changes']}\n"
        f"Reports: {steps.size}, last step={int(steps[-1])}"
    )
    axes[0].text(
        0.98, 0.03, summary,
        transform=axes[0].transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _surface_probe_series(states: list[dict], field_key: str, target_x: float, upper: bool) -> np.ndarray:
    """
    從 surface field 歷史抽取固定 x/c probe 時間序列
    """
    values = []
    for state in states:
        x = np.asarray(state["surface_x"], dtype=float)
        y = np.asarray(state["surface_y"], dtype=float)
        field = np.asarray(state[field_key], dtype=float)
        mask = y >= 0.0 if upper else y < 0.0
        xs = x[mask]
        fs = field[mask]
        order = np.argsort(xs)
        xs = xs[order]
        fs = fs[order]
        values.append(float(np.interp(target_x, xs, fs)))
    return np.asarray(values, dtype=float)


def plot_surface_history_diagnostics(state_files: list[str], field_key: str, save_path: str):
    """
    繪製 surface Cp / Cf 的時間歷程判讀圖

    What: 追蹤 surface field 對 final snapshot 的 RMS 差異，並輸出幾個固定 x/c probe
    Why:  完整 surface 曲線疊圖很難讀；RMS-to-final + probe history 更適合判斷
          單調趨穩、局部回擺或明顯震盪
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(state_files) < 2:
        return

    states = [np.load(path, allow_pickle=True).item() for path in state_files]
    times = np.array([float(state["time"]) for state in states], dtype=float)
    fields = [np.asarray(state[field_key], dtype=float) for state in states]
    final = fields[-1]
    rms_to_final = np.array(
        [float(np.sqrt(np.mean((field - final) ** 2))) for field in fields],
        dtype=float,
    )
    max_to_final = np.array(
        [float(np.max(np.abs(field - final))) for field in fields],
        dtype=float,
    )

    probes = [
        ("U x/c=0.2", True, 0.2),
        ("U x/c=0.5", True, 0.5),
        ("U x/c=0.8", True, 0.8),
        ("L x/c=0.2", False, 0.2),
        ("L x/c=0.5", False, 0.5),
        ("L x/c=0.8", False, 0.8),
    ]
    probe_series = []
    sign_change_counts = []
    for label, upper, x_target in probes:
        series = _surface_probe_series(states, field_key, x_target, upper)
        probe_series.append((label, series))
        sign_change_counts.append(describe_series_trend(series)["sign_changes"])

    field_label = field_key.upper()
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(times, rms_to_final, "k-o", ms=3, lw=1.4, label="RMS to final")
    axes[0].plot(times, max_to_final, "r-o", ms=3, lw=1.2, label="Max |Δ| to final")
    axes[0].set_ylabel(f"{field_label} mismatch", fontsize=12)
    axes[0].set_title(f"{field_label} Surface History Diagnostics", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    for label, series in probe_series[:3]:
        axes[1].plot(times, series, marker="o", ms=3, lw=1.3, label=label)
    axes[1].set_ylabel(f"Upper {field_label}", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, ncol=3)

    for label, series in probe_series[3:]:
        axes[2].plot(times, series, marker="o", ms=3, lw=1.3, label=label)
    axes[2].set_xlabel("Physical time", fontsize=12)
    axes[2].set_ylabel(f"Lower {field_label}", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, ncol=3)

    trend = describe_series_trend(rms_to_final)
    summary = (
        f"RMS-to-final: {trend['direction']}, sign flips={trend['sign_changes']}\n"
        f"Probe sign flips: max={max(sign_change_counts)}, mean={np.mean(sign_change_counts):.2f}\n"
        f"Snapshots: {len(states)}, final t={times[-1]:.4e}"
    )
    axes[0].text(
        0.98, 0.03, summary,
        transform=axes[0].transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def evaluate_plateau(history: list[dict], window: int, shock_tol: float, mach_tol: float) -> dict:
    """
    檢查 shock 位置與 Mach 峰值是否平台化
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


def _build_naca0012_ns_solver_controls(
        *,
        ref: dict,
        time_marching: str,
        pseudo_cfl_start: float | None,
        pseudo_ramp_steps: int,
        pseudo_precond_ref_mach: float,
        pseudo_precond_min_scale: float,
        residual_smoothing_eps: float,
        residual_smoothing_passes: int,
        adaptive_pseudo: bool,
        adaptive_interval: int,
        adaptive_cfl_growth: float,
        adaptive_cfl_shrink: float,
        adaptive_target_ratio: float,
        adaptive_fail_ratio: float,
        adaptive_smoothing_max_eps: float | None,
        cfl: float,
        turbulence_model: str,
        eddy_viscosity_ratio: float,
        turbulent_prandtl: float,
        sa_nu_tilde_inf_ratio: float) -> list[SolverControlDescriptor]:
    """
    建立 NACA0012 NS 的 solver control descriptors。

    What:
    - 將 far-field、time marching、pseudo-time 與 turbulence closure 提升為正式控制面

    Why:
    - 外流黏性翼型若仍在案例裡直接呼叫一串 solver setters，registry 只會變成薄殼
    """
    return [
        SolverControlDescriptor.time_marching(time_marching),
        SolverControlDescriptor.pseudo_time_controls(
            cfl_start=pseudo_cfl_start,
            ramp_steps=pseudo_ramp_steps,
            precond_ref_mach=pseudo_precond_ref_mach,
            precond_min_scale=pseudo_precond_min_scale,
        ),
        SolverControlDescriptor.residual_smoothing(
            epsilon=residual_smoothing_eps,
            passes=residual_smoothing_passes,
        ),
        SolverControlDescriptor.adaptive_pseudo_strategy(
            enabled=adaptive_pseudo,
            cfl_min=pseudo_cfl_start,
            cfl_max=cfl,
            growth=adaptive_cfl_growth,
            shrink=adaptive_cfl_shrink,
            target_ratio=adaptive_target_ratio,
            fail_ratio=adaptive_fail_ratio,
            interval=adaptive_interval,
            smoothing_max_eps=adaptive_smoothing_max_eps,
        ),
        SolverControlDescriptor.turbulence_model(
            model=turbulence_model,
            eddy_viscosity_ratio=eddy_viscosity_ratio,
            turbulent_prandtl=turbulent_prandtl,
            sa_nu_tilde_inf_ratio=sa_nu_tilde_inf_ratio,
        ),
        SolverControlDescriptor.far_field(
            location="j_max",
            rho=ref["rho_inf"],
            u=ref["u_inf"],
            v=ref["v_inf"],
            p=ref["p_inf"],
        ),
    ]


def _initialize_naca0012_ns_case(
        runner: CaseRunner,
        solver,
        _bc_handle,
        *,
        ref: dict,
        init_mode: str,
        init_bl_thickness: float,
        x_node: np.ndarray,
        y_node: np.ndarray):
    """
    NACA0012 NS 的正式初始化流程。

    What:
    - 建立 wall-aware primitive 初始場，並在需要時上傳 SA working variable

    Why:
    - 邊界與 transport closure 應先由 descriptor 套用，再由 initializer 處理物理初始狀態
    """
    w0_np, nu_tilde0_np = build_initial_primitive_state(
        solver,
        ref,
        init_mode=init_mode,
        init_bl_thickness=init_bl_thickness,
    )
    solver.init_from_primitive_numpy(w0_np)
    if nu_tilde0_np is not None:
        solver.set_sa_nu_tilde_from_numpy(nu_tilde0_np)
    solver.refresh_ghost_cells()

    geom = compute_geometry_monitor(solver)
    flow0 = compute_flow_diagnostics(
        solver,
        ref,
        dt=0.0,
        sim_time=0.0,
        initial_mass=0.0,
        geom=geom,
        x_node=x_node,
        y_node=y_node,
    )
    runner._naca0012_ns_geom = geom
    runner._naca0012_ns_initial_mass = float(flow0["mass"])


def _naca0012_ns_diagnostics_hook(
        runner: CaseRunner,
        solver,
        diagnostics: dict,
        *,
        alpha_rad: float,
        ref: dict,
        x_node: np.ndarray,
        y_node: np.ndarray) -> dict:
    """
    派生 NACA0012 NS 專屬 diagnostics。
    """
    geom = getattr(runner, "_naca0012_ns_geom", None)
    initial_mass = getattr(runner, "_naca0012_ns_initial_mass", None)
    if geom is None or initial_mass is None:
        return {}

    dt = float(solver.get_dt()) if runner.current_step > 0 else 0.0
    aero = compute_aero_forces(solver, alpha_rad, ref)
    flow = compute_flow_diagnostics(
        solver,
        ref,
        dt=dt,
        sim_time=runner.current_time,
        initial_mass=float(initial_mass),
        geom=geom,
        x_node=x_node,
        y_node=y_node,
    )
    transport = solver.get_transport_coefficients()
    return {
        "lift_coefficient": float(aero["CL"]),
        "drag_coefficient": float(aero["CD"]),
        "drag_coefficient_abs": float(abs(aero["CD"])),
        "pressure_drag_coefficient": float(aero["CD_p"]),
        "viscous_drag_coefficient": float(aero["CD_v"]),
        "y_plus_min": float(aero["y_plus_min"]),
        "y_plus_mean": float(aero["y_plus_mean"]),
        "y_plus_max": float(aero["y_plus_max"]),
        "mass_error": float(flow["mass_error"]),
        "cfl_est": float(flow["cfl_est"]),
        "supersonic_fraction": float(flow["supersonic_fraction"]),
        "entropy_rise_max": float(flow["entropy_rise_max"]),
        "shock_sensor_max": float(flow["shock_sensor_max"]),
        "shock_peak_x": float(flow["shock_peak_x"]),
        "shock_peak_y": float(flow["shock_peak_y"]),
        "time_marching_mode": flow["time_marching_mode"],
        "residual": float(solver.get_residual_norm()),
        "mu": float(transport["mu"]),
        "mu_t": float(transport["mu_t"]),
        "mu_eff": float(transport["mu_eff"]),
        "kappa_eff": float(transport["kappa_eff"]),
        "turbulence_model": transport["turbulence_model"],
    }


def build_naca0012_ns_runner(
        *,
        ni: int = 120,
        nj: int = 40,
        ma: float = 0.15,
        re: float = 500.0,
        aoa: float = 4.0,
        r_far: float = 12.0,
        cfl: float = 0.08,
        allow_transonic: bool = False,
        turbulence_model: str = "laminar",
        eddy_viscosity_ratio: float = 0.0,
        turbulent_prandtl: float = 0.9,
        sa_nu_tilde_inf_ratio: float = 3.0,
        wall_spacing_ratio: float = 1.0,
        time_marching: str = "global",
        pseudo_cfl_start: float | None = None,
        pseudo_ramp_steps: int = 0,
        pseudo_precond_ref_mach: float = 1.0,
        pseudo_precond_min_scale: float = 1.0,
        residual_smoothing_eps: float = 0.0,
        residual_smoothing_passes: int = 0,
        adaptive_pseudo: bool = False,
        adaptive_interval: int = 20,
        adaptive_cfl_growth: float = 1.03,
        adaptive_cfl_shrink: float = 0.7,
        adaptive_target_ratio: float = 0.995,
        adaptive_fail_ratio: float = 1.005,
        adaptive_smoothing_max_eps: float | None = None,
        initial_condition: str = "auto",
        init_bl_thickness: float = 0.02) -> tuple[CaseRunner, dict]:
    """
    建立 NACA0012 NS 的 CaseRunner。

    What:
    - 組裝 O-grid、airfoil no-slip wall、far-field characteristic BC 與黏性外流診斷

    Why:
    - external aero workflow 若只覆蓋 inviscid Euler，solver toolkit 仍缺有限 Re 主線
    """
    from fvm_taichi import NavierStokesSolver, generate_o_grid_naca0012

    if not (0.0 < ma < 1.0):
        raise ValueError(f"Ma must satisfy 0 < Ma < 1, got {ma}")
    transonic_mode = ma >= 0.7
    if transonic_mode and not allow_transonic:
        raise ValueError(
            f"Ma={ma:.3f} enters transonic regime. "
            "Use allow_transonic=True to build the experimental path."
        )
    if re <= 0.0:
        raise ValueError(f"Re must be positive, got {re}")
    if wall_spacing_ratio <= 0.0:
        raise ValueError(f"wall_spacing_ratio must be positive, got {wall_spacing_ratio}")

    validate_turbulence_inputs(
        turbulence_model,
        eddy_viscosity_ratio,
        turbulent_prandtl,
        sa_nu_tilde_inf_ratio,
    )
    validate_time_marching_inputs(time_marching, None)
    validate_pseudo_time_controls(
        time_marching=time_marching,
        cfl=cfl,
        pseudo_cfl_start=pseudo_cfl_start,
        pseudo_ramp_steps=pseudo_ramp_steps,
        pseudo_precond_ref_mach=pseudo_precond_ref_mach,
        pseudo_precond_min_scale=pseudo_precond_min_scale,
        residual_smoothing_eps=residual_smoothing_eps,
        residual_smoothing_passes=residual_smoothing_passes,
        adaptive_pseudo=adaptive_pseudo,
        adaptive_interval=adaptive_interval,
        adaptive_cfl_growth=adaptive_cfl_growth,
        adaptive_cfl_shrink=adaptive_cfl_shrink,
        adaptive_target_ratio=adaptive_target_ratio,
        adaptive_fail_ratio=adaptive_fail_ratio,
        adaptive_smoothing_max_eps=adaptive_smoothing_max_eps,
    )

    gamma = 1.4
    alpha_rad = float(np.deg2rad(aoa))
    ref = compute_reference_scales(ma, alpha_rad, gamma=gamma, re=re, chord=1.0)
    init_mode = resolve_initial_condition_mode(initial_condition, turbulence_model, re)
    x_node, y_node = generate_o_grid_naca0012(
        ni=ni,
        nj=nj,
        R_far=r_far,
        NG=NavierStokesSolver.NG,
        wall_spacing_ratio=wall_spacing_ratio,
    )

    runner = CaseRunner(
        name="naca0012_ns",
        method="fvm",
        equation="navier_stokes",
        regime="compressible",
        grid=CurvilinearGrid2D(x_node=x_node, y_node=y_node, ng=NavierStokesSolver.NG),
        solver_kwargs={
            "gamma": gamma,
            "cfl": cfl,
            "re": re,
            "pr": 0.72,
            "u_ref": max(ref["v_mag"], 1e-6),
            "length_scale": 1.0,
            "rho_ref": ref["rho_inf"],
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("airfoil", u_wall=0.0, v_wall=0.0),
        ],
        solver_controls=_build_naca0012_ns_solver_controls(
            ref=ref,
            time_marching=time_marching,
            pseudo_cfl_start=pseudo_cfl_start,
            pseudo_ramp_steps=pseudo_ramp_steps,
            pseudo_precond_ref_mach=pseudo_precond_ref_mach,
            pseudo_precond_min_scale=pseudo_precond_min_scale,
            residual_smoothing_eps=residual_smoothing_eps,
            residual_smoothing_passes=residual_smoothing_passes,
            adaptive_pseudo=adaptive_pseudo,
            adaptive_interval=adaptive_interval,
            adaptive_cfl_growth=adaptive_cfl_growth,
            adaptive_cfl_shrink=adaptive_cfl_shrink,
            adaptive_target_ratio=adaptive_target_ratio,
            adaptive_fail_ratio=adaptive_fail_ratio,
            adaptive_smoothing_max_eps=adaptive_smoothing_max_eps,
            cfl=cfl,
            turbulence_model=turbulence_model,
            eddy_viscosity_ratio=eddy_viscosity_ratio,
            turbulent_prandtl=turbulent_prandtl,
            sa_nu_tilde_inf_ratio=sa_nu_tilde_inf_ratio,
        ),
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_naca0012_ns_case(
            runner_obj,
            solver_obj,
            bc_handle,
            ref=ref,
            init_mode=init_mode,
            init_bl_thickness=init_bl_thickness,
            x_node=x_node,
            y_node=y_node,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _naca0012_ns_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            alpha_rad=alpha_rad,
            ref=ref,
            x_node=x_node,
            y_node=y_node,
        ),
    )
    return runner, {
        "alpha_rad": alpha_rad,
        "ref": ref,
        "init_mode": init_mode,
        "transonic_mode": transonic_mode,
    }


def run_naca0012_ns(
        ni: int = 180,
        nj: int = 60,
        ma: float = 0.15,
        re: float = 500.0,
        aoa: float = 4.0,
        steps: int = 12000,
        r_far: float = 15.0,
        cfl: float = 0.12,
        report: int = 500,
        tol: float = 5e-5,
        save_interval: int = 1000,
        output_dir: str | None = None,
        allow_transonic: bool = False,
        turbulence_model: str = "laminar",
        eddy_viscosity_ratio: float = 0.0,
        turbulent_prandtl: float = 0.9,
        sa_nu_tilde_inf_ratio: float = 3.0,
        wall_spacing_ratio: float = 1.0,
        time_marching: str = "global",
        pseudo_cfl_start: float | None = None,
        pseudo_ramp_steps: int = 0,
        pseudo_precond_ref_mach: float = 1.0,
        pseudo_precond_min_scale: float = 1.0,
        residual_smoothing_eps: float = 0.0,
        residual_smoothing_passes: int = 0,
        adaptive_pseudo: bool = False,
        adaptive_interval: int = 20,
        adaptive_cfl_growth: float = 1.03,
        adaptive_cfl_shrink: float = 0.7,
        adaptive_target_ratio: float = 0.995,
        adaptive_fail_ratio: float = 1.005,
        adaptive_smoothing_max_eps: float | None = None,
        initial_condition: str = "auto",
        init_bl_thickness: float = 0.02,
        target_sim_time: float | None = None,
        plateau_window: int = 5,
        shock_tol: float = 0.03,
        mach_tol: float = 0.05,
):
    """
    執行 NACA0012 finite-Re laminar NS 模擬
    """
    from cfd_taichi.configuration import apply_solver_control_descriptors
    from fvm_taichi import (
        NavierStokesSolver,
        generate_o_grid_naca0012,
        plot_cp,
        plot_surface_cp_cf,
        plot_wall_shear_distribution,
        plot_wall_unit_distribution,
        plot_flow_field,
        plot_cell_scalar_field,
    )

    if not (0.0 < ma < 1.0):
        raise ValueError(f"Ma must satisfy 0 < Ma < 1, got {ma}")
    transonic_mode = ma >= 0.7
    if transonic_mode and not allow_transonic:
        raise ValueError(
            f"Ma={ma:.3f} enters transonic regime. "
            "Use allow_transonic=True (CLI: --transonic) to run the experimental path."
        )
    if re <= 0.0:
        raise ValueError(f"Re must be positive, got {re}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if save_interval <= 0:
        raise ValueError(f"save_interval must be positive, got {save_interval}")
    if wall_spacing_ratio <= 0.0:
        raise ValueError(f"wall_spacing_ratio must be positive, got {wall_spacing_ratio}")
    validate_turbulence_inputs(
        turbulence_model,
        eddy_viscosity_ratio,
        turbulent_prandtl,
        sa_nu_tilde_inf_ratio,
    )
    validate_time_marching_inputs(time_marching, target_sim_time)
    validate_pseudo_time_controls(
        time_marching=time_marching,
        cfl=cfl,
        pseudo_cfl_start=pseudo_cfl_start,
        pseudo_ramp_steps=pseudo_ramp_steps,
        pseudo_precond_ref_mach=pseudo_precond_ref_mach,
        pseudo_precond_min_scale=pseudo_precond_min_scale,
        residual_smoothing_eps=residual_smoothing_eps,
        residual_smoothing_passes=residual_smoothing_passes,
        adaptive_pseudo=adaptive_pseudo,
        adaptive_interval=adaptive_interval,
        adaptive_cfl_growth=adaptive_cfl_growth,
        adaptive_cfl_shrink=adaptive_cfl_shrink,
        adaptive_target_ratio=adaptive_target_ratio,
        adaptive_fail_ratio=adaptive_fail_ratio,
        adaptive_smoothing_max_eps=adaptive_smoothing_max_eps,
    )
    if plateau_window <= 1:
        raise ValueError(f"plateau_window must be > 1, got {plateau_window}")
    if shock_tol <= 0.0 or mach_tol <= 0.0:
        raise ValueError(f"shock_tol and mach_tol must be positive, got {shock_tol}, {mach_tol}")

    gamma = 1.4
    alpha_rad = np.deg2rad(aoa)
    ref = compute_reference_scales(ma, alpha_rad, gamma=gamma, re=re, chord=1.0)
    init_mode = resolve_initial_condition_mode(initial_condition, turbulence_model, re)

    print("=" * 76)
    print("  NACA0012 Navier-Stokes (O-grid + Curved No-Slip Wall)")
    print("=" * 76)
    print(f"  Grid:      ni={ni} x nj={nj},  R_far={r_far}")
    print(f"  Radial:    wall_spacing_ratio={wall_spacing_ratio:.4f}")
    print(f"  Marching:  mode={time_marching}")
    if time_marching == "local_pseudo":
        pseudo_cfl_start_eff = cfl if pseudo_cfl_start is None else pseudo_cfl_start
        print(
            f"  Pseudo:    cfl_start={pseudo_cfl_start_eff:.4f}, "
            f"ramp_steps={pseudo_ramp_steps}, "
            f"precond_ref_mach={pseudo_precond_ref_mach:.3f}, "
            f"precond_min_scale={pseudo_precond_min_scale:.3f}"
        )
        print(
            f"  Smoothing: eps={residual_smoothing_eps:.3f}, "
            f"passes={residual_smoothing_passes}"
        )
        if adaptive_pseudo:
            adaptive_smoothing_max_eff = (
                residual_smoothing_eps if adaptive_smoothing_max_eps is None
                else adaptive_smoothing_max_eps
            )
            print(
                f"  Adaptive:  interval={adaptive_interval}, "
                f"growth={adaptive_cfl_growth:.3f}, shrink={adaptive_cfl_shrink:.3f}, "
                f"ratio=[{adaptive_target_ratio:.3f}, {adaptive_fail_ratio:.3f}], "
                f"smooth_max={adaptive_smoothing_max_eff:.3f}"
            )
    print(f"  Init:      mode={init_mode},  bl_thickness={init_bl_thickness:.4f}")
    print(f"  Ma / Re:   {ma:.3f} / {re:.1f},  AOA={aoa:.2f} deg")
    print(f"  V_inf:     ({ref['u_inf']:.5f}, {ref['v_inf']:.5f}), |V|={ref['v_mag']:.5f}")
    print(f"  p_inf:     {ref['p_inf']:.5f},  q_inf={ref['q_inf']:.5f},  mu_inf={ref['mu_inf']:.5e}")
    print(f"  Steps:     {steps},  CFL={cfl},  Residual tol={tol:.1e},  Save interval={save_interval}")
    if target_sim_time is not None:
        print(f"  Stop:      target_sim_time={target_sim_time:.6e}")
    if transonic_mode:
        print("  Mode:      experimental transonic")
        print("  Warning:   far-field BC remains subsonic characteristic; shock results are observational, not validated")

    t0 = time.time()
    print("\n[1] Generating O-grid...", end=" ", flush=True)
    x_node, y_node = generate_o_grid_naca0012(
        ni=ni, nj=nj, R_far=r_far, NG=NavierStokesSolver.NG,
        wall_spacing_ratio=wall_spacing_ratio)
    print(f"done  ({time.time() - t0:.2f}s)")

    print("[2] Initializing solver...", end=" ", flush=True)
    solver = NavierStokesSolver(
        ni=ni,
        nj=nj,
        gamma=gamma,
        cfl=cfl,
        re=re,
        pr=0.72,
        u_ref=max(ref["v_mag"], 1e-6),
        length_scale=1.0,
        rho_ref=ref["rho_inf"],
    )
    solver.set_curvilinear_grid(x_node, y_node)
    solver.set_periodic_bc(i_dir=True, j_dir=False)
    solver.set_no_slip_wall("airfoil", u_wall=0.0, v_wall=0.0, temperature=None)
    apply_solver_control_descriptors(
        solver,
        _build_naca0012_ns_solver_controls(
            ref=ref,
            time_marching=time_marching,
            pseudo_cfl_start=pseudo_cfl_start,
            pseudo_ramp_steps=pseudo_ramp_steps,
            pseudo_precond_ref_mach=pseudo_precond_ref_mach,
            pseudo_precond_min_scale=pseudo_precond_min_scale,
            residual_smoothing_eps=residual_smoothing_eps,
            residual_smoothing_passes=residual_smoothing_passes,
            adaptive_pseudo=adaptive_pseudo,
            adaptive_interval=adaptive_interval,
            adaptive_cfl_growth=adaptive_cfl_growth,
            adaptive_cfl_shrink=adaptive_cfl_shrink,
            adaptive_target_ratio=adaptive_target_ratio,
            adaptive_fail_ratio=adaptive_fail_ratio,
            adaptive_smoothing_max_eps=adaptive_smoothing_max_eps,
            cfl=cfl,
            turbulence_model=turbulence_model,
            eddy_viscosity_ratio=eddy_viscosity_ratio,
            turbulent_prandtl=turbulent_prandtl,
            sa_nu_tilde_inf_ratio=sa_nu_tilde_inf_ratio,
        ),
    )
    w0_np, nu_tilde0_np = build_initial_primitive_state(
        solver,
        ref,
        init_mode=init_mode,
        init_bl_thickness=init_bl_thickness,
    )
    solver.init_from_primitive_numpy(w0_np)
    if nu_tilde0_np is not None:
        solver.set_sa_nu_tilde_from_numpy(nu_tilde0_np)
    solver.refresh_ghost_cells()
    transport = solver.get_transport_coefficients()
    print(f"done  ({time.time() - t0:.2f}s)")
    print(
        f"  Transport: mu={transport['mu']:.5e}, mu_t={transport['mu_t']:.5e}, "
        f"mu_eff={transport['mu_eff']:.5e}, model={transport['turbulence_model']}"
    )
    if turbulence_model == "constant_eddy_viscosity":
        print(
            f"  Closure:   eddy_ratio={transport['eddy_viscosity_ratio']:.2f}, "
            f"Pr_t={transport['pr_t']:.2f}"
        )
    elif turbulence_model == "spalart_allmaras":
        print(
            f"  Closure:   SA, nu_tilde_inf/nu={transport['sa_nu_tilde_inf_ratio']:.2f}, "
            f"Pr_t={transport['pr_t']:.2f}"
        )

    x_cc, y_cc = get_airfoil_cc(x_node, y_node, ni)
    geom = compute_geometry_monitor(solver)

    print(f"\n[3] Running {steps} steps...")
    if transonic_mode:
        print(f"  {'step':>6}  {'dt':>10}  {'res':>10}  {'CL':>8}  {'CD':>8}  "
              f"{'M_max':>8}  {'y+_max':>8}  {'M>1%':>8}  {'shock':>9}  {'dS_max':>9}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
    else:
        print(f"  {'step':>6}  {'dt':>10}  {'res':>10}  {'CL':>8}  {'CD':>8}  "
              f"{'p_min':>9}  {'M_max':>8}  {'y+_max':>8}  {'mass_err':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  "
              f"{'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}")

    sim_start = time.time()
    converged = False
    plateau = False
    final_step = 0
    final_residual = np.nan
    stop_reason = "max_steps"
    aero = None
    flow = None
    history = []
    sim_time = 0.0
    time_label = "Physical time" if time_marching == "global" else "Pseudo time"
    plateau_status = {
        "window": int(plateau_window),
        "ready": False,
        "plateau": False,
        "shock_span": np.nan,
        "mach_span": np.nan,
    }

    aero0 = compute_aero_forces(solver, alpha_rad, ref)
    flow0 = compute_flow_diagnostics(
        solver, ref, dt=0.0, sim_time=0.0, initial_mass=0.0, geom=geom,
        x_node=x_node, y_node=y_node)
    initial_mass = flow0["mass"]
    flow0["mass_error"] = 0.0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_state_file(output_dir, 0, 0.0, solver, flow0, aero0, x_node, y_node, x_cc, y_cc)

    for step in range(1, steps + 1):
        try:
            dt = solver.step()
        except RuntimeError as exc:
            print(f"\n  ❌ Solver fail-fast at step {step}: {exc}")
            return None
        if time_marching == "global":
            sim_time += dt
        else:
            sim_time += solver.get_dt_diagnostics()["dt_mean"]
        reached_target_time = target_sim_time is not None and sim_time >= target_sim_time

        if step % report == 0 or step == 1 or reached_target_time:
            ti.sync()
            residual = solver.get_residual_norm()
            aero = compute_aero_forces(solver, alpha_rad, ref)
            flow = compute_flow_diagnostics(
                solver, ref, dt=dt, sim_time=sim_time, initial_mass=initial_mass, geom=geom,
                x_node=x_node, y_node=y_node)
            elapsed = time.time() - sim_start

            if transonic_mode:
                print(f"  {step:>6}  {dt:>10.4e}  {residual:>10.3e}  "
                      f"{aero['CL']:>8.4f}  {aero['CD']:>8.5f}  "
                      f"{flow['mach_max']:>8.4f}  {aero['y_plus_max']:>8.3f}  "
                      f"{100.0 * flow['supersonic_fraction']:>7.3f}%  "
                      f"{flow['shock_sensor_max']:>9.3e}  {flow['entropy_rise_max']:>9.3e}")
            else:
                print(f"  {step:>6}  {dt:>10.4e}  {residual:>10.3e}  "
                      f"{aero['CL']:>8.4f}  {aero['CD']:>8.5f}  "
                      f"{flow['p_min']:>9.5f}  {flow['mach_max']:>8.4f}  "
                      f"{aero['y_plus_max']:>8.3f}  {flow['mass_error']:>10.3e}")

            final_step = step
            final_residual = residual
            history.append({
                "step": step,
                "time": flow["time"],
                "dt": flow["dt"],
                "dt_min": flow["dt_min"],
                "dt_mean": flow["dt_mean"],
                "dt_max": flow["dt_max"],
                "pseudo_cfl": flow["pseudo_cfl"],
                "pseudo_cfl_target": flow["pseudo_cfl_target"],
                "pseudo_precond_scale_min": flow["pseudo_precond_scale_min"],
                "pseudo_precond_scale_mean": flow["pseudo_precond_scale_mean"],
                "pseudo_precond_scale_max": flow["pseudo_precond_scale_max"],
                "residual_smoothing_eps": flow["residual_smoothing_eps"],
                "residual_smoothing_base_eps": flow["residual_smoothing_base_eps"],
                "residual_smoothing_max_eps": flow["residual_smoothing_max_eps"],
                "residual_smoothing_passes": flow["residual_smoothing_passes"],
                "pseudo_adaptive_enabled": flow["pseudo_adaptive_enabled"],
                "pseudo_residual_ratio": flow["pseudo_residual_ratio"],
                "residual": residual,
                "CL": aero["CL"],
                "CD": aero["CD"],
                "CL_p": aero["CL_p"],
                "CD_p": aero["CD_p"],
                "CL_v": aero["CL_v"],
                "CD_v": aero["CD_v"],
                "rho_min": flow["rho_min"],
                "rho_max": flow["rho_max"],
                "p_min": flow["p_min"],
                "p_max": flow["p_max"],
                "u_max": flow["u_max"],
                "mach_max": flow["mach_max"],
                "supersonic_fraction": flow["supersonic_fraction"],
                "entropy_rise_max": flow["entropy_rise_max"],
                "shock_sensor_max": flow["shock_sensor_max"],
                "shock_peak_x": flow["shock_peak_x"],
                "shock_peak_y": flow["shock_peak_y"],
                "y_plus_mean": aero["y_plus_mean"],
                "y_plus_max": aero["y_plus_max"],
                "mass_error": flow["mass_error"],
                "cfl_est": flow["cfl_est"],
            })

            if transonic_mode:
                plateau_status = evaluate_plateau(
                    history, window=plateau_window, shock_tol=shock_tol, mach_tol=mach_tol)
                if plateau_status["ready"] and plateau_status["plateau"] and not plateau:
                    plateau = True
                    print(
                        f"  -> Shock plateau detected: Δx_shock={plateau_status['shock_span']:.4e}, "
                        f"ΔM_max={plateau_status['mach_span']:.4e} over last {plateau_window} reports"
                    )

            if output_dir and (step == 1 or step % save_interval == 0):
                save_state_file(
                    output_dir, step, sim_time, solver, flow, aero, x_node, y_node, x_cc, y_cc)

            if residual < tol:
                converged = True
                stop_reason = "residual_tol"
                print(f"\n  ✅ Residual converged at step {step} (RMS={residual:.3e})")
                break
            if reached_target_time:
                stop_reason = "target_sim_time"
                print(
                    f"\n  ✅ Target physical time reached at step {step} "
                    f"(t={sim_time:.6e})"
                )
                break
        elif output_dir and (step % save_interval == 0 or reached_target_time):
            ti.sync()
            aero = compute_aero_forces(solver, alpha_rad, ref)
            flow = compute_flow_diagnostics(
                solver, ref, dt=dt, sim_time=sim_time, initial_mass=initial_mass, geom=geom,
                x_node=x_node, y_node=y_node)
            save_state_file(
                output_dir, step, sim_time, solver, flow, aero, x_node, y_node, x_cc, y_cc)
            if reached_target_time:
                final_step = step
                final_residual = solver.get_residual_norm()
                stop_reason = "target_sim_time"
                print(
                    f"\n  ✅ Target physical time reached at step {step} "
                    f"(t={sim_time:.6e})"
                )
                break

    ti.sync()
    if aero is None:
        aero = compute_aero_forces(solver, alpha_rad, ref)
    if flow is None:
        flow = compute_flow_diagnostics(
            solver, ref, dt=dt if steps > 0 else 0.0, sim_time=sim_time,
            initial_mass=initial_mass, geom=geom, x_node=x_node, y_node=y_node)

    total_time = time.time() - t0
    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")
    print(f"\n{'='*76}")
    print(f"  Last reported step: {final_step}")
    print(f"  Stop reason:        {stop_reason}")
    print(f"  {time_label}:      {sim_time:.6e}")
    print(f"  Local dt range:     [{flow['dt_min']:.4e}, {flow['dt_max']:.4e}], mean={flow['dt_mean']:.4e}")
    if time_marching == "local_pseudo":
        print(
            f"  Pseudo controls:   CFL_eff={flow['pseudo_cfl']:.4f}, "
            f"target={flow['pseudo_cfl_target']:.4f}, "
            f"acoustic_scale=[{flow['pseudo_precond_scale_min']:.3f}, "
            f"{flow['pseudo_precond_scale_max']:.3f}], "
            f"mean={flow['pseudo_precond_scale_mean']:.3f}"
        )
        print(
            f"  Residual smooth:   eps={flow['residual_smoothing_eps']:.3f}, "
            f"base={flow['residual_smoothing_base_eps']:.3f}, "
            f"max={flow['residual_smoothing_max_eps']:.3f}, "
            f"passes={flow['residual_smoothing_passes']}"
        )
        if flow["pseudo_adaptive_enabled"]:
            print(f"  Adaptive ratio:    residual_ratio={flow['pseudo_residual_ratio']:.4f}")
    print(f"  Residual RMS:       {final_residual:.3e}")
    print(f"  Mass error:         {flow['mass_error']:.3e}")
    print(f"  Density range:      [{flow['rho_min']:.5f}, {flow['rho_max']:.5f}]")
    print(f"  Pressure range:     [{flow['p_min']:.5f}, {flow['p_max']:.5f}]")
    print(f"  Max |u| / Mach:     {flow['u_max']:.5f} / {flow['mach_max']:.5f}")
    print(f"  Wall units:         y+=[{aero['y_plus_min']:.3f}, {aero['y_plus_max']:.3f}], mean={aero['y_plus_mean']:.3f}")
    if transonic_mode:
        print(f"  Supersonic area:    {100.0 * flow['supersonic_fraction']:.3f}% of cells")
        print(f"  Shock sensor max:   {flow['shock_sensor_max']:.3e}  (p99={flow['shock_sensor_p99']:.3e})")
        print(f"  Entropy rise max:   {flow['entropy_rise_max']:.3e}")
        print(f"  Shock peak estimate:x={flow['shock_peak_x']:.4f}, y={flow['shock_peak_y']:.4f}")
    print(f"  CFL estimate:       {flow['cfl_est']:.5f}")
    if converged:
        print(f"  Converged forces:   CL={aero['CL']:.4f}, CD={aero['CD']:.5f}")
    else:
        print("  ⚠️  Not converged; forces below are last-step estimates")
        print(f"  Last-step forces:   CL={aero['CL']:.4f}, CD={aero['CD']:.5f}")
    print(f"  Pressure drag:      CD_p={aero['CD_p']:.5f}")
    print(f"  Viscous drag:       CD_v={aero['CD_v']:.5f}")
    print(f"  Lift components:    CL_p={aero['CL_p']:.5f}, CL_v={aero['CL_v']:.5f}")
    if transonic_mode:
        if plateau_status["ready"]:
            print(f"  Shock plateau:      {plateau_status['plateau']}")
            print(f"  Plateau spans:      Δx_shock={plateau_status['shock_span']:.4e}, ΔM_max={plateau_status['mach_span']:.4e}")
        else:
            print(f"  Shock plateau:      not enough history (need {plateau_window} reports)")
    print("=" * 76)

    if abs(aoa) < 0.1:
        upper = y_cc >= 0.0
        lower = y_cc < 0.0
        if upper.sum() > 1 and lower.sum() > 1:
            x_common = np.linspace(0.05, 0.95, 40)
            cp_u = np.interp(x_common, x_cc[upper][::-1], aero["Cp"][upper][::-1])
            cp_l = np.interp(x_common, x_cc[lower], aero["Cp"][lower])
            asymm = float(np.mean(np.abs(cp_u - cp_l)))
            print(f"  Cp symmetry error:  {asymm:.4e}")

    if output_dir:
        tag = f"ma{int(ma*100):03d}_re{int(re):05d}_aoa{int(round(aoa)):03d}"
        print(f"\n[4] Writing figures to {output_dir}/")
        plot_cp(
            solver, alpha_rad, ref["rho_inf"], ref["v_mag"], x_node, y_node,
            save_path=os.path.join(output_dir, f"cp_{tag}.png"))
        plot_surface_cp_cf(
            x_cc, y_cc, aero["Cp"], aero["Cf"],
            aoa_deg=aoa, re=re, ma=ma,
            save_path=os.path.join(output_dir, f"surface_cp_cf_{tag}.png"))
        plot_wall_shear_distribution(
            x_cc, y_cc, aero["tau_t"],
            aoa_deg=aoa, re=re, ma=ma,
            save_path=os.path.join(output_dir, f"wall_shear_{tag}.png"))
        plot_wall_unit_distribution(
            x_cc, y_cc, aero["y_plus"],
            aoa_deg=aoa, re=re, ma=ma,
            save_path=os.path.join(output_dir, f"wall_yplus_{tag}.png"))
        for field in ("velocity", "pressure", "density", "mach"):
            plot_flow_field(
                solver, x_node, y_node, field=field,
                save_path=os.path.join(output_dir, f"{field}_{tag}.png"))
        if transonic_mode:
            plot_cell_scalar_field(
                x_node,
                y_node,
                flow["shock_sensor"],
                field_label="Shock sensor",
                title=f"Shock Sensor — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                cmap="magma",
                save_path=os.path.join(output_dir, f"shock_sensor_{tag}.png"),
            )
            plot_cell_scalar_field(
                x_node,
                y_node,
                flow["entropy_rise"],
                field_label="Entropy rise",
                title=f"Entropy Rise — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                cmap="plasma",
                save_path=os.path.join(output_dir, f"entropy_rise_{tag}.png"),
            )
            history_steps = np.array([row["step"] for row in history], dtype=np.int32)
            if history_steps.size > 0:
                plot_history_series(
                    history_steps,
                    np.array([row["residual"] for row in history], dtype=np.float64),
                    ylabel="Residual RMS",
                    title=f"Residual History — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                    save_path=os.path.join(output_dir, f"residual_history_{tag}.png"),
                    logy=True,
                )
                plot_history_series(
                    history_steps,
                    np.array([row["shock_peak_x"] for row in history], dtype=np.float64),
                    ylabel="Shock Peak x",
                    title=f"Shock Position History — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                    save_path=os.path.join(output_dir, f"shock_peak_history_{tag}.png"),
                )
                plot_history_series(
                    history_steps,
                    np.array([row["mach_max"] for row in history], dtype=np.float64),
                    ylabel="Max Mach",
                    title=f"Max Mach History — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                    save_path=os.path.join(output_dir, f"mach_max_history_{tag}.png"),
                )
                plot_history_series(
                    history_steps,
                    np.array([row["y_plus_max"] for row in history], dtype=np.float64),
                    ylabel="y+ max",
                    title=f"y+ History — NACA0012, Ma={ma:.2f}, Re={re:.0f}, AOA={aoa:.1f} deg",
                    save_path=os.path.join(output_dir, f"yplus_history_{tag}.png"),
                )

        save_state_file(
            output_dir, final_step if final_step > 0 else steps, sim_time,
            solver, flow, aero, x_node, y_node, x_cc, y_cc)
        save_history_file(output_dir, history, {
            "ni": ni,
            "nj": nj,
            "ma": ma,
            "re": re,
            "aoa": aoa,
            "steps": steps,
            "r_far": r_far,
            "cfl": cfl,
            "tol": tol,
            "save_interval": save_interval,
            "wall_spacing_ratio": wall_spacing_ratio,
            "time_marching": time_marching,
            "pseudo_cfl_start": pseudo_cfl_start,
            "pseudo_ramp_steps": pseudo_ramp_steps,
            "pseudo_precond_ref_mach": pseudo_precond_ref_mach,
            "pseudo_precond_min_scale": pseudo_precond_min_scale,
            "residual_smoothing_eps": residual_smoothing_eps,
            "residual_smoothing_passes": residual_smoothing_passes,
            "adaptive_pseudo": adaptive_pseudo,
            "adaptive_interval": adaptive_interval,
            "adaptive_cfl_growth": adaptive_cfl_growth,
            "adaptive_cfl_shrink": adaptive_cfl_shrink,
            "adaptive_target_ratio": adaptive_target_ratio,
            "adaptive_fail_ratio": adaptive_fail_ratio,
            "adaptive_smoothing_max_eps": adaptive_smoothing_max_eps,
            "initial_condition": init_mode,
            "init_bl_thickness": init_bl_thickness,
            "target_sim_time": target_sim_time,
            "turbulence_model": turbulence_model,
            "eddy_viscosity_ratio": eddy_viscosity_ratio,
            "turbulent_prandtl": turbulent_prandtl,
            "sa_nu_tilde_inf_ratio": sa_nu_tilde_inf_ratio,
        })
        save_summary_file(output_dir, {
            "converged": converged,
            "stop_reason": stop_reason,
            "final_step": final_step,
            "sim_time": sim_time,
            "runtime_seconds": total_time,
            "residual": final_residual,
            "mass_error": flow["mass_error"],
            "rho_min": flow["rho_min"],
            "rho_max": flow["rho_max"],
            "p_min": flow["p_min"],
            "p_max": flow["p_max"],
            "u_max": flow["u_max"],
            "mach_max": flow["mach_max"],
            "supersonic_fraction": flow["supersonic_fraction"],
            "entropy_rise_max": flow["entropy_rise_max"],
            "shock_sensor_max": flow["shock_sensor_max"],
            "shock_peak_x": flow["shock_peak_x"],
            "shock_peak_y": flow["shock_peak_y"],
            "shock_plateau": plateau_status["plateau"],
            "shock_plateau_ready": plateau_status["ready"],
            "shock_peak_span": plateau_status["shock_span"],
            "mach_peak_span": plateau_status["mach_span"],
            "plateau_window": plateau_window,
            "shock_tol": shock_tol,
            "mach_tol": mach_tol,
            "cfl_est": flow["cfl_est"],
            "CL": aero["CL"],
            "CD": aero["CD"],
            "CL_p": aero["CL_p"],
            "CD_p": aero["CD_p"],
            "CL_v": aero["CL_v"],
            "CD_v": aero["CD_v"],
            "transonic_mode": transonic_mode,
            "wall_spacing_ratio": wall_spacing_ratio,
            "time_marching": time_marching,
            "pseudo_cfl_start": pseudo_cfl_start,
            "pseudo_ramp_steps": pseudo_ramp_steps,
            "pseudo_precond_ref_mach": pseudo_precond_ref_mach,
            "pseudo_precond_min_scale": pseudo_precond_min_scale,
            "residual_smoothing_eps": residual_smoothing_eps,
            "residual_smoothing_passes": residual_smoothing_passes,
            "adaptive_pseudo": adaptive_pseudo,
            "adaptive_interval": adaptive_interval,
            "adaptive_cfl_growth": adaptive_cfl_growth,
            "adaptive_cfl_shrink": adaptive_cfl_shrink,
            "adaptive_target_ratio": adaptive_target_ratio,
            "adaptive_fail_ratio": adaptive_fail_ratio,
            "adaptive_smoothing_max_eps": adaptive_smoothing_max_eps,
            "initial_condition": init_mode,
            "init_bl_thickness": init_bl_thickness,
            "target_sim_time": target_sim_time,
            "turbulence_model": turbulence_model,
            "eddy_viscosity_ratio": eddy_viscosity_ratio,
            "turbulent_prandtl": turbulent_prandtl,
            "sa_nu_tilde_inf_ratio": sa_nu_tilde_inf_ratio,
            "mu": transport["mu"],
            "mu_t": transport["mu_t"],
            "mu_eff": transport["mu_eff"],
            "kappa_eff": transport["kappa_eff"],
            "y_plus_min": aero["y_plus_min"],
            "y_plus_mean": aero["y_plus_mean"],
            "y_plus_max": aero["y_plus_max"],
            "pseudo_cfl": flow["pseudo_cfl"],
            "pseudo_cfl_target": flow["pseudo_cfl_target"],
            "pseudo_precond_scale_min": flow["pseudo_precond_scale_min"],
            "pseudo_precond_scale_mean": flow["pseudo_precond_scale_mean"],
            "pseudo_precond_scale_max": flow["pseudo_precond_scale_max"],
            "residual_smoothing_eps": flow["residual_smoothing_eps"],
            "residual_smoothing_base_eps": flow["residual_smoothing_base_eps"],
            "residual_smoothing_max_eps": flow["residual_smoothing_max_eps"],
            "residual_smoothing_passes": flow["residual_smoothing_passes"],
            "pseudo_adaptive_enabled": flow["pseudo_adaptive_enabled"],
            "pseudo_residual_ratio": flow["pseudo_residual_ratio"],
        })
        save_surface_file(output_dir, x_cc, y_cc, aero)

        if history:
            history_np = np.load(os.path.join(output_dir, "history.npy"), allow_pickle=True).item()
            plot_force_history_diagnostics(
                history_np,
                save_path=os.path.join(output_dir, f"force_history_analysis_{tag}.png"),
            )
        state_dir = os.path.join(output_dir, "states")
        if os.path.isdir(state_dir):
            state_files = sorted(
                os.path.join(state_dir, name)
                for name in os.listdir(state_dir)
                if name.startswith("state_") and name.endswith(".npy")
            )
            if len(state_files) >= 2:
                plot_surface_history_diagnostics(
                    state_files,
                    field_key="cp",
                    save_path=os.path.join(output_dir, f"cp_history_analysis_{tag}.png"),
                )
                plot_surface_history_diagnostics(
                    state_files,
                    field_key="cf",
                    save_path=os.path.join(output_dir, f"cf_history_analysis_{tag}.png"),
                )

    result = dict(aero)
    result.update({
        "ma": float(ma),
        "re": float(re),
        "aoa": float(aoa),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "final_step": int(final_step if final_step > 0 else steps),
        "sim_time": float(sim_time),
        "runtime_seconds": float(total_time),
        "residual": float(final_residual),
        "mass_error": float(flow["mass_error"]),
        "rho_min": float(flow["rho_min"]),
        "rho_max": float(flow["rho_max"]),
        "p_min": float(flow["p_min"]),
        "p_max": float(flow["p_max"]),
        "u_max": float(flow["u_max"]),
        "mach_max": float(flow["mach_max"]),
        "supersonic_fraction": float(flow["supersonic_fraction"]),
        "entropy_rise_max": float(flow["entropy_rise_max"]),
        "shock_sensor_max": float(flow["shock_sensor_max"]),
        "shock_peak_x": float(flow["shock_peak_x"]),
        "shock_peak_y": float(flow["shock_peak_y"]),
        "shock_plateau": bool(plateau_status["plateau"]),
        "shock_peak_span": float(plateau_status["shock_span"]),
        "mach_peak_span": float(plateau_status["mach_span"]),
        "cfl_est": float(flow["cfl_est"]),
        "output_dir": output_dir,
        "transonic_mode": bool(transonic_mode),
        "wall_spacing_ratio": float(wall_spacing_ratio),
        "time_marching": time_marching,
        "pseudo_cfl_start": None if pseudo_cfl_start is None else float(pseudo_cfl_start),
        "pseudo_ramp_steps": int(pseudo_ramp_steps),
        "pseudo_precond_ref_mach": float(pseudo_precond_ref_mach),
        "pseudo_precond_min_scale": float(pseudo_precond_min_scale),
        "residual_smoothing_eps": float(residual_smoothing_eps),
        "residual_smoothing_passes": int(residual_smoothing_passes),
        "adaptive_pseudo": bool(adaptive_pseudo),
        "adaptive_interval": int(adaptive_interval),
        "adaptive_cfl_growth": float(adaptive_cfl_growth),
        "adaptive_cfl_shrink": float(adaptive_cfl_shrink),
        "adaptive_target_ratio": float(adaptive_target_ratio),
        "adaptive_fail_ratio": float(adaptive_fail_ratio),
        "adaptive_smoothing_max_eps": (
            None if adaptive_smoothing_max_eps is None else float(adaptive_smoothing_max_eps)
        ),
        "initial_condition": init_mode,
        "init_bl_thickness": float(init_bl_thickness),
        "target_sim_time": None if target_sim_time is None else float(target_sim_time),
        "turbulence_model": turbulence_model,
        "eddy_viscosity_ratio": float(eddy_viscosity_ratio),
        "turbulent_prandtl": float(turbulent_prandtl),
        "sa_nu_tilde_inf_ratio": float(sa_nu_tilde_inf_ratio),
        "mu": float(transport["mu"]),
        "mu_t": float(transport["mu_t"]),
        "mu_eff": float(transport["mu_eff"]),
        "kappa_eff": float(transport["kappa_eff"]),
        "y_plus_min": float(aero["y_plus_min"]),
        "y_plus_mean": float(aero["y_plus_mean"]),
        "y_plus_max": float(aero["y_plus_max"]),
        "pseudo_cfl": float(flow["pseudo_cfl"]),
        "pseudo_cfl_target": float(flow["pseudo_cfl_target"]),
        "pseudo_precond_scale_min": float(flow["pseudo_precond_scale_min"]),
        "pseudo_precond_scale_mean": float(flow["pseudo_precond_scale_mean"]),
        "pseudo_precond_scale_max": float(flow["pseudo_precond_scale_max"]),
        "residual_smoothing_eps": float(flow["residual_smoothing_eps"]),
        "residual_smoothing_base_eps": float(flow["residual_smoothing_base_eps"]),
        "residual_smoothing_max_eps": float(flow["residual_smoothing_max_eps"]),
        "residual_smoothing_passes": int(flow["residual_smoothing_passes"]),
        "pseudo_adaptive_enabled": bool(flow["pseudo_adaptive_enabled"]),
        "pseudo_residual_ratio": float(flow["pseudo_residual_ratio"]),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="NACA0012 Navier-Stokes on O-grid")
    parser.add_argument("--ni", type=int, default=180, help="O-grid circumferential cells")
    parser.add_argument("--nj", type=int, default=60, help="O-grid radial cells")
    parser.add_argument("--ma", type=float, default=0.15, help="Freestream Mach number")
    parser.add_argument("--re", type=float, default=500.0, help="Chord-based Reynolds number")
    parser.add_argument("--aoa", type=float, default=4.0, help="Angle of attack in degrees")
    parser.add_argument("--steps", type=int, default=12000, help="Maximum time steps")
    parser.add_argument("--rfar", type=float, default=15.0, help="Far-field radius in chords")
    parser.add_argument("--cfl", type=float, default=0.12, help="Explicit CFL number")
    parser.add_argument("--report", type=int, default=500, help="Report interval")
    parser.add_argument("--tol", type=float, default=5e-5, help="Residual RMS tolerance")
    parser.add_argument("--save_interval", type=int, default=1000, help="State save interval")
    parser.add_argument(
        "--wall_spacing_ratio",
        type=float,
        default=1.0,
        help="First radial spacing relative to linear average 1/nj; <1 clusters near wall",
    )
    parser.add_argument(
        "--marching",
        choices=["global", "local_pseudo"],
        default="global",
        help="Time marching mode: physical global dt or steady local pseudo-time stepping",
    )
    parser.add_argument(
        "--pseudo_cfl_start",
        type=float,
        default=None,
        help="Initial local pseudo CFL before ramp; defaults to base CFL when omitted",
    )
    parser.add_argument(
        "--pseudo_ramp_steps",
        type=int,
        default=0,
        help="Number of local pseudo iterations used to ramp CFL to the target value",
    )
    parser.add_argument(
        "--pseudo_precond_ref_mach",
        type=float,
        default=1.0,
        help="Reference Mach for pseudo-time acoustic scaling; <1 reduces low-Mach stiffness",
    )
    parser.add_argument(
        "--pseudo_precond_min_scale",
        type=float,
        default=1.0,
        help="Minimum acoustic scaling factor for pseudo-time preconditioning",
    )
    parser.add_argument(
        "--residual_smoothing_eps",
        type=float,
        default=0.0,
        help="Jacobi-like residual smoothing strength for local_pseudo",
    )
    parser.add_argument(
        "--residual_smoothing_passes",
        type=int,
        default=0,
        help="Number of residual smoothing passes per pseudo-time RK stage",
    )
    parser.add_argument(
        "--adaptive_pseudo",
        action="store_true",
        help="Enable adaptive local CFL ramp and adaptive residual smoothing in local_pseudo mode",
    )
    parser.add_argument(
        "--adaptive_interval",
        type=int,
        default=20,
        help="Pseudo-iteration interval used to evaluate residual ratio and adapt controls",
    )
    parser.add_argument(
        "--adaptive_cfl_growth",
        type=float,
        default=1.03,
        help="CFL multiplier applied when residual ratio is improving",
    )
    parser.add_argument(
        "--adaptive_cfl_shrink",
        type=float,
        default=0.7,
        help="CFL multiplier applied when residual ratio degrades",
    )
    parser.add_argument(
        "--adaptive_target_ratio",
        type=float,
        default=0.995,
        help="Residual ratio threshold below which CFL is allowed to grow",
    )
    parser.add_argument(
        "--adaptive_fail_ratio",
        type=float,
        default=1.005,
        help="Residual ratio threshold above which CFL shrinks and smoothing increases",
    )
    parser.add_argument(
        "--adaptive_smoothing_max_eps",
        type=float,
        default=None,
        help="Upper bound for adaptive residual smoothing epsilon",
    )
    parser.add_argument(
        "--init",
        choices=["auto", "uniform", "wall_ramp"],
        default="auto",
        help="Initial primitive field; auto uses wall_ramp for high-Re/turbulent runs",
    )
    parser.add_argument(
        "--init_bl_thickness",
        type=float,
        default=0.02,
        help="Wall-ramp initialization thickness in chord units",
    )
    parser.add_argument(
        "--target_time",
        type=float,
        default=None,
        help="Stop once physical time reaches this value, even if steps remain",
    )
    parser.add_argument("--transonic", action="store_true", help="Enable experimental transonic path for Ma >= 0.7")
    parser.add_argument(
        "--turbulence",
        choices=["laminar", "constant_eddy_viscosity", "spalart_allmaras"],
        default="laminar",
        help="Transport closure model",
    )
    parser.add_argument(
        "--eddy_ratio",
        type=float,
        default=0.0,
        help="Eddy viscosity ratio mu_t / mu for constant_eddy_viscosity",
    )
    parser.add_argument(
        "--pr_t",
        type=float,
        default=0.9,
        help="Turbulent Prandtl number for constant_eddy_viscosity",
    )
    parser.add_argument(
        "--sa_nu_ratio",
        type=float,
        default=3.0,
        help="Freestream nu_tilde / nu ratio for Spalart-Allmaras",
    )
    parser.add_argument("--plateau_window", type=int, default=5, help="History window for shock plateau check")
    parser.add_argument("--shock_tol", type=float, default=0.03, help="Shock-position plateau tolerance")
    parser.add_argument("--mach_tol", type=float, default=0.05, help="Max-Mach plateau tolerance")
    parser.add_argument(
        "--output",
        type=str,
        default="output_naca0012_ns",
        help="Output directory for figures (empty string disables)",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        result = run_naca0012_ns(
            ni=args.ni,
            nj=args.nj,
            ma=args.ma,
            re=args.re,
            aoa=args.aoa,
            steps=args.steps,
            r_far=args.rfar,
            cfl=args.cfl,
            report=args.report,
            tol=args.tol,
            save_interval=args.save_interval,
            wall_spacing_ratio=args.wall_spacing_ratio,
            time_marching=args.marching,
            pseudo_cfl_start=args.pseudo_cfl_start,
            pseudo_ramp_steps=args.pseudo_ramp_steps,
            pseudo_precond_ref_mach=args.pseudo_precond_ref_mach,
            pseudo_precond_min_scale=args.pseudo_precond_min_scale,
            residual_smoothing_eps=args.residual_smoothing_eps,
            residual_smoothing_passes=args.residual_smoothing_passes,
            adaptive_pseudo=args.adaptive_pseudo,
            adaptive_interval=args.adaptive_interval,
            adaptive_cfl_growth=args.adaptive_cfl_growth,
            adaptive_cfl_shrink=args.adaptive_cfl_shrink,
            adaptive_target_ratio=args.adaptive_target_ratio,
            adaptive_fail_ratio=args.adaptive_fail_ratio,
            adaptive_smoothing_max_eps=args.adaptive_smoothing_max_eps,
            initial_condition=args.init,
            init_bl_thickness=args.init_bl_thickness,
            target_sim_time=args.target_time,
            output_dir=args.output if args.output else None,
            allow_transonic=args.transonic,
            turbulence_model=args.turbulence,
            eddy_viscosity_ratio=args.eddy_ratio,
            turbulent_prandtl=args.pr_t,
            sa_nu_tilde_inf_ratio=args.sa_nu_ratio,
            plateau_window=args.plateau_window,
            shock_tol=args.shock_tol,
            mach_tol=args.mach_tol,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ NACA0012 NS case failed: {exc}")
        sys.exit(1)

    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
