"""
Converging-Diverging Nozzle — Euler Validation Case
===================================================

What: 在對稱 converging-diverging nozzle 上求解 2D Euler，觀察亞音速/超音速演化
Why:  nozzle 幾何比翼型更單純，適合先驗證可壓縮加速與壓縮波路徑
When: 用於 EulerSolver 的 curvilinear 驗證；不含黏性與湍流模型
"""

import os
import sys

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
    SolverControlDescriptor,
    build_history_payload,
    build_state_payload,
)


def compute_cell_centers(x_node: np.ndarray, y_node: np.ndarray, ni: int, nj: int, NG: int) -> tuple[np.ndarray, np.ndarray]:
    """
    計算 interior cell center 座標
    """
    xc = 0.25 * (
        x_node[NG:NG+ni, NG:NG+nj]
        + x_node[NG+1:NG+ni+1, NG:NG+nj]
        + x_node[NG:NG+ni, NG+1:NG+nj+1]
        + x_node[NG+1:NG+ni+1, NG+1:NG+nj+1]
    )
    yc = 0.25 * (
        y_node[NG:NG+ni, NG:NG+nj]
        + y_node[NG+1:NG+ni+1, NG:NG+nj]
        + y_node[NG:NG+ni, NG+1:NG+nj+1]
        + y_node[NG+1:NG+ni+1, NG+1:NG+nj+1]
    )
    return xc, yc


def _isentropic_pressure_ratio_from_mach(mach: float, gamma: float) -> float:
    """
    計算等熵靜壓比 p/p0
    """
    fac = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    return fac ** (-gamma / (gamma - 1.0))


def _mass_flow_function(mach: float, gamma: float) -> float:
    """
    準一維等熵質量流率函數 G(M)
    """
    fac = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    exp = -(gamma + 1.0) / (2.0 * (gamma - 1.0))
    return mach * fac ** exp


def _solve_subsonic_mach_from_massflow(phi: float, gamma: float) -> float:
    """
    由 G(M)=phi 反解亞音速 M（0 < M < 1）
    """
    phi = max(min(phi, _mass_flow_function(0.999999, gamma)), 1e-12)
    lo = 1e-8
    hi = 0.999999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        g_mid = _mass_flow_function(mid, gamma)
        if g_mid < phi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_quasi_1d_nozzle_initial_field(
        x_node: np.ndarray,
        y_node: np.ndarray,
        ni: int,
        nj: int,
        NG: int,
        gamma: float,
        p0: float,
        t0: float,
        pb: float) -> np.ndarray:
    """
    以準一維等熵近似建立 nozzle 初始場

    What: 由 p0/t0/pb 與 A(x) 建立 (rho,u,v,p) 初值
    Why:  均勻初值會引入長暫態，難以快速形成 mdot 與喉部平台
    """
    area_face = np.maximum(
        y_node[NG:NG + ni + 1, NG + nj] - y_node[NG:NG + ni + 1, NG],
        1e-12,
    )
    area_cell = 0.5 * (area_face[:-1] + area_face[1:])
    a_out = float(area_face[-1])

    p_ratio = min(max(pb / max(p0, 1e-12), 1e-6), 0.999999)
    m_out = np.sqrt(
        max(0.0, 2.0 / (gamma - 1.0) * (p_ratio ** (-(gamma - 1.0) / gamma) - 1.0))
    )
    m_out = min(max(float(m_out), 1e-5), 0.999999)

    phi_out = _mass_flow_function(m_out, gamma)
    area_min = max(float(np.min(area_face)), 1e-12)
    phi_choked_limit = area_min / max(a_out, 1e-12) * _mass_flow_function(0.999999, gamma)
    phi_out = min(phi_out, phi_choked_limit)

    w0 = np.zeros((ni, nj, 4), dtype=np.float32)
    for i in range(ni):
        phi_i = phi_out * a_out / max(float(area_cell[i]), 1e-12)
        m_i = _solve_subsonic_mach_from_massflow(phi_i, gamma)
        t_i = t0 / (1.0 + 0.5 * (gamma - 1.0) * m_i * m_i)
        p_i = p0 * (t_i / t0) ** (gamma / (gamma - 1.0))
        rho_i = p_i / max(t_i, 1e-12)
        u_i = m_i * np.sqrt(max(gamma * t_i, 1e-12))
        w0[i, :, 0] = rho_i
        w0[i, :, 1] = u_i
        w0[i, :, 2] = 0.0
        w0[i, :, 3] = p_i
    return w0


def plot_centerline_mach(
        x_center: np.ndarray,
        mach_center: np.ndarray,
        area_ratio: np.ndarray,
        ma_in: float,
        save_path: str):
    """
    繪製中心線 Mach 與面積比
    """
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x_center, mach_center, "b-", lw=1.8, label="Centerline Mach")
    ax1.axhline(1.0, color="r", lw=0.8, ls="--", label="Mach 1")
    ax1.axhline(ma_in, color="0.35", lw=0.8, ls=":", label="Initial Mach")
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("Mach Number", fontsize=12, color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_center, area_ratio, "k--", lw=1.2, label="Area Ratio A/A*")
    ax2.set_ylabel("Area Ratio", fontsize=12, color="k")
    ax2.tick_params(axis="y", labelcolor="k")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")
    ax1.set_title("Centerline Mach and Nozzle Area Ratio", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mass_flow_profile(
        x_face: np.ndarray,
        mdot_profile: np.ndarray,
        throat_x: float,
        save_path: str):
    """
    繪製 nozzle 各截面質量流率分佈
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x_face, mdot_profile, "b-", lw=1.7, label="Section mass flow rate")
    ax.axvline(throat_x, color="r", ls="--", lw=1.0, label="Throat location")
    ax.axhline(np.mean(mdot_profile), color="0.35", ls=":", lw=1.0, label="Mean mdot")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Mass Flow Rate", fontsize=12)
    ax.set_title("Nozzle Section Mass Flow Distribution", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_nozzle_history_diagnostics(history_dict: dict, save_path: str):
    """
    繪製 nozzle 收斂/平台診斷歷程
    """
    steps = np.asarray(history_dict["steps"], dtype=float)
    residual = np.asarray(history_dict["residual"], dtype=float)
    mach_throat = np.asarray(history_dict["mach_throat"], dtype=float)
    p_throat = np.asarray(history_dict["p_throat"], dtype=float)
    mdot_throat = np.asarray(history_dict["mdot_throat"], dtype=float)
    mdot_balance = np.asarray(history_dict["mdot_balance"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    axes = axes.ravel()

    axes[0].semilogy(steps, np.maximum(residual, 1e-30), "k-", lw=1.3)
    axes[0].set_ylabel("Residual RMS")
    axes[0].set_title("Residual", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mach_throat, "b-", lw=1.4)
    axes[1].set_ylabel("Mach Number")
    axes[1].set_title("Throat Mach", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, p_throat, "r-", lw=1.4)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Pressure")
    axes[2].set_title("Throat Pressure", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    ax3 = axes[3]
    ax3.plot(steps, mdot_throat, "g-", lw=1.4, label="Throat mdot")
    ax3_t = ax3.twinx()
    ax3_t.plot(steps, 100.0 * mdot_balance, "m--", lw=1.2, label="mdot balance (%)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Mass Flow Rate")
    ax3_t.set_ylabel("Balance Error (%)")
    ax3.set_title("Throat mdot / Balance", fontsize=12)
    ax3.grid(True, alpha=0.3)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_t.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

    fig.suptitle("Nozzle Convergence Diagnostics", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_velocity_vector_field(
        x_node: np.ndarray,
        y_node: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        ma_in: float,
        save_path: str,
        stride_i: int = 6,
        stride_j: int = 4):
    """
    繪製速度向量場（cell centers）
    """
    ni, nj = u.shape
    NG = (x_node.shape[0] - 1 - ni) // 2
    x_in = x_node[NG:NG + ni + 1, NG:NG + nj + 1]
    y_in = y_node[NG:NG + ni + 1, NG:NG + nj + 1]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    vel_mag = np.sqrt(u**2 + v**2)
    mesh = ax.pcolormesh(x_in, y_in, vel_mag, shading="flat", cmap="viridis")
    cb = plt.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Velocity Magnitude", fontsize=10)
    ax.quiver(
        xc[::stride_i, ::stride_j],
        yc[::stride_i, ::stride_j],
        u[::stride_i, ::stride_j],
        v[::stride_i, ::stride_j],
        color="white",
        alpha=0.75,
        scale=35.0,
        width=0.0022,
        headwidth=3.0,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Velocity Vector Field — CD Nozzle, Ma_init={ma_in:.3f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_nozzle_cell_field(
        x_node: np.ndarray,
        y_node: np.ndarray,
        field_values: np.ndarray,
        field_label: str,
        title: str,
        cmap: str,
        save_path: str):
    """
    直接在 nozzle curvilinear cells 上繪製標量場（不做插值）
    """
    ni, nj = field_values.shape
    NG = (x_node.shape[0] - 1 - ni) // 2
    x_in = x_node[NG:NG + ni + 1, NG:NG + nj + 1]
    y_in = y_node[NG:NG + ni + 1, NG:NG + nj + 1]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    mesh = ax.pcolormesh(x_in, y_in, field_values, shading="flat", cmap=cmap)
    cb = plt.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(field_label, fontsize=10)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_internal_mesh(
        x_node: np.ndarray,
        y_node: np.ndarray,
        ni: int,
        nj: int,
        NG: int,
        save_path: str,
        line_skip_i: int = 1,
        line_skip_j: int = 1):
    """
    繪製噴嘴內部網格（只畫 interior nodes）

    What: 只取 [NG:NG+ni+1, NG:NG+nj+1] 節點，忽略 ghost/外圍
    Why:  幾何審視時需聚焦「內流道」本體，不應混入數值 ghost 區
    """
    if line_skip_i <= 0 or line_skip_j <= 0:
        raise ValueError(f"line_skip_i/line_skip_j must be positive, got {line_skip_i}, {line_skip_j}")

    x_in = x_node[NG:NG + ni + 1, NG:NG + nj + 1]
    y_in = y_node[NG:NG + ni + 1, NG:NG + nj + 1]

    fig, ax = plt.subplots(figsize=(10, 3.8))

    for j in range(0, nj + 1, line_skip_j):
        ax.plot(x_in[:, j], y_in[:, j], color="0.35", lw=0.6)
    for i in range(0, ni + 1, line_skip_i):
        ax.plot(x_in[i, :], y_in[i, :], color="0.35", lw=0.6)

    # 強調上下壁輪廓
    ax.plot(x_in[:, 0], y_in[:, 0], "k-", lw=1.8)
    ax.plot(x_in[:, -1], y_in[:, -1], "k-", lw=1.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Internal Mesh — Converging-Diverging Nozzle", fontsize=13)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def compute_nozzle_mass_flow_profile(
        x_node: np.ndarray,
        y_node: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        NG: int) -> tuple[np.ndarray, np.ndarray]:
    """
    計算每個 i-face 截面的質量流率分佈

    What: 對每條 i-face 線積分 `mdot = Σ ρ (V·S_i)`
    Why:  與 FVM 面通量定義一致，可直接檢查 nozzle 內流質量守恆
    """
    ni, nj = rho.shape
    x_face = np.zeros(ni + 1, dtype=np.float64)
    mdot_profile = np.zeros(ni + 1, dtype=np.float64)

    for i_face in range(ni + 1):
        i_node = NG + i_face
        mdot_val = 0.0
        x_acc = 0.0
        for j in range(nj):
            jn = NG + j
            x0 = float(x_node[i_node, jn])
            y0 = float(y_node[i_node, jn])
            x1 = float(x_node[i_node, jn + 1])
            y1 = float(y_node[i_node, jn + 1])
            sx = y1 - y0
            sy = -(x1 - x0)

            if i_face == 0:
                rho_f = float(rho[0, j])
                u_f = float(u[0, j])
                v_f = float(v[0, j])
            elif i_face == ni:
                rho_f = float(rho[ni - 1, j])
                u_f = float(u[ni - 1, j])
                v_f = float(v[ni - 1, j])
            else:
                rho_f = 0.5 * float(rho[i_face - 1, j] + rho[i_face, j])
                u_f = 0.5 * float(u[i_face - 1, j] + u[i_face, j])
                v_f = 0.5 * float(v[i_face - 1, j] + v[i_face, j])

            mdot_val += rho_f * (u_f * sx + v_f * sy)
            x_acc += 0.5 * (x0 + x1)

        x_face[i_face] = x_acc / max(float(nj), 1.0)
        mdot_profile[i_face] = mdot_val

    return x_face, mdot_profile


def evaluate_nozzle_plateau(
        history: list[dict],
        window: int,
        mdot_rel_tol: float,
        mach_tol: float,
        p_rel_tol: float,
        mdot_balance_tol: float) -> dict:
    """
    檢查 nozzle steady baseline 是否形成平台

    What: 以最近 `window` 筆資料檢查喉部狀態與質量流率穩定性
    Why:  單看 residual 可能誤判；平台判據可避免把暫態誤認為穩態
    """
    if len(history) < max(window, 2):
        return {
            "window": int(window),
            "ready": False,
            "plateau": False,
            "mdot_rel_span": np.nan,
            "mach_span": np.nan,
            "p_rel_span": np.nan,
            "mdot_balance_max": np.nan,
        }

    recent = history[-window:]
    mdot_vals = np.array([row["mdot_throat"] for row in recent], dtype=float)
    mach_vals = np.array([row["mach_throat"] for row in recent], dtype=float)
    p_vals = np.array([row["p_throat"] for row in recent], dtype=float)
    mdot_balance_vals = np.array([row["mdot_balance"] for row in recent], dtype=float)

    mdot_scale = max(abs(float(np.mean(mdot_vals))), 1e-12)
    p_scale = max(abs(float(np.mean(p_vals))), 1e-12)
    mdot_rel_span = float(np.max(mdot_vals) - np.min(mdot_vals)) / mdot_scale
    mach_span = float(np.max(mach_vals) - np.min(mach_vals))
    p_rel_span = float(np.max(p_vals) - np.min(p_vals)) / p_scale
    mdot_balance_max = float(np.max(mdot_balance_vals))
    plateau = (
        mdot_rel_span <= mdot_rel_tol
        and mach_span <= mach_tol
        and p_rel_span <= p_rel_tol
        and mdot_balance_max <= mdot_balance_tol
    )
    return {
        "window": int(window),
        "ready": True,
        "plateau": bool(plateau),
        "mdot_rel_span": mdot_rel_span,
        "mach_span": mach_span,
        "p_rel_span": p_rel_span,
        "mdot_balance_max": mdot_balance_max,
    }


def compute_nozzle_diagnostics(
        solver,
        x_node: np.ndarray,
        y_node: np.ndarray,
        h_throat: float) -> dict:
    """
    計算 nozzle 驗證關鍵量

    What: 輸出 Mach 場、中心線 Mach、喉部/出口 Mach、超音速面積比例
    Why:  nozzle 驗證核心在於加速能力與喉部行為，而非升阻力
    """
    rho_n, u_n, v_n, p_n = solver.get_primitive()
    gamma = solver.gamma
    vel_mag = np.sqrt(u_n**2 + v_n**2)
    temperature = p_n / np.maximum(rho_n, 1e-12)  # nondimensional: R=1
    c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, 1e-12))
    mach_n = vel_mag / c_n

    NG = solver.NG
    ni = solver.ni
    nj = solver.nj
    xc, yc = compute_cell_centers(x_node, y_node, ni, nj, NG)

    center_j = int(np.argmin(np.abs(np.mean(yc, axis=0))))
    centerline_x = xc[:, center_j]
    centerline_mach = mach_n[:, center_j]
    centerline_vel = vel_mag[:, center_j]
    centerline_pressure = p_n[:, center_j]
    centerline_temperature = temperature[:, center_j]

    y_bottom = 0.5 * (y_node[NG:NG+ni, NG] + y_node[NG+1:NG+ni+1, NG])
    y_top = 0.5 * (y_node[NG:NG+ni, NG+nj] + y_node[NG+1:NG+ni+1, NG+nj])
    height_x = np.maximum(y_top - y_bottom, 1e-12)
    area_ratio = height_x / max(h_throat, 1e-12)
    throat_idx = int(np.argmin(area_ratio))
    throat_x = float(centerline_x[throat_idx])

    x_face, mdot_profile = compute_nozzle_mass_flow_profile(
        x_node=x_node,
        y_node=y_node,
        rho=rho_n,
        u=u_n,
        v=v_n,
        NG=NG,
    )
    throat_face_idx = int(np.argmin(np.abs(x_face - throat_x)))
    mdot_inlet = float(mdot_profile[0])
    mdot_outlet = float(mdot_profile[-1])
    mdot_throat = float(mdot_profile[throat_face_idx])
    mdot_mean = float(np.mean(mdot_profile))
    mdot_span = float(np.max(mdot_profile) - np.min(mdot_profile))
    mdot_rel_span = mdot_span / max(abs(mdot_mean), 1e-12)
    mdot_balance = max(
        abs(mdot_inlet - mdot_throat),
        abs(mdot_outlet - mdot_throat),
    ) / max(abs(mdot_throat), 1e-12)

    band = max(3, ni // 20)
    mach_inlet = float(np.mean(centerline_mach[:band]))
    mach_outlet = float(np.mean(centerline_mach[-band:]))
    vel_inlet = float(np.mean(centerline_vel[:band]))
    vel_outlet = float(np.mean(centerline_vel[-band:]))
    p_inlet = float(np.mean(centerline_pressure[:band]))
    p_outlet = float(np.mean(centerline_pressure[-band:]))
    t_inlet = float(np.mean(centerline_temperature[:band]))
    t_outlet = float(np.mean(centerline_temperature[-band:]))

    return {
        "rho": rho_n,
        "u": u_n,
        "v": v_n,
        "p": p_n,
        "temperature": temperature,
        "vel_mag": vel_mag,
        "mach": mach_n,
        "xc": xc,
        "yc": yc,
        "centerline_x": centerline_x,
        "centerline_mach": centerline_mach,
        "centerline_vel": centerline_vel,
        "centerline_pressure": centerline_pressure,
        "centerline_temperature": centerline_temperature,
        "area_ratio": area_ratio,
        "throat_idx": throat_idx,
        "throat_x": throat_x,
        "mach_max": float(np.max(mach_n)),
        "mach_min": float(np.min(mach_n)),
        "mach_throat": float(centerline_mach[throat_idx]),
        "mach_inlet": mach_inlet,
        "mach_outlet": mach_outlet,
        "vel_throat": float(centerline_vel[throat_idx]),
        "vel_inlet": vel_inlet,
        "vel_outlet": vel_outlet,
        "p_throat": float(centerline_pressure[throat_idx]),
        "p_inlet": p_inlet,
        "p_outlet": p_outlet,
        "t_throat": float(centerline_temperature[throat_idx]),
        "t_inlet": t_inlet,
        "t_outlet": t_outlet,
        "x_face": x_face,
        "mdot_profile": mdot_profile,
        "throat_face_idx": throat_face_idx,
        "mdot_inlet": mdot_inlet,
        "mdot_throat": mdot_throat,
        "mdot_outlet": mdot_outlet,
        "mdot_mean": mdot_mean,
        "mdot_span": mdot_span,
        "mdot_rel_span": mdot_rel_span,
        "mdot_balance": mdot_balance,
        "supersonic_fraction": float(np.mean(mach_n > 1.0)),
    }


def _initialize_cd_nozzle_case(
        runner: CaseRunner,
        solver,
        _bc_handle,
        *,
        initial_condition: str,
        p0: float,
        t0: float,
        pb: float,
        rho_inf: float,
        u_inf: float,
        v_inf: float,
        p_inf: float):
    """
    CD nozzle 的正式初始化流程。

    What:
    - 只負責建立物理初始場

    Why:
    - solver controls 已升格為 descriptor；initializer 不再承擔 configuration glue
    """
    x_node = runner.grid.x_node
    y_node = runner.grid.y_node
    if initial_condition == "quasi1d":
        w0 = build_quasi_1d_nozzle_initial_field(
            x_node=x_node,
            y_node=y_node,
            ni=solver.ni,
            nj=solver.nj,
            NG=solver.NG,
            gamma=solver.gamma,
            p0=p0,
            t0=t0,
            pb=pb,
        )
        solver.init_from_primitive_numpy(w0)
    else:
        solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)
    solver._update_ghost()


def _build_cd_nozzle_solver_controls(
        *,
        p0: float,
        t0: float,
        pb: float,
        mach_inlet_bc_eff: float,
        inlet_relax: float,
        outlet_relax: float,
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
        adaptive_smoothing_max_eps: float,
        reconstruction_order: str,
        cfl: float) -> list[SolverControlDescriptor]:
    """
    建立 CD nozzle 的正式 solver control descriptors。

    Why:
    - nozzle BC、pseudo-time 與 reconstruction 都應成為可重用的 toolkit 配置，而非直接 method call
    """
    controls = [
        SolverControlDescriptor.nozzle_inlet(
            p0=p0,
            t0=t0,
            mach_in=mach_inlet_bc_eff,
            relax=inlet_relax,
        ),
        SolverControlDescriptor.nozzle_outlet(
            back_pressure=pb,
            relax=outlet_relax,
        ),
        SolverControlDescriptor.spatial_reconstruction(reconstruction_order),
        SolverControlDescriptor.time_marching(time_marching),
    ]
    if time_marching == "local_pseudo":
        pseudo_cfl_start_eff = cfl if pseudo_cfl_start is None else pseudo_cfl_start
        controls.extend(
            [
                SolverControlDescriptor.pseudo_time_controls(
                    cfl_start=pseudo_cfl_start_eff,
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
                    cfl_min=pseudo_cfl_start_eff,
                    cfl_max=cfl,
                    growth=adaptive_cfl_growth,
                    shrink=adaptive_cfl_shrink,
                    target_ratio=adaptive_target_ratio,
                    fail_ratio=adaptive_fail_ratio,
                    interval=adaptive_interval,
                    smoothing_max_eps=adaptive_smoothing_max_eps,
                ),
            ]
        )
    return controls


def _cd_nozzle_diagnostics_hook(
        runner: CaseRunner,
        solver,
        diagnostics: dict,
        *,
        h_throat: float) -> dict:
    """
    派生 CD nozzle benchmark 專屬 diagnostics。

    Why:
    - solver 只提供通用 Euler 指標；nozzle 驗證還需要喉部 Mach 與質量流率平台量
    """
    diag = compute_nozzle_diagnostics(
        solver,
        runner.grid.x_node,
        runner.grid.y_node,
        h_throat=h_throat,
    )
    mach_inlet = max(abs(float(diag["mach_inlet"])), 1e-12)
    return {
        "mach_inlet": float(diag["mach_inlet"]),
        "mach_throat": float(diag["mach_throat"]),
        "mach_outlet": float(diag["mach_outlet"]),
        "mdot_inlet": float(diag["mdot_inlet"]),
        "mdot_throat": float(diag["mdot_throat"]),
        "mdot_outlet": float(diag["mdot_outlet"]),
        "mdot_rel_span": float(diag["mdot_rel_span"]),
        "mdot_balance": float(diag["mdot_balance"]),
        "supersonic_fraction": float(diag["supersonic_fraction"]),
        "nozzle_acceleration_ratio": float(diag["mach_throat"] / mach_inlet),
    }


def build_cd_nozzle_runner(
        *,
        ni: int = 80,
        nj: int = 32,
        ma_init: float = 0.12,
        length: float = 3.0,
        h_inlet: float = 1.0,
        h_throat: float = 0.70,
        h_exit: float | None = 1.0,
        throat_x: float | None = None,
        conv_power: float = 2.4,
        div_power: float = 3.0,
        throat_blend: float = 0.35,
        p0: float = 0.0,
        t0: float = 0.0,
        pb: float = 0.0,
        pb_ratio: float = 0.985,
        mach_inlet_bc: float = 0.0,
        inlet_relax: float = 0.35,
        outlet_relax: float = 0.35,
        cfl: float = 0.20,
        time_marching: str = "local_pseudo",
        pseudo_cfl_start: float | None = 0.05,
        pseudo_ramp_steps: int = 800,
        pseudo_precond_ref_mach: float = 0.2,
        pseudo_precond_min_scale: float = 0.2,
        residual_smoothing_eps: float = 0.05,
        residual_smoothing_passes: int = 1,
        adaptive_pseudo: bool = False,
        adaptive_interval: int = 20,
        adaptive_cfl_growth: float = 1.03,
        adaptive_cfl_shrink: float = 0.70,
        adaptive_target_ratio: float = 0.995,
        adaptive_fail_ratio: float = 1.010,
        adaptive_smoothing_max_eps: float = 0.24,
        initial_condition: str = "quasi1d",
        reconstruction_order: str = "first") -> tuple[CaseRunner, dict]:
    """
    建立 CD nozzle Euler 的 CaseRunner。

    What:
    - 組裝 curvilinear nozzle grid、Euler solver、wall BC 與 characteristic nozzle BC 初始化

    Why:
    - nozzle 是比 transonic bump 更接近 1D 可驗證物理的 compressible benchmark，
      適合作為下一個正式 registry benchmark
    """
    from fvm_taichi import EulerSolver, generate_reference_nozzle_grid

    if ma_init <= 0.0:
        raise ValueError(f"ma_init must be positive, got {ma_init}")
    if cfl <= 0.0:
        raise ValueError(f"cfl must be positive, got {cfl}")
    if initial_condition not in {"uniform", "quasi1d"}:
        raise ValueError(f"initial_condition must be 'uniform' or 'quasi1d', got {initial_condition}")
    if reconstruction_order not in {"first", "second"}:
        raise ValueError(f"reconstruction_order must be 'first' or 'second', got {reconstruction_order}")
    if time_marching not in {"global", "local_pseudo"}:
        raise ValueError(f"time_marching must be 'global' or 'local_pseudo', got {time_marching}")

    gamma = 1.4
    rho_inf = 1.0
    p_inf = rho_inf / gamma
    u_inf = float(ma_init)
    v_inf = 0.0

    if h_exit is None:
        h_exit = h_inlet
    mach_inlet_bc_eff = ma_init if mach_inlet_bc <= 0.0 else mach_inlet_bc
    t_inf = p_inf / rho_inf
    fac0 = 1.0 + 0.5 * (gamma - 1.0) * mach_inlet_bc_eff**2
    if t0 <= 0.0:
        t0 = t_inf * fac0
    if p0 <= 0.0:
        p0 = p_inf * fac0 ** (gamma / (gamma - 1.0))
    if pb <= 0.0:
        pb = p0 * pb_ratio

    x_node, y_node = generate_reference_nozzle_grid(
        ni=ni,
        nj=nj,
        length=length,
        h_inlet=h_inlet,
        h_throat=h_throat,
        h_exit=h_exit,
        throat_x=throat_x,
        conv_power=conv_power,
        div_power=div_power,
        throat_blend=throat_blend,
        NG=EulerSolver.NG,
    )

    setup = {
        "gamma": gamma,
        "rho_inf": rho_inf,
        "u_inf": u_inf,
        "v_inf": v_inf,
        "p_inf": p_inf,
        "p0": float(p0),
        "t0": float(t0),
        "pb": float(pb),
        "mach_inlet_bc_eff": float(mach_inlet_bc_eff),
        "h_throat": float(h_throat),
    }
    solver_controls = _build_cd_nozzle_solver_controls(
        p0=setup["p0"],
        t0=setup["t0"],
        pb=setup["pb"],
        mach_inlet_bc_eff=setup["mach_inlet_bc_eff"],
        inlet_relax=inlet_relax,
        outlet_relax=outlet_relax,
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
        reconstruction_order=reconstruction_order,
        cfl=cfl,
    )

    runner = CaseRunner(
        name="cd_nozzle_euler",
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
        solver_controls=solver_controls,
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_cd_nozzle_case(
            runner_obj,
            solver_obj,
            bc_handle,
            initial_condition=initial_condition,
            p0=setup["p0"],
            t0=setup["t0"],
            pb=setup["pb"],
            rho_inf=setup["rho_inf"],
            u_inf=setup["u_inf"],
            v_inf=setup["v_inf"],
            p_inf=setup["p_inf"],
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _cd_nozzle_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            h_throat=setup["h_throat"],
        ),
    )
    return runner, setup


def run_cd_nozzle_case(
        ni: int = 240,
        nj: int = 80,
        ma_init: float = 0.30,
        steps: int = 3000,
        length: float = 3.0,
        h_inlet: float = 1.0,
        h_throat: float = 0.40,
        h_exit: float | None = None,
        throat_x: float | None = None,
        conv_power: float = 2.4,
        div_power: float = 3.0,
        throat_blend: float = 0.35,
        p0: float = 0.0,
        t0: float = 0.0,
        pb: float = 0.0,
        pb_ratio: float = 0.85,
        mach_inlet_bc: float = 0.0,
        inlet_relax: float = 0.35,
        outlet_relax: float = 0.35,
        cfl: float = 0.35,
        solver_model: str = "euler",
        re: float = 1.0e4,
        pr: float = 0.72,
        time_marching: str = "local_pseudo",
        pseudo_cfl_start: float | None = 0.08,
        pseudo_ramp_steps: int = 1200,
        pseudo_precond_ref_mach: float = 0.30,
        pseudo_precond_min_scale: float = 0.25,
        residual_smoothing_eps: float = 0.08,
        residual_smoothing_passes: int = 2,
        adaptive_pseudo: bool = False,
        adaptive_interval: int = 20,
        adaptive_cfl_growth: float = 1.03,
        adaptive_cfl_shrink: float = 0.70,
        adaptive_target_ratio: float = 0.995,
        adaptive_fail_ratio: float = 1.010,
        adaptive_smoothing_max_eps: float = 0.24,
        initial_condition: str = "quasi1d",
        reconstruction_order: str = "second",
        report: int = 200,
        tol: float = 1e-5,
        plateau_window: int = 6,
        plateau_mdot_rel_tol: float = 2e-3,
        plateau_mach_tol: float = 2e-3,
        plateau_p_rel_tol: float = 2e-3,
        plateau_mdot_balance_tol: float = 5e-3,
        require_plateau_for_converged: bool = True,
        output_dir: str | None = None) -> dict:
    """
    執行單一 nozzle case
    """
    from fvm_taichi import EulerSolver, NavierStokesSolver, generate_reference_nozzle_grid
    from cfd_taichi.configuration import apply_solver_control_descriptors

    if ma_init <= 0.0:
        raise ValueError(f"ma_init must be positive, got {ma_init}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if time_marching not in {"global", "local_pseudo"}:
        raise ValueError(f"time_marching must be 'global' or 'local_pseudo', got {time_marching}")
    if initial_condition not in {"uniform", "quasi1d"}:
        raise ValueError(f"initial_condition must be 'uniform' or 'quasi1d', got {initial_condition}")
    if reconstruction_order not in {"first", "second"}:
        raise ValueError(f"reconstruction_order must be 'first' or 'second', got {reconstruction_order}")
    if solver_model not in {"euler", "ns"}:
        raise ValueError(f"solver_model must be 'euler' or 'ns', got {solver_model}")
    if solver_model == "ns":
        if re <= 0.0:
            raise ValueError(f"re must be positive for NS mode, got {re}")
        if pr <= 0.0:
            raise ValueError(f"pr must be positive for NS mode, got {pr}")
    if plateau_window <= 1:
        raise ValueError(f"plateau_window must be > 1, got {plateau_window}")
    if plateau_mdot_rel_tol <= 0.0 or plateau_mach_tol <= 0.0 or plateau_p_rel_tol <= 0.0:
        raise ValueError(
            "plateau tolerances must be positive, got "
            f"{plateau_mdot_rel_tol}, {plateau_mach_tol}, {plateau_p_rel_tol}"
        )
    if plateau_mdot_balance_tol <= 0.0:
        raise ValueError(f"plateau_mdot_balance_tol must be positive, got {plateau_mdot_balance_tol}")

    gamma = 1.4
    rho_inf = 1.0
    p_inf = rho_inf / gamma
    u_inf = ma_init
    v_inf = 0.0
    mach_inlet_bc_eff = ma_init if mach_inlet_bc <= 0.0 else mach_inlet_bc
    if pb_ratio <= 0.0:
        raise ValueError(f"pb_ratio must be positive, got {pb_ratio}")
    if h_exit is None:
        h_exit = h_inlet
    t_inf = p_inf / rho_inf
    fac0 = 1.0 + 0.5 * (gamma - 1.0) * mach_inlet_bc_eff**2
    if t0 <= 0.0:
        t0 = t_inf * fac0
    if p0 <= 0.0:
        p0 = p_inf * fac0 ** (gamma / (gamma - 1.0))
    if p0 <= 0.0 or t0 <= 0.0:
        raise ValueError(f"p0 and t0 must be positive after initialization, got p0={p0}, t0={t0}")
    if pb <= 0.0:
        pb = p0 * pb_ratio
    if pb <= 0.0:
        raise ValueError(f"pb must be positive, got {pb}")
    if time_marching == "local_pseudo":
        if pseudo_cfl_start is not None and (pseudo_cfl_start <= 0.0 or pseudo_cfl_start > cfl):
            raise ValueError(
                f"pseudo_cfl_start must satisfy 0 < pseudo_cfl_start <= cfl ({cfl}), got {pseudo_cfl_start}"
            )
        if pseudo_ramp_steps < 0:
            raise ValueError(f"pseudo_ramp_steps must be >= 0, got {pseudo_ramp_steps}")
        if pseudo_precond_ref_mach <= 0.0:
            raise ValueError(
                f"pseudo_precond_ref_mach must be positive, got {pseudo_precond_ref_mach}"
            )
        if not (0.0 < pseudo_precond_min_scale <= 1.0):
            raise ValueError(
                "pseudo_precond_min_scale must satisfy 0 < value <= 1, "
                f"got {pseudo_precond_min_scale}"
            )
        if residual_smoothing_eps < 0.0:
            raise ValueError(f"residual_smoothing_eps must be >= 0, got {residual_smoothing_eps}")
        if residual_smoothing_passes < 0:
            raise ValueError(f"residual_smoothing_passes must be >= 0, got {residual_smoothing_passes}")
        if adaptive_interval <= 0:
            raise ValueError(f"adaptive_interval must be positive, got {adaptive_interval}")
        if adaptive_cfl_growth < 1.0:
            raise ValueError(f"adaptive_cfl_growth must be >= 1, got {adaptive_cfl_growth}")
        if not (0.0 < adaptive_cfl_shrink <= 1.0):
            raise ValueError(
                f"adaptive_cfl_shrink must satisfy 0 < value <= 1, got {adaptive_cfl_shrink}"
            )
        if not (0.0 < adaptive_target_ratio <= adaptive_fail_ratio):
            raise ValueError(
                "adaptive ratios must satisfy 0 < target <= fail, got "
                f"{adaptive_target_ratio}, {adaptive_fail_ratio}"
            )
        if adaptive_smoothing_max_eps < residual_smoothing_eps:
            raise ValueError(
                "adaptive_smoothing_max_eps must be >= residual_smoothing_eps, "
                f"got {adaptive_smoothing_max_eps} < {residual_smoothing_eps}"
            )

    print("=== Case Setup ===")
    print(f"Grid: ni={ni}, nj={nj}, length={length:.3f}")
    print(
        f"Solver: {solver_model.upper()} "
        f"({'no-slip walls' if solver_model == 'ns' else 'slip walls'})"
    )
    if solver_model == "ns":
        print(f"NS Params: Re={re:.3e}, Pr={pr:.3f}")
    print(
        f"Nozzle: h_inlet={h_inlet:.3f}, h_throat={h_throat:.3f}, h_exit={h_exit:.3f}, "
        f"conv_power={conv_power:.2f}, div_power={div_power:.2f}, throat_blend={throat_blend:.2f}"
    )
    print(f"Flow: Ma_init={ma_init:.3f}, rho_inf={rho_inf:.3f}, p_inf={p_inf:.5f}, CFL={cfl:.3f}")
    print(
        f"BC: p0={p0:.5f}, t0={t0:.5f}, pb={pb:.5f}, "
        f"pb/p0={pb/max(p0,1e-12):.4f}, Mach_in_bc={mach_inlet_bc_eff:.3f}"
    )
    print(
        f"Run: steps={steps}, report={report}, tol={tol:.1e}, "
        f"mode={time_marching}, init={initial_condition}, recon={reconstruction_order}"
    )
    print(
        "Plateau: "
        f"window={plateau_window}, mdot_span<={plateau_mdot_rel_tol:.2e}, "
        f"mach_span<={plateau_mach_tol:.2e}, p_span<={plateau_p_rel_tol:.2e}, "
        f"mdot_balance<={plateau_mdot_balance_tol:.2e}"
    )
    if time_marching == "local_pseudo":
        pseudo_cfl_start_eff = cfl if pseudo_cfl_start is None else pseudo_cfl_start
        print(
            f"Pseudo: cfl_start={pseudo_cfl_start_eff:.4f}, ramp={pseudo_ramp_steps}, "
            f"precond_ref_mach={pseudo_precond_ref_mach:.3f}, precond_min_scale={pseudo_precond_min_scale:.3f}"
        )
        print(
            f"Smoothing: eps={residual_smoothing_eps:.3f}, passes={residual_smoothing_passes}, "
            f"adaptive={adaptive_pseudo}"
        )

    case_start = time.time()
    x_node, y_node = generate_reference_nozzle_grid(
        ni=ni,
        nj=nj,
        length=length,
        h_inlet=h_inlet,
        h_throat=h_throat,
        h_exit=h_exit,
        throat_x=throat_x,
        conv_power=conv_power,
        div_power=div_power,
        throat_blend=throat_blend,
        NG=EulerSolver.NG,
    )

    if solver_model == "ns":
        solver = NavierStokesSolver(
            ni=ni,
            nj=nj,
            gamma=gamma,
            cfl=cfl,
            re=re,
            pr=pr,
            u_ref=max(abs(u_inf), 1e-6),
            length_scale=max(h_throat, 1e-6),
            rho_ref=rho_inf,
        )
    else:
        solver = EulerSolver(ni=ni, nj=nj, gamma=gamma, cfl=cfl)
    solver.set_curvilinear_grid(x_node, y_node)
    if solver_model == "ns":
        solver.set_no_slip_wall("j_min", u_wall=0.0, v_wall=0.0, temperature=None)
        solver.set_no_slip_wall("j_max", u_wall=0.0, v_wall=0.0, temperature=None)
    else:
        solver.set_slip_wall_j_min()
        solver.set_slip_wall_j_max()
    apply_solver_control_descriptors(
        solver,
        _build_cd_nozzle_solver_controls(
            p0=p0,
            t0=t0,
            pb=pb,
            mach_inlet_bc_eff=mach_inlet_bc_eff,
            inlet_relax=inlet_relax,
            outlet_relax=outlet_relax,
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
            reconstruction_order=reconstruction_order,
            cfl=cfl,
        ),
    )
    if initial_condition == "quasi1d":
        solver_ng = type(solver).NG
        w0 = build_quasi_1d_nozzle_initial_field(
            x_node=x_node,
            y_node=y_node,
            ni=ni,
            nj=nj,
            NG=solver_ng,
            gamma=gamma,
            p0=p0,
            t0=t0,
            pb=pb,
        )
        solver.init_from_primitive_numpy(w0)
    else:
        solver.init_uniform(rho=rho_inf, u=u_inf, v=v_inf, p=p_inf)
    solver._update_ghost()

    history = []
    converged = False
    final_step = 0
    final_residual = np.nan
    diag = compute_nozzle_diagnostics(solver, x_node, y_node, h_throat=h_throat)
    plateau_status = evaluate_nozzle_plateau(
        history=history,
        window=plateau_window,
        mdot_rel_tol=plateau_mdot_rel_tol,
        mach_tol=plateau_mach_tol,
        p_rel_tol=plateau_p_rel_tol,
        mdot_balance_tol=plateau_mdot_balance_tol,
    )

    print("\n=== Runtime Monitor ===")
    print(
        f"{'step':>8}  {'res':>10}  {'M_*':>8}  {'M_out':>8}  "
        f"{'mdot_span%':>10}  {'mdot_bal%':>10}  {'plateau':>7}"
    )
    print(f"{'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}")

    for step in range(1, steps + 1):
        solver.step()
        if step % report == 0 or step == 1:
            ti.sync()
            residual = solver.get_residual_norm()
            dt_diag = solver.get_dt_diagnostics()
            diag = compute_nozzle_diagnostics(solver, x_node, y_node, h_throat=h_throat)

            final_step = step
            final_residual = residual
            history_row = {
                "step": step,
                "residual": residual,
                "dt": float(dt_diag["dt_global"]),
                "dt_min": float(dt_diag["dt_min"]),
                "dt_mean": float(dt_diag["dt_mean"]),
                "dt_max": float(dt_diag["dt_max"]),
                "pseudo_cfl": float(dt_diag["pseudo_cfl"]),
                "pseudo_cfl_target": float(dt_diag["pseudo_cfl_target"]),
                "pseudo_residual_ratio": float(dt_diag["pseudo_residual_ratio"]),
                "mach_max": diag["mach_max"],
                "mach_throat": diag["mach_throat"],
                "mach_outlet": diag["mach_outlet"],
                "p_throat": diag["p_throat"],
                "mdot_inlet": diag["mdot_inlet"],
                "mdot_throat": diag["mdot_throat"],
                "mdot_outlet": diag["mdot_outlet"],
                "mdot_rel_span": diag["mdot_rel_span"],
                "mdot_balance": diag["mdot_balance"],
                "supersonic_fraction": diag["supersonic_fraction"],
            }
            history.append(history_row)
            plateau_status = evaluate_nozzle_plateau(
                history=history,
                window=plateau_window,
                mdot_rel_tol=plateau_mdot_rel_tol,
                mach_tol=plateau_mach_tol,
                p_rel_tol=plateau_p_rel_tol,
                mdot_balance_tol=plateau_mdot_balance_tol,
            )

            print(
                f"{step:8d}  {residual:10.3e}  {diag['mach_throat']:8.4f}  "
                f"{diag['mach_outlet']:8.4f}  {100.0 * diag['mdot_rel_span']:9.4f}%  "
                f"{100.0 * diag['mdot_balance']:9.4f}%  "
                f"{str(bool(plateau_status['plateau']) if plateau_status['ready'] else False):>7}"
            )

            plateau_ok = (
                (not require_plateau_for_converged)
                or (plateau_status["ready"] and plateau_status["plateau"])
            )
            if residual < tol and plateau_ok:
                converged = True
                print(f"\n  ✅ Residual + plateau converged at step {step}")
                break

    runtime = time.time() - case_start
    dt_diag_final = solver.get_dt_diagnostics()
    if not plateau_status["ready"]:
        plateau_status = evaluate_nozzle_plateau(
            history=history,
            window=plateau_window,
            mdot_rel_tol=plateau_mdot_rel_tol,
            mach_tol=plateau_mach_tol,
            p_rel_tol=plateau_p_rel_tol,
            mdot_balance_tol=plateau_mdot_balance_tol,
        )

    print("\n=== Case Summary ===")
    print(f"Final step:          {final_step}")
    print(f"Residual RMS:        {final_residual:.3e}")
    print(f"Time marching:       {time_marching}")
    if time_marching == "local_pseudo":
        print(f"Pseudo CFL eff:      {dt_diag_final['pseudo_cfl']:.4f}")
        print(f"Pseudo CFL target:   {dt_diag_final['pseudo_cfl_target']:.4f}")
        print(f"Pseudo res ratio:    {dt_diag_final['pseudo_residual_ratio']:.5f}")
    print(f"Mach range:          [{diag['mach_min']:.4f}, {diag['mach_max']:.4f}]")
    print(f"Centerline Mach in:  {diag['mach_inlet']:.4f}")
    print(f"Centerline Mach *:   {diag['mach_throat']:.4f}")
    print(f"Centerline Mach out: {diag['mach_outlet']:.4f}")
    print(f"Centerline |V| in:   {diag['vel_inlet']:.4f}")
    print(f"Centerline |V| *:    {diag['vel_throat']:.4f}")
    print(f"Centerline |V| out:  {diag['vel_outlet']:.4f}")
    print(f"Centerline p in:     {diag['p_inlet']:.5f}")
    print(f"Centerline p *:      {diag['p_throat']:.5f}")
    print(f"Centerline p out:    {diag['p_outlet']:.5f}")
    print(f"Centerline T in:     {diag['t_inlet']:.5f}")
    print(f"Centerline T *:      {diag['t_throat']:.5f}")
    print(f"Centerline T out:    {diag['t_outlet']:.5f}")
    print(f"Mass flow inlet:     {diag['mdot_inlet']:.6e}")
    print(f"Mass flow throat:    {diag['mdot_throat']:.6e}")
    print(f"Mass flow outlet:    {diag['mdot_outlet']:.6e}")
    print(f"Mass-flow span/mean: {100.0 * diag['mdot_rel_span']:.4f}%")
    print(f"Mass-flow balance:   {100.0 * diag['mdot_balance']:.4f}%")
    if plateau_status["ready"]:
        print(f"Plateau status:      {plateau_status['plateau']}")
        print(
            "Plateau spans:       "
            f"Δmdot_rel={plateau_status['mdot_rel_span']:.3e}, "
            f"ΔM*={plateau_status['mach_span']:.3e}, "
            f"Δp*_rel={plateau_status['p_rel_span']:.3e}, "
            f"max(mdot_bal)={plateau_status['mdot_balance_max']:.3e}"
        )
    else:
        print(f"Plateau status:      not enough history (need {plateau_window} reports)")
    print(f"Supersonic fraction: {100.0 * diag['supersonic_fraction']:.3f}%")
    print(f"Converged:           {converged}")
    print(f"Runtime:             {runtime:.2f}s")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tag = f"ma{int(round(ma_init * 1000)):04d}"
        history_dict = {
            "steps": np.array([row["step"] for row in history], dtype=np.int32),
            "residual": np.array([row["residual"] for row in history], dtype=np.float64),
            "dt": np.array([row["dt"] for row in history], dtype=np.float64),
            "dt_min": np.array([row["dt_min"] for row in history], dtype=np.float64),
            "dt_mean": np.array([row["dt_mean"] for row in history], dtype=np.float64),
            "dt_max": np.array([row["dt_max"] for row in history], dtype=np.float64),
            "pseudo_cfl": np.array([row["pseudo_cfl"] for row in history], dtype=np.float64),
            "pseudo_cfl_target": np.array([row["pseudo_cfl_target"] for row in history], dtype=np.float64),
            "pseudo_residual_ratio": np.array([row["pseudo_residual_ratio"] for row in history], dtype=np.float64),
            "mach_max": np.array([row["mach_max"] for row in history], dtype=np.float64),
            "mach_throat": np.array([row["mach_throat"] for row in history], dtype=np.float64),
            "mach_outlet": np.array([row["mach_outlet"] for row in history], dtype=np.float64),
            "p_throat": np.array([row["p_throat"] for row in history], dtype=np.float64),
            "mdot_inlet": np.array([row["mdot_inlet"] for row in history], dtype=np.float64),
            "mdot_throat": np.array([row["mdot_throat"] for row in history], dtype=np.float64),
            "mdot_outlet": np.array([row["mdot_outlet"] for row in history], dtype=np.float64),
            "mdot_rel_span": np.array([row["mdot_rel_span"] for row in history], dtype=np.float64),
            "mdot_balance": np.array([row["mdot_balance"] for row in history], dtype=np.float64),
            "supersonic_fraction": np.array([row["supersonic_fraction"] for row in history], dtype=np.float64),
        }

        plot_nozzle_cell_field(
            x_node,
            y_node,
            diag["mach"],
            field_label="Mach Number",
            title=f"Mach Field — CD Nozzle, Ma_init={ma_init:.3f}",
            cmap="hot_r",
            save_path=os.path.join(output_dir, f"mach_{tag}.png"),
        )
        plot_nozzle_cell_field(
            x_node,
            y_node,
            diag["vel_mag"],
            field_label="Velocity Magnitude",
            title=f"Velocity Field — CD Nozzle, Ma_init={ma_init:.3f}",
            cmap="viridis",
            save_path=os.path.join(output_dir, f"velocity_{tag}.png"),
        )
        plot_nozzle_cell_field(
            x_node,
            y_node,
            diag["p"],
            field_label="Pressure",
            title=f"Pressure Field — CD Nozzle, Ma_init={ma_init:.3f}",
            cmap="RdBu_r",
            save_path=os.path.join(output_dir, f"pressure_{tag}.png"),
        )
        plot_nozzle_cell_field(
            x_node,
            y_node,
            diag["temperature"],
            field_label="Temperature",
            title=f"Temperature Field — CD Nozzle, Ma_init={ma_init:.3f}",
            cmap="plasma",
            save_path=os.path.join(output_dir, f"temperature_{tag}.png"),
        )
        plot_centerline_mach(
            diag["centerline_x"],
            diag["centerline_mach"],
            diag["area_ratio"],
            ma_in=ma_init,
            save_path=os.path.join(output_dir, f"centerline_mach_{tag}.png"),
        )
        plot_velocity_vector_field(
            x_node,
            y_node,
            diag["xc"],
            diag["yc"],
            diag["u"],
            diag["v"],
            ma_in=ma_init,
            save_path=os.path.join(output_dir, f"velocity_vector_{tag}.png"),
            stride_i=max(1, ni // 45),
            stride_j=max(1, nj // 22),
        )
        plot_mass_flow_profile(
            diag["x_face"],
            diag["mdot_profile"],
            throat_x=diag["throat_x"],
            save_path=os.path.join(output_dir, f"mdot_profile_{tag}.png"),
        )
        plot_nozzle_history_diagnostics(
            history_dict,
            save_path=os.path.join(output_dir, f"history_diagnostics_{tag}.png"),
        )

        np.save(os.path.join(output_dir, "summary.npy"), {
            "ma_init": ma_init,
            "solver_model": solver_model,
            "re": re if solver_model == "ns" else np.nan,
            "pr": pr if solver_model == "ns" else np.nan,
            "ni": ni,
            "nj": nj,
            "length": length,
            "h_inlet": h_inlet,
            "h_throat": h_throat,
            "h_exit": h_exit,
            "conv_power": conv_power,
            "div_power": div_power,
            "throat_blend": throat_blend,
            "p0": p0,
            "t0": t0,
            "pb": pb,
            "pb_ratio": pb_ratio,
            "pb_over_p0": pb / max(p0, 1e-12),
            "mach_inlet_bc": mach_inlet_bc_eff,
            "inlet_relax": inlet_relax,
            "outlet_relax": outlet_relax,
            "time_marching": time_marching,
            "pseudo_cfl_start": cfl if pseudo_cfl_start is None else pseudo_cfl_start,
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
            "initial_condition": initial_condition,
            "reconstruction_order": reconstruction_order,
            "plateau_window": plateau_window,
            "plateau_mdot_rel_tol": plateau_mdot_rel_tol,
            "plateau_mach_tol": plateau_mach_tol,
            "plateau_p_rel_tol": plateau_p_rel_tol,
            "plateau_mdot_balance_tol": plateau_mdot_balance_tol,
            "require_plateau_for_converged": require_plateau_for_converged,
            "dt_final": float(dt_diag_final["dt_global"]),
            "dt_min_final": float(dt_diag_final["dt_min"]),
            "dt_mean_final": float(dt_diag_final["dt_mean"]),
            "dt_max_final": float(dt_diag_final["dt_max"]),
            "pseudo_cfl_final": float(dt_diag_final["pseudo_cfl"]),
            "pseudo_cfl_target_final": float(dt_diag_final["pseudo_cfl_target"]),
            "pseudo_residual_ratio_final": float(dt_diag_final["pseudo_residual_ratio"]),
            "final_step": final_step,
            "residual": final_residual,
            "mach_min": diag["mach_min"],
            "mach_max": diag["mach_max"],
            "mach_inlet": diag["mach_inlet"],
            "mach_throat": diag["mach_throat"],
            "mach_outlet": diag["mach_outlet"],
            "vel_inlet": diag["vel_inlet"],
            "vel_throat": diag["vel_throat"],
            "vel_outlet": diag["vel_outlet"],
            "p_inlet": diag["p_inlet"],
            "p_throat": diag["p_throat"],
            "p_outlet": diag["p_outlet"],
            "t_inlet": diag["t_inlet"],
            "t_throat": diag["t_throat"],
            "t_outlet": diag["t_outlet"],
            "mdot_inlet": diag["mdot_inlet"],
            "mdot_throat": diag["mdot_throat"],
            "mdot_outlet": diag["mdot_outlet"],
            "mdot_mean": diag["mdot_mean"],
            "mdot_span": diag["mdot_span"],
            "mdot_rel_span": diag["mdot_rel_span"],
            "mdot_balance": diag["mdot_balance"],
            "plateau_ready": plateau_status["ready"],
            "plateau": plateau_status["plateau"],
            "plateau_mdot_rel_span": plateau_status["mdot_rel_span"],
            "plateau_mach_span": plateau_status["mach_span"],
            "plateau_p_rel_span": plateau_status["p_rel_span"],
            "plateau_mdot_balance_max": plateau_status["mdot_balance_max"],
            "supersonic_fraction": diag["supersonic_fraction"],
            "converged": converged,
            "runtime_seconds": runtime,
        })
        history_payload = build_history_payload(
            solver=solver,
            steps=history_dict["steps"],
            params=history_dict.get("params"),
            extra_series={k: v for k, v in history_dict.items() if k not in {"steps", "params"}},
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
                "vel_mag": diag["vel_mag"],
                "temperature": diag["temperature"],
                "xc": diag["xc"],
                "yc": diag["yc"],
                "centerline_x": diag["centerline_x"],
                "centerline_mach": diag["centerline_mach"],
                "centerline_vel": diag["centerline_vel"],
                "centerline_pressure": diag["centerline_pressure"],
                "centerline_temperature": diag["centerline_temperature"],
                "area_ratio": diag["area_ratio"],
                "x_face": diag["x_face"],
                "mdot_profile": diag["mdot_profile"],
                "throat_face_idx": diag["throat_face_idx"],
            },
        )
        np.save(os.path.join(output_dir, "state_final.npy"), state_payload, allow_pickle=True)

    return {
        "ma_init": ma_init,
        "solver_model": solver_model,
        "re": re if solver_model == "ns" else np.nan,
        "pr": pr if solver_model == "ns" else np.nan,
        "final_step": final_step,
        "residual": final_residual,
        "mach_min": diag["mach_min"],
        "mach_max": diag["mach_max"],
        "mach_inlet": diag["mach_inlet"],
        "mach_throat": diag["mach_throat"],
        "mach_outlet": diag["mach_outlet"],
        "vel_inlet": diag["vel_inlet"],
        "vel_throat": diag["vel_throat"],
        "vel_outlet": diag["vel_outlet"],
        "p_inlet": diag["p_inlet"],
        "p_throat": diag["p_throat"],
        "p_outlet": diag["p_outlet"],
        "t_inlet": diag["t_inlet"],
        "t_throat": diag["t_throat"],
        "t_outlet": diag["t_outlet"],
        "mdot_inlet": diag["mdot_inlet"],
        "mdot_throat": diag["mdot_throat"],
        "mdot_outlet": diag["mdot_outlet"],
        "mdot_rel_span": diag["mdot_rel_span"],
        "mdot_balance": diag["mdot_balance"],
        "plateau_ready": plateau_status["ready"],
        "plateau": plateau_status["plateau"],
        "plateau_mdot_rel_span": plateau_status["mdot_rel_span"],
        "plateau_mach_span": plateau_status["mach_span"],
        "plateau_p_rel_span": plateau_status["p_rel_span"],
        "plateau_mdot_balance_max": plateau_status["mdot_balance_max"],
        "supersonic_fraction": diag["supersonic_fraction"],
        "converged": converged,
        "runtime_seconds": runtime,
        "time_marching": time_marching,
        "mach_inlet_bc": mach_inlet_bc_eff,
        "initial_condition": initial_condition,
        "reconstruction_order": reconstruction_order,
        "pb_ratio": pb / max(p0, 1e-12),
        "output_dir": output_dir,
    }


def run_cd_nozzle_pair(
        ni: int,
        nj: int,
        ma_sub: float,
        ma_sup: float,
        steps: int,
        length: float,
        h_inlet: float,
        h_throat: float,
        h_exit: float | None,
        throat_x: float | None,
        conv_power: float,
        div_power: float,
        throat_blend: float,
        p0: float,
        t0: float,
        pb_sub_ratio: float,
        pb_sup_ratio: float,
        mach_inlet_bc: float,
        inlet_relax: float,
        outlet_relax: float,
        cfl: float,
        solver_model: str,
        re: float,
        pr: float,
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
        adaptive_smoothing_max_eps: float,
        initial_condition: str,
        reconstruction_order: str,
        report: int,
        tol: float,
        plateau_window: int,
        plateau_mdot_rel_tol: float,
        plateau_mach_tol: float,
        plateau_p_rel_tol: float,
        plateau_mdot_balance_tol: float,
        require_plateau_for_converged: bool,
        output_dir: str | None) -> dict:
    """
    依序執行亞音速與超音速兩組 nozzle case
    """
    print("=" * 80)
    print(f"Converging-Diverging Nozzle Validation ({solver_model.upper()})")
    print("=" * 80)

    sub_dir = None if output_dir is None else os.path.join(output_dir, "subsonic_case")
    sup_dir = None if output_dir is None else os.path.join(output_dir, "supersonic_case")
    gamma = 1.4
    p_inf = 1.0 / gamma
    t_inf = p_inf
    mach_inlet_bc_eff = ma_sub if mach_inlet_bc <= 0.0 else mach_inlet_bc
    fac0 = 1.0 + 0.5 * (gamma - 1.0) * mach_inlet_bc_eff**2
    if t0 <= 0.0:
        t0 = t_inf * fac0
    if p0 <= 0.0:
        p0 = p_inf * fac0 ** (gamma / (gamma - 1.0))
    pb_sub = p0 * pb_sub_ratio
    pb_sup = p0 * pb_sup_ratio

    print("\n--- Running subsonic case ---")
    result_sub = run_cd_nozzle_case(
        ni=ni, nj=nj, ma_init=ma_sub, steps=steps, length=length,
        h_inlet=h_inlet, h_throat=h_throat, h_exit=h_exit, throat_x=throat_x,
        conv_power=conv_power, div_power=div_power, throat_blend=throat_blend,
        p0=p0, t0=t0, pb=pb_sub, mach_inlet_bc=mach_inlet_bc_eff,
        inlet_relax=inlet_relax, outlet_relax=outlet_relax,
        cfl=cfl,
        solver_model=solver_model,
        re=re,
        pr=pr,
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
        initial_condition=initial_condition,
        reconstruction_order=reconstruction_order,
        report=report,
        tol=tol,
        plateau_window=plateau_window,
        plateau_mdot_rel_tol=plateau_mdot_rel_tol,
        plateau_mach_tol=plateau_mach_tol,
        plateau_p_rel_tol=plateau_p_rel_tol,
        plateau_mdot_balance_tol=plateau_mdot_balance_tol,
        require_plateau_for_converged=require_plateau_for_converged,
        output_dir=sub_dir)

    print("\n--- Running supersonic case ---")
    result_sup = run_cd_nozzle_case(
        ni=ni, nj=nj, ma_init=ma_sup, steps=steps, length=length,
        h_inlet=h_inlet, h_throat=h_throat, h_exit=h_exit, throat_x=throat_x,
        conv_power=conv_power, div_power=div_power, throat_blend=throat_blend,
        p0=p0, t0=t0, pb=pb_sup, mach_inlet_bc=mach_inlet_bc_eff,
        inlet_relax=inlet_relax, outlet_relax=outlet_relax,
        cfl=cfl,
        solver_model=solver_model,
        re=re,
        pr=pr,
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
        initial_condition=initial_condition,
        reconstruction_order=reconstruction_order,
        report=report,
        tol=tol,
        plateau_window=plateau_window,
        plateau_mdot_rel_tol=plateau_mdot_rel_tol,
        plateau_mach_tol=plateau_mach_tol,
        plateau_p_rel_tol=plateau_p_rel_tol,
        plateau_mdot_balance_tol=plateau_mdot_balance_tol,
        require_plateau_for_converged=require_plateau_for_converged,
        output_dir=sup_dir)

    print("\n=== Comparison Summary ===")
    print(
        f"{'case':>12}  {'pb/p0':>8}  {'res':>10}  {'M_max':>8}  "
        f"{'M_throat':>9}  {'M_out':>8}  {'mdot_bal%':>10}  {'plateau':>8}"
    )
    print(f"{'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}")
    for label, row, ratio in (
            ("subsonic", result_sub, pb_sub_ratio),
            ("supersonic", result_sup, pb_sup_ratio)):
        print(
            f"{label:>12}  {ratio:8.3f}  {row['residual']:10.3e}  {row['mach_max']:8.4f}  "
            f"{row['mach_throat']:9.4f}  {row['mach_outlet']:8.4f}  "
            f"{100.0 * row['mdot_balance']:9.4f}%  {str(row['plateau']):>8}"
        )

    if output_dir:
        np.save(os.path.join(output_dir, "pair_summary.npy"), {
            "subsonic": result_sub,
            "supersonic": result_sup,
            "ni": ni,
            "nj": nj,
            "length": length,
            "h_inlet": h_inlet,
            "h_throat": h_throat,
            "h_exit": h_exit,
            "conv_power": conv_power,
            "div_power": div_power,
            "throat_blend": throat_blend,
            "p0": p0,
            "t0": t0,
            "pb_sub_ratio": pb_sub_ratio,
            "pb_sup_ratio": pb_sup_ratio,
            "pb_sub": pb_sub,
            "pb_sup": pb_sup,
            "mach_inlet_bc": mach_inlet_bc_eff,
            "inlet_relax": inlet_relax,
            "outlet_relax": outlet_relax,
            "steps": steps,
            "cfl": cfl,
            "solver_model": solver_model,
            "re": re if solver_model == "ns" else np.nan,
            "pr": pr if solver_model == "ns" else np.nan,
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
            "initial_condition": initial_condition,
            "reconstruction_order": reconstruction_order,
            "plateau_window": plateau_window,
            "plateau_mdot_rel_tol": plateau_mdot_rel_tol,
            "plateau_mach_tol": plateau_mach_tol,
            "plateau_p_rel_tol": plateau_p_rel_tol,
            "plateau_mdot_balance_tol": plateau_mdot_balance_tol,
            "require_plateau_for_converged": require_plateau_for_converged,
        })

    return {"subsonic": result_sub, "supersonic": result_sup}


def run_cd_nozzle_mesh_preview(
        ni: int,
        nj: int,
        length: float,
        h_inlet: float,
        h_throat: float,
        h_exit: float | None,
        throat_x: float | None,
        conv_power: float,
        div_power: float,
        throat_blend: float,
        output_dir: str | None):
    """
    只生成與輸出內部網格，不執行流場求解
    """
    from fvm_taichi import EulerSolver, generate_reference_nozzle_grid

    print("=== Mesh Preview ===")
    print(f"Grid: ni={ni}, nj={nj}")
    if h_exit is None:
        h_exit = h_inlet
    print(
        f"Nozzle: length={length:.3f}, h_inlet={h_inlet:.3f}, h_throat={h_throat:.3f}, "
        f"h_exit={h_exit:.3f}, conv_power={conv_power:.2f}, div_power={div_power:.2f}, "
        f"throat_blend={throat_blend:.2f}"
    )

    x_node, y_node = generate_reference_nozzle_grid(
        ni=ni,
        nj=nj,
        length=length,
        h_inlet=h_inlet,
        h_throat=h_throat,
        h_exit=h_exit,
        throat_x=throat_x,
        conv_power=conv_power,
        div_power=div_power,
        throat_blend=throat_blend,
        NG=EulerSolver.NG,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_internal_mesh(
            x_node,
            y_node,
            ni=ni,
            nj=nj,
            NG=EulerSolver.NG,
            save_path=os.path.join(output_dir, "mesh_internal.png"),
            line_skip_i=max(1, ni // 80),
            line_skip_j=1,
        )
        np.save(os.path.join(output_dir, "mesh_nodes_interior.npy"), {
            "x_node_interior": x_node[EulerSolver.NG:EulerSolver.NG + ni + 1, EulerSolver.NG:EulerSolver.NG + nj + 1],
            "y_node_interior": y_node[EulerSolver.NG:EulerSolver.NG + ni + 1, EulerSolver.NG:EulerSolver.NG + nj + 1],
            "ni": ni,
            "nj": nj,
            "length": length,
            "h_inlet": h_inlet,
            "h_throat": h_throat,
            "h_exit": h_exit,
            "throat_x": throat_x,
            "conv_power": conv_power,
            "div_power": div_power,
            "throat_blend": throat_blend,
        })


def run_cd_nozzle_physical_baseline(output_dir: str | None) -> dict:
    """
    執行固定的 Euler nozzle 物理基準案例

    Why:
      這組參數專門用來建立「可重現」的質量流率與喉部平台基準，
      驗證目標是 mdot_balance < 1% 且 plateau=True。
    """
    baseline = {
        "ni": 80,
        "nj": 32,
        "ma_init": 0.12,
        "length": 3.0,
        "h_inlet": 1.0,
        "h_throat": 0.70,
        "h_exit": 1.0,
        "conv_power": 2.4,
        "div_power": 3.0,
        "throat_blend": 0.35,
        "pb_ratio": 0.985,
        "cfl": 0.20,
        "time_marching": "local_pseudo",
        "pseudo_cfl_start": 0.05,
        "pseudo_ramp_steps": 800,
        "pseudo_precond_ref_mach": 0.2,
        "pseudo_precond_min_scale": 0.2,
        "residual_smoothing_eps": 0.05,
        "residual_smoothing_passes": 1,
        "adaptive_pseudo": False,
        "initial_condition": "quasi1d",
        "reconstruction_order": "first",
        "steps": 9000,
        "report": 300,
        "tol": 1e-7,
        "plateau_window": 3,
        "plateau_mdot_rel_tol": 1.0e-2,
        "plateau_mach_tol": 3.0e-3,
        "plateau_p_rel_tol": 6.0e-3,
        "plateau_mdot_balance_tol": 1.2e-2,
        "require_plateau_for_converged": True,
    }

    print("=" * 80)
    print("Euler CD Nozzle Physical Baseline")
    print("=" * 80)
    print("Target: mdot_balance < 1% with throat-state plateau")
    print(f"Fixed setup: ni={baseline['ni']}, nj={baseline['nj']}, Ma={baseline['ma_init']:.3f}, "
          f"h_throat={baseline['h_throat']:.3f}, pb_ratio={baseline['pb_ratio']:.3f}, "
          f"recon={baseline['reconstruction_order']}, init={baseline['initial_condition']}")

    if output_dir is None:
        output_dir = "output_cd_nozzle_baseline"

    result = run_cd_nozzle_case(output_dir=output_dir, **baseline)

    print("\n=== Baseline Check ===")
    print(f"mdot_balance: {100.0 * result['mdot_balance']:.3f}%")
    print(f"plateau:      {result['plateau']}")
    print(f"residual:     {result['residual']:.3e}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Converging-diverging nozzle Euler validation")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["mesh", "single", "both", "baseline"],
        help="mesh: internal grid only; single: one flow case; both: sub/supersonic pair; baseline: fixed physical baseline",
    )
    parser.add_argument("--ni", type=int, default=220, help="Streamwise interior cells")
    parser.add_argument("--nj", type=int, default=72, help="Wall-normal interior cells")
    parser.add_argument("--ma", type=float, default=0.3, help="Initialization Mach for single mode")
    parser.add_argument("--ma_sub", type=float, default=0.30, help="Initialization Mach for pair subsonic run")
    parser.add_argument("--ma_sup", type=float, default=1.20, help="Initialization Mach for pair supersonic run")
    parser.add_argument("--steps", type=int, default=2500, help="Maximum time steps")
    parser.add_argument("--length", type=float, default=3.0, help="Nozzle length")
    parser.add_argument("--h_inlet", type=float, default=1.0, help="Inlet/exit channel height")
    parser.add_argument("--h_throat", type=float, default=0.40, help="Throat height")
    parser.add_argument("--h_exit", type=float, default=None, help="Exit height (default: same as h_inlet)")
    parser.add_argument("--throat_x", type=float, default=None, help="Throat position (None -> center)")
    parser.add_argument("--conv_power", type=float, default=2.4, help="Converging-section exponent")
    parser.add_argument("--div_power", type=float, default=3.0, help="Diverging-section exponent")
    parser.add_argument("--throat_blend", type=float, default=0.35, help="Blend factor to shorten throat flatness")
    parser.add_argument("--p0", type=float, default=0.0, help="Inlet total pressure (<=0: auto from initial state)")
    parser.add_argument("--t0", type=float, default=0.0, help="Inlet total temperature (<=0: auto from initial state)")
    parser.add_argument("--pb", type=float, default=0.0, help="Outlet back pressure (single mode, <=0 uses pb_ratio*p0)")
    parser.add_argument("--pb_ratio", type=float, default=0.85, help="Single-mode back-pressure ratio when pb<=0")
    parser.add_argument("--pb_sub_ratio", type=float, default=0.85, help="Subsonic run back-pressure ratio pb/p0")
    parser.add_argument("--pb_sup_ratio", type=float, default=0.45, help="Supersonic run back-pressure ratio pb/p0")
    parser.add_argument("--mach_inlet_bc", type=float, default=0.0, help="Inlet BC reference Mach (<=0 uses ma/ma_sub)")
    parser.add_argument("--inlet_relax", type=float, default=0.35, help="Characteristic inlet relaxation")
    parser.add_argument("--outlet_relax", type=float, default=0.35, help="Characteristic outlet relaxation")
    parser.add_argument("--cfl", type=float, default=0.35, help="Explicit CFL")
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "ns"],
        help="Solver model: euler (slip wall) or ns (no-slip wall)",
    )
    parser.add_argument("--re", type=float, default=1.0e4, help="Reynolds number for NS mode")
    parser.add_argument("--pr", type=float, default=0.72, help="Prandtl number for NS mode")
    parser.add_argument(
        "--time_marching",
        type=str,
        default="local_pseudo",
        choices=["global", "local_pseudo"],
        help="Time marching mode",
    )
    parser.add_argument("--pseudo_cfl_start", type=float, default=0.08, help="Initial pseudo CFL (<= CFL)")
    parser.add_argument("--pseudo_ramp_steps", type=int, default=1200, help="Pseudo CFL ramp steps")
    parser.add_argument("--pseudo_precond_ref_mach", type=float, default=0.30, help="Pseudo preconditioning reference Mach")
    parser.add_argument("--pseudo_precond_min_scale", type=float, default=0.25, help="Pseudo preconditioning minimum acoustic scale")
    parser.add_argument("--residual_smoothing_eps", type=float, default=0.08, help="Residual smoothing epsilon")
    parser.add_argument("--residual_smoothing_passes", type=int, default=2, help="Residual smoothing passes")
    parser.add_argument("--adaptive_pseudo", action=argparse.BooleanOptionalAction, default=False, help="Enable adaptive pseudo CFL/smoothing")
    parser.add_argument("--adaptive_interval", type=int, default=20, help="Adaptive pseudo update interval")
    parser.add_argument("--adaptive_cfl_growth", type=float, default=1.03, help="Adaptive pseudo CFL growth factor")
    parser.add_argument("--adaptive_cfl_shrink", type=float, default=0.70, help="Adaptive pseudo CFL shrink factor")
    parser.add_argument("--adaptive_target_ratio", type=float, default=0.995, help="Adaptive pseudo target residual ratio")
    parser.add_argument("--adaptive_fail_ratio", type=float, default=1.010, help="Adaptive pseudo fail residual ratio")
    parser.add_argument("--adaptive_smoothing_max_eps", type=float, default=0.24, help="Adaptive maximum smoothing epsilon")
    parser.add_argument(
        "--initial_condition",
        type=str,
        default="quasi1d",
        choices=["uniform", "quasi1d"],
        help="Initial condition type",
    )
    parser.add_argument(
        "--reconstruction_order",
        type=str,
        default="second",
        choices=["first", "second"],
        help="Spatial reconstruction order",
    )
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Residual tolerance")
    parser.add_argument("--plateau_window", type=int, default=6, help="Report-window size for throat/mdot plateau check")
    parser.add_argument("--plateau_mdot_rel_tol", type=float, default=2e-3, help="Plateau tolerance of throat mdot relative span")
    parser.add_argument("--plateau_mach_tol", type=float, default=2e-3, help="Plateau tolerance of throat Mach span")
    parser.add_argument("--plateau_p_rel_tol", type=float, default=2e-3, help="Plateau tolerance of throat pressure relative span")
    parser.add_argument("--plateau_mdot_balance_tol", type=float, default=5e-3, help="Plateau tolerance of mdot inlet/throat/outlet mismatch")
    parser.add_argument(
        "--require_plateau_for_converged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require plateau criteria in addition to residual tolerance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_cd_nozzle",
        help="Output directory (empty string disables)",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        output_dir = args.output if args.output else None
        if args.mode == "mesh":
            run_cd_nozzle_mesh_preview(
                ni=args.ni,
                nj=args.nj,
                length=args.length,
                h_inlet=args.h_inlet,
                h_throat=args.h_throat,
                h_exit=args.h_exit,
                throat_x=args.throat_x,
                conv_power=args.conv_power,
                div_power=args.div_power,
                throat_blend=args.throat_blend,
                output_dir=output_dir,
            )
        elif args.mode == "single":
            run_cd_nozzle_case(
                ni=args.ni,
                nj=args.nj,
                ma_init=args.ma,
                steps=args.steps,
                length=args.length,
                h_inlet=args.h_inlet,
                h_throat=args.h_throat,
                h_exit=args.h_exit,
                throat_x=args.throat_x,
                conv_power=args.conv_power,
                div_power=args.div_power,
                throat_blend=args.throat_blend,
                p0=args.p0,
                t0=args.t0,
                pb=args.pb,
                pb_ratio=args.pb_ratio,
                mach_inlet_bc=args.mach_inlet_bc,
                inlet_relax=args.inlet_relax,
                outlet_relax=args.outlet_relax,
                cfl=args.cfl,
                solver_model=args.solver,
                re=args.re,
                pr=args.pr,
                time_marching=args.time_marching,
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
                initial_condition=args.initial_condition,
                reconstruction_order=args.reconstruction_order,
                report=args.report,
                tol=args.tol,
                plateau_window=args.plateau_window,
                plateau_mdot_rel_tol=args.plateau_mdot_rel_tol,
                plateau_mach_tol=args.plateau_mach_tol,
                plateau_p_rel_tol=args.plateau_p_rel_tol,
                plateau_mdot_balance_tol=args.plateau_mdot_balance_tol,
                require_plateau_for_converged=args.require_plateau_for_converged,
                output_dir=output_dir,
            )
        elif args.mode == "baseline":
            run_cd_nozzle_physical_baseline(output_dir=output_dir)
        else:
            run_cd_nozzle_pair(
                ni=args.ni,
                nj=args.nj,
                ma_sub=args.ma_sub,
                ma_sup=args.ma_sup,
                steps=args.steps,
                length=args.length,
                h_inlet=args.h_inlet,
                h_throat=args.h_throat,
                h_exit=args.h_exit,
                throat_x=args.throat_x,
                conv_power=args.conv_power,
                div_power=args.div_power,
                throat_blend=args.throat_blend,
                p0=args.p0,
                t0=args.t0,
                pb_sub_ratio=args.pb_sub_ratio,
                pb_sup_ratio=args.pb_sup_ratio,
                mach_inlet_bc=args.mach_inlet_bc,
                inlet_relax=args.inlet_relax,
                outlet_relax=args.outlet_relax,
                cfl=args.cfl,
                solver_model=args.solver,
                re=args.re,
                pr=args.pr,
                time_marching=args.time_marching,
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
                initial_condition=args.initial_condition,
                reconstruction_order=args.reconstruction_order,
                report=args.report,
                tol=args.tol,
                plateau_window=args.plateau_window,
                plateau_mdot_rel_tol=args.plateau_mdot_rel_tol,
                plateau_mach_tol=args.plateau_mach_tol,
                plateau_p_rel_tol=args.plateau_p_rel_tol,
                plateau_mdot_balance_tol=args.plateau_mdot_balance_tol,
                require_plateau_for_converged=args.require_plateau_for_converged,
                output_dir=output_dir,
            )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n❌ CD nozzle case failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
