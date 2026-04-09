"""
Laminar Poiseuille Flow — Incompressible Navier-Stokes Benchmark
================================================================

What:
- 驗證 body-force 驅動的週期通道層流 Poiseuille flow
- 提供 incompressible FVM 的正式 CaseRunner builder

Why:
- 這個案例是不可壓縮 projection solver 最直接的驗證之一：
  body force、no-slip wall、拋物線解析剖面三者必須同時成立
- 既然已遷移到 benchmark-driven workflow，就不再保留舊的 compressible fallback 路徑

When:
- 作為 incompressible FVM 的基準案例
- 作為 batch regression 與 CLI benchmark 的最小驗證
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import taichi as ti

from cfd_taichi import (
    BoundaryConditionDescriptor,
    CartesianGrid2D,
    CaseRunner,
)


def poiseuille_exact_profile(
    ny: int,
    nu: float,
    body_force_x: float,
    height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Poiseuille 解析解（cell center）。

    What:
    - 回傳 `u(y) = fx / (2 nu) * y * (H - y)` 的 cell-centered 剖面

    Why:
    - profile 誤差是 Poiseuille benchmark 最直接的物理解釋
    """
    y = (np.arange(ny, dtype=np.float32) + 0.5) * (height / ny)
    u = body_force_x * y * (height - y) / (2.0 * nu)
    return y, u


def _validate_poiseuille_inputs(
    *,
    ni: int,
    nj: int,
    re: float,
    u_max: float,
    report: int,
    tol: float,
    init_mode: str,
):
    if init_mode not in ("rest", "parabolic"):
        raise ValueError(
            f"Unknown init_mode={init_mode}; choose 'rest' or 'parabolic'"
        )
    if ni <= 0 or nj <= 0:
        raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
    if re <= 0.0:
        raise ValueError(f"re must be positive, got {re}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if u_max <= 0.0:
        raise ValueError(f"u_max must be positive, got {u_max}")


def _build_poiseuille_setup(
    *,
    nj: int,
    re: float,
    u_max: float,
    height: float = 1.0,
) -> dict[str, Any]:
    """
    建立 Poiseuille benchmark 的解析與驅動參數。

    Why:
    - exact profile 與 body force 是 benchmark 的核心物理解釋，應集中管理
    """
    nu = float(u_max * height / re)
    body_force_x = float(8.0 * nu * u_max / (height * height))
    y, u_exact = poiseuille_exact_profile(
        nj,
        nu=nu,
        body_force_x=body_force_x,
        height=height,
    )
    return {
        "height": float(height),
        "nu": nu,
        "body_force_x": body_force_x,
        "y": y,
        "u_exact": u_exact,
    }


def _initialize_poiseuille_case(
    _runner: CaseRunner,
    solver,
    _bc_handle,
    *,
    init_mode: str,
    u_exact: np.ndarray,
    p0: float,
):
    """
    Poiseuille 的正式初始化流程。

    What:
    - `rest`：靜止起算，測試 body-force 驅動啟動
    - `parabolic`：以解析剖面起算，測試 steady solution 能否保持

    Why:
    - benchmark builder 應把初始化語義固定下來，而不是留在主程式手寫
    """
    if init_mode == "rest":
        solver.init_uniform(u=0.0, v=0.0, p=p0)
        return

    w = np.zeros((solver.ni, solver.nj, 3), dtype=np.float32)
    w[:, :, 0] = u_exact[None, :]
    w[:, :, 1] = 0.0
    w[:, :, 2] = p0
    solver.init_from_primitive_numpy(w)


def _poiseuille_diagnostics_hook(
    _runner: CaseRunner,
    solver,
    diagnostics: dict[str, Any],
    *,
    u_exact: np.ndarray,
    body_force_x: float,
) -> dict[str, Any]:
    """
    派生 Poiseuille 專屬 diagnostics。

    Why:
    - solver 本體只輸出通用場量；解析解誤差屬於 benchmark contract，應掛在 workflow 層
    """
    u_field, _, _ = solver.get_primitive()
    u_profile = np.mean(u_field, axis=0)
    err = u_profile - u_exact
    center_idx = int(u_profile.shape[0] // 2)
    return {
        "l2_profile_error": float(np.sqrt(np.mean(err**2))),
        "linf_profile_error": float(np.max(np.abs(err))),
        "centerline_u": float(u_profile[center_idx]),
        "mean_u": float(np.mean(u_profile)),
        "body_force_x": float(body_force_x),
        "u_max": float(max(diagnostics.get("u_max", 0.0), np.max(np.abs(u_profile)))),
    }


def build_poiseuille_runner(
    *,
    ni: int = 128,
    nj: int = 64,
    re: float = 20.0,
    u_max: float = 0.05,
    init_mode: str = "parabolic",
) -> tuple[CaseRunner, dict[str, Any]]:
    """
    建立 Poiseuille 的 CaseRunner。

    What:
    - 使用 incompressible Navier-Stokes solver 與 Cartesian 通道設定

    Why:
    - 這是目前 toolkit 中缺的第三條正式 solver workflow：
      LBM / compressible FVM / incompressible FVM
    """
    _validate_poiseuille_inputs(
        ni=ni,
        nj=nj,
        re=re,
        u_max=u_max,
        report=1,
        tol=1e-12,
        init_mode=init_mode,
    )

    setup = _build_poiseuille_setup(nj=nj, re=re, u_max=u_max, height=1.0)
    p0 = 0.0

    runner = CaseRunner(
        name="poiseuille_flow_ns",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=ni, nj=nj, dx=1.0 / ni, dy=1.0 / nj),
        solver_kwargs={
            "re": re,
            "u_ref": u_max,
            "length_scale": setup["height"],
            "rho_ref": 1.0,
            "cfl": 0.2,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip("bottom", u_wall=0.0, v_wall=0.0),
            BoundaryConditionDescriptor.no_slip("top", u_wall=0.0, v_wall=0.0),
            BoundaryConditionDescriptor.body_force(
                fx=setup["body_force_x"],
                fy=0.0,
            ),
        ],
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_poiseuille_case(
            runner_obj,
            solver_obj,
            bc_handle,
            init_mode=init_mode,
            u_exact=setup["u_exact"],
            p0=p0,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _poiseuille_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            u_exact=setup["u_exact"],
            body_force_x=setup["body_force_x"],
        ),
    )
    setup["p0"] = p0
    return runner, setup


def run_poiseuille(
    ni: int = 128,
    nj: int = 64,
    re: float = 20.0,
    u_max: float = 0.05,
    steps: int = 8000,
    report: int = 200,
    tol: float = 1e-4,
    init_mode: str = "parabolic",
):
    """
    執行 laminar Poiseuille flow 驗證。

    What:
    - 只保留正式 incompressible CaseRunner 路徑
    """
    _validate_poiseuille_inputs(
        ni=ni,
        nj=nj,
        re=re,
        u_max=u_max,
        report=report,
        tol=tol,
        init_mode=init_mode,
    )

    runner, setup = build_poiseuille_runner(
        ni=ni,
        nj=nj,
        re=re,
        u_max=u_max,
        init_mode=init_mode,
    )
    solver = runner.build_solver()
    runner.configure()
    runner.initialize()

    transport = solver.get_transport_coefficients()

    print("=" * 76)
    print("  Laminar Poiseuille Flow — IncompressibleNavierStokesSolver")
    print("=" * 76)
    print(f"  Grid:       {ni} x {nj}")
    print("  Geometry:   cartesian")
    print(f"  Re:         {re:.1f}")
    print(f"  Target umax:{u_max:.5f}")
    print(f"  Body force: fx={setup['body_force_x']:.5e}")
    print(f"  Init:       {init_mode}")
    print(f"  Steps:      {steps}, report={report}, tol={tol:.1e}")
    print(f"  Transport:  {transport}")

    print("\n[1] Running Poiseuille benchmark...")
    print(
        f"  {'step':>6}  {'dt':>10}  {'L2(u)':>10}  {'Linf(u)':>10}  "
        f"{'u_mid':>10}  {'div':>10}  {'t(s)':>7}"
    )
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}")

    t0 = time.time()
    converged = False
    final_diag = runner.prepare_observables(reset_baseline=False) or {}
    final_step = 0

    for step in range(1, steps + 1):
        dt = runner.step_once()

        if step % report == 0 or step == 1:
            ti.sync()
            diagnostics = runner.prepare_observables(reset_baseline=False) or {}
            elapsed = time.time() - t0
            final_diag = diagnostics
            final_step = step

            print(
                f"  {step:>6}  {float(dt):>10.4e}  "
                f"{float(diagnostics['l2_profile_error']):>10.3e}  "
                f"{float(diagnostics['linf_profile_error']):>10.3e}  "
                f"{float(diagnostics['centerline_u']):>10.5f}  "
                f"{float(diagnostics['div_linf']):>10.3e}  "
                f"{elapsed:>7.2f}s"
            )

            if float(diagnostics["l2_profile_error"]) < tol:
                converged = True
                print(f"\n  ✅ Poiseuille profile converged at step {step}")
                break

    total_time = time.time() - t0
    final_diag = runner.prepare_observables(reset_baseline=False) or final_diag
    u_profile = np.mean(solver.get_primitive()[0], axis=0)

    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")
    print(f"  Last reported step: {final_step}")
    print(f"  L2 error:           {float(final_diag['l2_profile_error']):.3e}")
    print(f"  Linf error:         {float(final_diag['linf_profile_error']):.3e}")
    print("  Status:             " + ("converged" if converged else "not converged within step budget"))
    print("=" * 76)

    return {
        "y": setup["y"],
        "u_exact": setup["u_exact"],
        "u_profile": u_profile,
        "l2_error": float(final_diag["l2_profile_error"]),
        "linf_error": float(final_diag["linf_profile_error"]),
        "step": int(final_step),
        "converged": bool(converged),
        "body_force_x": float(setup["body_force_x"]),
        "solver_family": "fvm",
        "regime": "incompressible",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Laminar Poiseuille flow with NavierStokesSolver"
    )
    parser.add_argument("--ni", type=int, default=128, help="X resolution")
    parser.add_argument("--nj", type=int, default=64, help="Y resolution")
    parser.add_argument("--re", type=float, default=20.0, help="Reynolds number")
    parser.add_argument(
        "--u_max",
        type=float,
        default=0.05,
        help="Target centerline velocity",
    )
    parser.add_argument("--steps", type=int, default=8000, help="Maximum steps")
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=1e-4, help="L2 profile tolerance")
    parser.add_argument(
        "--init",
        type=str,
        default="parabolic",
        choices=("rest", "parabolic"),
        help="Initial condition mode",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        run_poiseuille(
            ni=args.ni,
            nj=args.nj,
            re=args.re,
            u_max=args.u_max,
            steps=args.steps,
            report=args.report,
            tol=args.tol,
            init_mode=args.init,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ Poiseuille case failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
