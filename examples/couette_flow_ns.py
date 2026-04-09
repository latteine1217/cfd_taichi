"""
Laminar Couette Flow — Incompressible Navier-Stokes Benchmark
=============================================================

What:
- 驗證兩平板間的層流 Couette flow
- 提供 incompressible FVM 的正式 CaseRunner builder

Why:
- 這是 moving-wall、黏性剪應力與線性解析剖面最直接的組合驗證
- 既然已遷移到 benchmark-driven workflow，就不再保留舊的 compressible fallback 路徑

When:
- 作為 incompressible FVM 的 wall-driven benchmark
- 作為 registry / matrix / CLI 的固定驗證案例
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


def couette_exact_profile(
    ny: int,
    u_bottom: float,
    u_top: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Couette 解析解（cell center）。

    What:
    - 回傳兩壁之間的線性速度剖面
    """
    y = (np.arange(ny, dtype=np.float32) + 0.5) / ny
    u = u_bottom + (u_top - u_bottom) * y
    return y, u


def _validate_couette_inputs(
    *,
    ni: int,
    nj: int,
    re: float,
    u_top: float,
    report: int,
    tol: float,
    init_mode: str,
):
    if init_mode not in ("rest", "linear"):
        raise ValueError(f"Unknown init_mode={init_mode}; choose 'rest' or 'linear'")
    if ni <= 0 or nj <= 0:
        raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
    if re <= 0.0:
        raise ValueError(f"re must be positive, got {re}")
    if report <= 0:
        raise ValueError(f"report must be positive, got {report}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")


def _build_couette_setup(
    *,
    nj: int,
    u_top: float,
    u_bottom: float = 0.0,
) -> dict[str, Any]:
    y, u_exact = couette_exact_profile(
        nj,
        u_bottom=u_bottom,
        u_top=u_top,
    )
    return {
        "u_bottom": float(u_bottom),
        "u_top": float(u_top),
        "v_wall": 0.0,
        "y": y,
        "u_exact": u_exact,
    }


def _initialize_couette_case(
    _runner: CaseRunner,
    solver,
    _bc_handle,
    *,
    init_mode: str,
    u_exact: np.ndarray,
    p0: float,
):
    """
    Couette 的正式初始化流程。
    """
    if init_mode == "rest":
        solver.init_uniform(u=0.0, v=0.0, p=p0)
        return

    w = np.zeros((solver.ni, solver.nj, 3), dtype=np.float32)
    w[:, :, 0] = u_exact[None, :]
    w[:, :, 1] = 0.0
    w[:, :, 2] = p0
    solver.init_from_primitive_numpy(w)


def _couette_diagnostics_hook(
    _runner: CaseRunner,
    solver,
    diagnostics: dict[str, Any],
    *,
    u_exact: np.ndarray,
    u_top: float,
) -> dict[str, Any]:
    """
    派生 Couette 專屬 diagnostics。
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
        "target_wall_speed": float(u_top),
        "u_max": float(max(diagnostics.get("u_max", 0.0), np.max(np.abs(u_profile)))),
    }


def build_couette_runner(
    *,
    ni: int = 128,
    nj: int = 64,
    re: float = 100.0,
    u_top: float = 0.1,
    init_mode: str = "linear",
) -> tuple[CaseRunner, dict[str, Any]]:
    """
    建立 Couette 的 CaseRunner。

    What:
    - 使用 incompressible Navier-Stokes solver 與 Cartesian moving-wall 通道設定
    """
    _validate_couette_inputs(
        ni=ni,
        nj=nj,
        re=re,
        u_top=u_top,
        report=1,
        tol=1e-12,
        init_mode=init_mode,
    )

    setup = _build_couette_setup(nj=nj, u_top=u_top)
    p0 = 0.0

    runner = CaseRunner(
        name="couette_flow_ns",
        method="fvm",
        equation="navier_stokes",
        regime="incompressible",
        grid=CartesianGrid2D(ni=ni, nj=nj, dx=1.0 / ni, dy=1.0 / nj),
        solver_kwargs={
            "re": re,
            "u_ref": max(abs(u_top), 1e-6),
            "length_scale": 1.0,
            "rho_ref": 1.0,
            "cfl": 0.2,
        },
        boundary_conditions=[
            BoundaryConditionDescriptor.periodic("x"),
            BoundaryConditionDescriptor.no_slip(
                "bottom",
                u_wall=setup["u_bottom"],
                v_wall=setup["v_wall"],
            ),
            BoundaryConditionDescriptor.no_slip(
                "top",
                u_wall=setup["u_top"],
                v_wall=setup["v_wall"],
            ),
        ],
        initializer=lambda runner_obj, solver_obj, bc_handle: _initialize_couette_case(
            runner_obj,
            solver_obj,
            bc_handle,
            init_mode=init_mode,
            u_exact=setup["u_exact"],
            p0=p0,
        ),
        diagnostics_hook=lambda runner_obj, solver_obj, diagnostics: _couette_diagnostics_hook(
            runner_obj,
            solver_obj,
            diagnostics,
            u_exact=setup["u_exact"],
            u_top=setup["u_top"],
        ),
    )
    setup["p0"] = p0
    return runner, setup


def run_couette(
    ni: int = 128,
    nj: int = 64,
    re: float = 100.0,
    u_top: float = 0.1,
    steps: int = 8000,
    report: int = 200,
    tol: float = 1e-4,
    init_mode: str = "rest",
):
    """
    執行 laminar Couette flow 驗證。
    """
    _validate_couette_inputs(
        ni=ni,
        nj=nj,
        re=re,
        u_top=u_top,
        report=report,
        tol=tol,
        init_mode=init_mode,
    )

    runner, setup = build_couette_runner(
        ni=ni,
        nj=nj,
        re=re,
        u_top=u_top,
        init_mode=init_mode,
    )
    solver = runner.build_solver()
    runner.configure()
    runner.initialize()

    transport = solver.get_transport_coefficients()

    print("=" * 72)
    print("  Laminar Couette Flow — IncompressibleNavierStokesSolver")
    print("=" * 72)
    print(f"  Grid:      {ni} x {nj}")
    print("  Geometry:  cartesian")
    print(f"  Re:        {re:.1f}")
    print(f"  Walls:     u_bottom={setup['u_bottom']:.4f}, u_top={setup['u_top']:.4f}")
    print(f"  Init:      {init_mode}")
    print(f"  Steps:     {steps}, report={report}, tol={tol:.1e}")
    print(f"  Transport: {transport}")

    print("\n[1] Running Couette benchmark...")
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
                print(f"\n  ✅ Couette profile converged at step {step}")
                break

    total_time = time.time() - t0
    final_diag = runner.prepare_observables(reset_baseline=False) or final_diag
    u_profile = np.mean(solver.get_primitive()[0], axis=0)

    print(f"\n--- Simulation completed in {total_time:.2f} seconds ---")
    print(f"  Last reported step: {final_step}")
    print(f"  L2 error:           {float(final_diag['l2_profile_error']):.3e}")
    print(f"  Linf error:         {float(final_diag['linf_profile_error']):.3e}")
    print("  Status:             " + ("converged" if converged else "not converged within step budget"))
    print("=" * 72)

    return {
        "y": setup["y"],
        "u_exact": setup["u_exact"],
        "u_profile": u_profile,
        "l2_error": float(final_diag["l2_profile_error"]),
        "linf_error": float(final_diag["linf_profile_error"]),
        "step": int(final_step),
        "converged": bool(converged),
        "solver_family": "fvm",
        "regime": "incompressible",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Laminar Couette flow with NavierStokesSolver"
    )
    parser.add_argument("--ni", type=int, default=128, help="X resolution")
    parser.add_argument("--nj", type=int, default=64, help="Y resolution")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--u_top", type=float, default=0.1, help="Top wall velocity")
    parser.add_argument("--steps", type=int, default=8000, help="Maximum steps")
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=1e-4, help="L2 profile tolerance")
    parser.add_argument(
        "--init",
        type=str,
        default="rest",
        choices=("rest", "linear"),
        help="Initial condition mode",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        run_couette(
            ni=args.ni,
            nj=args.nj,
            re=args.re,
            u_top=args.u_top,
            steps=args.steps,
            report=args.report,
            tol=args.tol,
            init_mode=args.init,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ Couette case failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
