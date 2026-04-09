"""
Rayleigh-Taylor CH Solver Parameter Sweep
=========================================
"""

import argparse
import csv
import itertools
import json
import os
import time
from typing import Dict, List

import numpy as np
import taichi as ti

from lbm_taichi.core import CHLBMSolver

try:
    from .rayleigh_taylor_ch import _build_initial_phi
except ImportError:
    from rayleigh_taylor_ch import _build_initial_phi


def _parse_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _interface_amp(phi: np.ndarray) -> float:
    yi = np.argmin(np.abs(phi), axis=1)
    return float(yi.max() - yi.min())


def _fronts(phi: np.ndarray) -> tuple[float, float]:
    neg = np.where(phi < -0.5)
    pos = np.where(phi > 0.5)
    bubble = float(np.max(neg[1])) if neg[1].size > 0 else np.nan
    spike = float(np.min(pos[1])) if pos[1].size > 0 else np.nan
    return bubble, spike


def _score_run(
    phi0: np.ndarray,
    phi_mid: np.ndarray,
    phi_end: np.ndarray,
    max_u: float,
    phi_drift: float,
) -> float:
    amp_growth = _interface_amp(phi_end) - _interface_amp(phi0)
    amp_late = _interface_amp(phi_end) - _interface_amp(phi_mid)
    # 偏好：有成長、速度適中、守恆較好
    return (
        amp_growth
        + 0.5 * amp_late
        - 8.0 * max(0.0, max_u - 0.12)
        - 20.0 * phi_drift
    )


def run_sweep(args):
    nx = int(args.aspect * args.res)
    ny = args.res
    thickness = max(1.0, args.thickness * ny)
    amp = max(0.0, args.perturb_amp * ny)
    g = -abs(args.g)

    phi0 = _build_initial_phi(nx, ny, args.interface, thickness, amp, args.perturb_mode)

    combos = list(
        itertools.product(
            _parse_list(args.tau_list),
            _parse_list(args.mobility_list),
            _parse_list(args.kappa_list),
            _parse_list(args.low_mach_list),
            _parse_list(args.compressibility_list),
        )
    )

    print("=== CH-RT Sweep ===")
    print(f"Grid: {nx}x{ny}")
    print(f"Total combinations: {len(combos)}")
    print(
        "| id | tau | M | kappa | lm_corr | comp | stable | objective | max_u | phi_drift | amp_grow | front_disp | score |"
    )
    print(
        "| -- | --- | --- | ----- | ------- | ---- | ------ | --------- | ----- | --------- | -------- | ---------- | ----- |"
    )

    os.makedirs(args.output, exist_ok=True)
    summary_csv = os.path.join(args.output, "sweep_summary.csv")
    best_json = os.path.join(args.output, "best_params.json")

    results: List[Dict] = []
    start = time.time()

    for idx, (tau, mobility, kappa, lm_corr, comp) in enumerate(combos, start=1):
        solver = CHLBMSolver(
            nx=nx,
            ny=ny,
            tau=tau,
            mobility=mobility,
            a=args.a,
            kappa=kappa,
            rho_heavy=args.rho_heavy,
            rho_light=args.rho_light,
            gravity=(0.0, g),
            u_cap=args.u_cap,
            force_cap=args.force_cap,
            boundary_y=args.boundary_y,
            low_mach_corr=lm_corr,
            compressibility=comp,
        )
        solver.set_initial_phi(phi0)

        max_u = 0.0
        stable = True
        objective = True
        phi_mid = None
        mid_step = max(1, args.steps // 2)
        for it in range(1, args.steps + 1):
            solver.step()
            fields = solver.get_fields()
            u_now = float(np.linalg.norm(fields["u"], axis=2).max())
            max_u = max(max_u, u_now)
            if it == mid_step:
                phi_mid = fields["phi"].copy()
            if not np.isfinite(u_now) or u_now > args.u_fail:
                stable = False
                break
            if not np.isfinite(fields["phi"]).all():
                stable = False
                break

        phi_end = solver.get_fields()["phi"]
        if phi_mid is None:
            phi_mid = phi_end.copy()
        phi_drift = float(abs(np.mean(phi_end) - np.mean(phi0)))
        amp_grow = float(_interface_amp(phi_end) - _interface_amp(phi0))
        amp_late = float(_interface_amp(phi_end) - _interface_amp(phi_mid))
        b0, s0 = _fronts(phi0)
        bm, sm = _fronts(phi_mid)
        b1, s1 = _fronts(phi_end)
        disp_candidates = []
        late_disp_candidates = []
        if np.isfinite(b0) and np.isfinite(b1):
            disp_candidates.append(abs(b1 - b0))
        if np.isfinite(bm) and np.isfinite(b1):
            late_disp_candidates.append(abs(b1 - bm))
        if np.isfinite(s0) and np.isfinite(s1):
            disp_candidates.append(abs(s1 - s0))
        if np.isfinite(sm) and np.isfinite(s1):
            late_disp_candidates.append(abs(s1 - sm))
        front_disp = float(max(disp_candidates)) if disp_candidates else 0.0
        front_disp_late = (
            float(max(late_disp_candidates)) if late_disp_candidates else 0.0
        )
        score = _score_run(phi0, phi_mid, phi_end, max_u, phi_drift)
        if not stable:
            score -= 50.0

        if phi_drift > args.phi_drift_fail:
            objective = False
            score -= 10.0
        if amp_grow < args.min_amp_growth:
            objective = False
            score -= 5.0
        if front_disp < args.min_front_disp:
            objective = False
            score -= 5.0
        if amp_late < args.min_amp_growth_late:
            objective = False
            score -= 5.0
        if front_disp_late < args.min_front_disp_late:
            objective = False
            score -= 5.0

        rec = {
            "id": idx,
            "tau": tau,
            "mobility": mobility,
            "kappa": kappa,
            "low_mach_corr": lm_corr,
            "compressibility": comp,
            "stable": stable,
            "objective": objective,
            "max_u": max_u,
            "phi_drift": phi_drift,
            "amp_growth": amp_grow,
            "amp_growth_late": amp_late,
            "front_disp": front_disp,
            "front_disp_late": front_disp_late,
            "score": score,
        }
        results.append(rec)

        print(
            f"| {idx} | {tau:.3f} | {mobility:.4f} | {kappa:.4f} | {lm_corr:.2f} | "
            f"{comp:.2f} | {stable} | {objective} | {max_u:.3e} | {phi_drift:.3e} | "
            f"{amp_grow:.2f} | {front_disp:.2f} | {score:.2f} |"
        )

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results_sorted[0] if results_sorted else {}
    stable_results = [r for r in results_sorted if r["stable"]]
    objective_results = [r for r in stable_results if r["objective"]]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results_sorted)

    payload = {
        "best": best,
        "stable_count": len(stable_results),
        "objective_count": len(objective_results),
        "total_count": len(results_sorted),
        "top5": results_sorted[:5],
        "elapsed_sec": time.time() - start,
    }
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Sweep Summary ===")
    print(f"stable_count: {len(stable_results)} / {len(results_sorted)}")
    print(f"objective_count: {len(objective_results)} / {len(results_sorted)}")
    if best:
        print(
            f"best: tau={best['tau']}, M={best['mobility']}, "
            f"kappa={best['kappa']}, lm_corr={best['low_mach_corr']}, "
            f"comp={best['compressibility']}, score={best['score']:.2f}"
        )
    print(f"CSV: {summary_csv}")
    print(f"JSON: {best_json}")


def main():
    parser = argparse.ArgumentParser(description="Sweep CH-RT parameter stability window")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--aspect", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--u_fail", type=float, default=0.25, help="Instability threshold")
    parser.add_argument(
        "--phi_drift_fail",
        type=float,
        default=0.08,
        help="Reject if mean(phi) drift exceeds this",
    )
    parser.add_argument(
        "--min_amp_growth",
        type=float,
        default=2.0,
        help="Minimum interface amplitude growth to count as RT growth",
    )
    parser.add_argument(
        "--min_front_disp",
        type=float,
        default=2.0,
        help="Minimum bubble/spike front displacement to count as RT growth",
    )
    parser.add_argument(
        "--min_amp_growth_late",
        type=float,
        default=0.8,
        help="Minimum amplitude growth in the second half",
    )
    parser.add_argument(
        "--min_front_disp_late",
        type=float,
        default=0.8,
        help="Minimum front displacement in the second half",
    )

    parser.add_argument("--tau_list", type=str, default="0.8,0.9,1.0")
    parser.add_argument("--mobility_list", type=str, default="0.0015,0.002,0.003")
    parser.add_argument("--kappa_list", type=str, default="0.02,0.04,0.06")
    parser.add_argument("--low_mach_list", type=str, default="0.0,0.1,0.2")
    parser.add_argument("--compressibility_list", type=str, default="0.0,0.2")

    parser.add_argument("--a", type=float, default=0.04)
    parser.add_argument("--g", type=float, default=1e-5)
    parser.add_argument("--rho_heavy", type=float, default=1.2)
    parser.add_argument("--rho_light", type=float, default=0.2)
    parser.add_argument("--interface", type=float, default=0.5)
    parser.add_argument("--thickness", type=float, default=0.02)
    parser.add_argument("--perturb_amp", type=float, default=0.03)
    parser.add_argument("--perturb_mode", type=int, default=1)
    parser.add_argument("--u_cap", type=float, default=0.08)
    parser.add_argument("--force_cap", type=float, default=5e-4)
    parser.add_argument(
        "--boundary_y", type=str, default="noslip", choices=["noslip", "freeslip"]
    )
    parser.add_argument("--output", type=str, default="output_rtch_sweep")
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)
    run_sweep(args)


if __name__ == "__main__":
    main()
