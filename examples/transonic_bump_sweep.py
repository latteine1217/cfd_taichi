"""
Transonic Bump Sweep
====================

What: 對 transonic bump Euler 驗證案例做 Ma 或 bump height 掃描
Why:  單一工作點只能看局部結果；shock 位置與強度對參數的趨勢更適合作為驗證基準
      掃描執行現在應重用 benchmark matrix preset，而不是維持獨立逐點 workflow
When: 用於建立 shock_peak_x / mach_max / supersonic_fraction 的基準曲線
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from cfd_taichi import (
    build_transonic_bump_sweep_preset,
    run_benchmark_matrix,
    save_benchmark_matrix_payload,
)


def parse_values(values_text: str | None, start: float, stop: float, step: float) -> list[float]:
    """
    解析 sweep 取樣點
    """
    if values_text:
        values = [float(token.strip()) for token in values_text.split(",") if token.strip()]
        if not values:
            raise ValueError("values must contain at least one numeric entry")
        return values

    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"stop must be >= start, got start={start}, stop={stop}")

    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = [start + k * step for k in range(count)]
    if values[-1] < stop - 1e-10:
        values.append(stop)
    return [float(value) for value in values]


def plot_series(
        x_values: np.ndarray,
        y_values: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: str):
    """
    繪製單一 sweep 指標曲線
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, y_values, "ko-", lw=1.6, ms=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_sweep_outputs(output_dir: str, sweep: str, results: list[dict], params: dict):
    """
    保存 sweep 結果
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "sweep_results.npy"), {
        "mode": sweep,
        "params": params,
        "results": results,
    })

    fieldnames = [
        "ma", "bump_height", "mach_max", "supersonic_fraction",
        "shock_sensor_max", "entropy_rise_max",
        "shock_peak_x", "shock_peak_y",
        "residual", "final_step", "shock_plateau",
        "shock_peak_span", "mach_peak_span", "runtime_seconds",
        "converged", "output_dir",
    ]
    with open(os.path.join(output_dir, "sweep_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_sweep_figures(output_dir: str, sweep: str, results: list[dict]):
    """
    繪製 sweep 摘要圖
    """
    if sweep == "ma":
        results = sorted(results, key=lambda row: row["ma"])
        x_values = np.array([row["ma"] for row in results], dtype=float)
        xlabel = "Freestream Mach  Ma"
        suffix = "ma"
    else:
        results = sorted(results, key=lambda row: row["bump_height"])
        x_values = np.array([row["bump_height"] for row in results], dtype=float)
        xlabel = "Bump Height"
        suffix = "bump_height"

    plot_series(
        x_values,
        np.array([row["shock_peak_x"] for row in results], dtype=float),
        xlabel=xlabel,
        ylabel="Shock Peak x",
        title=f"Shock Position vs {xlabel.split()[-1]} — Transonic Bump",
        save_path=os.path.join(output_dir, f"shock_peak_vs_{suffix}.png"),
    )
    plot_series(
        x_values,
        np.array([row["mach_max"] for row in results], dtype=float),
        xlabel=xlabel,
        ylabel="Max Mach",
        title=f"Max Mach vs {xlabel.split()[-1]} — Transonic Bump",
        save_path=os.path.join(output_dir, f"mach_max_vs_{suffix}.png"),
    )
    plot_series(
        x_values,
        100.0 * np.array([row["supersonic_fraction"] for row in results], dtype=float),
        xlabel=xlabel,
        ylabel="Supersonic Area (%)",
        title=f"Supersonic Fraction vs {xlabel.split()[-1]} — Transonic Bump",
        save_path=os.path.join(output_dir, f"supersonic_fraction_vs_{suffix}.png"),
    )


def run_sweep(
        sweep: str = "ma",
        values: list[float] | None = None,
        ni: int = 180,
        nj: int = 60,
        ma: float = 0.675,
        steps: int = 2000,
        length: float = 3.0,
        height: float = 1.0,
        bump_center: float = 1.5,
        bump_width: float = 1.0,
        bump_height: float = 0.12,
        cfl: float = 0.35,
        report: int = 200,
        tol: float = 1e-5,
        plateau_window: int = 5,
        shock_tol: float = 0.02,
        mach_tol: float = 0.03,
        output_dir: str | None = None,
        save_cases: bool = False):
    """
    執行 bump 參數掃描
    """
    if sweep not in {"ma", "bump_height"}:
        raise ValueError(f"sweep must be 'ma' or 'bump_height', got {sweep}")
    if values is None or len(values) == 0:
        raise ValueError("values must contain at least one sweep point")

    values = sorted(float(value) for value in values)
    print("=" * 96)
    print("  Transonic Bump Sweep")
    print("=" * 96)
    print(f"  Sweep mode: {sweep}")
    print(f"  Sweep values: {values}")
    print(f"  Grid: ni={ni} x nj={nj}, steps={steps}, report={report}, CFL={cfl}")
    print(f"  Geometry: length={length}, height={height}, center={bump_center}, width={bump_width}, bump_height={bump_height}")
    print("-" * 96)
    print(f"  {'value':>8}  {'step':>7}  {'res':>10}  {'M_max':>8}  {'M>1%':>8}  {'x_shock':>9}  {'plateau':>8}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")

    entries = build_transonic_bump_sweep_preset(
        sweep=sweep,
        values=values,
        ni=ni,
        nj=nj,
        ma=ma,
        steps=steps,
        length=length,
        height=height,
        bump_center=bump_center,
        bump_width=bump_width,
        bump_height=bump_height,
        cfl=cfl,
        report=report,
        tol=tol,
        plateau_window=plateau_window,
        shock_tol=shock_tol,
        mach_tol=mach_tol,
        output_dir=output_dir,
        save_cases=save_cases,
    )
    matrix_results = run_benchmark_matrix(entries)
    results = []
    for value, entry, matrix_result in zip(values, entries, matrix_results, strict=True):
        if matrix_result.error is not None:
            raise RuntimeError(
                f"Case failed during {sweep} sweep at value={value}: {matrix_result.error}"
            )

        diagnostics = matrix_result.diagnostics
        case_ma = float(entry.overrides["ma"])
        case_bump_height = float(entry.overrides["bump_height"])
        case_output = None
        if matrix_result.state_path is not None:
            case_output = os.path.dirname(matrix_result.state_path)
        elif entry.output_dir is not None:
            case_output = str(entry.output_dir)

        residual = float(diagnostics.get("residual_rms", np.nan))
        row = {
            "ma": case_ma,
            "bump_height": case_bump_height,
            "mach_max": float(diagnostics.get("mach_max", np.nan)),
            "supersonic_fraction": float(diagnostics.get("supersonic_fraction", np.nan)),
            "shock_sensor_max": float(diagnostics.get("shock_sensor_max", np.nan)),
            "entropy_rise_max": float(diagnostics.get("entropy_rise_max", np.nan)),
            "shock_peak_x": float(diagnostics.get("shock_peak_x", np.nan)),
            "shock_peak_y": float(diagnostics.get("shock_peak_y", np.nan)),
            "residual": residual,
            "final_step": int(matrix_result.step),
            "shock_plateau": False,
            "shock_peak_span": float("nan"),
            "mach_peak_span": float("nan"),
            "runtime_seconds": float("nan"),
            "converged": bool(np.isfinite(residual) and residual <= tol),
            "output_dir": case_output,
        }
        results.append(row)

        print(f"  {value:8.4f}  {row['final_step']:7d}  {row['residual']:10.3e}  {row['mach_max']:8.4f}  "
              f"{100.0 * row['supersonic_fraction']:7.3f}%  {row['shock_peak_x']:9.4f}  {str(row['shock_plateau']):>8}")

    print("=" * 96)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        params = {
            "sweep": sweep,
            "values": values,
            "ni": ni,
            "nj": nj,
            "ma": ma,
            "steps": steps,
            "length": length,
            "height": height,
            "bump_center": bump_center,
            "bump_width": bump_width,
            "bump_height": bump_height,
            "cfl": cfl,
            "report": report,
            "tol": tol,
            "plateau_window": plateau_window,
            "shock_tol": shock_tol,
            "mach_tol": mach_tol,
            "save_cases": save_cases,
        }
        save_sweep_outputs(output_dir, sweep, results, params)
        save_benchmark_matrix_payload(
            os.path.join(output_dir, "matrix_summary.npy"),
            matrix_results,
            additional_payload={
                "preset": "transonic_bump_sweep",
                "sweep": sweep,
                "values": np.asarray(values, dtype=np.float64),
            },
        )
        write_sweep_figures(output_dir, sweep, results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Transonic bump sweep")
    parser.add_argument("--sweep", choices=["ma", "bump_height"], default="ma", help="Sweep variable")
    parser.add_argument("--values", type=str, default="", help="Comma-separated sweep values; overrides start/stop/step")
    parser.add_argument("--start", type=float, default=0.60, help="Sweep start")
    parser.add_argument("--stop", type=float, default=0.70, help="Sweep stop")
    parser.add_argument("--step", type=float, default=0.025, help="Sweep step")
    parser.add_argument("--ni", type=int, default=180, help="Streamwise interior cells")
    parser.add_argument("--nj", type=int, default=60, help="Wall-normal interior cells")
    parser.add_argument("--ma", type=float, default=0.675, help="Fixed freestream Mach for bump_height sweep")
    parser.add_argument("--steps", type=int, default=2000, help="Maximum time steps")
    parser.add_argument("--length", type=float, default=3.0, help="Channel length")
    parser.add_argument("--height", type=float, default=1.0, help="Channel height")
    parser.add_argument("--bump_center", type=float, default=1.5, help="Bump center x")
    parser.add_argument("--bump_width", type=float, default=1.0, help="Bump support width")
    parser.add_argument("--bump_height", type=float, default=0.12, help="Fixed bump height for Ma sweep")
    parser.add_argument("--cfl", type=float, default=0.35, help="Explicit CFL")
    parser.add_argument("--report", type=int, default=200, help="Report interval")
    parser.add_argument("--tol", type=float, default=1e-5, help="Residual tolerance")
    parser.add_argument("--plateau_window", type=int, default=5, help="History window for shock plateau check")
    parser.add_argument("--shock_tol", type=float, default=0.02, help="Shock-position plateau tolerance")
    parser.add_argument("--mach_tol", type=float, default=0.03, help="Max-Mach plateau tolerance")
    parser.add_argument("--save_cases", action="store_true", help="Save full output for each work point")
    parser.add_argument(
        "--output",
        type=str,
        default="output_transonic_bump_sweep",
        help="Sweep output directory (empty string disables file output)",
    )
    args = parser.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    try:
        values = parse_values(
            values_text=args.values if args.values else None,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        run_sweep(
            sweep=args.sweep,
            values=values,
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
            plateau_window=args.plateau_window,
            shock_tol=args.shock_tol,
            mach_tol=args.mach_tol,
            output_dir=args.output if args.output else None,
            save_cases=args.save_cases,
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"\n  ❌ Transonic bump sweep failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
