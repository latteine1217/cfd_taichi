"""
CFD Taichi CLI
==============

What:
- 提供 benchmark registry 命令列入口

Why:
- toolkit-first workflow 應直接走 `benchmark registry -> matrix -> acceptance`
- 移除舊的案例腳本分派入口，避免 package surface 同時維護兩套路徑
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

from cfd_taichi.benchmark_matrix import (
    BenchmarkMatrixEntry,
    build_benchmark_matrix_payload,
    run_benchmark_matrix,
)
from cfd_taichi.benchmark_registry import get_benchmark_spec, list_benchmarks
SUMMARY_FIELD_ORDER = (
    "requested_name",
    "benchmark",
    "solver_family",
    "equation_set",
    "regime",
    "steps",
    "time",
    "history_samples",
    "acceptance_passed",
    "acceptance_failed_count",
    "u_max",
    "cfl",
    "mass_error",
    "mom_res_x",
    "mom_res_y",
    "div_linf",
    "projection_iters",
    "l2_profile_error",
    "linf_profile_error",
    "mach_max",
    "lift_coefficient",
    "drag_coefficient",
    "drag_coefficient_abs",
    "state_path",
    "history_path",
    "error",
)

INDEX_FIELD_ORDER = (
    "name",
    "aliases",
    "tags",
    "default_params",
    "acceptance_metrics",
    "description",
)


def _build_usage() -> str:
    lines = [
        "CFD Taichi CLI",
        "",
        "Usage:",
        "    cfd-taichi benchmark list",
        "    cfd-taichi benchmark <name> [options]",
    ]
    lines.extend(
        [
            "",
            "Benchmark options:",
            "    benchmark list",
            "    benchmark <name> --steps 10 --sample-interval 5 --set ni=64 --set nj=32",
        ]
    )
    return "\n".join(lines)


def _parse_override(raw: str) -> tuple[str, object]:
    key, sep, value = raw.partition("=")
    key = key.strip()
    if sep == "" or not key:
        raise ValueError(f"Override must use key=value form, got {raw!r}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return key, parsed


def _resolve_taichi_arch(name: str):
    import taichi as ti

    arch_name = name.strip().lower()
    arch_map = {
        "cpu": ti.cpu,
        "metal": ti.metal,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }
    if arch_name not in arch_map:
        raise ValueError(
            f"Unsupported Taichi arch {name!r}. Choose from: cpu, metal, cuda, vulkan."
        )
    return arch_map[arch_name]


def _init_taichi(arch_name: str):
    import taichi as ti

    ti.init(arch=_resolve_taichi_arch(arch_name), default_fp=ti.f32)


def _build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfd-taichi benchmark")
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        help="`list` or benchmark name; omitted defaults to list",
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Benchmark name when using explicit `run` form",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of time steps")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Record history every N steps",
    )
    parser.add_argument(
        "--record-initial",
        action="store_true",
        help="Record step 0 sample before stepping",
    )
    parser.add_argument(
        "--no-record-final",
        action="store_true",
        help="Disable implicit final sample recording",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for state/history outputs",
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Write standard state_final.npy under output dir",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Write standard history.npy under output dir",
    )
    parser.add_argument(
        "--payload-file",
        type=str,
        default=None,
        help="Optional `.npy` path for CLI schema payload",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Benchmark override in key=value form; repeatable",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="metal",
        help="Taichi arch for registry benchmark execution",
    )
    return parser


def _format_default_params(params: dict[str, object]) -> str:
    if not params:
        return "-"
    parts = [f"{key}={params[key]!r}" for key in sorted(params)]
    return ", ".join(parts)


def _format_acceptance_metrics(spec) -> str:
    if not spec.acceptance_criteria:
        return "-"
    metrics = []
    for criterion in spec.acceptance_criteria:
        bounds = []
        if criterion.min_value is not None:
            bounds.append(f"min={criterion.min_value}")
        if criterion.max_value is not None:
            bounds.append(f"max={criterion.max_value}")
        bounds_text = ", ".join(bounds) if bounds else "unbounded"
        metrics.append(
            f"{criterion.name}[metric={criterion.metric}; {bounds_text}; required={criterion.required}]"
        )
    return " | ".join(metrics)


def _build_benchmark_index_row(spec) -> dict[str, str]:
    """
    建立固定欄位的 benchmark index row。

    Why:
    - discovery 輸出也應該是穩定 schema，方便掃描與後續腳本擷取
    """
    return {
        "name": spec.name,
        "aliases": ", ".join(spec.aliases) if spec.aliases else "-",
        "tags": ", ".join(spec.tags) if spec.tags else "-",
        "default_params": _format_default_params(spec.default_params),
        "acceptance_metrics": _format_acceptance_metrics(spec),
        "description": spec.description,
    }


def _build_benchmark_index_payload(specs) -> dict[str, object]:
    """
    建立 benchmark index 的 `.npy` payload。

    What:
    - 將 CLI list schema 轉成固定欄位的 object-array payload

    Why:
    - discovery 與 stdout 應對應到同一份結構化資料，而不是各自發明格式
    """
    rows = [_build_benchmark_index_row(spec) for spec in specs]
    payload: dict[str, object] = {
        "schema_name": "benchmark_index",
        "field_order": np.asarray(INDEX_FIELD_ORDER, dtype=object),
        "count": int(len(rows)),
    }
    for key in INDEX_FIELD_ORDER:
        payload[key] = np.asarray([row[key] for row in rows], dtype=object)
    return payload


def _save_cli_payload(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    np.save(target, payload, allow_pickle=True)
    return target


def _print_benchmark_list(payload_path: str | None = None) -> None:
    specs = list_benchmarks()
    payload = _build_benchmark_index_payload(specs)
    print("=== Benchmark Index ===")
    for spec in specs:
        row = _build_benchmark_index_row(spec)
        print("---")
        for key in INDEX_FIELD_ORDER:
            print(f"{key}: {row[key]}")
    if payload_path is not None:
        saved = _save_cli_payload(payload_path, payload)
        print("=== Payload File ===")
        print(saved)


def _build_cli_summary_row(summary: dict[str, object], result) -> dict[str, object]:
    """
    建立固定欄位的 CLI benchmark summary row。

    Why:
    - stdout 應有穩定 schema，方便人眼掃描與外部工具擷取
    - 不應依不同 benchmark 的 diagnostics 多寡改變核心欄位集合
    """
    row = {
        "requested_name": result.requested_name,
        "benchmark": result.benchmark,
        "solver_family": result.solver_family,
        "equation_set": result.equation_set,
        "regime": result.regime,
        "steps": int(summary["steps"][0]),
        "time": float(summary["time"][0]),
        "history_samples": int(summary["history_samples"][0]),
        "acceptance_passed": bool(summary["acceptance_passed"][0]),
        "acceptance_failed_count": int(summary["acceptance_failed_count"][0]),
        "u_max": float(summary["u_max"][0]),
        "cfl": float(summary["cfl"][0]),
        "mass_error": float(summary["mass_error"][0]),
        "mom_res_x": float(summary["mom_res_x"][0]),
        "mom_res_y": float(summary["mom_res_y"][0]),
        "div_linf": float(summary["div_linf"][0]),
        "projection_iters": float(summary["projection_iters"][0]),
        "l2_profile_error": float(summary["l2_profile_error"][0]),
        "linf_profile_error": float(summary["linf_profile_error"][0]),
        "mach_max": float(summary["mach_max"][0]),
        "lift_coefficient": float(summary["lift_coefficient"][0]),
        "drag_coefficient": float(summary["drag_coefficient"][0]),
        "drag_coefficient_abs": float(summary["drag_coefficient_abs"][0]),
        "state_path": result.state_path,
        "history_path": result.history_path,
        "error": result.error,
    }
    return row


def _build_cli_summary_payload(row: dict[str, object]) -> dict[str, object]:
    """
    建立 benchmark summary 的 `.npy` payload。

    What:
    - 將單一 CLI summary row 以固定欄位寫成 payload
    """
    payload: dict[str, object] = {
        "schema_name": "benchmark_summary",
        "field_order": np.asarray(SUMMARY_FIELD_ORDER, dtype=object),
    }
    for key in SUMMARY_FIELD_ORDER:
        payload[key] = row.get(key)
    return payload


def _print_cli_summary(row: dict[str, object]) -> None:
    print("=== Benchmark Summary ===")
    for key in SUMMARY_FIELD_ORDER:
        print(f"{key}: {row.get(key)}")


def _run_benchmark_cli(argv: list[str]) -> None:
    parser = _build_benchmark_parser()
    normalized_argv = list(argv)
    if normalized_argv and normalized_argv[0] == "run":
        normalized_argv = normalized_argv[1:]
    args = parser.parse_args(normalized_argv)

    benchmark_name = args.command if args.name is None else args.name
    if args.command == "list" and args.name is None:
        _print_benchmark_list(args.payload_file)
        return

    overrides = dict(_parse_override(raw) for raw in args.overrides)
    spec = get_benchmark_spec(benchmark_name)

    _init_taichi(args.arch)

    entry = BenchmarkMatrixEntry(
        benchmark=spec.name,
        steps=args.steps,
        sample_interval=args.sample_interval,
        record_initial=args.record_initial,
        record_final=not args.no_record_final,
        output_dir=args.output_dir,
        save_state=args.save_state,
        save_history=args.save_history,
        overrides=overrides,
        history_params={"benchmark": spec.name, **overrides},
    )
    result = run_benchmark_matrix([entry])[0]
    summary = build_benchmark_matrix_payload([result])
    row = _build_cli_summary_row(summary, result)
    payload = _build_cli_summary_payload(row)

    if args.payload_file is not None:
        _save_cli_payload(args.payload_file, payload)

    _print_cli_summary(row)

    print("=== Diagnostics ===")
    for key in sorted(result.diagnostics):
        print(f"{key}: {result.diagnostics[key]}")

    print("=== Acceptance ===")
    for detail in result.acceptance_details:
        print(
            f"- {detail['name']}: passed={detail['passed']} "
            f"metric={detail['metric']} observed={detail['observed']} "
            f"reason={detail['reason']}"
        )

    if args.payload_file is not None:
        print("=== Payload File ===")
        print(args.payload_file)

    if result.error is not None:
        raise SystemExit(2)
    if result.acceptance_passed is False or bool(summary["acceptance_failed_count"][0]) is True:
        raise SystemExit(1)


def main() -> None:
    """分派至 benchmark registry。"""
    if len(sys.argv) < 2:
        print(_build_usage())
        raise SystemExit(1)

    command = sys.argv[1].lower()
    if command == "benchmark":
        _run_benchmark_cli(sys.argv[2:])
        return

    print(f"Unsupported command: {command}")
    print()
    print(_build_usage())
    raise SystemExit(1)
