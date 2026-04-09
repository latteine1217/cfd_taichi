"""
Benchmark Matrix
================

What:
- 提供 benchmark matrix 的批次執行、摘要輸出與回歸工作流入口

Why:
- `BenchmarkSpec -> CaseRunner` 只解決單一 benchmark 建立，還缺正式的 batch /
  regression 執行層
- matrix 應集中處理 history/state 儲存、錯誤狀態與摘要 payload，而不是散在
  notebook 或 shell script

When:
- 批次 benchmark smoke test
- regression matrix
- benchmark dashboard / automation 的上游資料來源
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_registry import get_benchmark_spec
from .case_runner import CaseRunner


@dataclass(frozen=True)
class BenchmarkMatrixEntry:
    """
    單一 matrix benchmark 執行設定。
    """

    benchmark: str
    steps: int
    overrides: dict[str, Any] = field(default_factory=dict)
    sample_interval: int = 0
    record_initial: bool = False
    record_final: bool = True
    output_dir: str | Path | None = None
    save_state: bool = False
    save_history: bool = False
    history_params: dict[str, Any] = field(default_factory=dict)
    additional_state_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkMatrixResult:
    """
    單一 matrix benchmark 執行結果。
    """

    requested_name: str
    benchmark: str
    solver_family: str
    equation_set: str
    regime: str
    tags: tuple[str, ...]
    step: int
    time: float
    history_samples: int
    diagnostics: dict[str, Any]
    acceptance_passed: bool | None = None
    acceptance_details: tuple[dict[str, Any], ...] = ()
    state_path: str | None = None
    history_path: str | None = None
    error: str | None = None


def _should_record_step(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def _record_history_sample(runner: CaseRunner, *, force: bool = False):
    if not force and runner.current_step <= 0:
        return
    runner.record_history_sample(extra={"time": float(runner.current_time)})


def _lookup_metric(result: BenchmarkMatrixResult, metric: str) -> Any:
    if metric in {
        "requested_name",
        "benchmark",
        "solver_family",
        "equation_set",
        "regime",
        "step",
        "time",
        "history_samples",
        "error",
    }:
        return getattr(result, metric)
    return result.diagnostics.get(metric)


def _evaluate_acceptance_criteria(
    spec,
    result: BenchmarkMatrixResult,
) -> tuple[bool | None, tuple[dict[str, Any], ...]]:
    criteria = tuple(spec.acceptance_criteria)
    if not criteria:
        return None, ()

    details: list[dict[str, Any]] = []
    all_passed = True
    for criterion in criteria:
        observed = _lookup_metric(result, criterion.metric)
        passed = True
        reason = ""

        if observed is None or (isinstance(observed, float) and not np.isfinite(observed)):
            if criterion.required:
                passed = False
                reason = f"Required metric '{criterion.metric}' is missing."
            else:
                reason = f"Optional metric '{criterion.metric}' is missing."
        else:
            observed_value = float(observed) if np.isscalar(observed) else observed
            if criterion.min_value is not None and float(observed_value) < float(criterion.min_value):
                passed = False
                reason = (
                    f"Metric '{criterion.metric}'={float(observed_value):.6g} is below "
                    f"minimum {float(criterion.min_value):.6g}."
                )
            if (
                passed
                and criterion.max_value is not None
                and float(observed_value) > float(criterion.max_value)
            ):
                passed = False
                reason = (
                    f"Metric '{criterion.metric}'={float(observed_value):.6g} exceeds "
                    f"maximum {float(criterion.max_value):.6g}."
                )
            if passed:
                reason = "Accepted."

        details.append(
            {
                "name": criterion.name,
                "metric": criterion.metric,
                "passed": bool(passed),
                "observed": observed,
                "min_value": criterion.min_value,
                "max_value": criterion.max_value,
                "required": criterion.required,
                "description": criterion.description,
                "reason": reason,
            }
        )
        all_passed = all_passed and passed
    return all_passed, tuple(details)


def _run_entry(entry: BenchmarkMatrixEntry) -> BenchmarkMatrixResult:
    if entry.steps < 0:
        raise ValueError(f"Matrix entry steps must be non-negative, got {entry.steps}")
    if entry.sample_interval < 0:
        raise ValueError(
            f"Matrix entry sample_interval must be non-negative, got {entry.sample_interval}"
        )

    spec = get_benchmark_spec(entry.benchmark)
    runner = spec.build_runner(**entry.overrides)
    runner.build_solver()
    runner.configure()
    runner.initialize()

    if entry.record_initial:
        runner.prepare_observables(reset_baseline=False)
        _record_history_sample(runner, force=True)

    last_sampled_step = runner.current_step if entry.record_initial else -1

    for _ in range(entry.steps):
        runner.step_once()
        if _should_record_step(runner.current_step, entry.sample_interval):
            runner.prepare_observables(reset_baseline=False)
            _record_history_sample(runner)
            last_sampled_step = runner.current_step

    diagnostics = runner.prepare_observables(reset_baseline=False) or {}
    if entry.record_final and runner.current_step > 0 and last_sampled_step != runner.current_step:
        _record_history_sample(runner)

    state_path: str | None = None
    history_path: str | None = None
    output_dir: Path | None = None
    if entry.output_dir is not None:
        output_dir = Path(entry.output_dir) / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)

    if entry.save_state:
        if output_dir is None:
            raise ValueError("save_state=True requires output_dir to be provided.")
        target = runner.save_state_file(
            output_dir / "state_final.npy",
            additional_data=entry.additional_state_data,
        )
        state_path = str(target)

    if entry.save_history:
        if output_dir is None:
            raise ValueError("save_history=True requires output_dir to be provided.")
        target = runner.save_history_file(
            output_dir / "history.npy",
            params=entry.history_params or {"benchmark": spec.name},
        )
        history_path = str(target)

    base_result = BenchmarkMatrixResult(
        requested_name=entry.benchmark,
        benchmark=spec.name,
        solver_family=str(getattr(runner.solver, "solver_family", "unknown")),
        equation_set=str(getattr(runner.solver, "equation_set", "unknown")),
        regime=str(getattr(runner.solver, "regime", "unknown")),
        tags=tuple(spec.tags),
        step=int(runner.current_step),
        time=float(runner.current_time),
        history_samples=int(len(runner._history_samples)),
        diagnostics=dict(diagnostics),
        state_path=state_path,
        history_path=history_path,
    )
    acceptance_passed, acceptance_details = _evaluate_acceptance_criteria(spec, base_result)
    return BenchmarkMatrixResult(
        requested_name=base_result.requested_name,
        benchmark=base_result.benchmark,
        solver_family=base_result.solver_family,
        equation_set=base_result.equation_set,
        regime=base_result.regime,
        tags=base_result.tags,
        step=base_result.step,
        time=base_result.time,
        history_samples=base_result.history_samples,
        diagnostics=base_result.diagnostics,
        acceptance_passed=acceptance_passed,
        acceptance_details=acceptance_details,
        state_path=base_result.state_path,
        history_path=base_result.history_path,
        error=base_result.error,
    )


def run_benchmark_matrix(
    entries: list[BenchmarkMatrixEntry] | tuple[BenchmarkMatrixEntry, ...],
    *,
    stop_on_error: bool = True,
) -> list[BenchmarkMatrixResult]:
    """
    執行 benchmark matrix。

    What:
    - 依序執行多個 benchmark entry
    - 每個 entry 可自訂步數、採樣頻率與輸出路徑

    Why:
    - toolkit 需要一個正式的 batch / regression 執行入口，而不是靠外部腳本重組
      `CaseRunner`
    """
    results: list[BenchmarkMatrixResult] = []
    for entry in entries:
        try:
            results.append(_run_entry(entry))
        except Exception as exc:
            if stop_on_error:
                raise
            spec_name = entry.benchmark.strip()
            results.append(
                BenchmarkMatrixResult(
                    requested_name=spec_name,
                    benchmark=spec_name,
                    solver_family="unknown",
                    equation_set="unknown",
                    regime="unknown",
                    tags=(),
                    step=0,
                    time=0.0,
                    history_samples=0,
                    diagnostics={},
                    acceptance_passed=False,
                    acceptance_details=(),
                    error=str(exc),
                )
            )
    return results


def build_benchmark_matrix_payload(
    results: list[BenchmarkMatrixResult] | tuple[BenchmarkMatrixResult, ...],
) -> dict[str, Any]:
    """
    建立 benchmark matrix 摘要 payload。
    """
    entries = list(results)
    n = len(entries)
    payload: dict[str, Any] = {
        "requested_names": np.asarray([item.requested_name for item in entries], dtype=object),
        "benchmarks": np.asarray([item.benchmark for item in entries], dtype=object),
        "solver_family": np.asarray([item.solver_family for item in entries], dtype=object),
        "equation_set": np.asarray([item.equation_set for item in entries], dtype=object),
        "regime": np.asarray([item.regime for item in entries], dtype=object),
        "tags": np.asarray([item.tags for item in entries], dtype=object),
        "steps": np.asarray([item.step for item in entries], dtype=np.int64),
        "time": np.asarray([item.time for item in entries], dtype=np.float64),
        "history_samples": np.asarray([item.history_samples for item in entries], dtype=np.int64),
        "acceptance_passed": np.asarray(
            [False if item.acceptance_passed is None else bool(item.acceptance_passed) for item in entries],
            dtype=bool,
        ),
        "acceptance_details": np.asarray([item.acceptance_details for item in entries], dtype=object),
        "errors": np.asarray([item.error for item in entries], dtype=object),
        "state_paths": np.asarray([item.state_path for item in entries], dtype=object),
        "history_paths": np.asarray([item.history_path for item in entries], dtype=object),
    }

    def _diag_series(name: str, default: float = np.nan) -> np.ndarray:
        return np.asarray(
            [float(item.diagnostics.get(name, default)) for item in entries],
            dtype=np.float64,
        ) if n > 0 else np.asarray([], dtype=np.float64)

    payload["u_max"] = _diag_series("u_max")
    payload["cfl"] = _diag_series("cfl", default=np.nan)
    payload["mass_error"] = _diag_series("mass_error", default=np.nan)
    payload["mom_res_x"] = _diag_series("mom_res_x", default=np.nan)
    payload["mom_res_y"] = _diag_series("mom_res_y", default=np.nan)
    payload["div_linf"] = _diag_series("div_linf", default=np.nan)
    payload["projection_iters"] = _diag_series("projection_iters", default=np.nan)
    payload["l2_profile_error"] = _diag_series("l2_profile_error", default=np.nan)
    payload["linf_profile_error"] = _diag_series("linf_profile_error", default=np.nan)
    payload["mach_inlet"] = _diag_series("mach_inlet", default=np.nan)
    payload["mach_throat"] = _diag_series("mach_throat", default=np.nan)
    payload["mach_outlet"] = _diag_series("mach_outlet", default=np.nan)
    payload["mach_max"] = _diag_series("mach_max", default=np.nan)
    payload["supersonic_fraction"] = _diag_series("supersonic_fraction", default=np.nan)
    payload["shock_sensor_max"] = _diag_series("shock_sensor_max", default=np.nan)
    payload["entropy_rise_max"] = _diag_series("entropy_rise_max", default=np.nan)
    payload["shock_peak_x"] = _diag_series("shock_peak_x", default=np.nan)
    payload["shock_peak_y"] = _diag_series("shock_peak_y", default=np.nan)
    payload["mdot_balance"] = _diag_series("mdot_balance", default=np.nan)
    payload["nozzle_acceleration_ratio"] = _diag_series("nozzle_acceleration_ratio", default=np.nan)
    payload["lift_coefficient"] = _diag_series("lift_coefficient", default=np.nan)
    payload["drag_coefficient"] = _diag_series("drag_coefficient", default=np.nan)
    payload["drag_coefficient_abs"] = _diag_series("drag_coefficient_abs", default=np.nan)
    payload["acceptance_failed_count"] = np.asarray(
        [
            int(sum(0 if detail.get("passed") else 1 for detail in item.acceptance_details))
            + (1 if item.error is not None else 0)
            for item in entries
        ],
        dtype=np.int64,
    ) if n > 0 else np.asarray([], dtype=np.int64)
    return payload


def save_benchmark_matrix_payload(
    path: str | Path,
    results: list[BenchmarkMatrixResult] | tuple[BenchmarkMatrixResult, ...],
    *,
    additional_payload: dict[str, Any] | None = None,
) -> Path:
    """
    將 benchmark matrix 摘要寫入 `.npy`。
    """
    payload = build_benchmark_matrix_payload(results)
    if additional_payload:
        payload.update(dict(additional_payload))
    target = Path(path)
    np.save(target, payload, allow_pickle=True)
    return target
