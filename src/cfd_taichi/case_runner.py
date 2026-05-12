"""
CaseRunner 工作流骨架
=====================

What:
- 提供共用的 case workflow：build / configure / initialize / step / save

Why:
- solver toolkit 不應只有 solver 類別，還要有一致的使用流程
- `examples/*.py` 目前仍大量手寫主迴圈；CaseRunner 是把這些流程正式工具化的第一步

When:
- 新 benchmark
- notebook / automation / batch runner
- 後續 benchmark registry 的執行核心
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import (
    apply_boundary_descriptors,
    apply_grid_descriptor,
    apply_solver_control_descriptors,
)
from .descriptors import BoundaryConditionDescriptor, SolverControlDescriptor
from .grid2d import Grid2D
from .output_schema import build_history_payload, build_state_payload
from .solver_factory import create_solver

InitializerFn = Callable[["CaseRunner", Any, Any], None]
StepperFn = Callable[["CaseRunner", int], Any]
PostStepFn = Callable[["CaseRunner", int, Any], None]
DiagnosticsHookFn = Callable[["CaseRunner", Any, dict[str, Any]], Mapping[str, Any] | None]
StateHookFn = Callable[["CaseRunner", Any, dict[str, Any]], Mapping[str, Any] | None]


class CaseRunner:
    """
    共用 case workflow 執行器。

    What:
    - 統一 solver 建立、grid/BC 套用、初始化、時間推進與標準輸出

    Why:
    - 讓案例從 script-first 邁向 toolkit-first
    """

    def __init__(
        self,
        *,
        name: str,
        method: str,
        grid: Grid2D,
        equation: str | None = None,
        regime: str | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        boundary_conditions: Sequence[BoundaryConditionDescriptor] | None = None,
        solver_controls: Sequence[SolverControlDescriptor] | None = None,
        initializer: InitializerFn | None = None,
        stepper: StepperFn | None = None,
        diagnostics_hook: DiagnosticsHookFn | None = None,
        state_hook: StateHookFn | None = None,
    ):
        self.name = str(name)
        self.method = str(method)
        self.equation = equation
        self.regime = regime
        self.grid = grid
        self.solver_kwargs = dict(solver_kwargs or {})
        self.boundary_conditions = list(boundary_conditions or [])
        self.solver_controls = list(solver_controls or [])
        self.initializer = initializer
        self.stepper = stepper
        self.diagnostics_hook = diagnostics_hook
        self.state_hook = state_hook

        self.solver: Any | None = None
        self.boundary_handle: Any | None = None
        self.control_handle: Any | None = None
        self.current_step = 0
        self.current_time = 0.0
        self._history_samples: list[dict[str, Any]] = []
        self._history_metadata: dict[str, Any] = {
            "solver_family": "lbm" if self.method.strip().lower() == "lbm" else "fvm",
            "equation_set": self.equation if self.equation is not None else "unknown",
            "regime": self.regime if self.regime is not None else "unknown",
        }

    def _merge_solver_kwargs_with_grid(self) -> dict[str, Any]:
        kwargs = dict(self.solver_kwargs)
        solver_family = "lbm" if self.method.strip().lower() == "lbm" else "fvm"
        size_kwargs = self.grid.solver_size_kwargs(solver_family)
        for key, value in size_kwargs.items():
            current = kwargs.get(key)
            if current is None:
                kwargs[key] = value
            elif int(current) != int(value):
                raise ValueError(
                    f"Grid size {key}={value} incompatible with solver_kwargs[{key}]={current}"
                )
        return kwargs

    def build_solver(self):
        """
        建立 solver instance。
        """
        kwargs = self._merge_solver_kwargs_with_grid()
        self.solver = create_solver(
            method=self.method,
            equation=self.equation,
            regime=self.regime,
            **kwargs,
        )
        self._history_metadata.update(
            {
                "solver_family": getattr(self.solver, "solver_family", self._history_metadata["solver_family"]),
                "equation_set": getattr(self.solver, "equation_set", self._history_metadata["equation_set"]),
                "regime": getattr(self.solver, "regime", self._history_metadata["regime"]),
            }
        )
        return self.solver

    def configure(self):
        """
        套用 grid、boundary 與 solver control descriptors。
        """
        if self.solver is None:
            self.build_solver()

        apply_grid_descriptor(self.solver, self.grid)
        self.boundary_handle = apply_boundary_descriptors(
            self.solver,
            self.boundary_conditions,
        ) if self.boundary_conditions else None
        self.control_handle = apply_solver_control_descriptors(
            self.solver,
            self.solver_controls,
        ) if self.solver_controls else None
        return self.solver

    def initialize(self):
        """
        執行使用者提供的初始化流程。
        """
        if self.solver is None:
            self.configure()
        if self.initializer is None:
            return self.solver
        self.initializer(self, self.solver, self.boundary_handle)
        return self.solver

    def prepare_observables(self, reset_baseline: bool = False):
        """
        更新可觀測量。

        What:
        - LBM: 以目前 active distribution 更新 macro/diagnostics
        - FVM: 直接讀取結構化 diagnostics
        """
        if self.solver is None:
            raise RuntimeError("Solver is not built. Call build_solver() first.")

        if getattr(self.solver, "solver_family", None) == "lbm" and hasattr(self.solver, "prepare_diagnostics"):
            diagnostics = self.solver.prepare_diagnostics(
                f_src=self.active_distribution_field(),
                reset_baseline=reset_baseline,
            )
            return self._augment_diagnostics(diagnostics)
        if hasattr(self.solver, "get_diagnostics"):
            diagnostics = self.solver.get_diagnostics()
            return self._augment_diagnostics(diagnostics)
        return None

    def _augment_diagnostics(self, diagnostics: Mapping[str, Any] | None):
        """
        將案例自定義 diagnostics 合併進共同診斷字典。

        Why:
        - 某些 benchmark 的 acceptance 需要解析解誤差、頻譜或幾何特定量
        - 這些派生量不該硬塞進 solver 本體，而應由 workflow 層提供可插拔擴充點
        """
        merged = dict(diagnostics or {})
        total_mass = merged.get("total_mass")
        initial_mass = merged.get("initial_mass")
        if total_mass is not None and initial_mass is not None and "mass_error" not in merged:
            merged["mass_error"] = float(
                abs(total_mass - initial_mass) / (abs(initial_mass) + 1e-12)
            )
        if "max_u" in merged and "u_max" not in merged:
            merged["u_max"] = float(merged["max_u"])
        if self.diagnostics_hook is None:
            return merged
        extra = self.diagnostics_hook(self, self.solver, merged)
        if extra:
            merged.update(dict(extra))
        return merged

    def active_distribution_field(self):
        """
        取得目前 LBM 主狀態分佈函數。
        """
        if self.solver is None or getattr(self.solver, "solver_family", None) != "lbm":
            return None
        return self.solver.f_new if self.current_step % 2 == 1 else self.solver.f

    def step_once(self):
        """
        執行一個時間步。
        """
        if self.solver is None:
            raise RuntimeError("Solver is not built. Call build_solver() first.")

        next_step = self.current_step + 1
        if getattr(self.solver, "solver_family", None) == "lbm":
            if self.stepper is not None:
                result = self.stepper(self, next_step)
            else:
                f_src = self.solver.f if next_step % 2 == 1 else self.solver.f_new
                f_dst = self.solver.f_new if next_step % 2 == 1 else self.solver.f
                result = self.solver.step(f_src, f_dst)
        else:
            result = self.solver.step()

        self.current_step = next_step
        if isinstance(result, (int, float, np.floating)):
            self.current_time += float(result)
        else:
            self.current_time = float(self.current_step)
        return result

    def run_steps(self, steps: int, post_step: PostStepFn | None = None):
        """
        連續執行多步。
        """
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        result = None
        for _ in range(steps):
            result = self.step_once()
            if post_step is not None:
                post_step(self, self.current_step, result)
        return result

    def save_state(
        self,
        *,
        step: int | None = None,
        time_value: float | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        建立標準 state payload。
        """
        if self.solver is None:
            raise RuntimeError("Solver is not built. Call build_solver() first.")
        if getattr(self.solver, "solver_family", None) == "lbm":
            self.prepare_observables(reset_baseline=False)
        extra_state = dict(additional_data or {})
        if self.state_hook is not None:
            hook_data = self.state_hook(self, self.solver, extra_state)
            if hook_data:
                extra_state.update(dict(hook_data))
        payload = build_state_payload(
            solver=self.solver,
            fields=self.solver.get_fields(),
            step=self.current_step if step is None else int(step),
            time_value=self.current_time if time_value is None else time_value,
            additional_data=extra_state or None,
        )
        return payload

    def save_state_file(
        self,
        path: str | Path,
        *,
        step: int | None = None,
        time_value: float | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> Path:
        """
        將標準 state payload 寫入 `.npy`。
        """
        payload = self.save_state(
            step=step,
            time_value=time_value,
            additional_data=additional_data,
        )
        target = Path(path)
        np.save(target, payload, allow_pickle=True)
        return target

    def _coerce_state_payload(self, source: Any) -> dict[str, Any]:
        if isinstance(source, (str, Path)):
            payload = np.load(Path(source), allow_pickle=True).item()
        elif isinstance(source, dict):
            payload = dict(source)
        else:
            raise TypeError(
                "state source must be a dict payload or a .npy path."
            )
        return payload

    def _ensure_ready_for_state_load(self):
        if self.solver is None:
            self.build_solver()
            self.configure()

    def _validate_restart_metadata(self, payload: dict[str, Any]):
        if self.solver is None:
            raise RuntimeError("Solver is not built.")
        expected = {
            "solver_family": getattr(self.solver, "solver_family", "unknown"),
            "equation_set": getattr(self.solver, "equation_set", "unknown"),
            "regime": getattr(self.solver, "regime", "unknown"),
        }
        for key, value in expected.items():
            found = payload.get(key)
            if found is None:
                continue
            if str(found) != str(value):
                raise ValueError(
                    f"Restart payload {key}={found!r} incompatible with current solver {value!r}."
                )

    def load_state(
        self,
        source: Any,
        *,
        apply_boundaries: bool = True,
        reset_baseline: bool = False,
        clear_history: bool = True,
    ) -> dict[str, Any]:
        """
        從標準 state payload 重啟 runner 狀態。

        What:
        - 支援直接吃 payload dict 或 `.npy` state file

        Why:
        - restart/load_state 屬於案例工作流責任，不應散落在 examples
        """
        payload = self._coerce_state_payload(source)
        self._ensure_ready_for_state_load()
        self._validate_restart_metadata(payload)

        if clear_history:
            self._history_samples.clear()

        if getattr(self.solver, "solver_family", None) == "lbm":
            velocity = payload.get("u")
            density = payload.get("rho")
            if velocity is None:
                raise ValueError("LBM restart payload requires 'u'.")
            if density is None:
                raise ValueError("LBM restart payload requires 'rho'.")
            self.solver.set_initial_condition(
                velocity=np.ascontiguousarray(velocity, dtype=np.float32),
                density=np.ascontiguousarray(density, dtype=np.float32),
                apply_boundaries=apply_boundaries,
                reset_baseline=reset_baseline,
            )
            mask = payload.get("mask")
            if mask is not None:
                current_mask = self.solver.get_fields().get("mask")
                if current_mask is not None and current_mask.shape == np.asarray(mask).shape:
                    if not np.array_equal(current_mask, np.asarray(mask)):
                        raise ValueError(
                            "Restart payload mask does not match current LBM geometry."
                        )
            self.prepare_observables(reset_baseline=reset_baseline)
        else:
            vel = payload.get("u")
            p = payload.get("p")
            if vel is None or p is None:
                raise ValueError("FVM restart payload requires 'u' and 'p'.")
            vel_arr = np.asarray(vel, dtype=np.float32)
            p_arr = np.asarray(p, dtype=np.float32)
            if getattr(self.solver, "regime", None) == "compressible":
                rho = payload.get("rho")
                if rho is None:
                    raise ValueError("Compressible FVM restart payload requires 'rho'.")
                w = np.zeros((self.solver.ni, self.solver.nj, 4), dtype=np.float32)
                w[:, :, 0] = np.asarray(rho, dtype=np.float32)
                w[:, :, 1:3] = vel_arr
                w[:, :, 3] = p_arr
                self.solver.init_from_primitive_numpy(w)
            else:
                w = np.zeros((self.solver.ni, self.solver.nj, 3), dtype=np.float32)
                w[:, :, :2] = vel_arr
                w[:, :, 2] = p_arr
                self.solver.init_from_primitive_numpy(w)

        self.current_step = int(payload.get("step", 0))
        self.current_time = float(payload.get("time", self.current_step))
        return payload

    def _coerce_history_payload(self, source: Any) -> dict[str, Any]:
        if isinstance(source, (str, Path)):
            payload = np.load(Path(source), allow_pickle=True).item()
        elif isinstance(source, dict):
            payload = dict(source)
        else:
            raise TypeError(
                "history source must be a dict payload or a .npy path."
            )
        return payload

    def record_history_sample(
        self,
        *,
        step: int | None = None,
        diagnostics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        收集一筆標準 history sample。
        """
        if self.solver is None:
            raise RuntimeError("Solver is not built. Call build_solver() first.")

        if diagnostics is None:
            diagnostics = self.prepare_observables(reset_baseline=False) or {}

        sample = {
            "step": int(self.current_step if step is None else step),
            "mass_error": np.nan,
            "mom_res_x": np.nan,
            "mom_res_y": np.nan,
            "u_max": float(diagnostics.get("u_max", np.nan)),
            "cfl": float(
                diagnostics.get(
                    "cfl",
                    diagnostics.get("pseudo_cfl", diagnostics.get("u_max", np.nan)),
                )
            ),
        }

        total_mass = diagnostics.get("total_mass")
        initial_mass = diagnostics.get("initial_mass")
        if total_mass is not None and initial_mass is not None:
            sample["mass_error"] = float(
                abs(total_mass - initial_mass) / (abs(initial_mass) + 1e-12)
            )

        if "mom_res_x" in diagnostics:
            sample["mom_res_x"] = float(diagnostics["mom_res_x"])
        if "mom_res_y" in diagnostics:
            sample["mom_res_y"] = float(diagnostics["mom_res_y"])

        for key, value in diagnostics.items():
            if key in sample:
                continue
            if isinstance(value, (str, bytes)):
                continue
            if np.isscalar(value):
                sample[key] = float(value)

        if extra:
            sample.update(dict(extra))
        self._history_samples.append(sample)
        return sample

    def build_history_payload(
        self,
        *,
        params: dict[str, Any] | None = None,
        additional_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        建立 runner 累積的 history payload。
        """
        if not self._history_samples:
            raise RuntimeError("No history samples recorded.")

        base_keys = {"step", "mass_error", "mom_res_x", "mom_res_y", "u_max", "cfl"}
        extra_series: dict[str, list[Any]] = {}
        for key in self._history_samples[0]:
            if key in base_keys:
                continue
            extra_series[key] = [sample.get(key, np.nan) for sample in self._history_samples]

        history_solver = self.solver
        if history_solver is None:
            history_solver = type("_HistoryMetadata", (), self._history_metadata)()

        payload = build_history_payload(
            solver=history_solver,
            steps=[sample["step"] for sample in self._history_samples],
            mass_error=[sample["mass_error"] for sample in self._history_samples],
            mom_res_x=[sample["mom_res_x"] for sample in self._history_samples],
            mom_res_y=[sample["mom_res_y"] for sample in self._history_samples],
            u_max=[sample["u_max"] for sample in self._history_samples],
            cfl=[sample["cfl"] for sample in self._history_samples],
            params=params,
            extra_series=extra_series or None,
        )
        if additional_payload:
            payload.update(dict(additional_payload))
        return payload

    def save_history_file(
        self,
        path: str | Path,
        *,
        params: dict[str, Any] | None = None,
        additional_payload: dict[str, Any] | None = None,
    ) -> Path:
        """
        將 runner 累積的 history payload 寫入 `.npy`。
        """
        payload = self.build_history_payload(
            params=params,
            additional_payload=additional_payload,
        )
        target = Path(path)
        np.save(target, payload, allow_pickle=True)
        return target

    def load_history(
        self,
        source: Any,
        *,
        clear_existing: bool = True,
    ) -> dict[str, Any]:
        """
        載入標準 history payload 並重建 runner 內部 sample 緩存。

        What:
        - 支援 `.npy` 或 dict payload
        - 將標準欄位與額外 series 還原為 `_history_samples`

        Why:
        - batch / regression workflow 需要把既有 history 重新掛回 runner，
          再做比較、聚合或延續輸出
        """
        payload = self._coerce_history_payload(source)
        steps = np.asarray(payload.get("steps"))
        if steps.ndim != 1:
            raise ValueError("History payload requires 1D 'steps'.")

        reserved = {
            "solver_family",
            "equation_set",
            "regime",
            "params",
            "headers",
            "data",
        }
        series_keys = [
            key
            for key, value in payload.items()
            if key not in reserved and isinstance(value, np.ndarray) and value.shape == steps.shape
        ]

        if clear_existing:
            self._history_samples.clear()
        self._history_metadata.update(
            {
                "solver_family": payload.get("solver_family", self._history_metadata["solver_family"]),
                "equation_set": payload.get("equation_set", self._history_metadata["equation_set"]),
                "regime": payload.get("regime", self._history_metadata["regime"]),
            }
        )

        for idx in range(int(steps.shape[0])):
            sample = {
                key: (
                    float(payload[key][idx])
                    if np.issubdtype(np.asarray(payload[key]).dtype, np.number)
                    else payload[key][idx]
                )
                for key in series_keys
            }
            if "step" not in sample:
                sample["step"] = int(steps[idx])
            self._history_samples.append(sample)

        if steps.size > 0:
            self.current_step = int(steps[-1])
            if "time" in payload and isinstance(payload["time"], np.ndarray) and payload["time"].shape == steps.shape:
                self.current_time = float(payload["time"][-1])
            else:
                self.current_time = float(self.current_step)
        return payload
