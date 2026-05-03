"""
統一輸出 schema
================

What:
- 提供跨 solver 共用的 state / history payload builder

Why:
- LBM / FVM 與不同案例不應各自發明 `.npy` 欄位格式
- 後處理、回歸比較與 benchmark 對照需要穩定的輸出 contract

When:
- diagnostics 儲存 state/history
- solver 或案例需要輸出標準化 payload
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def compute_vorticity_2d(
    velocity: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    計算 2D 速度場渦度。

    What:
    - 由 cell-centered `(u, v)` 場估算 `omega = dv/dx - du/dy`

    Why:
    - 渦度是跨方法比較最常用的流場診斷量
    - state schema 應預設提供，不該由每個案例各自重算
    """
    vel = np.asarray(velocity, dtype=np.float32)
    if vel.ndim != 3 or vel.shape[2] != 2:
        raise ValueError(
            f"Expected velocity shape (ni, nj, 2), got {vel.shape}"
        )

    u = vel[:, :, 0]
    v = vel[:, :, 1]
    du_dy = np.gradient(u, float(dy), axis=1, edge_order=1)
    dv_dx = np.gradient(v, float(dx), axis=0, edge_order=1)
    return np.asarray(dv_dx - du_dy, dtype=np.float32)


def _get_solver_metadata(solver: Any) -> dict[str, str]:
    return {
        "solver_family": str(getattr(solver, "solver_family", "unknown")),
        "equation_set": str(getattr(solver, "equation_set", "unknown")),
        "regime": str(getattr(solver, "regime", "unknown")),
    }


def _infer_spatial_shape(fields: Mapping[str, Any]) -> tuple[int, int]:
    for key in ("u", "rho", "p", "phi", "rhoA", "rhoB", "mask", "mach", "divergence"):
        value = fields.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 3:
            return int(arr.shape[0]), int(arr.shape[1])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("Unable to infer spatial shape from fields.")


def _ensure_mask(fields: Mapping[str, Any]) -> np.ndarray:
    mask = fields.get("mask")
    if mask is not None:
        return np.asarray(mask, dtype=np.int32)
    ni, nj = _infer_spatial_shape(fields)
    return np.zeros((ni, nj), dtype=np.int32)


def build_state_payload(
    *,
    solver: Any,
    fields: Mapping[str, Any],
    step: int,
    time_value: float | None = None,
    additional_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    建立統一 state payload。

    What:
    - 輸出最小共通欄位：`u`, `mask`, `step`, `time`, `vorticity`
    - 若 solver 有提供 `rho`, `p`, `phi`, `mach` 等欄位則一併保留

    Why:
    - 既有案例仍可攜帶額外欄位，但核心 schema 必須固定
    """
    payload = dict(_get_solver_metadata(solver))
    payload["step"] = int(step)
    payload["time"] = float(step if time_value is None else time_value)

    for key in ("u", "rho", "p", "phi", "rhoA", "rhoB", "mach", "divergence", "nu_sgs"):
        value = fields.get(key)
        if value is not None:
            payload[key] = np.asarray(value)

    payload["mask"] = _ensure_mask(fields)

    if "u" in payload:
        try:
            payload["vorticity"] = compute_vorticity_2d(payload["u"])
        except ValueError:
            payload["vorticity"] = None
    else:
        payload["vorticity"] = None

    if additional_data:
        payload.update(dict(additional_data))
    return payload


def _coerce_series(
    values: list[Any] | np.ndarray | None,
    *,
    size: int,
    dtype: Any,
    fill_value: float | int,
) -> np.ndarray:
    if values is None:
        return np.full(size, fill_value, dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    if arr.shape != (size,):
        raise ValueError(f"Expected series shape ({size},), got {arr.shape}")
    return arr


def build_history_payload(
    *,
    solver: Any,
    steps: list[Any] | np.ndarray,
    mass_error: list[Any] | np.ndarray | None = None,
    mom_res_x: list[Any] | np.ndarray | None = None,
    mom_res_y: list[Any] | np.ndarray | None = None,
    u_max: list[Any] | np.ndarray | None = None,
    cfl: list[Any] | np.ndarray | None = None,
    params: Mapping[str, Any] | None = None,
    legacy_headers: list[str] | None = None,
    legacy_rows: list[Any] | None = None,
    extra_series: Mapping[str, list[Any] | np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    建立統一 history payload。

    What:
    - 固定輸出 `steps`, `mass_error`, `mom_res_x`, `mom_res_y`, `u_max`, `cfl`
    - 保留 `headers` / `data` 舊格式，避免現有後處理直接斷裂

    Why:
    - history 是 benchmark regression 的主契約，欄位需要跨 solver 一致
    """
    steps_arr = np.asarray(steps, dtype=np.int64)
    if steps_arr.ndim != 1:
        raise ValueError(f"Expected 1D steps array, got {steps_arr.shape}")
    size = int(steps_arr.shape[0])

    payload: dict[str, Any] = dict(_get_solver_metadata(solver))
    payload["steps"] = steps_arr
    payload["mass_error"] = _coerce_series(
        mass_error, size=size, dtype=np.float64, fill_value=np.nan
    )
    payload["mom_res_x"] = _coerce_series(
        mom_res_x, size=size, dtype=np.float64, fill_value=np.nan
    )
    payload["mom_res_y"] = _coerce_series(
        mom_res_y, size=size, dtype=np.float64, fill_value=np.nan
    )
    payload["u_max"] = _coerce_series(
        u_max, size=size, dtype=np.float64, fill_value=np.nan
    )
    payload["cfl"] = _coerce_series(
        cfl, size=size, dtype=np.float64, fill_value=np.nan
    )

    if params is not None:
        payload["params"] = dict(params)
    if legacy_headers is not None:
        payload["headers"] = list(legacy_headers)
    if legacy_rows is not None:
        payload["data"] = list(legacy_rows)
    if extra_series:
        for key, values in extra_series.items():
            payload[key] = _coerce_series(
                values, size=size, dtype=np.float64, fill_value=np.nan
            )
    return payload
