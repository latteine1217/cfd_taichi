"""
多相 LBM 診斷與輸出
===================

What:
- 提供多相 LBM 的基礎診斷與輸出功能

Why:
- 多相 solver 獨立於單相流程，需要專用 diagnostics

When:
- RT 等多相案例輸出與監控
"""

from typing import Dict, List

import numpy as np
import os
import taichi as ti

from cfd_taichi.output_schema import build_history_payload, build_state_payload


@ti.data_oriented
class MultiphaseDiagnostics:
    """
    多相 LBM 診斷工具

    What:
    - 計算各相質量守恆、最大速度、總動能

    Why:
    - 多相模擬易失穩，需要可觀測的診斷量

    When:
    - 迭代過程中定期輸出
    """

    def __init__(
        self,
        solver,
        output_dir: str = "output_multiphase",
        enable_interface_metrics: bool = True,
        interface_threshold: float = 0.9,
    ):
        self.solver = solver
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.total_mass_a = ti.field(dtype=ti.f32, shape=())
        self.total_mass_b = ti.field(dtype=ti.f32, shape=())
        self.total_mass = ti.field(dtype=ti.f32, shape=())
        self.total_KE = ti.field(dtype=ti.f32, shape=())
        self.max_u = ti.field(dtype=ti.f32, shape=())

        self.initial_mass_a = 0.0
        self.initial_mass_b = 0.0

        self.enable_interface_metrics = enable_interface_metrics
        self.interface_threshold = interface_threshold

        self.history: List[List] = []
        self._headers: List[str] = []
        self._history_samples: List[Dict[str, float]] = []

    @ti.kernel
    def _update_diagnostics(self):
        self.total_mass_a[None] = 0.0
        self.total_mass_b[None] = 0.0
        self.total_mass[None] = 0.0
        self.total_KE[None] = 0.0
        self.max_u[None] = 0.0

        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            ig = i + 1
            jg = j + 1
            if self.solver.mask[ig, jg] == 0:
                rho_a = self.solver.rhoA[ig, jg]
                rho_b = self.solver.rhoB[ig, jg]
                u = self.solver.u[ig, jg]
                ti.atomic_add(self.total_mass_a[None], rho_a)
                ti.atomic_add(self.total_mass_b[None], rho_b)
                ti.atomic_add(self.total_mass[None], rho_a + rho_b)
                ti.atomic_add(
                    self.total_KE[None], 0.5 * (rho_a + rho_b) * u.norm_sqr()
                )
                ti.atomic_max(self.max_u[None], u.norm())

    def update(self) -> Dict[str, float]:
        """
        What: 更新診斷量
        Why: 取得即時物理量
        When: 每次輸出或檢查時
        """
        self._update_diagnostics()
        return {
            "total_mass_a": float(self.total_mass_a[None]),
            "total_mass_b": float(self.total_mass_b[None]),
            "total_mass": float(self.total_mass[None]),
            "total_KE": float(self.total_KE[None]),
            "max_u": float(self.max_u[None]),
        }

    def set_initial_baseline(self):
        """
        What: 設定初始質量基準
        Why: 計算相對質量誤差
        When: 初始化後第一步
        """
        diag = self.update()
        self.initial_mass_a = diag["total_mass_a"]
        self.initial_mass_b = diag["total_mass_b"]

    def print_header(self) -> List[str]:
        """
        What: 印出診斷表頭
        Why: 結構化輸出方便追蹤
        When: 模擬開始時
        """
        self._headers = [
            "step",
            "mass_err_A",
            "mass_err_B",
            "u_max",
            "KE",
            "mix_thick",
            "bubble_y",
            "spike_y",
            "speed(step/s)",
            "eta(s)",
        ]
        print("=== Diagnostics ===")
        print(
            "| "
            + " | ".join(self._headers)
            + " |"
        )
        print(
            "| "
            + " | ".join(["-" * len(h) for h in self._headers])
            + " |"
        )
        return self._headers

    def print_step_info(self, step: int, speed: float, eta_seconds: float) -> List:
        """
        What: 印出單步診斷資訊
        Why: 追蹤收斂與穩定性
        When: 每 N 步輸出
        """
        diag = self.update()
        mass_err_a = abs(diag["total_mass_a"] - self.initial_mass_a) / (
            self.initial_mass_a + 1e-12
        )
        mass_err_b = abs(diag["total_mass_b"] - self.initial_mass_b) / (
            self.initial_mass_b + 1e-12
        )

        mix_thick = ""
        bubble_y = ""
        spike_y = ""
        if self.enable_interface_metrics:
            mix_thick = f"{self._compute_mixing_layer():.1f}"
            bubble_y, spike_y = self._compute_bubble_spike_fronts()

        row = [
            step,
            f"{mass_err_a:.2e}",
            f"{mass_err_b:.2e}",
            f"{diag['max_u']:.3e}",
            f"{diag['total_KE']:.3e}",
            mix_thick,
            bubble_y,
            spike_y,
            f"{speed:.2f}",
            f"{eta_seconds:.1f}",
        ]
        total_initial = self.initial_mass_a + self.initial_mass_b
        total_mass = diag["total_mass"]
        self._history_samples.append(
            {
                "step": float(step),
                "mass_error": float(abs(total_mass - total_initial) / (total_initial + 1e-12)),
                "mom_res_x": np.nan,
                "mom_res_y": np.nan,
                "u_max": float(diag["max_u"]),
                "cfl": float(diag["max_u"]),
                "mass_err_A": float(mass_err_a),
                "mass_err_B": float(mass_err_b),
            }
        )
        print("| " + " | ".join(map(str, row)) + " |")
        return row

    def _compute_mixing_layer(self) -> float:
        """
        What: 計算混合層厚度（phi 閾值）
        Why: 追蹤 RT 混合層成長
        When: 需要界面診斷時
        """
        self.solver._compute_density()
        fields = self.solver.get_fields()
        phi = fields["phi"]
        mask = fields.get("mask", None)
        if mask is not None:
            phi = np.where(mask == 1, np.nan, phi)

        threshold = float(self.interface_threshold)
        valid = np.abs(phi) < threshold
        if not np.any(valid):
            return 0.0

        y_indices = np.where(np.any(valid, axis=0))[0]
        if y_indices.size == 0:
            return 0.0
        return float(y_indices.max() - y_indices.min() + 1)

    def _compute_bubble_spike_fronts(self) -> tuple[str, str]:
        """
        What: 計算 RT 氣泡/尖刺前沿位置（y 索引）
        Why: 量化界面是否形成向上氣泡與向下尖刺
        When: 介面診斷輸出時
        """
        self.solver._compute_density()
        phi = self.solver.get_fields()["phi"]

        pos = np.where(phi > 0.5)
        neg = np.where(phi < -0.5)

        bubble = ""
        spike = ""
        if neg[1].size > 0:
            bubble = f"{int(np.max(neg[1]))}"
        if pos[1].size > 0:
            spike = f"{int(np.min(pos[1]))}"
        return bubble, spike

    def save_data(self, step: int, time_value: float = 0.0):
        """
        What: 輸出多相場資料
        Why: 後處理與可視化
        When: 依照 interval 輸出
        """
        self.solver._compute_density()
        fields = self.solver.get_fields()
        data = build_state_payload(
            solver=self.solver,
            fields=fields,
            step=step,
            time_value=time_value,
        )
        np.save(os.path.join(self.output_dir, f"state_{step:06d}.npy"), data)

    def save_history(
        self,
        params: Dict | None = None,
        filename: str = "history.npy",
        additional_payload: Dict | None = None,
    ) -> str:
        steps = [int(sample["step"]) for sample in self._history_samples]
        payload = build_history_payload(
            solver=self.solver,
            steps=steps,
            mass_error=[sample["mass_error"] for sample in self._history_samples],
            mom_res_x=[sample["mom_res_x"] for sample in self._history_samples],
            mom_res_y=[sample["mom_res_y"] for sample in self._history_samples],
            u_max=[sample["u_max"] for sample in self._history_samples],
            cfl=[sample["cfl"] for sample in self._history_samples],
            params=params,
            legacy_headers=self._headers,
            legacy_rows=self.history,
            extra_series={
                "mass_err_A": [sample["mass_err_A"] for sample in self._history_samples],
                "mass_err_B": [sample["mass_err_B"] for sample in self._history_samples],
            },
        )
        if additional_payload:
            payload.update(dict(additional_payload))
        history_path = os.path.join(self.output_dir, filename)
        np.save(history_path, payload, allow_pickle=True)
        return history_path
