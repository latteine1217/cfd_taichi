"""
Cahn-Hilliard LBM Diagnostics
=============================
"""

from typing import List

import numpy as np
import os
import taichi as ti

from cfd_taichi.output_schema import build_history_payload, build_state_payload


@ti.data_oriented
class CHDiagnostics:
    def __init__(self, solver, output_dir: str):
        self.solver = solver
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.total_phi = ti.field(dtype=ti.f32, shape=())
        self.total_ke = ti.field(dtype=ti.f32, shape=())
        self.max_u = ti.field(dtype=ti.f32, shape=())

        self.initial_phi = 0.0
        self.n_cells = float(self.solver.nx * self.solver.ny)
        self.history: List[List] = []
        self.headers: List[str] = []
        self._history_samples: List[dict[str, float]] = []

    @ti.kernel
    def _update(self):
        self.total_phi[None] = 0.0
        self.total_ke[None] = 0.0
        self.max_u[None] = 0.0
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            ig = i + 1
            jg = j + 1
            rho = self.solver.rho[ig, jg]
            u = self.solver.u[ig, jg]
            phi = self.solver.phi[ig, jg]
            ti.atomic_add(self.total_phi[None], phi)
            ti.atomic_add(self.total_ke[None], 0.5 * rho * u.norm_sqr())
            ti.atomic_max(self.max_u[None], u.norm())

    def set_initial_baseline(self):
        self._update()
        self.initial_phi = float(self.total_phi[None])

    def print_header(self):
        self.headers = [
            "step",
            "phi_drift",
            "u_max",
            "KE",
            "mix_thick",
            "bubble_y",
            "spike_y",
            "speed(step/s)",
            "eta(s)",
        ]
        print("=== Diagnostics ===")
        print("| " + " | ".join(self.headers) + " |")
        print("| " + " | ".join(["-" * len(h) for h in self.headers]) + " |")
        return self.headers

    def _interface_metrics(self):
        phi = self.solver.get_fields()["phi"]
        valid = np.abs(phi) < 0.9
        if np.any(valid):
            ys = np.where(np.any(valid, axis=0))[0]
            mix_thick = float(ys.max() - ys.min() + 1)
        else:
            mix_thick = 0.0

        bubble = ""
        spike = ""
        neg = np.where(phi < -0.5)
        pos = np.where(phi > 0.5)
        if neg[1].size > 0:
            bubble = str(int(np.max(neg[1])))
        if pos[1].size > 0:
            spike = str(int(np.min(pos[1])))
        return mix_thick, bubble, spike

    def print_step_info(self, step: int, speed: float, eta_seconds: float):
        self._update()
        phi_drift = abs(float(self.total_phi[None]) - self.initial_phi) / (
            self.n_cells + 1e-12
        )
        mix_thick, bubble, spike = self._interface_metrics()
        row = [
            step,
            f"{phi_drift:.2e}",
            f"{float(self.max_u[None]):.3e}",
            f"{float(self.total_ke[None]):.3e}",
            f"{mix_thick:.1f}",
            bubble,
            spike,
            f"{speed:.2f}",
            f"{eta_seconds:.1f}",
        ]
        self._history_samples.append(
            {
                "step": float(step),
                "mass_error": float(phi_drift),
                "mom_res_x": np.nan,
                "mom_res_y": np.nan,
                "u_max": float(self.max_u[None]),
                "cfl": float(self.max_u[None]),
                "phi_drift": float(phi_drift),
            }
        )
        print("| " + " | ".join(map(str, row)) + " |")
        return row

    def save_data(self, step: int, time_value: float):
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
        params: dict | None = None,
        filename: str = "history.npy",
        additional_payload: dict | None = None,
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
            legacy_headers=self.headers,
            legacy_rows=self.history,
            extra_series={
                "phi_drift": [sample["phi_drift"] for sample in self._history_samples],
            },
        )
        if additional_payload:
            payload.update(dict(additional_payload))
        history_path = os.path.join(self.output_dir, filename)
        np.save(history_path, payload, allow_pickle=True)
        return history_path
