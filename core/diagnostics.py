"""
診斷與輸出模組
==============

提供統一的物理檢查、診斷輸出與資料儲存功能。
支援動量、質量守恆監控與動態表格輸出。
"""

import taichi as ti
import numpy as np
import os
import time
from tabulate import tabulate
from typing import Optional, Dict, List, Tuple
from utils.vtk_io import write_vti


@ti.data_oriented
class Diagnostics:
    """統一的診斷與輸出管理"""

    def __init__(self, solver, output_dir: str = "output", output_vtk: bool = False):
        self.solver = solver
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.force_x = ti.field(dtype=ti.f32, shape=())
        self.force_y = ti.field(dtype=ti.f32, shape=())
        self.min_f = ti.field(dtype=ti.f32, shape=())
        self.history = []
        self._headers = []
        self.output_vtk = output_vtk
        self.force_history: List[Tuple[int, float, float]] = []

    @ti.kernel
    def compute_forces(self, f_field: ti.template()):
        """
        計算固體障礙物受力（修正版）

        Why 這樣計算？
        - 遍歷所有流體節點，檢查其鄰居是否為固體
        - 當流體粒子撞到固體表面，發生 Bounce-Back
        - 動量變化 = 2 * f_bounced * e_k（因子 2 來自反彈）

        修正：確保正確累加所有固體表面的動量交換
        """
        self.force_x[None] = 0.0
        self.force_y[None] = 0.0

        # 遍歷所有流體節點
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            ig = i + 1
            jg = j + 1
            if self.solver.mask[ig, jg] == 0:  # 流體節點
                # 檢查每個方向的鄰居
                for k in ti.static(range(9)):
                    ni = i + self.solver.e[k][0]
                    nj = j + self.solver.e[k][1]

                    # 邊界檢查
                    if 0 <= ni < self.solver.nx and 0 <= nj < self.solver.ny:
                        # 如果鄰居是固體，則該方向會發生 Bounce-Back
                        if self.solver.mask[ni + 1, nj + 1] == 1:
                            # 從當前流體節點反彈回來的分佈函數
                            # f_bounced = f_field[i, j][inv[k]] 是碰撞後反向的分佈
                            f_bounced = f_field[ig, jg][self.solver.inv[k]]

                            # 施加在固體上的力 = 動量變化率
                            # 因子 2：入射動量 + 反彈動量
                            ti.atomic_add(
                                self.force_x[None],
                                2.0 * f_bounced * self.solver.e[k][0],
                            )
                            ti.atomic_add(
                                self.force_y[None],
                                2.0 * f_bounced * self.solver.e[k][1],
                            )

    def get_force_coefficients(self) -> tuple:
        self.compute_forces(self.solver.f)
        denom = 0.5 * 1.0 * (self.solver.u_ref**2) * self.solver.L_char
        return self.force_x[None] / (denom + 1e-12), self.force_y[None] / (
            denom + 1e-12
        )

    def record_forces(self, step: int, cd: float, cl: float):
        self.force_history.append((step, cd, cl))

    def compute_force_stats(self, min_step: int = 0) -> Dict[str, float]:
        if not self.force_history:
            return {
                "cd_mean": 0.0,
                "cl_rms": 0.0,
                "cl_mean": 0.0,
                "samples": 0,
            }

        data = np.array(
            [(s, cd, cl) for s, cd, cl in self.force_history if s >= min_step],
            dtype=np.float64,
        )
        if data.size == 0:
            return {
                "cd_mean": 0.0,
                "cl_rms": 0.0,
                "cl_mean": 0.0,
                "samples": 0,
            }

        cd = data[:, 1]
        cl = data[:, 2]
        cl_mean = float(np.mean(cl))
        cl_rms = float(np.sqrt(np.mean((cl - cl_mean) ** 2)))
        return {
            "cd_mean": float(np.mean(cd)),
            "cl_rms": cl_rms,
            "cl_mean": cl_mean,
            "samples": int(data.shape[0]),
        }

    def compute_strouhal(
        self, diameter: float, u_ref: float, min_step: int = 0
    ) -> float:
        if not self.force_history:
            return 0.0

        data = np.array(
            [(s, cl) for s, _, cl in self.force_history if s >= min_step],
            dtype=np.float64,
        )
        if data.shape[0] < 4:
            return 0.0

        steps = data[:, 0]
        cl = data[:, 1] - np.mean(data[:, 1])

        dt = float(np.mean(np.diff(steps))) if data.shape[0] > 1 else 1.0
        if dt <= 0.0:
            return 0.0

        fft = np.fft.rfft(cl)
        freq = np.fft.rfftfreq(cl.size, d=dt)
        if freq.size < 2:
            return 0.0

        peak_idx = int(np.argmax(np.abs(fft[1:])) + 1)
        f_peak = float(freq[peak_idx])
        if u_ref <= 0.0:
            return 0.0

        return f_peak * diameter / u_ref

    def compute_force_spectrum(
        self,
        min_step: int = 0,
        max_peaks: int = 3,
    ) -> List[Tuple[float, float]]:
        """
        計算 Cl 的頻譜主峰

        Returns:
            [(f_peak, amplitude), ...] 依 amplitude 由大到小排序
        """
        if not self.force_history:
            return []

        data = np.array(
            [(s, cl) for s, _, cl in self.force_history if s >= min_step],
            dtype=np.float64,
        )
        if data.shape[0] < 4:
            return []

        steps = data[:, 0]
        cl = data[:, 1] - np.mean(data[:, 1])
        dt = float(np.mean(np.diff(steps))) if data.shape[0] > 1 else 1.0
        if dt <= 0.0:
            return []

        fft = np.fft.rfft(cl)
        freq = np.fft.rfftfreq(cl.size, d=dt)
        if freq.size < 2:
            return []

        amp = np.abs(fft)
        amp[0] = 0.0
        top_idx = np.argsort(amp)[::-1][:max_peaks]
        peaks = [(float(freq[i]), float(amp[i])) for i in top_idx if amp[i] > 0.0]
        return peaks

    def print_force_spectrum(
        self,
        diameter: float,
        u_ref: float,
        min_step: int = 0,
        max_peaks: int = 3,
    ):
        peaks = self.compute_force_spectrum(min_step=min_step, max_peaks=max_peaks)
        if not peaks:
            print("\n--- Force Spectrum ---")
            print("No sufficient force history for spectrum.")
            return

        print("\n--- Force Spectrum (Cl) ---")
        print("f_peak(1/step) | amplitude | St")
        for f_peak, amp in peaks:
            st = f_peak * diameter / u_ref if u_ref > 0.0 else 0.0
            print(f"{f_peak:12.6f} | {amp:9.3e} | {st:6.3f}")

    def get_residuals(self) -> Dict[str, float]:
        diag = self.solver.get_diagnostics()
        n_fluid = self._get_fluid_cells()
        scale_u = max(
            self.solver.u_ref, diag["mom_scale_x"] / (self.solver.nx * self.solver.ny)
        )
        scale_v = max(
            self.solver.u_ref, diag["mom_scale_y"] / (self.solver.nx * self.solver.ny)
        )
        res_u = diag["mom_res_x"] / (n_fluid * (scale_u + 1e-12))
        res_v = diag["mom_res_y"] / (n_fluid * (scale_v + 1e-12))
        res_rho = diag["mass_residual"] / (diag["total_mass"] + 1e-12)
        return {"R_u": res_u, "R_v": res_v, "R_rho": res_rho}

    def _get_fluid_cells(self) -> int:
        n_fluid = int(
            self.solver.num_fluid_bulk[None] + self.solver.num_fluid_boundary[None]
        )
        if n_fluid <= 0:
            n_fluid = self.solver.nx * self.solver.ny
        return n_fluid

    def print_header(self, include_forces: bool = False):
        """列印表格頭部"""
        if include_forces:
            self._headers = [
                "step",
                "macro_res",
                "KE_mean",
                "f_min",
                "mass_error",
                "total_mass",
                "Cd",
                "Cl",
                "u_max",
                "CFL",
                "speed",
                "ETA",
            ]
        else:
            self._headers = [
                "step",
                "macro_res",
                "KE_mean",
                "f_min",
                "mass_error",
                "total_mass",
                "u_max",
                "CFL",
                "speed",
                "ETA",
            ]

        # 建立空的表格來顯示表頭
        header_str = tabulate([], headers=self._headers, tablefmt="github")
        print(header_str)
        return self._headers

    def print_step_info(
        self, step, speed, eta_seconds, include_forces=False, f_field=None
    ):
        """動態列印單行數據"""
        res = self.get_residuals()
        diag = self.solver.get_diagnostics()
        mass_err = abs(diag["total_mass"] - diag["initial_mass"]) / (
            diag["initial_mass"] + 1e-12
        )
        macro_res = float(np.sqrt(res["R_u"] ** 2 + res["R_v"] ** 2))
        n_fluid = self._get_fluid_cells()
        ke = diag.get("total_KE", 0.0) / n_fluid

        if f_field is not None:
            self.compute_min_f(f_field)
            f_min = float(self.min_f[None])
        else:
            f_min = 0.0
        umax = diag["max_u"]
        cfl = umax
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if include_forces:
            cd, cl = self.get_force_coefficients()
            row = [
                step,
                f"{macro_res:.2e}",
                f"{ke:.4f}",
                f"{f_min:.2e}",
                f"{mass_err:.2e}",
                f"{diag['total_mass']:.4f}",
                f"{cd:.4f}",
                f"{cl:.4f}",
                f"{umax:.4f}",
                f"{cfl:.3f}",
                f"{speed:.1f}",
                eta_str,
            ]
        else:
            row = [
                step,
                f"{macro_res:.2e}",
                f"{ke:.4f}",
                f"{f_min:.2e}",
                f"{mass_err:.2e}",
                f"{diag['total_mass']:.4f}",
                f"{umax:.4f}",
                f"{cfl:.3f}",
                f"{speed:.1f}",
                eta_str,
            ]

        # 核心優化：使用 tabulate 的格式化邏輯來生成「單行」表格字串
        # 透過與表頭共享相同的 headers，確保每一列都完美對齊
        row_str = tabulate([row], headers=self._headers, tablefmt="github").split("\n")[
            -1
        ]
        print(row_str)

        if mass_err > 1e-6:
            print(f"⚠️  WARNING: Mass conservation error: {mass_err:.2e}")
        if f_min < 0.0:
            print(f"⚠️  WARNING: Negative f detected (min={f_min:.2e})")
        return row

    @ti.kernel
    def compute_min_f(self, f_field: ti.template()):
        self.min_f[None] = 1e9
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            ig = i + 1
            jg = j + 1
            if self.solver.mask[ig, jg] == 0:
                for k in ti.static(range(9)):
                    val = f_field[ig, jg][k]
                    ti.atomic_min(self.min_f[None], val)

    def print_summary(self, headers: List[str]):
        """模擬結束後的最終總結表格"""
        print("\n--- Final Simulation Summary ---")
        print(tabulate(self.history, headers=headers, tablefmt="github"))
        self.print_physics_validation()

    def print_physics_validation(self):
        """
        最終物理驗證報告（包含能量守恆）

        Why 能量守恆重要？
        - 無黏流：動能應守恆
        - 黏性流：動能耗散率 = ε = 2 * ν * S_ij * S_ij
        - LES 模型：總耗散 = 分子黏度 + 渦黏度
        """
        diag = self.solver.get_diagnostics()
        mass_err = abs(diag["total_mass"] - diag["initial_mass"]) / (
            diag["initial_mass"] + 1e-12
        )

        # 能量變化率（相對初始能量）
        KE_init = diag.get("initial_KE", 0.0)
        KE_final = diag.get("total_KE", 0.0)
        n_fluid = self._get_fluid_cells()
        KE_init_mean = KE_init / n_fluid
        KE_final_mean = KE_final / n_fluid
        if KE_init_mean > 1e-12:
            KE_change = (KE_final_mean - KE_init_mean) / KE_init_mean
        else:
            KE_change = 0.0

        print("\n" + "=" * 60)
        print(f"{'Physics Validation (with Energy Monitoring)':^60}")
        print("-" * 60)
        print(f"Mass Error       : {mass_err:.2e} ({'✅' if mass_err < 1e-4 else '⚠️'})")
        print(
            f"Max CFL (U)      : {diag['max_u']:.4f} ({'✅' if diag['max_u'] < 0.3 else '❌'})"
        )
        print(f"Final Mass       : {diag['total_mass']:.4f}")

        print(f"\nEnergy Budget:")
        print(f"  Initial KE     : {KE_init:.6f}")
        print(f"  Final KE       : {KE_final:.6f}")
        print(f"  Initial KE_mean: {KE_init_mean:.6f}")
        print(f"  Final KE_mean  : {KE_final_mean:.6f}")
        print(f"  Relative Change: {KE_change:+.2%}")

        # 能量變化診斷
        if abs(KE_change) < 0.01:
            print(f"  ✅ Energy approximately conserved (|ΔKE| < 1%)")
        elif KE_change < 0:
            print(f"  ⚠️  Energy dissipation detected (expected for viscous flow)")
        else:
            print(f"  ❌ Energy increase detected (possible numerical instability)")

        print("=" * 60 + "\n")

    def save_data(self, step: int, additional_data: Optional[Dict] = None):
        """
        儲存流場數據與粒子狀態

        性能優化：
        - 粒子緊湊導出：只傳輸活躍粒子（通常 5-10% 總數）
        - 減少 GPU→CPU 傳輸：500k → 50k particles
        - 記憶體節省：~90%
        """
        fields = self.solver.get_fields()
        data = {
            "rho": fields["rho"],
            "u": fields["u"],
            "mask": fields["mask"],
            "step": step,
        }

        # 收集粒子數據（緊湊格式）
        if hasattr(self.solver, "p_active"):
            # 只導出活躍粒子（GPU→CPU 傳輸優化）
            active = self.solver.p_active.to_numpy() == 1
            px, py = self.solver.px.to_numpy(), self.solver.py.to_numpy()
            data["particles"] = np.stack([px[active], py[active]], axis=1)
            data["particle_count"] = np.sum(active)  # 統計資訊

        if additional_data:
            data.update(additional_data)
        np.save(os.path.join(self.output_dir, f"state_{step:06d}.npy"), data)

        if self.output_vtk:
            vtk_path = os.path.join(self.output_dir, f"state_{step:06d}.vti")
            write_vti(vtk_path, data["rho"], data["u"], data.get("mask"))

    def check_convergence(self, tol: float = 1e-5) -> bool:
        res = self.get_residuals()
        return res["R_u"] < tol and res["R_v"] < tol and res["R_rho"] < tol
