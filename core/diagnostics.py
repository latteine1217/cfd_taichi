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
from typing import Optional, Dict, List


@ti.data_oriented
class Diagnostics:
    """統一的診斷與輸出管理"""

    def __init__(self, solver, output_dir: str = "output"):
        self.solver = solver
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.force_x = ti.field(dtype=ti.f32, shape=())
        self.force_y = ti.field(dtype=ti.f32, shape=())
        self.history = []
        self._headers = []

    @ti.kernel
    def compute_forces(self, f_field: ti.template()):
        self.force_x[None] = 0.0
        self.force_y[None] = 0.0
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            if self.solver.mask[i, j] == 0:
                for k in ti.static(range(9)):
                    ni, nj = i + self.solver.e[k][0], j + self.solver.e[k][1]
                    if 0 <= ni < self.solver.nx and 0 <= nj < self.solver.ny:
                        if self.solver.mask[ni, nj] == 1:
                            f_bounced = f_field[i, j][self.solver.inv[k]]
                            self.force_x[None] += 2.0 * f_bounced * self.solver.e[k][0]
                            self.force_y[None] += 2.0 * f_bounced * self.solver.e[k][1]

    def get_force_coefficients(self) -> tuple:
        self.compute_forces(self.solver.f)
        denom = 0.5 * 1.0 * (self.solver.u_ref**2) * self.solver.L_char
        return self.force_x[None] / (denom + 1e-12), self.force_y[None] / (denom + 1e-12)

    def get_residuals(self) -> Dict[str, float]:
        diag = self.solver.get_diagnostics()
        res_u = diag['mom_res_x'] / (diag['mom_scale_x'] + 1e-12)
        res_v = diag['mom_res_y'] / (diag['mom_scale_y'] + 1e-12)
        res_rho = diag['mass_residual'] / (diag['total_mass'] + 1e-12)
        return {'R_u': res_u, 'R_v': res_v, 'R_rho': res_rho}

    def print_header(self, include_forces: bool = False):
        """列印表格頭部"""
        if include_forces:
            self._headers = ["step", "R_u", "R_v", "R_rho", "M_err", "Cd", "Cl", "Umax", "ETA"]
        else:
            self._headers = ["step", "R_u", "R_v", "R_rho", "M_err", "Umax", "ETA"]
        
        # 建立空的表格來顯示表頭
        header_str = tabulate([], headers=self._headers, tablefmt="github")
        print(header_str)
        return self._headers

    def print_step_info(self, step, speed, eta_seconds, include_forces=False):
        """動態列印單行數據"""
        res = self.get_residuals()
        diag = self.solver.get_diagnostics()
        mass_err = abs(diag['total_mass'] - diag['initial_mass']) / (diag['initial_mass'] + 1e-12)
        umax = diag['max_u']
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if include_forces:
            cd, cl = self.get_force_coefficients()
            row = [step, f"{res['R_u']:.2e}", f"{res['R_v']:.2e}", f"{res['R_rho']:.2e}", 
                   f"{mass_err:.2e}", f"{cd:.4f}", f"{cl:.4f}", f"{umax:.4f}", eta_str]
        else:
            row = [step, f"{res['R_u']:.2e}", f"{res['R_v']:.2e}", f"{res['R_rho']:.2e}", 
                   f"{mass_err:.2e}", f"{umax:.4f}", eta_str]

        # 核心優化：使用 tabulate 的格式化邏輯來生成「單行」表格字串
        # 透過與表頭共享相同的 headers，確保每一列都完美對齊
        row_str = tabulate([row], headers=self._headers, tablefmt="github").split('\n')[-1]
        print(row_str)
        
        self.history.append(row)
        if mass_err > 1e-3:
            print(f"⚠️  WARNING: Mass conservation error: {mass_err:.2e}")
        return row

    def print_summary(self, headers: List[str]):
        """模擬結束後的最終總結表格"""
        print("\n--- Final Simulation Summary ---")
        print(tabulate(self.history, headers=headers, tablefmt="github"))
        self.print_physics_validation()

    def print_physics_validation(self):
        diag = self.solver.get_diagnostics()
        mass_err = abs(diag['total_mass'] - diag['initial_mass']) / (diag['initial_mass'] + 1e-12)
        print("\n" + "="*50)
        print(f"{ 'Physics Validation':^50}")
        print("-" * 50)
        print(f"Mass Error    : {mass_err:.2e} ({'✅' if mass_err < 1e-4 else '⚠️'})")
        print(f"Max CFL (U)   : {diag['max_u']:.4f} ({'✅' if diag['max_u'] < 1.0 else '❌'})")
        print(f"Final Mass    : {diag['total_mass']:.4f}")
        print("="*50 + "\n")

    def save_data(self, step: int, additional_data: Optional[Dict] = None):
        fields = self.solver.get_fields()
        data = {'rho': fields['rho'], 'u': fields['u'], 'mask': fields['mask'], 'step': step}
        
        # 收集粒子數據
        if hasattr(self.solver, 'p_active'):
            active = self.solver.p_active.to_numpy() == 1
            px, py = self.solver.px.to_numpy(), self.solver.py.to_numpy()
            data['particles'] = np.stack([px[active], py[active]], axis=1)

        if additional_data: data.update(additional_data)
        np.save(os.path.join(self.output_dir, f"state_{step:06d}.npy"), data)

    def check_convergence(self, tol: float = 1e-5) -> bool:
        res = self.get_residuals()
        return res['R_u'] < tol and res['R_v'] < tol and res['R_rho'] < tol
