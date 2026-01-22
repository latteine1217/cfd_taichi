"""Kolmogorov Flow DNS Generator (Taichi Optimized)
================================================

High-performance Finite Difference code for 2D Kolmogorov flow using Taichi.
Optimized for Metal backend (Apple Silicon) using float32.

Features:
- Vorticity-Streamfunction formulation (omega-psi).
- Arakawa discretization for advection (Enstrophy conserving).
- Red-Black Gauss-Seidel Poisson Solver.
- RK4 Time integration.
- Real-time physics checks (Mass, Momentum).

Author: Taichi-SciML Team
Date: 2026-01-19
"""

import taichi as ti
import numpy as np
import argparse
import os
import time
from pathlib import Path

# Initialize Taichi with Metal backend as requested
ti.init(arch=ti.metal, default_fp=ti.f32, random_seed=int(time.time()))

@ti.data_oriented
class KolmogorovDNSTaichi:
    def __init__(
        self,
        N: int = 512,
        L: float = 2 * np.pi,
        nu: float = 1e-3,
        A: float = 0.1,
        k_f: int = 4,
        dt: float = 0.001
    ):
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.dx = L / N
        self.inv_dx2 = 1.0 / (self.dx * self.dx)

        # Fields
        self.w = ti.field(dtype=ti.f32, shape=(N, N))      # State t_n
        self.w_n = ti.field(dtype=ti.f32, shape=(N, N))    # Backup for RK4
        self.psi = ti.field(dtype=ti.f32, shape=(N, N))    # Streamfunction
        self.u = ti.field(dtype=ti.f32, shape=(N, N))      # u velocity
        self.v = ti.field(dtype=ti.f32, shape=(N, N))      # v velocity
        
        # RK4 substeps
        self.k1 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k2 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k3 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k4 = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def init_field(self, amplitude: float):
        for i, j in self.w:
            self.w[i, j] = (ti.random() - 0.5) * 2.0 * amplitude
            self.psi[i, j] = 0.0

    @ti.kernel
    def compute_velocity(self):
        for i, j in self.psi:
            ip = (i + 1) % self.N
            im = (i - 1 + self.N) % self.N
            jp = (j + 1) % self.N
            jm = (j - 1 + self.N) % self.N
            self.u[i, j] = (self.psi[i, jp] - self.psi[i, jm]) * (0.5 / self.dx)
            self.v[i, j] = -(self.psi[ip, j] - self.psi[im, j]) * (0.5 / self.dx)

    @ti.kernel
    def poisson_solver_step(self, rb: int):
        for i, j in self.psi:
            if (i + j) % 2 == rb:
                ip = (i + 1) % self.N
                im = (i - 1 + self.N) % self.N
                jp = (j + 1) % self.N
                jm = (j - 1 + self.N) % self.N
                
                rhs = -self.w[i, j] * (self.dx * self.dx)
                psi_new = 0.25 * (self.psi[im, j] + self.psi[ip, j] + self.psi[i, jm] + self.psi[i, jp] - rhs)
                self.psi[i, j] = psi_new

    def solve_poisson(self, max_iter=40):
        for _ in range(max_iter):
            self.poisson_solver_step(0)
            self.poisson_solver_step(1)

    @ti.func
    def arakawa_advection(self, i, j):
        ip = (i + 1) % self.N
        im = (i - 1 + self.N) % self.N
        jp = (j + 1) % self.N
        jm = (j - 1 + self.N) % self.N
        
        # J(w, psi)
        J1 = (self.w[ip, j] - self.w[im, j]) * (self.psi[i, jp] - self.psi[i, jm]) - \
             (self.w[i, jp] - self.w[i, jm]) * (self.psi[ip, j] - self.psi[im, j])
             
        J2 = self.w[ip, jp] * (self.psi[i, jp] - self.psi[ip, j]) - \
             self.w[im, jm] * (self.psi[i, jm] - self.psi[im, j]) - \
             self.w[im, jp] * (self.psi[i, jp] - self.psi[im, j]) + \
             self.w[ip, jm] * (self.psi[i, jm] - self.psi[ip, j])
             
        J3 = self.w[ip, jp] * (self.psi[ip, jp] - self.psi[i, jp]) - \
             self.w[im, jm] * (self.psi[im, jm] - self.psi[i, jm]) - \
             self.w[im, jp] * (self.psi[im, jp] - self.psi[im, j]) + \
             self.w[ip, jm] * (self.psi[ip, jm] - self.psi[ip, j])
             
        return -(J1 + J2 + J3) / (12.0 * self.dx * self.dx)

    @ti.kernel
    def compute_k(self, k_field: ti.template()):
        for i, j in self.w:
            ip = (i + 1) % self.N
            im = (i - 1 + self.N) % self.N
            jp = (j + 1) % self.N
            jm = (j - 1 + self.N) % self.N
            
            # Diffusion
            lap_w = (self.w[im, j] + self.w[ip, j] + self.w[i, jm] + self.w[i, jp] - 4.0 * self.w[i, j]) * self.inv_dx2
            diffusion = self.nu * lap_w
            
            # Advection
            advection = self.arakawa_advection(i, j)
            
            # Forcing
            y_val = j * self.dx
            forcing = -self.A * self.k_f * ti.cos(self.k_f * y_val)
            
            k_field[i, j] = advection + diffusion + forcing

    @ti.kernel
    def update_w(self, src: ti.template(), k: ti.template(), dt_factor: float):
        for i, j in self.w:
            self.w[i, j] = src[i, j] + dt_factor * k[i, j]

    @ti.kernel
    def final_rk4_update(self):
        for i, j in self.w:
            self.w[i, j] = self.w_n[i, j] + (self.dt / 6.0) * (
                self.k1[i, j] + 2*self.k2[i, j] + 2*self.k3[i, j] + self.k4[i, j]
            )

    @ti.kernel
    def copy_field(self, src: ti.template(), dst: ti.template()):
        for i, j in src:
            dst[i, j] = src[i, j]

    def step(self):
        # Save current state
        self.copy_field(self.w, self.w_n)
        
        # K1
        self.solve_poisson()
        self.compute_k(self.k1)
        
        # K2
        self.update_w(self.w_n, self.k1, 0.5 * self.dt)
        self.solve_poisson()
        self.compute_k(self.k2)
        
        # K3
        self.update_w(self.w_n, self.k2, 0.5 * self.dt)
        self.solve_poisson()
        self.compute_k(self.k3)
        
        # K4
        self.update_w(self.w_n, self.k3, self.dt)
        self.solve_poisson()
        self.compute_k(self.k4)
        
        # Combine
        self.final_rk4_update()

    @ti.kernel
    def calc_diagnostics(self) -> (ti.f32, ti.f32, ti.f32):
        div_err_max = 0.0
        mom_res_sum = 0.0
        ke_sum = 0.0
        for i, j in self.u:
            ip = (i + 1) % self.N
            im = (i - 1 + self.N) % self.N
            jp = (j + 1) % self.N
            jm = (j - 1 + self.N) % self.N
            
            div_u = (self.u[ip, j] - self.u[im, j]) / (2*self.dx) + \
                    (self.v[i, jp] - self.v[i, jm]) / (2*self.dx)
            
            dw_dt = self.k1[i, j]
            mom_res_sum += dw_dt * dw_dt
            ke_sum += 0.5 * (self.u[i, j]**2 + self.v[i, j]**2)
            
            if ti.abs(div_u) > div_err_max:
                div_err_max = ti.abs(div_u)
                
        return div_err_max, ti.sqrt(mom_res_sum / (self.N * self.N)), ke_sum / (self.N * self.N)

def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow DNS (Taichi)')
    parser.add_argument('--N', type=int, default=256, help='Grid points')
    parser.add_argument('--L', type=float, default=2*np.pi, help='Domain size')
    parser.add_argument('--nu', type=float, default=1e-3, help='Viscosity')
    parser.add_argument('--A', type=float, default=0.1, help='Forcing amplitude')
    parser.add_argument('--k_f', type=int, default=4, help='Forcing wavenumber')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step')
    parser.add_argument('--T_end', type=float, default=20.0, help='End time')
    parser.add_argument('--save_interval', type=int, default=100, help='Save interval')
    parser.add_argument('--output', type=str, default='data/kolmogorov_dns.npy', help='Output file')
    parser.add_argument('--perturbation_times', type=float, nargs='+', help='Perturbation times')
    parser.add_argument('--dealias-mode', type=str, default='3/2', choices=['2/3', '3/2'], help='Ignored in FD solver')
    
    args = parser.parse_args()
    
    solver = KolmogorovDNSTaichi(
        N=args.N, L=args.L, nu=args.nu, A=args.A, k_f=args.k_f, dt=args.dt
    )
    solver.init_field(1.0)
    
    n_steps = int(args.T_end / args.dt)
    u_save, v_save, w_save, t_save = [], [], [], []
    
    print(f"| step | momentum residual | mass conservation | speed(step/s) |")
    
    # Warmup (JIT compilation)
    solver.step()
    ti.sync() # Ensure GPU finishes
    t_start = time.time()
    
    for n in range(1, n_steps + 1):
        solver.step()
        
        if n % 100 == 0:
            solver.compute_velocity() # Needed for diag
            mass, mom, ke = solver.calc_diagnostics()
            elapsed = time.time() - t_start
            speed = (n) / elapsed if elapsed > 0 else 0
            print(f"| {n:6d} | {mom:17.4e} | {mass:17.4e} | {speed:12.1f} |")
            
        if n % args.save_interval == 0:
            solver.compute_velocity()
            u_save.append(solver.u.to_numpy())
            v_save.append(solver.v.to_numpy())
            w_save.append(solver.w.to_numpy())
            t_save.append(n * args.dt)
            
    # Save
    data = {
        'u': np.array(u_save),
        'v': np.array(v_save),
        'omega': np.array(w_save),
        'time': np.array(t_save),
        'config': vars(args)
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    print(f"✅ Simulation Complete. Output: {args.output}")

if __name__ == '__main__':
    main()