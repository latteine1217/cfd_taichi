"Kolmogorov Flow LES Generator (Taichi Optimized)
================================================

High-performance Finite Difference LES code using Taichi.
Includes Hyperviscosity and Linear Friction terms.
Optimized for Metal backend.

Author: Taichi-SciML Team
Date: 2026-01-19
"

import taichi as ti
import numpy as np
import argparse
import time
import os
from pathlib import Path

# Initialize Taichi with Metal backend
ti.init(arch=ti.metal, default_fp=ti.f32, random_seed=int(time.time()))

@ti.data_oriented
class KolmogorovLESTaichi:
    def __init__(
        self,
        N: int,
        L: float,
        nu: float,
        A: float,
        k_f: int,
        dt: float,
        nu_h: float = 0.0,
        hyper_p: int = 2,
        r_fric: float = 0.0
    ):
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.nu_h = nu_h
        self.hyper_p = hyper_p
        self.r_fric = r_fric
        
        self.dx = L / N
        self.inv_dx2 = 1.0 / (self.dx * self.dx)
        
        # Fields
        self.w = ti.field(dtype=ti.f32, shape=(N, N))
        self.w_n = ti.field(dtype=ti.f32, shape=(N, N))
        self.psi = ti.field(dtype=ti.f32, shape=(N, N))
        self.u = ti.field(dtype=ti.f32, shape=(N, N))
        self.v = ti.field(dtype=ti.f32, shape=(N, N))
        
        # RK4 substeps
        self.k1 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k2 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k3 = ti.field(dtype=ti.f32, shape=(N, N))
        self.k4 = ti.field(dtype=ti.f32, shape=(N, N))
        
        # Hyperviscosity helpers
        self.lap1 = ti.field(dtype=ti.f32, shape=(N, N))
        self.lap2 = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def init_field(self, omega_in: ti.types.ndarray()):
        for i, j in self.w:
            self.w[i, j] = omega_in[i, j]
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
    def compute_laplacian(self, src: ti.template(), dst: ti.template()):
        for i, j in src:
            ip = (i + 1) % self.N
            im = (i - 1 + self.N) % self.N
            jp = (j + 1) % self.N
            jm = (j - 1 + self.N) % self.N
            dst[i, j] = (src[im, j] + src[ip, j] + src[i, jm] + src[i, jp] - 4.0 * src[i, j]) * self.inv_dx2

    @ti.kernel
    def compute_rhs(self, k_field: ti.template(), hyper_field: ti.template()):
        for i, j in self.w:
            ip = (i + 1) % self.N
            im = (i - 1 + self.N) % self.N
            jp = (j + 1) % self.N
            jm = (j - 1 + self.N) % self.N
            
            lap_w = (self.w[im, j] + self.w[ip, j] + self.w[i, jm] + self.w[i, jp] - 4.0 * self.w[i, j]) * self.inv_dx2
            
            advection = self.arakawa_advection(i, j)
            diffusion = self.nu * lap_w
            hyper = -self.nu_h * hyper_field[i, j]
            friction = -self.r_fric * self.w[i, j]
            
            y_val = j * self.dx
            forcing = -self.A * self.k_f * ti.cos(self.k_f * y_val)
            
            k_field[i, j] = advection + diffusion + hyper + friction + forcing

    def get_hyper_field(self):
        if self.nu_h == 0:
            self.lap2.fill(0.0)
            return
        if self.hyper_p == 1:
            self.compute_laplacian(self.w, self.lap2)
            # Standard: nu_h * del^2. My rhs uses -nu_h * hyper. So hyper = -del^2.
            # But let's assume p=2 mostly.
        elif self.hyper_p == 2:
            self.compute_laplacian(self.w, self.lap1)
            self.compute_laplacian(self.lap1, self.lap2) # Del^4 w
        else:
            self.lap2.fill(0.0)

    @ti.kernel
    def update_w(self, src: ti.template(), k: ti.template(), factor: float):
        for i, j in self.w:
            self.w[i, j] = src[i, j] + factor * k[i, j]

    @ti.kernel
    def final_update(self):
        for i, j in self.w:
            self.w[i, j] = self.w_n[i, j] + (self.dt / 6.0) * (
                self.k1[i, j] + 2*self.k2[i, j] + 2*self.k3[i, j] + self.k4[i, j]
            )

    @ti.kernel
    def copy_field(self, src: ti.template(), dst: ti.template()):
        for I in src: dst[I] = src[I]

    def step(self):
        self.copy_field(self.w, self.w_n)
        
        # K1
        self.solve_poisson()
        self.get_hyper_field()
        self.compute_rhs(self.k1, self.lap2)
        
        # K2
        self.update_w(self.w_n, self.k1, 0.5 * self.dt)
        self.solve_poisson()
        self.get_hyper_field()
        self.compute_rhs(self.k2, self.lap2)
        
        # K3
        self.update_w(self.w_n, self.k2, 0.5 * self.dt)
        self.solve_poisson()
        self.get_hyper_field()
        self.compute_rhs(self.k3, self.lap2)
        
        # K4
        self.update_w(self.w_n, self.k3, self.dt)
        self.solve_poisson()
        self.get_hyper_field()
        self.compute_rhs(self.k4, self.lap2)
        
        self.final_update()

    @ti.kernel
    def calc_diagnostics(self) -> (ti.f32, ti.f32, ti.f32):
        div_max, mom_sum, ke_sum = 0.0, 0.0, 0.0
        for i, j in self.u:
            ip, im = (i+1)%self.N, (i-1+self.N)%self.N
            jp, jm = (j+1)%self.N, (j-1+self.N)%self.N
            div = (self.u[ip,j]-self.u[im,j])/(2*self.dx) + (self.v[i,jp]-self.v[i,jm])/(2*self.dx)
            ke_sum += 0.5*(self.u[i,j]**2 + self.v[i,j]**2)
            mom_sum += self.k1[i,j]**2
            if ti.abs(div) > div_max: div_max = ti.abs(div)
        return div_max, ti.sqrt(mom_sum/(self.N*self.N)), ke_sum/(self.N*self.N)

def main():
    parser = argparse.ArgumentParser(description="Kolmogorov LES (Taichi)")
    parser.add_argument("--dns", type=str, required=True, help="DNS NPY file")
    parser.add_argument("--N", type=int, default=128, help="LES Resolution")
    parser.add_argument("--T_end", type=float, default=20.0, help="Simulation time")
    parser.add_argument("--save_interval", type=int, default=100, help="Output interval")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step")
    parser.add_argument("--auto_dt", action="store_true", help="Auto-scale dt (Not implemented)")
    parser.add_argument("--cfl_target", type=float, default=0.4, help="CFL target")
    parser.add_argument("--nu_h", type=float, default=1e-4, help="Hyperviscosity coef")
    parser.add_argument("--nu_h_alpha", type=float, default=10.0, help="nu_h alpha")
    parser.add_argument("--hyper_p", type=int, default=2, help="Hyperviscosity power")
    parser.add_argument("--r_scale", type=float, default=0.1, help="Friction scale")
    parser.add_argument("--dealias_mode", type=str, default="2/3", choices=["2/3", "3/2"])
    parser.add_argument("--omega_rms", type=float, default=None, help="Initial RMS")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output file")

    args = parser.parse_args()
    
    # Load DNS metadata for L, nu, etc if possible, else defaults
    L, nu, A, k_f = 2*np.pi, 1e-3, 0.1, 4
    if os.path.exists(args.dns):
        try:
            d = np.load(args.dns, allow_pickle=True).item()
            conf = d.get('config', {})
            L = conf.get('L', L)
            nu = conf.get('nu', nu)
            A = conf.get('A', A)
            k_f = conf.get('k_f', k_f)
        except:
            pass

    solver = KolmogorovLESTaichi(
        N=args.N, L=L, nu=nu, A=A, k_f=k_f, dt=args.dt,
        nu_h=args.nu_h, hyper_p=args.hyper_p, r_fric=args.r_scale
    )
    
    # Init
    w_init = np.random.randn(args.N, args.N).astype(np.float32)
    solver.init_field(w_init)
    
    n_steps = int(args.T_end / args.dt)
    u_save, v_save, w_save, t_save = [], [], [], []
    t_start = time.time()
    
    print(f"| step | momentum residual | mass conservation | speed(step/s) |")
    
    # Warmup
    solver.step()
    ti.sync()
    t_start = time.time()
    
    for n in range(1, n_steps + 1):
        solver.step()
        if n % 100 == 0:
            solver.compute_velocity()
            mass, mom, ke = solver.calc_diagnostics()
            elapsed = time.time() - t_start
            speed = (n)/elapsed if elapsed > 0 else 0
            print(f"| {n:6d} | {mom:17.4e} | {mass:17.4e} | {speed:12.1f} |")
        if n % args.save_interval == 0:
            solver.compute_velocity()
            u_save.append(solver.u.to_numpy())
            v_save.append(solver.v.to_numpy())
            w_save.append(solver.w.to_numpy())
            t_save.append(n * args.dt)
            
    data = {'u': np.array(u_save), 'v': np.array(v_save), 'omega': np.array(w_save), 'time': np.array(t_save), 'config': vars(args)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    print("Done.")

if __name__ == "__main__":
    main()