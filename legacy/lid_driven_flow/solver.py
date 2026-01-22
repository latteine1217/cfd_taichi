import taichi as ti
import numpy as np
import os
import time
import argparse

# --- CLI Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="2D LBM Lid-Driven Cavity Solver (Taichi MRT)")
    parser.add_argument("--res", type=int, default=256, help="Grid resolution (NX=NY). Default: 256")
    parser.add_argument("--steps", type=int, default=50000, help="Total simulation steps. Default: 50000")
    parser.add_argument("--interval", type=int, default=1000, help="Output interval (steps). Default: 1000")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number. Default: 1000.0")
    parser.add_argument("--lid_vel", type=float, default=0.1, help="Lid velocity in lattice units (keep < 0.3). Default: 0.1")
    parser.add_argument("--cs", type=float, default=0.16, help="Smagorinsky constant for LES (0 to disable). Default: 0.16")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory. Default: 'output'")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance for normalized residuals. Default: 1e-3")
    return parser.parse_args()

args = parse_args()

# Initialize Taichi with Metal backend and float32 precision
ti.init(arch=ti.metal, default_fp=ti.f32)

# --- Configuration ---
NX = args.res
NY = args.res
STEPS = args.steps
OUTPUT_INTERVAL = args.interval
RE = args.re
LID_VEL = args.lid_vel
CS = args.cs  # Smagorinsky Constant

# Derived parameters
NU = LID_VEL * NX / RE  # Kinematic viscosity
TAU = 3.0 * NU + 0.5  # Relaxation time (Base/Molecular tau)

OUTPUT_DIR = args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- Simulation Setup ---")
print(f"Resolution: {NX}x{NY}")
print(f"Reynolds Number: {RE}")
print(f"Lid Velocity: {LID_VEL}")
print(f"Viscosity: {NU:.6f}")
print(f"Relaxation Time (tau_0): {TAU:.6f}")
print(f"Smagorinsky Constant (Cs): {CS}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Steps: {STEPS}")
print(f"------------------------")

# --- LBM Constants (D2Q9) ---
# Weights
w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
e_np = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                 [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)

# Weights and velocities fields (still useful for some operations, but we use static loops)
w = ti.field(dtype=ti.f32, shape=9)
e = ti.Vector.field(2, dtype=ti.i32, shape=9)

# --- Data Fields ---
# f: distribution function (2 sets for double buffering)
# Optimization 1: SOA (Structure of Arrays) Layout: (9, NX, NY)
f = ti.field(dtype=ti.f32, shape=(9, NX, NY))
f_new = ti.field(dtype=ti.f32, shape=(9, NX, NY))

# Macroscopic variables (AOS is fine for these 2D fields)
rho = ti.field(dtype=ti.f32, shape=(NX, NY))
u = ti.Vector.field(2, dtype=ti.f32, shape=(NX, NY))

# Diagnostics
total_mass = ti.field(dtype=ti.f32, shape=())
mass_residual = ti.field(dtype=ti.f32, shape=())
prev_rho = ti.field(dtype=ti.f32, shape=(NX, NY)) 

# Separate Momentum Diagnostics
mom_res_x = ti.field(dtype=ti.f32, shape=())
mom_res_y = ti.field(dtype=ti.f32, shape=())
mom_scale_x = ti.field(dtype=ti.f32, shape=())
mom_scale_y = ti.field(dtype=ti.f32, shape=())
prev_u = ti.Vector.field(2, dtype=ti.f32, shape=(NX, NY)) 

# --- MRT Constants ---
# Transformation Matrix M (D2Q9)
M_np = np.array([
    [1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],
    [4, -2, -2, -2, -2,  1,  1,  1,  1],
    [0,  1,  0, -1,  0,  1, -1, -1,  1],
    [0, -2,  0,  2,  0,  1, -1, -1,  1],
    [0,  0,  1,  0, -1,  1,  1, -1, -1],
    [0,  0, -2,  0,  2,  1,  1, -1, -1],
    [0,  1, -1,  1, -1,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  1, -1,  1, -1]
], dtype=np.float32)

M_inv_np = np.linalg.inv(M_np).astype(np.float32)

# Optimization 2: Use fields for large matrices to avoid compilation overhead
M_field = ti.field(dtype=ti.f32, shape=(9, 9))
M_inv_field = ti.field(dtype=ti.f32, shape=(9, 9))

@ti.func
def mat_vec_mul(M: ti.template(), v: ti.template()):
    res = ti.Vector([0.0]*9)
    for r in ti.static(range(9)):
        val = 0.0
        for c in ti.static(range(9)):
            val += M[r, c] * v[c]
        res[r] = val
    return res

S = ti.field(dtype=ti.f32, shape=9)  # Relaxation parameters

@ti.kernel
def init_constants():
    # Initialize Relaxation Rates
    s_nu = 1.0 / TAU
    s_other = 1.2
    
    S[0] = 0.0  # rho
    S[1] = s_other # e
    S[2] = s_other # epsilon
    S[3] = 0.0  # jx
    S[4] = s_other # qx
    S[5] = 0.0  # jy
    S[6] = s_other # qy
    S[7] = s_nu # pxx
    S[8] = s_nu # pxy

@ti.kernel
def init_fields():
    for i, j in rho:
        rho[i, j] = 1.0
        u[i, j] = ti.Vector([0.0, 0.0])
        prev_rho[i, j] = 1.0
        prev_u[i, j] = ti.Vector([0.0, 0.0])
        
        u_vec = u[i, j]
        rho_val = rho[i, j]
        u_sq = u_vec.norm_sqr()
        for k in ti.static(range(9)):
            eu = e[k].dot(u_vec)
            # SOA Access: f[k, i, j]
            f[k, i, j] = w[k] * rho_val * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
            f_new[k, i, j] = f[k, i, j]

# Optimization 1 & 3: Separate kernel to update macroscopic fields only when needed, with loop unrolling
@ti.kernel
def update_macro(f_src: ti.template()):
    for i, j in ti.ndrange(NX, NY):
        current_rho = 0.0
        current_u = ti.Vector([0.0, 0.0])
        for k in ti.static(range(9)):
            val = f_src[k, i, j]
            current_rho += val
            current_u += val * e[k]
        current_u /= current_rho
        rho[i, j] = current_rho
        u[i, j] = current_u

# Optimization 2: Use Template arguments to swap buffers without copying
@ti.kernel
def collide_and_stream(f_src: ti.template(), f_dst: ti.template()):
    # MRT Collision & Streaming
    for i, j in ti.ndrange(NX, NY):
        # 1. Load populations (Gather from SOA f_src)
        f_vec = ti.Vector([0.0]*9)
        for k in ti.static(range(9)):
            f_vec[k] = f_src[k, i, j]

        # 2. Compute Macroscopic variables (Local only, no global write)
        current_rho = 0.0
        current_u = ti.Vector([0.0, 0.0])
        for k in ti.static(range(9)):
            current_rho += f_vec[k]
            current_u += f_vec[k] * e[k]
        current_u /= current_rho

        # Optimization: We DO NOT write rho[i, j] and u[i, j] here anymore.

        # 3. Transform to Moment Space: m = M * f
        # Optimization: Use field based matrix mul
        m = mat_vec_mul(M_field, f_vec)

        # 4. Compute Equilibrium Moments
        ux = current_u[0]
        uy = current_u[1]
        u_sq = ux**2 + uy**2
        
        meq = ti.Vector([0.0]*9)
        meq[0] = current_rho
        meq[1] = -2.0 * current_rho + 3.0 * current_rho * u_sq
        meq[2] = current_rho - 3.0 * current_rho * u_sq
        meq[3] = current_rho * ux
        meq[4] = -current_rho * ux
        meq[5] = current_rho * uy
        meq[6] = -current_rho * uy
        meq[7] = current_rho * (ux**2 - uy**2)
        meq[8] = current_rho * ux * uy

        # 5. Relax Moments
        m_star = ti.Vector([0.0]*9)
        
        noneq_pxx = m[7] - meq[7]
        noneq_pxy = m[8] - meq[8]

        s_nu = S[7] 
        
        if CS > 0.0:
            Q = ti.sqrt(noneq_pxx**2 + noneq_pxy**2)
            delta = 18.0 * (CS**2) * Q / (current_rho + 1e-9)
            tau_eff = 0.5 * (TAU + ti.sqrt(TAU*TAU + delta))
            s_nu = 1.0 / tau_eff

        for k in ti.static(range(9)):
            if k == 7 or k == 8:
                m_star[k] = m[k] - s_nu * (m[k] - meq[k])
            else:
                m_star[k] = m[k] - S[k] * (m[k] - meq[k])

        # 6. Transform back: f* = M_inv * m*
        # Optimization: Use field based matrix mul
        f_post = mat_vec_mul(M_inv_field, m_star)

        # 7. Streaming with Half-way Bounce-Back
        inv_map = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        for k in ti.static(range(9)):
            dest_i = i + e[k][0]
            dest_j = j + e[k][1]
            
            if dest_i >= 0 and dest_i < NX and dest_j >= 0 and dest_j < NY:
                # SOA Write to f_dst
                f_dst[k, dest_i, dest_j] = f_post[k]
            else:
                # Bounce-Back
                inv_k = inv_map[k]
                if dest_j == NY:
                    # Moving Wall Correction
                    x_norm = float(i) / (NX - 1)
                    arg1 = 10.0 * (x_norm - 0.5)
                    ch1 = 0.5 * (ti.exp(arg1) + ti.exp(-arg1))
                    ch2 = 0.5 * (ti.exp(5.0) + ti.exp(-5.0))
                    u_wall_mag = LID_VEL * (1.0 - ch1 / ch2)
                    
                    momentum_term = 6.0 * w[k] * current_rho * (e[k].dot(ti.Vector([u_wall_mag, 0.0])))
                    f_dst[inv_k, i, j] = f_post[k] - momentum_term
                else:
                    f_dst[inv_k, i, j] = f_post[k]

@ti.kernel
def update_diagnostics():
    total_mass[None] = 0.0
    mass_residual[None] = 0.0
    mom_res_x[None] = 0.0
    mom_res_y[None] = 0.0
    mom_scale_x[None] = 0.0
    mom_scale_y[None] = 0.0
    
    for i, j in rho:
        total_mass[None] += rho[i, j]
        
        rho_diff = ti.abs(rho[i, j] - prev_rho[i, j])
        mass_residual[None] += rho_diff
        prev_rho[i, j] = rho[i, j]
        
        u_val = u[i, j]
        u_prev = prev_u[i, j]
        
        mom_res_x[None] += ti.abs(u_val[0] - u_prev[0])
        mom_res_y[None] += ti.abs(u_val[1] - u_prev[1])
        
        mom_scale_x[None] += ti.abs(u_val[0])
        mom_scale_y[None] += ti.abs(u_val[1])
        
        prev_u[i, j] = u_val

def save_data(step):
    rho_npy = rho.to_numpy()
    u_npy = u.to_numpy()
    filename = os.path.join(OUTPUT_DIR, f"state_{step:06d}.npy")
    np.save(filename, {'rho': rho_npy, 'u': u_npy})

def main():
    w.from_numpy(w_np)
    e.from_numpy(e_np)
    M_field.from_numpy(M_np)
    M_inv_field.from_numpy(M_inv_np)
    init_constants()
    init_fields()

    print(f"|{'step':<10}|{'R_u':<12}|{'R_v':<12}|{'R_rho':<12}|{'speed(step/s)':<15}|")
    print(f"|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*15}|")

    ti.sync()
    start_time = time.time()

    for step in range(1, STEPS + 1):
        # Optimization: Swap pointers instead of copying memory
        source_f = f if step % 2 == 1 else f_new
        dest_f = f_new if step % 2 == 1 else f
        
        collide_and_stream(source_f, dest_f)

        if step % 100 == 0:
            # We need rho/u for diagnostics, so update them now
            update_macro(dest_f)
            
            ti.sync()
            end_time = time.time()
            elapsed = end_time - start_time
            
            speed = 100.0 / elapsed if elapsed > 1e-6 else 0.0
            
            update_diagnostics()
            
            res_u = mom_res_x[None] / (mom_scale_x[None] + 1e-12)
            res_v = mom_res_y[None] / (mom_scale_y[None] + 1e-12)
            res_rho = mass_residual[None] / (total_mass[None] + 1e-12)
            
            print(f"|{step:<10}|{res_u:<12.4e}|{res_v:<12.4e}|{res_rho:<12.4e}|{speed:<15.2f}|")
            
            if res_u < args.tol and res_v < args.tol and res_rho < args.tol:
                print(f"---------------------------------------------------------------")
                print(f"Converged at step {step} (Tolerance: {args.tol})")
                save_data(step)
                break
            
            ti.sync() 
            start_time = time.time()
        
        elif step % OUTPUT_INTERVAL == 0:
            # Also update macro if we are saving (and didn't just do it for diag)
            update_macro(dest_f)
            save_data(step)

if __name__ == "__main__":
    main()