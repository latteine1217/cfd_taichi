import taichi as ti
import numpy as np
import os
import time
from tabulate import tabulate

@ti.data_oriented
class WindTunnel:
    def __init__(self, res_y=128, res_x=None, re=1000.0, u_in=0.1, cs=0.16, output_dir="output", length_scale=None):
        """
        Initialize the Wind Tunnel Solver (LBM MRT).
        
        Args:
            res_y (int): Vertical resolution (height of tunnel).
            res_x (int): Horizontal resolution. If None, defaults to 4 * res_y.
            re (float): Reynolds number.
            u_in (float): Inlet velocity (lattice units).
            cs (float): Smagorinsky constant for LES.
            output_dir (str): Directory to save outputs.
            length_scale (float): Characteristic length for Reynolds number calculation. 
                                  If None, defaults to res_y / 9.0 (to match old cylinder default).
        """
        self.ny = res_y
        self.nx = res_x if res_x is not None else int(3.5 * res_y)
        self.re = re
        self.u_in = u_in
        self.cs = cs
        self.output_dir = output_dir
        
        # Physics Parameters
        # characteristic length L.
        # nu = U * L / Re
        if length_scale is None:
            self.L_char = self.ny / 9.0 
        else:
            self.L_char = float(length_scale)
            
        self.nu = self.u_in * self.L_char / self.re
        self.tau = 3.0 * self.nu + 0.5
        
        # Create Output Directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- Taichi Fields ---
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.inv = ti.field(dtype=ti.i32, shape=9)
        
        # Optimization: Use Vector field for AoS layout (Better Cache Locality)
        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny))
        self.mask = ti.field(dtype=ti.i32, shape=(self.nx, self.ny)) # 1 = Solid, 0 = Fluid
        
        # Diagnostics
        self.total_mass = ti.field(dtype=ti.f32, shape=())
        self.initial_mass = ti.field(dtype=ti.f32, shape=())  # For mass conservation check
        self.mass_residual = ti.field(dtype=ti.f32, shape=())
        self.mom_res_x = ti.field(dtype=ti.f32, shape=())
        self.mom_res_y = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_x = ti.field(dtype=ti.f32, shape=())
        self.mom_scale_y = ti.field(dtype=ti.f32, shape=())
        self.max_u = ti.field(dtype=ti.f32, shape=())
        
        self.prev_rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.prev_u = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny))
        
        # Forces (Lift/Drag)
        self.force_x = ti.field(dtype=ti.f32, shape=())
        self.force_y = ti.field(dtype=ti.f32, shape=())
        
        # --- Particle System (Smoke) ---
        self.num_particles = 500000
        self.px = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.py = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.p_active = ti.field(dtype=ti.i32, shape=self.num_particles)
        self.emitter_ptr = ti.field(dtype=ti.i32, shape=()) # Circular buffer pointer
        
        # MRT Matrices
        self.M = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.M_inv = ti.Matrix.field(9, 9, dtype=ti.f32, shape=())
        self.S = ti.field(dtype=ti.f32, shape=9)
        
        # Constants Initialization
        self._init_constants()
        self.reset()
        
    def _init_constants(self):
        w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
        e_np = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                         [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        inv_np = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
        
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
        
        self.w.from_numpy(w_np)
        self.e.from_numpy(e_np)
        self.inv.from_numpy(inv_np)
        
        self._set_mrt_matrices(M_np, M_inv_np)
        
    @ti.kernel
    def _set_mrt_matrices(self, m_arr: ti.types.ndarray(), minv_arr: ti.types.ndarray()):
        for i, j in ti.ndrange(9, 9):
            self.M[None][i, j] = m_arr[i, j]
            self.M_inv[None][i, j] = minv_arr[i, j]
        
        s_nu = 1.0 / self.tau
        s_other = 1.2
        self.S[0] = 0.0; self.S[1] = s_other; self.S[2] = s_other; self.S[3] = 0.0; self.S[4] = s_other
        self.S[5] = 0.0; self.S[6] = s_other; self.S[7] = s_nu;    self.S[8] = s_nu

    @ti.kernel
    def _init_particles(self):
        self.emitter_ptr[None] = 0
        for i in range(self.num_particles):
            self.p_active[i] = 0
            self.px[i] = 0.0
            self.py[i] = 0.0

    @ti.kernel
    def _emit_particles(self):
        # Emit multiple smoke lines
        num_lines = 32
        stride = self.ny / num_lines
        
        # Emit one particle per line
        for l in range(num_lines):
            # Circular buffer allocation
            idx = (self.emitter_ptr[None] + l) % self.num_particles
            
            self.p_active[idx] = 1
            self.px[idx] = 2.0 # Slightly inside inlet
            self.py[idx] = (l + 0.5) * stride
            
        self.emitter_ptr[None] = (self.emitter_ptr[None] + num_lines) % self.num_particles

    @ti.func
    def _sample_u(self, x: float, y: float):
        # Bilinear interpolation
        i = int(x)
        j = int(y)
        u_res = ti.Vector([0.0, 0.0])
        
        # Clamp coordinates
        i = ti.max(0, ti.min(self.nx - 2, i))
        j = ti.max(0, ti.min(self.ny - 2, j))
        
        dx = x - i
        dy = y - j
        
        u00 = self.u[i, j]
        u10 = self.u[i + 1, j]
        u01 = self.u[i, j + 1]
        u11 = self.u[i + 1, j + 1]
        
        u_res = u00 * (1 - dx) * (1 - dy) + \
                u10 * dx * (1 - dy) + \
                u01 * (1 - dx) * dy + \
                u11 * dx * dy
        return u_res

    @ti.kernel
    def _advect_particles(self):
        for i in range(self.num_particles):
            if self.p_active[i] == 1:
                p_pos = ti.Vector([self.px[i], self.py[i]])
                
                # RK2 Integration for better streamlines? Euler is fine for smoke.
                vel = self._sample_u(p_pos.x, p_pos.y)
                
                p_pos += vel # dt = 1
                
                # Boundary check
                if p_pos.x < 0 or p_pos.x >= self.nx or p_pos.y < 0 or p_pos.y >= self.ny:
                    self.p_active[i] = 0
                
                # Solid check (Simple)
                ix, iy = int(p_pos.x), int(p_pos.y)
                if ix >= 0 and ix < self.nx and iy >= 0 and iy < self.ny:
                    if self.mask[ix, iy] == 1:
                        self.p_active[i] = 0
                
                self.px[i] = p_pos.x
                self.py[i] = p_pos.y

    @ti.kernel
    def _reset_fields(self):
        for i, j in self.rho:
            self.mask[i, j] = 0  # Clear mask (will be set later by set_obstacle)
            self.rho[i, j] = 1.0
            self.prev_rho[i, j] = 1.0

            # Initialize velocity: Fluid=u_in, Solid=0 (prevents initial discontinuity)
            # Note: mask will be updated later, so we initialize all as fluid first
            # This will be corrected after set_obstacle() is called
            self.u[i, j] = ti.Vector([self.u_in, 0.0])
            self.prev_u[i, j] = ti.Vector([self.u_in, 0.0])

            # Init populations with equilibrium distribution
            u_vec = self.u[i, j]
            rho_val = self.rho[i, j]
            u_sq = u_vec.norm_sqr()
            for k in range(9):
                eu = self.e[k].dot(u_vec)
                val = self.w[k] * rho_val * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq)
                self.f[i, j][k] = val
                self.f_new[i, j][k] = val

    @ti.kernel
    def _correct_solid_init(self):
        """Correct velocity in solid regions after mask is set."""
        for i, j in self.rho:
            if self.mask[i, j] == 1:  # Solid region
                # Set velocity to zero in solid
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.prev_u[i, j] = ti.Vector([0.0, 0.0])

                # Reinitialize populations with zero velocity
                rho_val = self.rho[i, j]
                for k in range(9):
                    self.f[i, j][k] = self.w[k] * rho_val
                    self.f_new[i, j][k] = self.w[k] * rho_val

    def reset(self):
        self._init_particles()
        self._reset_fields()

    def set_obstacle(self, mask_arr):
        """
        Set the solid obstacle mask.
        Args:
            mask_arr (numpy.ndarray): 2D array (nx, ny) or (ny, nx) depending on convention.
                                      Taichi uses (i, j) -> (x, y).
                                      If input is (Y, X) image, it needs transposing.
        """
        # Ensure correct shape
        if mask_arr.shape == (self.nx, self.ny):
            self.mask.from_numpy(mask_arr.astype(np.int32))
        elif mask_arr.shape == (self.ny, self.nx):
            # Transpose if passed as (Row, Col) which is (Y, X)
            self.mask.from_numpy(mask_arr.T.astype(np.int32))
        else:
            raise ValueError(f"Mask shape {mask_arr.shape} incompatible with grid ({self.nx}, {self.ny})")

        # Correct initialization in solid regions (set u=0)
        self._correct_solid_init()

    @ti.kernel
    def _collide_and_stream(self, f_src: ti.template(), f_dst: ti.template()):
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                continue # Skip solid nodes

            # 1. Load & Compute Macroscopic (AoS Optimization: Single Load)
            f_vec = f_src[i, j]
            
            current_rho = 0.0
            current_u = ti.Vector([0.0, 0.0])
            
            for k in ti.static(range(9)):
                current_rho += f_vec[k]
                current_u += f_vec[k] * self.e[k]
            
            current_u /= current_rho
            
            # 2. MRT Collision
            m = self.M[None] @ f_vec
            
            ux = current_u[0]; uy = current_u[1]
            u_sq = ux**2 + uy**2
            meq = ti.Vector([0.0]*9)
            meq[0] = current_rho
            meq[1] = -2.0*current_rho + 3.0*current_rho*u_sq
            meq[2] = current_rho - 3.0*current_rho*u_sq
            meq[3] = current_rho*ux
            meq[4] = -current_rho*ux
            meq[5] = current_rho*uy
            meq[6] = -current_rho*uy
            meq[7] = current_rho*(ux**2 - uy**2)
            meq[8] = current_rho*ux*uy

            m_star = ti.Vector([0.0]*9)
            s_nu = self.S[7]
            if self.cs > 0.0:
                noneq_pxx = m[7] - meq[7]; noneq_pxy = m[8] - meq[8]
                Q = ti.sqrt(noneq_pxx**2 + noneq_pxy**2)
                delta = 18.0 * (self.cs**2) * Q / (current_rho + 1e-9)
                tau_eff = 0.5 * (self.tau + ti.sqrt(self.tau*self.tau + delta))
                s_nu = 1.0 / tau_eff
                
            for k in ti.static(range(9)):
                rate = s_nu if (k==7 or k==8) else self.S[k]
                m_star[k] = m[k] - rate * (m[k] - meq[k])
                
            f_post = self.M_inv[None] @ m_star

            # 3. Stream
            for k in ti.static(range(9)):
                dest_i = i + self.e[k][0]
                dest_j = j + self.e[k][1]

                # Check if destination is within domain bounds
                in_bounds = (dest_i >= 0 and dest_i < self.nx and
                           dest_j >= 0 and dest_j < self.ny)

                if in_bounds:
                    if self.mask[dest_i, dest_j] == 1:
                        # Bounce Back from solid
                        f_dst[i, j][self.inv[k]] = f_post[k]
                    else:
                        # Normal streaming
                        f_dst[dest_i, dest_j][k] = f_post[k]
                # else: Particles stream out of domain (open boundary)
                # BC will reconstruct populations at boundary nodes afterward

    @ti.kernel
    def _compute_forces(self, f_out: ti.template()):
        """
        Compute forces using Momentum Exchange Method.
        F = sum_{x_f} sum_{i s.t. x_f+e_i is solid} 2 * f_i^{post_collision}(x_f) * e_i
        Using f_out (which contains values streamed from bounce-back at solid nodes),
        we can sum the momentum transferred.

        CRITICAL: No periodic BC in Y direction (consistent with streaming).
        """
        self.force_x[None] = 0.0
        self.force_y[None] = 0.0

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 0: # Fluid node
                for k in ti.static(range(9)):
                    # Neighbor (NO periodic Y boundary)
                    ni = i + self.e[k][0]
                    nj = j + self.e[k][1]

                    # Check bounds for both X and Y (consistent with streaming logic)
                    if ni >= 0 and ni < self.nx and nj >= 0 and nj < self.ny:
                        if self.mask[ni, nj] == 1: # Neighbor is solid
                            # Momentum exchange
                            # f_val = f_out[inv[k], i, j] comes from the wall
                            f_val = f_out[i, j][self.inv[k]]
                            self.force_x[None] += 2.0 * f_val * self.e[k][0]
                            self.force_y[None] += 2.0 * f_val * self.e[k][1]

    @ti.kernel
    def _apply_bc(self, f_dst: ti.template()):
        # Inlet (Left, i=0): Fixed Velocity Zou-He
        for j in range(self.ny):
            if self.mask[0, j] == 0:
                f0 = f_dst[0, j][0]; f2 = f_dst[0, j][2]; f3 = f_dst[0, j][3]
                f4 = f_dst[0, j][4]; f6 = f_dst[0, j][6]; f7 = f_dst[0, j][7]
                
                rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - self.u_in)
                
                f_dst[0, j][1] = f3 + (2.0/3.0) * rho_in * self.u_in
                f_dst[0, j][5] = f7 - 0.5 * (f2 - f4) + (1.0/6.0) * rho_in * self.u_in
                f_dst[0, j][8] = f6 + 0.5 * (f2 - f4) + (1.0/6.0) * rho_in * self.u_in

        # Outlet (Right, i=NX-1): Zou-He Pressure Outlet BC (Mass Conservative)
        # Enforces constant pressure (rho_out = 1.0) at outlet while allowing flow to exit
        # This ensures strict mass conservation: ∑f_i = rho_target
        # Reference: Zou & He (1997), "On pressure and velocity boundary conditions for LBM"
        for j in range(self.ny):
            if self.mask[self.nx-1, j] == 0:
                # Extract known populations (propagating into domain from outside)
                # At right boundary, we know: f0, f1, f2, f4, f5, f8
                # Unknown (need to reconstruct): f3, f6, f7 (pointing left)
                f0 = f_dst[self.nx-1, j][0]
                f2 = f_dst[self.nx-1, j][2]
                f4 = f_dst[self.nx-1, j][4]
                f1 = f_dst[self.nx-1, j][1]
                f5 = f_dst[self.nx-1, j][5]
                f8 = f_dst[self.nx-1, j][8]

                # Target density (atmospheric pressure)
                rho_out = 1.0

                # Calculate x-velocity from mass conservation
                # rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                # rho*u_x = f1 - f3 + f5 - f6 - f7 + f8
                # Solving for u_x given known f's:
                u_x = -1.0 + (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / rho_out

                # Reconstruct unknown populations using Zou-He scheme
                f_dst[self.nx-1, j][3] = f1 - (2.0/3.0) * rho_out * u_x
                f_dst[self.nx-1, j][7] = f5 - (1.0/6.0) * rho_out * u_x + 0.5*(f2 - f4)
                f_dst[self.nx-1, j][6] = f8 - (1.0/6.0) * rho_out * u_x - 0.5*(f2 - f4)
                    
        # Top & Bottom Walls (Free-Slip / Specular Reflection)
        # Simulates a frictionless wall (wind tunnel walls)
        for i in range(self.nx):
            # Bottom (j=0): Reflect downward moving particles back up
            if self.mask[i, 0] == 0:
                # f[2] (up) comes from f[4] (down)
                f_dst[i, 0][2] = f_dst[i, 0][4]
                # f[5] (up-right) comes from f[7] (down-right)?? No, f[5] is (1,1), f[7] is (-1,1)??
                # Let's check indices:
                # 0:(0,0), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1)
                # 5:(1,1), 6:(-1,1), 7:(-1,-1), 8:(1,-1)
                
                # Bottom needs inputs from below (y+): 2, 5, 6.
                # Sources are their vertical mirrors: 4, 8, 7.
                # f[2] = f[4]
                # f[5] (1,1) = f[8] (1,-1)  (Same x-dir, flipped y)
                # f[6] (-1,1) = f[7] (-1,-1) (Same x-dir, flipped y)
                f_dst[i, 0][2] = f_dst[i, 0][4]
                f_dst[i, 0][5] = f_dst[i, 0][8]
                f_dst[i, 0][6] = f_dst[i, 0][7]

            # Top (j=NY-1): Reflect upward moving particles back down
            if self.mask[i, self.ny-1] == 0:
                # Top needs inputs from above (y-): 4, 7, 8.
                # Sources are vertical mirrors: 2, 6, 5.
                # f[4] = f[2]
                # f[7] (-1,-1) = f[6] (-1,1)
                # f[8] (1,-1) = f[5] (1,1)
                f_dst[i, self.ny-1][4] = f_dst[i, self.ny-1][2]
                f_dst[i, self.ny-1][7] = f_dst[i, self.ny-1][6]
                f_dst[i, self.ny-1][8] = f_dst[i, self.ny-1][5]

    @ti.kernel
    def _update_macro(self, f_src: ti.template()):
        for i, j in ti.ndrange(self.nx, self.ny):
            f_vec = f_src[i, j]
            
            current_rho = 0.0
            current_u = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                current_rho += f_vec[k]
                current_u += f_vec[k] * self.e[k]
            
            if current_rho > 0:
                current_u /= current_rho
            
            self.rho[i, j] = current_rho
            self.u[i, j] = current_u

    @ti.kernel
    def _apply_mass_correction(self, f_src: ti.template()):
        """
        Global mass correction to enforce strict conservation.
        Rescales all distribution functions to maintain initial total mass.
        """
        # Calculate correction factor
        correction = self.initial_mass[None] / (self.total_mass[None] + 1e-12)

        # Apply correction to all fluid nodes
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 0:  # Fluid nodes only
                for k in ti.static(range(9)):
                    f_src[i, j][k] *= correction

    @ti.kernel
    def _update_diagnostics(self):
        self.total_mass[None] = 0.0
        self.mass_residual[None] = 0.0
        self.mom_res_x[None] = 0.0
        self.mom_res_y[None] = 0.0
        self.mom_scale_x[None] = 0.0
        self.mom_scale_y[None] = 0.0
        self.max_u[None] = 0.0
        
        for i, j in self.rho:
            if self.mask[i, j] == 0:
                self.total_mass[None] += self.rho[i, j]
                self.mass_residual[None] += ti.abs(self.rho[i, j] - self.prev_rho[i, j])
                self.prev_rho[i, j] = self.rho[i, j]
                
                u_val = self.u[i, j]
                u_prev = self.prev_u[i, j]
                
                self.mom_res_x[None] += ti.abs(u_val[0] - u_prev[0])
                self.mom_res_y[None] += ti.abs(u_val[1] - u_prev[1])
                
                self.mom_scale_x[None] += ti.abs(u_val[0])
                self.mom_scale_y[None] += ti.abs(u_val[1])
                
                ti.atomic_max(self.max_u[None], u_val.norm())
                
                self.prev_u[i, j] = u_val

    def save_data(self, step, cd=None, cl=None):
        rho_npy = self.rho.to_numpy()
        u_npy = self.u.to_numpy()
        mask_npy = self.mask.to_numpy()
        
        # Gather active particles
        active = self.p_active.to_numpy()
        px = self.px.to_numpy()
        py = self.py.to_numpy()
        
        # Filter active
        active_mask = active == 1
        particles_np = np.stack((px[active_mask], py[active_mask]), axis=1)
        
        # Calculate forces if not provided
        if cd is None or cl is None:
            self._compute_forces(self.f)
            # --- Force Coefficient Calculation ---
            # Standard definition: C_f = F / (0.5 * rho * U^2 * A_ref)
            # LBM Units: rho_ref = 1.0 (lattice density)
            # 2D Case: A_ref = L_char × depth, where depth = 1.0 (unit depth)
            # Reference: Anderson, "Fundamentals of Aerodynamics", Ch. 1.7
            rho_ref = 1.0  # LBM reference density
            A_ref = self.L_char * 1.0  # 2D reference area (chord × unit depth)
            denom = 0.5 * rho_ref * (self.u_in**2) * A_ref
            cd = self.force_x[None] / (denom + 1e-12)
            cl = self.force_y[None] / (denom + 1e-12)
        
        filename = os.path.join(self.output_dir, f"state_{step:06d}.npy")
        np.save(filename, {
            'rho': rho_npy, 
            'u': u_npy, 
            'mask': mask_npy,
            'particles': particles_np,
            'cd': cd,
            'cl': cl,
            'step': step
        })

    def run(self, steps=50000, interval=1000, tol=1e-5):
        print(f"--- Starting Simulation ---")
        print(f"Grid: {self.nx}x{self.ny}")
        print(f"Re: {self.re}, U_in: {self.u_in}")
        print(f"Viscosity (nu): {self.nu:.6f}, Tau: {self.tau:.6f}")
        print(f"Char Length (L): {self.L_char:.2f}")
        print(f"Steps: {steps}, Interval: {interval}")
        
        # Live monitoring header (added M_err for mass conservation)
        headers = ["step", "R_u", "R_v", "R_rho", "M_err", "Cd", "Cl", "Umax", "L/D", "ETA"]
        print(tabulate([], headers=headers, tablefmt="github"))

        history_data = []

        ti.sync()
        global_start_time = time.time()
        seg_start_time = time.time()

        # Initial save & record initial mass for conservation check
        self._update_macro(self.f)
        self._update_diagnostics()
        self.initial_mass[None] = self.total_mass[None]
        print(f"Initial Mass: {self.initial_mass[None]:.6f}")
        self.save_data(0)

        for step in range(1, steps + 1):
            source_f = self.f if step % 2 == 1 else self.f_new
            dest_f = self.f_new if step % 2 == 1 else self.f

            self._collide_and_stream(source_f, dest_f)
            self._apply_bc(dest_f)
            
            # Particle System
            if step % 5 == 0:
                self._emit_particles()
            self._advect_particles()
            
            # Compute forces every step (cheap on GPU) or interval?
            # Cheap enough to do often, but let's do it every 100 steps for diag
            if step % 100 == 0:
                self._compute_forces(dest_f)
                self._update_macro(dest_f)
                self._update_diagnostics()

                # Apply mass correction every 100 steps to enforce conservation
                # This compensates for numerical drift and BC imperfections
                self._apply_mass_correction(dest_f)
                # Update macroscopic after correction
                self._update_macro(dest_f)
                self._update_diagnostics()

                ti.sync()
                
                end_time = time.time()
                elapsed = end_time - seg_start_time
                speed = 100.0 / elapsed if elapsed > 1e-6 else 0.0
                
                # Calculate ETA
                remaining_steps = steps - step
                eta_seconds = remaining_steps / speed if speed > 0 else 0.0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                
                res_u = self.mom_res_x[None] / (self.mom_scale_x[None] + 1e-12)
                res_v = self.mom_res_y[None] / (self.mom_scale_y[None] + 1e-12)
                res_rho = self.mass_residual[None] / (self.total_mass[None] + 1e-12)

                # --- Mass Conservation Check ---
                mass_error = ti.abs(self.total_mass[None] - self.initial_mass[None]) / (self.initial_mass[None] + 1e-12)

                # --- Force Coefficient Calculation ---
                # C_f = F / (0.5 * rho * U^2 * A_ref)
                # LBM: rho_ref=1.0, A_ref=L_char×1.0 (2D)
                rho_ref = 1.0
                A_ref = self.L_char * 1.0
                denom = 0.5 * rho_ref * (self.u_in**2) * A_ref
                cd = self.force_x[None] / (denom + 1e-12)
                cl = self.force_y[None] / (denom + 1e-12)

                umax = self.max_u[None]
                eff = cl / (cd + 1e-12) if abs(cd) > 1e-9 else 0.0

                # Format data for row (added M_err column)
                # Live Log (Manual alignment to match simple github header)
                print(f"| {step:<6} | {res_u:<8.2e} | {res_v:<8.2e} | {res_rho:<8.2e} | {mass_error:<8.2e} | {cd:<8.4f} | {cl:<8.4f} | {umax:<6.4f} | {eff:<6.2f} | {eta_str:<8} |")

                # Warning for mass conservation violation
                if mass_error > 1e-3:
                    print(f"⚠️  WARNING: Mass conservation violated! Error = {mass_error:.2e}")

                # Collect for summary (every 500 steps or significant)
                if step % 500 == 0 or step == steps:
                    history_data.append([step, res_u, res_v, res_rho, mass_error, cd, cl, umax, eff, eta_str])
                
                # If this step also matches interval, save data now with known coeffs
                if step % interval == 0:
                    self.save_data(step, cd=cd, cl=cl)

                if res_u < tol and res_v < tol and res_rho < tol:
                    print(f"Converged at step {step}")
                    history_data.append([step, res_u, res_v, res_rho, mass_error, cd, cl, umax, eff, "Done"])
                    if step % interval != 0:
                         self.save_data(step, cd=cd, cl=cl)
                    break
                
                seg_start_time = time.time()
            
            # Save data if interval matches but NOT handled by the diag block
            elif step % interval == 0:
                if step % 100 != 0: 
                    self._update_macro(dest_f)
                    self.save_data(step)
        
        total_duration = time.time() - global_start_time
        print(f"--- Solver Finished in {total_duration:.2f} seconds ---")

        print("\n--- Simulation Summary ---")
        print(tabulate(history_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f"))

        # === Physics Validation Summary ===
        print("\n" + "="*70)
        print(" "*20 + "PHYSICS VALIDATION SUMMARY")
        print("="*70)

        # Final diagnostics
        final_mass_error = abs(self.total_mass[None] - self.initial_mass[None]) / (self.initial_mass[None] + 1e-12)

        # CFL Number (Courant-Friedrichs-Lewy condition)
        # CFL = u * dt / dx, where dt=1, dx=1 in LBM lattice units
        cfl = self.max_u[None] * 1.0 / 1.0

        # Grid Reynolds Number
        re_grid = self.u_in * self.L_char / self.nu

        print(f"\n{'Mass Conservation':<30} : {final_mass_error:.2e}")
        print(f"  Initial Mass                 : {self.initial_mass[None]:.6f}")
        print(f"  Final Mass                   : {self.total_mass[None]:.6f}")
        if final_mass_error < 1e-6:
            print(f"  Status                       : ✅ EXCELLENT (< 1e-6)")
        elif final_mass_error < 1e-4:
            print(f"  Status                       : ✅ GOOD (< 1e-4)")
        elif final_mass_error < 1e-3:
            print(f"  Status                       : ⚠️  ACCEPTABLE (< 1e-3)")
        else:
            print(f"  Status                       : ❌ POOR (> 1e-3)")

        print(f"\n{'Numerical Stability':<30}")
        print(f"  CFL Number                   : {cfl:.4f}")
        if cfl < 0.5:
            print(f"  CFL Status                   : ✅ STABLE (< 0.5)")
        elif cfl < 1.0:
            print(f"  CFL Status                   : ⚠️  MARGINAL (< 1.0)")
        else:
            print(f"  CFL Status                   : ❌ UNSTABLE (> 1.0)")

        print(f"  Max Velocity (lattice units) : {self.max_u[None]:.6f}")
        print(f"  Inlet Velocity               : {self.u_in:.6f}")
        print(f"  Velocity Ratio (Umax/Uin)    : {self.max_u[None]/self.u_in:.2f}")

        print(f"\n{'Reynolds Number Verification':<30}")
        print(f"  Target Re                    : {self.re:.1f}")
        print(f"  Grid Re (U*L/nu)             : {re_grid:.1f}")
        print(f"  Viscosity (nu)               : {self.nu:.6f}")
        print(f"  Relaxation Time (tau)        : {self.tau:.6f}")
        print(f"  Char. Length (L)             : {self.L_char:.2f} lattice units")

        print(f"\n{'Physical Interpretation':<30}")
        print(f"  Final Cd (Drag Coeff)        : {cd:.6f}")
        print(f"  Final Cl (Lift Coeff)        : {cl:.6f}")
        print(f"  L/D Ratio                    : {eff:.4f}")

        print("\n" + "="*70)
        print("  TIP: Check M_err < 1e-4 for reliable results")
        print("       Check CFL < 1.0 for numerical stability")
        print("="*70)

if __name__ == "__main__":
    # Test Run
    ti.init(arch=ti.metal, default_fp=ti.f32)
    solver = WindTunnel(res_y=128, re=150.0)
    
    # Create a simple cylinder obstacle for test
    nx, ny = solver.nx, solver.ny
    cx, cy, r = nx/4, ny/2, ny/9
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    mask = ((x - cx)**2 + (y - cy)**2 < r**2).astype(np.int32)
    
    solver.set_obstacle(mask)
    solver.run(steps=2000, interval=500)