# 2D Flow Over Cylinder (LBM-MRT)

This module implements a 2D Laminar/Turbulent flow simulation over a circular cylinder using the Lattice Boltzmann Method (LBM) with Multi-Relaxation Time (MRT) collision operator, accelerated by Taichi.

## Physics & Method

*   **Method**: Lattice Boltzmann Method (D2Q9 model).
*   **Collision**: MRT (Multi-Relaxation Time) for better stability at higher Reynolds numbers.
*   **Turbulence Model**: Smagorinsky LES (Large Eddy Simulation) subgrid scale model (optional, controlled by `Cs`).
*   **Geometry**:
    *   Channel dimensions: $L_x = 4 \times L_y$.
    *   Cylinder: Located at $(x, y) = (L_x/4, L_y/2)$ with diameter $D \approx L_y/4.5$.
*   **Boundary Conditions**:
    *   **Inlet (West)**: Zou-He Velocity Inlet (Fixed $u_{in}$).
    *   **Outlet (East)**: Zero-Gradient (Neumann) Extrapolation.
    *   **Top/Bottom**: Periodic Boundary Condition.
    *   **Cylinder Surface**: No-slip (Bounce-back).

## File Structure

*   `solver.py`: Main CFD solver. Handles physics, time-stepping, and data output.
*   `visualization.py`: Post-processing tool. Converts output `.npy` files into images and GIFs.

## Usage

### 1. Run the Solver

```bash
python flow_over_cylinder/solver.py [arguments]
```

**Common Arguments:**

*   `--res`: Vertical resolution (NY). The horizontal resolution (NX) is automatically set to `4 * NY`. (Default: 128)
*   `--re`: Reynolds number based on cylinder diameter ($Re = U_{in} D / \nu$). (Default: 150.0)
*   `--steps`: Total number of simulation steps. (Default: 50000)
*   `--interval`: Interval for saving `.npy` state files. (Default: 1000)
*   `--u_in`: Inlet velocity in lattice units. Keep low (< 0.2) for compressibility limit. (Default: 0.1)
*   `--cs`: Smagorinsky constant for LES. Set to 0 to disable. (Default: 0.16)
*   `--out_dir`: Directory to save simulation data. (Default: `output`)

**Example:**
Run a simulation at Re=200 with higher resolution:
```bash
python flow_over_cylinder/solver.py --res 256 --re 200.0 --steps 100000 --out_dir output_re200
```

### 2. Visualization

Generates velocity magnitude heatmaps with streamlines and saves an animation.

```bash
python flow_over_cylinder/visualization.py [arguments]
```

**Arguments:**

*   `--in_dir`: Input directory containing `.npy` files. (Default: `output`)
*   `--out_dir`: Output directory for images/GIF. (Default: `vis_output`)
*   `--fps`: Frames per second for the GIF. (Default: 15)

**Example:**
```bash
python flow_over_cylinder/visualization.py --in_dir output_re200 --out_dir vis_re200 --fps 20
```

## Output Format

The solver saves `.npy` files containing a dictionary with the following keys:

*   `rho`: Density field (scalar field, shape `[NX, NY]`).
*   `u`: Velocity field (vector field, shape `[NX, NY, 2]`).
*   `mask`: Geometry mask (1 = solid/cylinder, 0 = fluid).

## Notes

*   **Stability**: If the simulation blows up (NaNs), try increasing resolution (`--res`), reducing inlet velocity (`--u_in`), or lowering the Reynolds number.
*   **Performance**: The code automatically selects the best available backend (Metal on macOS, CUDA/Vulkan elsewhere) via `ti.init(arch=ti.metal)`.
