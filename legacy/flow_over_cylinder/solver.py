import numpy as np
import taichi as ti
import argparse
from wind_tunnel import WindTunnel, visualize

def main():
    parser = argparse.ArgumentParser(description="2D LBM Flow Over Cylinder Solver (Using WindTunnel Module)")
    parser.add_argument("--res", type=int, default=128, help="Channel height resolution (NY). NX will be 4x NY. Default: 128")
    parser.add_argument("--steps", type=int, default=50000, help="Total simulation steps. Default: 50000")
    parser.add_argument("--interval", type=int, default=1000, help="Output interval (steps). Default: 1000")
    parser.add_argument("--re", type=float, default=150.0, help="Reynolds number based on cylinder diameter. Default: 150.0")
    parser.add_argument("--u_in", type=float, default=0.1, help="Inlet velocity in lattice units. Default: 0.1")
    parser.add_argument("--cs", type=float, default=0.16, help="Smagorinsky constant for LES. Default: 0.16")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory. Default: 'output'")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance. Default: 1e-5")
    args = parser.parse_args()

    # Initialize Taichi
    ti.init(arch=ti.metal, default_fp=ti.f32)

    # Geometry Parameters (Matching original solver.py)
    # CYL_R = NY / 9.0
    # DIA = 2.0 * CYL_R
    cyl_r = args.res / 9.0
    dia = 2.0 * cyl_r
    
    print(f"--- Flow Over Cylinder Setup ---")
    print(f"Re (based on D={dia:.1f}): {args.re}")
    
    # Initialize Wind Tunnel
    # Important: Pass length_scale=dia to ensure Re matches original solver's definition
    tunnel = WindTunnel(
        res_y=args.res,
        re=args.re,
        u_in=args.u_in,
        cs=args.cs,
        output_dir=args.out_dir,
        length_scale=dia
    )

    # Create Cylinder Mask
    nx, ny = tunnel.nx, tunnel.ny
    cx, cy = nx / 4.0, ny / 2.0
    
    y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    # Transpose meshgrid to match (nx, ny) shape for mask[i, j]
    # np.meshgrid with indexing='ij' gives (ny, nx) if inputs are (ny, nx)??
    # Wait, np.meshgrid(y, x, indexing='ij') returns (len(y), len(x)). 
    # Let's use standard xy indexing to be safe and transpose.
    
    x_range = np.arange(nx)
    y_range = np.arange(ny)
    xv, yv = np.meshgrid(x_range, y_range, indexing='ij') # (nx, ny)
    
    # Distance squared
    dist_sq = (xv - cx)**2 + (yv - cy)**2
    mask = (dist_sq <= cyl_r**2).astype(np.int32)
    
    print(f"Setting Cylinder Obstacle: Center=({cx:.1f}, {cy:.1f}), Radius={cyl_r:.1f}")
    tunnel.set_obstacle(mask)
    
    # Run Simulation
    tunnel.run(steps=args.steps, interval=args.interval, tol=args.tol)
    
    # Visualization (Optional: Integrated here or run separately)
    # The original instructions said "visualization.py" is separate. 
    # But we can run it here for convenience if desired, or let user run it.
    # We'll leave it to separate execution or auto-run if steps > 0?
    # Let's just finish the solver part as requested.

if __name__ == "__main__":
    main()