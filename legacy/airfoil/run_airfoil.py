import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import argparse
import os
import sys
import taichi as ti

# Add project root to path for wind_tunnel import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wind_tunnel import WindTunnel, visualize, plot_forces

# ==================== High-Lift Configuration Constants ====================
# Based on typical commercial aircraft high-lift systems (Boeing 747, Airbus A320)
# Reference: Rudolph, P. K. C. "High-Lift Systems on Commercial Subsonic Airliners" (1996)

# Slat (Leading Edge Device) Configuration
SLAT_CHORD_RATIO = 0.15      # Slat chord / Main chord (typical: 0.10-0.20)
SLAT_GAP_X_RATIO = -0.015    # Horizontal gap / chord (negative = slat TE left of main LE)
SLAT_GAP_Y_RATIO = -0.01     # Vertical gap / chord (CORRECTED: negative = below main wing)
SLAT_NACA = "0012"           # Thin symmetric airfoil for leading edge acceleration

# Flap (Trailing Edge Device) Configuration
FLAP_CHORD_RATIO = 0.30      # Flap chord / Main chord (FIXED: 30%, was 35%)
FLAP_OVERLAP_RATIO = 1.03    # Flap LE position / chord (FIXED: 3% aft of TE, was 5% overlap)
FLAP_GAP_Y_RATIO = -0.025    # Vertical gap / chord (FIXED: 2.5%, was 1.5%)
FLAP_NACA = "4415"           # High camber airfoil for maximum lift generation

# Design Notes:
# - Slat gap: ~1-2% chord (allows leading edge slot flow)
# - Flap gap: 3-5% chord horizontal, 2-3% chord vertical (slot for boundary layer energization)
# - Negative Y values = device positioned below main wing chord line
#
# Airfoil Selection Rationale:
# - Slat (0012): Thin symmetric profile → low drag, creates leading edge suction peak
# - Main (2412): Moderate camber (2%) & thickness (12%) → balanced cruise/climb performance
# - Flap (4415): High camber (4%) & thickness (15%) → maximum lift generation at landing/takeoff
#
# Physical Configuration (Based on Real Aircraft):
# - Slat positioned BELOW main wing leading edge (negative Y)
# - Creates downward-directed slot flow to energize lower surface boundary layer
# - Prevents flow separation at high angles of attack
# - Reference: Typical Boeing 737/747, Airbus A320 high-lift systems

# Geometry Sizing
CHORD_TO_HEIGHT_RATIO = 0.6  # Chord length = 60% of vertical resolution
# ==========================================================================

def naca4(number, n_points=100):
    """Generate coordinates for a NACA 4-digit airfoil."""
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0
    
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1015 * x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if m == 0:
        x_upper, y_upper = x, yt
        x_lower, y_lower = x, -yt
    else:
        for i in range(len(x)):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
        
        theta = np.arctan(dyc_dx)
        x_upper = x - yt * np.sin(theta)
        y_upper = yc + yt * np.cos(theta)
        x_lower = x + yt * np.sin(theta)
        y_lower = yc - yt * np.cos(theta)
    
    x_coords = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    return x_coords, y_coords

def transform_coords(x, y, scale=1.0, rotate_deg=0.0, translate_x=0.0, translate_y=0.0, origin=(0,0)):
    """Apply affine transformation to coordinates."""
    # 1. Translate to local origin for rotation/scaling
    ox, oy = origin
    x_local = x - ox
    y_local = y - oy
    
    # 2. Scale
    x_local *= scale
    y_local *= scale
    
    # 3. Rotate (Standard math: CCW is positive. Aero: Nose up is usually positive AoA)
    # Here we define rotate_deg as "geometric rotation".
    # Positive rotates coordinates CCW.
    rad = np.radians(rotate_deg)
    xr = x_local * np.cos(rad) - y_local * np.sin(rad)
    yr = x_local * np.sin(rad) + y_local * np.cos(rad)
    
    # 4. Translate back + Offset
    x_final = xr + ox + translate_x
    y_final = yr + oy + translate_y
    
    return x_final, y_final

def generate_high_lift_mask(nx, ny, main_naca, chord, aoa, flap_deg, slat_deg, cx, cy):
    """
    Generate a Multi-Element Airfoil Mask.
    """
    # Grid setup
    x_grid = np.arange(nx)
    y_grid = np.arange(ny)
    xv, yv = np.meshgrid(x_grid, y_grid, indexing='ij')
    points = np.vstack((xv.flatten(), yv.flatten())).T
    
    final_mask_flat = np.zeros(nx * ny, dtype=bool)
    
    # --- 1. Main Element ---
    xm, ym = naca4(main_naca, 200)
    # Initially 0..1. Scale by chord.
    # We construct everything relative to Main Wing Leading Edge at (0,0) first.
    xm *= chord
    ym *= chord
    
    # --- 2. Slat (Leading Edge) ---
    if slat_deg > 0:
        # Slat geometry: smaller, cambered airfoil
        xs, ys = naca4(SLAT_NACA, 100)

        # Slat Configuration
        slat_scale = SLAT_CHORD_RATIO
        slat_chord_len = slat_scale * chord  # Actual slat chord length

        # Strategy: First rotate slat around its quarter-chord, then translate to position
        # This ensures the gap is measured correctly after rotation

        # Step 1: Scale the slat
        xs_scaled = xs * slat_chord_len
        ys_scaled = ys * slat_chord_len

        # Step 2: Rotate around slat's aerodynamic center (0.25 * slat_chord_len)
        slat_pivot_x = 0.25 * slat_chord_len
        slat_pivot_y = 0.0

        # First apply rotation only (no translation yet)
        xs_rot, ys_rot = transform_coords(xs_scaled, ys_scaled, scale=1.0, rotate_deg=-slat_deg,
                                          translate_x=0, translate_y=0,
                                          origin=(slat_pivot_x, slat_pivot_y))

        # Step 3: Find rotated TE position
        slat_te_idx = np.argmax(xs_rot)
        slat_te_x_rot = xs_rot[slat_te_idx]
        slat_te_y_rot = ys_rot[slat_te_idx]

        # Step 4: Calculate translation to position slat TE near main LE
        # Target: slat TE at (SLAT_GAP_X_RATIO * chord, SLAT_GAP_Y_RATIO * chord)
        target_te_x = SLAT_GAP_X_RATIO * chord
        target_te_y = SLAT_GAP_Y_RATIO * chord

        shift_x = target_te_x - slat_te_x_rot
        shift_y = target_te_y - slat_te_y_rot

        # Step 5: Apply final translation
        xs = xs_rot + shift_x
        ys = ys_rot + shift_y
        
    # --- 3. Flap (Trailing Edge) ---
    if flap_deg > 0:
        # Flap geometry: cambered airfoil for high lift
        xf, yf = naca4(FLAP_NACA, 100)
        xf *= chord
        yf *= chord

        # Flap Configuration
        flap_scale = FLAP_CHORD_RATIO

        # Position: Overlap with main wing
        # Main TE is at x=chord.
        # We want Flap LE (x=0) to be at x=FLAP_OVERLAP_RATIO*chord (5% overlap)
        flap_pos_x = chord * FLAP_OVERLAP_RATIO
        flap_pos_y = FLAP_GAP_Y_RATIO * chord  # Vertical gap

        # Rotate flap downwards
        xf, yf = transform_coords(xf, yf, scale=flap_scale, rotate_deg=-flap_deg,
                                  translate_x=flap_pos_x,
                                  translate_y=flap_pos_y)

    # --- 4. Global Rotation (AoA) & Translation to Grid Center ---
    # Combine all parts into a list for processing
    parts = []
    parts.append((xm, ym))
    if slat_deg > 0: parts.append((xs, ys))
    if flap_deg > 0: parts.append((xf, yf))

    # Rotate entire assembly around Main Wing Aerodynamic Center (0.25 * chord, 0)
    pivot_x = 0.25 * chord
    pivot_y = 0.0

    # Center shift (User provided cx, cy is where 0.25 chord should end up)
    shift_x = cx - pivot_x
    shift_y = cy - pivot_y

    for (x_part, y_part) in parts:
        # --- AoA Rotation Logic ---
        # IMPORTANT: Using -aoa (negative) is CORRECT for coordinate frame rotation.
        # Positive AoA means "nose up relative to freestream".
        # We achieve this by rotating the coordinate system (not the object).
        # Frame rotation direction = opposite of object rotation.
        # Example: AoA=+10° → LE should be higher than TE → rotate_deg=-10° ✓
        x_rot, y_rot = transform_coords(x_part, y_part, scale=1.0, rotate_deg=-aoa,
                                        translate_x=shift_x, translate_y=shift_y,
                                        origin=(pivot_x, pivot_y))
        
        # Rasterize this part
        poly_verts = np.vstack((x_rot, y_rot)).T
        path = Path(poly_verts)
        mask_part = path.contains_points(points)
        final_mask_flat = np.logical_or(final_mask_flat, mask_part)

    return final_mask_flat.reshape(nx, ny).astype(np.int32)

def main():
    parser = argparse.ArgumentParser(description="Run High-Lift Airfoil Simulation")
    parser.add_argument("--naca", type=str, default="2412", help="Main wing NACA code")
    parser.add_argument("--aoa", type=float, default=10.0, help="Angle of attack (deg)")
    parser.add_argument("--flap", type=float, default=0.0, help="Flap deflection (deg)")
    parser.add_argument("--slat", type=float, default=0.0, help="Slat deflection (deg)")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--res", type=int, default=512, help="Vertical resolution")
    parser.add_argument("--u_in", type=float, default=0.05, help="Inlet velocity (lattice units)")
    parser.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    parser.add_argument("--out", type=str, default="airfoil_output", help="Output directory")
    args = parser.parse_args()
    
    ti.init(arch=ti.metal, default_fp=ti.f32)
    
    print(f"--- Configuration ---")
    print(f"Airfoil: NACA {args.naca}")
    print(f"AoA: {args.aoa} deg")
    print(f"Slat: {args.slat} deg, Flap: {args.flap} deg")
    print(f"Re: {args.re}, U_in: {args.u_in}")
    
    # 0. Calculate Geometry Parameters
    # Chord length: reasonable relative to resolution, leaves room for wake
    chord_len = args.res * CHORD_TO_HEIGHT_RATIO
    
    # 1. Init Tunnel
    # Pass chord_len as length_scale for correct Cd/Cl normalization
    tunnel = WindTunnel(res_y=args.res, re=args.re, u_in=args.u_in, output_dir=args.out, length_scale=chord_len)
    
    # 2. Generate Geometry
    mask = generate_high_lift_mask(tunnel.nx, tunnel.ny, 
                                   main_naca=args.naca,
                                   chord=chord_len,
                                   aoa=args.aoa,
                                   flap_deg=args.flap,
                                   slat_deg=args.slat,
                                   cx=tunnel.nx/3.5, # Position slightly forward
                                   cy=tunnel.ny/2)
    
    tunnel.set_obstacle(mask)
    
    # 3. Run
    tunnel.run(steps=args.steps, interval=100)
    
    # 4. Visualize
    print(f"--- Starting Visualization (This may take some time) ---")
    visualize(input_dir=args.out, output_dir=f"{args.out}_vis")
    plot_forces(input_dir=args.out, output_dir=f"{args.out}_vis")

if __name__ == "__main__":
    main()