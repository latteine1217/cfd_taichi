import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio.v2 as imageio
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LBM Lid-Driven Cavity Flow results")
    parser.add_argument("--in_dir", type=str, default="output", help="Input directory containing .npy files. Default: 'output'")
    parser.add_argument("--out_dir", type=str, default="vis_output", help="Output directory for frames and GIF. Default: 'vis_output'")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF. Default: 10")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max velocity for colorbar. If None, it will be detected automatically.")
    return parser.parse_args()

def visualize():
    args = parse_args()
    input_dir = args.in_dir
    output_dir = args.out_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # Find all state files
    state_files = sorted(glob.glob(f"{input_dir}/state_*.npy"))
    if not state_files:
        print(f"No output files found in {input_dir}!")
        return

    print(f"Found {len(state_files)} files in {input_dir}.")
    
    # Determine global vmax for consistent colorbar
    vmax = args.vmax
    if vmax is None:
        print("Scanning files for global max velocity...")
        vmax = 0.0
        # Scan files to find global max to fix colorbar
        for f_path in state_files:
            try:
                data = np.load(f_path, allow_pickle=True).item()
                u = data['u']
                u_mag = np.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)
                vmax = max(vmax, np.max(u_mag))
            except:
                continue
        # Add a small buffer to avoid clipping at the very top of the colormap
        vmax = vmax * 1.05 if vmax > 0 else 0.1
        print(f"Global vmax detected: {vmax:.4f}")

    print(f"Processing frames to {output_dir}...")

    images = []
    
    for idx, f_path in enumerate(state_files):
        # Extract step number from filename
        filename = os.path.basename(f_path)
        step_str = filename.replace("state_", "").replace(".npy", "")
        step = int(step_str)
        
        # Load Data
        try:
            data = np.load(f_path, allow_pickle=True).item()
            u = data['u']
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            continue
        
        # Calculate Magnitude
        u_mag = np.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)
        
        # Transpose for plotting (to match [row, col] -> [y, x] convention)
        u_mag_t = u_mag.T
        u_x_t = u[:, :, 0].T
        u_y_t = u[:, :, 1].T
        
        # Setup Plot with fixed size and fixed DPI
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Magnitude Heatmap with FIXED vmin and vmax
        im = ax.imshow(u_mag_t, origin='lower', cmap='jet', interpolation='spline16', 
                       vmin=0, vmax=vmax)
        
        # Fixed size colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
        
        ax.set_title(f"Lid Driven Cavity Flow - Step {step}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        
        # Use tight_layout and fixed bbox to prevent jumping
        fig.tight_layout()
        
        # Save Frame
        frame_path = f"{output_dir}/frame_{step:06d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        images.append(imageio.imread(frame_path))
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(state_files):
            print(f"Processed {idx + 1}/{len(state_files)} frames...")

    # Save GIF
    gif_path = f"{output_dir}/flow_animation.gif"
    imageio.mimsave(gif_path, images, fps=args.fps)
    print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    visualize()