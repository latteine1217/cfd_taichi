import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio.v2 as imageio
import argparse

def plot_forces(input_dir, output_dir):
    """
    Reads .npy files and plots the history of Lift (Cl) and Drag (Cd) coefficients.
    Saves the plot to force_history.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    state_files = sorted(glob.glob(os.path.join(input_dir, "state_*.npy")))
    
    steps = []
    cds = []
    cls = []
    
    print("Extracting force history...")
    for f_path in state_files:
        try:
            data = np.load(f_path, allow_pickle=True).item()
            if 'cd' in data and 'cl' in data:
                steps.append(data.get('step', 0))
                cds.append(data['cd'])
                cls.append(data['cl'])
        except:
            continue
            
    if not steps:
        print("No force data found in output files.")
        return

    # Sort data
    sorted_indices = np.argsort(steps)
    steps = np.array(steps)[sorted_indices]
    cds = np.array(cds)[sorted_indices]
    cls = np.array(cls)[sorted_indices]
    
    # Calculate Efficiency L/D
    eff = np.zeros_like(cds)
    valid_mask = np.abs(cds) > 1e-9
    eff[valid_mask] = cls[valid_mask] / cds[valid_mask]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
    
    color = 'tab:green'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Lift Coefficient ($C_l$)', color=color)
    l1 = ax1.plot(steps, cls, color=color, linewidth=2, label='$C_l$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Drag Coefficient ($C_d$)', color=color)  # we already handled the x-label with ax1
    l2 = ax2.plot(steps, cds, color=color, linewidth=2, label='$C_d$')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Added lines for legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    plt.title('Aerodynamic Forces History')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    out_path = os.path.join(output_dir, "force_history.png")
    plt.savefig(out_path)
    print(f"Force history saved to {out_path}")
    plt.close(fig)

def visualize(input_dir, output_dir, fps=15, vmax=None):
    os.makedirs(output_dir, exist_ok=True)

    state_files = sorted(glob.glob(os.path.join(input_dir, "state_*.npy")))
    if not state_files:
        print(f"No output files found in {input_dir}!")
        return

    print(f"Found {len(state_files)} files in {input_dir}.")

    # Determine global vmax if not provided
    if vmax is None:
        print("Scanning files for global max velocity (using 99.9th percentile for contrast)...")
        vmax = 0.0
        for f_path in state_files:
            try:
                data = np.load(f_path, allow_pickle=True).item()
                u = data['u']
                u_mag = np.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)
                # Use 99.9th percentile to ignore outliers and increase contrast
                local_max = np.percentile(u_mag, 99.9)
                vmax = max(vmax, local_max)
            except:
                continue
        # Add a tiny buffer
        vmax = vmax * 1.02 if vmax > 0 else 0.1
        print(f"Global robust vmax detected: {vmax:.4f}")

    print(f"Processing frames to {output_dir}...")
    images = []
    
    for idx, f_path in enumerate(state_files):
        try:
            filename = os.path.basename(f_path)
            step_str = filename.replace("state_", "").replace(".npy", "")
            step = int(step_str)
            
            data = np.load(f_path, allow_pickle=True).item()
            u = data['u']
            mask = data.get('mask', None)
            
            # Compute Magnitude
            u_mag = np.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)
            
            # Transpose [x, y] -> [y, x] for imshow
            u_mag_t = u_mag.T
            
            ny, nx = u_mag_t.shape
            ratio = nx / ny
            fig, ax = plt.subplots(figsize=(6 * ratio, 6), dpi=100)
            
            # Plot Velocity with 'turbo' colormap for better perceptual contrast
            im = ax.imshow(u_mag_t, origin='lower', cmap='turbo', interpolation='bicubic', vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, label='Velocity Magnitude |u|')
            
            # Overlay Particles (Smoke)
            if 'particles' in data:
                p = data['particles']
                if len(p) > 0:
                    # p[:, 0] is x, p[:, 1] is y
                    ax.scatter(p[:, 0], p[:, 1], s=0.5, c='white', alpha=0.6)
            
            # Overlay Mask
            if mask is not None:
                mask_t = mask.T
                ax.contour(mask_t, levels=[0.5], colors='black', linewidths=2, origin='lower')
                # Overlay semi-transparent gray on solid parts
                masked_data = np.ma.masked_where(mask_t == 0, mask_t)
                ax.imshow(masked_data, origin='lower', cmap='gray_r', alpha=0.5, vmin=0, vmax=1)

            ax.set_title(f"Wind Tunnel Simulation - Step {step}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis('tight')
            
            fig.tight_layout()
            
            frame_path = os.path.join(output_dir, f"frame_{step:06d}.png")
            plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
            images.append(imageio.imread(frame_path))
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(state_files)} frames...")

        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    if images:
        gif_path = os.path.join(output_dir, "simulation.gif")
        imageio.mimsave(gif_path, images, fps=fps)
        print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Wind Tunnel Results")
    parser.add_argument("--in_dir", type=str, default="output", help="Input directory")
    parser.add_argument("--out_dir", type=str, default="vis_output", help="Output directory")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    args = parser.parse_args()
    
    visualize(args.in_dir, args.out_dir, args.fps)
    plot_forces(args.in_dir, args.out_dir)
