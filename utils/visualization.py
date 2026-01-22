"""
可視化工具模組
==============

提供統一、專業且高品質的流體力學數據可視化功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Optional, List, Dict, Any
import time


class Visualizer:
    """專業 CFD 數據可視化管理類別"""

    def __init__(self, output_dir: str = "output", style: str = "white"):
        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        
        if style == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'figure.dpi': 150,
            'savefig.dpi': 300
        })

    def _get_figure(self, nx: int, ny: int) -> tuple:
        aspect_ratio = ny / nx
        fig_width = 12
        fig_height = max(fig_width * aspect_ratio * 1.2, 4)
        return plt.subplots(figsize=(fig_width, fig_height))

    def plot_velocity(self, u, mask=None, particles=None, title="Velocity", show_quiver=False, save_path=None):
        u_mag = np.linalg.norm(u, axis=2)
        if mask is not None:
            u_mag = np.ma.masked_where(mask == 1, u_mag)

        fig, ax = self._get_figure(u.shape[0], u.shape[1])
        im = ax.imshow(u_mag.T, origin='lower', cmap='turbo', interpolation='bilinear', aspect='equal')
        
        # 繪製粒子 (煙線效果)
        if particles is not None and len(particles) > 0:
            ax.scatter(particles[:, 0], particles[:, 1], s=0.1, c='white', alpha=0.5, edgecolors='none')

        if show_quiver:
            nx, ny = u.shape[0], u.shape[1]
            step = max(nx // 30, 1)
            y, x = np.mgrid[0:ny:step, 0:nx:step]
            ux, uy = u[::step, ::step, 0].T, u[::step, ::step, 1].T
            ax.quiver(x, y, ux, uy, color='white', alpha=0.5)

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='|U|', fraction=0.046, pad=0.04)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight'); plt.close()

    def plot_vorticity(self, u, mask=None, title="Vorticity", save_path=None):
        dvdx = np.gradient(u[:, :, 1], axis=0)
        dudy = np.gradient(u[:, :, 0], axis=1)
        vorticity = dvdx - dudy
        if mask is not None:
            vorticity = np.ma.masked_where(mask == 1, vorticity)

        fig, ax = self._get_figure(u.shape[0], u.shape[1])
        v_data = np.abs(vorticity.compressed()) if mask is not None else np.abs(vorticity)
        limit = np.percentile(v_data, 99) if len(v_data) > 0 else 1.0
        
        im = ax.imshow(vorticity.T, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='bilinear', aspect='equal')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=r'$\omega$', fraction=0.046, pad=0.04)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight'); plt.close()

    def process_all(self, plot_types: List[str] = ["velocity", "vorticity"]):
        files = sorted(glob.glob(os.path.join(self.output_dir, "state_*.npy")))
        if not files: return
        total = len(files)
        print(f"🚀 Rendering {total} steps...")
        start_time = time.time()
        
        for i, f in enumerate(files):
            data = np.load(f, allow_pickle=True).item()
            step, u, rho, mask = data['step'], data['u'], data['rho'], data['mask']
            particles = data.get('particles', None)
            
            for p_type in plot_types:
                save_name = os.path.join(self.fig_dir, f"{p_type}_{{step:06d}}.png")
                if p_type == "velocity": self.plot_velocity(u, mask, particles, f"Velocity Step {step}", save_path=save_name)
                elif p_type == "vorticity": self.plot_vorticity(u, mask, f"Vorticity Step {step}", save_path=save_name)
            
            # 動態進度條
            if (i + 1) % 5 == 0 or (i + 1) == total:
                percent = (i + 1) / total * 100
                bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
                print(f"\rRender Progress: |{bar}| {i+1}/{total} ({percent:.1f}%)", end="", flush=True)
        
        print(f"\n✅ Finished rendering in {time.time() - start_time:.1f}s")

    def create_gif(self, prefix: str, gif_name: str, fps: int = 15):
        try:
            import imageio.v2 as imageio
        except ImportError: return
        img_files = sorted(glob.glob(os.path.join(self.fig_dir, f"{prefix}_*.png")))
        if not img_files: return
        total = len(img_files)
        print(f"🎬 Creating GIF: {gif_name} ({total} frames)...")
        
        with imageio.get_writer(os.path.join(self.output_dir, gif_name), mode='I', fps=fps) as writer:
            for i, f in enumerate(img_files):
                writer.append_data(imageio.imread(f))
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(f"\rGIF Progress: {i+1}/{total} frames encoded", end="", flush=True)
        
        print(f"\n✅ GIF saved to {os.path.join(self.output_dir, gif_name)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--type', type=str, nargs='+', default=['velocity', 'vorticity'])
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()
    if not os.path.exists(args.dir): return
    vis = Visualizer(output_dir=args.dir)
    vis.process_all(plot_types=args.type)
    if args.gif:
        for p_type in args.type:
            vis.create_gif(prefix=p_type, gif_name=f"{p_type}_anim.gif", fps=args.fps)


if __name__ == "__main__":
    main()
