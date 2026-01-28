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
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        plt.rcParams.update(
            {"font.family": "sans-serif", "figure.dpi": 150, "savefig.dpi": 300}
        )

        self._render_cache = {}

    def _get_figure(self, nx: int, ny: int) -> tuple:
        aspect_ratio = ny / nx
        fig_width = 12
        fig_height = max(fig_width * aspect_ratio * 1.2, 4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_velocity(
        self,
        u,
        mask=None,
        particles=None,
        title="Velocity",
        show_quiver=False,
        save_path=None,
    ):
        u_mag = np.linalg.norm(u, axis=2)
        if mask is not None:
            u_mag = np.ma.masked_where(mask == 1, u_mag)

        nx, ny = u.shape[0], u.shape[1]
        fig, ax = self._get_figure(nx, ny)
        im = ax.imshow(
            u_mag.T,
            origin="lower",
            cmap="turbo",
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
        )

        if particles is not None and len(particles) > 0:
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=0.3,
                c="white",
                alpha=0.8,
                edgecolors="none",
            )

        if show_quiver:
            step = max(nx // 30, 1)
            y, x = np.mgrid[0:ny:step, 0:nx:step]
            ux, uy = u[::step, ::step, 0].T, u[::step, ::step, 1].T
            ax.quiver(x, y, ux, uy, color="white", alpha=0.5)

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="|U|", fraction=0.046, pad=0.04)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    def plot_vorticity(self, u, mask=None, title="Vorticity", save_path=None):
        dvdx = np.gradient(u[:, :, 1], axis=0)
        dudy = np.gradient(u[:, :, 0], axis=1)
        vorticity = dvdx - dudy
        if mask is not None:
            vorticity = np.ma.masked_where(mask == 1, vorticity)

        nx, ny = u.shape[0], u.shape[1]
        fig, ax = self._get_figure(nx, ny)
        v_data = (
            np.abs(vorticity.compressed()) if mask is not None else np.abs(vorticity)
        )
        limit = np.percentile(v_data, 99) if len(v_data) > 0 else 1.0

        im = ax.imshow(
            vorticity.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
        )
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Vorticity", fraction=0.046, pad=0.04)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    def plot_eddy_viscosity(
        self, nu_sgs, mask=None, title="Eddy Viscosity (nu_sgs)", save_path=None
    ):
        """
        繪製 Smagorinsky 渦黏度場

        Why 渦黏度重要？
        - 驗證 LES 模型是否在正確位置激活
        - 應在高剪切區域（分離、尾流）最大
        - 自由流區域應接近零
        """
        if nu_sgs is None:
            print("⚠️  Eddy viscosity data not available (LES model disabled?)")
            return

        if mask is not None:
            nu_sgs = np.ma.masked_where(mask == 1, nu_sgs)

        nx, ny = nu_sgs.shape[0], nu_sgs.shape[1]
        fig, ax = self._get_figure(nx, ny)

        # 使用對數尺度顯示（渦黏度範圍跨度大）
        nu_data = nu_sgs.compressed() if mask is not None else nu_sgs.ravel()
        nu_data = nu_data[nu_data > 1e-10]  # 過濾接近零的值

        if len(nu_data) > 0:
            vmin = np.percentile(nu_data, 1)
            vmax = np.percentile(nu_data, 99)
        else:
            vmin, vmax = 1e-6, 1e-3

        im = ax.imshow(
            nu_sgs.T,
            origin="lower",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
        )
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="nu_sgs", fraction=0.046, pad=0.04, format="%.2e")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    def _init_velocity_render(self, nx: int, ny: int, vmin: float, vmax: float):
        fig, ax = self._get_figure(nx, ny)
        im = ax.imshow(
            np.zeros((ny, nx)),
            origin="lower",
            cmap="turbo",
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        scatter = ax.scatter([], [], s=0.3, c="white", alpha=0.8, edgecolors="none")
        plt.colorbar(im, ax=ax, label="|U|", fraction=0.046, pad=0.04)
        return {"fig": fig, "ax": ax, "im": im, "scatter": scatter}

    def _update_velocity_render(self, cache, u_mag, particles, title, save_path):
        cache["im"].set_data(u_mag.T)
        cache["ax"].set_title(title)
        if particles is not None and len(particles) > 0:
            cache["scatter"].set_offsets(particles[:, :2])
        else:
            cache["scatter"].set_offsets(np.empty((0, 2)))
        cache["fig"].savefig(save_path, bbox_inches="tight")

    def _init_vorticity_render(self, nx: int, ny: int, vmin: float, vmax: float):
        fig, ax = self._get_figure(nx, ny)
        im = ax.imshow(
            np.zeros((ny, nx)),
            origin="lower",
            cmap="RdBu_r",
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="Vorticity", fraction=0.046, pad=0.04)
        return {"fig": fig, "ax": ax, "im": im}

    def _update_vorticity_render(self, cache, vorticity, title, save_path):
        cache["im"].set_data(vorticity.T)
        cache["ax"].set_title(title)
        cache["fig"].savefig(save_path, bbox_inches="tight")

    def _init_eddy_viscosity_render(self, nx: int, ny: int, vmin: float, vmax: float):
        fig, ax = self._get_figure(nx, ny)
        im = ax.imshow(
            np.zeros((ny, nx)),
            origin="lower",
            cmap="YlOrRd",
            interpolation="bilinear",
            extent=[0, nx, 0, ny],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="nu_sgs", fraction=0.046, pad=0.04, format="%.2e")
        return {"fig": fig, "ax": ax, "im": im}

    def _update_eddy_viscosity_render(self, cache, nu_sgs, title, save_path):
        cache["im"].set_data(nu_sgs.T)
        cache["ax"].set_title(title)
        cache["fig"].savefig(save_path, bbox_inches="tight")

    def _scan_global_ranges(
        self, files: List[str], plot_types: List[str]
    ) -> Dict[str, Dict[str, float]]:
        ranges = {
            "velocity": {"min": np.inf, "max": -np.inf},
            "vorticity": {"min": np.inf, "max": -np.inf},
            "eddy_viscosity": {"min": np.inf, "max": -np.inf},
        }

        for f in files:
            data = np.load(f, allow_pickle=True).item()
            u = data["u"]
            mask = data.get("mask", None)
            nu_sgs = data.get("nu_sgs", None)

            if "velocity" in plot_types:
                u_mag = np.linalg.norm(u, axis=2)
                if mask is not None:
                    u_mag = u_mag[mask == 0]
                if u_mag.size > 0:
                    ranges["velocity"]["min"] = min(
                        ranges["velocity"]["min"], float(u_mag.min())
                    )
                    ranges["velocity"]["max"] = max(
                        ranges["velocity"]["max"], float(u_mag.max())
                    )

            if "vorticity" in plot_types:
                dvdx = np.gradient(u[:, :, 1], axis=0)
                dudy = np.gradient(u[:, :, 0], axis=1)
                vorticity = dvdx - dudy
                if mask is not None:
                    vorticity = vorticity[mask == 0]
                if vorticity.size > 0:
                    ranges["vorticity"]["min"] = min(
                        ranges["vorticity"]["min"], float(vorticity.min())
                    )
                    ranges["vorticity"]["max"] = max(
                        ranges["vorticity"]["max"], float(vorticity.max())
                    )

            if "eddy_viscosity" in plot_types and nu_sgs is not None:
                if mask is not None:
                    nu_sgs = nu_sgs[mask == 0]
                nu_sgs = nu_sgs[nu_sgs > 1e-10]
                if nu_sgs.size > 0:
                    ranges["eddy_viscosity"]["min"] = min(
                        ranges["eddy_viscosity"]["min"], float(nu_sgs.min())
                    )
                    ranges["eddy_viscosity"]["max"] = max(
                        ranges["eddy_viscosity"]["max"], float(nu_sgs.max())
                    )

        if ranges["velocity"]["min"] == np.inf:
            ranges["velocity"] = {"min": 0.0, "max": 1.0}
        if ranges["vorticity"]["min"] == np.inf:
            ranges["vorticity"] = {"min": -1.0, "max": 1.0}
        if ranges["eddy_viscosity"]["min"] == np.inf:
            ranges["eddy_viscosity"] = {"min": 1e-6, "max": 1e-3}

        return ranges

    def process_all(
        self,
        plot_types: List[str] = ["velocity", "vorticity"],
        max_particles: int = 50000,
    ):
        files = sorted(glob.glob(os.path.join(self.output_dir, "state_*.npy")))
        if not files:
            return
        total = len(files)
        print(f"🚀 Rendering {total} steps...")
        start_time = time.time()

        ranges = self._scan_global_ranges(files, plot_types)
        rng = np.random.default_rng(0)

        for f in files[:1]:
            data = np.load(f, allow_pickle=True).item()
            nx, ny = data["u"].shape[0], data["u"].shape[1]
            if "velocity" in plot_types:
                self._render_cache["velocity"] = self._init_velocity_render(
                    nx,
                    ny,
                    ranges["velocity"]["min"],
                    ranges["velocity"]["max"],
                )
            if "vorticity" in plot_types:
                self._render_cache["vorticity"] = self._init_vorticity_render(
                    nx,
                    ny,
                    ranges["vorticity"]["min"],
                    ranges["vorticity"]["max"],
                )
            if "eddy_viscosity" in plot_types:
                self._render_cache["eddy_viscosity"] = self._init_eddy_viscosity_render(
                    nx,
                    ny,
                    ranges["eddy_viscosity"]["min"],
                    ranges["eddy_viscosity"]["max"],
                )

        for i, f in enumerate(files):
            data = np.load(f, allow_pickle=True).item()
            step, u, rho, mask = data["step"], data["u"], data["rho"], data["mask"]
            particles = data.get("particles", None)
            nu_sgs = data.get("nu_sgs", None)

            if particles is not None and len(particles) > max_particles:
                idx = rng.choice(len(particles), size=max_particles, replace=False)
                particles = particles[idx]

            for p_type in plot_types:
                # 修正這裡的格式化命名
                save_name = os.path.join(self.fig_dir, f"{p_type}_{step:06d}.png")
                if p_type == "velocity":
                    u_mag = np.linalg.norm(u, axis=2)
                    if mask is not None:
                        u_mag = np.ma.masked_where(mask == 1, u_mag)
                    cache = self._render_cache["velocity"]
                    self._update_velocity_render(
                        cache,
                        u_mag,
                        particles,
                        f"Velocity Step {step}",
                        save_name,
                    )
                elif p_type == "vorticity":
                    dvdx = np.gradient(u[:, :, 1], axis=0)
                    dudy = np.gradient(u[:, :, 0], axis=1)
                    vorticity = dvdx - dudy
                    if mask is not None:
                        vorticity = np.ma.masked_where(mask == 1, vorticity)
                    cache = self._render_cache["vorticity"]
                    self._update_vorticity_render(
                        cache,
                        vorticity,
                        f"Vorticity Step {step}",
                        save_name,
                    )
                elif p_type == "eddy_viscosity":
                    if nu_sgs is None:
                        continue
                    if mask is not None:
                        nu_sgs = np.ma.masked_where(mask == 1, nu_sgs)
                    cache = self._render_cache["eddy_viscosity"]
                    self._update_eddy_viscosity_render(
                        cache,
                        nu_sgs,
                        f"Eddy Viscosity Step {step}",
                        save_name,
                    )

            if (i + 1) % 5 == 0 or (i + 1) == total:
                percent = (i + 1) / total * 100
                bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
                print(
                    f"\rRender Progress: |{bar}| {i + 1}/{total} ({percent:.1f}%)",
                    end="",
                    flush=True,
                )

        print(f"\n✅ Finished rendering in {time.time() - start_time:.1f}s")

        for cache in self._render_cache.values():
            plt.close(cache["fig"])

    def create_gif(self, prefix: str, gif_name: str, fps: int = 15):
        try:
            import imageio.v2 as imageio
        except ImportError:
            return
        img_files = sorted(glob.glob(os.path.join(self.fig_dir, f"{prefix}_*.png")))
        if not img_files:
            return
        total = len(img_files)
        print(f"🎬 Creating GIF: {gif_name} ({total} frames)...")

        with imageio.get_writer(
            os.path.join(self.output_dir, gif_name), mode="I", fps=fps
        ) as writer:
            for i, f in enumerate(img_files):
                writer.append_data(imageio.imread(f))
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(
                        f"\rGIF Progress: {i + 1}/{total} frames encoded",
                        end="",
                        flush=True,
                    )

        print(f"\n✅ GIF saved to {os.path.join(self.output_dir, gif_name)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument(
        "--type", type=str, nargs="+", default=["velocity", "vorticity"]
    )
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        return
    vis = Visualizer(output_dir=args.dir)
    vis.process_all(plot_types=args.type)
    if args.gif:
        for p_type in args.type:
            vis.create_gif(prefix=p_type, gif_name=f"{p_type}_anim.gif", fps=args.fps)


if __name__ == "__main__":
    main()
