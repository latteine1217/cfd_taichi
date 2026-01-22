import argparse
from wind_tunnel import visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Flow Over Cylinder Results")
    parser.add_argument("--in_dir", type=str, default="output", help="Input directory")
    parser.add_argument("--out_dir", type=str, default="vis_output", help="Output directory")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max velocity")
    args = parser.parse_args()
    
    visualize(args.in_dir, args.out_dir, args.fps, args.vmax)
