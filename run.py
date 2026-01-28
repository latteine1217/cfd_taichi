"""
CFD Taichi - 統一執行入口
=========================

Usage:
    python run.py <case> [options]

Available cases:
    ldc         - Lid-Driven Cavity Flow
    cylinder    - Flow Over Cylinder
    airfoil     - High-Lift Airfoil System

Examples:
    python run.py ldc --res 256 --re 1000
    python run.py cylinder --res 128 --re 150
    python run.py airfoil --res 256 --aoa 10 --re 1000
"""

import os
import sys
import argparse

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    case = sys.argv[1].lower()

    # 移除 case 名稱，讓各個 case 的 parser 處理剩餘參數
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if case in ["ldc", "lid", "cavity"]:
        from examples.lid_driven_cavity import main as ldc_main

        ldc_main()

    elif case in ["cylinder", "cyl"]:
        from examples.flow_over_cylinder import main as cyl_main

        cyl_main()

    elif case in ["airfoil", "wing"]:
        from examples.airfoil import main as airfoil_main

        airfoil_main()

    else:
        print(f"Unknown case: {case}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
