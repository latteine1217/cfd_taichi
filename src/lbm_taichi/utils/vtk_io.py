"""
VTK 輸出工具
============

What:
- 將 2D LBM 場輸出為 VTK ImageData (.vti)

Why:
- 方便在 ParaView 進行後處理與量測

When:
- 需要導出速度場、密度場或 mask 供外部可視化時
"""

from __future__ import annotations

import numpy as np


def write_vti(
    path: str, rho: np.ndarray, u: np.ndarray, mask: np.ndarray | None = None
):
    """
    寫出 VTK ImageData (.vti)

    Args:
        path: 輸出路徑
        rho: (nx, ny) 密度場
        u: (nx, ny, 2) 速度場
        mask: (nx, ny) 障礙物遮罩（選用）
    """
    nx, ny = rho.shape

    rho_flat = rho.astype(np.float32).ravel(order="F")

    u_flat = np.zeros((nx * ny, 3), dtype=np.float32)
    u_flat[:, 0] = u[:, :, 0].astype(np.float32).ravel(order="F")
    u_flat[:, 1] = u[:, :, 1].astype(np.float32).ravel(order="F")
    u_flat[:, 2] = 0.0

    if mask is not None:
        mask_flat = mask.astype(np.int32).ravel(order="F")
    else:
        mask_flat = None

    extent = f"0 {nx - 1} 0 {ny - 1} 0 0"

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">')
    lines.append(f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">')
    lines.append(f'    <Piece Extent="{extent}">')
    lines.append('      <PointData Scalars="rho" Vectors="u">')

    lines.append('        <DataArray type="Float32" Name="rho" format="ascii">')
    lines.append("          " + " ".join(f"{v:.6e}" for v in rho_flat))
    lines.append("        </DataArray>")

    lines.append(
        '        <DataArray type="Float32" Name="u" NumberOfComponents="3" format="ascii">'
    )
    lines.append("          " + " ".join(f"{v:.6e}" for v in u_flat.ravel()))
    lines.append("        </DataArray>")

    if mask_flat is not None:
        lines.append('        <DataArray type="Int32" Name="mask" format="ascii">')
        lines.append("          " + " ".join(str(int(v)) for v in mask_flat))
        lines.append("        </DataArray>")

    lines.append("      </PointData>")
    lines.append("      <CellData/>")
    lines.append("    </Piece>")
    lines.append("  </ImageData>")
    lines.append("</VTKFile>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
