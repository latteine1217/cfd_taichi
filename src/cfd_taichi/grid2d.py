"""
Grid2D 幾何物件
================

What:
- 提供 solver case layer 可直接持有的 2D 網格物件

Why:
- descriptor 只解決「如何套用」，但沒有承載幾何本體與尺寸語義
- solver toolkit 需要正式的 grid object，讓 runner 與 benchmark registry
  可以依賴穩定的 geometry contract

When:
- CaseRunner 建立 solver 與配置網格
- benchmark / notebook / workflow 層需要共通幾何物件
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .descriptors import GridDescriptor


@dataclass(frozen=True)
class Grid2D:
    """
    2D 網格抽象基類。

    What:
    - 提供共同尺寸與 descriptor 轉換介面

    Why:
    - case layer 不應再直接操作裸露的 `dx/dy/x_node/y_node`
    """

    kind: str

    def to_descriptor(self) -> GridDescriptor:
        raise NotImplementedError

    def solver_size_kwargs(self, solver_family: str) -> dict[str, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class CartesianGrid2D(Grid2D):
    """
    均勻 Cartesian cell-centered grid。
    """

    ni: int
    nj: int
    dx: float
    dy: float

    def __init__(self, ni: int, nj: int, dx: float, dy: float):
        if ni <= 0 or nj <= 0:
            raise ValueError(f"ni and nj must be positive, got ni={ni}, nj={nj}")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")
        object.__setattr__(self, "kind", "cartesian")
        object.__setattr__(self, "ni", int(ni))
        object.__setattr__(self, "nj", int(nj))
        object.__setattr__(self, "dx", float(dx))
        object.__setattr__(self, "dy", float(dy))

    def to_descriptor(self) -> GridDescriptor:
        return GridDescriptor.cartesian(dx=self.dx, dy=self.dy)

    def solver_size_kwargs(self, solver_family: str) -> dict[str, int]:
        if solver_family == "fvm":
            return {"ni": self.ni, "nj": self.nj}
        if solver_family == "lbm":
            return {"nx": self.ni, "ny": self.nj}
        raise ValueError(f"Unsupported solver_family: {solver_family}")


@dataclass(frozen=True)
class CurvilinearGrid2D(Grid2D):
    """
    曲線 structured grid。
    """

    x_node: np.ndarray
    y_node: np.ndarray
    ng: int = 2

    def __init__(self, x_node: np.ndarray, y_node: np.ndarray, ng: int = 2):
        xn = np.asarray(x_node, dtype=np.float32)
        yn = np.asarray(y_node, dtype=np.float32)
        if xn.shape != yn.shape:
            raise ValueError(f"x_node and y_node must share shape, got {xn.shape} and {yn.shape}")
        if xn.ndim != 2 or xn.shape[0] < 2 or xn.shape[1] < 2:
            raise ValueError(f"Expected 2D node arrays with shape >= (2, 2), got {xn.shape}")
        if ng < 0:
            raise ValueError(f"ng must be non-negative, got {ng}")
        object.__setattr__(self, "kind", "curvilinear")
        object.__setattr__(self, "x_node", xn)
        object.__setattr__(self, "y_node", yn)
        object.__setattr__(self, "ng", int(ng))

    @property
    def ni(self) -> int:
        return int(self.x_node.shape[0] - 1 - 2 * self.ng)

    @property
    def nj(self) -> int:
        return int(self.x_node.shape[1] - 1 - 2 * self.ng)

    def to_descriptor(self) -> GridDescriptor:
        return GridDescriptor.curvilinear(self.x_node, self.y_node)

    def solver_size_kwargs(self, solver_family: str) -> dict[str, int]:
        if solver_family != "fvm":
            raise ValueError("CurvilinearGrid2D currently targets FVM solvers only.")
        if self.ni <= 0 or self.nj <= 0:
            raise ValueError(
                f"Curvilinear node shape {self.x_node.shape} with ng={self.ng} gives "
                f"non-positive interior size ({self.ni}, {self.nj})."
            )
        return {"ni": self.ni, "nj": self.nj}


@dataclass(frozen=True)
class LatticeGrid2D(Grid2D):
    """
    LBM lattice grid。
    """

    nx: int
    ny: int

    def __init__(self, nx: int, ny: int):
        if nx <= 0 or ny <= 0:
            raise ValueError(f"nx and ny must be positive, got nx={nx}, ny={ny}")
        object.__setattr__(self, "kind", "lattice")
        object.__setattr__(self, "nx", int(nx))
        object.__setattr__(self, "ny", int(ny))

    def to_descriptor(self) -> GridDescriptor:
        return GridDescriptor.lattice(nx=self.nx, ny=self.ny)

    def solver_size_kwargs(self, solver_family: str) -> dict[str, int]:
        if solver_family != "lbm":
            raise ValueError("LatticeGrid2D currently targets LBM solvers only.")
        return {"nx": self.nx, "ny": self.ny}
