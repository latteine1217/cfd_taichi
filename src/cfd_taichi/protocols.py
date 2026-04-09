"""
Solver Protocols
================

What:
- 定義跨 solver family 的最小共同 protocol

Why:
- `lbm` / `fvm` 的內部數值流程不同，但工具層至少需要一致的欄位與診斷出口
- protocol 先固定資料交換面，再逐步收斂執行流程
"""

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SolverProtocol(Protocol):
    """
    跨 solver family 的最小共同介面。
    """

    solver_family: str
    equation_set: str
    regime: str

    def get_fields(self) -> dict[str, np.ndarray]:
        """
        回傳標準化場資料。
        """

    def get_diagnostics(self) -> Mapping[str, Any]:
        """
        回傳結構化診斷量。
        """
