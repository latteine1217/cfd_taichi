# 🌊 CFD Taichi: Unified LBM Solver

[![Taichi](https://img.shields.io/badge/backend-Taichi-red.svg)](https://taichi-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**CFD Taichi** 是基於 Taichi 的高效能 LBM 求解器，專注於清晰的物理模型、可驗證的診斷與可重現的案例流程。

---

## 🚀 核心特性

- **統一核心架構**：單一 `LBMSolver` 管理碰撞、流傳、診斷與案例流程。
- **進階物理模型**：MRT-LBM + Dynamic Smagorinsky LES（自動估計 Cs）。
- **邊界條件系統**：Zou-He、Orlanski、Free-Slip、Moving Wall 等可插拔設計。
- **粒子煙線可視化**：支援 50 萬粒子並行追蹤。
- **診斷監控**：`macro_res`、`KE_mean`、`mass_error` 等即時輸出。

---

## 📦 專案結構

```
cfd_taichi/
├── src/
│   └── lbm_taichi/
│       ├── core/                     # 求解器核心
│       └── utils/                    # 幾何與可視化工具
├── examples/                         # 案例腳本
├── tests/                            # 測試與基準
├── docs/                             # 技術文件
└── run.py                            # 統一入口
```

---

## ⚙️ 安裝

```bash
uv pip install taichi numpy matplotlib tabulate imageio
```

---

## ▶️ 執行方式

**統一入口**
```bash
uv run python run.py cylinder --res 128 --re 150
```

**直接執行案例**
```bash
uv run python examples/flow_over_cylinder.py --res 128 --re 150
```

**切換碰撞模型**
```bash
uv run python examples/flow_over_cylinder.py --collision emrt
```

**VTK 輸出**
```bash
uv run python examples/flow_over_cylinder.py --vtk
```

---

## 📈 可視化

```bash
uv run python src/lbm_taichi/utils/visualization.py output_airfoil --gif --type velocity
```

---

## 🧩 套件匯入

```python
from lbm_taichi import LBMSolver, BoundaryConditions, Diagnostics
```

---

## 🧪 經典案例

- **Lid-Driven Cavity**：邊界與收斂驗證
- **Flow Over Cylinder**：卡門渦街與 Strouhal
- **Multi-Element Airfoil**：高升力配置
- **Kelvin-Helmholtz**：剪切不穩定性

---

## 🙏 致謝

特別感謝 [Taichi 團隊](https://github.com/taichi-dev) 提供高效能計算框架。

**開發者**: latteine1217  \
**維護狀態**: ✅ 主動開發中  \
**授權**: MIT License
