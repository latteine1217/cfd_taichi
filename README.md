# CFD Taichi - 統一 LBM 流體力學求解器

基於 Taichi 的統一 Lattice Boltzmann Method (LBM) 框架，支援多種 CFD 案例模擬。

**版本 2.1** - 加入粒子系統與強化可視化

---

## ✨ 核心特色

- 🏗️ **統一架構**: 單一 LBM 核心支援所有 CFD 案例
- 🚀 **高效能**: Taichi Metal 後端，Apple Silicon GPU 加速
- 🎯 **MRT-LBM**: Multiple-Relaxation-Time 數值穩定性優異
- 🌪️ **LES 湍流**: Smagorinsky 大渦模擬
- 💨 **粒子系統**: 支援 50 萬級粒子並行平流，實現風洞煙線（Smoke lines）效果
- 📊 **物理驗證**: 即時質量守恆、動量殘差、力係數計算
- 🔧 **模組化**: 可插拔邊界條件、解耦的碰撞模型、診斷系統

---

## 🏗️ 專案架構

```
cfd_taichi/
├── core/                       # 統一 LBM 核心
│   ├── lbm_solver.py          # D2Q9 MRT-LBM 求解器 + 粒子系統
│   ├── boundary_conditions.py  # 邊界條件 (Zou-He, Free-Slip, Moving Wall)
│   ├── collision_models.py     # 碰撞模型工廠 (MRT, BGK)
│   └── diagnostics.py          # 動態表格輸出與數據存儲
│
├── utils/                      # 輔助工具
│   ├── geometry.py             # 機翼系統幾何 (修正旋轉與比例邏輯)
│   └── visualization.py        # 強化版 CLI 可視化工具 (進度條, Turbo, GIF)
│
├── cases/                      # 統一的 CFD 案例
│   ├── lid_driven_cavity.py    # 方腔上蓋驅動流
│   ├── flow_over_cylinder.py   # 圓柱繞流 (卡門渦街)
│   └── airfoil.py              # 高升力機翼系統 (含煙線可視化)
│
├── run.py                      # 統一執行入口
└── legacy/                     # 舊版與實驗性實現
```

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
uv pip install taichi numpy matplotlib tabulate imageio
```

### 2. 執行模擬 (以機翼為例)

```bash
# 執行高升力機翼系統模擬 (AoA=12°, 含 50 萬粒子煙線)
python run.py airfoil --res 256 --aoa 12 --slat 15 --flap 15 --steps 5000
```

### 3. 可視化結果 (CLI 工具)

```bash
# 處理數據、生成圖片與 GIF (帶有動態進度條)
python utils/visualization.py output_airfoil --gif --type velocity
```

---

## 📊 輸出說明

### 即時動態監控
系統現在支援動態表格輸出，您可以即時看到每一列數據完美對齊地出現：
| step   | R_u      | R_v      | R_rho    | M_err    | Cd       | Cl       | Umax   | ETA      |
|--------|----------|----------|----------|----------|----------|----------|--------|----------|
| 100    | 1.23e-03 | 5.67e-04 | 2.34e-05 | 1.45e-07 | 0.0234   | 0.5678   | 0.0823 | 00:15:23 |

### 數據存儲
- **場數據**: `state_XXXXXX.npy` 包含 rho, u, mask, particles。
- **分析數據**: `history.npy` 包含完整殘差歷史與模擬參數。

---

## 📝 版本歷史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| v2.1 | 2026-01-23 | ✨ 整合粒子系統，修正機翼幾何邏輯，新增可視化 CLI 與進度條 |
| v2.0 | 2026-01-23 | 🎉 統一架構重構，模組化設計 |

---

## 🤝 貢獻與授權

歡迎提交 Pull Request！本專案採用 MIT 授權。

**最後更新**: 2026-01-23
**維護狀態**: ✅ Active Development