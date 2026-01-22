# 🌊 CFD Taichi: Unified LBM Solver

[![Taichi](https://img.shields.io/badge/backend-Taichi-red.svg)](https://taichi-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**CFD Taichi** 是一個基於 [Taichi Programming Language](https://github.com/taichi-dev/taichi) 開發的高效能流體力學求解器。本專案採用 **Lattice Boltzmann Method (LBM)** 框架，旨在提供一個高效、模組化且易於擴展的流體模擬平台。

---

## 🔥 為什麼選擇 Taichi？

本專案的核心計算引擎完全由 [Taichi](https://taichi-lang.org/) 驅動。Taichi 為高效能計算（HPC）提供了極佳的開發體驗：

*   **跨平台硬體加速**: 透過 Taichi 的 JIT 編譯技術，同一套代碼可無縫運行在 **Apple Metal**, **CUDA**, 與 **Vulkan** 後端，在 Apple Silicon M 系列晶片上擁有卓越的表現。
*   **Pythonic 高效能**: 讓開發者能以純 Python 的語法撰寫高性能的 GPU Kernel，大幅降低了實作複雜數值演算法（如 MRT-LBM）的門檻。
*   **靈活的記憶體佈局**: 透過 Taichi 的 SNode 系統，我們能針對不同的硬體架構優化場變數（Fields）的記憶體排列，最大化頻寬利用率。

---

## 🚀 核心優勢

*   🏗️ **統一核心架構**: 單一 `LBMSolver` 核心透過模組化設計（BC, Collision, Diagnostics），完美支援所有模擬場景。
*   🌪️ **進階物理模型**: 支援 **MRT-LBM** (Multiple-Relaxation-Time) 碰撞算子與 **Smagorinsky LES** (Large Eddy Simulation) 湍流模型。
*   💨 **粒子煙線系統**: 內建並行粒子系統，支援同時追蹤 50 萬個流體粒子，重現真實風洞中的煙線（Smoke Lines）可視化。
*   📊 **動態監控系統**: 具備即時殘差監控與質量守恆檢查，確保物理模擬的準確性。

---

## 🎨 經典案例 (Case Studies)

### 1. 方腔上蓋驅動流 (Lid-Driven Cavity)
典型的封閉系統流動，用於驗證邊界條件處理與渦流形成。是驗證 LBM 穩定性的基礎工具。

### 2. 圓柱繞流 (Flow Over Cylinder)
觀察流體在不同 Reynolds 數下的行為。當 $Re > 47$ 時，可觀測到週期性的 **卡門渦街 (Kármán Vortex Street)**。

### 3. 高升力機翼系統 (Multi-Element Airfoil)
模擬商用飛機起降階段的複雜氣動力配置（縫翼 + 主翼 + 襟翼）。精確模擬 **Slot (引流縫隙)** 效應，搭配粒子系統直觀觀察升力產生的物理過程。

---

## ⚙️ 安裝與執行

### 1. 環境準備
建議使用 `uv` 或 `pip` 安裝：
```bash
uv pip install taichi numpy matplotlib tabulate imageio
```

### 2. 執行模擬
```bash
# 執行機翼模擬
python run.py airfoil --res 256 --aoa 12 --re 1000 --steps 5000
```

### 3. 數據可視化
```bash
# 生成速度場圖片與 GIF 影片（附帶動態進度條）
python utils/visualization.py output_airfoil --gif --type velocity
```

---

## 🙏 致謝

特別感謝 [Taichi 團隊](https://github.com/taichi-dev) 開發了如此強大的程式語言，使得高性能物理模擬能以如此優雅的方式在個人電腦上運行。

*   **Taichi Repo**: [https://github.com/taichi-dev/taichi](https://github.com/taichi-dev/taichi)
*   **Documentation**: [https://docs.taichi-lang.org/](https://docs.taichi-lang.org/)

---
**開發者**: [latteine1217]
**維護狀態**: ✅ 主動開發中
**授權**: MIT License