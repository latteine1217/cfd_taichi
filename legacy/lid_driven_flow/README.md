# Lid-Driven Cavity Flow Solver (Taichi + LBM)

本專案使用 **Taichi** (Metal Backend) 實作 **Lattice Boltzmann Method (LBM)** 來模擬二維 Lid-Driven Cavity Flow (頂蓋驅動方腔流)。

專案強調物理正確性與數值穩定性，結合了 **MRT (Multiple Relaxation Time)** 碰撞算子與 **LES (Large Eddy Simulation)** 湍流模型，適用於高雷諾數模擬。

## 特色算法

1.  **Lattice Boltzmann Method (D2Q9)**:
    *   基於 D2Q9 晶格模型。
    *   使用 **MRT (Multiple Relaxation Time)** 碰撞算子，相比 BGK 具有更好的數值穩定性與精度。

2.  **Boundary Conditions (邊界條件)**:
    *   **Half-way Bounce-back**: 二階精度的反彈邊界，用於靜止壁面。
    *   **Regularized Moving Wall**: 頂蓋速度採用 $U(x) = U_{lid} 	imes (1 - \frac{\cosh(10(x-0.5))}{\cosh(5)})$ 進行正則化，消除了角落速度不連續導致的奇異點 (Singularity)。

3.  **Large Eddy Simulation (LES)**:
    *   實作 **Smagorinsky-Lilly Model**。
    *   動態調整局部鬆弛時間 ($	au_{eff}$)，在高應變率區域引入渦流黏滯性 (Eddy Viscosity) 以穩定高雷諾數流場。

4.  **Backend Optimization**:
    *   使用 Taichi Metal Backend (`ti.metal`) 在 macOS 上進行 GPU 加速。
    *   全流程使用 `float32` 精度運算。

## 檔案結構

*   `solver.py`: 核心求解器。包含 LBM 算法、MRT 碰撞、LES 模型與邊界處理。
*   `visualization.py`: 視覺化工具。將輸出的 `.npy` 數據轉換為速度場熱圖與流線圖 GIF。

## 安裝依賴

需安裝 `taichi`, `numpy`, `matplotlib`, `imageio`。

```bash
pip install taichi numpy matplotlib imageio
```

## 收斂標準 (Convergence Criteria)

本專案採用正規化殘差 (Normalized Residuals) 來判斷流場是否收斂。當殘差降至 $10^{-5}$ 以下時，可認為流場已達到高度穩態。

| 物理量 | 一般工程標準 | 高精度要求 |
| :--- | :--- | :--- |
| 連續方程 (Continuity) | $10^{-3}$ | $10^{-5} \sim 10^{-6}$ |
| 動量方程 (Velocity) | $10^{-3}$ | $10^{-5} \sim 10^{-6}$ |
| 湍流模型 (LES) | $10^{-3}$ | $10^{-4}$ |

在 `solver.py` 的 CLI 輸出中，殘差已針對第 100 步的初始值進行縮放 (Scaled)。

## 使用方法

### 1. 執行模擬 (Solver)

使用 `solver.py` 進行計算。支援多種 CLI 參數以控制物理條件。

**基本指令：**
```bash
python lid_driven_flow/solver.py
```

**進階指令 (高雷諾數範例)：**
```bash
python lid_driven_flow/solver.py --res 512 --re 5000 --steps 100000 --out_dir output_re5000 --cs 0.16
```

**參數說明：**
*   `--res`: 網格解析度 (NxN)。預設 `256`。
*   `--re`: 雷諾數 (Reynolds Number)。預設 `1000.0`。
*   `--steps`: 總模擬步數。預設 `50000`。
*   `--interval`: 存檔頻率。預設 `1000`。
*   `--lid_vel`: 頂蓋驅動速度 (Lattice Units)。建議小於 0.3 以滿足不可壓縮假設。預設 `0.1`。
*   `--cs`: Smagorinsky 常數 (LES)。設為 `0` 則關閉 LES。預設 `0.16`。
*   `--tol`: 收斂門檻 (Tolerance)。當正規化殘差低於此值時自動停止。預設 `1e-3`。
*   `--out_dir`: 輸出檔案目錄。預設 `output`。

程式執行時會即時顯示 **Momentum Residual** (動量殘差) 與 **Mass** (總質量)，用於監控收斂性與守恆性。

### 2. 視覺化 (Visualization)

使用 `visualization.py` 讀取模擬結果並生成動畫。

**基本指令：**
```bash
python lid_driven_flow/visualization.py
```

**指定輸入目錄：**
```bash
python lid_driven_flow/visualization.py --in_dir output_re5000 --out_dir vis_re5000 --fps 20
```

**參數說明：**
*   `--in_dir`: 讀取 `.npy` 檔的目錄。預設 `output`。
*   `--out_dir`: 輸出圖片與 GIF 的目錄。預設 `vis_output`。
*   `--fps`: GIF 動畫的幀率。預設 `10`。

## 結果輸出

*   **Solver**: 在 `output/` (或指定目錄) 生成 `rho_*.npy` (密度場) 與 `u_*.npy` (速度場)。
*   **Visualization**: 在 `vis_output/` 生成 `frame_*.png` (單幀圖) 與 `flow_animation.gif` (完整流場動畫)。
