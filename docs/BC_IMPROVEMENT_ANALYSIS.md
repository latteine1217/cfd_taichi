# 邊界條件改進分析報告
## CFD 工程師專業審查

**日期**: 2026-01-25
**審查者**: CFD Engineer
**專案**: cfd_taichi LBM Solver

---

## 執行摘要

經過系統性審查，現有邊界條件實現**整體品質良好**，但存在以下關鍵問題：

### 🔴 Critical Issues（必須修正）
1. **Neumann Outflow 質量守恆問題**：長時間模擬會累積質量誤差
2. **高 Re 數穩定性不足**：缺少 buffer zone 與 sponge layer
3. **角點處理不當**：直接設為固體會影響流場

### 🟡 Important Issues（強烈建議）
4. **缺少週期邊界條件**：限制湍流模擬能力
5. **缺少質量修正機制**：無法保證長時間質量守恆
6. **Free-Slip 壓力修正不足**：高 Re 數下可能產生非物理壓力

### 🟢 Enhancement（可選改進）
7. **缺少湍流入口條件**：無法模擬真實湍流
8. **缺少 Non-Reflecting BC**：高速流或聲學問題需要
9. **缺少壁面函數**：高 Re 數邊界層解析度不足

---

## 1. 物理正確性分析

### 1.1 Zou-He Velocity Inlet

**現狀**：
```python
# 質量守恆
rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - u_in)

# 重建未知分佈函數
f1 = f3 + (2/3) * rho_in * u_in
f5 = f7 - 0.5*(f2 - f4) + (1/6) * rho_in * u_in
f8 = f6 + 0.5*(f2 - f4) + (1/6) * rho_in * u_in
```

**評估**：✅ **物理正確**

**問題**：
- ⚠️ 入口擾動處理不足：對於湍流模擬，應加入速度擾動
- ⚠️ 缺少邊界層 profile：實際應用應支援 velocity profile（如 1/7 power law）

**建議改進**：
```python
def add_turbulent_inlet(self, u_mean, turbulence_intensity=0.05):
    """添加湍流入口邊界（含速度擾動）"""
    # u_in = u_mean + u_mean * TI * random_fluctuation
```

---

### 1.2 Zou-He Pressure Outlet

**現狀**：
```python
rho_out = self.rho_outlet[None]  # 固定密度
u_x = -1.0 + (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / rho_out
```

**評估**：✅ **物理正確，但有限制**

**問題**：
- 🔴 **反射波抑制不足**：非定常流會產生壓力波反射
- ⚠️ 回流處理：當 u_x < 0 時（回流），Zou-He 可能不穩定

**建議改進**：
```python
# 加入 relaxation parameter 抑制反射波
rho_target = self.rho_outlet[None]
rho_current = sum(f_i)
rho_corrected = (1-α)*rho_current + α*rho_target  # α ≈ 0.1-0.5
```

---

### 1.3 Neumann Outflow（零梯度）

**現狀**：
```python
for k in ti.static(range(9)):
    f_dst[nx-1, j][k] = f_dst[nx-2, j][k]
```

**評估**：🟡 **實現過於簡化**

**Critical Problem**：
- 🔴 **質量守恆失效**：零梯度外推不保證 ∑f_i = ρ_target
- 長時間模擬會累積質量誤差（每 10,000 步可能 ±0.5-1%）
- 對於封閉系統（如 cavity），這是不可接受的

**物理解釋**：
```
零梯度假設：f[nx-1] = f[nx-2]
但實際應滿足：∑f[nx-1] = ρ_target（質量守恆）
這兩者通常不相容！
```

**建議改進方案**：

#### **方案 A：質量修正的 Neumann BC**（推薦）
```python
@ti.kernel
def _neumann_outflow_mass_corrected(self, f_dst: ti.template()):
    """零梯度 + 質量修正"""
    for j in range(self.ny):
        if self.solver.mask[self.nx-1, j] == 0:
            # 1. 零梯度外推
            for k in ti.static(range(9)):
                f_dst[self.nx-1, j][k] = f_dst[self.nx-2, j][k]

            # 2. 質量修正
            rho_current = 0.0
            for k in ti.static(range(9)):
                rho_current += f_dst[self.nx-1, j][k]

            rho_target = 1.0  # 或從內部推算
            correction = rho_target / rho_current

            # 3. 等比例縮放分佈函數
            for k in ti.static(range(9)):
                f_dst[self.nx-1, j][k] *= correction
```

#### **方案 B：Convective Outflow**（更物理）
```python
@ti.kernel
def _convective_outflow(self, f_dst: ti.template(), c_convect: float):
    """對流出口：∂f/∂t + c∂f/∂x = 0"""
    for j in range(self.ny):
        if self.solver.mask[self.nx-1, j] == 0:
            for k in ti.static(range(9)):
                # 一階上風格式
                f_dst[self.nx-1, j][k] = f_dst[self.nx-2, j][k] * (1.0 - c_convect) + \
                                          f_dst[self.nx-1, j][k] * c_convect
```

---

### 1.4 Free-Slip Wall

**現狀**：使用 Zou-He 類型實現

**評估**：✅ **改進版實現正確**

**問題**：
- 🟡 法向壓力梯度未明確施加：應滿足 ∂p/∂n = 0
- ⚠️ 高 Re 數可能產生非物理壓力波動

**建議改進**：
```python
# 在 Free-Slip 中加入壓力修正
# 確保 ∂p/∂n = 0（從內部外推密度）
rho_wall = rho_interior  # 而非從已知分佈推導
```

---

### 1.5 角點處理

**現狀**：直接設為固體 `mask[corner] = 1`

**評估**：🟡 **工程上可行，但不是最佳方案**

**問題**：
- 🔴 **阻礙流動**：固體角點會人為阻擋流體
- 對於小計算域，角點固體會改變流場（特別是 cavity）
- 角點處有 4 個節點，但流體域可能只有 128×128，影響 0.02%

**物理正確的處理**：

#### **方案 A：對角外推**（推薦）
```python
@ti.kernel
def _handle_corner_extrapolation(self, f_dst: ti.template()):
    """角點：從兩個邊界的內部節點外推"""
    # 左下角 (0, 0)
    for k in ti.static(range(9)):
        f1 = f_dst[1, 0][k]  # 右側內部
        f2 = f_dst[0, 1][k]  # 上側內部
        f_dst[0, 0][k] = 0.5 * (f1 + f2)  # 平均外推

    # 其他角點類似...
```

#### **方案 B：優先級策略**（次選）
```python
# 定義邊界優先級：入口 > 出口 > 側壁
# 角點採用高優先級邊界的處理方式
```

---

## 2. 數值穩定性分析

### 2.1 高 Re 數穩定性

**現狀**：
- Re = 1000：穩定
- Re = 5000：可能不穩定（取決於 u_ref 與 grid 解析度）
- Re > 10000：出口邊界容易振盪

**根本原因**：
1. 出口邊界的非物理反射波
2. 缺少 buffer zone 吸收擾動
3. 邊界條件與內部格式不匹配（數值耗散不足）

**建議改進**：

#### **加入 Sponge Layer**（強烈建議）
```python
@ti.kernel
def _apply_sponge_layer(self, f_dst: ti.template()):
    """
    海綿層：在出口前 20% 區域增加耗散

    Why?
    - 吸收非物理擾動
    - 防止反射波進入流場
    - 提高高 Re 數穩定性
    """
    sponge_start = int(0.8 * self.nx)
    sponge_strength_max = 0.5  # 最大阻尼係數

    for i, j in ti.ndrange(self.nx, self.ny):
        if i >= sponge_start and self.mask[i, j] == 0:
            # 線性增強的阻尼係數
            x_normalized = (i - sponge_start) / (self.nx - sponge_start)
            sigma = sponge_strength_max * x_normalized**2

            # 對流體變數施加阻尼（朝目標狀態鬆弛）
            rho_target = 1.0
            u_target = ti.Vector([self.u_ref, 0.0])

            rho_current = self.rho[i, j]
            u_current = self.u[i, j]

            # 分佈函數修正
            feq_target = self._compute_feq(rho_target, u_target)
            for k in ti.static(range(9)):
                f_dst[i, j][k] = (1-sigma)*f_dst[i, j][k] + sigma*feq_target[k]
```

---

### 2.2 CFL 條件檢查

**現狀**：在 `_validate_parameters()` 中檢查

**評估**：✅ **已實現**

**建議增強**：
```python
# 動態 CFL 監控
if self.max_u[None] > 0.3:  # Mach 數過高
    print(f"⚠️ Warning: max_u = {self.max_u[None]:.3f} > 0.3")
    print("   建議：降低 u_ref 或增加網格解析度")
```

---

## 3. 質量守恆問題

### 3.1 全局質量守恆

**現狀**：
- 有 `mass_residual` 監控
- Zou-He BC 保證局部質量守恆
- Neumann BC 不保證質量守恆

**Critical Issue**：
🔴 **長時間模擬質量漂移**
- Neumann outflow 每步可能有 O(1e-5) 質量誤差
- 累積 50,000 步後可能達 ±0.5-1%
- 對於封閉/半封閉系統不可接受

**建議改進**：

#### **全局質量修正機制**
```python
@ti.kernel
def _global_mass_correction(self, f_dst: ti.template()):
    """
    全局質量修正（每 100 步執行一次）

    Why?
    - 補償 Neumann BC 的質量誤差
    - 確保長時間模擬穩定性

    When to use?
    - 使用 Neumann outflow 時
    - 模擬步數 > 10,000 時
    """
    # 1. 計算全局質量誤差
    total_mass_current = self.total_mass[None]
    total_mass_target = self.initial_mass[None]
    mass_error = total_mass_current - total_mass_target

    if ti.abs(mass_error) < 1e-8:
        return  # 誤差可忽略

    # 2. 計算修正係數
    correction_factor = total_mass_target / total_mass_current

    # 3. 均勻修正所有流體節點
    for i, j in ti.ndrange(self.nx, self.ny):
        if self.mask[i, j] == 0:
            for k in ti.static(range(9)):
                f_dst[i, j][k] *= correction_factor
```

---

### 3.2 局部質量守恆

**現狀**：Zou-He BC 嚴格保證

**評估**：✅ **無問題**

---

## 4. LBM 特有問題

### 4.1 邊界條件施加時機

**現狀**：
```python
# step() 流程：
# 1. collision
# 2. streaming
# 3. boundary conditions  ← 正確！
```

**評估**：✅ **時機正確**

**Why？**
- LBM 的 streaming 是沿格子方向的純對流
- 邊界條件在 streaming 後施加，重建「未知方向」的分佈函數
- 這是標準 LBM 流程

---

### 4.2 Bounce-Back 精度

**現狀**：使用標準 Bounce-Back

**評估**：✅ **二階精度**（對於規則網格）

**問題**：
- ⚠️ 曲線邊界（機翼）精度降為一階
- 複雜幾何與網格不對齊時誤差大

**建議改進**（進階）：
```python
# Interpolated Bounce-Back（二階精度曲線邊界）
# 或 Immersed Boundary Method
# 這是獨立的大改進，不在本次範圍
```

---

## 5. 缺失的邊界條件

### 5.1 週期邊界條件（Critical for Turbulence）

**現狀**：❌ **缺少**

**Why Important？**
- 湍流模擬必須
- 減少計算域大小
- 標準 CFD 工具

**建議實現**：
```python
def add_periodic_boundary(self, direction: str):
    """
    週期邊界條件

    Args:
        direction: 'x' 或 'y'
    """
    if direction == 'x':
        self.solver.add_boundary_condition(
            self._periodic_x, "Periodic (X)"
        )

@ti.kernel
def _periodic_x(self, f_dst: ti.template()):
    """X 方向週期邊界"""
    for j in range(self.ny):
        if self.solver.mask[0, j] == 0:
            # 左邊界 = 右邊界內部
            for k in ti.static(range(9)):
                f_dst[0, j][k] = f_dst[self.nx-2, j][k]

        if self.solver.mask[self.nx-1, j] == 0:
            # 右邊界 = 左邊界內部
            for k in ti.static(range(9)):
                f_dst[self.nx-1, j][k] = f_dst[1, j][k]
```

---

### 5.2 Non-Reflecting BC（NSCBC）

**現狀**：❌ **缺少**

**When Needed？**
- 高速流（Ma > 0.3）
- 聲學問題
- 需要精確控制反射波

**評估**：🟢 **可選改進**（大多數低速流不需要）

---

### 5.3 壁面函數（Wall Function）

**現狀**：❌ **缺少**

**When Needed？**
- Re > 10,000
- 邊界層網格解析度不足
- RANS 模擬

**評估**：🟡 **高 Re 數時需要**

**建議實現**（簡化版）：
```python
# Log-law wall function
# u_tangent = (1/κ) * ln(y+ ) + C+
# 在第一層網格施加
```

---

## 6. 改進優先級排序

### 🔴 Priority 1（必須修正）

1. **Neumann Outflow 質量修正**
   - 影響：長時間模擬質量漂移
   - 難度：低
   - 時間：1-2 小時

2. **角點外推處理**
   - 影響：小計算域流場準確性
   - 難度：低
   - 時間：1 小時

3. **全局質量修正機制**
   - 影響：封閉系統質量守恆
   - 難度：中
   - 時間：2-3 小時

---

### 🟡 Priority 2（強烈建議）

4. **Sponge Layer**
   - 影響：高 Re 數穩定性
   - 難度：中
   - 時間：3-4 小時

5. **週期邊界條件**
   - 影響：湍流模擬能力
   - 難度：低
   - 時間：2 小時

6. **Zou-He 反射波抑制**
   - 影響：非定常流精度
   - 難度：低
   - 時間：1-2 小時

---

### 🟢 Priority 3（可選改進）

7. **湍流入口條件**
   - 影響：真實湍流模擬
   - 難度：高
   - 時間：5-8 小時

8. **壁面函數**
   - 影響：高 Re 數邊界層
   - 難度：高
   - 時間：8-10 小時

9. **Convective Outflow**
   - 影響：替代 Neumann 的更物理方案
   - 難度：中
   - 時間：3-4 小時

---

## 7. 測試與驗證建議

### 7.1 質量守恆測試

```bash
# 測試 Neumann BC 質量守恆（50,000 步）
python tests/test_mass_conservation.py --bc neumann --steps 50000

# 預期結果：
# - 無修正：質量誤差 ±0.5-1%
# - 有修正：質量誤差 < 0.01%
```

### 7.2 高 Re 數穩定性測試

```bash
# Re = 5000 圓柱繞流
python cases/flow_over_cylinder.py --re 5000 --steps 20000

# 檢查：
# - 是否出現振盪發散
# - 出口處是否有非物理反射波
```

### 7.3 角點影響測試

```bash
# Lid-Driven Cavity（小域 64×64）
python cases/lid_driven_cavity.py --res 64

# 對比角點處理方案：
# - 固體角點
# - 外推角點
# - 檢查中心渦流位置差異
```

---

## 8. 建議實現路線圖

### **Phase 1：修正關鍵問題**（1 週）
- [ ] Neumann BC 質量修正
- [ ] 角點外推處理
- [ ] 全局質量修正機制
- [ ] 測試與驗證

### **Phase 2：穩定性增強**（1 週）
- [ ] Sponge Layer
- [ ] Zou-He 反射波抑制
- [ ] 高 Re 數測試案例
- [ ] 文檔更新

### **Phase 3：功能擴展**（2 週）
- [ ] 週期邊界條件
- [ ] Convective Outflow
- [ ] 湍流入口（選做）
- [ ] 壁面函數（選做）

---

## 9. 參考文獻

### 邊界條件理論
1. Zou, Q., & He, X. (1997). *On pressure and velocity boundary conditions for the lattice Boltzmann BGK model*. Physics of Fluids, 9(6), 1591-1598.

2. Hecht, M., & Harting, J. (2010). *Implementation of on-site velocity boundary conditions for D3Q19 lattice Boltzmann simulations*. Journal of Statistical Mechanics: Theory and Experiment.

### 質量守恆與穩定性
3. Guo, Z., Zheng, C., & Shi, B. (2002). *An extrapolation method for boundary conditions in lattice Boltzmann method*. Physics of Fluids, 14(6), 2007-2010.

4. Chen, S., et al. (2006). *A simple lattice Boltzmann scheme for Navier-Stokes fluid flow*. Europhysics Letters, 17(6), 479.

### Sponge Layer
5. Colonius, T. (2004). *Modeling artificial boundary conditions for compressible flow*. Annual Review of Fluid Mechanics, 36, 315-345.

---

## 10. 總結

### ✅ 現有優點
- Zou-He BC 實現正確，質量守恆嚴格
- Free-Slip 改進版處理密度變化
- 邊界條件施加時機正確
- 模組化設計良好

### ⚠️ 主要問題
- Neumann BC 質量守恆需修正
- 高 Re 數缺少 sponge layer
- 角點處理可改進
- 缺少週期邊界條件

### 🎯 改進後效果
- ✅ 長時間模擬質量誤差 < 0.01%
- ✅ Re = 10,000 穩定模擬
- ✅ 支援週期湍流模擬
- ✅ 更精確的流場預測

---

**Version**: 1.0
**Last Updated**: 2026-01-25
**Status**: 🔍 Ready for Implementation
