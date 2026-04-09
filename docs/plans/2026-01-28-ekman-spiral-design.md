# Ekman 螺旋多層模擬器設計文檔

**日期**: 2026-01-28
**作者**: latteine1217
**狀態**: 設計完成，待實現

---

## 1. 概述

### 目標

模擬風驅動的海洋表面流動，捕捉科氏力引起的 **Ekman 螺旋現象**：
- 表面水流偏離風向約 45°
- 深度越深，偏轉角度越大
- 速度隨深度指數衰減
- 形成順時針旋轉的速度剖面（北半球）

### 物理背景

經典 Ekman 理論描述旋轉參考系下風驅動的海洋邊界層流動。關鍵物理過程：
1. 風應力驅動表面水體
2. 科氏力使流動向右偏轉（北半球）
3. 垂直剪應力向下傳遞動量
4. 各層累積不同的偏轉角度
5. 底部摩擦阻尼深層流動

### 實現策略

採用 **多層 2D LBM 近似**（方案 A）：
- 垂直方向離散為 20 層
- 每層使用獨立的 D2Q9 LBM 求解器
- 層間通過顯式剪應力耦合
- 使用現有 `LBMSolver` 核心架構

**選擇理由**：
- ✅ 快速開發（使用現有框架）
- ✅ 可驗證概念（與解析解對比）
- ✅ 記憶體可控（M3 16GB 限制）
- ❌ 非真實 3D（物理近似）

---

## 2. 物理參數

### 地球物理參數

| 參數 | 符號 | 數值 | 單位 | 說明 |
|------|------|------|------|------|
| 緯度 | φ | 45° | degree | 中緯度 |
| 科氏參數 | f | 1.0×10⁻⁴ | s⁻¹ | 2Ω sin(φ) |
| 慣性週期 | T_i | 17.5 | hr | 2π/f |

### 流體性質

| 參數 | 符號 | 數值 | 單位 | 說明 |
|------|------|------|------|------|
| 海水密度 | ρ | 1025 | kg/m³ | 典型值 |
| 分子黏度 | ν | 1.0×10⁻⁶ | m²/s | 運動黏度 |
| 垂直渦黏度 | ν_v | 1.0×10⁻³ | m²/s | 湍流混合 |

**關鍵比值**：
- 垂直/水平黏度比：ν_v / ν ≈ 1000（各向異性）
- Ekman 深度：D_E = π√(2ν_v/f) ≈ 100 m

### 垂直結構

| 參數 | 數值 | 說明 |
|------|------|------|
| 層數 | 20 | 中等解析度 |
| 總深度 | 100 m | ≈ Ekman 深度 |
| 層厚度 | 5 m | 均勻分層 |

### 風應力與底摩擦

| 參數 | 符號 | 數值 | 單位 | 說明 |
|------|------|------|------|------|
| 10m 風速 | U_10 | 10 | m/s | 中等風速 |
| 阻力係數 | C_d | 1.3×10⁻³ | - | 典型值 |
| 風應力 | τ_w | 0.156 | N/m² | ρ_air C_d U_10² |
| 底摩擦係數 | r | 1.0×10⁻⁴ | s⁻¹ | 線性拖曳 |

### 水平網格

| 參數 | 數值 | 說明 |
|------|------|------|
| nx × ny | 256 × 256 | 或 512×256 |
| 邊界條件 | 週期性 | 理想化海洋 |

---

## 3. 數學模型

### 控制方程

每層滿足旋轉參考系下的 2D Navier-Stokes 方程：

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + F_coriolis + F_shear
```

其中：
- **科氏力**：`F_coriolis = (f·v, -f·u)`
- **剪應力力**：`F_shear = ∂τ_z/∂z`

### 層間耦合

**剪應力計算**（第 k 層）：

```python
# 上界面剪應力
if k == 0:
    # 頂層：風應力
    τ_top = τ_wind
else:
    # 黏性剪應力
    τ_top = ρ ν_v (u[k-1] - u[k]) / dz

# 下界面剪應力
if k == n_layers - 1:
    # 底層：底摩擦
    τ_bottom = -ρ r u[k] dz
else:
    # 黏性剪應力
    τ_bottom = ρ ν_v (u[k] - u[k+1]) / dz

# 剪應力梯度 → 體積力
F_shear[k] = (τ_top - τ_bottom) / (ρ dz)
```

**時間推進順序**：

```
For each time step:
  1. 計算科氏力：F_coriolis[k] = (f·v[k], -f·u[k])
  2. 計算層間剪應力：τ[k] = ρ ν_v Δu/dz
  3. 計算剪應力梯度：F_shear[k]
  4. 施加風應力（頂層）+ 底摩擦（底層）
  5. 總外力：F_total[k] = F_coriolis[k] + F_shear[k]
  6. 每層獨立執行 LBM step（含外力）
```

### 解析解（驗證用）

經典 Ekman 螺旋解析解（無限深，恆定風應力）：

```
u(z) = u_0 exp(-z/D_E) cos(π/4 - z/D_E)
v(z) = u_0 exp(-z/D_E) sin(π/4 - z/D_E)
```

其中：
- `u_0 = τ_wind / (ρ √(ν_v f))`：表面速度幅值
- `D_E = π √(2ν_v/f)`：Ekman 深度
- z = 0：海洋表面

**關鍵特徵**：
- 表面流偏離風向 45°
- 速度在 D_E 處衰減至 e^(-π) ≈ 4% 表面值
- Hodograph 呈對數螺旋

---

## 4. 實現架構

### 文件結構

```
examples/ekman_spiral.py          # 主模擬腳本（單一檔案）
```

### 核心類

```python
class MultiLayerEkmanSolver:
    """
    多層 Ekman 螺旋求解器

    What: 垂直離散化的 2D LBM 多層模型
    Why: 捕捉風驅動海洋流動的 Ekman 螺旋結構
    When: 需要模擬科氏力影響下的深度依賴性流動

    Attributes:
        n_layers: 垂直層數
        layers: List[LBMSolver]，每層的 2D 求解器
        shear_stress: 層間剪應力場
        coriolis_force: 科氏力場
    """

    def __init__(self, n_layers, nx, ny, f, nu_v, tau_wind, r_bottom, dz)

    @ti.kernel
    def compute_shear_stress(self)
        # 計算所有層間剪應力

    @ti.kernel
    def compute_coriolis_force(self)
        # 計算所有層的科氏力

    def apply_vertical_coupling(self)
        # 將層間耦合轉換為體積力

    def step(self)
        # 單步時間推進

    def get_velocity_profile(self) -> np.ndarray
        # 提取垂直速度剖面

    def compute_ekman_transport(self) -> (float, float, float)
        # 計算 Ekman 傳輸

    def save_state(self, step: int)
        # 保存狀態
```

### 邊界條件

**水平方向**（每層）：
- 週期性邊界（periodic）
- 模擬無限大海洋

**垂直方向**：
- 頂層（k=0）：風應力驅動
- 底層（k=19）：線性底摩擦
- 中間層：剪應力耦合

### 初始條件

```python
# 推薦：靜止海洋
u[:, :, :] = 0.0
rho[:, :, :] = 1.0

# 或：加入小擾動加速收斂
u[:, :, :] = 0.01 * wind_direction + random_noise(0.001)
```

---

## 5. 診斷與可視化

### 關鍵診斷量

**1. Ekman 剖面**（空間平均的垂直速度）

```python
u_profile[k] = mean(u[k, :, :])  # (n_layers, 2)
```

**2. Ekman 傳輸**（垂直積分）

```python
M_x = Σ u[k] * dz
M_y = Σ v[k] * dz
transport_angle = atan2(M_y, M_x)  # 應 ≈ 90°（垂直風向）
```

**3. 動能**

```python
KE[k] = 0.5 * ρ * mean(u[k]² + v[k]²)
```

**4. 質量守恆**

```python
mass_error[k] = |M(t) - M(0)| / M(0)  # 每層獨立檢查
```

### CLI 輸出格式

```
| step  | time(hr) | u_surf | angle_surf | u_bot  | KE_total | transport_angle | mass_err |
|-------|----------|--------|------------|--------|----------|-----------------|----------|
| 1000  | 2.5      | 0.087  | 43.2°      | 0.003  | 0.152    | 89.7°           | 2.3e-07  |
```

### 可視化輸出

**1. Hodograph**（速度矢量圖）
- X軸：東向速度 u
- Y軸：北向速度 v
- 每個點代表一層
- 經典 Ekman 螺旋：順時針對數螺旋

**2. 深度剖面**
- 速度大小 vs 深度
- 速度方向 vs 深度
- 與解析解對比

**3. 水平切片**（某層的 2D 流場）
- 速度矢量場
- 渦度場

**4. 時間演化動畫**
- Hodograph 動畫
- 粒子追蹤（每層不同顏色）

---

## 6. 驗證策略

### 階段 1：單點驗證（0D+垂直）

```python
# 設定 nx=1, ny=1，退化為垂直一維問題
solver = MultiLayerEkmanSolver(
    n_layers=20, nx=1, ny=1, ...
)

# 運行至穩態
solver.run(steps=50000)

# 與解析解對比
u_num, v_num = solver.get_velocity_profile()
u_ana, v_ana = ekman_analytical_solution(...)

# 相對誤差應 < 10%
error = |u_num - u_ana| / max(u_ana)
assert error.mean() < 0.10
```

### 階段 2：物理一致性檢查

**質量守恆**：
```python
for k in range(n_layers):
    assert layers[k].mass_error < 1e-6
```

**Ekman 傳輸方向**：
```python
# 理論：垂直於風向（90°）
transport_angle = compute_transport_angle()
assert abs(transport_angle - 90.0) < 5.0  # ±5° 容差
```

**表面流偏角**：
```python
# 理論：偏離風向 45°
surface_angle = atan2(v_surf, u_surf) * 180/pi
assert abs(surface_angle - 45.0) < 10.0  # ±10° 容差
```

**速度衰減**：
```python
# Ekman 深度處速度 < 10% 表面值
u_at_D_e = u_profile[int(D_E / dz)]
assert u_at_D_e / u_surf < 0.1
```

### 成功標準

```
✅ 質量守恆誤差 < 1e-6（每層）
✅ 表面流偏角 = 45° ± 10°
✅ Ekman 傳輸方向 = 90° ± 5°
✅ 速度剖面與解析解誤差 < 10%
✅ 穩態時間 < 3 個慣性週期（~52 小時）
✅ Hodograph 呈現順時針螺旋
```

---

## 7. 預期結果

### 時間演化

**初期（0 - 0.5 T_i）**：
- 表面快速加速
- 深層幾乎靜止

**中期（0.5 - 2 T_i）**：
- 螺旋結構逐漸形成
- 動量向下傳播

**後期（2 - 3 T_i）**：
- 達到準穩態
- Hodograph 穩定為對數螺旋

### 空間結構

**垂直方向**：
- 表面（k=0）：u_surf ≈ 0.06 m/s，偏離風向 45°
- 中層（k=10, 50m）：速度 ≈ 0.02 m/s，偏轉角 ≈ 135°
- 底層（k=19, 95m）：速度 ≈ 0.001 m/s，接近靜止

**水平方向**：
- 由於週期性邊界，應保持空間均勻
- 任意擾動會觸發慣性振盪（週期 = T_i）

---

## 8. 已知限制

### 物理近似

1. **非真實 3D**：層間耦合是離散近似，非連續的垂直導數
2. **各向異性假設**：ν_v >> ν_h（垂直混合主導）
3. **線性底摩擦**：實際海底邊界層更複雜

### 數值限制

1. **層數限制**：20 層可能不足以完全解析 Ekman 深度
2. **時間步長**：需滿足 CFL 與慣性穩定性（dt < 0.1/f）
3. **數值擴散**：LBM 固有擴散可能污染層間耦合

### 記憶體需求

```
估算（256×256 網格，20 層）：
- 每層 LBM 場：9 × 256 × 256 × 4 bytes ≈ 2.4 MB
- 20 層：~48 MB
- 加上剪應力、科氏力等輔助場：~100 MB
- 總計：< 500 MB（遠低於 16GB 限制）
```

---

## 9. 後續擴展方向

### 短期（Phase 2）

- [ ] 增加層數至 50 層（提升解析度）
- [ ] 時變風場（旋轉風、陣風）
- [ ] 粒子追蹤可視化（每層不同顏色）

### 中期（Phase 3）

- [ ] 封閉盆地模擬（觀察環流）
- [ ] β 平面效應（緯度依賴的科氏參數）
- [ ] 熱驅動浮力（密度分層）

### 長期（Phase 4）

- [ ] 完整 3D LBM（D3Q19）
- [ ] 地形影響（斜坡底部）
- [ ] 與觀測數據對比（實際海洋剖面）

---

## 10. 參考資料

### 理論基礎

- Ekman, V. W. (1905). "On the influence of the earth's rotation on ocean-currents"
- Cushman-Roisin & Beckers (2011). *Introduction to Geophysical Fluid Dynamics*
- Vallis (2017). *Atmospheric and Oceanic Fluid Dynamics*

### LBM 實現

- 專案 `CLAUDE.md`：邊界條件、外力實現
- `core/lbm_solver.py`：D2Q9 MRT-LBM 核心
- `examples/lid_driven_cavity.py`：參考架構

---

**最後更新**: 2026-01-28
**下一步**: 進入實現階段（使用 `superpowers:writing-plans`）
