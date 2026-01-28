# Debug Playbook: Solver Stability Degradation Analysis

**任務編號**: debug_stability_001  
**建立時間**: 2026-01-27  
**狀態**: 分析完成  
**分析者**: Debug Specialist Agent

---

## 1. 背景（Background）

### 1.1 問題描述
近期修改後，LBM solver 穩定性顯著下降，可能表現為：
- 質量守恆誤差增加（mass_error > 1e-6）
- 速度場震盪或發散
- CFL 條件違反頻率增加
- NaN 出現於流場

### 1.2 Recent Changes（觸發因素）
根據用戶提供的資訊，近期關鍵修改包括：

1. **Gradient-based LES** (Line 766-818)
   - 使用速度梯度計算 Smagorinsky 渦黏度
   - 直接有限差分計算 `du/dx`, `dv/dy`, `du/dy`, `dv/dx`
   
2. **step() 調用順序變更** (Line 1010-1034)
   - 現在在 `_collide_and_stream()` **之前**調用：
     - `_update_macro(f_src)` (Line 1024)
     - `_update_smagorinsky_viscosity()` (Line 1025)
   
3. **移除 Out-of-Bounds Bounce-Back** (Line 732-734)
   - 舊版：超出邊界時執行 bounce-back
   - 新版：註釋「由邊界條件負責重建」
   
4. **CFL check 調用 _update_macro** (Line 1070)
   - `check_cfl_condition()` 中重複計算巨觀量
   
5. **Diagnostics 變更** (Line 820-873)
   - `_update_diagnostics()` 單次遍歷優化

---

## 2. 根本原因分析（Root Cause Analysis）

### 🔴 **P0: 時間步內多次 _update_macro() 調用（Critical）**

**問題機制**：
```python
# step() 執行順序：
def step(f_src, f_dst):
    self._update_macro(f_src)               # ← 第一次計算 rho, u
    self._update_smagorinsky_viscosity()    # ← 使用上述 rho, u
    self._collide_and_stream(f_src, f_dst) # ← 使用 self.rho, self.u（但基於 f_src）
    bc_func(f_dst)                          # ← 邊界條件
```

**根本衝突**：
1. **Line 1024**: `_update_macro(f_src)` 計算 `self.rho[i,j]`, `self.u[i,j]`
2. **Line 678-679**: `_collide_and_stream()` 內部**再次讀取** `self.rho[i,j]`, `self.u[i,j]`
3. **時間不一致**：
   - Smagorinsky 使用的是 `f_src` 對應的速度場
   - 但 `_collide_and_stream()` 內部**假設** `self.rho/u` 已由 `f_src` 更新

**為何導致不穩定**：
- **舊版**（假設）：`_collide_and_stream()` 內部自行計算區域 `rho`, `u`（Line 676-679 的註釋暗示）
- **新版**：依賴外部預計算的 `self.rho/u`，但在雙緩衝切換時可能讀到**上一步的殘留值**

**數值後果**：
- MRT 碰撞算子使用錯誤的 `current_rho`, `current_u` → 平衡分佈函數 `meq` 錯誤
- Effective viscosity `tau_eff` 基於錯誤的應變率 → 耗散率不正確
- 累積誤差導致質量/動量守恆違反

---

### 🟠 **P1: Gradient-based LES 邊界處理不當（High）**

**問題位置**: Line 792-799

```python
# 邊界處理邏輯：
if self.mask[i_plus, j] == 1:
    u_ip = self.u[i, j]  # ← 單側差分 fallback
if self.mask[i_minus, j] == 1:
    u_im = self.u[i, j]
```

**問題**：
1. **固體邊界附近**：使用 `self.u[i,j]` 替代固體節點速度，等效於**單側差分**
2. **開放邊界附近**：`i_plus = min(i+1, nx-1)` 在邊界處會使用**同一點兩次**
   - 例如 `i = nx-1` 時，`i_plus = nx-1`，導致 `du/dx = 0`（虛假結果）

**物理後果**：
- **邊界層附近**：應變率 `|S|` 被低估 → `nu_sgs` 偏小 → 湍流耗散不足
- **出口邊界**：梯度計算錯誤 → 渦黏度突變 → 壓力/速度振盪
- **分離點**：固體邊界處梯度不連續 → 數值噪聲

**與舊版差異**：
- 舊版（moment-based）使用非平衡矩量，在邊界處較平滑
- 新版直接差分速度場，對邊界處理敏感

---

### 🟠 **P2: 移除 Out-of-Bounds Bounce-Back（Medium-High）**

**問題位置**: Line 732-734

```python
else:
    # 邊界外方向：由邊界條件負責重建
    pass
```

**設計假設**：
- 假設邊界條件**總是**正確重建所有未知分佈函數
- 但部分邊界條件（如 Neumann outflow）可能**無法完全**重建所有方向

**風險場景**：
1. **角落節點**：同時受兩個邊界影響，可能有遺漏的方向
2. **動態邊界**：Moving wall 等條件可能在某些步驟未完全覆蓋
3. **邊界條件順序**：多個 BC 重疊時，後者可能覆蓋前者

**物理後果**：
- 未初始化的 `f_dst` 分量 → 隨機值或零值
- 邊界處質量/動量不守恆
- 數值不穩定性從邊界傳播至內部

---

### 🟡 **P3: CFL Check 重複計算巨觀量（Medium）**

**問題位置**: Line 1070

```python
def check_cfl_condition(self, warn_only=True):
    self._update_macro(self.f)  # ← 額外開銷
    has_violation = self._check_cfl_violation()
```

**問題**：
- 如果在 `step()` 之後立即調用 `check_cfl_condition()`，會**重複計算**巨觀量
- `self.f` 與 `f_dst` 在雙緩衝模式下可能指向不同緩衝區

**數值後果**：
- 效能損失（診斷開銷 > 1%）
- 更嚴重：**讀到錯誤的緩衝區** → CFL 檢查基於舊資料 → 誤判

---

### 🟢 **P4: Diagnostics 單次遍歷優化（Low Risk）**

**問題位置**: Line 820-873

**分析**：
- 優化本身是**正確**的（減少記憶體訪問）
- 但依賴於 `self.rho/u` 已正確更新
- 如果 P0 問題存在，診斷結果也會錯誤

**結論**：這是**症狀**而非根因。

---

## 3. 驗證方法（Diagnosis Procedures）

### 3.1 最小可重現案例（Minimal Reproducible Example）

**測試 Case**: Lid-Driven Cavity（最簡單，無障礙物）

```bash
# 運行基準測試
python examples/lid_driven_cavity.py --res 128 --re 100 --steps 10000
```

**預期行為**：
- ✅ 穩定：`mass_error < 1e-6`
- ✅ 收斂：`mom_res < 1e-5`
- ❌ 不穩定：質量誤差指數增長或 NaN

### 3.2 驗證步驟（按優先級）

#### 驗證 P0：時間步內一致性

**Test 1**: 單步追蹤
```python
# 在 step() 開頭添加：
print(f"Before _update_macro: rho[50,50]={self.rho[50,50]:.6f}")
self._update_macro(f_src)
print(f"After _update_macro: rho[50,50]={self.rho[50,50]:.6f}")

# 在 _collide_and_stream 內部 (Line 678):
print(f"In collision: current_rho={current_rho:.6f}, self.rho={self.rho[i,j]:.6f}")
```

**Expected**：兩者應**完全一致**。如果不一致 → 確認 P0。

**Test 2**: 移除 step() 中的 _update_macro()
```python
# 暫時註釋 Line 1024
def step(f_src, f_dst):
    # self._update_macro(f_src)  # ← 註釋掉
    self._update_smagorinsky_viscosity()
    self._collide_and_stream(f_src, f_dst)
    ...
```

**Expected**：如果穩定性**改善** → 確認 P0 是主因。

---

#### 驗證 P1：Gradient-based LES 邊界影響

**Test 3**: 關閉 Smagorinsky
```python
solver = LBMSolver(..., cs=0.0)  # ← 強制關閉 LES
```

**Expected**：如果穩定性**恢復** → 確認 LES 實作有問題。

**Test 4**: 檢查邊界處渦黏度
```python
# 在主迴圈中添加：
nu_sgs_field = solver.nu_sgs.to_numpy()
print(f"nu_sgs at boundary: max={nu_sgs_field[0,:].max()}, mean={nu_sgs_field[0,:].mean()}")
```

**Expected**：邊界處 `nu_sgs` 應平滑變化，不應出現突變或負值。

---

#### 驗證 P2：Out-of-Bounds Handling

**Test 5**: 檢查邊界質量守恆
```python
# 計算邊界區域的質量誤差
boundary_mask = np.zeros((nx, ny), dtype=bool)
boundary_mask[0,:] = boundary_mask[-1,:] = boundary_mask[:,0] = boundary_mask[:,-1] = True

boundary_mass_error = np.abs(rho[boundary_mask] - 1.0).mean()
print(f"Boundary mass error: {boundary_mass_error:.2e}")
```

**Expected**：邊界質量誤差應 < 1e-4。如果 > 1e-3 → 確認邊界條件遺漏。

---

#### 驗證 P3：CFL Check 時機

**Test 6**: 移除 runtime CFL check
```python
# 在主迴圈中暫時註釋：
# solver.check_cfl_condition()
```

**Expected**：如果穩定性改善 → CFL check 干擾了流場計算。

---

### 3.3 關鍵觀測指標

監控以下物理量的**時間演化**（繪製 step vs. value）：

| 指標 | 正常行為 | 異常信號 | 對應根因 |
|------|----------|---------|---------|
| `mass_error` | 單調遞減至 < 1e-6 | 振盪或指數增長 | P0, P2 |
| `mom_res` | 單調遞減至 < 1e-5 | 停滯或反彈 | P0, P1 |
| `max(nu_sgs)` | 平滑變化，< 10*nu | 突變或 > 100*nu | P1 |
| `max_u` | 穩定在 u_ref 附近 | 持續增長 | P0, P1 |
| `rho` 邊界值 | 接近 1.0 (±0.01) | > 1.05 或 < 0.95 | P2 |

---

## 4. 修正建議（Fix Recommendations）

### 🔴 **Fix P0: 移除 step() 中的冗餘 _update_macro() 調用（Critical）**

**原因**：
- `_collide_and_stream()` 內部已經讀取 `self.rho[i,j]`, `self.u[i,j]`（Line 678-679）
- 外部預計算與內部使用之間存在時間不一致風險

**方案 A（推薦）**: 在碰撞核內部重新計算區域巨觀量

```python
@ti.kernel
def _collide_and_stream(self, f_src: ti.template(), f_dst: ti.template()):
    for i, j in ti.ndrange(self.nx, self.ny):
        if self.mask[i, j] == 1:
            continue
        
        # === 1. 從 f_src 重新計算區域巨觀量（移除對 self.rho/u 的依賴）===
        f_vec = f_src[i, j]
        current_rho = 0.0
        current_u = ti.Vector([0.0, 0.0])
        
        for k in ti.static(range(9)):
            current_rho += f_vec[k]
            current_u += f_vec[k] * self.e[k]
        
        if current_rho > 1e-12:
            current_u /= current_rho
        
        # === 2. 使用區域計算的 rho, u（不依賴 self.rho/u）===
        m = self.M[None] @ f_vec
        # ... MRT 碰撞（使用 current_rho, current_u）
```

**方案 B**: 明確分離 Smagorinsky 更新與碰撞

```python
def step(f_src, f_dst):
    # 第一階段：更新診斷用的全局場（不影響碰撞）
    self._update_macro_for_diagnostics(f_src)
    self._update_smagorinsky_viscosity()
    
    # 第二階段：碰撞使用區域計算（內部自洽）
    self._collide_and_stream(f_src, f_dst)
    
    # 第三階段：邊界條件
    for bc_func, _ in self.bc_functions:
        bc_func(f_dst)
```

**驗證**：執行 Test 1, 2，確認時間一致性。

---

### 🟠 **Fix P1: 改進 Gradient-based LES 邊界處理（High）**

**問題**：邊界處梯度計算使用單側差分，不準確且引入噪聲。

**修正方案**：

```python
@ti.kernel
def _update_smagorinsky_viscosity(self):
    for i, j in ti.ndrange(self.nx, self.ny):
        if self.mask[i, j] == 0:
            if ti.static(self.cs > 0.0):
                # === 改進：邊界距離檢查 ===
                distance_to_boundary = ti.min(
                    ti.min(i, self.nx - 1 - i),
                    ti.min(j, self.ny - 1 - j)
                )
                
                # 開放邊界附近（距離 < 3）：降低 LES 強度或關閉
                if distance_to_boundary < 3:
                    self.nu_sgs[i, j] = 0.0  # 或 *= 0.5
                    continue
                
                # === 固體邊界處理：檢查所有鄰居是否為固體 ===
                has_solid_neighbor = (
                    self.mask[i+1, j] == 1 or
                    self.mask[i-1, j] == 1 or
                    self.mask[i, j+1] == 1 or
                    self.mask[i, j-1] == 1
                )
                
                if has_solid_neighbor:
                    # 固體附近：使用壁面函數或關閉 LES
                    self.nu_sgs[i, j] = 0.0
                    continue
                
                # === 內部流場：正常計算（確保鄰居都是流體）===
                # ... 原始梯度計算 ...
```

**替代方案**：使用 moment-based Smagorinsky（更穩健）

```python
# 恢復舊版非平衡矩量方法（參考 Kolmogorov case）
q_neq = ti.sqrt(m_star[4]**2 + m_star[6]**2)  # 非平衡動量通量
self.nu_sgs[i, j] = (self.cs * delta)**2 * q_neq / self.nu
```

**驗證**：執行 Test 3, 4，確認邊界處 `nu_sgs` 無異常。

---

### 🟠 **Fix P2: 恢復 Out-of-Bounds 安全處理（Medium-High）**

**修正方案**：

```python
@ti.kernel
def _collide_and_stream(self, f_src, f_dst):
    for i, j in ti.ndrange(self.nx, self.ny):
        # ... 碰撞計算 ...
        
        for k in ti.static(range(9)):
            dest_i = i + self.e[k][0]
            dest_j = j + self.e[k][1]
            
            in_bounds = (
                dest_i >= 0 and dest_i < self.nx and
                dest_j >= 0 and dest_j < self.ny
            )
            
            if in_bounds:
                if self.mask[dest_i, dest_j] == 1:
                    f_dst[i, j][self.inv[k]] = f_post[k]
                else:
                    f_dst[dest_i, dest_j][k] = f_post[k]
            else:
                # === 恢復：邊界外 Bounce-Back（安全保底）===
                f_dst[i, j][self.inv[k]] = f_post[k]
                
                # 選項：添加警告標記（診斷用）
                # ti.atomic_add(self.boundary_violation_count[None], 1)
```

**理由**：
- 邊界條件應是**優化**，而非**必須依賴**
- Bounce-back 是物理上合理的 fallback（代表剛性壁面）
- 防禦性程式設計：永不假設外部正確性

**驗證**：執行 Test 5，確認邊界質量守恆。

---

### 🟡 **Fix P3: 優化 CFL Check 調用時機（Medium）**

**修正方案**：

```python
# 選項 A：移除 check_cfl_condition 中的重複計算
def check_cfl_condition(self, warn_only=True):
    # 假設 _update_macro 已在 step() 後執行
    # self._update_macro(self.f)  # ← 移除
    has_violation = self._check_cfl_violation()
    ...

# 選項 B：明確指定緩衝區
def check_cfl_condition(self, f_current: ti.template(), warn_only=True):
    self._update_macro(f_current)  # ← 使用正確的緩衝區
    has_violation = self._check_cfl_violation()
    ...
```

**建議**：在主迴圈中控制調用頻率

```python
# 每 100 步檢查一次（減少開銷）
if step % 100 == 0:
    solver._update_macro(f_dst)  # 確保使用最新資料
    solver._check_cfl_violation()
```

**驗證**：執行 Test 6，確認 CFL check 不干擾流場。

---

## 5. 實施優先級與時間表（Priority & Timeline）

| 優先級 | 修正項目 | 預期效果 | 實施風險 | 測試時間 |
|--------|---------|---------|---------|---------|
| **P0** | 移除 step() 中冗餘 _update_macro | 根本解決時間不一致 | 低（區域變數） | 1 小時 |
| **P1** | 改進 LES 邊界處理 | 消除邊界振盪 | 中（需測試多種邊界） | 2 小時 |
| **P2** | 恢復 out-of-bounds bounce-back | 增強邊界穩定性 | 低（保守修復） | 30 分鐘 |
| **P3** | 優化 CFL check 時機 | 降低診斷開銷 | 低（僅效能優化） | 30 分鐘 |

**建議實施順序**：
1. **立即**：P0（最關鍵，阻塞所有其他測試）
2. **次要**：P2（安全修復，無副作用）
3. **驗證後**：P1（需基於穩定基線測試）
4. **最後**：P3（效能優化，非穩定性問題）

---

## 6. 預防措施（Prevention Strategy）

### 6.1 程式設計原則
- **單一真相來源（Single Source of Truth）**：
  - 巨觀量應從**單一**分佈函數緩衝區計算
  - 避免多處存儲 `self.rho/u` 導致不一致
  
- **區域可推理性（Local Reasoning）**：
  - Kernel 內部應盡可能自洽，不依賴外部狀態
  - `_collide_and_stream()` 應從 `f_src` 計算所需所有量

### 6.2 測試策略
- **單元測試**：每個 kernel 獨立測試（質量守恆、對稱性）
- **回歸測試**：建立基準資料集（Lid-Driven Cavity Re=100, 400, 1000）
- **邊界測試**：專門測試邊界節點（角落、固體附近、出口）

### 6.3 監控指標
- 在 CI 中自動檢查：
  - `mass_error < 1e-6`（全域）
  - `mass_error < 1e-4`（邊界區域）
  - `nu_sgs < 10 * nu`（LES 合理性）

---

## 7. 參考資料（References）

### 內部文檔
- `src/lbm_taichi/core/lbm_solver.py`: Line 1010-1034（step 實作）
- `src/lbm_taichi/core/lbm_solver.py`: Line 766-818（gradient-based LES）
- `AGENTS.md`: 物理正確性優先原則

### 外部理論
- **MRT-LBM 穩定性**：d'Humières et al. (2002) - "Multiple-relaxation-time lattice Boltzmann models"
- **Smagorinsky 邊界處理**：Pope (2000) - "Turbulent Flows", Chapter 13.3
- **LBM 邊界條件**：Krüger et al. (2017) - "The Lattice Boltzmann Method", Chapter 5

---

## 8. 下一步行動（Next Actions）

### 立即執行
1. ✅ **完成本分析報告**（已完成）
2. 📋 **執行 Test 1, 2**（驗證 P0 假設）
3. 🔧 **實施 Fix P0**（移除冗餘調用）
4. ✅ **回歸測試**（Lid-Driven Cavity 基準）

### 後續追蹤
- 如果 P0 修復後仍不穩定 → 深入分析 P1（LES 實作細節）
- 收集多個 case 的穩定性數據（Cylinder, Airfoil）
- 更新 `AGENTS.md` 添加「時間一致性」設計原則

---

## 9. 總結（Executive Summary）

**最可能根因**：  
🔴 **P0 - step() 中的冗餘 _update_macro() 調用**導致時間不一致，MRT 碰撞算子使用錯誤的巨觀量計算平衡分佈函數，累積誤差破壞質量/動量守恆。

**修復策略**：  
在 `_collide_and_stream()` 內部從 `f_src` 重新計算區域 `current_rho`, `current_u`，移除對全局 `self.rho/u` 的依賴。

**驗證指標**：  
- 質量守恆誤差 < 1e-6
- 動量殘差單調遞減至 < 1e-5
- 邊界處渦黏度無異常突變

**風險評估**：  
- 低風險修復（使用區域變數）
- 高收益（根本解決穩定性問題）
- 預期 1-2 小時完成修復與驗證

---

**報告完成時間**: 2026-01-27  
**下一步**: 請閱讀此文件後，指示是否執行驗證測試或直接實施修復。
