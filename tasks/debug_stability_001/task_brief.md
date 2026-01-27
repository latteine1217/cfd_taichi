# Task Brief: Solver Stability Analysis

**任務編號**: debug_stability_001  
**建立時間**: 2026-01-27  
**負責角色**: Debug Specialist (Research/Planning Agent)  
**狀態**: ✅ 分析完成

---

## 任務描述

分析近期修改後 LBM solver 穩定性下降的根本原因，並提供修正建議（不實施 code）。

---

## Recent Changes（觸發因素）

1. Gradient-based LES（使用速度梯度計算 Smagorinsky 渦黏度）
2. `step()` 現在調用 `_update_macro(f_src)` 和 `_update_smagorinsky_viscosity()`
3. 移除 out-of-bounds bounce-back
4. CFL check 更新巨觀量
5. Diagnostics 變更

---

## 關鍵發現

**🔴 最可能根因（Critical Priority）**:

**P0: 時間步內多次 `_update_macro()` 調用導致時間不一致**

- `step()` 在碰撞前調用 `_update_macro(f_src)`
- `_collide_and_stream()` 內部讀取 `self.rho[i,j]`, `self.u[i,j]`
- 雙緩衝切換時可能讀到上一步的殘留值
- MRT 碰撞算子使用錯誤的巨觀量 → 平衡分佈函數錯誤 → 數值發散

---

## 其他風險因素

- **P1 (High)**: Gradient-based LES 邊界處理不當（單側差分、邊界處梯度錯誤）
- **P2 (Medium-High)**: 移除 out-of-bounds bounce-back 可能導致邊界遺漏
- **P3 (Medium)**: CFL check 重複計算巨觀量且可能讀錯緩衝區

---

## 推薦修正方案

### Fix P0（立即執行）
在 `_collide_and_stream()` 內部從 `f_src` **重新計算**區域 `current_rho`, `current_u`，移除對全局 `self.rho/u` 的依賴。

```python
# 碰撞核內部：
for i, j in ti.ndrange(self.nx, self.ny):
    f_vec = f_src[i, j]
    
    # 區域重新計算（不依賴 self.rho/u）
    current_rho = sum(f_vec[k] for k in range(9))
    current_u = sum(f_vec[k] * e[k] for k in range(9)) / current_rho
    
    # 使用 current_rho, current_u 計算 MRT...
```

### Fix P1（次要）
改進邊界附近梯度計算，或在邊界 3 格內關閉 LES。

### Fix P2（安全修復）
恢復 out-of-bounds bounce-back 作為安全保底機制。

---

## 驗證方法

1. **Lid-Driven Cavity** (Re=100)：最簡單基準測試
2. 監控指標：
   - `mass_error < 1e-6`
   - `mom_res` 單調遞減
   - 邊界處 `nu_sgs` 無異常
3. 執行測試：
   - 移除 `step()` 中的 `_update_macro()` → 觀察穩定性變化
   - 關閉 LES (`cs=0.0`) → 隔離 LES 影響

---

## 產出文件

📄 `/tasks/debug_stability_001/debug_playbook.md`

包含：
- 詳細根因分析
- 驗證程序（最小可重現案例）
- 具體修復方案（含代碼示例）
- 實施優先級與時間表
- 預防措施與參考資料

---

## 下一步

請閱讀完整的 `debug_playbook.md` 後決定：

1. 執行驗證測試（Test 1-6）確認假設
2. 直接實施 Fix P0（最關鍵修復）
3. 需要更多資訊或替代分析

---

**完成時間**: 2026-01-27  
**報告品質**: 符合「根本原因分析專家」角色要求
