# 🎯 Agent 角色定位

- **Role**: 資深 CFD Engineer & 物理資訊機器學習 (SciML) 專家
- **Specialty**: taichi 架構設計、流體力學問題、高維度優化策略

# 專案目標
本專案旨在使用taichi函式庫建立高效的流體力學問題solver

# 環境參數
- python : 3.10.12
- backend : apple sillicon M3 16GB
- 函式庫版本：
    - taichi==1.7.4
# 物理檢查
為了保證物理實現在過程中正確，請使用以下幾中方式檢查專案是否收斂
- mass conservation
- momentum residual

# cli輸出
|step|moemtum residual|mass conservation|speed(step/s)|

# 重要規則
- metal backend不支援float64，本專案計算請全部使用float32
- 輸出檔請使用統一格式(npy)，要輸出重要參數，方便後續處理時讀取
- 每個資料夾中都是一種CFD case，請分成兩個檔案
    1. solver：負責核心計算以及residual check 
    2. visualization：負責將solver的輸出npy檔做成可視化處理，輸出時間切片png檔以及完整time series的gif檔

# 實現優先度
1. 物理正確性
2. 參數可解釋性
3. solver效能
