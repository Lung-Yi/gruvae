# 變更日誌 (Changelog)

## [v2.0.0] - 2024-12-09

### 新增 (Added)
- ✨ **配置檔案系統**: 新增 YAML 配置檔案支援
  - 所有超參數現在可以通過 `train.yaml` 管理
  - 支援命令列指定自訂配置: `python train.py --config my_config.yaml`
  - 配置檔案包含：資料、模型、訓練、驗證、檢查點、設備設定
  - 新增 `CONFIG_GUIDE.md` 詳細說明配置使用方式

### 改進 (Improved)
- 🔧 **Trainer 類別增強**:
  - 新增 `grad_clip_max_norm` 參數（可配置梯度裁剪）
  - 新增 `save_interval` 參數（可配置檢查點保存間隔）
  - 新增 `num_reconstruct` 和 `num_sample` 參數（可配置驗證顯示數量）

- 📝 **文檔更新**:
  - 更新 `README.md` 包含配置檔案使用說明
  - 新增 `CONFIG_GUIDE.md` 配置檔案完整指南
  - 更新專案結構說明

### 檔案變更
- 修改: `train.py` - 新增配置檔案載入功能
- 新增: `train.yaml` - 預設配置檔案
- 新增: `CONFIG_GUIDE.md` - 配置檔案使用指南
- 新增: `CHANGELOG.md` - 本檔案

---

## [v1.0.0] - 2024-12-09

### 新增 (Added)
- ✨ **重建準確度計算**:
  - 在驗證階段計算 canonicalized SMILES 匹配率
  - 顯示格式: `Val Recon Acc: XX.XX% (XXXX/3791)`

- ✨ **學習率和 KL Weight 顯示**:
  - 每個 epoch 結束時顯示當前學習率
  - 顯示當前 KL weight 值
  - 追蹤到訓練歷史中

- ✨ **KL Weight Annealing**:
  - 實作線性 KL weight 退火
  - 從 0.0 開始，逐漸增加到 0.1
  - 避免後驗坍塌問題
  - 可配置起始值、結束值和退火期間

### 檔案變更
- 修改: `train.py` - 實作三個新功能
- 新增: `UPDATES.md` - 功能更新說明

---

## [v0.1.0] - 初始版本

### 新增 (Added)
- 🎉 **GRU-VAE 基礎架構**:
  - GRU Encoder 和 Decoder
  - 變分自編碼器 (VAE) 實作
  - Teacher forcing 支援
  - 重參數化技巧

- 📊 **資料處理**:
  - SMILES tokenizer
  - 資料增強（隨機化 SMILES 輸入，規範化輸出）
  - 自訂 Dataset 和 DataLoader
  - SMILES 預處理腳本

- 🏋️ **訓練功能**:
  - 完整的訓練循環
  - 驗證循環
  - 梯度裁剪
  - 自動保存最佳模型
  - 定期檢查點保存

- 📈 **監控功能**:
  - 訓練/驗證 loss 追蹤
  - 重建 loss 和 KL loss 分別顯示
  - 分子重建範例展示
  - 潛在空間採樣展示

- 📝 **文檔**:
  - `README.md` - 完整的專案說明
  - 模型架構說明
  - 使用範例

### 檔案結構
- `tokenizer.py` - SMILES 分詞器
- `dataset.py` - Dataset 和 DataLoader
- `model.py` - GRU-VAE 模型
- `train.py` - 訓練腳本
- `process_smiles.py` - 資料預處理
- `README.md` - 專案文檔

---

## 版本號規則

採用語義化版本 (Semantic Versioning):
- **主版本號 (Major)**: 不相容的 API 變更
- **次版本號 (Minor)**: 向下相容的新功能
- **修訂號 (Patch)**: 向下相容的問題修正

## 未來計劃

### v2.1.0 (計劃中)
- [ ] 增加更多 learning rate scheduler 選項
- [ ] 支援斷點續訓
- [ ] TensorBoard 整合
- [ ] 更多資料增強選項

### v2.2.0 (計劃中)
- [ ] 支援多 GPU 訓練
- [ ] 混合精度訓練 (AMP)
- [ ] 更多採樣策略（Beam Search, Top-k）

### v3.0.0 (構想中)
- [ ] 條件式 VAE (CVAE)
- [ ] 屬性預測整合
- [ ] 分子優化功能
